"""
Dolphin-inspired audio-only source separator for ASS.

This module adapts the transferable part of Dolphin:
- single-pass multi-scale encoder/decoder separator,
- global/local feature modeling in each separator layer,
- a lightweight semantic-prior branch.

The original Dolphin paper uses visual DP-LipCoder tokens.  For generic audio
source separation we replace that path with an audio-derived source-prior coder
over SFC band tokens, so the model remains usable for speech/music/effects.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectral_feature_compression.core.model.online_sfc_2d import (
    CausalConv2d,
    RMSNorm2d,
    SpectralCompressor2d,
    SpectralDecoder2d,
    _runtime_assert,
)


def _validate_even_pyramid(n_bands: int, num_scales: int) -> None:
    divisor = 2 ** max(num_scales - 1, 0)
    if n_bands % divisor != 0:
        raise ValueError(f"n_bands={n_bands} must be divisible by {divisor} for {num_scales} scales.")


class FrozenDolphinBandSpec2d(nn.Module):
    """
    Deterministic frozen band constants for DolphinSFCNPU.

    This intentionally avoids `librosa` or any environment-dependent fallback.
    `musical` uses deterministic log-spaced triangular bands, while `linear`
    uses uniformly spaced triangular bands.
    """

    def __init__(self, n_freq: int, n_bands: int, band_config: str = "musical"):
        super().__init__()
        if n_freq <= 0 or n_bands <= 0:
            raise ValueError("n_freq and n_bands must be positive.")
        self.n_freq = n_freq
        self.n_bands = n_bands
        self.band_config = band_config
        basis = self._build_basis(n_freq=n_freq, n_bands=n_bands, band_config=band_config)
        self.register_buffer("basis", basis.view(1, n_bands, 1, n_freq))

    @staticmethod
    def _build_basis(n_freq: int, n_bands: int, band_config: str) -> torch.Tensor:
        if band_config == "linear":
            edges = torch.linspace(0.0, float(n_freq - 1), steps=n_bands + 2)
        elif band_config == "musical":
            # Low-frequency bins receive narrower bands, without relying on
            # librosa/midi conversion during export or validation.
            max_pos = torch.log1p(torch.tensor(float(n_freq - 1)))
            edges = torch.expm1(torch.linspace(0.0, float(max_pos), steps=n_bands + 2))
        else:
            raise ValueError(f"Unsupported frozen band_config: {band_config!r}")

        freq_pos = torch.arange(n_freq, dtype=torch.float32)
        basis = torch.zeros(n_bands, n_freq, dtype=torch.float32)
        for band_idx in range(n_bands):
            left = edges[band_idx]
            center = edges[band_idx + 1]
            right = torch.maximum(edges[band_idx + 2], center + 1.0)
            rising = (freq_pos - left) / (center - left).clamp_min(1.0)
            falling = (right - freq_pos) / (right - center).clamp_min(1.0)
            basis[band_idx] = torch.clamp(torch.minimum(rising, falling), min=0.0, max=1.0)
            if basis[band_idx].amax() <= 0:
                nearest = int(torch.clamp(center.round(), min=0, max=n_freq - 1).item())
                basis[band_idx, nearest] = 1.0
        return basis

    def band_bias(self) -> torch.Tensor:
        peak = self.basis.amax(dim=-1, keepdim=True).clamp_min(1e-6)
        return 2.0 * (self.basis / peak) - 1.0

    def decode_basis(self) -> torch.Tensor:
        return self.basis / self.basis.sum(dim=1, keepdim=True).clamp_min(1e-6)


class DolphinPointwiseFFN2d(nn.Module):
    def __init__(self, channels: int, expansion: int = 2, kernel_size: tuple[int, int] = (3, 3)):
        super().__init__()
        hidden = channels * expansion
        self.norm = RMSNorm2d(channels)
        self.in_proj = nn.Conv2d(channels, hidden * 2, kernel_size=1, bias=True)
        self.dw = CausalConv2d(hidden, hidden, kernel_size=kernel_size, groups=hidden, bias=True)
        self.out_proj = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        a, b = self.in_proj(y).chunk(2, dim=1)
        y = a * torch.sigmoid(b)
        y = F.silu(self.dw(y))
        return x + self.out_proj(y)

    def init_stream_state(self, batch_size: int, freq_bins: int, device=None, dtype=None) -> torch.Tensor:
        return self.dw.init_stream_state(batch_size, freq_bins=freq_bins, device=device, dtype=dtype)

    def forward_stream(self, x: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y = self.norm(x)
        a, b = self.in_proj(y).chunk(2, dim=1)
        y = a * torch.sigmoid(b)
        y, new_state = self.dw.forward_stream(y, state)
        y = F.silu(y)
        return x + self.out_proj(y), new_state


class DolphinGlobalLocalBlock2d(nn.Module):
    """
    NPU-friendly approximation of Dolphin GLA.

    The paper's GA uses coarse self-attention and LA uses DCT heat diffusion.
    Runtime DCT/FFT and dynamic attention constants are awkward for the ASS NPU
    path, so this block keeps the same division of labor with causal conv2d:
    longer temporal depthwise conv for global context, shorter depthwise conv
    for local heat-like smoothing, then a gated FFN.
    """

    def __init__(
        self,
        channels: int,
        global_kernel: int = 7,
        local_kernel: tuple[int, int] = (3, 3),
        ffn_expansion: int = 2,
    ):
        super().__init__()
        if (global_kernel - 1) >= 14:
            raise ValueError("global_kernel violates ASS NPU span limit.")
        self.global_norm = RMSNorm2d(channels)
        self.global_dw = CausalConv2d(
            channels,
            channels,
            kernel_size=(global_kernel, 1),
            groups=channels,
            bias=True,
        )
        self.global_gate = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.global_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

        self.local_norm = RMSNorm2d(channels)
        self.local_dw = CausalConv2d(
            channels,
            channels,
            kernel_size=local_kernel,
            groups=channels,
            bias=True,
        )
        self.local_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.ffn = DolphinPointwiseFFN2d(channels, expansion=ffn_expansion, kernel_size=local_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.global_dw(self.global_norm(x))
        x = x + self.global_proj(g * torch.sigmoid(self.global_gate(x)))
        l = self.local_dw(self.local_norm(x))
        x = x + self.local_proj(F.silu(l))
        return self.ffn(x)

    def init_stream_state(self, batch_size: int, freq_bins: int, device=None, dtype=None):
        return (
            self.global_dw.init_stream_state(batch_size, freq_bins=freq_bins, device=device, dtype=dtype),
            self.local_dw.init_stream_state(batch_size, freq_bins=freq_bins, device=device, dtype=dtype),
            self.ffn.init_stream_state(batch_size, freq_bins=freq_bins, device=device, dtype=dtype),
        )

    def forward_stream(self, x: torch.Tensor, state) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        global_state, local_state, ffn_state = state
        g, new_global_state = self.global_dw.forward_stream(self.global_norm(x), global_state)
        x = x + self.global_proj(g * torch.sigmoid(self.global_gate(x)))
        l, new_local_state = self.local_dw.forward_stream(self.local_norm(x), local_state)
        x = x + self.local_proj(F.silu(l))
        x, new_ffn_state = self.ffn.forward_stream(x, ffn_state)
        return x, (new_global_state, new_local_state, new_ffn_state)


class DolphinSourcePriorCoder2d(nn.Module):
    """
    Audio-only replacement for Dolphin's visual semantic token path.

    It produces a compact gate over SFC band tokens.  This is not a VQ module:
    VQ/lip distillation is useful for AVSS, but unsuitable for the online
    three-stem ASS deployment path without a video stream.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.norm = RMSNorm2d(channels)
        self.temporal = CausalConv2d(channels, channels, kernel_size=(5, 1), groups=channels, bias=True)
        self.mix = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        prior = self.mix(F.silu(self.temporal(self.norm(z))))
        return z * (1.0 + torch.sigmoid(prior))

    def init_stream_state(self, batch_size: int, freq_bins: int, device=None, dtype=None) -> torch.Tensor:
        return self.temporal.init_stream_state(batch_size, freq_bins=freq_bins, device=device, dtype=dtype)

    def forward_stream(self, z: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        prior, new_state = self.temporal.forward_stream(self.norm(z), state)
        prior = self.mix(F.silu(prior))
        return z * (1.0 + torch.sigmoid(prior)), new_state


class DolphinEncoderStage2d(nn.Module):
    def __init__(self, channels: int, do_downsample: bool):
        super().__init__()
        self.block = DolphinGlobalLocalBlock2d(channels)
        self.do_downsample = do_downsample
        if do_downsample:
            self.down = CausalConv2d(channels, channels, kernel_size=(3, 3), groups=1, bias=True)
            self.band_down = nn.Conv2d(channels, channels, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1), bias=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.block(x)
        skip = x
        if self.do_downsample:
            x = F.silu(self.down(x))
            x = self.band_down(x)
        return x, skip

    def init_stream_state(self, batch_size: int, freq_bins: int, device=None, dtype=None):
        block_state = self.block.init_stream_state(batch_size, freq_bins=freq_bins, device=device, dtype=dtype)
        if not self.do_downsample:
            return (block_state,)
        down_state = self.down.init_stream_state(batch_size, freq_bins=freq_bins, device=device, dtype=dtype)
        return (block_state, down_state)

    def forward_stream(self, x: torch.Tensor, state) -> tuple[torch.Tensor, torch.Tensor, tuple]:
        block_state = state[0]
        x, new_block_state = self.block.forward_stream(x, block_state)
        skip = x
        if not self.do_downsample:
            return x, skip, (new_block_state,)
        down_state = state[1]
        x, new_down_state = self.down.forward_stream(x, down_state)
        x = self.band_down(F.silu(x))
        return x, skip, (new_block_state, new_down_state)


class DolphinDecoderStage2d(nn.Module):
    def __init__(self, channels: int, do_upsample: bool):
        super().__init__()
        self.do_upsample = do_upsample
        if do_upsample:
            self.band_up = nn.ConvTranspose2d(
                channels,
                channels,
                kernel_size=(1, 4),
                stride=(1, 2),
                padding=(0, 1),
                bias=True,
            )
            self.merge = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=True)
        self.block = DolphinGlobalLocalBlock2d(channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if self.do_upsample:
            x = self.band_up(x)
            x = self.merge(torch.cat([x, skip], dim=1))
        return self.block(x)

    def init_stream_state(self, batch_size: int, freq_bins: int, device=None, dtype=None):
        return self.block.init_stream_state(batch_size, freq_bins=freq_bins, device=device, dtype=dtype)

    def forward_stream(self, x: torch.Tensor, skip: torch.Tensor, state) -> tuple[torch.Tensor, tuple]:
        if self.do_upsample:
            x = self.band_up(x)
            x = self.merge(torch.cat([x, skip], dim=1))
        return self.block.forward_stream(x, state)


class DolphinSFCNPUSeparator(nn.Module):
    """
    Audio-only Dolphin/SFC separator.

    Input/Output contract:
      x: (B, 2 * n_chan, T, F), packed real/imag STFT
      y: (B, 2 * n_src * n_chan, T, F)
    """

    def __init__(
        self,
        n_freq: int,
        n_bands: int = 64,
        n_fft: int | None = None,
        sample_rate: int | None = None,
        band_config: str = "musical",
        n_src: int = 3,
        n_chan: int = 1,
        d_model: int = 288,
        num_scales: int = 3,
        masking: bool = True,
    ):
        super().__init__()
        _validate_even_pyramid(n_bands, num_scales)
        self.n_freq = n_freq
        self.n_bands = n_bands
        self.n_src = n_src
        self.n_chan = n_chan
        self.d_model = d_model
        self.num_scales = num_scales
        self.masking = masking

        self.band_spec = self._build_band_spec(
            n_freq=n_freq,
            n_bands=n_bands,
            n_fft=n_fft,
            sample_rate=sample_rate,
            band_config=band_config,
        )
        self.in_proj = nn.Sequential(nn.Conv2d(2 * n_chan, d_model, kernel_size=1), RMSNorm2d(d_model))
        self.compressor = SpectralCompressor2d(d_model, self.band_spec, causal=True)
        self.prior = DolphinSourcePriorCoder2d(d_model)

        self.encoder = nn.ModuleList(
            DolphinEncoderStage2d(d_model, do_downsample=idx < num_scales - 1)
            for idx in range(num_scales)
        )
        self.decoder = nn.ModuleList(
            DolphinDecoderStage2d(d_model, do_upsample=idx > 0)
            for idx in range(num_scales)
        )

        self.decoder_to_freq = SpectralDecoder2d(d_model, self.band_spec)
        out_ch = n_src * n_chan if masking else 2 * n_src * n_chan
        self.out_proj = nn.Conv2d(d_model, out_ch, kernel_size=1)

    @staticmethod
    def _build_band_spec(
        n_freq: int,
        n_bands: int,
        n_fft: int | None,
        sample_rate: int | None,
        band_config: str,
    ) -> FrozenDolphinBandSpec2d:
        _ = n_fft, sample_rate
        return FrozenDolphinBandSpec2d(n_freq=n_freq, n_bands=n_bands, band_config=band_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _runtime_assert(x.ndim == 4, f"Expected (B,C,T,F), got {x.shape}")
        _runtime_assert(x.shape[-1] == self.n_freq, f"{x.shape[-1]} vs {self.n_freq}")
        z = self.compressor(self.in_proj(x))
        z = self.prior(z)

        skips = []
        for stage in self.encoder:
            z, skip = stage(z)
            skips.append(skip)

        for stage, skip in zip(self.decoder, reversed(skips)):
            z = stage(z, skip)

        y = self.out_proj(self.decoder_to_freq(z))
        if self.masking:
            y = apply_source_gain_mask_4d(x, y, n_src=self.n_src, n_chan=self.n_chan)
        return y

    def init_stream_state(self, batch_size: int = 1, *, device=None, dtype=None):
        comp = self.compressor.init_stream_state(batch_size, freq_bins=self.n_freq, device=device, dtype=dtype)
        prior = self.prior.init_stream_state(batch_size, freq_bins=self.n_bands, device=device, dtype=dtype)

        enc_states = []
        bands = self.n_bands
        for idx, stage in enumerate(self.encoder):
            enc_states.append(stage.init_stream_state(batch_size, freq_bins=bands, device=device, dtype=dtype))
            if idx < self.num_scales - 1:
                bands = bands // 2

        dec_states = []
        for idx, stage in enumerate(self.decoder):
            dec_states.append(stage.init_stream_state(batch_size, freq_bins=bands, device=device, dtype=dtype))
            if idx < self.num_scales - 1:
                bands = bands * 2

        return (comp, prior, tuple(enc_states), tuple(dec_states))

    def forward_stream(self, x: torch.Tensor, state=None):
        _runtime_assert(x.ndim == 4, f"Expected (B,C,T,F), got {x.shape}")
        _runtime_assert(x.shape[2] == 1, "forward_stream expects one frame at a time.")
        if state is None:
            state = self.init_stream_state(batch_size=x.shape[0], device=x.device, dtype=x.dtype)

        comp_state, prior_state, enc_states, dec_states = state
        z, new_comp_state = self.compressor.forward_stream(self.in_proj(x), comp_state)
        z, new_prior_state = self.prior.forward_stream(z, prior_state)

        skips = []
        new_enc_states = []
        for stage, stage_state in zip(self.encoder, enc_states):
            z, skip, new_stage_state = stage.forward_stream(z, stage_state)
            skips.append(skip)
            new_enc_states.append(new_stage_state)

        new_dec_states = []
        for stage, skip, stage_state in zip(self.decoder, reversed(skips), dec_states):
            z, new_stage_state = stage.forward_stream(z, skip, stage_state)
            new_dec_states.append(new_stage_state)

        y = self.out_proj(self.decoder_to_freq(z))
        if self.masking:
            y = apply_source_gain_mask_4d(x, y, n_src=self.n_src, n_chan=self.n_chan)
        return y, (new_comp_state, new_prior_state, tuple(new_enc_states), tuple(new_dec_states))

    def state_numel(self, batch_size: int = 1) -> int:
        state = self.init_stream_state(
            batch_size=batch_size,
            device=self.out_proj.weight.device,
            dtype=self.out_proj.weight.dtype,
        )
        return _tree_numel(state)

    def state_size_bytes(self, batch_size: int = 1, dtype: torch.dtype = torch.float16) -> int:
        return self.state_numel(batch_size=batch_size) * torch.tensor([], dtype=dtype).element_size()


def _tree_numel(tree) -> int:
    if isinstance(tree, torch.Tensor):
        return int(tree.numel())
    return sum(_tree_numel(item) for item in tree)


def apply_source_gain_mask_4d(x: torch.Tensor, mask_logits: torch.Tensor, n_src: int, n_chan: int) -> torch.Tensor:
    """Apply real-valued source gains to packed complex input using 4D tensors only."""

    _runtime_assert(x.shape[1] == 2 * n_chan, f"{x.shape[1]} vs {2 * n_chan}")
    _runtime_assert(mask_logits.shape[1] == n_src * n_chan, f"{mask_logits.shape[1]} vs {n_src * n_chan}")
    gains = torch.sigmoid(mask_logits)
    outputs = []
    for src_idx in range(n_src):
        for chan_idx in range(n_chan):
            gain = gains[:, src_idx * n_chan + chan_idx : src_idx * n_chan + chan_idx + 1, :, :]
            real = x[:, 2 * chan_idx : 2 * chan_idx + 1, :, :] * gain
            imag = x[:, 2 * chan_idx + 1 : 2 * chan_idx + 2, :, :] * gain
            outputs.extend([real, imag])
    return torch.cat(outputs, dim=1)


class DolphinSFCNPUStreamingExportWrapper(nn.Module):
    def __init__(self, core: DolphinSFCNPUSeparator, batch_size: int = 1, dtype: torch.dtype = torch.float32):
        super().__init__()
        from spectral_feature_compression.utils.onnx_streaming import flatten_tensor_tree

        self.core = core
        example_state = core.init_stream_state(batch_size=batch_size, dtype=dtype)
        flat_state, state_spec = flatten_tensor_tree(example_state)
        self.state_spec = state_spec
        self.state_tensor_count = len(flat_state)

    def forward(self, x: torch.Tensor, *flat_state: torch.Tensor):
        from spectral_feature_compression.utils.onnx_streaming import flatten_tensor_tree, unflatten_tensor_tree

        state = unflatten_tensor_tree(flat_state, self.state_spec)
        y, new_state = self.core.forward_stream(x, state)
        flat_new_state, _ = flatten_tensor_tree(new_state)
        return (y, *flat_new_state)


def build_dolphin_sfc_npu_preset(
    preset: str,
    *,
    n_freq: int,
    n_fft: int | None = None,
    sample_rate: int | None = None,
    n_src: int = 3,
    n_chan: int = 1,
    band_config: str = "musical",
    masking: bool = True,
) -> DolphinSFCNPUSeparator:
    """
    Build named DolphinSFC configurations.

    `edge_small` keeps streaming state modest for structural/export tests.
    `large_8m` is the recommended first performance recipe; it intentionally
    spends more cache and compute and should be trained/evaluated before any
    later cache-compression pass.
    """

    presets = {
        "edge_small": dict(n_bands=32, d_model=16, num_scales=3),
        "large_6m": dict(n_bands=64, d_model=256, num_scales=3),
        "large_8m": dict(n_bands=64, d_model=288, num_scales=3),
    }
    if preset not in presets:
        names = ", ".join(sorted(presets))
        raise ValueError(f"Unknown DolphinSFC preset {preset!r}. Available presets: {names}")

    return DolphinSFCNPUSeparator(
        n_freq=n_freq,
        n_fft=n_fft,
        sample_rate=sample_rate,
        band_config=band_config,
        n_src=n_src,
        n_chan=n_chan,
        masking=masking,
        **presets[preset],
    )
