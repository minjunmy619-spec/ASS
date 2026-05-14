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
    """
    ONNX export wrapper that collapses the nested streaming-state tree of
    `DolphinSFCNPUSeparator` into a single packed tensor.

    Rationale (see AGENT.md):
      - rule 14 asks the exported graph to have a small number of input/output
        parameters. Exposing every leaf cache tensor as its own ONNX input
        would produce 23 inputs and 23 outputs for the three-scale preset.
      - The core separator keeps its ergonomic tree-state API for Python
        callers and training. Only the export boundary is flattened.

    Packed shape convention:
      * state tensor is 2-D ``(B, total_numel)``. Using 2-D (rather than
        ``(B, 1, 1, total_numel)``) keeps the graph free from rule-4 batch-dim
        bookkeeping for this leaf tensor and adds no extra Unsqueeze nodes.
      * per-leaf ``(C_i, T_i, F_i)`` shapes and offsets are baked in at
        construction time, so unpack is a static list of ``Slice`` + ``Reshape``
        ops (both on the NPU-allowed list) and repack is per-leaf ``Reshape``
        plus one ``Concat``.

    The core ``forward_stream`` API is intentionally untouched; this wrapper is
    the only place that should be used for ONNX/MLIR export.
    """

    def __init__(self, core: DolphinSFCNPUSeparator, batch_size: int = 1, dtype: torch.dtype = torch.float32):
        super().__init__()
        from spectral_feature_compression.utils.onnx_streaming import flatten_tensor_tree

        self.core = core
        self.batch_size = batch_size
        example_state = core.init_stream_state(batch_size=batch_size, dtype=dtype)
        flat_state, state_spec = flatten_tensor_tree(example_state)
        self.state_spec = state_spec
        self.state_tensor_count = len(flat_state)

        per_shapes: list[tuple[int, ...]] = []
        per_numels: list[int] = []
        for tensor in flat_state:
            if tensor.shape[0] != batch_size:
                raise ValueError(
                    f"All leaf state tensors must start with batch dim {batch_size}; "
                    f"got {tuple(tensor.shape)}."
                )
            shape_wo_batch = tuple(int(d) for d in tensor.shape[1:])
            numel = int(tensor.numel() // max(batch_size, 1))
            if numel == 0:
                raise ValueError(
                    "DolphinSFCNPUStreamingExportWrapper does not support zero-sized leaf "
                    f"state tensors (shape {tuple(tensor.shape)}); such tensors carry no "
                    "information and break the static `Reshape(-1, ...)` used during unpack. "
                    "Drop them from the streaming state tree instead."
                )
            per_shapes.append(shape_wo_batch)
            per_numels.append(numel)
        self.per_shapes: tuple[tuple[int, ...], ...] = tuple(per_shapes)
        self.per_numels: tuple[int, ...] = tuple(per_numels)
        self.total_numel: int = sum(per_numels)

    # -- internal helpers ----------------------------------------------------

    def _unpack_state(self, packed: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Unpack ``(B, total_numel)`` into the tree of per-leaf state tensors."""
        from spectral_feature_compression.utils.onnx_streaming import unflatten_tensor_tree

        _runtime_assert(
            packed.ndim == 2 and int(packed.shape[1]) == self.total_numel,
            f"Expected packed state shape (B, {self.total_numel}), got {tuple(packed.shape)}",
        )
        leaves: list[torch.Tensor] = []
        offset = 0
        for numel, shape in zip(self.per_numels, self.per_shapes):
            # Static slice bounds -> ONNX ``Slice``; fixed reshape with ``-1`` for
            # the batch dim -> a single ``Reshape`` with a constant shape tensor.
            chunk = packed[:, offset : offset + numel]
            leaves.append(chunk.reshape((-1,) + shape))
            offset += numel
        return unflatten_tensor_tree(tuple(leaves), self.state_spec)

    def _pack_state(self, state_tree) -> torch.Tensor:
        """Pack the tree of per-leaf state tensors back into ``(B, total_numel)``."""
        from spectral_feature_compression.utils.onnx_streaming import flatten_tensor_tree

        flat, _ = flatten_tensor_tree(state_tree)
        _runtime_assert(
            len(flat) == self.state_tensor_count,
            f"State tree has {len(flat)} leaves but {self.state_tensor_count} were expected.",
        )
        # ``flatten(start_dim=1)`` lowers to a single ``Reshape`` per leaf,
        # then one ``Concat`` over dim=1 produces the packed state.
        flat_2d = [torch.flatten(t, start_dim=1) for t in flat]
        return torch.cat(flat_2d, dim=1)

    # -- public API ----------------------------------------------------------

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state_tree = self._unpack_state(state)
        y, new_state_tree = self.core.forward_stream(x, state_tree)
        packed_new_state = self._pack_state(new_state_tree)
        return y, packed_new_state

    def init_packed_state(
        self,
        batch_size: int | None = None,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Return a zero packed state compatible with :meth:`forward`."""
        b = batch_size if batch_size is not None else self.batch_size
        state_tree = self.core.init_stream_state(batch_size=b, device=device, dtype=dtype)
        return self._pack_state(state_tree)

    def pack_state(self, state_tree) -> torch.Tensor:
        """Public helper to convert a tree-state into the packed ONNX layout."""
        return self._pack_state(state_tree)

    def unpack_state(self, packed: torch.Tensor):
        """Public helper to convert a packed state back into the tree layout."""
        return self._unpack_state(packed)


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
