"""BandSFCNetNPU core.

This model combines the deployment-friendly SFC frequency transport with the
BandSCNetNPU separation prior:

* SFC-style F -> K compression and K -> F expansion keep the recurrent state
  tied to a small latent band axis.
* BandSCNet-style cross-band and narrow-band stages model local frequency and
  causal temporal structure on those latent bands.
* Optional frequency-pooled channel mixers add trainable capacity without
  increasing persistent streaming state.

The deployable core operates on packed-real 2D STFT tensors shaped
``[B, 2*n_chan, T, F]`` and returns packed complex estimates or raw mask logits
depending on ``masking``.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from BandSCNetNPU.blocks import BoundedCausalAttn, CrossBandBlock, GatedAct, PooledChannelMixer
from spectral_feature_compression.core.model.online_crossattn_query_sfc_2d import (
    NPUSafeCrossAttnDecoder2d,
    NPUSafeCrossAttnEncoder2d,
)
from spectral_feature_compression.core.model.online_sfc_2d import (
    CausalConv2d,
    RMSNorm2d,
    _runtime_assert,
    pack_complex_stft_as_2d,
    unpack_2d_to_complex_stft,
)
from spectral_feature_compression.core.model.online_soft_band_sfc_2d import (
    SoftBandCompressor2d,
    SoftBandSpec2d,
)


def _normalize_dilation_schedule(
    num_stages: int,
    dilation_cycle: tuple[int, ...] | list[int] | None,
) -> tuple[int, ...]:
    if dilation_cycle is None:
        dilation_cycle = (1, 1, 2, 4, 6)
    cycle = tuple(int(d) for d in dilation_cycle)
    if len(cycle) == 0:
        raise ValueError("dilation_cycle must not be empty")
    if any(d <= 0 for d in cycle):
        raise ValueError(f"All dilations must be positive, got {cycle}")
    return tuple(cycle[idx % len(cycle)] for idx in range(num_stages))


def _apply_packed_complex_mask_no_repeat(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    n_src: int,
    n_chan: int,
) -> torch.Tensor:
    """Packed complex multiply without repeat/Tile-prone expansion."""

    _runtime_assert(x.shape[1] == 2 * n_chan, f"{x.shape[1]} vs {2 * n_chan}")
    _runtime_assert(y.shape[1] == 2 * n_src * n_chan, f"{y.shape[1]} vs {2 * n_src * n_chan}")
    outputs: list[torch.Tensor] = []
    for src_idx in range(n_src):
        for chan_idx in range(n_chan):
            in_r = x[:, 2 * chan_idx : 2 * chan_idx + 1, :, :]
            in_i = x[:, 2 * chan_idx + 1 : 2 * chan_idx + 2, :, :]
            mask_base = 2 * (src_idx * n_chan + chan_idx)
            mask_r = y[:, mask_base : mask_base + 1, :, :]
            mask_i = y[:, mask_base + 1 : mask_base + 2, :, :]
            outputs.append(in_r * mask_r - in_i * mask_i)
            outputs.append(in_r * mask_i + in_i * mask_r)
    return torch.cat(outputs, dim=1)


class DilatedNarrowBandBlock(nn.Module):
    """BandSCNet narrow-band temporal block with optional NPU-valid dilation."""

    def __init__(
        self,
        channels: int,
        *,
        time_kernel: int = 3,
        time_dilation: int = 1,
        use_attn: bool = False,
        attn_window: int = 16,
        num_heads: int = 4,
        head_dim: int = 8,
    ):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError(f"channels must be even, got {channels}")
        self.channels = channels
        self.time_kernel = time_kernel
        self.time_dilation = time_dilation
        self.use_attn = use_attn

        self.norm = RMSNorm2d(channels)
        self.causal_dw = CausalConv2d(
            channels,
            2 * channels,
            kernel_size=(time_kernel, 1),
            dilation=(time_dilation, 1),
            groups=channels,
            bias=True,
        )
        self.act = GatedAct()
        self.attn = (
            BoundedCausalAttn(channels, window=attn_window, num_heads=num_heads, head_dim=head_dim)
            if use_attn
            else None
        )
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _runtime_assert(x.ndim == 4, f"Expected 4D input, got {tuple(x.shape)}")
        y = self.norm(x)
        y = self.causal_dw(y)
        y = self.act(y)
        if self.attn is not None:
            y = y + self.attn(x)
        y = self.pointwise(y)
        return x + y

    def stream_context_frames(self) -> int:
        return self.causal_dw.stream_context_frames()

    def init_stream_state(
        self,
        batch_size: int,
        *,
        freq_bins: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, ...]:
        dw_state = self.causal_dw.init_stream_state(
            batch_size,
            freq_bins=freq_bins,
            device=device,
            dtype=dtype,
        )
        if self.attn is None:
            return (dw_state,)
        attn_state = self.attn.init_stream_state(
            batch_size,
            freq_bins=freq_bins,
            device=device,
            dtype=dtype,
        )
        return (dw_state, attn_state)

    def forward_stream(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, ...] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        _runtime_assert(x.ndim == 4, f"Expected 4D input, got {tuple(x.shape)}")
        if state is None:
            state = self.init_stream_state(
                x.shape[0],
                freq_bins=x.shape[-1],
                device=x.device,
                dtype=x.dtype,
            )
        y = self.norm(x)
        y, new_dw_state = self.causal_dw.forward_stream(y, state[0])
        y = self.act(y)
        if self.attn is not None:
            attn_out, new_attn_state = self.attn.forward_stream(x, state[1])
            y = y + attn_out
            new_state: tuple[torch.Tensor, ...] = (new_dw_state, new_attn_state)
        else:
            new_state = (new_dw_state,)
        y = self.pointwise(y)
        return x + y, new_state


class BandSFCStage(nn.Module):
    """Cross-band, dilated narrow-band, and optional pooled capacity branch."""

    def __init__(
        self,
        channels: int,
        *,
        time_kernel: int,
        freq_kernel: int,
        time_dilation: int,
        use_attn: bool,
        attn_window: int,
        num_heads: int,
        head_dim: int,
        pooled_mixer_hidden: int,
    ):
        super().__init__()
        self.cross = CrossBandBlock(channels, freq_kernel=freq_kernel)
        self.narrow = DilatedNarrowBandBlock(
            channels,
            time_kernel=time_kernel,
            time_dilation=time_dilation,
            use_attn=use_attn,
            attn_window=attn_window,
            num_heads=num_heads,
            head_dim=head_dim,
        )
        self.pooled_mixer = (
            PooledChannelMixer(channels, pooled_mixer_hidden)
            if pooled_mixer_hidden > 0
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cross(x)
        x = self.narrow(x)
        return self.pooled_mixer(x)

    def stream_context_frames(self) -> int:
        return self.narrow.stream_context_frames()

    def init_stream_state(
        self,
        batch_size: int,
        *,
        freq_bins: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, ...]:
        return self.narrow.init_stream_state(batch_size, freq_bins=freq_bins, device=device, dtype=dtype)

    def forward_stream(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        x = self.cross(x)
        x, state = self.narrow.forward_stream(x, state)
        return self.pooled_mixer(x), state


class SoftBandExpanderNoClip2d(nn.Module):
    """Soft-band K -> F expander that avoids ONNX Clip from clamp_min."""

    def __init__(self, channels: int, band_spec: SoftBandSpec2d):
        super().__init__()
        self.channels = channels
        self.band_spec = band_spec
        self.n_bands = band_spec.n_bands
        self.n_freq = band_spec.n_freq
        self.pre_norm = RMSNorm2d(channels)
        self.pre_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.band_gain = nn.Conv2d(channels, 1, kernel_size=1, bias=True)
        self.gain_scale = nn.Parameter(torch.tensor(1.0))
        self.basis_scale = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("expansion_basis", band_spec.expansion_basis())

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        _runtime_assert(z.shape[-1] == self.n_bands, f"{z.shape} vs {self.n_bands}")
        h = self.pre_proj(self.pre_norm(z))
        h = h * torch.sigmoid(h)
        batch, channels, n_frames, n_bands = h.shape

        gains = 1.0 + torch.sigmoid(self.band_gain(h)) * self.gain_scale
        gains = gains.permute(0, 3, 2, 1)
        coeff = self.expansion_basis * (self.basis_scale + gains)
        denom = coeff.sum(dim=1, keepdim=True) + 1e-6
        coeff = coeff / denom

        h_btck = h.permute(0, 2, 1, 3).reshape(batch * n_frames, channels, n_bands)
        coeff_btkf = coeff.permute(0, 2, 1, 3).reshape(batch * n_frames, n_bands, self.n_freq)
        expanded_btcf = torch.bmm(h_btck, coeff_btkf)
        return expanded_btcf.reshape(batch, n_frames, channels, self.n_freq).permute(0, 2, 1, 3)


class BandSFCNetNPU(nn.Module):
    """Online NPU core with SFC transport and BandSCNet latent separation."""

    def __init__(
        self,
        n_freq: int,
        *,
        n_bands: int = 64,
        n_fft: int | None = None,
        sample_rate: int | None = None,
        band_config: str = "musical",
        n_src: int = 3,
        n_chan: int = 1,
        channels: int = 32,
        num_stages: int = 4,
        time_kernel: int = 3,
        freq_kernel: int = 3,
        dilation_cycle: tuple[int, ...] | list[int] | None = None,
        transport: str = "soft",
        query_type: str = "adaptive",
        routing_normalization: str = "softmax",
        use_attn: bool = False,
        attn_window: int = 16,
        num_heads: int = 4,
        head_dim: int = 8,
        pooled_mixer_hidden: int = 0,
        causal: bool = True,
        masking: bool = True,
    ):
        super().__init__()
        if transport not in {"soft", "crossattn"}:
            raise ValueError(f"transport must be 'soft' or 'crossattn', got {transport!r}")
        if not causal:
            raise ValueError("BandSFCNetNPU currently targets causal online deployment only")
        if channels % 2 != 0:
            raise ValueError(f"channels must be even, got {channels}")

        self.n_freq = n_freq
        self.n_bands = n_bands
        self.n_src = n_src
        self.n_chan = n_chan
        self.channels = channels
        self.num_stages = num_stages
        self.transport = transport
        self.causal = causal
        self.masking = masking
        self.dilation_schedule = _normalize_dilation_schedule(num_stages, dilation_cycle)

        in_ch = 2 * n_chan
        out_ch = 2 * n_src * n_chan
        self.in_proj = nn.Sequential(
            nn.Conv2d(in_ch, channels, kernel_size=1, bias=True),
            RMSNorm2d(channels),
        )
        band_spec = SoftBandSpec2d(
            n_freq=n_freq,
            n_bands=n_bands,
            n_fft=n_fft,
            sample_rate=sample_rate,
            band_config=band_config,
        )

        if transport == "soft":
            self.encoder = SoftBandCompressor2d(
                channels=channels,
                band_spec=band_spec,
                kernel_size=(time_kernel, freq_kernel),
                causal=causal,
                normalization=routing_normalization,
            )
            self.decoder = SoftBandExpanderNoClip2d(channels=channels, band_spec=band_spec)
        else:
            self.encoder = NPUSafeCrossAttnEncoder2d(
                channels=channels,
                band_spec=band_spec,
                kernel_size=(time_kernel, freq_kernel),
                causal=causal,
                query_type=query_type,
                routing_normalization=routing_normalization,
            )
            self.decoder = NPUSafeCrossAttnDecoder2d(
                channels=channels,
                band_spec=band_spec,
                query_type=query_type,
                routing_normalization=routing_normalization,
            )

        self.stages = nn.ModuleList(
            [
                BandSFCStage(
                    channels,
                    time_kernel=time_kernel,
                    freq_kernel=freq_kernel,
                    time_dilation=dilation,
                    use_attn=use_attn,
                    attn_window=attn_window,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    pooled_mixer_hidden=pooled_mixer_hidden,
                )
                for dilation in self.dilation_schedule
            ]
        )
        self.out_proj = nn.Conv2d(channels, out_ch, kernel_size=1, bias=True)

    def _encode(self, x: torch.Tensor):
        if self.transport == "soft":
            z, _weights = self.encoder(x)
            return z, None
        return self.encoder(x)

    def _encode_stream(self, x: torch.Tensor, state: torch.Tensor | None):
        if self.transport == "soft":
            z, new_state = self.encoder.forward_stream(x, state)
            return z, None, new_state
        (z, side), new_state = self.encoder.forward_stream(x, state)
        return z, side, new_state

    def _decode(self, z: torch.Tensor, side: torch.Tensor | None) -> torch.Tensor:
        if self.transport == "soft":
            return self.decoder(z)
        if side is None:
            raise RuntimeError("crossattn transport requires a decoder side path")
        return self.decoder(z, side)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _runtime_assert(x.ndim == 4, f"Expected 4D input (B,C,T,F), got {tuple(x.shape)}")
        _runtime_assert(x.shape[1] == 2 * self.n_chan, f"{x.shape[1]} vs {2 * self.n_chan}")
        _runtime_assert(x.shape[-1] == self.n_freq, f"{x.shape[-1]} vs {self.n_freq}")

        h = self.in_proj(x)
        z, side = self._encode(h)
        for stage in self.stages:
            z = stage(z)
        h = self._decode(z, side)
        y = self.out_proj(h)
        if self.masking:
            return _apply_packed_complex_mask_no_repeat(x=x, y=y, n_src=self.n_src, n_chan=self.n_chan)
        return y

    def stream_context_frames(self) -> int:
        return self.encoder.stream_context_frames() + sum(stage.stream_context_frames() for stage in self.stages)

    def init_stream_state(
        self,
        batch_size: int = 1,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, ...]:
        enc = self.encoder.init_stream_state(batch_size, freq_bins=self.n_freq, device=device, dtype=dtype)
        sep = tuple(
            stage.init_stream_state(batch_size, freq_bins=self.n_bands, device=device, dtype=dtype)
            for stage in self.stages
        )
        return (enc, *sep)

    def forward_stream(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, ...] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        _runtime_assert(x.ndim == 4, f"Expected 4D input, got {tuple(x.shape)}")
        _runtime_assert(x.shape[-1] == self.n_freq, f"{x.shape[-1]} vs {self.n_freq}")
        if state is None:
            state = self.init_stream_state(batch_size=x.shape[0], device=x.device, dtype=x.dtype)

        h = self.in_proj(x)
        z, side, new_enc_state = self._encode_stream(h, state[0])
        new_sep_states: list[tuple[torch.Tensor, ...]] = []
        for stage, stage_state in zip(self.stages, state[1:]):
            z, new_stage_state = stage.forward_stream(z, stage_state)
            new_sep_states.append(new_stage_state)
        h = self._decode(z, side)
        y = self.out_proj(h)
        if self.masking:
            y = _apply_packed_complex_mask_no_repeat(x=x, y=y, n_src=self.n_src, n_chan=self.n_chan)
        return y, (new_enc_state, *new_sep_states)

    def layer_cache_numel(self, batch_size: int = 1) -> int:
        states = self.init_stream_state(
            batch_size=batch_size,
            device=self.out_proj.weight.device,
            dtype=self.out_proj.weight.dtype,
        )
        return _tree_numel(states)

    def input_history_numel(self, batch_size: int = 1) -> int:
        return batch_size * 2 * self.n_chan * self.stream_context_frames() * self.n_freq

    def state_size_bytes(
        self,
        *,
        batch_size: int = 1,
        dtype: torch.dtype = torch.float16,
        mode: str = "layer_cache",
    ) -> int:
        element_size = torch.tensor([], dtype=dtype).element_size()
        if mode == "layer_cache":
            return self.layer_cache_numel(batch_size=batch_size) * element_size
        if mode == "input_history":
            return self.input_history_numel(batch_size=batch_size) * element_size
        raise ValueError(f"Unsupported state mode: {mode}")


class BandSFCNetNPUModel(nn.Module):
    """Complex-STFT wrapper matching the existing online model contract."""

    def __init__(self, core: BandSFCNetNPU):
        super().__init__()
        self.core = core
        self.n_src = core.n_src
        self.n_chan = core.n_chan

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x2d = pack_complex_stft_as_2d(x)
        y2d = self.core(x2d)
        return unpack_2d_to_complex_stft(y2d, n_src=self.n_src, n_chan=self.n_chan)

    def init_stream_state(self, batch_size: int = 1, *, device=None, dtype=None):
        return self.core.init_stream_state(batch_size=batch_size, device=device, dtype=dtype)

    def forward_stream(self, x2d: torch.Tensor, state=None):
        return self.core.forward_stream(x2d, state)


def _tree_numel(tree) -> int:
    if isinstance(tree, torch.Tensor):
        return int(tree.numel())
    return sum(_tree_numel(item) for item in tree)
