"""
Strict online Adaptive Mel-SFC Locoformer-Lite separator.

This is the deployable Proposal-A student path: adaptive overlapped mel SFC
compression/expansion plus a compact TF-Locoformer-like separator implemented
only with 2D convolutions, elementwise gates, and bmm inside the SFC router.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectral_feature_compression.core.model.adaptive_mel_sfc_2d import (
    AdaptiveMelBandSpec2d,
    _apply_packed_complex_mask_no_repeat,
    _as_pair,
)
from spectral_feature_compression.core.model.frequency_preprocessing import (
    FrequencyPreprocessedOnlineModel,
    build_frequency_preprocessor,
    build_pcen_preprocessor,
    resolve_preprocessed_n_freq,
)
from spectral_feature_compression.core.model.npu_capacity_blocks_2d import build_capacity_mixers
from spectral_feature_compression.core.model.online_model_wrapper import OnlineModelWrapper
from spectral_feature_compression.core.model.online_sfc_2d import (
    CausalConv2d,
    RMSNorm2d,
    _runtime_assert,
    _validate_npu_kernel_dilation_limit,
    pack_complex_stft_as_2d,
    unpack_2d_to_complex_stft,
)
from spectral_feature_compression.core.model.online_soft_band_query_sfc_2d import (
    SoftBandQueryCompressor2d,
    SoftBandQueryExpander2d,
)


def _normalize_dilation_schedule(n_layers: int, dilation_cycle: Sequence[int] | None) -> tuple[int, ...]:
    if n_layers <= 0:
        raise ValueError(f"n_layers must be positive, got {n_layers}")
    if dilation_cycle is None:
        dilation_cycle = (1, 2, 1, 2)
    cycle = tuple(int(v) for v in dilation_cycle)
    if len(cycle) == 0:
        raise ValueError("dilation_cycle must not be empty")
    if any(v <= 0 for v in cycle):
        raise ValueError(f"dilation values must be positive, got {cycle}")
    return tuple(cycle[layer_idx % len(cycle)] for layer_idx in range(n_layers))


class AdaptiveMelLocoformerLiteBlock2d(nn.Module):
    """NPU-safe TF-Locoformer-lite block over compressed mel-band tokens."""

    def __init__(
        self,
        *,
        channels: int,
        expansion: int = 2,
        ffn_expansion: int = 2,
        time_kernel_size: int = 3,
        band_kernel_size: int = 3,
        time_dilation: int = 1,
        causal: bool = True,
    ):
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if expansion <= 0 or ffn_expansion <= 0:
            raise ValueError("expansion factors must be positive")
        if band_kernel_size % 2 != 1:
            raise ValueError(f"band_kernel_size must be odd, got {band_kernel_size}")
        if not causal:
            raise ValueError("AdaptiveMelLocoformerLiteBlock2d currently supports causal=True only")
        _validate_npu_kernel_dilation_limit(time_kernel_size, time_dilation, axis="time")
        _validate_npu_kernel_dilation_limit(band_kernel_size, 1, axis="band")

        hidden = int(channels) * int(expansion)
        ffn_hidden = int(channels) * int(ffn_expansion)
        self.channels = int(channels)
        self.hidden = hidden
        self.time_kernel_size = int(time_kernel_size)
        self.band_kernel_size = int(band_kernel_size)
        self.time_dilation = int(time_dilation)

        self.time_norm = RMSNorm2d(channels)
        self.time_in = nn.Conv2d(channels, 2 * hidden, kernel_size=1, bias=True)
        self.time_dw = CausalConv2d(
            hidden,
            hidden,
            kernel_size=(time_kernel_size, 1),
            dilation=(time_dilation, 1),
            groups=hidden,
            bias=True,
        )
        self.time_out = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)

        self.band_norm = RMSNorm2d(channels)
        self.band_in = nn.Conv2d(channels, 2 * hidden, kernel_size=1, bias=True)
        self.band_dw = nn.Conv2d(
            hidden,
            hidden,
            kernel_size=(1, band_kernel_size),
            padding=(0, band_kernel_size // 2),
            groups=hidden,
            bias=True,
        )
        self.band_out = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)

        self.ffn_norm = RMSNorm2d(channels)
        self.ffn_in = nn.Conv2d(channels, 2 * ffn_hidden, kernel_size=1, bias=True)
        self.ffn_out = nn.Conv2d(ffn_hidden, channels, kernel_size=1, bias=True)

    @staticmethod
    def _gated(x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=1)
        return a * torch.sigmoid(b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self._gated(self.time_in(self.time_norm(x)))
        y = F.silu(self.time_dw(y))
        x = x + self.time_out(y)

        y = self._gated(self.band_in(self.band_norm(x)))
        y = F.silu(self.band_dw(y))
        x = x + self.band_out(y)

        y = self._gated(self.ffn_in(self.ffn_norm(x)))
        return x + self.ffn_out(y)

    def stream_context_frames(self) -> int:
        return self.time_dw.stream_context_frames()

    def init_stream_state(
        self,
        batch_size: int = 1,
        *,
        freq_bins: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        return self.time_dw.init_stream_state(batch_size, freq_bins=freq_bins, device=device, dtype=dtype)

    def forward_stream(self, x: torch.Tensor, state: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        y = self._gated(self.time_in(self.time_norm(x)))
        y, new_state = self.time_dw.forward_stream(y, state)
        y = F.silu(y)
        x = x + self.time_out(y)

        y = self._gated(self.band_in(self.band_norm(x)))
        y = F.silu(self.band_dw(y))
        x = x + self.band_out(y)

        y = self._gated(self.ffn_in(self.ffn_norm(x)))
        return x + self.ffn_out(y), new_state


class OnlineAdaptiveMelLocoformerLiteSFC2D(nn.Module):
    """Adaptive overlapped-mel SFC with strict online Locoformer-lite blocks."""

    def __init__(
        self,
        n_freq: int,
        n_bands: int = 80,
        n_fft: int | None = None,
        sample_rate: int = 44100,
        n_src: int = 3,
        n_chan: int = 1,
        d_model: int = 32,
        n_layers: int = 4,
        kernel_size: Sequence[int] | tuple[int, int] = (3, 3),
        routing_kernel_size: Sequence[int] | tuple[int, int] = (1, 3),
        dilation_cycle: Sequence[int] | None = (1, 2, 1, 2),
        expansion: int = 2,
        ffn_expansion: int = 2,
        capacity_mixer_hidden: int = 0,
        capacity_mixer_layers: int = 0,
        low_freq_hz: float = 1000.0,
        low_freq_band_fraction: float = 0.45,
        overlap_factor: float = 1.5,
        low_freq_overlap_factor: float = 2.0,
        causal: bool = True,
        masking: bool = True,
        routing_normalization: str = "softmax",
    ):
        super().__init__()
        del n_fft
        if not causal:
            raise ValueError("OnlineAdaptiveMelLocoformerLiteSFC2D supports causal=True only")
        kernel_size = _as_pair(kernel_size, name="kernel_size")
        routing_kernel_size = _as_pair(routing_kernel_size, name="routing_kernel_size")
        self.n_freq = int(n_freq)
        self.n_bands = int(n_bands)
        self.n_src = int(n_src)
        self.n_chan = int(n_chan)
        self.masking = bool(masking)
        self.causal = bool(causal)
        self.dilation_schedule = _normalize_dilation_schedule(n_layers, dilation_cycle)

        band_spec = AdaptiveMelBandSpec2d(
            n_freq=n_freq,
            n_bands=n_bands,
            sample_rate=sample_rate,
            low_freq_hz=low_freq_hz,
            low_freq_band_fraction=low_freq_band_fraction,
            overlap_factor=overlap_factor,
            low_freq_overlap_factor=low_freq_overlap_factor,
        )
        self.band_spec = band_spec
        self.in_proj = nn.Sequential(
            nn.Conv2d(2 * n_chan, d_model, kernel_size=1, bias=True),
            RMSNorm2d(d_model),
        )
        self.compressor = SoftBandQueryCompressor2d(
            channels=d_model,
            band_spec=band_spec,
            kernel_size=routing_kernel_size,
            causal=causal,
            normalization=routing_normalization,
        )
        self.separator = nn.ModuleList(
            [
                AdaptiveMelLocoformerLiteBlock2d(
                    channels=d_model,
                    expansion=expansion,
                    ffn_expansion=ffn_expansion,
                    time_kernel_size=kernel_size[0],
                    band_kernel_size=kernel_size[1],
                    time_dilation=dilation,
                    causal=causal,
                )
                for dilation in self.dilation_schedule
            ]
        )
        self.capacity_mixers = build_capacity_mixers(
            channels=d_model,
            hidden_channels=capacity_mixer_hidden,
            n_layers=capacity_mixer_layers,
        )
        self.expander = SoftBandQueryExpander2d(channels=d_model, band_spec=band_spec)
        self.out_proj = nn.Conv2d(d_model, 2 * n_src * n_chan, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _runtime_assert(x.ndim == 4, f"Expected 4D input (B,C,T,F), got {x.shape}")
        _runtime_assert(x.shape[-1] == self.n_freq, f"{x.shape} vs {self.n_freq}")
        h = self.in_proj(x)
        z, query_tokens = self.compressor(h)
        for block_idx, block in enumerate(self.separator):
            z = block(z)
            if block_idx < len(self.capacity_mixers):
                z = self.capacity_mixers[block_idx](z)
        for block_idx in range(len(self.separator), len(self.capacity_mixers)):
            z = self.capacity_mixers[block_idx](z)
        y = self.out_proj(self.expander(z, query_tokens))
        if self.masking:
            return _apply_packed_complex_mask_no_repeat(x=x, y=y, n_src=self.n_src, n_chan=self.n_chan)
        return y

    def stream_context_frames(self) -> int:
        return self.compressor.stream_context_frames() + sum(block.stream_context_frames() for block in self.separator)

    def init_stream_state(
        self,
        batch_size: int = 1,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, ...]:
        comp = self.compressor.init_stream_state(batch_size, freq_bins=self.n_freq, device=device, dtype=dtype)
        sep = tuple(
            block.init_stream_state(batch_size, freq_bins=self.n_bands, device=device, dtype=dtype)
            for block in self.separator
        )
        return (comp, *sep)

    def forward_stream(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, ...] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        _runtime_assert(x.ndim == 4, f"Expected 4D input (B,C,T,F), got {x.shape}")
        _runtime_assert(x.shape[-1] == self.n_freq, f"{x.shape} vs {self.n_freq}")
        if state is None:
            state = self.init_stream_state(batch_size=x.shape[0], device=x.device, dtype=x.dtype)
        _runtime_assert(len(state) == 1 + len(self.separator), f"Unexpected state tuple: {len(state)}")

        h = self.in_proj(x)
        (z, query_tokens), new_comp_state = self.compressor.forward_stream(h, state[0])
        new_sep_states = []
        for block_idx, (block, block_state) in enumerate(zip(self.separator, state[1:])):
            z, block_state = block.forward_stream(z, block_state)
            if block_idx < len(self.capacity_mixers):
                z = self.capacity_mixers[block_idx](z)
            new_sep_states.append(block_state)
        for block_idx in range(len(self.separator), len(self.capacity_mixers)):
            z = self.capacity_mixers[block_idx](z)
        y = self.out_proj(self.expander(z, query_tokens))
        if self.masking:
            y = _apply_packed_complex_mask_no_repeat(x=x, y=y, n_src=self.n_src, n_chan=self.n_chan)
        return y, (new_comp_state, *new_sep_states)

    def init_input_history(
        self,
        batch_size: int = 1,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        history_frames = self.stream_context_frames()
        return torch.zeros(batch_size, 2 * self.n_chan, history_frames, self.n_freq, device=device, dtype=dtype)

    def forward_stream_recompute(
        self,
        x: torch.Tensor,
        history: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError(
            "Exact low-memory recomputation from raw input history is not implemented for "
            "OnlineAdaptiveMelLocoformerLiteSFC2D. Use forward_stream with layer caches for strict equivalence."
        )

    def layer_cache_numel(self, batch_size: int = 1) -> int:
        states = self.init_stream_state(
            batch_size=batch_size,
            device=self.out_proj.weight.device,
            dtype=self.out_proj.weight.dtype,
        )
        return sum(int(state.numel()) for state in states)

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


class OnlineAdaptiveMelLocoformerLiteSFCModel(nn.Module):
    """Complex-STFT wrapper around OnlineAdaptiveMelLocoformerLiteSFC2D."""

    def __init__(self, *, n_freq: int, n_src: int = 3, n_chan: int = 1, **kwargs):
        super().__init__()
        self.core = OnlineAdaptiveMelLocoformerLiteSFC2D(n_freq=n_freq, n_src=n_src, n_chan=n_chan, **kwargs)
        self.n_src = n_src
        self.n_chan = n_chan

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        del kwargs
        x2d = pack_complex_stft_as_2d(x)
        y2d = self.core(x2d)
        return unpack_2d_to_complex_stft(y2d, n_src=self.n_src, n_chan=self.n_chan)


def build_adaptive_mel_locoformer_lite_sfc_system(
    *,
    n_fft: int,
    hop_length: int,
    fs: int,
    n_src: int = 3,
    n_chan: int = 1,
    n_bands: int = 80,
    d_model: int = 32,
    n_layers: int = 4,
    kernel_size: Sequence[int] | tuple[int, int] = (3, 3),
    routing_kernel_size: Sequence[int] | tuple[int, int] = (1, 3),
    dilation_cycle: Sequence[int] | None = (1, 2, 1, 2),
    expansion: int = 2,
    ffn_expansion: int = 2,
    capacity_mixer_hidden: int = 6144,
    capacity_mixer_layers: int = 4,
    low_freq_hz: float = 1000.0,
    low_freq_band_fraction: float = 0.45,
    overlap_factor: float = 1.5,
    low_freq_overlap_factor: float = 2.0,
    causal: bool = True,
    masking: bool = True,
    routing_normalization: str = "softmax",
    freq_preprocess_enabled: bool = True,
    freq_preprocess_keep_bins: int | None = 475,
    freq_preprocess_target_bins: int | None = 512,
    freq_preprocess_mode: str = "triangular",
    dc_bypass_enabled: bool = False,
    dc_policy: str = "zero",
    pcen_preprocess_enabled: bool = False,
    pcen_smooth_coef: float = 0.98,
    pcen_alpha: float = 0.5,
    pcen_delta: float = 2.0,
    pcen_root: float = 0.5,
    pcen_eps: float = 1e-6,
    pcen_gain_floor: float = 0.05,
    pcen_gain_ceiling: float = 20.0,
    scaling: bool = False,
    css_segment_size: int = 12,
    css_shift_size: int = 6,
    css_batch_size: int = 1,
) -> OnlineModelWrapper:
    full_n_freq = (n_fft // 2) + 1
    core_n_freq = resolve_preprocessed_n_freq(
        full_n_freq,
        enabled=freq_preprocess_enabled,
        keep_bins=freq_preprocess_keep_bins,
        target_bins=freq_preprocess_target_bins,
        dc_bypass_enabled=dc_bypass_enabled,
    )
    core_n_fft = 2 * (core_n_freq - 1)
    freq_preprocessor = build_frequency_preprocessor(
        full_n_freq,
        enabled=freq_preprocess_enabled,
        keep_bins=freq_preprocess_keep_bins,
        target_bins=freq_preprocess_target_bins,
        mode=freq_preprocess_mode,
        dc_bypass_enabled=dc_bypass_enabled,
    )
    pcen_preprocessor = build_pcen_preprocessor(
        n_chan=n_chan,
        enabled=pcen_preprocess_enabled,
        smooth_coef=pcen_smooth_coef,
        alpha=pcen_alpha,
        delta=pcen_delta,
        root=pcen_root,
        eps=pcen_eps,
        gain_floor=pcen_gain_floor,
        gain_ceiling=pcen_gain_ceiling,
    )
    core = OnlineAdaptiveMelLocoformerLiteSFC2D(
        n_freq=core_n_freq,
        n_bands=n_bands,
        n_fft=core_n_fft,
        sample_rate=fs,
        n_src=n_src,
        n_chan=n_chan,
        d_model=d_model,
        n_layers=n_layers,
        kernel_size=kernel_size,
        routing_kernel_size=routing_kernel_size,
        dilation_cycle=dilation_cycle,
        expansion=expansion,
        ffn_expansion=ffn_expansion,
        capacity_mixer_hidden=capacity_mixer_hidden,
        capacity_mixer_layers=capacity_mixer_layers,
        low_freq_hz=low_freq_hz,
        low_freq_band_fraction=low_freq_band_fraction,
        overlap_factor=overlap_factor,
        low_freq_overlap_factor=low_freq_overlap_factor,
        causal=causal,
        masking=masking,
        routing_normalization=routing_normalization,
    )
    model = FrequencyPreprocessedOnlineModel(
        core=core,
        n_src=n_src,
        n_chan=n_chan,
        freq_preprocessor=freq_preprocessor,
        pcen_preprocessor=pcen_preprocessor,
        dc_bypass_enabled=dc_bypass_enabled,
        dc_policy=dc_policy,
    )
    return OnlineModelWrapper(
        model=model,
        n_fft=n_fft,
        hop_length=hop_length,
        fs=fs,
        scaling=scaling,
        css_segment_size=css_segment_size,
        css_shift_size=css_shift_size,
        css_batch_size=css_batch_size,
    )
