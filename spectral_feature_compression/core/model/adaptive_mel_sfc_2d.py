"""
Adaptive overlapped mel-band SFC front-end.

This module makes the band-mapping ablation explicit instead of relying on the
generic ``band_config: mel`` path.  It exposes 80-band overlapped mel routing by
default, low-frequency band allocation, and extra low-frequency overlap for bass
and music preservation while keeping the same 4D online tensor contract.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

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
    OnlineConvBlock,
    RMSNorm2d,
    _runtime_assert,
    pack_complex_stft_as_2d,
    unpack_2d_to_complex_stft,
)
from spectral_feature_compression.core.model.online_soft_band_query_sfc_2d import (
    SoftBandQueryCompressor2d,
    SoftBandQueryExpander2d,
)


def _as_pair(value: Sequence[int] | int, *, name: str) -> tuple[int, int]:
    pair = (value, value) if isinstance(value, int) else tuple(int(v) for v in value)
    if len(pair) != 2:
        raise ValueError(f"{name} must contain exactly two values, got {value}.")
    return pair


def _hz_to_mel(hz: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (torch.pow(10.0, mel / 2595.0) - 1.0)


def _apply_packed_complex_mask_no_repeat(
    *,
    x: torch.Tensor,
    y: torch.Tensor,
    n_src: int,
    n_chan: int,
) -> torch.Tensor:
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


class AdaptiveMelBandSpec2d(nn.Module):
    """Explicit overlapped mel basis with low-frequency density controls."""

    def __init__(
        self,
        *,
        n_freq: int,
        n_bands: int = 80,
        sample_rate: int = 44100,
        f_min_hz: float = 0.0,
        f_max_hz: float | None = None,
        low_freq_hz: float = 1000.0,
        low_freq_band_fraction: float = 0.45,
        overlap_factor: float = 1.5,
        low_freq_overlap_factor: float = 2.0,
    ):
        super().__init__()
        if n_freq <= 0:
            raise ValueError(f"n_freq must be positive, got {n_freq}.")
        if n_bands <= 1:
            raise ValueError(f"n_bands must be > 1, got {n_bands}.")
        nyquist = 0.5 * float(sample_rate)
        if nyquist <= 0.0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}.")
        f_max = nyquist if f_max_hz is None else min(float(f_max_hz), nyquist)
        f_min = max(0.0, min(float(f_min_hz), f_max - 1.0))
        low_freq = max(f_min + 1.0, min(float(low_freq_hz), f_max - 1.0))
        if not (0.0 < low_freq_band_fraction < 1.0):
            raise ValueError(f"low_freq_band_fraction must be in (0, 1), got {low_freq_band_fraction}.")
        if overlap_factor < 1.0 or low_freq_overlap_factor < 1.0:
            raise ValueError("overlap factors must be >= 1.0")

        basis, centers_hz = self._build_basis(
            n_freq=n_freq,
            n_bands=n_bands,
            nyquist=nyquist,
            f_min_hz=f_min,
            f_max_hz=f_max,
            low_freq_hz=low_freq,
            low_freq_band_fraction=low_freq_band_fraction,
            overlap_factor=overlap_factor,
            low_freq_overlap_factor=low_freq_overlap_factor,
        )
        starts, ends = self._basis_bounds(basis)

        self.n_freq = int(n_freq)
        self.n_bands = int(n_bands)
        self.sample_rate = int(sample_rate)
        self.f_min_hz = float(f_min)
        self.f_max_hz = float(f_max)
        self.low_freq_hz = float(low_freq)
        self.low_freq_band_fraction = float(low_freq_band_fraction)
        self.overlap_factor = float(overlap_factor)
        self.low_freq_overlap_factor = float(low_freq_overlap_factor)
        self.register_buffer("starts", starts)
        self.register_buffer("ends", ends)
        self.register_buffer("centers_hz", centers_hz)
        self.register_buffer("basis", basis.view(1, n_bands, 1, n_freq))

    @staticmethod
    def _build_basis(
        *,
        n_freq: int,
        n_bands: int,
        nyquist: float,
        f_min_hz: float,
        f_max_hz: float,
        low_freq_hz: float,
        low_freq_band_fraction: float,
        overlap_factor: float,
        low_freq_overlap_factor: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_low = int(round(n_bands * low_freq_band_fraction))
        n_low = min(max(n_low, 1), n_bands - 1)
        n_high = n_bands - n_low

        mel_min = _hz_to_mel(torch.tensor(f_min_hz, dtype=torch.float32))
        mel_low = _hz_to_mel(torch.tensor(low_freq_hz, dtype=torch.float32))
        mel_max = _hz_to_mel(torch.tensor(f_max_hz, dtype=torch.float32))
        low_centers = torch.linspace(mel_min, mel_low, steps=n_low + 2, dtype=torch.float32)[1:-1]
        high_centers = torch.linspace(mel_low, mel_max, steps=n_high + 2, dtype=torch.float32)[1:-1]
        centers_mel = torch.cat([low_centers, high_centers], dim=0)

        center_hz = _mel_to_hz(centers_mel).clamp(f_min_hz, f_max_hz)
        freqs_hz = torch.linspace(0.0, nyquist, steps=n_freq, dtype=torch.float32)
        freqs_mel = _hz_to_mel(freqs_hz)

        midpoint_left = torch.empty_like(centers_mel)
        midpoint_right = torch.empty_like(centers_mel)
        midpoint_left[0] = mel_min
        midpoint_left[1:] = 0.5 * (centers_mel[:-1] + centers_mel[1:])
        midpoint_right[:-1] = 0.5 * (centers_mel[:-1] + centers_mel[1:])
        midpoint_right[-1] = mel_max

        basis = torch.zeros(n_bands, n_freq, dtype=torch.float32)
        for band_idx in range(n_bands):
            center = centers_mel[band_idx]
            factor = low_freq_overlap_factor if center_hz[band_idx] <= low_freq_hz else overlap_factor
            left = center - (center - midpoint_left[band_idx]) * factor
            right = center + (midpoint_right[band_idx] - center) * factor
            rising = (freqs_mel - left) / (center - left).clamp_min(1e-6)
            falling = (right - freqs_mel) / (right - center).clamp_min(1e-6)
            tri = torch.minimum(rising, falling).clamp(min=0.0)
            tri = torch.where((freqs_hz >= f_min_hz) & (freqs_hz <= f_max_hz), tri, torch.zeros_like(tri))
            if float(tri.max().item()) <= 0.0:
                nearest = int(torch.argmin(torch.abs(freqs_hz - center_hz[band_idx])).item())
                tri[nearest] = 1.0
            basis[band_idx] = tri
        return basis, center_hz

    @staticmethod
    def _basis_bounds(basis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        starts = []
        ends = []
        n_freq = basis.shape[1]
        for band in basis:
            active = torch.nonzero(band > 0.0, as_tuple=False).flatten()
            if active.numel() == 0:
                starts.append(0)
                ends.append(1)
            else:
                starts.append(int(active[0].item()))
                ends.append(min(int(active[-1].item()) + 1, n_freq))
        return torch.tensor(starts, dtype=torch.long), torch.tensor(ends, dtype=torch.long)

    def routing_bias(self) -> torch.Tensor:
        peak = self.basis.amax(dim=-1, keepdim=True).clamp_min(1e-6)
        return 2.0 * (self.basis / peak) - 1.0

    def expansion_basis(self) -> torch.Tensor:
        return self.basis / self.basis.sum(dim=1, keepdim=True).clamp_min(1e-6)

    def manifest(self) -> dict[str, object]:
        return {
            "type": "adaptive_overlapped_mel",
            "n_freq": self.n_freq,
            "n_bands": self.n_bands,
            "sample_rate": self.sample_rate,
            "f_min_hz": self.f_min_hz,
            "f_max_hz": self.f_max_hz,
            "low_freq_hz": self.low_freq_hz,
            "low_freq_band_fraction": self.low_freq_band_fraction,
            "overlap_factor": self.overlap_factor,
            "low_freq_overlap_factor": self.low_freq_overlap_factor,
        }


class OnlineAdaptiveMelSFC2D(nn.Module):
    """Soft-band query SFC using explicit overlapped/adaptive mel bands."""

    def __init__(
        self,
        n_freq: int,
        n_bands: int = 80,
        n_fft: int | None = None,
        sample_rate: int = 44100,
        n_src: int = 3,
        n_chan: int = 1,
        d_model: int = 24,
        n_layers: int = 6,
        capacity_mixer_hidden: int = 0,
        capacity_mixer_layers: int = 0,
        kernel_size: Sequence[int] | tuple[int, int] = (3, 3),
        routing_kernel_size: Sequence[int] | tuple[int, int] = (1, 3),
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
        kernel_size = _as_pair(kernel_size, name="kernel_size")
        routing_kernel_size = _as_pair(routing_kernel_size, name="routing_kernel_size")
        self.n_freq = int(n_freq)
        self.n_bands = int(n_bands)
        self.n_src = int(n_src)
        self.n_chan = int(n_chan)
        self.masking = masking

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
            [OnlineConvBlock(d_model, expansion=2, kernel_size=kernel_size, causal=causal) for _ in range(n_layers)]
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
        if not isinstance(self.compressor.dw, CausalConv2d):
            return 0
        return self.compressor.stream_context_frames() + sum(block.stream_context_frames() for block in self.separator)

    def init_stream_state(self, batch_size: int = 1, *, device=None, dtype=None) -> tuple[torch.Tensor, ...]:
        if not isinstance(self.compressor.dw, CausalConv2d):
            raise RuntimeError("Streaming state is only supported when causal=True.")
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

    def init_input_history(self, batch_size: int = 1, *, device=None, dtype=None) -> torch.Tensor:
        history_frames = self.stream_context_frames()
        return torch.zeros(batch_size, 2 * self.n_chan, history_frames, self.n_freq, device=device, dtype=dtype)

    def forward_stream_recompute(
        self,
        x: torch.Tensor,
        history: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError(
            "Exact low-memory recomputation from raw input history is not implemented for OnlineAdaptiveMelSFC2D. "
            "Use forward_stream with layer caches for strict realtime equivalence."
        )

    def layer_cache_numel(self, batch_size: int = 1) -> int:
        states = self.init_stream_state(
            batch_size=batch_size,
            device=self.out_proj.weight.device,
            dtype=self.out_proj.weight.dtype,
        )
        return sum(int(state.numel()) for state in states)

    def state_size_bytes(self, *, batch_size: int = 1, dtype: torch.dtype = torch.float16) -> int:
        return self.layer_cache_numel(batch_size=batch_size) * torch.tensor([], dtype=dtype).element_size()


class OnlineAdaptiveMelSFCModel(nn.Module):
    """Complex-STFT wrapper around OnlineAdaptiveMelSFC2D."""

    def __init__(self, *, n_freq: int, n_src: int = 3, n_chan: int = 1, **kwargs):
        super().__init__()
        self.core = OnlineAdaptiveMelSFC2D(n_freq=n_freq, n_src=n_src, n_chan=n_chan, **kwargs)
        self.n_src = n_src
        self.n_chan = n_chan

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        del kwargs
        x2d = pack_complex_stft_as_2d(x)
        y2d = self.core(x2d)
        return unpack_2d_to_complex_stft(y2d, n_src=self.n_src, n_chan=self.n_chan)


def build_adaptive_mel_sfc_system(
    *,
    n_fft: int,
    hop_length: int,
    fs: int,
    n_src: int = 3,
    n_chan: int = 1,
    n_bands: int = 80,
    d_model: int = 24,
    n_layers: int = 6,
    capacity_mixer_hidden: int = 8192,
    capacity_mixer_layers: int = 4,
    kernel_size: Sequence[int] | tuple[int, int] = (3, 3),
    routing_kernel_size: Sequence[int] | tuple[int, int] = (1, 3),
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
    css_segment_size: int = 6,
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
    core = OnlineAdaptiveMelSFC2D(
        n_freq=core_n_freq,
        n_bands=n_bands,
        n_fft=core_n_fft,
        sample_rate=fs,
        n_src=n_src,
        n_chan=n_chan,
        d_model=d_model,
        n_layers=n_layers,
        capacity_mixer_hidden=capacity_mixer_hidden,
        capacity_mixer_layers=capacity_mixer_layers,
        kernel_size=kernel_size,
        routing_kernel_size=routing_kernel_size,
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
