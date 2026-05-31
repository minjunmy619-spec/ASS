"""
SFC separator with a second-stage residual refinement branch.

This is the deployment-friendly part of the Mamba2/TS-BSMamba2 research gap: a
long temporal branch after SFC compression plus a correction head that refines
the first estimate.  It intentionally does not import Mamba2 kernels; the
``Mamba2LiteTemporalBranch2d`` name marks the targeted ablation role while using
causal dilated 2D blocks that stay within the existing online/NPU tensor rules.
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
from spectral_feature_compression.core.model.online_model_wrapper import OnlineModelWrapper
from spectral_feature_compression.core.model.online_sfc_2d import (
    OnlineConvBlock,
    RMSNorm2d,
    _runtime_assert,
    pack_complex_stft_as_2d,
    unpack_2d_to_complex_stft,
)
from spectral_feature_compression.core.model.online_soft_band_dilated_sfc_2d import (
    DilatedBandMixBlock2d,
    _normalize_dilation_schedule,
)
from spectral_feature_compression.core.model.online_soft_band_query_sfc_2d import (
    SoftBandQueryCompressor2d,
    SoftBandQueryExpander2d,
)
from spectral_feature_compression.core.model.online_soft_band_sfc_2d import SoftBandSpec2d


def _as_pair(value: Sequence[int] | int, *, name: str) -> tuple[int, int]:
    pair = (value, value) if isinstance(value, int) else tuple(int(v) for v in value)
    if len(pair) != 2:
        raise ValueError(f"{name} must contain exactly two values, got {value}.")
    return pair


def _as_dilation_cycle(value: Sequence[int] | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    cycle = tuple(int(v) for v in value)
    if len(cycle) == 0:
        raise ValueError("dilation_cycle must not be empty")
    if any(v <= 0 for v in cycle):
        raise ValueError(f"dilation values must be positive, got {cycle}")
    return cycle


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


class Mamba2LiteTemporalBranch2d(nn.Module):
    """Long temporal latent-band branch using exportable dilated 2D blocks."""

    def __init__(
        self,
        *,
        channels: int,
        n_bands: int,
        n_layers: int = 2,
        kernel_size: tuple[int, int] = (3, 3),
        dilation_cycle: Sequence[int] | None = (1, 2, 4),
        causal: bool = True,
    ):
        super().__init__()
        self.channels = int(channels)
        self.n_bands = int(n_bands)
        self.causal = causal
        self.dilation_schedule = _normalize_dilation_schedule(n_layers, _as_dilation_cycle(dilation_cycle))
        self.blocks = nn.ModuleList(
            [
                DilatedBandMixBlock2d(
                    channels=channels,
                    expansion=2,
                    time_kernel_size=kernel_size[0],
                    band_kernel_size=kernel_size[1],
                    time_dilation=dilation,
                    causal=causal,
                )
                for dilation in self.dilation_schedule
            ]
        )
        self.gated_delta = nn.Conv2d(channels, 2 * channels, kernel_size=1, bias=True)
        self.delta_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        for block in self.blocks:
            x = block(x)
        value, gate = self.gated_delta(x).chunk(2, dim=1)
        return residual + value * torch.sigmoid(gate) * self.delta_scale

    def stream_context_frames(self) -> int:
        return sum(block.stream_context_frames() for block in self.blocks)

    def init_stream_state(self, batch_size: int = 1, *, device=None, dtype=None) -> tuple[torch.Tensor, ...]:
        if not self.causal:
            raise RuntimeError("Streaming state is only supported when causal=True.")
        return tuple(
            block.init_stream_state(batch_size, freq_bins=self.n_bands, device=device, dtype=dtype)
            for block in self.blocks
        )

    def forward_stream(
        self,
        x: torch.Tensor,
        states: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        _runtime_assert(len(states) == len(self.blocks), f"Expected {len(self.blocks)} states, got {len(states)}")
        residual = x
        new_states = []
        for block, state in zip(self.blocks, states):
            x, state = block.forward_stream(x, state)
            new_states.append(state)
        value, gate = self.gated_delta(x).chunk(2, dim=1)
        return residual + value * torch.sigmoid(gate) * self.delta_scale, tuple(new_states)


class ResidualCorrectionHead2d(nn.Module):
    """Second-stage full-band correction from mixture, estimate, and token context."""

    def __init__(
        self,
        *,
        channels: int,
        n_freq: int,
        n_src: int,
        n_chan: int,
        n_layers: int = 1,
        kernel_size: tuple[int, int] = (3, 3),
        causal: bool = True,
    ):
        super().__init__()
        self.channels = int(channels)
        self.n_freq = int(n_freq)
        self.n_src = int(n_src)
        self.n_chan = int(n_chan)
        self.causal = causal
        out_ch = 2 * n_src * n_chan
        self.mix_proj = nn.Conv2d(2 * n_chan, channels, kernel_size=1, bias=True)
        self.estimate_proj = nn.Conv2d(out_ch, channels, kernel_size=1, bias=True)
        self.context_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.fuse = nn.Sequential(
            RMSNorm2d(3 * channels),
            nn.Conv2d(3 * channels, channels, kernel_size=1, bias=True),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [OnlineConvBlock(channels, expansion=2, kernel_size=kernel_size, causal=causal) for _ in range(n_layers)]
        )
        self.out_proj = nn.Conv2d(channels, out_ch, kernel_size=1, bias=True)
        self.correction_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor, estimate: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        h = self.fuse(torch.cat([self.mix_proj(x), self.estimate_proj(estimate), self.context_proj(context)], dim=1))
        for block in self.blocks:
            h = block(h)
        return self.out_proj(h) * self.correction_scale

    def stream_context_frames(self) -> int:
        return sum(block.stream_context_frames() for block in self.blocks)

    def init_stream_state(self, batch_size: int = 1, *, device=None, dtype=None) -> tuple[torch.Tensor, ...]:
        if not self.causal:
            raise RuntimeError("Streaming state is only supported when causal=True.")
        return tuple(
            block.init_stream_state(batch_size, freq_bins=self.n_freq, device=device, dtype=dtype)
            for block in self.blocks
        )

    def forward_stream(
        self,
        x: torch.Tensor,
        estimate: torch.Tensor,
        context: torch.Tensor,
        states: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        _runtime_assert(len(states) == len(self.blocks), f"Expected {len(self.blocks)} states, got {len(states)}")
        h = self.fuse(torch.cat([self.mix_proj(x), self.estimate_proj(estimate), self.context_proj(context)], dim=1))
        new_states = []
        for block, state in zip(self.blocks, states):
            h, state = block.forward_stream(h, state)
            new_states.append(state)
        return self.out_proj(h) * self.correction_scale, tuple(new_states)


class OnlineResidualRefinementSFC2D(nn.Module):
    """SFC core with primary estimate plus second-stage residual correction."""

    def __init__(
        self,
        n_freq: int,
        n_bands: int = 64,
        n_fft: int | None = None,
        sample_rate: int | None = None,
        band_config: str = "musical",
        n_src: int = 3,
        n_chan: int = 1,
        d_model: int = 24,
        n_layers: int = 2,
        refinement_layers: int = 1,
        long_branch_layers: int = 1,
        kernel_size: Sequence[int] | tuple[int, int] = (3, 3),
        routing_kernel_size: Sequence[int] | tuple[int, int] = (1, 3),
        dilation_cycle: Sequence[int] | None = (1, 2, 4),
        long_branch_dilation_cycle: Sequence[int] | None = (1, 2, 4),
        causal: bool = True,
        masking: bool = True,
        routing_normalization: str = "softmax",
    ):
        super().__init__()
        kernel_size = _as_pair(kernel_size, name="kernel_size")
        routing_kernel_size = _as_pair(routing_kernel_size, name="routing_kernel_size")
        self.n_freq = int(n_freq)
        self.n_bands = int(n_bands)
        self.n_src = int(n_src)
        self.n_chan = int(n_chan)
        self.masking = masking
        self.causal = causal
        self.primary_dilation_schedule = _normalize_dilation_schedule(n_layers, _as_dilation_cycle(dilation_cycle))

        out_ch = 2 * n_src * n_chan
        band_spec = SoftBandSpec2d(
            n_freq=n_freq,
            n_bands=n_bands,
            n_fft=n_fft,
            sample_rate=sample_rate,
            band_config=band_config,
        )
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
        self.primary_separator = nn.ModuleList(
            [
                DilatedBandMixBlock2d(
                    channels=d_model,
                    expansion=2,
                    time_kernel_size=kernel_size[0],
                    band_kernel_size=kernel_size[1],
                    time_dilation=dilation,
                    causal=causal,
                )
                for dilation in self.primary_dilation_schedule
            ]
        )
        self.long_temporal_refiner = Mamba2LiteTemporalBranch2d(
            channels=d_model,
            n_bands=n_bands,
            n_layers=long_branch_layers,
            kernel_size=kernel_size,
            dilation_cycle=long_branch_dilation_cycle,
            causal=causal,
        )
        self.expander = SoftBandQueryExpander2d(channels=d_model, band_spec=band_spec)
        self.primary_out = nn.Conv2d(d_model, out_ch, kernel_size=1, bias=True)
        self.correction_head = ResidualCorrectionHead2d(
            channels=d_model,
            n_freq=n_freq,
            n_src=n_src,
            n_chan=n_chan,
            n_layers=refinement_layers,
            kernel_size=kernel_size,
            causal=causal,
        )

    def _primary_estimate(self, x: torch.Tensor, mask_or_mapping: torch.Tensor) -> torch.Tensor:
        if not self.masking:
            return mask_or_mapping
        return _apply_packed_complex_mask_no_repeat(
            x=x,
            y=mask_or_mapping,
            n_src=self.n_src,
            n_chan=self.n_chan,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _runtime_assert(x.ndim == 4, f"Expected 4D input (B,C,T,F), got {x.shape}")
        _runtime_assert(x.shape[-1] == self.n_freq, f"{x.shape} vs {self.n_freq}")
        h = self.in_proj(x)
        z, query_tokens = self.compressor(h)
        for block in self.primary_separator:
            z = block(z)

        z_refined = self.long_temporal_refiner(z)
        primary_context = self.expander(z, query_tokens)
        residual_context = self.expander(z_refined, query_tokens)
        primary_estimate = self._primary_estimate(x, self.primary_out(primary_context))
        correction = self.correction_head(x, primary_estimate, residual_context)
        return primary_estimate + correction

    def stream_context_frames(self) -> int:
        if not self.causal:
            return 0
        return (
            self.compressor.stream_context_frames()
            + sum(block.stream_context_frames() for block in self.primary_separator)
            + self.long_temporal_refiner.stream_context_frames()
            + self.correction_head.stream_context_frames()
        )

    def init_stream_state(
        self,
        batch_size: int = 1,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, ...]:
        if not self.causal:
            raise RuntimeError("Streaming state is only supported when causal=True.")
        comp = self.compressor.init_stream_state(batch_size, freq_bins=self.n_freq, device=device, dtype=dtype)
        primary = tuple(
            block.init_stream_state(batch_size, freq_bins=self.n_bands, device=device, dtype=dtype)
            for block in self.primary_separator
        )
        long_branch = self.long_temporal_refiner.init_stream_state(batch_size=batch_size, device=device, dtype=dtype)
        correction = self.correction_head.init_stream_state(batch_size=batch_size, device=device, dtype=dtype)
        return (comp, *primary, *long_branch, *correction)

    def forward_stream(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, ...] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if not self.causal:
            raise RuntimeError("forward_stream is only supported when causal=True.")
        _runtime_assert(x.ndim == 4, f"Expected 4D input (B,C,T,F), got {x.shape}")
        _runtime_assert(x.shape[-1] == self.n_freq, f"{x.shape} vs {self.n_freq}")
        if state is None:
            state = self.init_stream_state(batch_size=x.shape[0], device=x.device, dtype=x.dtype)

        primary_count = len(self.primary_separator)
        long_count = len(self.long_temporal_refiner.blocks)
        correction_count = len(self.correction_head.blocks)
        expected_states = 1 + primary_count + long_count + correction_count
        _runtime_assert(len(state) == expected_states, f"Expected {expected_states} states, got {len(state)}")

        h = self.in_proj(x)
        (z, query_tokens), new_comp_state = self.compressor.forward_stream(h, state[0])
        new_primary_states = []
        for block, block_state in zip(self.primary_separator, state[1 : 1 + primary_count]):
            z, block_state = block.forward_stream(z, block_state)
            new_primary_states.append(block_state)

        long_start = 1 + primary_count
        correction_start = long_start + long_count
        z_refined, new_long_states = self.long_temporal_refiner.forward_stream(
            z,
            state[long_start:correction_start],
        )
        primary_context = self.expander(z, query_tokens)
        residual_context = self.expander(z_refined, query_tokens)
        primary_estimate = self._primary_estimate(x, self.primary_out(primary_context))
        correction, new_correction_states = self.correction_head.forward_stream(
            x,
            primary_estimate,
            residual_context,
            state[correction_start:],
        )
        return primary_estimate + correction, (
            new_comp_state,
            *new_primary_states,
            *new_long_states,
            *new_correction_states,
        )

    def init_input_history(self, batch_size: int = 1, *, device=None, dtype=None) -> torch.Tensor:
        history_frames = self.stream_context_frames()
        return torch.zeros(batch_size, 2 * self.n_chan, history_frames, self.n_freq, device=device, dtype=dtype)

    def forward_stream_recompute(
        self,
        x: torch.Tensor,
        history: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError(
            "Exact low-memory recomputation from raw input history is not implemented "
            "for OnlineResidualRefinementSFC2D. "
            "Use forward_stream with layer caches for strict realtime equivalence."
        )

    def layer_cache_numel(self, batch_size: int = 1) -> int:
        states = self.init_stream_state(
            batch_size=batch_size,
            device=self.primary_out.weight.device,
            dtype=self.primary_out.weight.dtype,
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


class OnlineResidualRefinementSFCModel(nn.Module):
    """Complex-STFT wrapper around OnlineResidualRefinementSFC2D."""

    def __init__(self, *, n_freq: int, n_src: int = 3, n_chan: int = 1, **kwargs):
        super().__init__()
        self.core = OnlineResidualRefinementSFC2D(n_freq=n_freq, n_src=n_src, n_chan=n_chan, **kwargs)
        self.n_src = n_src
        self.n_chan = n_chan

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        del kwargs
        x2d = pack_complex_stft_as_2d(x)
        y2d = self.core(x2d)
        return unpack_2d_to_complex_stft(y2d, n_src=self.n_src, n_chan=self.n_chan)

    def init_stream_state(self, batch_size: int = 1, *, device=None, dtype=None):
        return self.core.init_stream_state(batch_size=batch_size, device=device, dtype=dtype)

    def forward_stream(self, x2d: torch.Tensor, state=None):
        return self.core.forward_stream(x2d, state)

    def init_input_history(self, batch_size: int = 1, *, device=None, dtype=None):
        return self.core.init_input_history(batch_size=batch_size, device=device, dtype=dtype)

    def forward_stream_recompute(self, x2d: torch.Tensor, history=None):
        return self.core.forward_stream_recompute(x2d, history)


def build_residual_refinement_sfc_system(
    *,
    n_fft: int,
    hop_length: int,
    fs: int,
    n_bands: int = 64,
    band_config: str = "musical",
    n_src: int = 3,
    n_chan: int = 1,
    d_model: int = 24,
    n_layers: int = 2,
    refinement_layers: int = 1,
    long_branch_layers: int = 1,
    kernel_size: Sequence[int] | tuple[int, int] = (3, 3),
    routing_kernel_size: Sequence[int] | tuple[int, int] = (1, 3),
    dilation_cycle: Sequence[int] | None = (1, 2, 4),
    long_branch_dilation_cycle: Sequence[int] | None = (1, 2, 4),
    causal: bool = True,
    masking: bool = True,
    routing_normalization: str = "softmax",
    freq_preprocess_enabled: bool = False,
    freq_preprocess_keep_bins: int | None = None,
    freq_preprocess_target_bins: int | None = None,
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
    core = OnlineResidualRefinementSFC2D(
        n_freq=core_n_freq,
        n_bands=n_bands,
        n_fft=core_n_fft,
        sample_rate=fs,
        band_config=band_config,
        n_src=n_src,
        n_chan=n_chan,
        d_model=d_model,
        n_layers=n_layers,
        refinement_layers=refinement_layers,
        long_branch_layers=long_branch_layers,
        kernel_size=kernel_size,
        routing_kernel_size=routing_kernel_size,
        dilation_cycle=dilation_cycle,
        long_branch_dilation_cycle=long_branch_dilation_cycle,
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
