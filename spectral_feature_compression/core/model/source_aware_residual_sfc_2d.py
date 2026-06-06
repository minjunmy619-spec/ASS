"""
Source-aware residual SFC separator for strict online/NPU deployment.

This module is a deliberately integrated DnR separator design, not only a wider
variant of the earlier pooled BandSFC branch.  The architecture follows the
separation problem structure under the current NPU rules:

1. input-adaptive SFC compression keeps the spectral representation compact;
2. dilated causal band-mix blocks model multi-scale temporal structure on the
   compressed band axis;
3. an early source split gives Speech/Music/Effects separate token streams while
   sharing the refiner weights for parameter efficiency;
4. a cross-source shared decoder reconstructs primary complex masks;
5. a low-rank full-band residual head corrects mask artifacts from mixture,
   primary masks, and long-context shared tokens.

All runtime tensors stay 4D and the default deployment recipe avoids persistent
full-frequency cache except in the narrow residual correction head.
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
from spectral_feature_compression.core.model.residual_refinement_sfc_2d import Mamba2LiteTemporalBranch2d
from spectral_feature_compression.core.model.source_split_sfc_2d import (
    SharedSourceRefiner2d,
    SourceSharedReconstructionDecoder2d,
    SourceTokenSplitter2d,
)


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
    """Packed complex multiplication without repeat/Tile-prone expansion."""

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


class LowRankResidualCorrectionHead2d(nn.Module):
    """Full-band correction head with a small channel rank.

    A full-band correction stage is important for musical transients and
    overlapping source leakage, but using the main token width here would spend
    too much persistent state.  This head projects mixture, primary complex
    masks, and decoded long-context tokens to
    ``correction_channels`` first, then runs a short causal 2D stack on that
    narrow representation.  It returns a mask-domain delta, so exporting with
    ``masking=False`` remains equivalent to host-side complex mask application.
    """

    def __init__(
        self,
        *,
        context_channels: int,
        correction_channels: int,
        n_freq: int,
        n_src: int,
        n_chan: int,
        n_layers: int = 1,
        kernel_size: tuple[int, int] = (3, 3),
        causal: bool = True,
    ):
        super().__init__()
        if correction_channels <= 0:
            raise ValueError(f"correction_channels must be positive, got {correction_channels}")
        if n_layers < 0:
            raise ValueError(f"n_layers must be non-negative, got {n_layers}")
        self.context_channels = int(context_channels)
        self.correction_channels = int(correction_channels)
        self.n_freq = int(n_freq)
        self.n_src = int(n_src)
        self.n_chan = int(n_chan)
        self.causal = causal

        in_ch = 2 * n_chan
        out_ch = 2 * n_src * n_chan
        self.mix_proj = nn.Conv2d(in_ch, correction_channels, kernel_size=1, bias=True)
        self.estimate_proj = nn.Conv2d(out_ch, correction_channels, kernel_size=1, bias=True)
        self.context_proj = nn.Conv2d(context_channels, correction_channels, kernel_size=1, bias=True)
        self.fuse = nn.Sequential(
            RMSNorm2d(3 * correction_channels),
            nn.Conv2d(3 * correction_channels, correction_channels, kernel_size=1, bias=True),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [
                OnlineConvBlock(
                    correction_channels,
                    expansion=2,
                    kernel_size=kernel_size,
                    causal=causal,
                )
                for _ in range(n_layers)
            ]
        )
        self.out_proj = nn.Conv2d(correction_channels, out_ch, kernel_size=1, bias=True)
        self.correction_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor, primary_masks: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        h = self.fuse(
            torch.cat(
                [
                    self.mix_proj(x),
                    self.estimate_proj(primary_masks),
                    self.context_proj(context),
                ],
                dim=1,
            )
        )
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
        primary_masks: torch.Tensor,
        context: torch.Tensor,
        states: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        _runtime_assert(len(states) == len(self.blocks), f"Expected {len(self.blocks)} states, got {len(states)}")
        h = self.fuse(
            torch.cat(
                [
                    self.mix_proj(x),
                    self.estimate_proj(primary_masks),
                    self.context_proj(context),
                ],
                dim=1,
            )
        )
        new_states = []
        for block, state in zip(self.blocks, states):
            h, state = block.forward_stream(h, state)
            new_states.append(state)
        return self.out_proj(h) * self.correction_scale, tuple(new_states)


class OnlineSourceAwareResidualSFC2D(nn.Module):
    """Source-aware SFC core with low-rank full-band residual refinement."""

    def __init__(
        self,
        n_freq: int,
        n_bands: int = 56,
        n_fft: int | None = None,
        sample_rate: int | None = None,
        band_config: str = "musical",
        n_src: int = 3,
        n_chan: int = 1,
        d_model: int = 28,
        n_shared_layers: int = 2,
        n_source_layers: int = 2,
        long_branch_layers: int = 1,
        correction_layers: int = 1,
        correction_channels: int = 12,
        shared_capacity_hidden: int = 8192,
        shared_capacity_layers: int = 4,
        kernel_size: Sequence[int] | tuple[int, int] = (3, 3),
        routing_kernel_size: Sequence[int] | tuple[int, int] = (1, 3),
        dilation_cycle: Sequence[int] | None = (1, 2, 4),
        long_branch_dilation_cycle: Sequence[int] | None = (1, 2, 4),
        causal: bool = True,
        masking: bool = True,
        routing_normalization: str = "softmax",
    ):
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if n_shared_layers < 0 or n_source_layers < 0 or long_branch_layers < 0 or correction_layers < 0:
            raise ValueError("layer counts must be non-negative")

        kernel_size = _as_pair(kernel_size, name="kernel_size")
        routing_kernel_size = _as_pair(routing_kernel_size, name="routing_kernel_size")
        self.n_freq = int(n_freq)
        self.n_bands = int(n_bands)
        self.n_src = int(n_src)
        self.n_chan = int(n_chan)
        self.d_model = int(d_model)
        self.n_shared_layers = int(n_shared_layers)
        self.n_source_layers = int(n_source_layers)
        self.long_branch_layers = int(long_branch_layers)
        self.correction_layers = int(correction_layers)
        self.correction_channels = int(correction_channels)
        self.causal = causal
        self.masking = masking
        self.dilation_schedule = _normalize_dilation_schedule(n_shared_layers, _as_dilation_cycle(dilation_cycle))

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
        self.shared_analysis = nn.ModuleList(
            [
                DilatedBandMixBlock2d(
                    channels=d_model,
                    expansion=2,
                    time_kernel_size=kernel_size[0],
                    band_kernel_size=kernel_size[1],
                    time_dilation=dilation,
                    causal=causal,
                )
                for dilation in self.dilation_schedule
            ]
        )
        self.shared_capacity_mixers = build_capacity_mixers(
            channels=d_model,
            hidden_channels=shared_capacity_hidden,
            n_layers=shared_capacity_layers,
        )
        self.long_temporal_refiner = Mamba2LiteTemporalBranch2d(
            channels=d_model,
            n_bands=n_bands,
            n_layers=long_branch_layers,
            kernel_size=kernel_size,
            dilation_cycle=long_branch_dilation_cycle,
            causal=causal,
        )
        self.source_splitter = SourceTokenSplitter2d(channels=d_model, n_src=n_src)
        self.source_refiner = SharedSourceRefiner2d(
            channels=d_model,
            n_src=n_src,
            n_bands=n_bands,
            n_layers=n_source_layers,
            kernel_size=kernel_size,
            causal=causal,
        )
        self.primary_decoder = SourceSharedReconstructionDecoder2d(
            channels=d_model,
            n_src=n_src,
            n_chan=n_chan,
            band_spec=band_spec,
        )
        self.residual_expander = SoftBandQueryExpander2d(channels=d_model, band_spec=band_spec)
        self.correction_head = LowRankResidualCorrectionHead2d(
            context_channels=d_model,
            correction_channels=correction_channels,
            n_freq=n_freq,
            n_src=n_src,
            n_chan=n_chan,
            n_layers=correction_layers,
            kernel_size=kernel_size,
            causal=causal,
        )

    @property
    def _has_compressor_state(self) -> bool:
        return self.compressor.stream_context_frames() > 0

    def _run_shared(self, z: torch.Tensor) -> torch.Tensor:
        for block_idx, block in enumerate(self.shared_analysis):
            z = block(z)
            if block_idx < len(self.shared_capacity_mixers):
                z = self.shared_capacity_mixers[block_idx](z)
        for block_idx in range(len(self.shared_analysis), len(self.shared_capacity_mixers)):
            z = self.shared_capacity_mixers[block_idx](z)
        return z

    def _apply_masks(self, x: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        return _apply_packed_complex_mask_no_repeat(
            x=x,
            y=masks,
            n_src=self.n_src,
            n_chan=self.n_chan,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _runtime_assert(x.ndim == 4, f"Expected 4D input (B,C,T,F), got {x.shape}")
        _runtime_assert(x.shape[-1] == self.n_freq, f"{x.shape} vs {self.n_freq}")
        h = self.in_proj(x)
        z, query_tokens = self.compressor(h)
        z = self._run_shared(z)

        z_long = self.long_temporal_refiner(z)
        source_tokens = self.source_splitter(z)
        source_tokens = self.source_refiner(source_tokens)
        primary_masks = self.primary_decoder(source_tokens, query_tokens, z)
        residual_context = self.residual_expander(z_long, query_tokens)
        mask_delta = self.correction_head(x, primary_masks, residual_context)
        final_masks = primary_masks + mask_delta
        if self.masking:
            return self._apply_masks(x, final_masks)
        return final_masks

    def stream_context_frames(self) -> int:
        if not self.causal:
            return 0
        compressor_ctx = self.compressor.stream_context_frames()
        shared_ctx = sum(block.stream_context_frames() for block in self.shared_analysis)
        source_ctx = self.source_refiner.stream_context_frames()
        long_ctx = self.long_temporal_refiner.stream_context_frames()
        correction_ctx = self.correction_head.stream_context_frames()
        return compressor_ctx + shared_ctx + max(source_ctx, long_ctx) + correction_ctx

    def init_stream_state(
        self,
        batch_size: int = 1,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, ...]:
        if not self.causal:
            raise RuntimeError("Streaming state is only supported when causal=True.")
        states: list[torch.Tensor] = []
        if self._has_compressor_state:
            states.append(
                self.compressor.init_stream_state(batch_size, freq_bins=self.n_freq, device=device, dtype=dtype)
            )
        states.extend(
            block.init_stream_state(batch_size, freq_bins=self.n_bands, device=device, dtype=dtype)
            for block in self.shared_analysis
        )
        states.extend(self.source_refiner.init_stream_state(batch_size=batch_size, device=device, dtype=dtype))
        states.extend(self.long_temporal_refiner.init_stream_state(batch_size=batch_size, device=device, dtype=dtype))
        states.extend(self.correction_head.init_stream_state(batch_size=batch_size, device=device, dtype=dtype))
        return tuple(states)

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

        comp_count = 1 if self._has_compressor_state else 0
        shared_count = len(self.shared_analysis)
        source_count = len(self.source_refiner.blocks) * self.n_src
        long_count = len(self.long_temporal_refiner.blocks)
        correction_count = len(self.correction_head.blocks)
        expected_states = comp_count + shared_count + source_count + long_count + correction_count
        _runtime_assert(len(state) == expected_states, f"Expected {expected_states} states, got {len(state)}")

        state_idx = 0
        h = self.in_proj(x)
        if self._has_compressor_state:
            (z, query_tokens), new_comp_state = self.compressor.forward_stream(h, state[state_idx])
            state_idx += 1
        else:
            (z, query_tokens), new_comp_state = self.compressor.forward_stream(h, None)

        new_states: list[torch.Tensor] = []
        if self._has_compressor_state:
            new_states.append(new_comp_state)

        for block_idx, block in enumerate(self.shared_analysis):
            z, block_state = block.forward_stream(z, state[state_idx])
            state_idx += 1
            if block_idx < len(self.shared_capacity_mixers):
                z = self.shared_capacity_mixers[block_idx](z)
            new_states.append(block_state)
        for block_idx in range(len(self.shared_analysis), len(self.shared_capacity_mixers)):
            z = self.shared_capacity_mixers[block_idx](z)

        source_end = state_idx + source_count
        long_end = source_end + long_count
        correction_end = long_end + correction_count

        z_long, new_long_states = self.long_temporal_refiner.forward_stream(z, state[source_end:long_end])
        source_tokens = self.source_splitter(z)
        source_tokens, new_source_states = self.source_refiner.forward_stream(
            source_tokens, state[state_idx:source_end]
        )
        primary_masks = self.primary_decoder(source_tokens, query_tokens, z)
        residual_context = self.residual_expander(z_long, query_tokens)
        mask_delta, new_correction_states = self.correction_head.forward_stream(
            x,
            primary_masks,
            residual_context,
            state[long_end:correction_end],
        )
        final_masks = primary_masks + mask_delta
        y = self._apply_masks(x, final_masks) if self.masking else final_masks
        _runtime_assert(correction_end == len(state), f"Unused stream states: {len(state) - correction_end}")
        return y, (
            *new_states,
            *new_source_states,
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
            "for OnlineSourceAwareResidualSFC2D. Use forward_stream with layer caches for strict equivalence."
        )

    def layer_cache_numel(self, batch_size: int = 1) -> int:
        states = self.init_stream_state(
            batch_size=batch_size,
            device=self.correction_head.out_proj.weight.device,
            dtype=self.correction_head.out_proj.weight.dtype,
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


class OnlineSourceAwareResidualSFCModel(nn.Module):
    """Complex-STFT wrapper around OnlineSourceAwareResidualSFC2D."""

    def __init__(self, *, n_freq: int, n_src: int = 3, n_chan: int = 1, **kwargs):
        super().__init__()
        self.core = OnlineSourceAwareResidualSFC2D(n_freq=n_freq, n_src=n_src, n_chan=n_chan, **kwargs)
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


def build_source_aware_residual_sfc_system(
    *,
    n_fft: int,
    hop_length: int,
    fs: int,
    n_bands: int = 56,
    band_config: str = "musical",
    n_src: int = 3,
    n_chan: int = 1,
    d_model: int = 28,
    n_shared_layers: int = 2,
    n_source_layers: int = 2,
    long_branch_layers: int = 1,
    correction_layers: int = 1,
    correction_channels: int = 12,
    shared_capacity_hidden: int = 8192,
    shared_capacity_layers: int = 4,
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
    core = OnlineSourceAwareResidualSFC2D(
        n_freq=core_n_freq,
        n_bands=n_bands,
        n_fft=core_n_fft,
        sample_rate=fs,
        band_config=band_config,
        n_src=n_src,
        n_chan=n_chan,
        d_model=d_model,
        n_shared_layers=n_shared_layers,
        n_source_layers=n_source_layers,
        long_branch_layers=long_branch_layers,
        correction_layers=correction_layers,
        correction_channels=correction_channels,
        shared_capacity_hidden=shared_capacity_hidden,
        shared_capacity_layers=shared_capacity_layers,
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
