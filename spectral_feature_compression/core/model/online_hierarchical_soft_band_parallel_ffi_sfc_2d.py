"""
Online / realtime hierarchical soft-band family with parallel temporal FFI blocks.

This variant keeps the same hierarchical SFC-aware backbone as the other
hierarchical families, but replaces the single temporal path inside each
interleaved block with multiple parallel causal branches of different temporal
receptive fields.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectral_feature_compression.core.model.frequency_preprocessing import (
    FrequencyPreprocessedOnlineModel,
    build_frequency_preprocessor,
    resolve_preprocessed_n_freq,
)
from spectral_feature_compression.core.model.online_model_wrapper import OnlineModelWrapper
from spectral_feature_compression.core.model.online_sfc_2d import (
    CausalConv2d,
    RMSNorm2d,
    _runtime_assert,
    _validate_npu_kernel_dilation_limit,
    apply_packed_complex_mask,
    pack_complex_stft_as_2d,
    unpack_2d_to_complex_stft,
)
from spectral_feature_compression.core.model.online_soft_band_sfc_2d import SoftBandCompressor2d, SoftBandExpander2d
from spectral_feature_compression.core.model.online_hierarchical_soft_band_sfc_2d import (
    HierarchicalBandSpec2d,
    HierarchicalSkipFuse2d,
)


def _normalize_branch_specs(
    kernel_sizes: tuple[int, ...] | list[int],
    dilations: tuple[int, ...] | list[int],
) -> tuple[tuple[int, int], ...]:
    kernel_sizes = tuple(int(k) for k in kernel_sizes)
    dilations = tuple(int(d) for d in dilations)
    if len(kernel_sizes) == 0 or len(kernel_sizes) != len(dilations):
        raise ValueError(
            f"time_branch_kernel_sizes and time_branch_dilations must be non-empty and have the same length, got "
            f"{kernel_sizes} and {dilations}"
        )
    specs = []
    for kernel, dilation in zip(kernel_sizes, dilations):
        _validate_npu_kernel_dilation_limit(kernel, dilation, axis="time")
        specs.append((kernel, dilation))
    return tuple(specs)


def _split_hidden_channels(total_hidden: int, n_branches: int) -> tuple[int, ...]:
    base = total_hidden // n_branches
    rem = total_hidden % n_branches
    parts = []
    for idx in range(n_branches):
        parts.append(base + (1 if idx < rem else 0))
    if any(p <= 0 for p in parts):
        raise ValueError(f"Invalid hidden split: total_hidden={total_hidden}, n_branches={n_branches}, parts={parts}")
    return tuple(parts)


def _recursive_numel(state) -> int:
    if state is None:
        return 0
    if isinstance(state, torch.Tensor):
        return int(state.numel())
    if isinstance(state, (tuple, list)):
        return sum(_recursive_numel(s) for s in state)
    raise TypeError(f"Unsupported state type: {type(state)!r}")


class ParallelTemporalFFIBlock2d(nn.Module):
    """
    Interleaved frequency -> parallel temporal block.

    Frequency path:
      local, stateless, band-axis conv

    Temporal path:
      multiple parallel causal depthwise branches
      each branch has its own kernel/dilation and its own streaming cache
    """

    def __init__(
        self,
        channels: int,
        expansion: int = 2,
        freq_kernel_size: int = 3,
        time_branch_kernel_sizes: tuple[int, ...] | list[int] = (3, 3),
        time_branch_dilations: tuple[int, ...] | list[int] = (1, 6),
        causal: bool = True,
    ):
        super().__init__()
        if freq_kernel_size % 2 == 0:
            raise ValueError(f"freq_kernel_size must be odd, got {freq_kernel_size}")
        _validate_npu_kernel_dilation_limit(freq_kernel_size, 1, axis="frequency")

        self.branch_specs = _normalize_branch_specs(time_branch_kernel_sizes, time_branch_dilations)
        self.causal = causal

        hidden_total = channels * expansion
        self.branch_hidden = _split_hidden_channels(hidden_total, len(self.branch_specs))

        self.freq_norm = RMSNorm2d(channels)
        self.freq_pw1 = nn.Conv2d(channels, hidden_total * 2, kernel_size=1, bias=True)
        self.freq_dw = nn.Conv2d(
            hidden_total,
            hidden_total,
            kernel_size=(1, freq_kernel_size),
            padding=(0, freq_kernel_size // 2),
            groups=hidden_total,
            bias=True,
        )
        self.freq_pw2 = nn.Conv2d(hidden_total, channels, kernel_size=1, bias=True)

        self.time_norm = RMSNorm2d(channels)
        self.time_pw1 = nn.Conv2d(channels, hidden_total * 2, kernel_size=1, bias=True)
        self.time_branches = nn.ModuleList(
            [
                CausalConv2d(
                    hidden,
                    hidden,
                    kernel_size=(kernel, 1),
                    dilation=(dilation, 1),
                    groups=hidden,
                    bias=True,
                ) if causal else nn.Conv2d(
                    hidden,
                    hidden,
                    kernel_size=(kernel, 1),
                    padding=((kernel - 1) * dilation // 2, 0),
                    dilation=(dilation, 1),
                    groups=hidden,
                    bias=True,
                )
                for hidden, (kernel, dilation) in zip(self.branch_hidden, self.branch_specs)
            ]
        )
        self.time_pw2 = nn.Conv2d(hidden_total, channels, kernel_size=1, bias=True)
        self.out_norm = RMSNorm2d(channels)

    def _gated(self, x: torch.Tensor, proj: nn.Conv2d) -> torch.Tensor:
        a, b = proj(x).chunk(2, dim=1)
        return a * torch.sigmoid(b)

    def _freq_path(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, K)
        y = self._gated(self.freq_norm(x), self.freq_pw1)
        y = self.freq_dw(y)
        y = F.silu(y)
        y = self.freq_pw2(y)
        return x + y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, K)
        x = self._freq_path(x)
        y = self._gated(self.time_norm(x), self.time_pw1)
        # y_parts[i]: (B, hidden_i, T, K)
        y_parts = torch.split(y, self.branch_hidden, dim=1)
        branch_outs = []
        for y_i, branch in zip(y_parts, self.time_branches):
            branch_outs.append(F.silu(branch(y_i)))
        y = torch.cat(branch_outs, dim=1)
        y = self.time_pw2(y)
        return self.out_norm(x + y)

    def stream_context_frames(self) -> int:
        return max(
            branch.stream_context_frames() if isinstance(branch, CausalConv2d) else 0
            for branch in self.time_branches
        )

    def init_stream_state(
        self,
        batch_size: int,
        *,
        freq_bins: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, ...]:
        if not self.causal:
            raise RuntimeError("Streaming state is only supported when causal=True.")
        return tuple(
            branch.init_stream_state(batch_size, freq_bins=freq_bins, device=device, dtype=dtype)
            for branch in self.time_branches
        )

    def forward_stream(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, ...] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if not self.causal:
            raise RuntimeError("forward_stream is only supported when causal=True.")
        if state is None:
            raise RuntimeError("ParallelTemporalFFIBlock2d.forward_stream requires an explicit state tuple.")
        _runtime_assert(len(state) == len(self.time_branches), f"Unexpected branch state tuple: {len(state)}")

        x = self._freq_path(x)
        y = self._gated(self.time_norm(x), self.time_pw1)
        y_parts = torch.split(y, self.branch_hidden, dim=1)
        branch_outs = []
        new_states = []
        for y_i, branch, branch_state in zip(y_parts, self.time_branches, state):
            branch_y, branch_state = branch.forward_stream(y_i, branch_state)
            branch_outs.append(F.silu(branch_y))
            new_states.append(branch_state)
        y = torch.cat(branch_outs, dim=1)
        y = self.time_pw2(y)
        return self.out_norm(x + y), tuple(new_states)


def _build_parallel_ffi_blocks(
    channels: int,
    n_layers: int,
    kernel_size: tuple[int, int],
    causal: bool,
    time_branch_kernel_sizes: tuple[int, ...] | list[int],
    time_branch_dilations: tuple[int, ...] | list[int],
) -> nn.ModuleList:
    return nn.ModuleList(
        [
            ParallelTemporalFFIBlock2d(
                channels=channels,
                expansion=2,
                freq_kernel_size=kernel_size[1],
                time_branch_kernel_sizes=time_branch_kernel_sizes,
                time_branch_dilations=time_branch_dilations,
                causal=causal,
            )
            for _ in range(n_layers)
        ]
    )


class OnlineHierarchicalSoftBandParallelFFISFC2D(nn.Module):
    def __init__(
        self,
        n_freq: int,
        pre_bands: int = 128,
        mid_bands: int = 96,
        bottleneck_bands: int = 48,
        n_fft: int | None = None,
        sample_rate: int | None = None,
        band_config: str = "musical",
        n_src: int = 2,
        n_chan: int = 1,
        d_model: int = 20,
        pre_layers: int = 0,
        mid_layers: int = 1,
        bottleneck_layers: int = 1,
        kernel_size: tuple[int, int] = (3, 3),
        causal: bool = True,
        masking: bool = True,
        routing_normalization: str = "softmax",
        hierarchical_prior_mode: str = "inherited",
        time_branch_kernel_sizes: tuple[int, ...] | list[int] = (3, 3),
        time_branch_dilations: tuple[int, ...] | list[int] = (1, 6),
    ):
        super().__init__()
        if not (0 < bottleneck_bands <= mid_bands <= pre_bands < n_freq):
            raise ValueError(
                "Expected 0 < bottleneck_bands <= mid_bands <= pre_bands < n_freq, got "
                f"{bottleneck_bands}, {mid_bands}, {pre_bands}, {n_freq}"
            )

        self.n_freq = n_freq
        self.pre_bands = pre_bands
        self.mid_bands = mid_bands
        self.bottleneck_bands = bottleneck_bands
        self.n_src = n_src
        self.n_chan = n_chan
        self.masking = masking
        self.hierarchical_prior_mode = hierarchical_prior_mode
        self.time_branch_specs = _normalize_branch_specs(time_branch_kernel_sizes, time_branch_dilations)

        in_ch = 2 * n_chan
        out_ch = 2 * n_src * n_chan

        pre_spec = HierarchicalBandSpec2d.from_original(
            n_freq=n_freq,
            n_bands=pre_bands,
            n_fft=n_fft,
            sample_rate=sample_rate,
            band_config=band_config,
        )

        if hierarchical_prior_mode == "inherited":
            mid_down_spec = HierarchicalBandSpec2d.from_parent(pre_spec, n_bands=mid_bands, band_config=band_config)
            bottleneck_down_spec = HierarchicalBandSpec2d.from_parent(
                mid_down_spec,
                n_bands=bottleneck_bands,
                band_config=band_config,
            )
            mid_up_spec = HierarchicalBandSpec2d.from_parent(
                mid_down_spec,
                n_bands=bottleneck_bands,
                band_config=band_config,
            )
            pre_up_spec = HierarchicalBandSpec2d.from_parent(pre_spec, n_bands=mid_bands, band_config=band_config)
        elif hierarchical_prior_mode == "uniform":
            from spectral_feature_compression.core.model.online_soft_band_sfc_2d import SoftBandSpec2d

            mid_down_spec = SoftBandSpec2d(n_freq=pre_bands, n_bands=mid_bands)
            bottleneck_down_spec = SoftBandSpec2d(n_freq=mid_bands, n_bands=bottleneck_bands)
            mid_up_spec = SoftBandSpec2d(n_freq=mid_bands, n_bands=bottleneck_bands)
            pre_up_spec = SoftBandSpec2d(n_freq=pre_bands, n_bands=mid_bands)
        else:
            raise ValueError(f"Unsupported hierarchical_prior_mode: {hierarchical_prior_mode}")

        self.in_proj = nn.Sequential(
            nn.Conv2d(in_ch, d_model, kernel_size=1, bias=True),
            RMSNorm2d(d_model),
        )

        self.pre_compressor = SoftBandCompressor2d(
            channels=d_model,
            band_spec=pre_spec,
            kernel_size=kernel_size,
            causal=causal,
            normalization=routing_normalization,
        )
        self.pre_temporal = _build_parallel_ffi_blocks(
            d_model, pre_layers, kernel_size, causal, time_branch_kernel_sizes, time_branch_dilations
        )

        self.mid_compressor = SoftBandCompressor2d(
            channels=d_model,
            band_spec=mid_down_spec,
            kernel_size=kernel_size,
            causal=causal,
            normalization=routing_normalization,
        )
        self.mid_temporal = _build_parallel_ffi_blocks(
            d_model, mid_layers, kernel_size, causal, time_branch_kernel_sizes, time_branch_dilations
        )

        self.bottleneck_compressor = SoftBandCompressor2d(
            channels=d_model,
            band_spec=bottleneck_down_spec,
            kernel_size=kernel_size,
            causal=causal,
            normalization=routing_normalization,
        )
        self.bottleneck_temporal = _build_parallel_ffi_blocks(
            d_model, bottleneck_layers, kernel_size, causal, time_branch_kernel_sizes, time_branch_dilations
        )

        self.mid_expander = SoftBandExpander2d(channels=d_model, band_spec=mid_up_spec)
        self.pre_expander = SoftBandExpander2d(channels=d_model, band_spec=pre_up_spec)
        self.out_expander = SoftBandExpander2d(channels=d_model, band_spec=pre_spec)
        self.fuse_mid = HierarchicalSkipFuse2d(d_model)
        self.fuse_pre = HierarchicalSkipFuse2d(d_model)
        self.out_proj = nn.Conv2d(d_model, out_ch, kernel_size=1, bias=True)

    def _run_blocks(self, x: torch.Tensor, blocks: nn.ModuleList) -> torch.Tensor:
        for block in blocks:
            x = block(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _runtime_assert(x.ndim == 4, f"Expected 4D input (B,C,T,F), got {x.shape}")
        _runtime_assert(x.shape[-1] == self.n_freq, f"{x.shape} vs {self.n_freq}")

        h = self.in_proj(x)
        z0, _ = self.pre_compressor(h)
        z0 = self._run_blocks(z0, self.pre_temporal)
        z1, _ = self.mid_compressor(z0)
        z1 = self._run_blocks(z1, self.mid_temporal)
        z2, _ = self.bottleneck_compressor(z1)
        z2 = self._run_blocks(z2, self.bottleneck_temporal)

        u1 = self.mid_expander(z2)
        u1 = self.fuse_mid(u1, z1)
        u0 = self.pre_expander(u1)
        u0 = self.fuse_pre(u0, z0)
        h_out = self.out_expander(u0)
        y = self.out_proj(h_out)
        if self.masking:
            return apply_packed_complex_mask(x=x, y=y, n_src=self.n_src, n_chan=self.n_chan)
        return y

    def stream_context_frames(self) -> int:
        compressors = [self.pre_compressor, self.mid_compressor, self.bottleneck_compressor]
        blocks = [*self.pre_temporal, *self.mid_temporal, *self.bottleneck_temporal]
        if not isinstance(self.pre_compressor.dw, CausalConv2d):
            return 0
        return sum(comp.stream_context_frames() for comp in compressors) + sum(block.stream_context_frames() for block in blocks)

    def init_stream_state(
        self,
        batch_size: int = 1,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        if not isinstance(self.pre_compressor.dw, CausalConv2d):
            raise RuntimeError("Streaming state is only supported when causal=True.")
        return (
            self.pre_compressor.init_stream_state(batch_size, freq_bins=self.n_freq, device=device, dtype=dtype),
            *[
                block.init_stream_state(batch_size, freq_bins=self.pre_bands, device=device, dtype=dtype)
                for block in self.pre_temporal
            ],
            self.mid_compressor.init_stream_state(batch_size, freq_bins=self.pre_bands, device=device, dtype=dtype),
            *[
                block.init_stream_state(batch_size, freq_bins=self.mid_bands, device=device, dtype=dtype)
                for block in self.mid_temporal
            ],
            self.bottleneck_compressor.init_stream_state(batch_size, freq_bins=self.mid_bands, device=device, dtype=dtype),
            *[
                block.init_stream_state(batch_size, freq_bins=self.bottleneck_bands, device=device, dtype=dtype)
                for block in self.bottleneck_temporal
            ],
        )

    def forward_stream(self, x: torch.Tensor, state=None):
        if not isinstance(self.pre_compressor.dw, CausalConv2d):
            raise RuntimeError("forward_stream is only supported when causal=True.")
        _runtime_assert(x.ndim == 4, f"Expected 4D input (B,C,T,F), got {x.shape}")
        _runtime_assert(x.shape[-1] == self.n_freq, f"{x.shape} vs {self.n_freq}")
        if state is None:
            state = self.init_stream_state(batch_size=x.shape[0], device=x.device, dtype=x.dtype)

        expected_states = 3 + len(self.pre_temporal) + len(self.mid_temporal) + len(self.bottleneck_temporal)
        _runtime_assert(len(state) == expected_states, f"Unexpected state tuple: {len(state)} vs {expected_states}")

        h = self.in_proj(x)
        idx = 0
        z0, s = self.pre_compressor.forward_stream(h, state[idx])
        new_states = [s]
        idx += 1
        for block in self.pre_temporal:
            z0, s = block.forward_stream(z0, state[idx])
            new_states.append(s)
            idx += 1

        z1, s = self.mid_compressor.forward_stream(z0, state[idx])
        new_states.append(s)
        idx += 1
        for block in self.mid_temporal:
            z1, s = block.forward_stream(z1, state[idx])
            new_states.append(s)
            idx += 1

        z2, s = self.bottleneck_compressor.forward_stream(z1, state[idx])
        new_states.append(s)
        idx += 1
        for block in self.bottleneck_temporal:
            z2, s = block.forward_stream(z2, state[idx])
            new_states.append(s)
            idx += 1

        u1 = self.mid_expander(z2)
        u1 = self.fuse_mid(u1, z1)
        u0 = self.pre_expander(u1)
        u0 = self.fuse_pre(u0, z0)
        h_out = self.out_expander(u0)
        y = self.out_proj(h_out)
        if self.masking:
            y = apply_packed_complex_mask(x=x, y=y, n_src=self.n_src, n_chan=self.n_chan)
        return y, tuple(new_states)

    def init_input_history(self, batch_size: int = 1, *, device=None, dtype=None):
        history_frames = self.stream_context_frames()
        return torch.zeros(batch_size, 2 * self.n_chan, history_frames, self.n_freq, device=device, dtype=dtype)

    def forward_stream_recompute(self, x: torch.Tensor, history=None):
        raise RuntimeError(
            "Exact low-memory recomputation from raw input history is not implemented for this model. "
            "Use forward_stream with layer caches for strict realtime equivalence."
        )

    def layer_cache_numel(self, batch_size: int = 1) -> int:
        states = self.init_stream_state(batch_size=batch_size, device=self.out_proj.weight.device, dtype=self.out_proj.weight.dtype)
        return _recursive_numel(states)

    def input_history_numel(self, batch_size: int = 1) -> int:
        return batch_size * 2 * self.n_chan * self.stream_context_frames() * self.n_freq

    def state_size_bytes(self, *, batch_size: int = 1, dtype: torch.dtype = torch.float16, mode: str = "layer_cache") -> int:
        element_size = torch.tensor([], dtype=dtype).element_size()
        if mode == "layer_cache":
            return self.layer_cache_numel(batch_size=batch_size) * element_size
        if mode == "input_history":
            return self.input_history_numel(batch_size=batch_size) * element_size
        raise ValueError(f"Unsupported state mode: {mode}")


class OnlineHierarchicalSoftBandParallelFFISFCModel(nn.Module):
    def __init__(
        self,
        n_freq: int,
        pre_bands: int = 128,
        mid_bands: int = 96,
        bottleneck_bands: int = 48,
        n_fft: int | None = None,
        sample_rate: int | None = None,
        band_config: str = "musical",
        n_src: int = 2,
        n_chan: int = 1,
        d_model: int = 20,
        pre_layers: int = 0,
        mid_layers: int = 1,
        bottleneck_layers: int = 1,
        kernel_size: tuple[int, int] = (3, 3),
        causal: bool = True,
        masking: bool = True,
        routing_normalization: str = "softmax",
        hierarchical_prior_mode: str = "inherited",
        time_branch_kernel_sizes: tuple[int, ...] | list[int] = (3, 3),
        time_branch_dilations: tuple[int, ...] | list[int] = (1, 6),
    ):
        super().__init__()
        self.core = OnlineHierarchicalSoftBandParallelFFISFC2D(
            n_freq=n_freq,
            pre_bands=pre_bands,
            mid_bands=mid_bands,
            bottleneck_bands=bottleneck_bands,
            n_fft=n_fft,
            sample_rate=sample_rate,
            band_config=band_config,
            n_src=n_src,
            n_chan=n_chan,
            d_model=d_model,
            pre_layers=pre_layers,
            mid_layers=mid_layers,
            bottleneck_layers=bottleneck_layers,
            kernel_size=kernel_size,
            causal=causal,
            masking=masking,
            routing_normalization=routing_normalization,
            hierarchical_prior_mode=hierarchical_prior_mode,
            time_branch_kernel_sizes=time_branch_kernel_sizes,
            time_branch_dilations=time_branch_dilations,
        )
        self.n_src = n_src
        self.n_chan = n_chan

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
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


def build_online_hierarchical_soft_band_parallel_ffi_sfc_system(
    *,
    n_fft: int,
    hop_length: int,
    fs: int,
    pre_bands: int = 128,
    mid_bands: int = 96,
    bottleneck_bands: int = 48,
    band_config: str = "musical",
    n_src: int = 2,
    n_chan: int = 1,
    d_model: int = 20,
    pre_layers: int = 0,
    mid_layers: int = 1,
    bottleneck_layers: int = 1,
    kernel_size: tuple[int, int] = (3, 3),
    causal: bool = True,
    masking: bool = True,
    routing_normalization: str = "softmax",
    hierarchical_prior_mode: str = "inherited",
    time_branch_kernel_sizes: tuple[int, ...] | list[int] = (3, 3),
    time_branch_dilations: tuple[int, ...] | list[int] = (1, 6),
    freq_preprocess_enabled: bool = False,
    freq_preprocess_keep_bins: int | None = None,
    freq_preprocess_target_bins: int | None = None,
    freq_preprocess_mode: str = "triangular",
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
    )
    freq_preprocessor = build_frequency_preprocessor(
        full_n_freq,
        enabled=freq_preprocess_enabled,
        keep_bins=freq_preprocess_keep_bins,
        target_bins=freq_preprocess_target_bins,
        mode=freq_preprocess_mode,
    )
    core = OnlineHierarchicalSoftBandParallelFFISFC2D(
        n_freq=core_n_freq,
        pre_bands=pre_bands,
        mid_bands=mid_bands,
        bottleneck_bands=bottleneck_bands,
        n_fft=n_fft,
        sample_rate=fs,
        band_config=band_config,
        n_src=n_src,
        n_chan=n_chan,
        d_model=d_model,
        pre_layers=pre_layers,
        mid_layers=mid_layers,
        bottleneck_layers=bottleneck_layers,
        kernel_size=kernel_size,
        causal=causal,
        masking=masking,
        routing_normalization=routing_normalization,
        hierarchical_prior_mode=hierarchical_prior_mode,
        time_branch_kernel_sizes=time_branch_kernel_sizes,
        time_branch_dilations=time_branch_dilations,
    )
    model = FrequencyPreprocessedOnlineModel(
        core=core,
        n_src=n_src,
        n_chan=n_chan,
        freq_preprocessor=freq_preprocessor,
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
