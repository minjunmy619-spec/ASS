"""
Online / realtime hierarchical soft-band SFC variant.

This family keeps an SFC-style front-end inductive bias on the original STFT
frequency axis, then performs interleaved frequency compression and temporal
modeling on progressively smaller latent band axes:

    F -> K0 -> K1 -> K2 -> K1 -> K0 -> F

The first compression step preserves the original "musical" / "mel" prior over
real STFT bins. Later hierarchical stages can either:
- inherit approximate original-frequency semantics onto the latent band axes
- or fall back to a lighter uniform latent prior for ablations
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectral_feature_compression.core.model.frequency_preprocessing import (
    FrequencyPreprocessedOnlineModel,
    build_frequency_preprocessor,
    resolve_preprocessed_n_freq,
)
from spectral_feature_compression.core.model.online_model_wrapper import OnlineModelWrapper
from spectral_feature_compression.core.model.online_sfc_2d import CausalConv2d, RMSNorm2d, _runtime_assert
from spectral_feature_compression.core.model.online_sfc_2d import apply_packed_complex_mask
from spectral_feature_compression.core.model.online_sfc_2d import pack_complex_stft_as_2d
from spectral_feature_compression.core.model.online_sfc_2d import unpack_2d_to_complex_stft
from spectral_feature_compression.core.model.online_soft_band_dilated_sfc_2d import (
    DilatedBandMixBlock2d,
    _normalize_dilation_schedule,
)
from spectral_feature_compression.core.model.online_soft_band_sfc_2d import (
    SoftBandCompressor2d,
    SoftBandExpander2d,
    SoftBandSpec2d,
)


def _build_blocks(
    channels: int,
    n_layers: int,
    kernel_size: tuple[int, int],
    causal: bool,
    dilation_cycle: tuple[int, ...] | list[int] | None,
) -> nn.ModuleList:
    dilation_schedule = _normalize_dilation_schedule(n_layers, dilation_cycle)
    return nn.ModuleList(
        [
            DilatedBandMixBlock2d(
                channels=channels,
                expansion=2,
                time_kernel_size=kernel_size[0],
                band_kernel_size=kernel_size[1],
                time_dilation=dilation,
                causal=causal,
            )
            for dilation in dilation_schedule
        ]
    )


class HierarchicalBandSpec2d(nn.Module):
    """
    Band specification for hierarchical stages.

    It stores both:
    - the basis over the current input axis (`basis`)
    - the approximate semantic footprint over the original STFT axis
      (`orig_basis`)

    This lets later latent stages inherit a band-aware prior from earlier
    stages instead of falling back to a purely uniform latent prior.
    """

    def __init__(
        self,
        *,
        n_freq: int,
        n_bands: int,
        basis_2d: torch.Tensor,
        orig_basis_2d: torch.Tensor,
        band_config: str,
        source_n_fft: int | None,
        source_sample_rate: int | None,
    ):
        super().__init__()
        self.n_freq = n_freq
        self.n_bands = n_bands
        self.band_config = band_config
        self.orig_n_freq = int(orig_basis_2d.shape[-1])
        self.source_n_fft = source_n_fft
        self.source_sample_rate = source_sample_rate

        starts, ends = self._support_from_basis(basis_2d)
        self.register_buffer("starts", starts)
        self.register_buffer("ends", ends)
        self.register_buffer("basis", basis_2d.view(1, n_bands, 1, n_freq))
        self.register_buffer("orig_basis", orig_basis_2d)

    @staticmethod
    def _support_from_basis(basis_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        starts = []
        ends = []
        for row in basis_2d > 0:
            nz = torch.nonzero(row, as_tuple=False).view(-1)
            if nz.numel() == 0:
                starts.append(0)
                ends.append(1)
            else:
                starts.append(int(nz[0].item()))
                ends.append(int(nz[-1].item()) + 1)
        return torch.tensor(starts, dtype=torch.long), torch.tensor(ends, dtype=torch.long)

    @classmethod
    def from_original(
        cls,
        *,
        n_freq: int,
        n_bands: int,
        n_fft: int | None,
        sample_rate: int | None,
        band_config: str,
    ) -> "HierarchicalBandSpec2d":
        spec = SoftBandSpec2d(
            n_freq=n_freq,
            n_bands=n_bands,
            n_fft=n_fft,
            sample_rate=sample_rate,
            band_config=band_config,
        )
        basis_2d = spec.basis.view(n_bands, n_freq).detach().clone()
        return cls(
            n_freq=n_freq,
            n_bands=n_bands,
            basis_2d=basis_2d,
            orig_basis_2d=basis_2d.clone(),
            band_config=band_config,
            source_n_fft=n_fft,
            source_sample_rate=sample_rate,
        )

    @classmethod
    def from_parent(
        cls,
        parent_spec: "HierarchicalBandSpec2d",
        *,
        n_bands: int,
        band_config: str,
    ) -> "HierarchicalBandSpec2d":
        target_orig = HierarchicalBandSpec2d.from_original(
            n_freq=parent_spec.orig_n_freq,
            n_bands=n_bands,
            n_fft=parent_spec.source_n_fft,
            sample_rate=parent_spec.source_sample_rate,
            band_config=band_config,
        )

        # parent_footprint: (K_parent, F_orig), normalized over original freq bins.
        parent_footprint = parent_spec.orig_basis / parent_spec.orig_basis.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        # inherited latent basis: (K_child, K_parent)
        latent_basis = torch.matmul(target_orig.orig_basis, parent_footprint.transpose(0, 1)).clamp_min(0.0)
        return cls(
            n_freq=parent_spec.n_bands,
            n_bands=n_bands,
            basis_2d=latent_basis,
            orig_basis_2d=target_orig.orig_basis,
            band_config=band_config,
            source_n_fft=parent_spec.source_n_fft,
            source_sample_rate=parent_spec.source_sample_rate,
        )

    def routing_bias(self) -> torch.Tensor:
        peak = self.basis.amax(dim=-1, keepdim=True).clamp_min(1e-6)
        return 2.0 * (self.basis / peak) - 1.0

    def expansion_basis(self) -> torch.Tensor:
        return self.basis / self.basis.sum(dim=1, keepdim=True).clamp_min(1e-6)


class HierarchicalSkipFuse2d(nn.Module):
    """
    Stateless skip fusion used on the decoder path.

    A simple concat + 1x1 projection keeps the graph NPU-friendly while letting
    the decoder merge current and skip-scale features with different semantics.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.norm = RMSNorm2d(channels * 2)
        self.proj = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if not torch.onnx.is_in_onnx_export():
            _runtime_assert(x.shape == skip.shape, f"Skip mismatch: {x.shape} vs {skip.shape}")
        # x / skip: (B, C, T, K_stage)
        y = torch.cat([x, skip], dim=1)
        # y after concat: (B, 2C, T, K_stage)
        y = self.norm(y)
        y = self.proj(y)
        # fused output: (B, C, T, K_stage)
        return F.silu(y)


class OnlineHierarchicalSoftBandSFC2D(nn.Module):
    """
    Hierarchical online separator with:
    - front-end soft-band compression from real STFT bins to K0
    - interleaved temporal modeling at K0 / K1 / K2
    - symmetric soft-band expansion back to the original STFT grid
    """

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
        d_model: int = 96,
        pre_layers: int = 1,
        mid_layers: int = 2,
        bottleneck_layers: int = 2,
        kernel_size: tuple[int, int] = (3, 3),
        causal: bool = True,
        masking: bool = True,
        routing_normalization: str = "softmax",
        dilation_cycle: tuple[int, ...] | list[int] | None = None,
        hierarchical_prior_mode: str = "inherited",
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

        in_ch = 2 * n_chan
        out_ch = 2 * n_src * n_chan

        # Front-end uses the original STFT-aware prior.
        pre_spec = HierarchicalBandSpec2d.from_original(
            n_freq=n_freq,
            n_bands=pre_bands,
            n_fft=n_fft,
            sample_rate=sample_rate,
            band_config=band_config,
        )
        if hierarchical_prior_mode == "inherited":
            # Internal latent stages inherit original-frequency semantics from
            # the preceding stage instead of using a purely uniform latent prior.
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
            # Baseline mode kept for ablations and direct comparison.
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
        self.pre_temporal = _build_blocks(d_model, pre_layers, kernel_size, causal, dilation_cycle)

        self.mid_compressor = SoftBandCompressor2d(
            channels=d_model,
            band_spec=mid_down_spec,
            kernel_size=kernel_size,
            causal=causal,
            normalization=routing_normalization,
        )
        self.mid_temporal = _build_blocks(d_model, mid_layers, kernel_size, causal, dilation_cycle)

        self.bottleneck_compressor = SoftBandCompressor2d(
            channels=d_model,
            band_spec=bottleneck_down_spec,
            kernel_size=kernel_size,
            causal=causal,
            normalization=routing_normalization,
        )
        self.bottleneck_temporal = _build_blocks(d_model, bottleneck_layers, kernel_size, causal, dilation_cycle)

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
        # x: (B, 2*M, T, F)
        _runtime_assert(x.ndim == 4, f"Expected 4D input (B,C,T,F), got {x.shape}")
        _runtime_assert(x.shape[-1] == self.n_freq, f"{x.shape} vs {self.n_freq}")

        # h: (B, D, T, F)
        h = self.in_proj(x)

        # z0: (B, D, T, K0=pre_bands)
        z0, _ = self.pre_compressor(h)
        z0 = self._run_blocks(z0, self.pre_temporal)

        # z1: (B, D, T, K1=mid_bands)
        z1, _ = self.mid_compressor(z0)
        z1 = self._run_blocks(z1, self.mid_temporal)

        # z2: (B, D, T, K2=bottleneck_bands)
        z2, _ = self.bottleneck_compressor(z1)
        z2 = self._run_blocks(z2, self.bottleneck_temporal)

        # u1: (B, D, T, K1)
        u1 = self.mid_expander(z2)
        u1 = self.fuse_mid(u1, z1)

        # u0: (B, D, T, K0)
        u0 = self.pre_expander(u1)
        u0 = self.fuse_pre(u0, z0)

        # h: (B, D, T, F), y: (B, 2*N*M, T, F)
        h = self.out_expander(u0)
        y = self.out_proj(h)
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
    ) -> tuple[torch.Tensor, ...]:
        if not isinstance(self.pre_compressor.dw, CausalConv2d):
            raise RuntimeError("Streaming state is only supported when causal=True.")
        states = [
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
        ]
        return tuple(states)

    def forward_stream(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, ...] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if not isinstance(self.pre_compressor.dw, CausalConv2d):
            raise RuntimeError("forward_stream is only supported when causal=True.")

        _runtime_assert(x.ndim == 4, f"Expected 4D input (B,C,T,F), got {x.shape}")
        _runtime_assert(x.shape[-1] == self.n_freq, f"{x.shape} vs {self.n_freq}")

        if state is None:
            state = self.init_stream_state(batch_size=x.shape[0], device=x.device, dtype=x.dtype)

        expected_states = 3 + len(self.pre_temporal) + len(self.mid_temporal) + len(self.bottleneck_temporal)
        _runtime_assert(len(state) == expected_states, f"Unexpected state tuple: {len(state)} vs {expected_states}")

        # h:  (B, D, T_chunk, F)
        h = self.in_proj(x)
        idx = 0

        # z0: (B, D, T_chunk, K0)
        z0, s = self.pre_compressor.forward_stream(h, state[idx])
        new_states = [s]
        idx += 1
        for block in self.pre_temporal:
            z0, s = block.forward_stream(z0, state[idx])
            new_states.append(s)
            idx += 1

        # z1: (B, D, T_chunk, K1)
        z1, s = self.mid_compressor.forward_stream(z0, state[idx])
        new_states.append(s)
        idx += 1
        for block in self.mid_temporal:
            z1, s = block.forward_stream(z1, state[idx])
            new_states.append(s)
            idx += 1

        # z2: (B, D, T_chunk, K2)
        z2, s = self.bottleneck_compressor.forward_stream(z1, state[idx])
        new_states.append(s)
        idx += 1
        for block in self.bottleneck_temporal:
            z2, s = block.forward_stream(z2, state[idx])
            new_states.append(s)
            idx += 1

        # u1: (B, D, T_chunk, K1), u0: (B, D, T_chunk, K0)
        u1 = self.mid_expander(z2)
        u1 = self.fuse_mid(u1, z1)
        u0 = self.pre_expander(u1)
        u0 = self.fuse_pre(u0, z0)
        # h: (B, D, T_chunk, F), y: (B, 2*N*M, T_chunk, F)
        h = self.out_expander(u0)
        y = self.out_proj(h)
        if self.masking:
            y = apply_packed_complex_mask(x=x, y=y, n_src=self.n_src, n_chan=self.n_chan)
        return y, tuple(new_states)

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
            "Exact low-memory recomputation from raw input history is not implemented for this model. "
            "Use forward_stream with layer caches for strict realtime equivalence."
        )

    def layer_cache_numel(self, batch_size: int = 1) -> int:
        states = self.init_stream_state(
            batch_size=batch_size,
            device=self.out_proj.weight.device,
            dtype=self.out_proj.weight.dtype,
        )
        return sum(int(s.numel()) for s in states)

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


class OnlineHierarchicalSoftBandSFCModel(nn.Module):
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
        d_model: int = 96,
        pre_layers: int = 1,
        mid_layers: int = 2,
        bottleneck_layers: int = 2,
        kernel_size: tuple[int, int] = (3, 3),
        causal: bool = True,
        masking: bool = True,
        routing_normalization: str = "softmax",
        dilation_cycle: tuple[int, ...] | list[int] | None = None,
        hierarchical_prior_mode: str = "inherited",
    ):
        super().__init__()
        self.core = OnlineHierarchicalSoftBandSFC2D(
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
            dilation_cycle=dilation_cycle,
            hierarchical_prior_mode=hierarchical_prior_mode,
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


def build_online_hierarchical_soft_band_sfc_system(
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
    d_model: int = 96,
    pre_layers: int = 1,
    mid_layers: int = 2,
    bottleneck_layers: int = 2,
    kernel_size: tuple[int, int] = (3, 3),
    causal: bool = True,
    masking: bool = True,
    routing_normalization: str = "softmax",
    dilation_cycle: tuple[int, ...] | list[int] | None = None,
    hierarchical_prior_mode: str = "inherited",
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
    core = OnlineHierarchicalSoftBandSFC2D(
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
        dilation_cycle=dilation_cycle,
        hierarchical_prior_mode=hierarchical_prior_mode,
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
