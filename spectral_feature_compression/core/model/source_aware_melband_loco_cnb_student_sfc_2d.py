"""
Source-aware MelBand Loco-CNB strict-NPU student.

This model is a separate deployment-target version that combines the strongest
source-aware MelBand student priors with the causal Loco-CNB/FSMN temporal
memory used by BandSFCNetNPU:

* adaptive overlapped Mel routing on packed 4D STFT tensors;
* shared Loco-CNB backbone with local TF detail, cross-band mixing, causal FSMN
  memory, and rank-4 compressed-band attention;
* stateless source-aware seeding and source competition decoder;
* query-conditioned Mel expansion and source-shared complex mask head;
* optional stateless full-band mask correction and mixture consistency.

The recurrent state is intentionally spent on the compressed shared backbone,
not on source-specific or full-band branches.  This keeps the model suitable for
online/NPU export while giving it much longer temporal memory than the current
strong source-aware student.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from spectral_feature_compression.core.model.frequency_preprocessing import (
    FrequencyPreprocessedOnlineModel,
    build_frequency_preprocessor,
    build_hybrid_frequency_bin_frequencies,
    build_pcen_preprocessor,
    resolve_frequency_input_n_freq,
    resolve_preprocessed_n_freq,
)
from spectral_feature_compression.core.model.online_model_wrapper import OnlineModelWrapper
from spectral_feature_compression.core.model.online_sfc_2d import (
    CausalConv2d,
    RMSNorm2d,
    _runtime_assert,
    _validate_npu_kernel_dilation_limit,
    pack_complex_stft_as_2d,
    unpack_2d_to_complex_stft,
)
from spectral_feature_compression.core.model.source_aware_melband_strong_student_sfc_2d import (
    StrongAdaptiveMelRouter2d,
    StrongMaskCorrectionHead2d,
    StrongMelBandExpander2d,
    StrongSourceDecoder2d,
    StrongSourceMaskHead2d,
    StrongSourceSeed2d,
    StrongTokenFFN2d,
    _apply_packed_complex_mask_no_repeat,
    _as_pair,
    _packed_complex_features,
)


def _as_positive_int_tuple(value: Sequence[int] | None, *, name: str) -> tuple[int, ...] | None:
    if value is None:
        return None
    result = tuple(int(v) for v in value)
    if len(result) == 0:
        raise ValueError(f"{name} must not be empty")
    if any(v <= 0 for v in result):
        raise ValueError(f"{name} values must be positive, got {result}")
    return result


def _normalize_hidden_schedule(
    n_layers: int,
    *,
    hidden_channels: int,
    hidden_schedule: Sequence[int] | None,
) -> tuple[int, ...]:
    if n_layers < 0:
        raise ValueError(f"n_layers must be non-negative, got {n_layers}")
    if hidden_schedule is None:
        return tuple(int(hidden_channels) for _ in range(n_layers))
    schedule = tuple(int(v) for v in hidden_schedule)
    if len(schedule) != n_layers:
        raise ValueError(f"hidden_schedule must contain {n_layers} entries, got {schedule}")
    if any(v < 0 for v in schedule):
        raise ValueError(f"hidden schedule values must be >= 0, got {schedule}")
    return schedule


def _split_two(x: torch.Tensor, channels: int) -> tuple[torch.Tensor, torch.Tensor]:
    first, second = torch.split(x, channels, dim=1)
    return first, second


def _tree_numel(tree) -> int:
    if isinstance(tree, torch.Tensor):
        return int(tree.numel())
    return sum(_tree_numel(item) for item in tree)


class LocoPooledChannelMixer2d(nn.Module):
    """Frequency-pooled stateless channel capacity mixer.

    The expensive 1x1 projections run at band width 1 after a ReduceMean, so the
    block adds parameters cheaply and carries no streaming state.
    """

    def __init__(self, channels: int, hidden_channels: int):
        super().__init__()
        if hidden_channels <= 0:
            raise ValueError(f"hidden_channels must be positive, got {hidden_channels}")
        self.channels = int(channels)
        self.hidden_channels = int(hidden_channels)
        self.norm = RMSNorm2d(channels)
        self.expand = nn.Conv2d(channels, 2 * hidden_channels, kernel_size=1, bias=True)
        self.project = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _runtime_assert(x.ndim == 4, f"Expected 4D input, got {x.shape}")
        value, gate = _split_two(self.expand(self.norm(x).mean(dim=3, keepdim=True)), self.hidden_channels)
        y = value * torch.sigmoid(gate)
        return x + self.project(y) * self.scale


class LocoLocalTFMixer2d(nn.Module):
    """TF-Locoformer-style local current/detail mixer on compressed bands."""

    def __init__(
        self,
        channels: int,
        *,
        expansion: int = 1,
        ffn_expansion: int = 2,
        time_kernel: int = 3,
        band_kernel: int = 3,
        time_dilation: int = 1,
    ):
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if expansion <= 0 or ffn_expansion <= 0:
            raise ValueError("expansion and ffn_expansion must be positive")
        if band_kernel <= 0 or band_kernel % 2 != 1:
            raise ValueError(f"band_kernel must be a positive odd integer, got {band_kernel}")
        _validate_npu_kernel_dilation_limit(time_kernel, time_dilation, axis="time")
        _validate_npu_kernel_dilation_limit(band_kernel, 1, axis="band")

        hidden = int(channels) * int(expansion)
        ffn_hidden = int(channels) * int(ffn_expansion)
        self.channels = int(channels)
        self.hidden = int(hidden)
        self.ffn_hidden = int(ffn_hidden)
        self.time_norm = RMSNorm2d(channels)
        self.time_in = nn.Conv2d(channels, 2 * hidden, kernel_size=1, bias=True)
        self.time_dw = CausalConv2d(
            hidden,
            hidden,
            kernel_size=(time_kernel, 1),
            dilation=(time_dilation, 1),
            groups=hidden,
            bias=True,
        )
        self.time_out = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
        self.time_scale = nn.Parameter(torch.tensor(0.1))

        self.band_norm = RMSNorm2d(channels)
        self.band_in = nn.Conv2d(channels, 2 * hidden, kernel_size=1, bias=True)
        self.band_dw = nn.Conv2d(
            hidden,
            hidden,
            kernel_size=(1, band_kernel),
            padding=(0, band_kernel // 2),
            groups=hidden,
            bias=True,
        )
        self.band_out = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
        self.band_scale = nn.Parameter(torch.tensor(0.1))

        self.ffn_norm = RMSNorm2d(channels)
        self.ffn_in = nn.Conv2d(channels, 2 * ffn_hidden, kernel_size=1, bias=True)
        self.ffn_out = nn.Conv2d(ffn_hidden, channels, kernel_size=1, bias=True)
        self.ffn_scale = nn.Parameter(torch.tensor(0.1))

    @staticmethod
    def _gate(x: torch.Tensor, channels: int) -> torch.Tensor:
        value, gate = _split_two(x, channels)
        return value * torch.sigmoid(gate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _runtime_assert(x.ndim == 4, f"Expected 4D input, got {x.shape}")
        y = self._gate(self.time_in(self.time_norm(x)), self.hidden)
        y = self.time_dw(y)
        x = x + self.time_out(y) * self.time_scale

        y = self._gate(self.band_in(self.band_norm(x)), self.hidden)
        y = self.band_dw(y)
        x = x + self.band_out(y) * self.band_scale

        y = self._gate(self.ffn_in(self.ffn_norm(x)), self.ffn_hidden)
        return x + self.ffn_out(y) * self.ffn_scale

    def stream_context_frames(self) -> int:
        return self.time_dw.stream_context_frames()

    def init_stream_state(self, batch_size: int = 1, *, freq_bins: int, device=None, dtype=None) -> torch.Tensor:
        return self.time_dw.init_stream_state(batch_size, freq_bins=freq_bins, device=device, dtype=dtype)

    def forward_stream(self, x: torch.Tensor, state: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        _runtime_assert(x.ndim == 4, f"Expected 4D input, got {x.shape}")
        _runtime_assert(x.shape[2] == 1, f"Expected single-frame input, got T={x.shape[2]}")
        y = self._gate(self.time_in(self.time_norm(x)), self.hidden)
        y, new_state = self.time_dw.forward_stream(y, state)
        x = x + self.time_out(y) * self.time_scale

        y = self._gate(self.band_in(self.band_norm(x)), self.hidden)
        y = self.band_dw(y)
        x = x + self.band_out(y) * self.band_scale

        y = self._gate(self.ffn_in(self.ffn_norm(x)), self.ffn_hidden)
        return x + self.ffn_out(y) * self.ffn_scale, new_state


class LocoCrossBandMixer2d(nn.Module):
    """Grouped neighbouring-band mixer for compressed Mel tokens."""

    def __init__(self, channels: int, *, freq_kernel: int = 3):
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if freq_kernel <= 0 or freq_kernel % 2 != 1:
            raise ValueError(f"freq_kernel must be a positive odd integer, got {freq_kernel}")
        _validate_npu_kernel_dilation_limit(freq_kernel, 1, axis="band")
        self.channels = int(channels)
        self.norm = RMSNorm2d(channels)
        self.freq_conv = nn.Conv2d(
            channels,
            2 * channels,
            kernel_size=(1, freq_kernel),
            padding=(0, freq_kernel // 2),
            groups=channels,
            bias=True,
        )
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _runtime_assert(x.ndim == 4, f"Expected 4D input, got {x.shape}")
        value, gate = _split_two(self.freq_conv(self.norm(x)), self.channels)
        y = value * torch.sigmoid(gate)
        return self.pointwise(y) * self.scale


class LocoFSMNBandMixer2d(nn.Module):
    """Causal FSMN memory mixer on compressed bands.

    The default keeps separate dilation branches for compatibility with the
    original Loco-CNB design.  ``merge_dilations=True`` collapses those branches
    into one depthwise causal convolution whose kernel covers the same maximum
    look-back.  This removes small branch Conv/Add nodes from the deploy graph
    while preserving the streaming-state length.
    """

    def __init__(
        self,
        channels: int,
        *,
        kernel_t: int = 4,
        dilation_schedule: Sequence[int] = (1, 2, 3),
        merge_dilations: bool = False,
    ):
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        schedule = _as_positive_int_tuple(dilation_schedule, name="dilation_schedule")
        assert schedule is not None
        for dilation in schedule:
            _validate_npu_kernel_dilation_limit(kernel_t, dilation, axis="time")
        self.channels = int(channels)
        self.kernel_t = int(kernel_t)
        self.dilation_schedule = schedule
        self.max_context = max((self.kernel_t - 1) * dilation for dilation in schedule)
        self.merge_dilations = bool(merge_dilations)
        self.norm = RMSNorm2d(channels)
        self.in_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        if self.merge_dilations:
            self.memory = CausalConv2d(
                channels,
                channels,
                kernel_size=(self.max_context + 1, 1),
                dilation=(1, 1),
                groups=channels,
                bias=True,
            )
        else:
            self.memory = nn.ModuleList(
                [
                    CausalConv2d(
                        channels,
                        channels,
                        kernel_size=(self.kernel_t, 1),
                        dilation=(dilation, 1),
                        groups=channels,
                        bias=True,
                    )
                    for dilation in schedule
                ]
            )
        self.gate = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def _mix_branches_full(self, y: torch.Tensor) -> torch.Tensor:
        if self.merge_dilations:
            return self.memory(y)
        outputs = [branch(y) for branch in self.memory]
        mixed = outputs[0]
        for output in outputs[1:]:
            mixed = mixed + output
        return mixed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _runtime_assert(x.ndim == 4, f"Expected 4D input, got {x.shape}")
        y = self.in_proj(self.norm(x))
        y_mem = self._mix_branches_full(y)
        y = y_mem * torch.sigmoid(self.gate(y))
        return self.out_proj(y) * self.scale

    def stream_context_frames(self) -> int:
        return int(self.max_context)

    def init_stream_state(self, batch_size: int = 1, *, freq_bins: int, device=None, dtype=None) -> torch.Tensor:
        return torch.zeros(batch_size, self.channels, self.max_context, freq_bins, device=device, dtype=dtype)

    def forward_stream(self, x: torch.Tensor, state: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        _runtime_assert(x.ndim == 4, f"Expected 4D input, got {x.shape}")
        _runtime_assert(x.shape[2] == 1, f"Expected single-frame input, got T={x.shape[2]}")
        if state is None:
            state = self.init_stream_state(x.shape[0], freq_bins=x.shape[-1], device=x.device, dtype=x.dtype)
        y = self.in_proj(self.norm(x))
        if self.merge_dilations:
            y_mem, new_state = self.memory.forward_stream(y, state)
        else:
            history = torch.cat([state, y], dim=2)
            outputs: list[torch.Tensor] = []
            for branch in self.memory:
                outputs.append(branch.conv(history)[:, :, -1:, :])
            y_mem = outputs[0]
            for output in outputs[1:]:
                y_mem = y_mem + output
            new_state = history[:, :, 0:0, :] if self.max_context == 0 else history[:, :, -self.max_context :, :]
        y = y_mem * torch.sigmoid(self.gate(y))
        return self.out_proj(y) * self.scale, new_state


class LocoCompressedBandAttentionFusion2d(nn.Module):
    """Stateless compressed-band attention using rank-4 MatMul only."""

    def __init__(self, channels: int, *, num_heads: int = 4, head_dim: int = 8):
        super().__init__()
        if num_heads <= 0 or head_dim <= 0:
            raise ValueError("num_heads and head_dim must be positive")
        self.channels = int(channels)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.inner_dim = self.num_heads * self.head_dim
        self.norm = RMSNorm2d(channels)
        self.qkv_proj = nn.Conv2d(channels, 3 * self.inner_dim, kernel_size=1, bias=True)
        self.out_proj = nn.Conv2d(self.inner_dim, channels, kernel_size=1, bias=True)
        self.scale_value = 1.0 / float(self.head_dim) ** 0.5
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _runtime_assert(x.ndim == 4, f"Expected 4D input, got {x.shape}")
        q, k, v = torch.split(self.qkv_proj(self.norm(x)), self.inner_dim, dim=1)
        q_btkd = q.permute(0, 2, 3, 1)
        k_btdk = k.permute(0, 2, 1, 3)
        scores = torch.matmul(q_btkd, k_btdk) * self.scale_value
        attn = torch.softmax(scores, dim=-1)
        v_btkd = v.permute(0, 2, 3, 1)
        out = torch.matmul(attn, v_btkd).permute(0, 3, 1, 2)
        return self.out_proj(out) * self.residual_scale

    def forward_stream(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class SourceAwareLocoCNBBlock2d(nn.Module):
    """Shared Loco-CNB separator stage for source-aware MelBand tokens."""

    def __init__(
        self,
        channels: int,
        *,
        freq_kernel: int = 3,
        cnb_kernel: int = 4,
        cnb_dilation_schedule: Sequence[int] = (1, 2, 3),
        cnb_merge_dilations: bool = False,
        num_heads: int = 4,
        head_dim: int = 8,
        attention_enabled: bool = False,
        pooled_mixer_hidden: int = 0,
        loco_expansion: int = 1,
        loco_ffn_expansion: int = 2,
        loco_time_kernel: int = 3,
        loco_band_kernel: int = 3,
        loco_time_dilation: int = 1,
    ):
        super().__init__()
        self.local = LocoLocalTFMixer2d(
            channels,
            expansion=loco_expansion,
            ffn_expansion=loco_ffn_expansion,
            time_kernel=loco_time_kernel,
            band_kernel=loco_band_kernel,
            time_dilation=loco_time_dilation,
        )
        self.cross_band = LocoCrossBandMixer2d(channels, freq_kernel=freq_kernel)
        self.narrow_band = LocoFSMNBandMixer2d(
            channels,
            kernel_t=cnb_kernel,
            dilation_schedule=cnb_dilation_schedule,
            merge_dilations=cnb_merge_dilations,
        )
        self.band_attention = (
            LocoCompressedBandAttentionFusion2d(channels, num_heads=num_heads, head_dim=head_dim)
            if attention_enabled
            else None
        )
        self.pooled_mixer = (
            LocoPooledChannelMixer2d(channels, pooled_mixer_hidden) if int(pooled_mixer_hidden) > 0 else nn.Identity()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.local(z)
        z = z + self.cross_band(z)
        z = z + self.narrow_band(z)
        if self.band_attention is not None:
            z = z + self.band_attention(z)
        return self.pooled_mixer(z)

    def stream_context_frames(self) -> int:
        return self.local.stream_context_frames() + self.narrow_band.stream_context_frames()

    def init_stream_state(
        self, batch_size: int = 1, *, freq_bins: int, device=None, dtype=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        local_state = self.local.init_stream_state(batch_size, freq_bins=freq_bins, device=device, dtype=dtype)
        narrow_state = self.narrow_band.init_stream_state(batch_size, freq_bins=freq_bins, device=device, dtype=dtype)
        return local_state, narrow_state

    def forward_stream(
        self,
        z: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if state is None:
            state = self.init_stream_state(z.shape[0], freq_bins=z.shape[-1], device=z.device, dtype=z.dtype)
        local_state, narrow_state = state
        z, new_local_state = self.local.forward_stream(z, local_state)
        z = z + self.cross_band(z)
        narrow_out, new_narrow_state = self.narrow_band.forward_stream(z, narrow_state)
        z = z + narrow_out
        if self.band_attention is not None:
            z = z + self.band_attention.forward_stream(z)
        return self.pooled_mixer(z), (new_local_state, new_narrow_state)


class OnlineSourceAwareMelBandLocoCNBStudentSFC2D(nn.Module):
    """Source-aware MelBand student with a shared causal Loco-CNB backbone."""

    def __init__(
        self,
        n_freq: int,
        *,
        n_fft: int | None = None,
        sample_rate: int = 44100,
        n_src: int = 3,
        n_chan: int = 1,
        n_bands: int = 56,
        state_channels: int = 36,
        source_channels: int = 48,
        n_loco_layers: int = 4,
        n_source_layers: int = 4,
        source_local_expansion: int = 2,
        source_local_ffn_mult: int = 4,
        source_fusion_hidden: int = 192,
        source_seed_hidden: int = 192,
        expander_hidden: int = 128,
        mask_hidden: int = 160,
        correction_layers: int = 1,
        correction_channels: int = 16,
        correction_kernel_size: Sequence[int] | int = (1, 5),
        routing_kernel_size: Sequence[int] | int = (1, 3),
        loco_freq_kernel: int = 3,
        cnb_kernel: int = 4,
        cnb_dilation_schedule: Sequence[int] | None = (1, 2, 3),
        cnb_merge_dilations: bool = False,
        cnb_num_heads: int = 4,
        cnb_head_dim: int = 8,
        cnb_attention_enabled: bool = False,
        pooled_mixer_hidden: int = 0,
        pooled_mixer_hidden_schedule: Sequence[int] | None = None,
        loco_expansion: int = 1,
        loco_ffn_expansion: int = 2,
        loco_time_kernel: int = 3,
        loco_band_kernel: int = 3,
        loco_time_dilation: int = 1,
        low_freq_hz: float = 1000.0,
        low_freq_band_fraction: float = 0.45,
        overlap_factor: float = 1.5,
        low_freq_overlap_factor: float = 2.0,
        bin_frequencies_hz: torch.Tensor | Sequence[float] | None = None,
        include_magnitude_features: bool = True,
        include_logmag_features: bool = False,
        causal: bool = True,
        masking: bool = True,
        mixture_consistency: bool = True,
        routing_normalization: str = "softmax",
    ):
        super().__init__()
        del n_fft
        if not causal:
            raise ValueError("OnlineSourceAwareMelBandLocoCNBStudentSFC2D targets causal deployment only")
        if state_channels <= 0 or source_channels <= 0:
            raise ValueError("state_channels and source_channels must be positive")
        if n_loco_layers < 0 or n_source_layers < 0 or correction_layers < 0:
            raise ValueError("layer counts must be non-negative")
        routing_kernel_size = _as_pair(routing_kernel_size, name="routing_kernel_size")
        correction_kernel_size = _as_pair(correction_kernel_size, name="correction_kernel_size")
        if correction_kernel_size[0] != 1:
            raise ValueError(
                "Source-aware MelBand Loco-CNB keeps full-band correction stateless; "
                f"expected correction_kernel_size[0] == 1, got {correction_kernel_size}."
            )
        schedule = _as_positive_int_tuple(cnb_dilation_schedule, name="cnb_dilation_schedule") or (1, 2, 3)
        hidden_schedule = _normalize_hidden_schedule(
            n_loco_layers,
            hidden_channels=pooled_mixer_hidden,
            hidden_schedule=pooled_mixer_hidden_schedule,
        )

        self.n_freq = int(n_freq)
        self.n_bands = int(n_bands)
        self.n_src = int(n_src)
        self.n_chan = int(n_chan)
        self.state_channels = int(state_channels)
        self.source_channels = int(source_channels)
        self.d_model = self.source_channels
        self.n_loco_layers = int(n_loco_layers)
        self.n_source_layers = int(n_source_layers)
        self.correction_layers = int(correction_layers)
        self.correction_channels = int(correction_channels)
        self.cnb_merge_dilations = bool(cnb_merge_dilations)
        self.causal = True
        self.masking = bool(masking)
        self.mixture_consistency = bool(mixture_consistency)
        self.include_magnitude_features = bool(include_magnitude_features)
        self.include_logmag_features = bool(include_logmag_features)

        feature_channels = 2 * n_chan
        if include_magnitude_features:
            feature_channels += n_chan
        if include_logmag_features:
            feature_channels += n_chan

        from spectral_feature_compression.core.model.adaptive_mel_sfc_2d import AdaptiveMelBandSpec2d

        band_spec = AdaptiveMelBandSpec2d(
            n_freq=n_freq,
            n_bands=n_bands,
            sample_rate=sample_rate,
            low_freq_hz=low_freq_hz,
            low_freq_band_fraction=low_freq_band_fraction,
            overlap_factor=overlap_factor,
            low_freq_overlap_factor=low_freq_overlap_factor,
            bin_frequencies_hz=bin_frequencies_hz,
        )
        self.band_spec = band_spec
        self.frontend = nn.Sequential(
            nn.Conv2d(feature_channels, state_channels, kernel_size=1, bias=True),
            RMSNorm2d(state_channels),
            StrongTokenFFN2d(state_channels, max(state_channels * 3, source_seed_hidden // 2), residual_scale=0.1),
        )
        self.router = StrongAdaptiveMelRouter2d(
            channels=state_channels,
            band_spec=band_spec,
            kernel_size=routing_kernel_size,
            causal=True,
            normalization=routing_normalization,
        )
        self.backbone = nn.ModuleList(
            [
                SourceAwareLocoCNBBlock2d(
                    state_channels,
                    freq_kernel=loco_freq_kernel,
                    cnb_kernel=cnb_kernel,
                    cnb_dilation_schedule=schedule,
                    cnb_merge_dilations=self.cnb_merge_dilations,
                    num_heads=cnb_num_heads,
                    head_dim=cnb_head_dim,
                    attention_enabled=cnb_attention_enabled,
                    pooled_mixer_hidden=hidden,
                    loco_expansion=loco_expansion,
                    loco_ffn_expansion=loco_ffn_expansion,
                    loco_time_kernel=loco_time_kernel,
                    loco_band_kernel=loco_band_kernel,
                    loco_time_dilation=loco_time_dilation,
                )
                for hidden in hidden_schedule
            ]
        )
        self.source_project = nn.Sequential(
            RMSNorm2d(state_channels),
            nn.Conv2d(state_channels, source_channels, kernel_size=1, bias=True),
            RMSNorm2d(source_channels),
        )
        self.query_project = nn.Sequential(
            RMSNorm2d(state_channels),
            nn.Conv2d(state_channels, source_channels, kernel_size=1, bias=True),
            RMSNorm2d(source_channels),
        )
        self.source_seed = StrongSourceSeed2d(
            channels=source_channels,
            n_src=n_src,
            hidden_channels=source_seed_hidden,
        )
        self.source_decoder = StrongSourceDecoder2d(
            channels=source_channels,
            n_src=n_src,
            n_layers=n_source_layers,
            local_expansion=source_local_expansion,
            local_ffn_mult=source_local_ffn_mult,
            fusion_hidden_channels=source_fusion_hidden,
            band_kernel_size=loco_band_kernel if loco_band_kernel % 2 == 1 else 3,
        )
        self.mask_head = StrongSourceMaskHead2d(
            channels=source_channels,
            n_src=n_src,
            n_chan=n_chan,
            band_spec=band_spec,
            expander_hidden_channels=expander_hidden,
            mask_hidden_channels=mask_hidden,
            fullband_kernel_size=correction_kernel_size[1],
        )
        self.context_expander = StrongMelBandExpander2d(
            channels=source_channels,
            band_spec=band_spec,
            hidden_channels=expander_hidden,
        )
        self.context_fuse = nn.Sequential(
            RMSNorm2d(2 * source_channels),
            nn.Conv2d(2 * source_channels, 2 * source_channels, kernel_size=1, bias=True),
            nn.GLU(dim=1),
        )
        self.correction = StrongMaskCorrectionHead2d(
            context_channels=source_channels,
            correction_channels=correction_channels,
            n_freq=n_freq,
            n_src=n_src,
            n_chan=n_chan,
            n_layers=correction_layers,
            kernel_size=correction_kernel_size,
            causal=True,
        )

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = _packed_complex_features(
            x,
            n_chan=self.n_chan,
            include_magnitude=self.include_magnitude_features,
            include_logmag=self.include_logmag_features,
        )
        h = self.frontend(features)
        z, query_tokens = self.router(h)
        for block in self.backbone:
            z = block(z)
        return self.source_project(z), self.query_project(query_tokens)

    def _apply_mixture_consistency(self, estimates: torch.Tensor, mixture: torch.Tensor) -> torch.Tensor:
        if not self.mixture_consistency:
            return estimates
        chunks = torch.split(estimates, 2 * self.n_chan, dim=1)
        total = chunks[0]
        for chunk in chunks[1:]:
            total = total + chunk
        correction = (mixture - total) / float(self.n_src)
        return torch.cat([chunk + correction for chunk in chunks], dim=1)

    def _decode(self, mixture: torch.Tensor, z: torch.Tensor, query_tokens: torch.Tensor) -> torch.Tensor:
        source_tokens = self.source_seed(z)
        source_tokens = self.source_decoder(source_tokens, z)
        masks, source_context = self.mask_head(source_tokens, query_tokens)
        mixture_context = self.context_expander(z, query_tokens)
        correction_context = self.context_fuse(torch.cat([source_context, mixture_context], dim=1))
        masks = masks + self.correction(mixture, masks, correction_context)
        if not self.masking:
            return masks
        estimates = _apply_packed_complex_mask_no_repeat(
            x=mixture,
            y=masks,
            n_src=self.n_src,
            n_chan=self.n_chan,
        )
        return self._apply_mixture_consistency(estimates, mixture)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _runtime_assert(x.ndim == 4, f"Expected [B,2M,T,F], got {x.shape}")
        _runtime_assert(x.shape[1] == 2 * self.n_chan, f"Expected {2 * self.n_chan} channels, got {x.shape}")
        _runtime_assert(x.shape[-1] == self.n_freq, f"Expected F={self.n_freq}, got {x.shape}")
        z, query_tokens = self._encode(x)
        return self._decode(x, z, query_tokens)

    def stream_context_frames(self) -> int:
        return self.router.stream_context_frames() + sum(block.stream_context_frames() for block in self.backbone)

    def _router_state_count(self) -> int:
        return 1 if self.router.stream_context_frames() > 0 else 0

    def init_stream_state(self, batch_size: int = 1, *, device=None, dtype=None) -> tuple:
        states: list = []
        states.extend(self.router.init_stream_state(batch_size, freq_bins=self.n_freq, device=device, dtype=dtype))
        states.extend(
            block.init_stream_state(batch_size, freq_bins=self.n_bands, device=device, dtype=dtype)
            for block in self.backbone
        )
        return tuple(states)

    def forward_stream(self, x: torch.Tensor, state: tuple | None = None) -> tuple[torch.Tensor, tuple]:
        _runtime_assert(x.ndim == 4, f"Expected [B,2M,T,F], got {x.shape}")
        _runtime_assert(x.shape[2] == 1, f"Expected single-frame input, got T={x.shape[2]}")
        _runtime_assert(x.shape[1] == 2 * self.n_chan, f"Expected {2 * self.n_chan} channels, got {x.shape}")
        _runtime_assert(x.shape[-1] == self.n_freq, f"Expected F={self.n_freq}, got {x.shape}")
        if state is None:
            state = self.init_stream_state(batch_size=x.shape[0], device=x.device, dtype=x.dtype)
        router_count = self._router_state_count()
        expected = router_count + len(self.backbone)
        _runtime_assert(len(state) == expected, f"Expected {expected} states, got {len(state)}")

        features = _packed_complex_features(
            x,
            n_chan=self.n_chan,
            include_magnitude=self.include_magnitude_features,
            include_logmag=self.include_logmag_features,
        )
        h = self.frontend(features)
        (z, query_tokens), new_router_states = self.router.forward_stream(h, state[:router_count])
        new_states: list = [*new_router_states]
        state_idx = router_count
        for block in self.backbone:
            z, block_state = block.forward_stream(z, state[state_idx])
            state_idx += 1
            new_states.append(block_state)
        z = self.source_project(z)
        query_tokens = self.query_project(query_tokens)
        y = self._decode(x, z, query_tokens)
        return y, tuple(new_states)

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
            "for OnlineSourceAwareMelBandLocoCNBStudentSFC2D. Use forward_stream with layer caches."
        )

    def layer_cache_numel(self, batch_size: int = 1) -> int:
        states = self.init_stream_state(
            batch_size=batch_size,
            device=next(self.parameters()).device,
            dtype=next(self.parameters()).dtype,
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


class OnlineSourceAwareMelBandLocoCNBStudentSFCModel(nn.Module):
    """Complex-STFT wrapper around the Loco-CNB source-aware core."""

    def __init__(self, *, n_freq: int, n_src: int = 3, n_chan: int = 1, **kwargs):
        super().__init__()
        self.core = OnlineSourceAwareMelBandLocoCNBStudentSFC2D(n_freq=n_freq, n_src=n_src, n_chan=n_chan, **kwargs)
        self.n_src = int(n_src)
        self.n_chan = int(n_chan)

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


def _core_bin_frequencies_hz(
    *,
    n_fft: int,
    sample_rate: int,
    core_n_freq: int,
    freq_preprocess_enabled: bool,
    freq_preprocess_keep_bins: int | None,
    freq_preprocess_target_bins: int | None,
    freq_preprocess_mode: str,
    dc_bypass_enabled: bool,
) -> torch.Tensor | None:
    full_n_freq = (int(n_fft) // 2) + 1
    body_n_freq = resolve_frequency_input_n_freq(full_n_freq, dc_bypass_enabled=dc_bypass_enabled)
    if freq_preprocess_enabled:
        if freq_preprocess_keep_bins is None or freq_preprocess_target_bins is None:
            raise ValueError("keep_bins and target_bins are required for frequency preprocessing")
        return build_hybrid_frequency_bin_frequencies(
            body_n_freq,
            keep_bins=int(freq_preprocess_keep_bins),
            target_bins=int(freq_preprocess_target_bins),
            n_fft=n_fft,
            sample_rate=sample_rate,
            mode=freq_preprocess_mode,
            dc_bypass_enabled=dc_bypass_enabled,
        )
    if dc_bypass_enabled:
        first_bin = 1
        bin_indices = torch.arange(first_bin, first_bin + core_n_freq, dtype=torch.float32)
        return bin_indices * (float(sample_rate) / float(n_fft))
    return None


def build_source_aware_melband_loco_cnb_student_sfc_system(
    *,
    n_fft: int,
    hop_length: int,
    fs: int,
    n_src: int = 3,
    n_chan: int = 1,
    core_n_src: int | None = None,
    n_bands: int = 56,
    state_channels: int = 36,
    source_channels: int = 48,
    d_model: int | None = None,
    n_loco_layers: int = 4,
    n_source_layers: int = 4,
    n_layers: int | None = None,
    n_decoder_layers: int | None = None,
    source_local_expansion: int = 2,
    source_local_ffn_mult: int = 4,
    source_fusion_hidden: int = 192,
    source_seed_hidden: int = 192,
    expander_hidden: int = 128,
    mask_hidden: int = 160,
    correction_layers: int = 1,
    correction_channels: int = 16,
    correction_kernel_size: Sequence[int] | int = (1, 5),
    routing_kernel_size: Sequence[int] | int = (1, 3),
    loco_freq_kernel: int = 3,
    cnb_kernel: int = 4,
    cnb_dilation_schedule: Sequence[int] | None = (1, 2, 3),
    cnb_merge_dilations: bool = False,
    cnb_num_heads: int = 4,
    cnb_head_dim: int = 8,
    cnb_attention_enabled: bool = False,
    pooled_mixer_hidden: int = 0,
    pooled_mixer_hidden_schedule: Sequence[int] | None = None,
    loco_expansion: int = 1,
    loco_ffn_expansion: int = 2,
    loco_time_kernel: int = 3,
    loco_band_kernel: int = 3,
    loco_time_dilation: int = 1,
    low_freq_hz: float = 1000.0,
    low_freq_band_fraction: float = 0.45,
    overlap_factor: float = 1.5,
    low_freq_overlap_factor: float = 2.0,
    include_magnitude_features: bool = True,
    include_logmag_features: bool = False,
    causal: bool = True,
    masking: bool = True,
    mixture_consistency: bool = True,
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
    residual_source_enabled: bool = False,
    residual_source_index: int | None = None,
    scaling: bool = False,
    css_segment_size: int = 12,
    css_shift_size: int = 6,
    css_batch_size: int = 1,
) -> OnlineModelWrapper:
    if d_model is not None:
        source_channels = int(d_model)
    if n_layers is not None:
        n_loco_layers = int(n_layers)
    if n_decoder_layers is not None:
        n_source_layers = int(n_decoder_layers)
    explicit_n_src = (
        int(core_n_src) if core_n_src is not None else (int(n_src) - 1 if residual_source_enabled else int(n_src))
    )
    if residual_source_enabled and explicit_n_src != int(n_src) - 1:
        raise ValueError(f"residual_source_enabled expects core_n_src=n_src-1={int(n_src) - 1}, got {explicit_n_src}")
    if not residual_source_enabled and explicit_n_src != int(n_src):
        raise ValueError(f"core_n_src={explicit_n_src} requires residual_source_enabled=true when n_src={int(n_src)}")
    if residual_source_enabled and mixture_consistency:
        raise ValueError(
            "residual_source_enabled requires mixture_consistency=False in the explicit core; "
            "otherwise explicit stems are forced to sum to the mixture and the residual source collapses."
        )

    full_n_freq = (n_fft // 2) + 1
    core_n_freq = resolve_preprocessed_n_freq(
        full_n_freq,
        enabled=freq_preprocess_enabled,
        keep_bins=freq_preprocess_keep_bins,
        target_bins=freq_preprocess_target_bins,
        dc_bypass_enabled=dc_bypass_enabled,
    )
    core_n_fft = 2 * (core_n_freq - 1)
    bin_frequencies_hz = _core_bin_frequencies_hz(
        n_fft=n_fft,
        sample_rate=fs,
        core_n_freq=core_n_freq,
        freq_preprocess_enabled=freq_preprocess_enabled,
        freq_preprocess_keep_bins=freq_preprocess_keep_bins,
        freq_preprocess_target_bins=freq_preprocess_target_bins,
        freq_preprocess_mode=freq_preprocess_mode,
        dc_bypass_enabled=dc_bypass_enabled,
    )
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
    core = OnlineSourceAwareMelBandLocoCNBStudentSFC2D(
        n_freq=core_n_freq,
        n_fft=core_n_fft,
        sample_rate=fs,
        n_src=explicit_n_src,
        n_chan=n_chan,
        n_bands=n_bands,
        state_channels=state_channels,
        source_channels=source_channels,
        n_loco_layers=n_loco_layers,
        n_source_layers=n_source_layers,
        source_local_expansion=source_local_expansion,
        source_local_ffn_mult=source_local_ffn_mult,
        source_fusion_hidden=source_fusion_hidden,
        source_seed_hidden=source_seed_hidden,
        expander_hidden=expander_hidden,
        mask_hidden=mask_hidden,
        correction_layers=correction_layers,
        correction_channels=correction_channels,
        correction_kernel_size=correction_kernel_size,
        routing_kernel_size=routing_kernel_size,
        loco_freq_kernel=loco_freq_kernel,
        cnb_kernel=cnb_kernel,
        cnb_dilation_schedule=cnb_dilation_schedule,
        cnb_merge_dilations=cnb_merge_dilations,
        cnb_num_heads=cnb_num_heads,
        cnb_head_dim=cnb_head_dim,
        cnb_attention_enabled=cnb_attention_enabled,
        pooled_mixer_hidden=pooled_mixer_hidden,
        pooled_mixer_hidden_schedule=pooled_mixer_hidden_schedule,
        loco_expansion=loco_expansion,
        loco_ffn_expansion=loco_ffn_expansion,
        loco_time_kernel=loco_time_kernel,
        loco_band_kernel=loco_band_kernel,
        loco_time_dilation=loco_time_dilation,
        low_freq_hz=low_freq_hz,
        low_freq_band_fraction=low_freq_band_fraction,
        overlap_factor=overlap_factor,
        low_freq_overlap_factor=low_freq_overlap_factor,
        bin_frequencies_hz=bin_frequencies_hz,
        include_magnitude_features=include_magnitude_features,
        include_logmag_features=include_logmag_features,
        causal=causal,
        masking=masking,
        mixture_consistency=mixture_consistency,
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
        residual_source_enabled=residual_source_enabled,
        residual_source_index=residual_source_index,
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
