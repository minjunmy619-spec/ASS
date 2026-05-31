"""
Online / realtime SFC variant with query side-path and dilated separation.

This combines the two existing soft-band branches:
- ``online_soft_band_query_sfc_2d``: compressor emits latent tokens plus an
  explicit query side-path consumed by the decoder.
- ``online_soft_band_dilated_sfc_2d``: separator uses causal dilated time
  mixing and stateless local band-axis mixing.

The result keeps the same online/NPU contract as the sibling SFC models:
packed real/imag 2D tensors, explicit layer-cache streaming state, and fixed
frequency shapes for export.
"""

from __future__ import annotations

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
    CausalConv2d,
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


class BandAwareCapacityMixer2d(nn.Module):
    """State-free band-aware capacity branch.

    The expensive channel MLP runs on a pooled current-frame context so it does
    not add streaming state, but its output gates and colors a local band-mixed
    feature map. This gives the extra parameters a concrete separation role:
    condition current-frame local band features on a learned global latent-band
    summary instead of merely adding a broadcast bias.
    """

    def __init__(self, channels: int, hidden_channels: int):
        super().__init__()
        if hidden_channels <= 0:
            raise ValueError(f"hidden_channels must be positive, got {hidden_channels}")
        self.norm = RMSNorm2d(channels)
        self.band_dw = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, 3),
            padding=(0, 1),
            groups=channels,
            bias=True,
        )
        self.context_expand = nn.Conv2d(channels, 2 * hidden_channels, kernel_size=1, bias=True)
        self.context_project = nn.Conv2d(hidden_channels, 2 * channels, kernel_size=1, bias=True)
        self.local_project = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.band_dw(self.norm(x))
        context = local.mean(dim=3, keepdim=True)
        a, b = self.context_expand(context).chunk(2, dim=1)
        context = a * torch.sigmoid(b)
        gate, bias = self.context_project(context).chunk(2, dim=1)
        local = self.local_project(local * torch.sigmoid(gate))
        return x + local + bias


class OnlineSoftBandQueryDilatedSFC2D(nn.Module):
    """2D-only SFC core with query transport and dilated latent separator."""

    def __init__(
        self,
        n_freq: int,
        n_bands: int = 64,
        n_fft: int | None = None,
        sample_rate: int | None = None,
        band_config: str = "musical",
        n_src: int = 2,
        n_chan: int = 1,
        d_model: int = 96,
        n_layers: int = 12,
        kernel_size: tuple[int, int] = (3, 3),
        causal: bool = True,
        masking: bool = True,
        routing_normalization: str = "softmax",
        dilation_cycle: tuple[int, ...] | list[int] | None = None,
        pooled_mixer_hidden: int = 0,
        stateless_separator_layers: int = 0,
    ):
        super().__init__()
        if stateless_separator_layers < 0:
            raise ValueError(f"stateless_separator_layers must be non-negative, got {stateless_separator_layers}")
        if stateless_separator_layers > n_layers:
            raise ValueError(
                f"stateless_separator_layers={stateless_separator_layers} exceeds n_layers={n_layers}"
            )
        self.n_freq = n_freq
        self.n_bands = n_bands
        self.n_src = n_src
        self.n_chan = n_chan
        self.masking = masking
        self.dilation_schedule = _normalize_dilation_schedule(n_layers, dilation_cycle)
        self.pooled_mixer_hidden = pooled_mixer_hidden
        self.stateless_separator_layers = stateless_separator_layers

        in_ch = 2 * n_chan
        out_ch = 2 * n_src * n_chan

        band_spec = SoftBandSpec2d(
            n_freq=n_freq,
            n_bands=n_bands,
            n_fft=n_fft,
            sample_rate=sample_rate,
            band_config=band_config,
        )
        self.in_proj = nn.Sequential(
            nn.Conv2d(in_ch, d_model, kernel_size=1, bias=True),
            RMSNorm2d(d_model),
        )
        self.compressor = SoftBandQueryCompressor2d(
            channels=d_model,
            band_spec=band_spec,
            kernel_size=kernel_size,
            causal=causal,
            normalization=routing_normalization,
        )
        self.separator = nn.ModuleList(
            [
                DilatedBandMixBlock2d(
                    channels=d_model,
                    expansion=2,
                    time_kernel_size=1 if layer_idx < stateless_separator_layers else kernel_size[0],
                    band_kernel_size=kernel_size[1],
                    time_dilation=1 if layer_idx < stateless_separator_layers else dilation,
                    causal=causal,
                )
                for layer_idx, dilation in enumerate(self.dilation_schedule)
            ]
        )
        self.separator_uses_state = tuple(block.stream_context_frames() > 0 for block in self.separator)
        self.pooled_mixers = nn.ModuleList(
            [
                BandAwareCapacityMixer2d(d_model, pooled_mixer_hidden)
                if pooled_mixer_hidden > 0
                else nn.Identity()
                for _ in self.dilation_schedule
            ]
        )
        self.expander = SoftBandQueryExpander2d(channels=d_model, band_spec=band_spec)
        self.out_proj = nn.Conv2d(d_model, out_ch, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _runtime_assert(x.ndim == 4, f"Expected 4D input (B,C,T,F), got {x.shape}")
        _runtime_assert(x.shape[-1] == self.n_freq, f"{x.shape} vs {self.n_freq}")

        h = self.in_proj(x)
        z, q = self.compressor(h)
        for block, mixer in zip(self.separator, self.pooled_mixers):
            z = block(z)
            z = mixer(z)
        h = self.expander(z, q)
        y = self.out_proj(h)
        if self.masking:
            return _apply_packed_complex_mask_no_repeat(x=x, y=y, n_src=self.n_src, n_chan=self.n_chan)
        return y

    def stream_context_frames(self) -> int:
        if not isinstance(self.compressor.dw, CausalConv2d):
            return 0
        return self.compressor.stream_context_frames() + sum(block.stream_context_frames() for block in self.separator)

    def init_stream_state(
        self,
        batch_size: int = 1,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, ...]:
        if not isinstance(self.compressor.dw, CausalConv2d):
            raise RuntimeError("Streaming state is only supported when causal=True.")
        comp = self.compressor.init_stream_state(batch_size, freq_bins=self.n_freq, device=device, dtype=dtype)
        sep = tuple(
            block.init_stream_state(batch_size, freq_bins=self.n_bands, device=device, dtype=dtype)
            for block, uses_state in zip(self.separator, self.separator_uses_state)
            if uses_state
        )
        return (comp, *sep)

    def forward_stream(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, ...] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if not isinstance(self.compressor.dw, CausalConv2d):
            raise RuntimeError("forward_stream is only supported when causal=True.")

        _runtime_assert(x.ndim == 4, f"Expected 4D input (B,C,T,F), got {x.shape}")
        _runtime_assert(x.shape[-1] == self.n_freq, f"{x.shape} vs {self.n_freq}")

        if state is None:
            state = self.init_stream_state(batch_size=x.shape[0], device=x.device, dtype=x.dtype)

        expected_states = 1 + sum(1 for uses_state in self.separator_uses_state if uses_state)
        _runtime_assert(len(state) == expected_states, f"Unexpected state tuple: {len(state)}")
        comp_state = state[0]
        sep_state = state[1:]

        h = self.in_proj(x)
        (z, q), new_comp_state = self.compressor.forward_stream(h, comp_state)
        new_sep_states = []
        sep_state_idx = 0
        for block, mixer, uses_state in zip(self.separator, self.pooled_mixers, self.separator_uses_state):
            if uses_state:
                block_state = sep_state[sep_state_idx]
                z, block_state = block.forward_stream(z, block_state)
            else:
                z = block(z)
                block_state = None
            z = mixer(z)
            if uses_state:
                new_sep_states.append(block_state)
                sep_state_idx += 1
        h = self.expander(z, q)
        y = self.out_proj(h)
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


class OnlineSoftBandQueryDilatedSFCModel(nn.Module):
    """Torch wrapper that matches the existing complex STFT model contract."""

    def __init__(
        self,
        n_freq: int,
        n_bands: int = 64,
        n_fft: int | None = None,
        sample_rate: int | None = None,
        band_config: str = "musical",
        n_src: int = 2,
        n_chan: int = 1,
        d_model: int = 96,
        n_layers: int = 12,
        kernel_size: tuple[int, int] = (3, 3),
        causal: bool = True,
        masking: bool = True,
        routing_normalization: str = "softmax",
        dilation_cycle: tuple[int, ...] | list[int] | None = None,
        pooled_mixer_hidden: int = 0,
        stateless_separator_layers: int = 0,
    ):
        super().__init__()
        self.core = OnlineSoftBandQueryDilatedSFC2D(
            n_freq=n_freq,
            n_bands=n_bands,
            n_fft=n_fft,
            sample_rate=sample_rate,
            band_config=band_config,
            n_src=n_src,
            n_chan=n_chan,
            d_model=d_model,
            n_layers=n_layers,
            kernel_size=kernel_size,
            causal=causal,
            masking=masking,
            routing_normalization=routing_normalization,
            dilation_cycle=dilation_cycle,
            pooled_mixer_hidden=pooled_mixer_hidden,
            stateless_separator_layers=stateless_separator_layers,
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


def build_online_soft_band_query_dilated_sfc_system(
    *,
    n_fft: int,
    hop_length: int,
    fs: int,
    n_bands: int = 64,
    band_config: str = "musical",
    n_src: int = 2,
    n_chan: int = 1,
    d_model: int = 96,
    n_layers: int = 12,
    kernel_size: tuple[int, int] = (3, 3),
    causal: bool = True,
    masking: bool = True,
    routing_normalization: str = "softmax",
    dilation_cycle: tuple[int, ...] | list[int] | None = None,
    pooled_mixer_hidden: int = 0,
    stateless_separator_layers: int = 0,
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
    core = OnlineSoftBandQueryDilatedSFC2D(
        n_freq=core_n_freq,
        n_bands=n_bands,
        n_fft=core_n_fft,
        sample_rate=fs,
        band_config=band_config,
        n_src=n_src,
        n_chan=n_chan,
        d_model=d_model,
        n_layers=n_layers,
        kernel_size=kernel_size,
        causal=causal,
        masking=masking,
        routing_normalization=routing_normalization,
        dilation_cycle=dilation_cycle,
        pooled_mixer_hidden=pooled_mixer_hidden,
        stateless_separator_layers=stateless_separator_layers,
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
