"""
Online / realtime SFC variant with soft band routing and a stronger separator.

This keeps the soft-band compressor / expander path, but replaces the plain
stacked 3x3 separator with a more expressive block:
- time-only causal depthwise conv with optional dilation
- band-axis depthwise mixing without extra streaming state
- 1x1 channel mixing before and after the separable conv stack

The design goal is to improve time receptive field efficiency while preserving:
- strict realtime causal behavior
- <= 4D tensors for ONNX / NPU deployment
- explicit and budgetable per-layer cache state
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
from spectral_feature_compression.core.model.online_sfc_2d import _validate_npu_kernel_dilation_limit
from spectral_feature_compression.core.model.online_sfc_2d import apply_packed_complex_mask
from spectral_feature_compression.core.model.online_sfc_2d import pack_complex_stft_as_2d
from spectral_feature_compression.core.model.online_sfc_2d import unpack_2d_to_complex_stft
from spectral_feature_compression.core.model.online_soft_band_sfc_2d import SoftBandCompressor2d
from spectral_feature_compression.core.model.online_soft_band_sfc_2d import SoftBandExpander2d
from spectral_feature_compression.core.model.online_soft_band_sfc_2d import SoftBandSpec2d


def _normalize_dilation_schedule(
    n_layers: int,
    dilation_cycle: tuple[int, ...] | list[int] | None,
) -> tuple[int, ...]:
    if dilation_cycle is None:
        dilation_cycle = (1, 2, 1, 2)
    dilation_cycle = tuple(int(d) for d in dilation_cycle)
    if len(dilation_cycle) == 0:
        raise ValueError("dilation_cycle must not be empty")
    if any(d <= 0 for d in dilation_cycle):
        raise ValueError(f"All dilations must be positive, got {dilation_cycle}")
    return tuple(dilation_cycle[layer_idx % len(dilation_cycle)] for layer_idx in range(n_layers))


class DilatedBandMixBlock2d(nn.Module):
    """
    Streaming-friendly separator block with split temporal and band mixing.

    The only streaming state comes from the causal time depthwise convolution.
    Band mixing stays local and stateless by using a centered depthwise conv
    along the compressed band axis.
    """

    def __init__(
        self,
        channels: int,
        expansion: int = 2,
        time_kernel_size: int = 3,
        band_kernel_size: int = 3,
        time_dilation: int = 1,
        causal: bool = True,
    ):
        super().__init__()
        if band_kernel_size % 2 == 0:
            raise ValueError(f"band_kernel_size must be odd, got {band_kernel_size}")
        _validate_npu_kernel_dilation_limit(time_kernel_size, time_dilation, axis="time")
        _validate_npu_kernel_dilation_limit(band_kernel_size, 1, axis="frequency")

        hidden = channels * expansion
        self.causal = causal

        self.norm1 = RMSNorm2d(channels)
        self.pw1 = nn.Conv2d(channels, hidden * 2, kernel_size=1, bias=True)
        self.time_dw = CausalConv2d(
            hidden,
            hidden,
            kernel_size=(time_kernel_size, 1),
            dilation=(time_dilation, 1),
            groups=hidden,
            bias=True,
        ) if causal else nn.Conv2d(
            hidden,
            hidden,
            kernel_size=(time_kernel_size, 1),
            padding=((time_kernel_size - 1) * time_dilation // 2, 0),
            dilation=(time_dilation, 1),
            groups=hidden,
            bias=True,
        )
        self.band_dw = nn.Conv2d(
            hidden,
            hidden,
            kernel_size=(1, band_kernel_size),
            padding=(0, band_kernel_size // 2),
            groups=hidden,
            bias=True,
        )
        self.pw2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
        self.norm2 = RMSNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, K)
        y = self.norm1(x)
        a, b = self.pw1(y).chunk(2, dim=1)
        # after gating: (B, hidden, T, K)
        y = a * torch.sigmoid(b)
        # time mixing keeps K and is causal only on T.
        y = self.time_dw(y)
        y = F.silu(y)
        # band mixing is local on K and introduces no streaming state.
        y = self.band_dw(y)
        y = F.silu(y)
        y = self.pw2(y)
        # output: (B, C, T, K)
        return self.norm2(x + y)

    def stream_context_frames(self) -> int:
        if isinstance(self.time_dw, CausalConv2d):
            return self.time_dw.stream_context_frames()
        return 0

    def init_stream_state(
        self,
        batch_size: int,
        *,
        freq_bins: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if not isinstance(self.time_dw, CausalConv2d):
            raise RuntimeError("Streaming state is only supported when causal=True.")
        return self.time_dw.init_stream_state(batch_size, freq_bins=freq_bins, device=device, dtype=dtype)

    def forward_stream(self, x: torch.Tensor, state: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(self.time_dw, CausalConv2d):
            raise RuntimeError("forward_stream is only supported when causal=True.")

        # x: (B, C, T_chunk, K), state: (B, hidden, ctx, K)
        y = self.norm1(x)
        a, b = self.pw1(y).chunk(2, dim=1)
        y = a * torch.sigmoid(b)
        y, new_state = self.time_dw.forward_stream(y, state)
        y = F.silu(y)
        y = self.band_dw(y)
        y = F.silu(y)
        y = self.pw2(y)
        # output: (B, C, T_chunk, K), new_state: (B, hidden, ctx, K)
        return self.norm2(x + y), new_state


class OnlineSoftBandDilatedSFC2D(nn.Module):
    """
    2D-only separator core with soft band routing and dilated separation.
    """

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
    ):
        super().__init__()
        self.n_freq = n_freq
        self.n_bands = n_bands
        self.n_src = n_src
        self.n_chan = n_chan
        self.masking = masking
        self.dilation_schedule = _normalize_dilation_schedule(n_layers, dilation_cycle)

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
        self.compressor = SoftBandCompressor2d(
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
                    time_kernel_size=kernel_size[0],
                    band_kernel_size=kernel_size[1],
                    time_dilation=dilation,
                    causal=causal,
                )
                for dilation in self.dilation_schedule
            ]
        )
        self.expander = SoftBandExpander2d(channels=d_model, band_spec=band_spec)
        self.out_proj = nn.Conv2d(d_model, out_ch, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _runtime_assert(x.ndim == 4, f"Expected 4D input (B,C,T,F), got {x.shape}")
        _runtime_assert(x.shape[-1] == self.n_freq, f"{x.shape} vs {self.n_freq}")

        h = self.in_proj(x)
        z, _ = self.compressor(h)
        for block in self.separator:
            z = block(z)
        h = self.expander(z)
        y = self.out_proj(h)

        if self.masking:
            return apply_packed_complex_mask(x=x, y=y, n_src=self.n_src, n_chan=self.n_chan)
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
            for block in self.separator
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

        _runtime_assert(len(state) == 1 + len(self.separator), f"Unexpected state tuple: {len(state)}")
        comp_state = state[0]
        sep_state = state[1:]

        h = self.in_proj(x)
        z, new_comp_state = self.compressor.forward_stream(h, comp_state)
        new_sep_states = []
        for block, block_state in zip(self.separator, sep_state):
            z, block_state = block.forward_stream(z, block_state)
            new_sep_states.append(block_state)
        h = self.expander(z)
        y = self.out_proj(h)
        if self.masking:
            y = apply_packed_complex_mask(x=x, y=y, n_src=self.n_src, n_chan=self.n_chan)
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


class OnlineSoftBandDilatedSFCModel(nn.Module):
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
    ):
        super().__init__()
        self.core = OnlineSoftBandDilatedSFC2D(
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


def build_online_soft_band_dilated_sfc_system(
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
    core = OnlineSoftBandDilatedSFC2D(
        n_freq=core_n_freq,
        n_bands=n_bands,
        n_fft=n_fft,
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
