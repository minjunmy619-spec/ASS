"""32-band SFC + Conv2D Locoformer student for NPU deployment.

This variant is meant to match the user's strongest observed lite-SFC direction:
one adaptive SFC compression/expansion pass, a very small number of Conv2D
Locoformer blocks, and explicit speech/music masks with SFX reconstructed by
the outer frequency wrapper when requested.
"""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectral_feature_compression.core.model.adaptive_mel_locoformer_lite_sfc_2d import (
    _normalize_dilation_schedule,
)
from spectral_feature_compression.core.model.adaptive_mel_sfc_2d import (
    AdaptiveMelBandSpec2d,
    _apply_packed_complex_mask_no_repeat,
    _as_pair,
)
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
from spectral_feature_compression.core.model.online_soft_band_query_sfc_2d import (
    SoftBandQueryCompressor2d,
    SoftBandQueryExpander2d,
)


def _logit(value: float) -> float:
    value = min(max(float(value), 1.0e-4), 1.0 - 1.0e-4)
    return math.log(value / (1.0 - value))


class SourceWiseSigmoidTanhComplexHead2d(nn.Module):
    """Small source-aware complex mask head using regular Conv2D only."""

    def __init__(
        self,
        *,
        in_channels: int,
        n_src: int,
        n_chan: int,
        hidden_channels: int = 128,
        refine_layers: int = 1,
        source_kernel_size: int = 5,
        real_mask_scale: float = 1.5,
        imag_mask_scale: float = 0.12,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or hidden_channels <= 0:
            raise ValueError("in_channels and hidden_channels must be positive")
        if n_src <= 0 or n_chan <= 0:
            raise ValueError("n_src and n_chan must be positive")
        if refine_layers < 0:
            raise ValueError(f"refine_layers must be non-negative, got {refine_layers}")
        if source_kernel_size <= 0 or source_kernel_size % 2 != 1:
            raise ValueError(f"source_kernel_size must be a positive odd integer, got {source_kernel_size}")
        if real_mask_scale <= 0.0 or imag_mask_scale <= 0.0:
            raise ValueError("mask scales must be positive")
        del source_kernel_size

        self.n_src = int(n_src)
        self.n_chan = int(n_chan)
        self.hidden_channels = int(hidden_channels)
        self.real_mask_scale = float(real_mask_scale)
        self.imag_mask_scale = float(imag_mask_scale)

        self.seed = nn.Sequential(
            nn.Conv2d(in_channels, self.hidden_channels, kernel_size=1, bias=True),
            nn.SiLU(),
        )
        self.refine = nn.ModuleList()
        for _ in range(int(refine_layers)):
            self.refine.append(nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=1, bias=True))
        self.out_norm = RMSNorm2d(self.hidden_channels)
        self.mask = nn.Conv2d(
            self.hidden_channels,
            2 * self.n_src * self.n_chan,
            kernel_size=1,
            bias=True,
        )
        self._init_mask_bias()

    def _init_mask_bias(self) -> None:
        target_real = min(0.45 / max(self.real_mask_scale, 1.0e-6), 0.95)
        with torch.no_grad():
            self.mask.bias.zero_()
            self.mask.bias[0::2].fill_(_logit(target_real))

    def _mask_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        real = torch.sigmoid(logits[:, 0::2, :, :]) * self.real_mask_scale
        imag = torch.tanh(logits[:, 1::2, :, :]) * self.imag_mask_scale
        chunks: list[torch.Tensor] = []
        for idx in range(self.n_src * self.n_chan):
            chunks.append(real[:, idx : idx + 1, :, :])
            chunks.append(imag[:, idx : idx + 1, :, :])
        return torch.cat(chunks, dim=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.seed(x)
        for layer in self.refine:
            h = h + F.silu(layer(h))
        logits = self.mask(self.out_norm(h))
        return self._mask_from_logits(logits), logits


class RegularConv2DLocoformerBlock2d(nn.Module):
    """Conv2D Locoformer-lite block without grouped/depthwise Conv nodes."""

    def __init__(
        self,
        *,
        channels: int,
        expansion: int = 2,
        ffn_expansion: int = 4,
        time_kernel_size: int = 3,
        band_kernel_size: int = 3,
        time_dilation: int = 1,
        causal: bool = True,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if expansion <= 0 or ffn_expansion <= 0:
            raise ValueError("expansion factors must be positive")
        if band_kernel_size % 2 != 1:
            raise ValueError(f"band_kernel_size must be odd, got {band_kernel_size}")
        if not causal:
            raise ValueError("RegularConv2DLocoformerBlock2d supports causal=True only")
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
        self.time_conv = CausalConv2d(
            hidden,
            hidden,
            kernel_size=(time_kernel_size, 1),
            dilation=(time_dilation, 1),
            groups=1,
            bias=True,
        )
        self.time_out = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)

        self.band_norm = RMSNorm2d(channels)
        self.band_in = nn.Conv2d(channels, 2 * hidden, kernel_size=1, bias=True)
        self.band_conv = nn.Conv2d(
            hidden,
            hidden,
            kernel_size=(1, band_kernel_size),
            padding=(0, band_kernel_size // 2),
            groups=1,
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
        y = F.silu(self.time_conv(y))
        x = x + self.time_out(y)

        y = self._gated(self.band_in(self.band_norm(x)))
        y = F.silu(self.band_conv(y))
        x = x + self.band_out(y)

        y = self._gated(self.ffn_in(self.ffn_norm(x)))
        return x + self.ffn_out(y)

    def stream_context_frames(self) -> int:
        return self.time_conv.stream_context_frames()

    def init_stream_state(self, batch_size: int = 1, *, freq_bins: int, device=None, dtype=None) -> torch.Tensor:
        return self.time_conv.init_stream_state(batch_size, freq_bins=freq_bins, device=device, dtype=dtype)

    def forward_stream(self, x: torch.Tensor, state: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        y = self._gated(self.time_in(self.time_norm(x)))
        y, new_state = self.time_conv.forward_stream(y, state)
        y = F.silu(y)
        x = x + self.time_out(y)

        y = self._gated(self.band_in(self.band_norm(x)))
        y = F.silu(self.band_conv(y))
        x = x + self.band_out(y)

        y = self._gated(self.ffn_in(self.ffn_norm(x)))
        return x + self.ffn_out(y), new_state


class OnlineSFCLocoformerConv2D32BNPU2D(nn.Module):
    """NPU-oriented 32-band SFC student with a 2-layer Conv2D Locoformer trunk."""

    def __init__(
        self,
        *,
        n_freq: int,
        n_bands: int = 32,
        n_fft: int | None = None,
        sample_rate: int = 24000,
        n_src: int = 2,
        n_chan: int = 1,
        d_model: int = 192,
        n_loco_layers: int = 2,
        kernel_size: Sequence[int] | int = (3, 3),
        routing_kernel_size: Sequence[int] | int = (1, 3),
        dilation_cycle: Sequence[int] | None = (1, 2),
        expansion: int = 2,
        ffn_expansion: int = 4,
        source_head_channels: int = 128,
        source_refine_layers: int = 1,
        source_kernel_size: int = 5,
        real_mask_scale: float = 1.5,
        imag_mask_scale: float = 0.12,
        low_freq_hz: float = 1000.0,
        low_freq_band_fraction: float = 0.45,
        overlap_factor: float = 1.5,
        low_freq_overlap_factor: float = 2.0,
        bin_frequencies_hz: torch.Tensor | Sequence[float] | None = None,
        causal: bool = True,
        masking: bool = True,
        routing_normalization: str = "softmax",
    ) -> None:
        super().__init__()
        del n_fft
        if not causal:
            raise ValueError("OnlineSFCLocoformerConv2D32BNPU2D supports causal=True only")
        kernel_size = _as_pair(kernel_size, name="kernel_size")
        routing_kernel_size = _as_pair(routing_kernel_size, name="routing_kernel_size")

        self.n_freq = int(n_freq)
        self.n_bands = int(n_bands)
        self.n_src = int(n_src)
        self.n_chan = int(n_chan)
        self.d_model = int(d_model)
        self.masking = bool(masking)
        self.causal = bool(causal)
        self.dilation_schedule = _normalize_dilation_schedule(int(n_loco_layers), dilation_cycle)

        self.band_spec = AdaptiveMelBandSpec2d(
            n_freq=self.n_freq,
            n_bands=self.n_bands,
            sample_rate=sample_rate,
            low_freq_hz=low_freq_hz,
            low_freq_band_fraction=low_freq_band_fraction,
            overlap_factor=overlap_factor,
            low_freq_overlap_factor=low_freq_overlap_factor,
            bin_frequencies_hz=bin_frequencies_hz,
        )
        self.in_proj = nn.Sequential(
            nn.Conv2d(2 * self.n_chan, self.d_model, kernel_size=1, bias=True),
            RMSNorm2d(self.d_model),
        )
        self.compressor = SoftBandQueryCompressor2d(
            channels=self.d_model,
            band_spec=self.band_spec,
            kernel_size=routing_kernel_size,
            causal=True,
            normalization=routing_normalization,
        )
        self.separator = nn.ModuleList(
            [
                RegularConv2DLocoformerBlock2d(
                    channels=self.d_model,
                    expansion=expansion,
                    ffn_expansion=ffn_expansion,
                    time_kernel_size=kernel_size[0],
                    band_kernel_size=kernel_size[1],
                    time_dilation=dilation,
                    causal=True,
                )
                for dilation in self.dilation_schedule
            ]
        )
        self.expander = SoftBandQueryExpander2d(channels=self.d_model, band_spec=self.band_spec)
        self.source_head = SourceWiseSigmoidTanhComplexHead2d(
            in_channels=self.d_model,
            n_src=self.n_src,
            n_chan=self.n_chan,
            hidden_channels=source_head_channels,
            refine_layers=source_refine_layers,
            source_kernel_size=source_kernel_size,
            real_mask_scale=real_mask_scale,
            imag_mask_scale=imag_mask_scale,
        )

    def _encode_decode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        z, query_tokens = self.compressor(h)
        for block in self.separator:
            z = block(z)
        return self.expander(z, query_tokens)

    def _decode_masks(self, x: torch.Tensor, h: torch.Tensor, *, return_aux: bool = False):
        masks, logits = self.source_head(h)
        if self.masking:
            y = _apply_packed_complex_mask_no_repeat(x=x, y=masks, n_src=self.n_src, n_chan=self.n_chan)
        else:
            y = masks
        if not return_aux:
            return y
        return y, {
            "mask": masks,
            "mask_domain": "packed_complex_mask",
            "mask_logits": logits,
            "mask_logits_domain": "sfc_locoformer_conv2d_32b_complex_mask_logits",
            "mask_logits_transform": "sigmoid_tanh_complex_mask",
            "mask_logits_real_scale": self.source_head.real_mask_scale,
            "mask_logits_imag_scale": self.source_head.imag_mask_scale,
        }

    def forward(self, x: torch.Tensor, *, return_aux: bool = False):
        _runtime_assert(x.ndim == 4, f"Expected 4D input (B,C,T,F), got {x.shape}")
        _runtime_assert(x.shape[1] == 2 * self.n_chan, f"Expected {2 * self.n_chan} input channels, got {x.shape}")
        _runtime_assert(x.shape[-1] == self.n_freq, f"{x.shape} vs {self.n_freq}")
        h = self._encode_decode(x)
        return self._decode_masks(x, h, return_aux=return_aux)

    def stream_context_frames(self) -> int:
        return self.compressor.stream_context_frames() + sum(block.stream_context_frames() for block in self.separator)

    def init_stream_state(self, batch_size: int = 1, *, device=None, dtype=None) -> tuple[torch.Tensor, ...]:
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
        _runtime_assert(x.shape[2] == 1, f"Expected single-frame streaming input, got T={x.shape[2]}")
        _runtime_assert(x.shape[-1] == self.n_freq, f"{x.shape} vs {self.n_freq}")
        if state is None:
            state = self.init_stream_state(batch_size=x.shape[0], device=x.device, dtype=x.dtype)
        _runtime_assert(len(state) == 1 + len(self.separator), f"Unexpected state tuple: {len(state)}")

        h = self.in_proj(x)
        (z, query_tokens), new_comp_state = self.compressor.forward_stream(h, state[0])
        new_sep_states = []
        for block, block_state in zip(self.separator, state[1:]):
            z, block_state = block.forward_stream(z, block_state)
            new_sep_states.append(block_state)
        h = self.expander(z, query_tokens)
        y = self._decode_masks(x, h, return_aux=False)
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
            "Exact low-memory recomputation is not implemented for OnlineSFCLocoformerConv2D32BNPU2D. "
            "Use forward_stream with layer caches for strict equivalence."
        )

    def layer_cache_numel(self, batch_size: int = 1) -> int:
        states = self.init_stream_state(
            batch_size=batch_size,
            device=next(self.parameters()).device,
            dtype=next(self.parameters()).dtype,
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


class OnlineSFCLocoformerConv2D32BNPUModel(nn.Module):
    """Complex-STFT wrapper around the 32-band Conv2D SFC-Locoformer core."""

    def __init__(self, *, n_freq: int, n_src: int = 2, n_chan: int = 1, **kwargs) -> None:
        super().__init__()
        self.core = OnlineSFCLocoformerConv2D32BNPU2D(n_freq=n_freq, n_src=n_src, n_chan=n_chan, **kwargs)
        self.n_src = int(n_src)
        self.n_chan = int(n_chan)

    def forward(self, x: torch.Tensor, **kwargs):
        kwargs.pop("ref", None)
        return_aux = bool(kwargs.pop("return_aux", False))
        x2d = pack_complex_stft_as_2d(x)
        core_output = self.core(x2d, return_aux=return_aux)
        if isinstance(core_output, tuple):
            y2d, aux = core_output
        else:
            y2d = core_output
            aux = {}
        estimate = unpack_2d_to_complex_stft(y2d, n_src=self.n_src, n_chan=self.n_chan)
        if return_aux:
            return estimate, aux
        return estimate

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
        basis_mode = (
            "triangular" if freq_preprocess_mode in {"learnable_query", "sfclite_query"} else freq_preprocess_mode
        )
        return build_hybrid_frequency_bin_frequencies(
            body_n_freq,
            keep_bins=int(freq_preprocess_keep_bins),
            target_bins=int(freq_preprocess_target_bins),
            n_fft=n_fft,
            sample_rate=sample_rate,
            mode=basis_mode,
            dc_bypass_enabled=dc_bypass_enabled,
        )
    if dc_bypass_enabled:
        first_bin = 1
        bin_indices = torch.arange(first_bin, first_bin + core_n_freq, dtype=torch.float32)
        return bin_indices * (float(sample_rate) / float(n_fft))
    return None


def build_sfc_locoformer_conv2d_32b_npu_system(
    *,
    n_fft: int,
    hop_length: int,
    fs: int,
    n_src: int = 3,
    n_chan: int = 1,
    core_n_src: int | None = None,
    n_bands: int = 32,
    d_model: int = 192,
    n_loco_layers: int = 2,
    kernel_size: Sequence[int] | int = (3, 3),
    routing_kernel_size: Sequence[int] | int = (1, 3),
    dilation_cycle: Sequence[int] | None = (1, 2),
    expansion: int = 2,
    ffn_expansion: int = 4,
    source_head_channels: int = 128,
    source_refine_layers: int = 1,
    source_kernel_size: int = 5,
    real_mask_scale: float = 1.5,
    imag_mask_scale: float = 0.12,
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
    residual_source_enabled: bool = True,
    residual_source_index: int | None = None,
    scaling: bool = False,
    css_segment_size: int = 12,
    css_shift_size: int = 6,
    css_batch_size: int = 1,
) -> OnlineModelWrapper:
    explicit_n_src = (
        int(core_n_src) if core_n_src is not None else (int(n_src) - 1 if residual_source_enabled else int(n_src))
    )
    if residual_source_enabled and explicit_n_src != int(n_src) - 1:
        raise ValueError(f"residual_source_enabled expects core_n_src=n_src-1={int(n_src) - 1}, got {explicit_n_src}")
    if not residual_source_enabled and explicit_n_src != int(n_src):
        raise ValueError(f"core_n_src={explicit_n_src} requires residual_source_enabled=true when n_src={int(n_src)}")

    full_n_freq = (int(n_fft) // 2) + 1
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
    core = OnlineSFCLocoformerConv2D32BNPU2D(
        n_freq=core_n_freq,
        n_bands=n_bands,
        n_fft=core_n_fft,
        sample_rate=fs,
        n_src=explicit_n_src,
        n_chan=n_chan,
        d_model=d_model,
        n_loco_layers=n_loco_layers,
        kernel_size=kernel_size,
        routing_kernel_size=routing_kernel_size,
        dilation_cycle=dilation_cycle,
        expansion=expansion,
        ffn_expansion=ffn_expansion,
        source_head_channels=source_head_channels,
        source_refine_layers=source_refine_layers,
        source_kernel_size=source_kernel_size,
        real_mask_scale=real_mask_scale,
        imag_mask_scale=imag_mask_scale,
        low_freq_hz=low_freq_hz,
        low_freq_band_fraction=low_freq_band_fraction,
        overlap_factor=overlap_factor,
        low_freq_overlap_factor=low_freq_overlap_factor,
        bin_frequencies_hz=bin_frequencies_hz,
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
