"""
Online / realtime SFC variant with soft band routing and a ConvGRU separator.

The separator is implemented only with Conv2d + elementwise ops so it remains
friendly to ONNX / NPU compilation while introducing explicit recurrent state.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spectral_feature_compression.core.model.frequency_preprocessing import (
    FrequencyPreprocessedOnlineModel,
    build_frequency_preprocessor,
    resolve_preprocessed_n_freq,
)
from spectral_feature_compression.core.model.online_model_wrapper import OnlineModelWrapper
from spectral_feature_compression.core.model.online_sfc_2d import RMSNorm2d, _runtime_assert
from spectral_feature_compression.core.model.online_sfc_2d import _validate_npu_kernel_dilation_limit
from spectral_feature_compression.core.model.online_sfc_2d import apply_packed_complex_mask
from spectral_feature_compression.core.model.online_sfc_2d import pack_complex_stft_as_2d
from spectral_feature_compression.core.model.online_sfc_2d import unpack_2d_to_complex_stft
from spectral_feature_compression.core.model.online_soft_band_sfc_2d import CausalConv2d
from spectral_feature_compression.core.model.online_soft_band_sfc_2d import SoftBandCompressor2d
from spectral_feature_compression.core.model.online_soft_band_sfc_2d import SoftBandExpander2d
from spectral_feature_compression.core.model.online_soft_band_sfc_2d import SoftBandSpec2d


class ConvGRUBandCell2d(nn.Module):
    """
    GRU-like recurrent cell operating on one time frame at a time.

    The recurrence is explicit in the hidden state, while all learnable
    transforms are Conv2d with kernel sizes compatible with the target NPU.
    """

    def __init__(self, channels: int, band_kernel_size: int = 3):
        super().__init__()
        if band_kernel_size % 2 == 0:
            raise ValueError(f"band_kernel_size must be odd, got {band_kernel_size}")
        _validate_npu_kernel_dilation_limit(band_kernel_size, 1, axis="frequency")

        self.channels = channels
        self.input_norm = RMSNorm2d(channels)
        self.hidden_norm = RMSNorm2d(channels)

        self.x_proj = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=True)
        self.h_mix = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, band_kernel_size),
            padding=(0, band_kernel_size // 2),
            groups=channels,
            bias=True,
        )
        self.h_proj = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=True)
        self.out_norm = RMSNorm2d(channels)

    def init_state(
        self,
        batch_size: int,
        *,
        freq_bins: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        return torch.zeros(batch_size, self.channels, 1, freq_bins, device=device, dtype=dtype)

    def forward_frame(self, x_t: torch.Tensor, h_prev: torch.Tensor | None) -> torch.Tensor:
        _runtime_assert(x_t.ndim == 4 and x_t.shape[2] == 1, f"Expected single-frame 4D input, got {x_t.shape}")
        if h_prev is None:
            h_prev = self.init_state(
                x_t.shape[0],
                freq_bins=x_t.shape[-1],
                device=x_t.device,
                dtype=x_t.dtype,
            )

        xg = self.x_proj(self.input_norm(x_t))
        hg = self.h_proj(self.h_mix(self.hidden_norm(h_prev)))

        xr, xz, xn = xg.chunk(3, dim=1)
        hr, hz, hn = hg.chunk(3, dim=1)

        reset = torch.sigmoid(xr + hr)
        update = torch.sigmoid(xz + hz)
        cand = torch.tanh(xn + reset * hn)
        h_new = (1.0 - update) * cand + update * h_prev
        return self.out_norm(h_new + x_t)


class ConvGRUSeparator2d(nn.Module):
    def __init__(self, channels: int, n_layers: int, band_kernel_size: int = 3):
        super().__init__()
        self.layers = nn.ModuleList([ConvGRUBandCell2d(channels, band_kernel_size=band_kernel_size) for _ in range(n_layers)])
        self.channels = channels

    def init_stream_state(
        self,
        batch_size: int,
        *,
        freq_bins: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, ...]:
        return tuple(
            layer.init_state(batch_size, freq_bins=freq_bins, device=device, dtype=dtype)
            for layer in self.layers
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _runtime_assert(x.ndim == 4, f"Expected 4D latent tensor, got {x.shape}")
        states = [None] * len(self.layers)
        outputs = []
        for t in range(x.shape[2]):
            y_t = x[:, :, t : t + 1, :]
            new_states = []
            for layer, state in zip(self.layers, states):
                y_t = layer.forward_frame(y_t, state)
                new_states.append(y_t)
            states = new_states
            outputs.append(y_t)
        return torch.cat(outputs, dim=2)

    def forward_stream(
        self,
        x: torch.Tensor,
        states: tuple[torch.Tensor, ...] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        _runtime_assert(x.ndim == 4, f"Expected 4D latent tensor, got {x.shape}")
        if states is None:
            states = tuple([None] * len(self.layers))
        _runtime_assert(len(states) == len(self.layers), f"{len(states)} vs {len(self.layers)}")

        outputs = []
        current_states = list(states)
        for t in range(x.shape[2]):
            y_t = x[:, :, t : t + 1, :]
            new_states = []
            for layer, state in zip(self.layers, current_states):
                y_t = layer.forward_frame(y_t, state)
                new_states.append(y_t)
            current_states = new_states
            outputs.append(y_t)
        return torch.cat(outputs, dim=2), tuple(current_states)


class OnlineSoftBandGRUSFC2D(nn.Module):
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
        gru_band_kernel_size: int = 3,
    ):
        super().__init__()
        if not causal:
            raise ValueError("OnlineSoftBandGRUSFC2D currently supports only causal=True.")

        self.n_freq = n_freq
        self.n_bands = n_bands
        self.n_src = n_src
        self.n_chan = n_chan
        self.masking = masking

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
        self.separator = ConvGRUSeparator2d(channels=d_model, n_layers=n_layers, band_kernel_size=gru_band_kernel_size)
        self.expander = SoftBandExpander2d(channels=d_model, band_spec=band_spec)
        self.out_proj = nn.Conv2d(d_model, out_ch, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _runtime_assert(x.ndim == 4, f"Expected 4D input (B,C,T,F), got {x.shape}")
        _runtime_assert(x.shape[-1] == self.n_freq, f"{x.shape} vs {self.n_freq}")

        h = self.in_proj(x)
        z, _ = self.compressor(h)
        z = self.separator(z)
        h = self.expander(z)
        y = self.out_proj(h)
        if self.masking:
            return apply_packed_complex_mask(x=x, y=y, n_src=self.n_src, n_chan=self.n_chan)
        return y

    def stream_context_frames(self) -> int:
        if not isinstance(self.compressor.dw, CausalConv2d):
            return 0
        return self.compressor.stream_context_frames()

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
        sep = self.separator.init_stream_state(batch_size, freq_bins=self.n_bands, device=device, dtype=dtype)
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

        _runtime_assert(len(state) == 1 + len(self.separator.layers), f"Unexpected state tuple: {len(state)}")
        comp_state = state[0]
        sep_state = state[1:]

        h = self.in_proj(x)
        z, new_comp_state = self.compressor.forward_stream(h, comp_state)
        z, new_sep_states = self.separator.forward_stream(z, sep_state)
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
            "Use forward_stream with recurrent states for strict realtime equivalence."
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


class OnlineSoftBandGRUSFCModel(nn.Module):
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
        gru_band_kernel_size: int = 3,
    ):
        super().__init__()
        self.core = OnlineSoftBandGRUSFC2D(
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
            gru_band_kernel_size=gru_band_kernel_size,
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


def build_online_soft_band_gru_sfc_system(
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
    gru_band_kernel_size: int = 3,
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
    core = OnlineSoftBandGRUSFC2D(
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
        gru_band_kernel_size=gru_band_kernel_size,
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
