"""
Online / realtime SFC variant with soft band routing.

This module keeps the SFC-style inductive bias without attention:
- a fixed number of band slots compresses F bins into K latent bands,
- each slot has a static frequency prior derived from a band specification,
- the actual routing weights are still input-adaptive on each frame,
- all runtime tensors stay <= 4D and the separator uses only 2D ops.
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
from spectral_feature_compression.core.model.bandit_split import get_band_specs
from spectral_feature_compression.core.model.online_model_wrapper import OnlineModelWrapper
from spectral_feature_compression.core.model.online_sfc_2d import (
    CausalConv2d,
    OnlineConvBlock,
    RMSNorm2d,
    SameConv2d,
    _runtime_assert,
    apply_packed_complex_mask,
    pack_complex_stft_as_2d,
    unpack_2d_to_complex_stft,
)


class SoftBandSpec2d(nn.Module):
    """
    Static band definitions used as soft priors for routing and expansion.
    """

    def __init__(
        self,
        n_freq: int,
        n_bands: int,
        n_fft: int | None = None,
        sample_rate: int | None = None,
        band_config: str = "musical",
    ):
        super().__init__()
        assert n_freq > 0
        assert n_bands > 0

        starts, ends, basis = self._build_basis(
            n_freq=n_freq,
            n_bands=n_bands,
            n_fft=n_fft,
            sample_rate=sample_rate,
            band_config=band_config,
        )
        self.n_freq = n_freq
        self.n_bands = n_bands
        self.band_config = band_config

        self.register_buffer("starts", starts)
        self.register_buffer("ends", ends)
        self.register_buffer("basis", basis.view(1, n_bands, 1, n_freq))

    @staticmethod
    def _build_speech_lowfreq_narrow_basis(
        n_freq: int,
        n_bands: int,
        sample_rate: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if sample_rate is None:
            raise ValueError("speech_lowfreq_narrow requires sample_rate.")

        nyquist = 0.5 * float(sample_rate)
        hz_edges = [0.0, min(1000.0, nyquist), min(2000.0, nyquist), min(4000.0, nyquist), nyquist]
        # Higher slope => more bands allocated in that region.
        slopes = [1.0 / 25.0, 1.0 / 100.0, 1.0 / 250.0, 1.0 / 500.0]

        importance_breaks = [0.0]
        for idx, slope in enumerate(slopes):
            span_hz = max(hz_edges[idx + 1] - hz_edges[idx], 0.0)
            importance_breaks.append(importance_breaks[-1] + span_hz * slope)
        importance_breaks = torch.tensor(importance_breaks, dtype=torch.float32)

        def hz_to_importance(hz: torch.Tensor) -> torch.Tensor:
            value = torch.zeros_like(hz)
            for idx, slope in enumerate(slopes):
                left = hz_edges[idx]
                right = hz_edges[idx + 1]
                clipped = torch.clamp(hz - left, min=0.0, max=max(right - left, 0.0))
                value = value + clipped * slope
            return value

        def importance_to_hz(importance: torch.Tensor) -> torch.Tensor:
            hz = torch.zeros_like(importance)
            remaining = importance.clone()
            for idx, slope in enumerate(slopes):
                seg_importance = importance_breaks[idx + 1] - importance_breaks[idx]
                taken = torch.clamp(remaining, min=0.0, max=float(seg_importance))
                hz = hz + taken / slope
                remaining = remaining - taken
            return hz

        total_importance = float(importance_breaks[-1].item())
        imp_edges = torch.linspace(0.0, total_importance, steps=n_bands + 1, dtype=torch.float32)
        hz_band_edges = importance_to_hz(imp_edges).clamp(0.0, nyquist)
        bin_edges = hz_band_edges / nyquist * max(n_freq - 1, 1)

        starts = torch.floor(bin_edges[:-1]).to(torch.long)
        ends = torch.ceil(bin_edges[1:]).to(torch.long) + 1
        ends = torch.maximum(ends, starts + 1)
        ends = torch.clamp(ends, max=n_freq)

        basis = torch.zeros(n_bands, n_freq, dtype=torch.float32)
        freq_positions = torch.arange(n_freq, dtype=torch.float32)
        for band_idx in range(n_bands):
            start = int(starts[band_idx].item())
            end = int(ends[band_idx].item())
            center = 0.5 * (start + end - 1)
            half_width = max(0.5 * (end - start), 1.0)
            tri = 1.0 - torch.abs(freq_positions[start:end] - center) / half_width
            basis[band_idx, start:end] = torch.clamp(tri, min=0.0)
        return starts, ends, basis

    @staticmethod
    def _build_basis(
        n_freq: int,
        n_bands: int,
        n_fft: int | None,
        sample_rate: int | None,
        band_config: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if band_config == "speech_lowfreq_narrow":
            return SoftBandSpec2d._build_speech_lowfreq_narrow_basis(
                n_freq=n_freq,
                n_bands=n_bands,
                sample_rate=sample_rate,
            )

        if n_fft is not None and sample_rate is not None:
            band_specs, freq_weights, _ = get_band_specs(band_config, n_fft, sample_rate, n_bands=n_bands)
            starts = torch.tensor([start for start, _ in band_specs], dtype=torch.long)
            ends = torch.tensor([end for _, end in band_specs], dtype=torch.long)
            basis = torch.zeros(n_bands, n_freq, dtype=torch.float32)
            for band_idx, ((start, end), weights) in enumerate(zip(band_specs, freq_weights)):
                end = min(end, n_freq)
                width = end - start
                if width <= 0:
                    continue
                w = weights.to(torch.float32)
                if w.numel() != width:
                    w = F.interpolate(
                        w.view(1, 1, -1),
                        size=width,
                        mode="linear",
                        align_corners=False,
                    ).view(-1)
                basis[band_idx, start:end] = w
            return starts, ends, basis

        edges = torch.linspace(0, n_freq, steps=n_bands + 1)
        starts = torch.floor(edges[:-1]).to(torch.long)
        ends = torch.ceil(edges[1:]).to(torch.long)
        ends = torch.maximum(ends, starts + 1)
        ends = torch.clamp(ends, max=n_freq)

        basis = torch.zeros(n_bands, n_freq, dtype=torch.float32)
        freq_positions = torch.arange(n_freq, dtype=torch.float32)
        for band_idx in range(n_bands):
            start = int(starts[band_idx].item())
            end = int(ends[band_idx].item())
            center = 0.5 * (start + end - 1)
            half_width = max(0.5 * (end - start), 1.0)
            tri = 1.0 - torch.abs(freq_positions[start:end] - center) / half_width
            basis[band_idx, start:end] = torch.clamp(tri, min=0.0)
        return starts, ends, basis

    def routing_bias(self) -> torch.Tensor:
        peak = self.basis.amax(dim=-1, keepdim=True).clamp_min(1e-6)
        return 2.0 * (self.basis / peak) - 1.0

    def expansion_basis(self) -> torch.Tensor:
        return self.basis / self.basis.sum(dim=1, keepdim=True).clamp_min(1e-6)


def normalize_band_scores(scores: torch.Tensor, mode: str = "softmax") -> torch.Tensor:
    if mode == "softmax":
        return torch.softmax(scores, dim=-1)
    if mode == "relu_l1":
        weights = F.relu(scores)
        return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    raise ValueError(f"Unsupported normalization mode: {mode}")


class SoftBandCompressor2d(nn.Module):
    """
    Compress F frequency bins into K latent band slots.

    Static band priors tell each slot where it should look by default, while
    the input-conditioned score map adapts the actual pooling weights.
    """

    def __init__(
        self,
        channels: int,
        band_spec: SoftBandSpec2d,
        kernel_size: tuple[int, int] = (3, 3),
        causal: bool = True,
        normalization: str = "softmax",
    ):
        super().__init__()
        Conv = CausalConv2d if causal else SameConv2d
        self.channels = channels
        self.band_spec = band_spec
        self.n_bands = band_spec.n_bands
        self.normalization = normalization
        self.causal = causal

        self.norm = RMSNorm2d(channels)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.dw = Conv(channels, channels, kernel_size=kernel_size, groups=channels, bias=True)
        self.score = nn.Conv2d(channels, self.n_bands, kernel_size=1, bias=True)
        self.value = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.score_scale = nn.Parameter(torch.tensor(1.0))
        self.prior_scale = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("routing_bias", band_spec.routing_bias())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, C, T, F_in)
        _runtime_assert(x.shape[-1] == self.band_spec.n_freq, f"{x.shape} vs {self.band_spec.n_freq}")
        # h: (B, C, T, F_in)
        h = self.dw(self.pw(self.norm(x)))
        h = F.silu(h)
        # values: (B, C, T, F_in)
        values = self.value(h)
        # scores / weights: (B, K_out, T, F_in)
        scores = self.score(h) * self.score_scale + self.routing_bias * self.prior_scale
        weights = normalize_band_scores(scores, mode=self.normalization)

        batch, channels, n_frames, n_freq = values.shape
        # values_btfc:   (B*T, F_in, C)
        values_btfc = values.permute(0, 2, 3, 1).reshape(batch * n_frames, n_freq, channels)
        # weights_btkf:  (B*T, K_out, F_in)
        weights_btkf = weights.permute(0, 2, 1, 3).reshape(batch * n_frames, self.n_bands, n_freq)
        # latent_btkc:   (B*T, K_out, C)
        latent_btkc = torch.bmm(weights_btkf, values_btfc)
        # latent: (B, C, T, K_out)
        latent = latent_btkc.reshape(batch, n_frames, self.n_bands, channels).permute(0, 3, 1, 2)
        return latent, weights

    def stream_context_frames(self) -> int:
        if isinstance(self.dw, CausalConv2d):
            return self.dw.stream_context_frames()
        return 0

    def init_stream_state(
        self,
        batch_size: int,
        *,
        freq_bins: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if not isinstance(self.dw, CausalConv2d):
            raise RuntimeError("Streaming state is only supported when causal=True.")
        return self.dw.init_stream_state(batch_size, freq_bins=freq_bins, device=device, dtype=dtype)

    def forward_stream(self, x: torch.Tensor, state: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(self.dw, CausalConv2d):
            raise RuntimeError("forward_stream is only supported when causal=True.")

        # x: (B, C, T_chunk, F_in)
        _runtime_assert(x.shape[-1] == self.band_spec.n_freq, f"{x.shape} vs {self.band_spec.n_freq}")
        # h: (B, C, T_chunk, F_in), new_state: (B, C, ctx, F_in)
        h = self.norm(x)
        h = self.pw(h)
        h, new_state = self.dw.forward_stream(h, state)
        h = F.silu(h)
        # values: (B, C, T_chunk, F_in)
        values = self.value(h)
        # weights: (B, K_out, T_chunk, F_in)
        scores = self.score(h) * self.score_scale + self.routing_bias * self.prior_scale
        weights = normalize_band_scores(scores, mode=self.normalization)
        batch, channels, n_frames, n_freq = values.shape
        # values_btfc:   (B*T_chunk, F_in, C)
        values_btfc = values.permute(0, 2, 3, 1).reshape(batch * n_frames, n_freq, channels)
        # weights_btkf:  (B*T_chunk, K_out, F_in)
        weights_btkf = weights.permute(0, 2, 1, 3).reshape(batch * n_frames, self.n_bands, n_freq)
        # latent: (B, C, T_chunk, K_out)
        latent_btkc = torch.bmm(weights_btkf, values_btfc)
        latent = latent_btkc.reshape(batch, n_frames, self.n_bands, channels).permute(0, 3, 1, 2)
        return latent, new_state


class SoftBandExpander2d(nn.Module):
    """
    Expand K latent band slots back to F bins.

    The static basis preserves band locality and a light dynamic gate lets the
    model adjust how much each slot contributes per frame.
    """

    def __init__(self, channels: int, band_spec: SoftBandSpec2d):
        super().__init__()
        self.channels = channels
        self.band_spec = band_spec
        self.n_bands = band_spec.n_bands
        self.n_freq = band_spec.n_freq

        self.pre = nn.Sequential(
            RMSNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.SiLU(),
        )
        self.band_gain = nn.Conv2d(channels, 1, kernel_size=1, bias=True)
        self.gain_scale = nn.Parameter(torch.tensor(1.0))
        self.basis_scale = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("expansion_basis", band_spec.expansion_basis())

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, C, T, K_in)
        _runtime_assert(z.shape[-1] == self.n_bands, f"{z.shape} vs {self.n_bands}")
        # h: (B, C, T, K_in)
        h = self.pre(z)
        batch, channels, n_frames, n_bands = h.shape

        # gains: (B, K_in, T, 1)
        gains = 1.0 + torch.sigmoid(self.band_gain(h)) * self.gain_scale
        gains = gains.permute(0, 3, 2, 1)  # (B, K, T, 1)

        # coeff: (B, K_in, T, F_out)
        coeff = self.expansion_basis * (self.basis_scale + gains)
        coeff = coeff / coeff.sum(dim=1, keepdim=True).clamp_min(1e-6)

        # h_btck: (B*T, C, K_in)
        h_btck = h.permute(0, 2, 1, 3).reshape(batch * n_frames, channels, n_bands)
        # coeff_btkf: (B*T, K_in, F_out)
        coeff_btkf = coeff.permute(0, 2, 1, 3).reshape(batch * n_frames, n_bands, self.n_freq)
        # expanded: (B, C, T, F_out)
        expanded_btcf = torch.bmm(h_btck, coeff_btkf)
        return expanded_btcf.reshape(batch, n_frames, channels, self.n_freq).permute(0, 2, 1, 3)


class OnlineSoftBandSFC2D(nn.Module):
    """
    2D-only separator core with soft band routing.

    input (B, 2*M, T, F)
      -> in_proj
      -> soft compressor: (B, D, T, F) -> (B, D, T, K)
      -> causal/non-causal conv separator on K bands
      -> soft expander: (B, D, T, K) -> (B, D, T, F)
      -> out_proj
      -> packed complex masking or mapping
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
    ):
        super().__init__()
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
        self.separator = nn.ModuleList(
            [OnlineConvBlock(d_model, expansion=2, kernel_size=kernel_size, causal=causal) for _ in range(n_layers)]
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
        states = self.init_stream_state(batch_size=batch_size, device=self.out_proj.weight.device, dtype=self.out_proj.weight.dtype)
        return sum(int(s.numel()) for s in states)

    def input_history_numel(self, batch_size: int = 1) -> int:
        return batch_size * 2 * self.n_chan * self.stream_context_frames() * self.n_freq

    def state_size_bytes(self, *, batch_size: int = 1, dtype: torch.dtype = torch.float16, mode: str = "layer_cache") -> int:
        element_size = torch.tensor([], dtype=dtype).element_size()
        if mode == "layer_cache":
            return self.layer_cache_numel(batch_size=batch_size) * element_size
        if mode == "input_history":
            return self.input_history_numel(batch_size=batch_size) * element_size
        raise ValueError(f"Unsupported state mode: {mode}")


class OnlineSoftBandSFCModel(nn.Module):
    """
    Torch wrapper that matches the existing complex STFT model contract.
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
    ):
        super().__init__()
        self.core = OnlineSoftBandSFC2D(
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


def build_online_soft_band_sfc_system(
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
    core = OnlineSoftBandSFC2D(
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
