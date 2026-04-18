"""
Online / realtime SFC variant with explicit query side-path.

This family is intentionally closer to the paper's SFC spirit than the plain
online soft-band model while remaining edge/NPU friendly:
- compression is still input-adaptive and band-aware,
- the compressor emits both latent tokens and an explicit query tensor,
- the decoder uses that query tensor as a side-path during reconstruction,
- the whole graph stays within Conv2d / bmm / basic tensor ops.
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
from spectral_feature_compression.core.model.online_soft_band_sfc_2d import SoftBandSpec2d, normalize_band_scores


class SoftBandQueryCompressor2d(nn.Module):
    """
    Input-adaptive F -> K compression with an explicit query side-output.

    The returned `latent` tensor is the compressed representation processed by
    the separator, while `query_tokens` preserve a second adaptive summary that
    is consumed later by the decoder.
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

        self.norm = RMSNorm2d(channels)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.dw = Conv(channels, channels, kernel_size=kernel_size, groups=channels, bias=True)
        self.score = nn.Conv2d(channels, self.n_bands, kernel_size=1, bias=True)
        self.value = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.query = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.score_scale = nn.Parameter(torch.tensor(1.0))
        self.prior_scale = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("routing_bias", band_spec.routing_bias())

    def _pool_tokens(self, values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        batch, channels, n_frames, n_freq = values.shape
        values_btfc = values.permute(0, 2, 3, 1).reshape(batch * n_frames, n_freq, channels)
        weights_btkf = weights.permute(0, 2, 1, 3).reshape(batch * n_frames, self.n_bands, n_freq)
        pooled_btkc = torch.bmm(weights_btkf, values_btfc)
        return pooled_btkc.reshape(batch, n_frames, self.n_bands, channels).permute(0, 3, 1, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _runtime_assert(x.shape[-1] == self.band_spec.n_freq, f"{x.shape} vs {self.band_spec.n_freq}")
        h = self.dw(self.pw(self.norm(x)))
        h = F.silu(h)

        scores = self.score(h) * self.score_scale + self.routing_bias * self.prior_scale
        weights = normalize_band_scores(scores, mode=self.normalization)
        latent = self._pool_tokens(self.value(h), weights)
        query_tokens = self._pool_tokens(self.query(h), weights)
        return latent, query_tokens

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

    def forward_stream(self, x: torch.Tensor, state: torch.Tensor | None) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        if not isinstance(self.dw, CausalConv2d):
            raise RuntimeError("forward_stream is only supported when causal=True.")

        _runtime_assert(x.shape[-1] == self.band_spec.n_freq, f"{x.shape} vs {self.band_spec.n_freq}")
        h = self.norm(x)
        h = self.pw(h)
        h, new_state = self.dw.forward_stream(h, state)
        h = F.silu(h)

        scores = self.score(h) * self.score_scale + self.routing_bias * self.prior_scale
        weights = normalize_band_scores(scores, mode=self.normalization)
        latent = self._pool_tokens(self.value(h), weights)
        query_tokens = self._pool_tokens(self.query(h), weights)
        return (latent, query_tokens), new_state


class SoftBandQueryExpander2d(nn.Module):
    """
    Decoder that reconstructs F bins from latent tokens plus an explicit query
    side-path, approximating the encoder/decoder query contract of SFC.
    """

    def __init__(self, channels: int, band_spec: SoftBandSpec2d):
        super().__init__()
        self.channels = channels
        self.band_spec = band_spec
        self.n_bands = band_spec.n_bands
        self.n_freq = band_spec.n_freq

        self.latent_pre = nn.Sequential(
            RMSNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.SiLU(),
        )
        self.query_pre = nn.Sequential(
            RMSNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.SiLU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=1, bias=True),
            nn.SiLU(),
        )
        self.band_gain = nn.Conv2d(channels, 1, kernel_size=1, bias=True)
        self.query_skip_scale = nn.Parameter(torch.tensor(1.0))
        self.gain_scale = nn.Parameter(torch.tensor(1.0))
        self.basis_scale = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("expansion_basis", band_spec.expansion_basis())

    def forward(self, latent: torch.Tensor, query_tokens: torch.Tensor) -> torch.Tensor:
        _runtime_assert(latent.shape[-1] == self.n_bands, f"{latent.shape} vs {self.n_bands}")

        latent_h = self.latent_pre(latent)
        query_h = self.query_pre(query_tokens)
        fused = self.fuse(torch.cat([latent_h, query_h], dim=1))

        gains = 1.0 + torch.sigmoid(self.band_gain(fused)) * self.gain_scale
        gains = gains.permute(0, 3, 2, 1)  # (B, K, T, 1)

        coeff = self.expansion_basis * (self.basis_scale + gains)
        coeff = coeff / coeff.sum(dim=1, keepdim=True).clamp_min(1e-6)

        tokens = latent_h + query_h * self.query_skip_scale
        batch, channels, n_frames, n_bands = tokens.shape
        tokens_btck = tokens.permute(0, 2, 1, 3).reshape(batch * n_frames, channels, n_bands)
        coeff_btkf = coeff.permute(0, 2, 1, 3).reshape(batch * n_frames, n_bands, self.n_freq)
        expanded_btcf = torch.bmm(tokens_btck, coeff_btkf)
        return expanded_btcf.reshape(batch, n_frames, channels, self.n_freq).permute(0, 2, 1, 3)


class OnlineSoftBandQuerySFC2D(nn.Module):
    """
    Online soft-band SFC with an explicit query side-path from compressor to
    decoder. This is the closest NPU-friendly approximation in the repo to the
    paper's encoder/decoder query contract.
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
        self.compressor = SoftBandQueryCompressor2d(
            channels=d_model,
            band_spec=band_spec,
            kernel_size=kernel_size,
            causal=causal,
            normalization=routing_normalization,
        )
        self.separator = nn.ModuleList(
            [OnlineConvBlock(d_model, expansion=2, kernel_size=kernel_size, causal=causal) for _ in range(n_layers)]
        )
        self.expander = SoftBandQueryExpander2d(channels=d_model, band_spec=band_spec)
        self.out_proj = nn.Conv2d(d_model, out_ch, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _runtime_assert(x.ndim == 4, f"Expected 4D input (B,C,T,F), got {x.shape}")
        _runtime_assert(x.shape[-1] == self.n_freq, f"{x.shape} vs {self.n_freq}")

        h = self.in_proj(x)
        z, q = self.compressor(h)
        for block in self.separator:
            z = block(z)
        h = self.expander(z, q)
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
        (z, q), new_comp_state = self.compressor.forward_stream(h, comp_state)
        new_sep_states = []
        for block, block_state in zip(self.separator, sep_state):
            z, block_state = block.forward_stream(z, block_state)
            new_sep_states.append(block_state)
        h = self.expander(z, q)
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


class OnlineSoftBandQuerySFCModel(nn.Module):
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
        self.core = OnlineSoftBandQuerySFC2D(
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


def build_online_soft_band_query_sfc_system(
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
    core = OnlineSoftBandQuerySFC2D(
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
