"""
Handcrafted strong NPU student for the source-aware MelBand RoFormer teacher.

This is intentionally separate from the conservative student implementation.  It
keeps the teacher's useful separation priors, but every separator stage here is
custom-built from NPU-compatible primitives instead of selecting the repo's
existing separator blocks or hiding capacity in pooled mixers:

* custom adaptive mel router with learned mixture/query token side paths;
* custom gated temporal-band encoder blocks with local band mixing and
  zero-state token FFNs;
* explicit learned source seeding and repeated source/other/mixture competition;
* custom query-conditioned mel expander and full-band source-shared mask head;
* custom low-rank full-band mask correction and 4D mixture consistency.

The deployment recipe keeps temporal state bounded by making the source decoder
stateless in time while spending most extra capacity on per-token/source/frequency
1x1 and band-local transformations that add parameters without recurrent cache.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectral_feature_compression.core.model.adaptive_mel_sfc_2d import AdaptiveMelBandSpec2d
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
    pack_complex_stft_as_2d,
    unpack_2d_to_complex_stft,
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


def _normalize_dilation_schedule(n_layers: int, cycle: Sequence[int] | None) -> tuple[int, ...]:
    if n_layers < 0:
        raise ValueError(f"n_layers must be non-negative, got {n_layers}")
    if n_layers == 0:
        return ()
    cycle = (1, 2) if cycle is None else tuple(int(v) for v in cycle)
    if len(cycle) == 0:
        raise ValueError("dilation cycle must not be empty")
    if any(v <= 0 for v in cycle):
        raise ValueError(f"dilation values must be positive, got {cycle}")
    return tuple(cycle[idx % len(cycle)] for idx in range(n_layers))


def _packed_complex_features(
    x: torch.Tensor,
    *,
    n_chan: int,
    include_magnitude: bool,
    include_logmag: bool,
) -> torch.Tensor:
    _runtime_assert(x.shape[1] == 2 * n_chan, f"Expected {2 * n_chan} packed channels, got {x.shape}")
    if not include_magnitude and not include_logmag:
        return x

    feats: list[torch.Tensor] = [x]
    mags: list[torch.Tensor] = []
    ri_channels = torch.split(x, 1, dim=1)
    for chan_idx in range(n_chan):
        real = ri_channels[2 * chan_idx]
        imag = ri_channels[2 * chan_idx + 1]
        mags.append(torch.sqrt(real * real + imag * imag + 1.0e-8))
    mag = torch.cat(mags, dim=1)
    if include_magnitude:
        feats.append(mag)
    if include_logmag:
        feats.append(torch.log1p(mag))
    return torch.cat(feats, dim=1)


def _source_chunks(x: torch.Tensor, *, n_src: int, channels: int) -> list[torch.Tensor]:
    _runtime_assert(x.shape[1] == n_src * channels, f"Expected {n_src * channels} channels, got {x.shape}")
    return list(torch.split(x, channels, dim=1))


def _sum_chunks(chunks: list[torch.Tensor]) -> torch.Tensor:
    total = chunks[0]
    for chunk in chunks[1:]:
        total = total + chunk
    return total


def _apply_packed_complex_mask_no_repeat(
    *,
    x: torch.Tensor,
    y: torch.Tensor,
    n_src: int,
    n_chan: int,
) -> torch.Tensor:
    _runtime_assert(x.shape[1] == 2 * n_chan, f"{x.shape[1]} vs {2 * n_chan}")
    _runtime_assert(y.shape[1] == 2 * n_src * n_chan, f"{y.shape[1]} vs {2 * n_src * n_chan}")
    outputs: list[torch.Tensor] = []
    mixture_channels = torch.split(x, 1, dim=1)
    mask_channels = torch.split(y, 1, dim=1)
    for src_idx in range(n_src):
        for chan_idx in range(n_chan):
            in_r = mixture_channels[2 * chan_idx]
            in_i = mixture_channels[2 * chan_idx + 1]
            mask_base = 2 * (src_idx * n_chan + chan_idx)
            mask_r = mask_channels[mask_base]
            mask_i = mask_channels[mask_base + 1]
            outputs.append(in_r * mask_r - in_i * mask_i)
            outputs.append(in_r * mask_i + in_i * mask_r)
    return torch.cat(outputs, dim=1)


class StrongCausalOrFrameConv2d(nn.Module):
    """Causal time depthwise/regular conv that becomes stateless for kt=1."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: tuple[int, int],
        dilation: tuple[int, int] = (1, 1),
        groups: int = 1,
        causal: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        kt, kf = kernel_size
        dt, df = dilation
        self.causal = bool(causal)
        self.is_stateless = kt == 1 or not causal
        if causal and kt > 1:
            self.conv = CausalConv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        else:
            pad_t = 0 if kt == 1 else ((kt - 1) * dt) // 2
            pad_f = ((kf - 1) * df) // 2
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=(pad_t, pad_f),
                groups=groups,
                bias=bias,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

    def stream_context_frames(self) -> int:
        if isinstance(self.conv, CausalConv2d):
            return self.conv.stream_context_frames()
        return 0

    def init_stream_state(self, batch_size: int, *, freq_bins: int, device=None, dtype=None) -> torch.Tensor:
        if isinstance(self.conv, CausalConv2d):
            return self.conv.init_stream_state(batch_size, freq_bins=freq_bins, device=device, dtype=dtype)
        return torch.zeros(batch_size, 0, 0, freq_bins, device=device, dtype=dtype)

    def forward_stream(self, x: torch.Tensor, state: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(self.conv, CausalConv2d):
            return self.conv.forward_stream(x, state)
        return self.conv(x), self.init_stream_state(x.shape[0], freq_bins=x.shape[-1], device=x.device, dtype=x.dtype)


class StrongTokenFFN2d(nn.Module):
    """Zero-state gated token/channel expansion applied at each time-frequency token."""

    def __init__(self, channels: int, hidden_channels: int, *, residual_scale: float = 0.1):
        super().__init__()
        if hidden_channels <= 0:
            raise ValueError(f"hidden_channels must be positive, got {hidden_channels}")
        self.hidden_channels = int(hidden_channels)
        self.norm = RMSNorm2d(channels)
        self.expand = nn.Conv2d(channels, 2 * hidden_channels, kernel_size=1, bias=True)
        self.project = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True)
        self.scale = nn.Parameter(torch.tensor(float(residual_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = torch.split(self.expand(self.norm(x)), self.hidden_channels, dim=1)
        return x + self.project(value * torch.sigmoid(gate)) * self.scale


class StrongTemporalBandBlock2d(nn.Module):
    """Custom gated time/band/local-channel block for compressed or full bands."""

    def __init__(
        self,
        channels: int,
        *,
        expansion: int = 2,
        token_ffn_mult: int = 4,
        time_kernel_size: int = 3,
        band_kernel_size: int = 5,
        time_dilation: int = 1,
        causal: bool = True,
        residual_scale: float = 0.1,
    ):
        super().__init__()
        if band_kernel_size % 2 == 0:
            raise ValueError(f"band_kernel_size must be odd, got {band_kernel_size}")
        hidden = int(channels) * int(expansion)
        ffn_hidden = int(channels) * int(token_ffn_mult)
        self.channels = int(channels)
        self.hidden = int(hidden)
        self.causal = bool(causal)
        self.norm = RMSNorm2d(channels)
        self.in_proj = nn.Conv2d(channels, 2 * hidden, kernel_size=1, bias=True)
        self.time_dw = StrongCausalOrFrameConv2d(
            hidden,
            hidden,
            kernel_size=(time_kernel_size, 1),
            dilation=(time_dilation, 1),
            groups=hidden,
            causal=causal,
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
        wide_kernel = min(max(band_kernel_size + 2, 3), 7)
        if wide_kernel % 2 == 0:
            wide_kernel += 1
        self.wide_band_dw = nn.Conv2d(
            hidden,
            hidden,
            kernel_size=(1, wide_kernel),
            padding=(0, wide_kernel // 2),
            groups=hidden,
            bias=True,
        )
        self.band_fuse = nn.Conv2d(2 * hidden, hidden, kernel_size=1, bias=True)
        self.out_proj = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
        self.mix_scale = nn.Parameter(torch.tensor(float(residual_scale)))
        self.ffn = StrongTokenFFN2d(channels, ffn_hidden, residual_scale=residual_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = torch.split(self.in_proj(self.norm(x)), self.hidden, dim=1)
        y = value * torch.sigmoid(gate)
        y = F.silu(self.time_dw(y))
        band = F.silu(self.band_dw(y))
        wide = F.silu(self.wide_band_dw(y))
        y = self.band_fuse(torch.cat([band, wide], dim=1))
        x = x + self.out_proj(y) * self.mix_scale
        return self.ffn(x)

    def stream_context_frames(self) -> int:
        return self.time_dw.stream_context_frames()

    def init_stream_state(
        self, batch_size: int = 1, *, freq_bins: int, device=None, dtype=None
    ) -> tuple[torch.Tensor, ...]:
        if not self.causal:
            raise RuntimeError("Streaming state is only supported when causal=True.")
        if self.stream_context_frames() == 0:
            return ()
        return (self.time_dw.init_stream_state(batch_size, freq_bins=freq_bins, device=device, dtype=dtype),)

    def forward_stream(
        self,
        x: torch.Tensor,
        states: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if not self.causal:
            raise RuntimeError("forward_stream is only supported when causal=True.")
        expected = 1 if self.stream_context_frames() > 0 else 0
        _runtime_assert(len(states) == expected, f"Expected {expected} block states, got {len(states)}")
        value, gate = torch.split(self.in_proj(self.norm(x)), self.hidden, dim=1)
        y = value * torch.sigmoid(gate)
        if expected == 0:
            y, new_state = self.time_dw.forward_stream(y, None)
            new_states: tuple[torch.Tensor, ...] = ()
        else:
            y, new_state = self.time_dw.forward_stream(y, states[0])
            new_states = (new_state,)
        y = F.silu(y)
        band = F.silu(self.band_dw(y))
        wide = F.silu(self.wide_band_dw(y))
        y = self.band_fuse(torch.cat([band, wide], dim=1))
        x = x + self.out_proj(y) * self.mix_scale
        return self.ffn(x), new_states


class StrongAdaptiveMelRouter2d(nn.Module):
    """Custom input-adaptive mel-band router with query side path."""

    def __init__(
        self,
        *,
        channels: int,
        band_spec: AdaptiveMelBandSpec2d,
        kernel_size: tuple[int, int] = (1, 3),
        causal: bool = True,
        normalization: str = "softmax",
    ):
        super().__init__()
        self.channels = int(channels)
        self.band_spec = band_spec
        self.n_bands = int(band_spec.n_bands)
        self.normalization = normalization
        self.norm = RMSNorm2d(channels)
        self.pre = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.local = StrongCausalOrFrameConv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            groups=channels,
            causal=causal,
            bias=True,
        )
        self.score = nn.Conv2d(channels, self.n_bands, kernel_size=1, bias=True)
        self.value = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.query = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.detail = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.token_fuse = nn.Conv2d(2 * channels, channels, kernel_size=1, bias=True)
        self.score_scale = nn.Parameter(torch.tensor(1.0))
        self.prior_scale = nn.Parameter(torch.tensor(1.0))
        self.detail_scale = nn.Parameter(torch.tensor(0.1))
        self.register_buffer("routing_bias", band_spec.routing_bias())

    def _normalize(self, scores: torch.Tensor) -> torch.Tensor:
        if self.normalization == "softmax":
            return torch.softmax(scores, dim=-1)
        if self.normalization == "relu_l1":
            weights = F.relu(scores)
            return weights / (weights.sum(dim=-1, keepdim=True) + 1.0e-6)
        raise ValueError(f"Unsupported routing normalization: {self.normalization}")

    def _pool_tokens(self, values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        values_btfc = values.permute(0, 2, 3, 1)
        weights_btkf = weights.permute(0, 2, 1, 3)
        pooled_btkc = torch.matmul(weights_btkf, values_btfc)
        return pooled_btkc.permute(0, 3, 1, 2)

    def _route(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.score(h) * self.score_scale + self.routing_bias * self.prior_scale
        weights = self._normalize(scores)
        value_tokens = self._pool_tokens(self.value(h), weights)
        detail_tokens = self._pool_tokens(self.detail(h), weights)
        latent = self.token_fuse(torch.cat([value_tokens, detail_tokens], dim=1))
        query = self._pool_tokens(self.query(h), weights)
        return latent + detail_tokens * self.detail_scale, query

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _runtime_assert(x.shape[-1] == self.band_spec.n_freq, f"{x.shape} vs {self.band_spec.n_freq}")
        h = F.silu(self.local(self.pre(self.norm(x))))
        return self._route(h)

    def stream_context_frames(self) -> int:
        return self.local.stream_context_frames()

    def init_stream_state(
        self, batch_size: int = 1, *, freq_bins: int, device=None, dtype=None
    ) -> tuple[torch.Tensor, ...]:
        if self.stream_context_frames() == 0:
            return ()
        return (self.local.init_stream_state(batch_size, freq_bins=freq_bins, device=device, dtype=dtype),)

    def forward_stream(
        self,
        x: torch.Tensor,
        states: tuple[torch.Tensor, ...],
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, ...]]:
        expected = 1 if self.stream_context_frames() > 0 else 0
        _runtime_assert(len(states) == expected, f"Expected {expected} router states, got {len(states)}")
        h = self.pre(self.norm(x))
        if expected == 0:
            h, new_state = self.local.forward_stream(h, None)
            new_states: tuple[torch.Tensor, ...] = ()
        else:
            h, new_state = self.local.forward_stream(h, states[0])
            new_states = (new_state,)
        return self._route(F.silu(h)), new_states


class StrongMelBandExpander2d(nn.Module):
    """Custom query-conditioned K->F mel expander."""

    def __init__(self, *, channels: int, band_spec: AdaptiveMelBandSpec2d, hidden_channels: int | None = None):
        super().__init__()
        self.channels = int(channels)
        self.n_bands = int(band_spec.n_bands)
        self.n_freq = int(band_spec.n_freq)
        hidden_channels = channels if hidden_channels is None else int(hidden_channels)
        self.latent_pre = nn.Sequential(RMSNorm2d(channels), nn.Conv2d(channels, channels, kernel_size=1), nn.SiLU())
        self.query_pre = nn.Sequential(RMSNorm2d(channels), nn.Conv2d(channels, channels, kernel_size=1), nn.SiLU())
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * channels, 2 * hidden_channels, kernel_size=1, bias=True),
            nn.GLU(dim=1),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True),
            nn.SiLU(),
        )
        self.band_gain = nn.Conv2d(channels, 1, kernel_size=1, bias=True)
        self.query_scale = nn.Parameter(torch.tensor(0.5))
        self.gain_scale = nn.Parameter(torch.tensor(1.0))
        self.basis_scale = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("expansion_basis", band_spec.expansion_basis())

    def forward(self, latent: torch.Tensor, query_tokens: torch.Tensor) -> torch.Tensor:
        _runtime_assert(latent.shape[-1] == self.n_bands, f"{latent.shape} vs {self.n_bands}")
        latent_h = self.latent_pre(latent)
        query_h = self.query_pre(query_tokens)
        tokens = self.fuse(torch.cat([latent_h, query_h], dim=1)) + query_h * self.query_scale
        gains = 1.0 + torch.sigmoid(self.band_gain(tokens)) * self.gain_scale
        gains = gains.permute(0, 3, 2, 1)
        coeff = self.expansion_basis * (self.basis_scale + gains)
        coeff_tr = coeff.transpose(1, -1)
        coeff = coeff / (coeff_tr.sum(dim=-1, keepdim=True).transpose(1, -1) + 1.0e-6)

        tokens_btck = tokens.permute(0, 2, 1, 3)
        coeff_btkf = coeff.permute(0, 2, 1, 3)
        expanded_btcf = torch.matmul(tokens_btck, coeff_btkf)
        return expanded_btcf.permute(0, 2, 1, 3)


class StrongSourceSeed2d(nn.Module):
    """Learned source-specific seeding from shared mixture tokens."""

    def __init__(self, *, channels: int, n_src: int, hidden_channels: int):
        super().__init__()
        self.channels = int(channels)
        self.n_src = int(n_src)
        self.seed = nn.Sequential(
            RMSNorm2d(channels),
            nn.Conv2d(channels, 2 * hidden_channels, kernel_size=1, bias=True),
            nn.GLU(dim=1),
            nn.Conv2d(hidden_channels, n_src * channels, kernel_size=1, bias=True),
        )
        self.shared = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.source_bias = nn.Parameter(torch.zeros(1, n_src * channels, 1, 1))
        self.seed_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, mixture_tokens: torch.Tensor) -> torch.Tensor:
        source_delta = self.seed(mixture_tokens) * self.seed_scale + self.source_bias
        shared = self.shared(mixture_tokens)
        chunks = _source_chunks(source_delta, n_src=self.n_src, channels=self.channels)
        return torch.cat([chunk + shared for chunk in chunks], dim=1)


class StrongSourceCompetitionBlock2d(nn.Module):
    """Stateless-in-time source competition over source/other/mixture context."""

    def __init__(
        self,
        *,
        channels: int,
        n_src: int,
        local_expansion: int = 2,
        local_ffn_mult: int = 4,
        fusion_hidden_channels: int = 192,
        band_kernel_size: int = 5,
    ):
        super().__init__()
        self.channels = int(channels)
        self.n_src = int(n_src)
        self.local = StrongTemporalBandBlock2d(
            channels,
            expansion=local_expansion,
            token_ffn_mult=local_ffn_mult,
            time_kernel_size=1,
            band_kernel_size=band_kernel_size,
            time_dilation=1,
            causal=True,
            residual_scale=0.1,
        )
        in_channels = 5 * channels
        self.fusion_hidden_channels = int(fusion_hidden_channels)
        self.competition_norm = RMSNorm2d(in_channels)
        self.competition_in = nn.Conv2d(in_channels, 2 * fusion_hidden_channels, kernel_size=1, bias=True)
        self.competition_mid = nn.Conv2d(fusion_hidden_channels, fusion_hidden_channels, kernel_size=1, bias=True)
        self.competition_out = nn.Conv2d(fusion_hidden_channels, channels, kernel_size=1, bias=True)
        self.competition_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, source_tokens: torch.Tensor, mixture_tokens: torch.Tensor) -> torch.Tensor:
        chunks = _source_chunks(source_tokens, n_src=self.n_src, channels=self.channels)
        chunks = [self.local(chunk) for chunk in chunks]
        source_mean = _sum_chunks(chunks) / float(self.n_src)
        fused = []
        for chunk in chunks:
            if self.n_src > 1:
                other_mean = (source_mean * float(self.n_src) - chunk) / float(self.n_src - 1)
            else:
                other_mean = source_mean
            comp = torch.cat([chunk, mixture_tokens, other_mean, chunk - other_mean, mixture_tokens - chunk], dim=1)
            value, gate = torch.split(
                self.competition_in(self.competition_norm(comp)),
                self.fusion_hidden_channels,
                dim=1,
            )
            y = F.silu(self.competition_mid(value * torch.sigmoid(gate)))
            fused.append(chunk + self.competition_out(y) * self.competition_scale)
        return torch.cat(fused, dim=1)


class StrongSourceDecoder2d(nn.Module):
    """Repeated custom source competition blocks."""

    def __init__(
        self,
        *,
        channels: int,
        n_src: int,
        n_layers: int,
        local_expansion: int,
        local_ffn_mult: int,
        fusion_hidden_channels: int,
        band_kernel_size: int,
    ):
        super().__init__()
        self.channels = int(channels)
        self.n_src = int(n_src)
        self.blocks = nn.ModuleList(
            [
                StrongSourceCompetitionBlock2d(
                    channels=channels,
                    n_src=n_src,
                    local_expansion=local_expansion,
                    local_ffn_mult=local_ffn_mult,
                    fusion_hidden_channels=fusion_hidden_channels,
                    band_kernel_size=band_kernel_size,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, source_tokens: torch.Tensor, mixture_tokens: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            source_tokens = block(source_tokens, mixture_tokens)
        return source_tokens

    def init_stream_state(self, batch_size: int = 1, *, device=None, dtype=None) -> tuple[torch.Tensor, ...]:
        del batch_size, device, dtype
        return ()

    def forward_stream(
        self,
        source_tokens: torch.Tensor,
        mixture_tokens: torch.Tensor,
        states: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        _runtime_assert(len(states) == 0, f"Expected no source decoder states, got {len(states)}")
        return self.forward(source_tokens, mixture_tokens), ()


class StrongSourceMaskHead2d(nn.Module):
    """Source-shared custom expander + full-band local mask head."""

    def __init__(
        self,
        *,
        channels: int,
        n_src: int,
        n_chan: int,
        band_spec: AdaptiveMelBandSpec2d,
        expander_hidden_channels: int,
        mask_hidden_channels: int,
        fullband_kernel_size: int = 5,
    ):
        super().__init__()
        self.channels = int(channels)
        self.n_src = int(n_src)
        self.n_chan = int(n_chan)
        self.expander = StrongMelBandExpander2d(
            channels=channels,
            band_spec=band_spec,
            hidden_channels=expander_hidden_channels,
        )
        self.fullband_local = StrongTemporalBandBlock2d(
            channels,
            expansion=1,
            token_ffn_mult=max(1, mask_hidden_channels // channels),
            time_kernel_size=1,
            band_kernel_size=fullband_kernel_size,
            causal=True,
            residual_scale=0.1,
        )
        self.mask = nn.Sequential(
            RMSNorm2d(channels),
            nn.Conv2d(channels, 2 * mask_hidden_channels, kernel_size=1, bias=True),
            nn.GLU(dim=1),
            nn.Conv2d(mask_hidden_channels, 2 * n_chan, kernel_size=1, bias=True),
        )

    def forward(self, source_tokens: torch.Tensor, query_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        chunks = _source_chunks(source_tokens, n_src=self.n_src, channels=self.channels)
        masks = []
        contexts = []
        for chunk in chunks:
            fullband = self.fullband_local(self.expander(chunk, query_tokens))
            contexts.append(fullband)
            masks.append(self.mask(fullband))
        return torch.cat(masks, dim=1), _sum_chunks(contexts) / float(self.n_src)


class StrongMaskCorrectionHead2d(nn.Module):
    """Custom full-band low-rank correction head in mask domain."""

    def __init__(
        self,
        *,
        context_channels: int,
        correction_channels: int,
        n_freq: int,
        n_src: int,
        n_chan: int,
        n_layers: int = 1,
        kernel_size: tuple[int, int] = (3, 5),
        causal: bool = True,
    ):
        super().__init__()
        self.context_channels = int(context_channels)
        self.correction_channels = int(correction_channels)
        self.n_freq = int(n_freq)
        self.n_src = int(n_src)
        self.n_chan = int(n_chan)
        self.causal = bool(causal)
        out_channels = 2 * n_src * n_chan
        self.mix_proj = nn.Conv2d(2 * n_chan, correction_channels, kernel_size=1, bias=True)
        self.mask_proj = nn.Conv2d(out_channels, correction_channels, kernel_size=1, bias=True)
        self.context_proj = nn.Conv2d(context_channels, correction_channels, kernel_size=1, bias=True)
        self.fuse = nn.Sequential(
            RMSNorm2d(3 * correction_channels),
            nn.Conv2d(3 * correction_channels, 2 * correction_channels, kernel_size=1, bias=True),
            nn.GLU(dim=1),
        )
        self.blocks = nn.ModuleList(
            [
                StrongTemporalBandBlock2d(
                    correction_channels,
                    expansion=2,
                    token_ffn_mult=3,
                    time_kernel_size=kernel_size[0],
                    band_kernel_size=kernel_size[1],
                    time_dilation=1,
                    causal=causal,
                    residual_scale=0.1,
                )
                for _ in range(n_layers)
            ]
        )
        self.out = nn.Conv2d(correction_channels, out_channels, kernel_size=1, bias=True)
        self.scale = nn.Parameter(torch.tensor(0.05))

    def forward(self, mixture: torch.Tensor, masks: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        h = self.fuse(torch.cat([self.mix_proj(mixture), self.mask_proj(masks), self.context_proj(context)], dim=1))
        for block in self.blocks:
            h = block(h)
        return self.out(h) * self.scale

    def stream_context_frames(self) -> int:
        return sum(block.stream_context_frames() for block in self.blocks)

    def state_tensor_count(self) -> int:
        return sum(1 if block.stream_context_frames() > 0 else 0 for block in self.blocks)

    def init_stream_state(self, batch_size: int = 1, *, device=None, dtype=None) -> tuple[torch.Tensor, ...]:
        if not self.causal:
            raise RuntimeError("Streaming state is only supported when causal=True.")
        states: list[torch.Tensor] = []
        for block in self.blocks:
            states.extend(block.init_stream_state(batch_size, freq_bins=self.n_freq, device=device, dtype=dtype))
        return tuple(states)

    def forward_stream(
        self,
        mixture: torch.Tensor,
        masks: torch.Tensor,
        context: torch.Tensor,
        states: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        expected = self.state_tensor_count()
        _runtime_assert(len(states) == expected, f"Expected {expected} correction states, got {len(states)}")
        h = self.fuse(torch.cat([self.mix_proj(mixture), self.mask_proj(masks), self.context_proj(context)], dim=1))
        new_states = []
        state_idx = 0
        for block in self.blocks:
            state_count = 1 if block.stream_context_frames() > 0 else 0
            h, block_states = block.forward_stream(h, states[state_idx : state_idx + state_count])
            state_idx += state_count
            new_states.extend(block_states)
        return self.out(h) * self.scale, tuple(new_states)


class OnlineSourceAwareMelBandStrongStudentSFC2D(nn.Module):
    """Handcrafted strongest-current NPU student for RoFormer-teacher distillation."""

    def __init__(
        self,
        n_freq: int,
        *,
        n_fft: int | None = None,
        sample_rate: int = 44100,
        n_src: int = 3,
        n_chan: int = 1,
        n_bands: int = 80,
        d_model: int = 48,
        n_encoder_layers: int = 2,
        n_source_layers: int = 5,
        correction_layers: int = 1,
        encoder_expansion: int = 2,
        encoder_ffn_mult: int = 4,
        source_local_expansion: int = 2,
        source_local_ffn_mult: int = 4,
        source_fusion_hidden: int = 192,
        source_seed_hidden: int = 192,
        expander_hidden: int = 128,
        mask_hidden: int = 192,
        correction_channels: int = 24,
        kernel_size: Sequence[int] | int = (3, 5),
        routing_kernel_size: Sequence[int] | int = (1, 3),
        encoder_dilation_cycle: Sequence[int] | None = (1, 2),
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
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        kernel_size = _as_pair(kernel_size, name="kernel_size")
        routing_kernel_size = _as_pair(routing_kernel_size, name="routing_kernel_size")
        self.n_freq = int(n_freq)
        self.n_bands = int(n_bands)
        self.n_src = int(n_src)
        self.n_chan = int(n_chan)
        self.d_model = int(d_model)
        self.n_encoder_layers = int(n_encoder_layers)
        self.n_source_layers = int(n_source_layers)
        self.correction_layers = int(correction_layers)
        self.correction_channels = int(correction_channels)
        self.causal = bool(causal)
        self.masking = bool(masking)
        self.mixture_consistency = bool(mixture_consistency)
        self.include_magnitude_features = bool(include_magnitude_features)
        self.include_logmag_features = bool(include_logmag_features)
        self.encoder_dilation_schedule = _normalize_dilation_schedule(
            n_encoder_layers,
            _as_dilation_cycle(encoder_dilation_cycle),
        )

        feature_channels = 2 * n_chan
        if include_magnitude_features:
            feature_channels += n_chan
        if include_logmag_features:
            feature_channels += n_chan
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
            nn.Conv2d(feature_channels, d_model, kernel_size=1, bias=True),
            RMSNorm2d(d_model),
            StrongTokenFFN2d(d_model, max(d_model * 3, source_seed_hidden // 2), residual_scale=0.1),
        )
        self.router = StrongAdaptiveMelRouter2d(
            channels=d_model,
            band_spec=band_spec,
            kernel_size=routing_kernel_size,
            causal=causal,
            normalization=routing_normalization,
        )
        self.encoder = nn.ModuleList(
            [
                StrongTemporalBandBlock2d(
                    d_model,
                    expansion=encoder_expansion,
                    token_ffn_mult=encoder_ffn_mult,
                    time_kernel_size=kernel_size[0],
                    band_kernel_size=kernel_size[1],
                    time_dilation=dilation,
                    causal=causal,
                    residual_scale=0.1,
                )
                for dilation in self.encoder_dilation_schedule
            ]
        )
        self.source_seed = StrongSourceSeed2d(channels=d_model, n_src=n_src, hidden_channels=source_seed_hidden)
        self.source_decoder = StrongSourceDecoder2d(
            channels=d_model,
            n_src=n_src,
            n_layers=n_source_layers,
            local_expansion=source_local_expansion,
            local_ffn_mult=source_local_ffn_mult,
            fusion_hidden_channels=source_fusion_hidden,
            band_kernel_size=kernel_size[1],
        )
        self.mask_head = StrongSourceMaskHead2d(
            channels=d_model,
            n_src=n_src,
            n_chan=n_chan,
            band_spec=band_spec,
            expander_hidden_channels=expander_hidden,
            mask_hidden_channels=mask_hidden,
            fullband_kernel_size=kernel_size[1],
        )
        self.context_expander = StrongMelBandExpander2d(
            channels=d_model,
            band_spec=band_spec,
            hidden_channels=expander_hidden,
        )
        self.context_fuse = nn.Sequential(
            RMSNorm2d(2 * d_model),
            nn.Conv2d(2 * d_model, 2 * d_model, kernel_size=1, bias=True),
            nn.GLU(dim=1),
        )
        self.correction = StrongMaskCorrectionHead2d(
            context_channels=d_model,
            correction_channels=correction_channels,
            n_freq=n_freq,
            n_src=n_src,
            n_chan=n_chan,
            n_layers=correction_layers,
            kernel_size=kernel_size,
            causal=causal,
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
        for block in self.encoder:
            z = block(z)
        return z, query_tokens

    def _apply_mixture_consistency(self, estimates: torch.Tensor, mixture: torch.Tensor) -> torch.Tensor:
        if not self.mixture_consistency:
            return estimates
        chunks = _source_chunks(estimates, n_src=self.n_src, channels=2 * self.n_chan)
        correction = (mixture - _sum_chunks(chunks)) / float(self.n_src)
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
        _runtime_assert(x.shape[1] == 2 * self.n_chan, f"Expected {2 * self.n_chan} packed channels, got {x.shape}")
        _runtime_assert(x.shape[-1] == self.n_freq, f"Expected F={self.n_freq}, got {x.shape}")
        z, query_tokens = self._encode(x)
        return self._decode(x, z, query_tokens)

    def stream_context_frames(self) -> int:
        if not self.causal:
            return 0
        return (
            self.router.stream_context_frames()
            + sum(block.stream_context_frames() for block in self.encoder)
            + self.correction.stream_context_frames()
        )

    def init_stream_state(self, batch_size: int = 1, *, device=None, dtype=None) -> tuple[torch.Tensor, ...]:
        if not self.causal:
            raise RuntimeError("Streaming state is only supported when causal=True.")
        states: list[torch.Tensor] = []
        states.extend(self.router.init_stream_state(batch_size, freq_bins=self.n_freq, device=device, dtype=dtype))
        for block in self.encoder:
            states.extend(block.init_stream_state(batch_size, freq_bins=self.n_bands, device=device, dtype=dtype))
        states.extend(self.correction.init_stream_state(batch_size=batch_size, device=device, dtype=dtype))
        return tuple(states)

    def forward_stream(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, ...] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if not self.causal:
            raise RuntimeError("forward_stream is only supported when causal=True.")
        _runtime_assert(x.ndim == 4, f"Expected [B,2M,T,F], got {x.shape}")
        _runtime_assert(x.shape[1] == 2 * self.n_chan, f"Expected {2 * self.n_chan} packed channels, got {x.shape}")
        _runtime_assert(x.shape[-1] == self.n_freq, f"Expected F={self.n_freq}, got {x.shape}")
        if state is None:
            state = self.init_stream_state(batch_size=x.shape[0], device=x.device, dtype=x.dtype)

        router_count = 1 if self.router.stream_context_frames() > 0 else 0
        encoder_count = sum(1 if block.stream_context_frames() > 0 else 0 for block in self.encoder)
        correction_count = self.correction.state_tensor_count()
        expected = router_count + encoder_count + correction_count
        _runtime_assert(len(state) == expected, f"Expected {expected} states, got {len(state)}")

        features = _packed_complex_features(
            x,
            n_chan=self.n_chan,
            include_magnitude=self.include_magnitude_features,
            include_logmag=self.include_logmag_features,
        )
        h = self.frontend(features)
        state_idx = 0
        router_end = router_count
        (z, query_tokens), new_router_states = self.router.forward_stream(h, state[state_idx:router_end])
        state_idx = router_end
        new_encoder_states = []
        for block in self.encoder:
            block_count = 1 if block.stream_context_frames() > 0 else 0
            z, block_states = block.forward_stream(z, state[state_idx : state_idx + block_count])
            state_idx += block_count
            new_encoder_states.extend(block_states)

        source_tokens = self.source_seed(z)
        source_tokens, _source_states = self.source_decoder.forward_stream(source_tokens, z, ())
        masks, source_context = self.mask_head(source_tokens, query_tokens)
        mixture_context = self.context_expander(z, query_tokens)
        correction_context = self.context_fuse(torch.cat([source_context, mixture_context], dim=1))
        mask_delta, new_correction_states = self.correction.forward_stream(
            x,
            masks,
            correction_context,
            state[state_idx:],
        )
        masks = masks + mask_delta
        if self.masking:
            estimates = _apply_packed_complex_mask_no_repeat(
                x=x,
                y=masks,
                n_src=self.n_src,
                n_chan=self.n_chan,
            )
            y = self._apply_mixture_consistency(estimates, x)
        else:
            y = masks
        return y, (*new_router_states, *new_encoder_states, *new_correction_states)

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
            "for OnlineSourceAwareMelBandStrongStudentSFC2D. Use forward_stream with layer caches."
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


class OnlineSourceAwareMelBandStrongStudentSFCModel(nn.Module):
    """Complex-STFT wrapper around OnlineSourceAwareMelBandStrongStudentSFC2D."""

    def __init__(self, *, n_freq: int, n_src: int = 3, n_chan: int = 1, **kwargs):
        super().__init__()
        self.core = OnlineSourceAwareMelBandStrongStudentSFC2D(n_freq=n_freq, n_src=n_src, n_chan=n_chan, **kwargs)
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


def build_source_aware_melband_strong_student_sfc_system(
    *,
    n_fft: int,
    hop_length: int,
    fs: int,
    n_src: int = 3,
    n_chan: int = 1,
    n_bands: int = 80,
    d_model: int = 48,
    n_encoder_layers: int = 2,
    n_source_layers: int = 5,
    correction_layers: int = 1,
    encoder_expansion: int = 2,
    encoder_ffn_mult: int = 4,
    source_local_expansion: int = 2,
    source_local_ffn_mult: int = 4,
    source_fusion_hidden: int = 192,
    source_seed_hidden: int = 192,
    expander_hidden: int = 128,
    mask_hidden: int = 192,
    correction_channels: int = 24,
    kernel_size: Sequence[int] | int = (3, 5),
    routing_kernel_size: Sequence[int] | int = (1, 3),
    encoder_dilation_cycle: Sequence[int] | None = (1, 2),
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
    core = OnlineSourceAwareMelBandStrongStudentSFC2D(
        n_freq=core_n_freq,
        n_fft=core_n_fft,
        sample_rate=fs,
        n_src=n_src,
        n_chan=n_chan,
        n_bands=n_bands,
        d_model=d_model,
        n_encoder_layers=n_encoder_layers,
        n_source_layers=n_source_layers,
        correction_layers=correction_layers,
        encoder_expansion=encoder_expansion,
        encoder_ffn_mult=encoder_ffn_mult,
        source_local_expansion=source_local_expansion,
        source_local_ffn_mult=source_local_ffn_mult,
        source_fusion_hidden=source_fusion_hidden,
        source_seed_hidden=source_seed_hidden,
        expander_hidden=expander_hidden,
        mask_hidden=mask_hidden,
        correction_channels=correction_channels,
        kernel_size=kernel_size,
        routing_kernel_size=routing_kernel_size,
        encoder_dilation_cycle=encoder_dilation_cycle,
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
