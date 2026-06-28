"""
Dolphin-inspired audio-only source separator for ASS (slim NPU variant).

This is the second generation of ``DolphinSFCNPU``.  It keeps the transferable
ideas from Dolphin:

- single-pass multi-scale U-shape separator,
- one temporal recurrence per separator block (serves the role of the old
  global + local conv pair),
- a lightweight source-prior mechanism,

but the block shapes and the compressor have been redesigned to respect
AGENT.md rule 13 (the 192 KiB DSP quota for streaming state).  Only one
streaming cache per block survives, and the compressor / downsamples are
stateless along the time axis.  The packed-state export wrapper added in the
previous revision is preserved so the exported graph still has exactly
``(x, state) -> (y, next_state)`` (AGENT.md rule 14).

Why this shape is chosen:

- The dominant source-separation features on 2D spectrograms are local-in-time
  and wide-in-frequency.  The slim block therefore keeps a causal depthwise
  conv on the time axis (cached) and a regular depthwise conv on the frequency
  axis (stateless, wider kernel), instead of paying for both a long temporal
  global conv and a short local conv.
- The source-prior role that the old ``DolphinSourcePriorCoder2d`` served is
  implemented in the block itself via a pointwise SiLU gate on the compressed
  band axis; this keeps the semantic-emphasis intent but costs zero state.
- The band compressor no longer caches left context: its temporal receptive
  field was only 3 frames and cost ~O(d_model * n_freq) bytes, which dominated
  the budget at full frequency resolution.  The separator still sees temporal
  context via its per-block caches, just downstream of the band compression
  so the width is n_bands << n_freq.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectral_feature_compression.core.model.online_sfc_2d import (
    BandSpec2d,
    CausalConv2d,
    RMSNorm2d,
    SpectralDecoder2d,
    _runtime_assert,
)


def _validate_even_pyramid(n_bands: int, num_scales: int) -> None:
    divisor = 2 ** max(num_scales - 1, 0)
    if n_bands % divisor != 0:
        raise ValueError(f"n_bands={n_bands} must be divisible by {divisor} for {num_scales} scales.")


# ---------------------------------------------------------------------------
# Band specification
# ---------------------------------------------------------------------------


class FrozenDolphinBandSpec2d(nn.Module):
    """
    Deterministic frozen band constants for DolphinSFCNPU.

    ``musical`` uses deterministic log-spaced triangular bands, ``linear`` uses
    uniformly spaced triangular bands.  No ``librosa`` dependency; the basis is
    a plain buffer so the graph folds it as a constant at export time.
    """

    def __init__(self, n_freq: int, n_bands: int, band_config: str = "musical"):
        super().__init__()
        if n_freq <= 0 or n_bands <= 0:
            raise ValueError("n_freq and n_bands must be positive.")
        self.n_freq = n_freq
        self.n_bands = n_bands
        self.band_config = band_config
        basis = self._build_basis(n_freq=n_freq, n_bands=n_bands, band_config=band_config)
        self.register_buffer("basis", basis.view(1, n_bands, 1, n_freq))

    @staticmethod
    def _build_basis(n_freq: int, n_bands: int, band_config: str) -> torch.Tensor:
        if band_config == "linear":
            edges = torch.linspace(0.0, float(n_freq - 1), steps=n_bands + 2)
        elif band_config == "musical":
            max_pos = torch.log1p(torch.tensor(float(n_freq - 1)))
            edges = torch.expm1(torch.linspace(0.0, float(max_pos), steps=n_bands + 2))
        else:
            raise ValueError(f"Unsupported frozen band_config: {band_config!r}")

        freq_pos = torch.arange(n_freq, dtype=torch.float32)
        basis = torch.zeros(n_bands, n_freq, dtype=torch.float32)
        for band_idx in range(n_bands):
            left = edges[band_idx]
            center = edges[band_idx + 1]
            right = torch.maximum(edges[band_idx + 2], center + 1.0)
            rising = (freq_pos - left) / (center - left).clamp_min(1.0)
            falling = (right - freq_pos) / (right - center).clamp_min(1.0)
            basis[band_idx] = torch.clamp(torch.minimum(rising, falling), min=0.0, max=1.0)
            if basis[band_idx].amax() <= 0:
                nearest = int(torch.clamp(center.round(), min=0, max=n_freq - 1).item())
                basis[band_idx, nearest] = 1.0
        return basis

    def band_bias(self) -> torch.Tensor:
        peak = self.basis.amax(dim=-1, keepdim=True).clamp_min(1e-6)
        return 2.0 * (self.basis / peak) - 1.0

    def decode_basis(self) -> torch.Tensor:
        return self.basis / self.basis.sum(dim=1, keepdim=True).clamp_min(1e-6)

    def routing_bias(self) -> torch.Tensor:
        """Alias used by the query/cross-attention Dolphin variants."""

        return self.band_bias()

    def expansion_basis(self) -> torch.Tensor:
        """Alias used by the query/cross-attention Dolphin variants."""

        return self.decode_basis()


# ---------------------------------------------------------------------------
# Stateless building blocks
# ---------------------------------------------------------------------------


class StatelessBandCompressor2d(nn.Module):
    """
    Compress (B, C, T, F) -> (B, C, T, K) with a stateless frequency-only
    depthwise refinement and a soft band-pooling basis.

    The old ``SpectralCompressor2d`` kept a ``CausalConv2d((3, 3))`` cache at
    full ``n_freq`` resolution, which dominated the streaming-state budget.
    Here the temporal receptive field is folded into the downstream separator
    blocks (which operate on ``n_bands`` instead of ``n_freq``, so their
    caches are much cheaper), and the compressor itself is time-stateless.
    """

    def __init__(self, channels: int, band_spec: FrozenDolphinBandSpec2d | BandSpec2d, freq_kernel: int = 3):
        super().__init__()
        if freq_kernel % 2 == 0:
            raise ValueError(f"freq_kernel must be odd; got {freq_kernel}.")
        self.channels = channels
        self.band_spec = band_spec
        self.n_bands = band_spec.n_bands

        self.norm = RMSNorm2d(channels)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        # Frequency-only depthwise refinement: stateless along time.
        self.dw_f = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, freq_kernel),
            padding=(0, freq_kernel // 2),
            groups=channels,
            bias=True,
        )
        self.score = nn.Conv2d(channels, self.n_bands, kernel_size=1, bias=True)
        self.score_scale = nn.Parameter(torch.tensor(1.0))
        self.bias_scale = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("band_bias", band_spec.band_bias())

    def _compress(self, h: torch.Tensor) -> torch.Tensor:
        scores = self.score(h) * self.score_scale + self.band_bias * self.bias_scale
        weights = torch.softmax(scores, dim=-1)
        batch, channels, n_frames, n_freq = h.shape

        # Batched band pooling from F bins to K band tokens; matches the
        # original SFC compressor contract so downstream shapes are unchanged.
        h_btfc = h.permute(0, 2, 3, 1).reshape(batch * n_frames, n_freq, channels)
        w_btkf = weights.permute(0, 2, 1, 3).reshape(batch * n_frames, self.n_bands, n_freq)
        z_btkc = torch.bmm(w_btkf, h_btfc)
        return z_btkc.reshape(batch, n_frames, self.n_bands, channels).permute(0, 3, 1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _runtime_assert(x.shape[-1] == self.band_spec.n_freq, f"{x.shape} vs {self.band_spec.n_freq}")
        h = self.dw_f(self.pw(self.norm(x)))
        return self._compress(h)

    # Stateless along time: streaming == offline.
    def forward_stream(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class StatelessSoftBandQueryCompressor2d(nn.Module):
    """Dolphin compressor that also emits K-band query tokens.

    This is the low-risk query variant: both the latent and query side-path stay
    on the compressed band axis, so it adds no streaming cache and avoids a
    full-resolution attention decoder.
    """

    def __init__(self, channels: int, band_spec: FrozenDolphinBandSpec2d | BandSpec2d, freq_kernel: int = 3):
        super().__init__()
        if freq_kernel % 2 == 0:
            raise ValueError(f"freq_kernel must be odd; got {freq_kernel}.")
        self.channels = channels
        self.band_spec = band_spec
        self.n_bands = band_spec.n_bands

        self.norm = RMSNorm2d(channels)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.dw_f = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, freq_kernel),
            padding=(0, freq_kernel // 2),
            groups=channels,
            bias=True,
        )
        self.score = nn.Conv2d(channels, self.n_bands, kernel_size=1, bias=True)
        self.value = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.query = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.score_scale = nn.Parameter(torch.tensor(1.0))
        self.bias_scale = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("band_bias", band_spec.band_bias())

    def _pool(self, h: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        batch, channels, n_frames, n_freq = h.shape
        h_btfc = h.permute(0, 2, 3, 1).reshape(batch * n_frames, n_freq, channels)
        w_btkf = weights.permute(0, 2, 1, 3).reshape(batch * n_frames, self.n_bands, n_freq)
        z_btkc = torch.bmm(w_btkf, h_btfc)
        return z_btkc.reshape(batch, n_frames, self.n_bands, channels).permute(0, 3, 1, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _runtime_assert(x.shape[-1] == self.band_spec.n_freq, f"{x.shape} vs {self.band_spec.n_freq}")
        h = F.silu(self.dw_f(self.pw(self.norm(x))))
        scores = self.score(h) * self.score_scale + self.band_bias * self.bias_scale
        weights = torch.softmax(scores, dim=-1)
        return self._pool(self.value(h), weights), self._pool(self.query(h), weights)

    def forward_stream(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward(x)


class DolphinSoftBandQueryDecoder2d(nn.Module):
    """K-band query-conditioned decoder for DolphinSFCNPU."""

    def __init__(self, channels: int, query_channels: int, band_spec: FrozenDolphinBandSpec2d | BandSpec2d):
        super().__init__()
        self.channels = channels
        self.query_channels = query_channels
        self.band_spec = band_spec
        self.n_bands = band_spec.n_bands
        self.n_freq = band_spec.n_freq

        self.latent_pre = nn.Sequential(RMSNorm2d(channels), nn.Conv2d(channels, channels, kernel_size=1, bias=True))
        self.query_pre = nn.Sequential(
            RMSNorm2d(query_channels),
            nn.Conv2d(query_channels, channels, kernel_size=1, bias=True),
        )
        self.fuse = nn.Conv2d(2 * channels, channels, kernel_size=1, bias=True)
        self.band_gain = nn.Conv2d(channels, 1, kernel_size=1, bias=True)
        self.query_skip_scale = nn.Parameter(torch.tensor(1.0))
        self.gain_scale = nn.Parameter(torch.tensor(1.0))
        self.basis_scale = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("decode_basis", band_spec.decode_basis())

    def forward(self, z: torch.Tensor, query_tokens: torch.Tensor) -> torch.Tensor:
        _runtime_assert(z.shape[-1] == self.n_bands, f"{z.shape} vs {self.n_bands}")
        _runtime_assert(query_tokens.shape[-1] == self.n_bands, f"{query_tokens.shape} vs {self.n_bands}")
        latent_h = F.silu(self.latent_pre(z))
        query_h = F.silu(self.query_pre(query_tokens))
        fused = F.silu(self.fuse(torch.cat([latent_h, query_h], dim=1)))

        gains = 1.0 + torch.sigmoid(self.band_gain(fused)) * self.gain_scale
        gains = gains.permute(0, 3, 2, 1)  # (B, K, T, 1)
        coeff = self.decode_basis * (self.basis_scale + gains)
        coeff = coeff / (coeff.sum(dim=1, keepdim=True) + 1.0e-6)

        tokens = latent_h + query_h * self.query_skip_scale
        batch, channels, n_frames, _ = tokens.shape
        tokens_btck = tokens.permute(0, 2, 1, 3).reshape(batch * n_frames, channels, self.n_bands)
        coeff_btkf = coeff.permute(0, 2, 1, 3).reshape(batch * n_frames, self.n_bands, self.n_freq)
        expanded = torch.bmm(tokens_btck, coeff_btkf)
        return expanded.reshape(batch, n_frames, channels, self.n_freq).permute(0, 2, 1, 3)


class StatelessCrossAttentionQueryCompressor2d(nn.Module):
    """Stateless F->K cross-attention compressor for DolphinSFCNPU."""

    def __init__(
        self,
        channels: int,
        band_spec: FrozenDolphinBandSpec2d | BandSpec2d,
        *,
        freq_kernel: int = 3,
        query_type: str = "adaptive",
    ):
        super().__init__()
        if query_type not in {"adaptive", "learnable"}:
            raise ValueError(f"Unsupported query_type={query_type!r}.")
        if freq_kernel % 2 == 0:
            raise ValueError(f"freq_kernel must be odd; got {freq_kernel}.")
        self.channels = channels
        self.band_spec = band_spec
        self.n_bands = band_spec.n_bands
        self.n_freq = band_spec.n_freq
        self.query_type = query_type

        self.norm = RMSNorm2d(channels)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.dw_f = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, freq_kernel),
            padding=(0, freq_kernel // 2),
            groups=channels,
            bias=True,
        )
        self.q_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.k_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.v_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.score_scale = nn.Parameter(torch.tensor(1.0 / math.sqrt(max(channels, 1)), dtype=torch.float32))
        self.prior_scale = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("routing_bias", band_spec.routing_bias())
        self.register_buffer("query_basis", band_spec.expansion_basis())
        if query_type == "learnable":
            self.query = nn.Parameter(torch.randn(1, channels, 1, self.n_bands) * 0.02)
        else:
            self.query = None

    def _pool_query_tokens(self, h: torch.Tensor) -> torch.Tensor:
        batch, channels, n_frames, n_freq = h.shape
        h_btfc = h.permute(0, 2, 3, 1).reshape(batch * n_frames, n_freq, channels)
        basis_kf = self.query_basis.reshape(self.n_bands, n_freq).to(dtype=h.dtype)
        pooled = torch.matmul(basis_kf, h_btfc)
        return pooled.reshape(batch, n_frames, self.n_bands, channels).permute(0, 3, 1, 2)

    def _prepare_query(self, h: torch.Tensor) -> torch.Tensor:
        if self.query is not None:
            return self.query.expand(h.shape[0], -1, h.shape[2], -1)
        return self._pool_query_tokens(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _runtime_assert(x.shape[-1] == self.n_freq, f"{x.shape} vs {self.n_freq}")
        h = F.silu(self.dw_f(self.pw(self.norm(x))))
        query_seed = self._prepare_query(h)
        keys = self.k_proj(h)
        values = self.v_proj(h)
        queries = self.q_proj(query_seed)

        batch, channels, n_frames, n_freq = h.shape
        q = queries.permute(0, 2, 3, 1).reshape(batch * n_frames, self.n_bands, channels)
        k = keys.permute(0, 2, 3, 1).reshape(batch * n_frames, n_freq, channels)
        v = values.permute(0, 2, 3, 1).reshape(batch * n_frames, n_freq, channels)

        scores = torch.bmm(q, k.transpose(1, 2)) * self.score_scale.to(dtype=h.dtype)
        scores = scores.reshape(batch, n_frames, self.n_bands, n_freq)
        bias = self.routing_bias.permute(0, 2, 1, 3).to(dtype=scores.dtype)
        scores = scores + bias * self.prior_scale.to(dtype=scores.dtype)
        weights = torch.softmax(scores.reshape(batch * n_frames, self.n_bands, n_freq), dim=-1)
        z = torch.bmm(weights, v)
        z = z.reshape(batch, n_frames, self.n_bands, channels).permute(0, 3, 1, 2)
        return self.out_proj(z), h

    def forward_stream(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward(x)


class DolphinCrossAttentionQueryDecoder2d(nn.Module):
    """Full-resolution query decoder: F queries attend to K Dolphin tokens."""

    def __init__(self, channels: int, side_channels: int, band_spec: FrozenDolphinBandSpec2d | BandSpec2d):
        super().__init__()
        self.channels = channels
        self.side_channels = side_channels
        self.band_spec = band_spec
        self.n_bands = band_spec.n_bands
        self.n_freq = band_spec.n_freq

        self.latent_pre = nn.Sequential(RMSNorm2d(channels), nn.Conv2d(channels, channels, kernel_size=1, bias=True))
        self.query_pre = nn.Sequential(
            RMSNorm2d(side_channels),
            nn.Conv2d(side_channels, channels, kernel_size=1, bias=True),
        )
        self.q_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.k_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.v_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.score_scale = nn.Parameter(torch.tensor(1.0 / math.sqrt(max(channels, 1)), dtype=torch.float32))
        self.prior_scale = nn.Parameter(torch.tensor(1.0))
        self.query_skip_scale = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("expansion_basis", band_spec.expansion_basis())

    def forward(self, z: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
        _runtime_assert(z.shape[-1] == self.n_bands, f"{z.shape} vs {self.n_bands}")
        _runtime_assert(side.shape[-1] == self.n_freq, f"{side.shape} vs {self.n_freq}")
        latent_h = F.silu(self.latent_pre(z))
        query_h = F.silu(self.query_pre(side))

        batch, channels, n_frames, _ = query_h.shape
        q = self.q_proj(query_h).permute(0, 2, 3, 1).reshape(batch * n_frames, self.n_freq, channels)
        k = self.k_proj(latent_h).permute(0, 2, 3, 1).reshape(batch * n_frames, self.n_bands, channels)
        v = self.v_proj(latent_h).permute(0, 2, 3, 1).reshape(batch * n_frames, self.n_bands, channels)

        scores = torch.bmm(q, k.transpose(1, 2)) * self.score_scale.to(dtype=query_h.dtype)
        scores = scores.reshape(batch, n_frames, self.n_freq, self.n_bands)
        bias = self.expansion_basis.squeeze(0).squeeze(1).transpose(0, 1).reshape(1, 1, self.n_freq, self.n_bands)
        scores = scores + bias.to(dtype=scores.dtype) * self.prior_scale.to(dtype=scores.dtype)
        weights = torch.softmax(scores.reshape(batch * n_frames, self.n_freq, self.n_bands), dim=-1)
        y = torch.bmm(weights, v)
        y = y.reshape(batch, n_frames, self.n_freq, channels).permute(0, 3, 1, 2)
        return self.out_proj(y) + query_h * self.query_skip_scale


class StatelessBandDown(nn.Module):
    """
    Downsample n_bands -> n_bands/2 with a stride-2 frequency conv only.

    No cached state along the time axis; the separator block *before* this
    downsample is the one that carries temporal receptive field.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, 4),
            stride=(1, 2),
            padding=(0, 1),
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class StatelessBandUp(nn.Module):
    """Upsample n_bands/2 -> n_bands via stride-2 frequency transposed conv."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            channels,
            channels,
            kernel_size=(1, 4),
            stride=(1, 2),
            padding=(0, 1),
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ---------------------------------------------------------------------------
# Slim separator block — one streaming cache only
# ---------------------------------------------------------------------------


class DolphinSFCNPUSlimBlock(nn.Module):
    """
    Single-cache separator block.

    Structure (all 2-D, all causal along time, all NPU-friendly):

      residual ->
        RMSNorm -> pointwise(2*C) -> SiLU gate (halves to C) ->
        causal depthwise over time (kt x 1)  [<-- the only streaming cache]
        pointwise(C) -> residual add
      -> RMSNorm -> pointwise(hidden*2) -> SiLU gate ->
        depthwise over frequency (1 x kf)  [stateless]
        pointwise(C) -> residual add

    Two residual sub-blocks: a temporal time-token mixer (one cached conv) and
    a frequency-channel mixer (no cache).  The temporal sub-block's gate plays
    the source-prior role by modulating which features propagate through the
    time cache, which is why the old standalone ``DolphinSourcePriorCoder2d``
    is no longer needed.

    Streaming state is a single tensor shaped ``(B, C, kt-1, bands)``.
    """

    def __init__(
        self,
        channels: int,
        time_kernel: int = 3,
        freq_kernel: int = 3,
        ffn_expansion: int = 2,
    ):
        super().__init__()
        if time_kernel < 1:
            raise ValueError("time_kernel must be >= 1.")
        if freq_kernel % 2 == 0:
            raise ValueError(f"freq_kernel must be odd; got {freq_kernel}.")
        if (time_kernel - 1) >= 14:
            raise ValueError("time_kernel violates AGENT.md rule 5.")
        if (freq_kernel - 1) >= 14:
            raise ValueError("freq_kernel violates AGENT.md rule 5.")

        self.channels = channels
        self.time_kernel = time_kernel
        self.freq_kernel = freq_kernel

        # --- Temporal sub-block (owns the single streaming cache) ---
        self.t_norm = RMSNorm2d(channels)
        self.t_in = nn.Conv2d(channels, channels * 2, kernel_size=1, bias=True)
        self.t_dw = CausalConv2d(
            channels,
            channels,
            kernel_size=(time_kernel, 1),
            groups=channels,
            bias=True,
        )
        self.t_out = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

        # --- Frequency / channel-mix sub-block (stateless) ---
        hidden = channels * ffn_expansion
        self.f_norm = RMSNorm2d(channels)
        self.f_in = nn.Conv2d(channels, hidden * 2, kernel_size=1, bias=True)
        self.f_dw = nn.Conv2d(
            hidden,
            hidden,
            kernel_size=(1, freq_kernel),
            padding=(0, freq_kernel // 2),
            groups=hidden,
            bias=True,
        )
        self.f_out = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)

    @property
    def streaming_context_frames(self) -> int:
        return self.t_dw.stream_context_frames()

    # -- offline / training path ---------------------------------------------

    def _temporal(self, x: torch.Tensor) -> torch.Tensor:
        y = self.t_norm(x)
        a, b = self.t_in(y).chunk(2, dim=1)
        y = a * torch.sigmoid(b)  # gate plays the source-prior role
        y = self.t_dw(y)
        y = F.silu(y)
        return x + self.t_out(y)

    def _freq_channel(self, x: torch.Tensor) -> torch.Tensor:
        y = self.f_norm(x)
        a, b = self.f_in(y).chunk(2, dim=1)
        y = a * torch.sigmoid(b)
        y = F.silu(self.f_dw(y))
        return x + self.f_out(y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._freq_channel(self._temporal(x))

    # -- streaming path -------------------------------------------------------

    def init_stream_state(
        self,
        batch_size: int,
        *,
        freq_bins: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        return self.t_dw.init_stream_state(batch_size, freq_bins=freq_bins, device=device, dtype=dtype)

    def forward_stream(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y = self.t_norm(x)
        a, b = self.t_in(y).chunk(2, dim=1)
        y = a * torch.sigmoid(b)
        y, new_state = self.t_dw.forward_stream(y, state)
        y = F.silu(y)
        x = x + self.t_out(y)
        x = self._freq_channel(x)
        return x, new_state


# ---------------------------------------------------------------------------
# Encoder / decoder stages (state comes from the block, not the resampler)
# ---------------------------------------------------------------------------


class DolphinSFCNPUSlimEncoderStage(nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        num_blocks: int,
        time_kernel: int,
        freq_kernel: int,
        do_downsample: bool,
        ffn_expansion: int = 2,
    ):
        super().__init__()
        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1.")
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.do_downsample = do_downsample

        # Channel projection: stateless 1x1 projection at the top of the stage.
        if channels_in != channels_out:
            self.project = nn.Conv2d(channels_in, channels_out, kernel_size=1, bias=True)
        else:
            self.project = nn.Identity()

        self.blocks = nn.ModuleList(
            DolphinSFCNPUSlimBlock(
                channels=channels_out,
                time_kernel=time_kernel,
                freq_kernel=freq_kernel,
                ffn_expansion=ffn_expansion,
            )
            for _ in range(num_blocks)
        )

        if do_downsample:
            self.down = StatelessBandDown(channels_out)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.project(x)
        for block in self.blocks:
            x = block(x)
        skip = x
        if self.do_downsample:
            x = self.down(x)
        return x, skip

    def init_stream_state(
        self,
        batch_size: int,
        *,
        freq_bins: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, ...]:
        return tuple(
            block.init_stream_state(batch_size, freq_bins=freq_bins, device=device, dtype=dtype)
            for block in self.blocks
        )

    def forward_stream(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        x = self.project(x)
        new_states: list[torch.Tensor] = []
        for block, block_state in zip(self.blocks, state):
            x, new_state = block.forward_stream(x, block_state)
            new_states.append(new_state)
        skip = x
        if self.do_downsample:
            x = self.down(x)
        return x, skip, tuple(new_states)


class DolphinSFCNPUSlimDecoderStage(nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        num_blocks: int,
        time_kernel: int,
        freq_kernel: int,
        do_upsample: bool,
        ffn_expansion: int = 2,
    ):
        super().__init__()
        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1.")
        self.do_upsample = do_upsample
        self.channels_in = channels_in
        self.channels_out = channels_out

        if do_upsample:
            self.up = StatelessBandUp(channels_in)
            self.merge = nn.Conv2d(channels_in + channels_out, channels_out, kernel_size=1, bias=True)
        else:
            if channels_in != channels_out:
                self.project = nn.Conv2d(channels_in, channels_out, kernel_size=1, bias=True)
            else:
                self.project = nn.Identity()

        self.blocks = nn.ModuleList(
            DolphinSFCNPUSlimBlock(
                channels=channels_out,
                time_kernel=time_kernel,
                freq_kernel=freq_kernel,
                ffn_expansion=ffn_expansion,
            )
            for _ in range(num_blocks)
        )

    def _join(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if self.do_upsample:
            x = self.up(x)
            x = self.merge(torch.cat([x, skip], dim=1))
        else:
            x = self.project(x)
        return x

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self._join(x, skip)
        for block in self.blocks:
            x = block(x)
        return x

    def init_stream_state(
        self,
        batch_size: int,
        *,
        freq_bins: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, ...]:
        return tuple(
            block.init_stream_state(batch_size, freq_bins=freq_bins, device=device, dtype=dtype)
            for block in self.blocks
        )

    def forward_stream(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        x = self._join(x, skip)
        new_states: list[torch.Tensor] = []
        for block, block_state in zip(self.blocks, state):
            x, new_state = block.forward_stream(x, block_state)
            new_states.append(new_state)
        return x, tuple(new_states)


# ---------------------------------------------------------------------------
# Source-aware compressed-token refinement
# ---------------------------------------------------------------------------


class DolphinSourceTokenRefinementBlock2d(nn.Module):
    """Stateless per-source compressed-token refinement block."""

    def __init__(self, channels: int, freq_kernel: int = 5, expansion: int = 2):
        super().__init__()
        if freq_kernel % 2 == 0:
            raise ValueError(f"freq_kernel must be odd; got {freq_kernel}.")
        if (freq_kernel - 1) >= 14:
            raise ValueError("freq_kernel violates AGENT.md rule 5.")
        if expansion < 1:
            raise ValueError(f"expansion must be >= 1, got {expansion}.")

        hidden = channels * expansion
        self.norm = RMSNorm2d(channels)
        self.in_proj = nn.Conv2d(channels, hidden * 2, kernel_size=1, bias=True)
        self.freq_dw = nn.Conv2d(
            hidden,
            hidden,
            kernel_size=(1, freq_kernel),
            padding=(0, freq_kernel // 2),
            groups=hidden,
            bias=True,
        )
        self.out_proj = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.in_proj(self.norm(x)).chunk(2, dim=1)
        y = a * torch.sigmoid(b)
        y = F.silu(self.freq_dw(y))
        return x + self.out_proj(y)


class DolphinSourceTokenRefiner2d(nn.Module):
    """Per-source token/head refinement on the compressed band axis."""

    def __init__(
        self,
        channels: int,
        *,
        layers: int = 2,
        freq_kernel: int = 5,
        expansion: int = 2,
    ):
        super().__init__()
        if layers < 0:
            raise ValueError(f"layers must be non-negative, got {layers}.")
        self.in_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.blocks = nn.ModuleList(
            DolphinSourceTokenRefinementBlock2d(channels, freq_kernel=freq_kernel, expansion=expansion)
            for _ in range(layers)
        )
        self.out_norm = RMSNorm2d(channels)

    def forward(self, shared_tokens: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.in_proj(shared_tokens))
        for block in self.blocks:
            x = block(x)
        return self.out_norm(x)


# ---------------------------------------------------------------------------
# Top-level separator
# ---------------------------------------------------------------------------


class DolphinSFCNPUSeparator(nn.Module):
    """
    Audio-only Dolphin/SFC separator, slim NPU variant.

    Input/output contract:
      x: (B, 2 * n_chan, T, F), packed real/imag STFT
      y: (B, 2 * n_src * n_chan, T, F)
    """

    def __init__(
        self,
        n_freq: int,
        n_bands: int = 48,
        n_fft: int | None = None,
        sample_rate: int | None = None,
        band_config: str = "musical",
        n_src: int = 3,
        n_chan: int = 1,
        d_model: int = 128,
        num_scales: int = 3,
        widths: tuple[int, ...] | None = None,
        blocks_per_scale: tuple[int, ...] | None = None,
        time_kernels: tuple[int, ...] | None = None,
        freq_kernels: tuple[int, ...] | None = None,
        compressor_freq_kernel: int = 3,
        ffn_expansion: int = 2,
        masking: bool = True,
        mask_activation: str = "sigmoid",
        query_variant: str = "none",
        query_type: str = "adaptive",
    ):
        super().__init__()
        _validate_even_pyramid(n_bands, num_scales)
        if mask_activation not in {"sigmoid", "softmax"}:
            raise ValueError(f"Unsupported mask_activation={mask_activation!r}; expected 'sigmoid' or 'softmax'.")
        if query_variant not in {"none", "soft_band_query", "crossattn_query"}:
            raise ValueError(
                f"Unsupported query_variant={query_variant!r}; "
                "expected 'none', 'soft_band_query', or 'crossattn_query'."
            )
        if query_type not in {"adaptive", "learnable"}:
            raise ValueError(f"Unsupported query_type={query_type!r}; expected 'adaptive' or 'learnable'.")
        if widths is None:
            widths = tuple(d_model * (2**i) for i in range(num_scales))
        if len(widths) != num_scales:
            raise ValueError(f"widths must have {num_scales} entries, got {widths}.")
        if blocks_per_scale is None:
            blocks_per_scale = (1,) * num_scales
        if len(blocks_per_scale) != num_scales:
            raise ValueError(f"blocks_per_scale must have {num_scales} entries, got {blocks_per_scale}.")
        if time_kernels is None:
            time_kernels = (3,) * num_scales
        if len(time_kernels) != num_scales:
            raise ValueError(f"time_kernels must have {num_scales} entries, got {time_kernels}.")
        if freq_kernels is None:
            freq_kernels = (3,) * num_scales
        if len(freq_kernels) != num_scales:
            raise ValueError(f"freq_kernels must have {num_scales} entries, got {freq_kernels}.")
        if compressor_freq_kernel % 2 == 0:
            raise ValueError(f"compressor_freq_kernel must be odd, got {compressor_freq_kernel}.")
        if (compressor_freq_kernel - 1) >= 14:
            raise ValueError("compressor_freq_kernel violates AGENT.md rule 5.")
        if ffn_expansion < 1:
            raise ValueError(f"ffn_expansion must be >= 1, got {ffn_expansion}.")

        self.n_freq = n_freq
        self.n_bands = n_bands
        self.n_src = n_src
        self.n_chan = n_chan
        self.d_model = d_model
        self.num_scales = num_scales
        self.widths = tuple(widths)
        self.blocks_per_scale = tuple(blocks_per_scale)
        self.time_kernels = tuple(time_kernels)
        self.freq_kernels = tuple(freq_kernels)
        self.compressor_freq_kernel = compressor_freq_kernel
        self.ffn_expansion = ffn_expansion
        self.masking = masking
        self.mask_activation = mask_activation
        self.query_variant = query_variant
        self.query_type = query_type

        self.band_spec = self._build_band_spec(
            n_freq=n_freq,
            n_bands=n_bands,
            n_fft=n_fft,
            sample_rate=sample_rate,
            band_config=band_config,
        )
        self.in_proj = nn.Sequential(nn.Conv2d(2 * n_chan, d_model, kernel_size=1), RMSNorm2d(d_model))
        if query_variant == "soft_band_query":
            self.compressor = StatelessSoftBandQueryCompressor2d(
                d_model,
                self.band_spec,
                freq_kernel=compressor_freq_kernel,
            )
        elif query_variant == "crossattn_query":
            self.compressor = StatelessCrossAttentionQueryCompressor2d(
                d_model,
                self.band_spec,
                freq_kernel=compressor_freq_kernel,
                query_type=query_type,
            )
        else:
            self.compressor = StatelessBandCompressor2d(d_model, self.band_spec, freq_kernel=compressor_freq_kernel)

        encoder_stages: list[DolphinSFCNPUSlimEncoderStage] = []
        prev_channels = d_model
        for idx in range(num_scales):
            encoder_stages.append(
                DolphinSFCNPUSlimEncoderStage(
                    channels_in=prev_channels,
                    channels_out=self.widths[idx],
                    num_blocks=self.blocks_per_scale[idx],
                    time_kernel=self.time_kernels[idx],
                    freq_kernel=self.freq_kernels[idx],
                    do_downsample=idx < num_scales - 1,
                    ffn_expansion=ffn_expansion,
                )
            )
            prev_channels = self.widths[idx]
        self.encoder = nn.ModuleList(encoder_stages)

        decoder_stages: list[DolphinSFCNPUSlimDecoderStage] = []
        # Decoder mirrors the encoder with shared block-count / kernel pattern.
        for idx in range(num_scales):
            # idx==0 corresponds to the deepest level (no upsample).
            scale_idx = num_scales - 1 - idx
            channels_in = self.widths[scale_idx] if idx == 0 else self.widths[scale_idx + 1]
            decoder_stages.append(
                DolphinSFCNPUSlimDecoderStage(
                    channels_in=channels_in,
                    channels_out=self.widths[scale_idx],
                    num_blocks=self.blocks_per_scale[scale_idx],
                    time_kernel=self.time_kernels[scale_idx],
                    freq_kernel=self.freq_kernels[scale_idx],
                    do_upsample=idx > 0,
                    ffn_expansion=ffn_expansion,
                )
            )
        self.decoder = nn.ModuleList(decoder_stages)

        if query_variant == "soft_band_query":
            self.decoder_to_freq = DolphinSoftBandQueryDecoder2d(self.widths[0], d_model, self.band_spec)
        elif query_variant == "crossattn_query":
            self.decoder_to_freq = DolphinCrossAttentionQueryDecoder2d(self.widths[0], d_model, self.band_spec)
        else:
            self.decoder_to_freq = SpectralDecoder2d(self.widths[0], self.band_spec)
        out_ch = n_src * n_chan if masking else 2 * n_src * n_chan
        self.out_proj = nn.Conv2d(self.widths[0], out_ch, kernel_size=1)
        self._init_output_head()

    def _init_output_head(self) -> None:
        """Start from a mixture split instead of random source gains."""

        nn.init.zeros_(self.out_proj.weight)
        if self.out_proj.bias is None:
            return
        nn.init.zeros_(self.out_proj.bias)
        if not self.masking:
            return

        if self.mask_activation == "softmax":
            return

        eps = 1.0e-4
        source_gain = min(max(1.0 / float(self.n_src), eps), 1.0 - eps)
        source_logit = math.log(source_gain / (1.0 - source_gain))
        with torch.no_grad():
            self.out_proj.bias.fill_(source_logit)

    @staticmethod
    def _build_band_spec(
        n_freq: int,
        n_bands: int,
        n_fft: int | None,
        sample_rate: int | None,
        band_config: str,
    ) -> FrozenDolphinBandSpec2d:
        _ = n_fft, sample_rate
        return FrozenDolphinBandSpec2d(n_freq=n_freq, n_bands=n_bands, band_config=band_config)

    # -- offline ------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _runtime_assert(x.ndim == 4, f"Expected (B,C,T,F), got {x.shape}")
        _runtime_assert(x.shape[-1] == self.n_freq, f"{x.shape[-1]} vs {self.n_freq}")
        query_side = None
        compressed = self.compressor(self.in_proj(x))
        if self.query_variant == "none":
            z = compressed
        else:
            z, query_side = compressed

        skips: list[torch.Tensor] = []
        for stage in self.encoder:
            z, skip = stage(z)
            skips.append(skip)

        # Decoder iterates from the deepest stage upward.  The deepest stage
        # has ``do_upsample=False`` and simply refines the bottleneck with its
        # own blocks; stages above it take the upsampled lower feature map and
        # concatenate with the matching encoder skip.
        for idx, stage in enumerate(self.decoder):
            scale_idx = self.num_scales - 1 - idx
            z = stage(z, skips[scale_idx])

        if self.query_variant == "none":
            decoded = self.decoder_to_freq(z)
        else:
            _runtime_assert(query_side is not None, "Query side-path was not produced by the compressor.")
            decoded = self.decoder_to_freq(z, query_side)
        y = self.out_proj(decoded)
        if self.masking:
            y = apply_source_gain_mask_4d(
                x,
                y,
                n_src=self.n_src,
                n_chan=self.n_chan,
                activation=self.mask_activation,
            )
        return y

    # -- streaming ----------------------------------------------------------

    def init_stream_state(self, batch_size: int = 1, *, device=None, dtype=None):
        enc_states: list[tuple[torch.Tensor, ...]] = []
        bands = self.n_bands
        for idx, stage in enumerate(self.encoder):
            enc_states.append(stage.init_stream_state(batch_size, freq_bins=bands, device=device, dtype=dtype))
            if idx < self.num_scales - 1:
                bands = bands // 2

        dec_states: list[tuple[torch.Tensor, ...]] = []
        # Deepest stage first, mirroring forward_stream order.
        for idx, stage in enumerate(self.decoder):
            dec_states.append(stage.init_stream_state(batch_size, freq_bins=bands, device=device, dtype=dtype))
            if idx < self.num_scales - 1:
                bands = bands * 2

        return (tuple(enc_states), tuple(dec_states))

    def forward_stream(self, x: torch.Tensor, state=None):
        _runtime_assert(x.ndim == 4, f"Expected (B,C,T,F), got {x.shape}")
        _runtime_assert(x.shape[2] == 1, "forward_stream expects one frame at a time.")
        if state is None:
            state = self.init_stream_state(batch_size=x.shape[0], device=x.device, dtype=x.dtype)

        enc_states, dec_states = state
        query_side = None
        compressed = self.compressor.forward_stream(self.in_proj(x))
        if self.query_variant == "none":
            z = compressed
        else:
            z, query_side = compressed

        skips: list[torch.Tensor] = []
        new_enc_states: list[tuple[torch.Tensor, ...]] = []
        for stage, stage_state in zip(self.encoder, enc_states):
            z, skip, new_state = stage.forward_stream(z, stage_state)
            skips.append(skip)
            new_enc_states.append(new_state)

        new_dec_states: list[tuple[torch.Tensor, ...]] = []
        for idx, (stage, stage_state) in enumerate(zip(self.decoder, dec_states)):
            scale_idx = self.num_scales - 1 - idx
            skip = skips[scale_idx]
            z, new_state = stage.forward_stream(z, skip, stage_state)
            new_dec_states.append(new_state)

        if self.query_variant == "none":
            decoded = self.decoder_to_freq(z)
        else:
            _runtime_assert(query_side is not None, "Query side-path was not produced by the compressor.")
            decoded = self.decoder_to_freq(z, query_side)
        y = self.out_proj(decoded)
        if self.masking:
            y = apply_source_gain_mask_4d(
                x,
                y,
                n_src=self.n_src,
                n_chan=self.n_chan,
                activation=self.mask_activation,
            )
        return y, (tuple(new_enc_states), tuple(new_dec_states))

    # -- state accounting ---------------------------------------------------

    def state_numel(self, batch_size: int = 1) -> int:
        state = self.init_stream_state(
            batch_size=batch_size,
            device=self.out_proj.weight.device,
            dtype=self.out_proj.weight.dtype,
        )
        return _tree_numel(state)

    def state_size_bytes(self, batch_size: int = 1, dtype: torch.dtype = torch.float16) -> int:
        return self.state_numel(batch_size=batch_size) * torch.tensor([], dtype=dtype).element_size()


def _tree_numel(tree) -> int:
    if isinstance(tree, torch.Tensor):
        return int(tree.numel())
    return sum(_tree_numel(item) for item in tree)


# ---------------------------------------------------------------------------
# Real-valued source-gain masking (unchanged from the previous revision)
# ---------------------------------------------------------------------------


def apply_source_gain_mask_4d(
    x: torch.Tensor,
    mask_logits: torch.Tensor,
    n_src: int,
    n_chan: int,
    *,
    activation: str = "sigmoid",
) -> torch.Tensor:
    """Apply real-valued source gains to packed complex input using 4D tensors only."""

    _runtime_assert(x.shape[1] == 2 * n_chan, f"{x.shape[1]} vs {2 * n_chan}")
    _runtime_assert(mask_logits.shape[1] == n_src * n_chan, f"{mask_logits.shape[1]} vs {n_src * n_chan}")
    if activation not in {"sigmoid", "softmax"}:
        raise ValueError(f"Unsupported activation={activation!r}; expected 'sigmoid' or 'softmax'.")

    gains = torch.sigmoid(mask_logits)
    softmax_gains = []
    if activation == "softmax":
        for chan_idx in range(n_chan):
            chan_logits = torch.cat(
                [
                    mask_logits[:, src_idx * n_chan + chan_idx : src_idx * n_chan + chan_idx + 1, :, :]
                    for src_idx in range(n_src)
                ],
                dim=1,
            )
            softmax_gains.append(torch.softmax(chan_logits, dim=1))

    outputs = []
    for src_idx in range(n_src):
        for chan_idx in range(n_chan):
            if activation == "sigmoid":
                gain = gains[:, src_idx * n_chan + chan_idx : src_idx * n_chan + chan_idx + 1, :, :]
            else:
                gain = softmax_gains[chan_idx][:, src_idx : src_idx + 1, :, :]
            real = x[:, 2 * chan_idx : 2 * chan_idx + 1, :, :] * gain
            imag = x[:, 2 * chan_idx + 1 : 2 * chan_idx + 2, :, :] * gain
            outputs.extend([real, imag])
    return torch.cat(outputs, dim=1)


class SourceAwareDolphinSFCNPUSeparator(DolphinSFCNPUSeparator):
    """
    DolphinSFCNPU variant with source-aware compressed-token refinement.

    The shared Dolphin/SFC trunk still operates on compressed K-band tokens.
    Speech/music are refined by separate stateless token refiners and predicted
    as explicit masks; the final source is reconstructed from the residual by
    default to preserve mixture consistency.
    """

    def __init__(
        self,
        *args,
        explicit_source_count: int | None = None,
        residual_source_index: int | None = None,
        source_refine_layers: int = 2,
        source_refine_freq_kernel: int = 5,
        source_refine_expansion: int = 2,
        source_head_type: str = "complex_residual",
        sfx_residual_mode: str = "residual",
        real_mask_scale: float = 1.0,
        imag_mask_scale: float = 0.12,
        **kwargs,
    ):
        if source_head_type not in {"real_residual", "complex_residual"}:
            raise ValueError(
                "source_head_type must be 'real_residual' or 'complex_residual', "
                f"got {source_head_type!r}."
            )
        if sfx_residual_mode not in {"residual", "gated_residual"}:
            raise ValueError(
                "sfx_residual_mode must be 'residual' or 'gated_residual', "
                f"got {sfx_residual_mode!r}."
            )
        super().__init__(*args, **kwargs)

        if self.n_src < 2:
            raise ValueError(f"Source-aware Dolphin requires at least two sources, got {self.n_src}.")
        if explicit_source_count is None:
            explicit_source_count = self.n_src - 1
        if explicit_source_count != self.n_src - 1:
            raise ValueError(
                "Source-aware Dolphin currently expects exactly one residual source; "
                f"got explicit_source_count={explicit_source_count}, n_src={self.n_src}."
            )
        if residual_source_index is None:
            residual_source_index = self.n_src - 1
        if residual_source_index != explicit_source_count:
            raise ValueError(
                "Source-aware Dolphin keeps explicit sources first and residual source last; "
                f"got residual_source_index={residual_source_index}, explicit_source_count={explicit_source_count}."
            )
        if real_mask_scale <= 0.0:
            raise ValueError(f"real_mask_scale must be positive, got {real_mask_scale}.")
        if imag_mask_scale < 0.0:
            raise ValueError(f"imag_mask_scale must be non-negative, got {imag_mask_scale}.")

        self.explicit_source_count = int(explicit_source_count)
        self.residual_source_index = int(residual_source_index)
        self.source_head_type = str(source_head_type)
        self.sfx_residual_mode = str(sfx_residual_mode)
        self.real_mask_scale = float(real_mask_scale)
        self.imag_mask_scale = float(imag_mask_scale)

        self.out_proj = nn.Identity()
        self.source_refiners = nn.ModuleList(
            [
                DolphinSourceTokenRefiner2d(
                    self.widths[0],
                    layers=source_refine_layers,
                    freq_kernel=source_refine_freq_kernel,
                    expansion=source_refine_expansion,
                )
                for _ in range(self.explicit_source_count)
            ]
        )
        head_channels = self.n_chan if self.source_head_type == "real_residual" else 2 * self.n_chan
        self.source_heads = nn.ModuleList(
            [
                nn.Conv2d(self.widths[0], head_channels, kernel_size=1, bias=True)
                for _ in range(self.explicit_source_count)
            ]
        )
        self.residual_gate_head = (
            nn.Conv2d(self.widths[0], self.n_chan, kernel_size=1, bias=True)
            if self.sfx_residual_mode == "gated_residual"
            else None
        )
        self._init_source_heads()

    def _init_source_heads(self) -> None:
        eps = 1.0e-4
        source_gain = min(max(1.0 / float(self.n_src), eps), 1.0 - eps)
        source_logit = math.log(source_gain / (1.0 - source_gain))
        with torch.no_grad():
            for head in self.source_heads:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
                if self.source_head_type == "real_residual":
                    head.bias.fill_(source_logit)
                else:
                    for chan_idx in range(self.n_chan):
                        head.bias[2 * chan_idx].fill_(source_logit)
                        head.bias[2 * chan_idx + 1].zero_()
            if self.residual_gate_head is not None:
                nn.init.zeros_(self.residual_gate_head.weight)
                nn.init.zeros_(self.residual_gate_head.bias)

    def _shared_tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        query_side = None
        compressed = self.compressor(self.in_proj(x))
        if self.query_variant == "none":
            z = compressed
        else:
            z, query_side = compressed

        skips: list[torch.Tensor] = []
        for stage in self.encoder:
            z, skip = stage(z)
            skips.append(skip)

        for idx, stage in enumerate(self.decoder):
            scale_idx = self.num_scales - 1 - idx
            z = stage(z, skips[scale_idx])
        return z, query_side

    def _shared_tokens_stream(
        self,
        x: torch.Tensor,
        state,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]]:
        enc_states, dec_states = state
        query_side = None
        compressed = self.compressor.forward_stream(self.in_proj(x))
        if self.query_variant == "none":
            z = compressed
        else:
            z, query_side = compressed

        skips: list[torch.Tensor] = []
        new_enc_states: list[tuple[torch.Tensor, ...]] = []
        for stage, stage_state in zip(self.encoder, enc_states):
            z, skip, new_state = stage.forward_stream(z, stage_state)
            skips.append(skip)
            new_enc_states.append(new_state)

        new_dec_states: list[tuple[torch.Tensor, ...]] = []
        for idx, (stage, stage_state) in enumerate(zip(self.decoder, dec_states)):
            scale_idx = self.num_scales - 1 - idx
            z, new_state = stage.forward_stream(z, skips[scale_idx], stage_state)
            new_dec_states.append(new_state)
        return z, query_side, (tuple(new_enc_states), tuple(new_dec_states))

    def _decode_tokens(self, tokens: torch.Tensor, query_side: torch.Tensor | None) -> torch.Tensor:
        if self.query_variant == "none":
            return self.decoder_to_freq(tokens)
        _runtime_assert(query_side is not None, "Query side-path was not produced by the compressor.")
        return self.decoder_to_freq(tokens, query_side)

    def _explicit_estimates(
        self,
        x: torch.Tensor,
        shared_tokens: torch.Tensor,
        query_side: torch.Tensor | None,
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        estimates: list[torch.Tensor] = []
        mask_logits: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        for refiner, head in zip(self.source_refiners, self.source_heads):
            decoded = self._decode_tokens(refiner(shared_tokens), query_side)
            logits = head(decoded)
            mask_logits.append(logits)
            if self.source_head_type == "real_residual":
                gain = torch.sigmoid(logits) * self.real_mask_scale
                masks.append(gain)
                parts = []
                for chan_idx in range(self.n_chan):
                    chan_gain = gain[:, chan_idx : chan_idx + 1]
                    real = x[:, 2 * chan_idx : 2 * chan_idx + 1] * chan_gain
                    imag = x[:, 2 * chan_idx + 1 : 2 * chan_idx + 2] * chan_gain
                    parts.extend([real, imag])
                estimates.append(torch.cat(parts, dim=1))
            else:
                parts = []
                mask_parts = []
                for chan_idx in range(self.n_chan):
                    real_logit = logits[:, 2 * chan_idx : 2 * chan_idx + 1]
                    imag_logit = logits[:, 2 * chan_idx + 1 : 2 * chan_idx + 2]
                    real_mask = torch.sigmoid(real_logit) * self.real_mask_scale
                    imag_mask = torch.tanh(imag_logit) * self.imag_mask_scale
                    mix_real = x[:, 2 * chan_idx : 2 * chan_idx + 1]
                    mix_imag = x[:, 2 * chan_idx + 1 : 2 * chan_idx + 2]
                    parts.extend(
                        [
                            mix_real * real_mask - mix_imag * imag_mask,
                            mix_imag * real_mask + mix_real * imag_mask,
                        ]
                    )
                    mask_parts.extend([real_mask, imag_mask])
                estimates.append(torch.cat(parts, dim=1))
                masks.append(torch.cat(mask_parts, dim=1))
        return estimates, torch.cat(masks, dim=1), torch.cat(mask_logits, dim=1), shared_tokens

    def _residual_estimate(
        self,
        x: torch.Tensor,
        explicit_estimates: Sequence[torch.Tensor],
        shared_tokens: torch.Tensor,
        query_side: torch.Tensor | None,
    ) -> torch.Tensor:
        residual_parts = []
        for chan_idx in range(self.n_chan):
            real = x[:, 2 * chan_idx : 2 * chan_idx + 1]
            imag = x[:, 2 * chan_idx + 1 : 2 * chan_idx + 2]
            for estimate in explicit_estimates:
                real = real - estimate[:, 2 * chan_idx : 2 * chan_idx + 1]
                imag = imag - estimate[:, 2 * chan_idx + 1 : 2 * chan_idx + 2]
            residual_parts.extend([real, imag])
        residual = torch.cat(residual_parts, dim=1)
        if self.residual_gate_head is None:
            return residual
        decoded = self._decode_tokens(shared_tokens, query_side)
        gate = torch.sigmoid(self.residual_gate_head(decoded))
        gated_parts = []
        for chan_idx in range(self.n_chan):
            chan_gate = gate[:, chan_idx : chan_idx + 1]
            gated_parts.extend(
                [
                    residual[:, 2 * chan_idx : 2 * chan_idx + 1] * chan_gate,
                    residual[:, 2 * chan_idx + 1 : 2 * chan_idx + 2] * chan_gate,
                ]
            )
        return torch.cat(gated_parts, dim=1)

    def _assemble_output(
        self,
        x: torch.Tensor,
        shared_tokens: torch.Tensor,
        query_side: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        explicit_estimates, mask, mask_logits, shared_tokens = self._explicit_estimates(x, shared_tokens, query_side)
        residual = self._residual_estimate(x, explicit_estimates, shared_tokens, query_side)
        y = torch.cat([*explicit_estimates, residual], dim=1)
        aux = {
            "mask": mask,
            "mask_domain": "packed_complex_mask" if self.source_head_type == "complex_residual" else "real_gain_mask",
            "mask_logits": mask_logits,
            "mask_logits_domain": (
                "source_aware_dolphin_complex_mask_logits"
                if self.source_head_type == "complex_residual"
                else "source_aware_dolphin_real_gain_logits"
            ),
            "mask_logits_transform": (
                "sigmoid_tanh_complex_mask"
                if self.source_head_type == "complex_residual"
                else "sigmoid_real_gain_mask"
            ),
            "mask_logits_real_scale": self.real_mask_scale,
            "mask_logits_imag_scale": self.imag_mask_scale,
            "explicit_source_count": self.explicit_source_count,
            "residual_source_index": self.residual_source_index,
            "sfx_residual_mode": self.sfx_residual_mode,
        }
        return y, aux

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        shared_tokens, query_side = self._shared_tokens(x)
        y, aux = self._assemble_output(x, shared_tokens, query_side)
        if return_aux:
            return y, aux
        return y

    def forward_stream(self, x: torch.Tensor, state=None, return_aux: bool = False):
        _runtime_assert(x.ndim == 4, f"Expected (B,C,T,F), got {x.shape}")
        _runtime_assert(x.shape[2] == 1, "forward_stream expects one frame at a time.")
        if state is None:
            state = self.init_stream_state(batch_size=x.shape[0], device=x.device, dtype=x.dtype)
        shared_tokens, query_side, new_state = self._shared_tokens_stream(x, state)
        y, aux = self._assemble_output(x, shared_tokens, query_side)
        if return_aux:
            return y, new_state, aux
        return y, new_state

    def state_numel(self, batch_size: int = 1) -> int:
        parameter = next(self.parameters())
        state = self.init_stream_state(batch_size=batch_size, device=parameter.device, dtype=parameter.dtype)
        return _tree_numel(state)


# ---------------------------------------------------------------------------
# Packed-state ONNX export wrapper (unchanged contract, smaller leaf count)
# ---------------------------------------------------------------------------


class DolphinSFCNPUStreamingExportWrapper(nn.Module):
    """
    ONNX export wrapper that collapses the nested streaming-state tree into a
    single packed 2-D tensor.  See AGENT.md rule 14 (small input/output count)
    and rule 13 (tight DSP quota).

    The underlying slim separator already has far fewer per-block caches than
    the previous generation, so the packed state here is genuinely smaller —
    not just fewer ONNX edges.  The wrapper itself is unchanged: one Slice +
    Reshape per leaf at unpack, per-leaf Flatten + one Concat at pack.
    """

    def __init__(self, core: DolphinSFCNPUSeparator, batch_size: int = 1, dtype: torch.dtype = torch.float32):
        super().__init__()
        from spectral_feature_compression.utils.onnx_streaming import flatten_tensor_tree

        self.core = core
        self.batch_size = batch_size
        example_state = core.init_stream_state(batch_size=batch_size, dtype=dtype)
        flat_state, state_spec = flatten_tensor_tree(example_state)
        self.state_spec = state_spec
        self.state_tensor_count = len(flat_state)

        per_shapes: list[tuple[int, ...]] = []
        per_numels: list[int] = []
        for tensor in flat_state:
            if tensor.shape[0] != batch_size:
                raise ValueError(
                    f"All leaf state tensors must start with batch dim {batch_size}; got {tuple(tensor.shape)}."
                )
            shape_wo_batch = tuple(int(d) for d in tensor.shape[1:])
            numel = int(tensor.numel() // max(batch_size, 1))
            if numel == 0:
                raise ValueError(
                    "DolphinSFCNPUStreamingExportWrapper does not support zero-sized leaf "
                    f"state tensors (shape {tuple(tensor.shape)}); such tensors carry no "
                    "information and break the static `Reshape(-1, ...)` used during unpack. "
                    "Drop them from the streaming state tree instead."
                )
            per_shapes.append(shape_wo_batch)
            per_numels.append(numel)
        self.per_shapes: tuple[tuple[int, ...], ...] = tuple(per_shapes)
        self.per_numels: tuple[int, ...] = tuple(per_numels)
        self.total_numel: int = sum(per_numels)

    # -- internal helpers ----------------------------------------------------

    def _unpack_state(self, packed: torch.Tensor) -> tuple[torch.Tensor, ...]:
        from spectral_feature_compression.utils.onnx_streaming import unflatten_tensor_tree

        _runtime_assert(
            packed.ndim == 2 and int(packed.shape[1]) == self.total_numel,
            f"Expected packed state shape (B, {self.total_numel}), got {tuple(packed.shape)}",
        )
        leaves: list[torch.Tensor] = []
        offset = 0
        for numel, shape in zip(self.per_numels, self.per_shapes):
            chunk = packed[:, offset : offset + numel]
            leaves.append(chunk.reshape((-1,) + shape))
            offset += numel
        return unflatten_tensor_tree(tuple(leaves), self.state_spec)

    def _pack_state(self, state_tree) -> torch.Tensor:
        from spectral_feature_compression.utils.onnx_streaming import flatten_tensor_tree

        flat, _ = flatten_tensor_tree(state_tree)
        _runtime_assert(
            len(flat) == self.state_tensor_count,
            f"State tree has {len(flat)} leaves but {self.state_tensor_count} were expected.",
        )
        flat_2d = [torch.flatten(t, start_dim=1) for t in flat]
        return torch.cat(flat_2d, dim=1)

    # -- public API ----------------------------------------------------------

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state_tree = self._unpack_state(state)
        y, new_state_tree = self.core.forward_stream(x, state_tree)
        packed_new_state = self._pack_state(new_state_tree)
        return y, packed_new_state

    def init_packed_state(
        self,
        batch_size: int | None = None,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        b = batch_size if batch_size is not None else self.batch_size
        state_tree = self.core.init_stream_state(batch_size=b, device=device, dtype=dtype)
        return self._pack_state(state_tree)

    def pack_state(self, state_tree) -> torch.Tensor:
        return self._pack_state(state_tree)

    def unpack_state(self, packed: torch.Tensor):
        return self._unpack_state(packed)


# ---------------------------------------------------------------------------
# Presets targeting the 3-8M parameter window with <192 KB state at fp16
# ---------------------------------------------------------------------------


_PRESETS: dict[str, dict[str, object]] = {
    # Tiny smoke/export target; not for quality.
    "edge_small": dict(
        n_bands=32,
        d_model=16,
        num_scales=3,
        widths=(16, 32, 64),
        blocks_per_scale=(1, 1, 1),
        time_kernels=(3, 3, 3),
        freq_kernels=(3, 3, 3),
    ),
    # ~3.6M params, ~144 KiB fp16 state at n_freq=257.
    "slim_4m": dict(
        n_bands=48,
        d_model=128,
        num_scales=3,
        widths=(128, 192, 256),
        blocks_per_scale=(1, 2, 1),
        time_kernels=(3, 3, 3),
        freq_kernels=(3, 3, 3),
    ),
    # ~5.0M params, ~162 KiB fp16 state at n_freq=257.
    "slim_6m": dict(
        n_bands=48,
        d_model=128,
        num_scales=3,
        widths=(128, 224, 320),
        blocks_per_scale=(1, 2, 1),
        time_kernels=(3, 3, 3),
        freq_kernels=(3, 3, 3),
    ),
    # ~6.5M params, ~174 KiB fp16 state at n_freq=257.
    "slim_8m": dict(
        n_bands=48,
        d_model=128,
        num_scales=3,
        widths=(128, 240, 384),
        blocks_per_scale=(1, 2, 1),
        time_kernels=(3, 3, 3),
        freq_kernels=(3, 3, 3),
    ),
}

_PRESETS["large_6m"] = dict(_PRESETS["slim_6m"])
_PRESETS["large_8m"] = dict(_PRESETS["slim_8m"])

# Query variants are named presets for recipe convenience.  They intentionally
# reuse the same width/depth/state profile; only the compressor/decoder contract
# changes via ``query_variant``.
for _base_name in ("edge_small", "slim_4m", "slim_6m", "slim_8m", "large_6m", "large_8m"):
    _PRESETS[f"{_base_name}_soft_query"] = dict(_PRESETS[_base_name], query_variant="soft_band_query")
    _PRESETS[f"{_base_name}_soft_band_query"] = dict(_PRESETS[_base_name], query_variant="soft_band_query")
    _PRESETS[f"{_base_name}_crossattn_query"] = dict(_PRESETS[_base_name], query_variant="crossattn_query")

_PRESETS["source_aware_6m"] = dict(
    _PRESETS["slim_6m"],
    n_bands=56,
    query_variant="soft_band_query",
    source_aware=True,
    explicit_source_count=2,
    source_refine_layers=2,
    source_refine_freq_kernel=5,
    source_refine_expansion=2,
    source_head_type="complex_residual",
    sfx_residual_mode="residual",
)
_PRESETS["source_aware_6m_gated_sfx"] = dict(
    _PRESETS["source_aware_6m"],
    sfx_residual_mode="gated_residual",
)
_PRESETS["source_aware_6m_crossattn"] = dict(
    _PRESETS["source_aware_6m"],
    query_variant="crossattn_query",
)


def build_dolphin_sfc_npu_from_config(
    *,
    preset: str,
    n_freq: int,
    n_fft: int | None = None,
    sample_rate: int | None = None,
    n_src: int = 3,
    n_chan: int = 1,
    band_config: str = "musical",
    masking: bool = True,
    mask_activation: str = "sigmoid",
    query_variant: str | None = None,
    query_type: str = "adaptive",
    n_bands: int | None = None,
    d_model: int | None = None,
    num_scales: int | None = None,
    widths: tuple[int, ...] | list[int] | None = None,
    blocks_per_scale: tuple[int, ...] | list[int] | None = None,
    time_kernels: tuple[int, ...] | list[int] | None = None,
    freq_kernels: tuple[int, ...] | list[int] | None = None,
    compressor_freq_kernel: int | None = None,
    ffn_expansion: int | None = None,
    source_aware: bool | None = None,
    explicit_source_count: int | None = None,
    residual_source_index: int | None = None,
    source_refine_layers: int | None = None,
    source_refine_freq_kernel: int | None = None,
    source_refine_expansion: int | None = None,
    source_head_type: str | None = None,
    sfx_residual_mode: str | None = None,
    real_mask_scale: float | None = None,
    imag_mask_scale: float | None = None,
) -> DolphinSFCNPUSeparator:
    if preset not in _PRESETS:
        names = ", ".join(sorted(_PRESETS))
        raise ValueError(f"Unknown DolphinSFCNPU preset {preset!r}. Available presets: {names}")

    cfg = deepcopy(_PRESETS[preset])
    preset_source_aware = bool(cfg.pop("source_aware", False))
    if source_aware is None:
        source_aware = preset_source_aware
    preset_query_variant = cfg.pop("query_variant", "none")
    if query_variant is None:
        query_variant = str(preset_query_variant)
    source_cfg = {
        "explicit_source_count": cfg.pop("explicit_source_count", None),
        "residual_source_index": cfg.pop("residual_source_index", None),
        "source_refine_layers": cfg.pop("source_refine_layers", 2),
        "source_refine_freq_kernel": cfg.pop("source_refine_freq_kernel", 5),
        "source_refine_expansion": cfg.pop("source_refine_expansion", 2),
        "source_head_type": cfg.pop("source_head_type", "complex_residual"),
        "sfx_residual_mode": cfg.pop("sfx_residual_mode", "residual"),
        "real_mask_scale": cfg.pop("real_mask_scale", 1.0),
        "imag_mask_scale": cfg.pop("imag_mask_scale", 0.12),
    }
    source_overrides = {
        "explicit_source_count": explicit_source_count,
        "residual_source_index": residual_source_index,
        "source_refine_layers": source_refine_layers,
        "source_refine_freq_kernel": source_refine_freq_kernel,
        "source_refine_expansion": source_refine_expansion,
        "source_head_type": source_head_type,
        "sfx_residual_mode": sfx_residual_mode,
        "real_mask_scale": real_mask_scale,
        "imag_mask_scale": imag_mask_scale,
    }
    for key, value in source_overrides.items():
        if value is not None:
            source_cfg[key] = value
    overrides = {
        "n_bands": n_bands,
        "d_model": d_model,
        "num_scales": num_scales,
        "widths": tuple(widths) if widths is not None else None,
        "blocks_per_scale": tuple(blocks_per_scale) if blocks_per_scale is not None else None,
        "time_kernels": tuple(time_kernels) if time_kernels is not None else None,
        "freq_kernels": tuple(freq_kernels) if freq_kernels is not None else None,
        "compressor_freq_kernel": compressor_freq_kernel,
        "ffn_expansion": ffn_expansion,
    }
    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value
    if source_aware and not masking:
        raise ValueError("Source-aware Dolphin currently requires masking=True.")
    separator_cls = SourceAwareDolphinSFCNPUSeparator if source_aware else DolphinSFCNPUSeparator
    return separator_cls(
        n_freq=n_freq,
        n_fft=n_fft,
        sample_rate=sample_rate,
        band_config=band_config,
        n_src=n_src,
        n_chan=n_chan,
        masking=masking,
        mask_activation=mask_activation,
        query_variant=query_variant,
        query_type=query_type,
        **(source_cfg if source_aware else {}),
        **cfg,  # type: ignore[arg-type]
    )


def build_dolphin_sfc_npu_preset(
    preset: str,
    *,
    n_freq: int,
    n_fft: int | None = None,
    sample_rate: int | None = None,
    n_src: int = 3,
    n_chan: int = 1,
    band_config: str = "musical",
    masking: bool = True,
    mask_activation: str = "sigmoid",
    query_variant: str | None = None,
    query_type: str = "adaptive",
    source_aware: bool | None = None,
) -> DolphinSFCNPUSeparator:
    """
    Build a named DolphinSFCNPU configuration.

    - ``edge_small``: tiny smoke/export model, not for quality.
    - ``slim_4m``, ``slim_6m``, ``slim_8m``: 3-8M parameter range, designed to
      stay under the 192 KiB streaming-state budget at fp16 with batch=1 while
      offering useful separation capacity at the bottleneck.
    - append ``_soft_query`` or ``_crossattn_query`` to any preset name to use
      the new explicit-query compressor/decoder variants.  The same effect can
      be requested with ``query_variant=...``.
    """

    return build_dolphin_sfc_npu_from_config(
        preset=preset,
        n_freq=n_freq,
        n_fft=n_fft,
        sample_rate=sample_rate,
        n_src=n_src,
        n_chan=n_chan,
        band_config=band_config,
        masking=masking,
        mask_activation=mask_activation,
        query_variant=query_variant,
        query_type=query_type,
        source_aware=source_aware,
    )
