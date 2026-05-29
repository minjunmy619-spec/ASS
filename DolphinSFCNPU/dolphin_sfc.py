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
        masking: bool = True,
    ):
        super().__init__()
        _validate_even_pyramid(n_bands, num_scales)
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
        self.masking = masking

        self.band_spec = self._build_band_spec(
            n_freq=n_freq,
            n_bands=n_bands,
            n_fft=n_fft,
            sample_rate=sample_rate,
            band_config=band_config,
        )
        self.in_proj = nn.Sequential(nn.Conv2d(2 * n_chan, d_model, kernel_size=1), RMSNorm2d(d_model))
        self.compressor = StatelessBandCompressor2d(d_model, self.band_spec, freq_kernel=3)

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
                )
            )
        self.decoder = nn.ModuleList(decoder_stages)

        self.decoder_to_freq = SpectralDecoder2d(self.widths[0], self.band_spec)
        out_ch = n_src * n_chan if masking else 2 * n_src * n_chan
        self.out_proj = nn.Conv2d(self.widths[0], out_ch, kernel_size=1)

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
        z = self.compressor(self.in_proj(x))

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

        y = self.out_proj(self.decoder_to_freq(z))
        if self.masking:
            y = apply_source_gain_mask_4d(x, y, n_src=self.n_src, n_chan=self.n_chan)
        return y

    # -- streaming ----------------------------------------------------------

    def init_stream_state(self, batch_size: int = 1, *, device=None, dtype=None):
        enc_states: list[tuple[torch.Tensor, ...]] = []
        bands = self.n_bands
        for idx, stage in enumerate(self.encoder):
            enc_states.append(
                stage.init_stream_state(batch_size, freq_bins=bands, device=device, dtype=dtype)
            )
            if idx < self.num_scales - 1:
                bands = bands // 2

        dec_states: list[tuple[torch.Tensor, ...]] = []
        # Deepest stage first, mirroring forward_stream order.
        for idx, stage in enumerate(self.decoder):
            dec_states.append(
                stage.init_stream_state(batch_size, freq_bins=bands, device=device, dtype=dtype)
            )
            if idx < self.num_scales - 1:
                bands = bands * 2

        return (tuple(enc_states), tuple(dec_states))

    def forward_stream(self, x: torch.Tensor, state=None):
        _runtime_assert(x.ndim == 4, f"Expected (B,C,T,F), got {x.shape}")
        _runtime_assert(x.shape[2] == 1, "forward_stream expects one frame at a time.")
        if state is None:
            state = self.init_stream_state(batch_size=x.shape[0], device=x.device, dtype=x.dtype)

        enc_states, dec_states = state
        z = self.compressor.forward_stream(self.in_proj(x))

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

        y = self.out_proj(self.decoder_to_freq(z))
        if self.masking:
            y = apply_source_gain_mask_4d(x, y, n_src=self.n_src, n_chan=self.n_chan)
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


def apply_source_gain_mask_4d(x: torch.Tensor, mask_logits: torch.Tensor, n_src: int, n_chan: int) -> torch.Tensor:
    """Apply real-valued source gains to packed complex input using 4D tensors only."""

    _runtime_assert(x.shape[1] == 2 * n_chan, f"{x.shape[1]} vs {2 * n_chan}")
    _runtime_assert(mask_logits.shape[1] == n_src * n_chan, f"{mask_logits.shape[1]} vs {n_src * n_chan}")
    gains = torch.sigmoid(mask_logits)
    outputs = []
    for src_idx in range(n_src):
        for chan_idx in range(n_chan):
            gain = gains[:, src_idx * n_chan + chan_idx : src_idx * n_chan + chan_idx + 1, :, :]
            real = x[:, 2 * chan_idx : 2 * chan_idx + 1, :, :] * gain
            imag = x[:, 2 * chan_idx + 1 : 2 * chan_idx + 2, :, :] * gain
            outputs.extend([real, imag])
    return torch.cat(outputs, dim=1)


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
                    f"All leaf state tensors must start with batch dim {batch_size}; "
                    f"got {tuple(tensor.shape)}."
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
) -> DolphinSFCNPUSeparator:
    """
    Build a named DolphinSFCNPU configuration.

    - ``edge_small``: tiny smoke/export model, not for quality.
    - ``slim_4m``, ``slim_6m``, ``slim_8m``: 3-8M parameter range, designed to
      stay under the 192 KiB streaming-state budget at fp16 with batch=1 while
      offering useful separation capacity at the bottleneck.
    """

    if preset not in _PRESETS:
        names = ", ".join(sorted(_PRESETS))
        raise ValueError(f"Unknown DolphinSFCNPU preset {preset!r}. Available presets: {names}")

    cfg = dict(_PRESETS[preset])
    return DolphinSFCNPUSeparator(
        n_freq=n_freq,
        n_fft=n_fft,
        sample_rate=sample_rate,
        band_config=band_config,
        n_src=n_src,
        n_chan=n_chan,
        masking=masking,
        **cfg,  # type: ignore[arg-type]
    )
