"""Band-split dual-path separator for causal TV-NPU streaming (BandDualPathNPU).

Design summary
--------------

The model predicts complex masks for ``n_src`` stems (speech / music / effects)
from a single-frame packed STFT and is built for the ONE -> NPU flow:

* **Host-side band layout.** The host (DSP) already computes the STFT. It also
  applies power-law magnitude compression and lays the spectrum out as a few
  "band regions".  Region ``r`` covers bins ``[start, end)`` split into
  ``K_r`` bands of ``W_r`` bins, delivered as ``[B, 2*M*W_r, T, K_r]``: the
  bins of one band sit on the channel axis.  The NPU graph therefore starts
  with a plain 1x1 Conv2d (a BSRNN band-split projection) instead of a
  full-resolution cross-attention, pyramid, Slice or Gather.
* **Time path: per-band GRU.**  Each block has a causal GRU shared across bands
  (narrow-band modelling).  Training uses ``torch.nn.GRU`` (cuDNN); the export
  path evaluates the *same weights* as six 1x1 Conv2d plus sigmoid/tanh, so the
  streaming state is only ``[B, H, 1, K]`` per block while the temporal
  receptive field is unbounded (a conv stack with kernel 2-3 only sees
  ~100-200 ms).
* **Global scene path: band-pooled GRU + FiLM.**  Band-split trunks spend
  ``K`` MACs per weight, so their useful parameter count is capped by the MAC
  budget.  A GRU on the band-mean of the frame (``[B, C, 1, 1]``) with a wide
  hidden state models long-term scene context (commentary vs concert vs
  cinematic) and modulates all bands with FiLM, adding ~1M parameters for
  ~0.1 GMAC/s.
* **Frequency path: in-frame band attention + depthwise ConvGLU.**  Attention
  runs across the ``K`` band tokens of the *current frame* (global harmonic
  context, causal by construction).  A depthwise ``(1, k)`` conv plus a gated
  1x1 FFN adds local cross-band mixing.
* **Quantization-friendly numerics.**  RMSNorm over channels per (frame, band)
  position is written as ``x / sqrt(mean(x*x) + eps)`` so ONE's
  ``transform_sqrt_div_to_rsqrt_mul`` + ``fuse_rmsnorm`` can turn it into one
  ``RmsNorm`` op; the norm gains are folded into the consuming convolutions for
  export.  No BatchNorm folding (the root cause of the int8/int16 collapse in
  ``NPU_QUANT_ISSUE.md``), bounded inputs (power-law compression), bounded GRU
  state (tanh) and bounded masks (``mask_bound * tanh``).
* **Compact ABI.**  Inputs: one tensor per band region, one packed band-GRU
  state ``[B, H, n_blocks, K]`` and one packed scene-GRU state
  ``[B, H_s, n_scene, 1]``; outputs: one mask tensor per region plus the two
  next states.  All tensors are 4D with batch first, no Tile/Expand/Gather/
  ConstantOfShape, no BatchNorm, no scalar-constant multiplies.

Shapes use ``[B, C, T, K]`` (channels, frames, bands) throughout.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import autocast
import torch.nn as nn
import torch.nn.functional as F

# (start_bin, end_bin, bins_per_band) for n_fft=2048 at 44.1 kHz (21.5 Hz bins):
#   0.00-2.07 kHz : 24 bands x  86 Hz  (speech F0/formants, bass, pitch)
#   2.07-7.58 kHz : 16 bands x 345 Hz  (consonants, presence, attacks)
#   7.58-22.05 kHz:  8 bands x 1.81 kHz (air, cymbals, noise-like effects)
# 48 band tokens in total.  Bin 1024 (Nyquist) reuses the mask of bin 1023.
DEFAULT_REGIONS: tuple[tuple[int, int, int], ...] = ((0, 96, 4), (96, 352, 16), (352, 1024, 84))


def _check_span(kernel_size: int, dilation: int = 1, *, name: str) -> None:
    span = (int(kernel_size) - 1) * int(dilation)
    if span > 14:
        raise ValueError(f"{name}: (kernel_size - 1) * dilation = {span} violates the NPU limit of 14")


@dataclass(frozen=True)
class BandRegion:
    start: int
    end: int
    width: int

    @property
    def n_bands(self) -> int:
        return (self.end - self.start) // self.width


class BandLayout:
    """Host-side mapping between full STFT frames and band-region tensors.

    ``pack`` / ``unpack`` are the exact reference for the host implementation.
    Channel index inside a region tensor is ``c * W + w`` where ``c`` is the
    packed input channel (``m*2 + re/im``) and ``w`` the bin inside the band.
    """

    def __init__(self, n_freq: int, regions: Sequence[Sequence[int]] = DEFAULT_REGIONS) -> None:
        self.n_freq = int(n_freq)
        parsed = tuple(BandRegion(int(s), int(e), int(w)) for s, e, w in regions)
        if not parsed:
            raise ValueError("At least one band region is required")
        expected_start = 0
        for region in parsed:
            if region.start != expected_start:
                raise ValueError(f"Band regions must be contiguous from bin 0, got {parsed}")
            if region.width <= 0 or region.end <= region.start:
                raise ValueError(f"Invalid band region {region}")
            if (region.end - region.start) % region.width != 0:
                raise ValueError(f"Region {region} is not divisible into {region.width}-bin bands")
            expected_start = region.end
        tail = self.n_freq - expected_start
        if tail not in (0, 1):
            raise ValueError(
                f"Band regions end at bin {expected_start}; only a single Nyquist tail bin may be left "
                f"uncovered for n_freq={self.n_freq}"
            )
        self.regions = parsed
        self.covered_bins = expected_start
        self.tail_bins = tail

    @property
    def n_bands(self) -> int:
        return sum(region.n_bands for region in self.regions)

    @property
    def band_counts(self) -> tuple[int, ...]:
        return tuple(region.n_bands for region in self.regions)

    def pack(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """``[B, C, T, F]`` -> tuple of ``[B, C*W_r, T, K_r]``."""
        bsz, channels, n_frames, n_freq = x.shape
        if n_freq != self.n_freq:
            raise ValueError(f"Expected {self.n_freq} frequency bins, got {n_freq}")
        parts = []
        for region in self.regions:
            chunk = x[..., region.start : region.end]
            chunk = chunk.reshape(bsz, channels, n_frames, region.n_bands, region.width)
            chunk = chunk.permute(0, 1, 4, 2, 3).reshape(bsz, channels * region.width, n_frames, region.n_bands)
            parts.append(chunk)
        return tuple(parts)

    def unpack(self, parts: Sequence[torch.Tensor], channels: int) -> torch.Tensor:
        """Inverse of ``pack`` for ``channels`` output channels; replicates the last bin into the tail."""
        pieces = []
        for region, part in zip(self.regions, parts, strict=True):
            bsz, _, n_frames, n_bands = part.shape
            piece = part.reshape(bsz, channels, region.width, n_frames, n_bands)
            piece = piece.permute(0, 1, 3, 4, 2).reshape(bsz, channels, n_frames, n_bands * region.width)
            pieces.append(piece)
        if self.tail_bins:
            pieces.append(pieces[-1][..., -1:].expand(*pieces[-1].shape[:-1], self.tail_bins))
        return torch.cat(pieces, dim=-1)


def compress_packed_spectrum(x: torch.Tensor, exponent: float, eps: float = 1e-12) -> torch.Tensor:
    """Power-law magnitude compression of a packed ``[B, 2*M, T, F]`` spectrum (host-side).

    Each (re, im) pair is scaled by ``|X|^(exponent - 1)`` so the phase is kept
    and the magnitude becomes ``|X|^exponent``.
    """
    if exponent == 1.0:
        return x
    bsz, channels, n_frames, n_freq = x.shape
    pairs = x.reshape(bsz, channels // 2, 2, n_frames, n_freq)
    power = pairs.square().sum(dim=2, keepdim=True)
    scale = (power + eps).pow(0.5 * (exponent - 1.0))
    return (pairs * scale).reshape(bsz, channels, n_frames, n_freq)


class RMSNorm2d(nn.Module):
    """Channel RMSNorm per (frame, band) position in the ONE ``fuse_rmsnorm`` form."""

    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.affine = True
        # "mean": ReduceMean over channels (ONE fuse_rmsnorm pattern).
        # "conv": fixed 1x1 averaging Conv2d, a fallback for backends without a fast MEAN.
        self.reduce = "mean"
        self.register_buffer("avg_weight", torch.full((1, channels, 1, 1), 1.0 / channels), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        low_precision = in_dtype in (torch.float16, torch.bfloat16)
        if low_precision:
            x = x.float()
        if self.reduce == "conv":
            mean_square = F.conv2d(x * x, self.avg_weight.to(x.dtype))
        else:
            mean_square = (x * x).mean(dim=1, keepdim=True)
        y = x / torch.sqrt(mean_square + self.eps)
        if self.affine:
            y = y * self.weight
        return y.to(in_dtype) if low_precision else y

    @torch.no_grad()
    def take_gain(self) -> torch.Tensor:
        """Return the per-channel gain and turn it into identity (for folding into consumers)."""
        gain = self.weight.detach().reshape(-1).clone()
        self.weight.fill_(1.0)
        self.affine = False
        return gain


@torch.no_grad()
def _fold_gain_into_conv_input(conv: nn.Conv2d, gain: torch.Tensor) -> None:
    if conv.groups == 1:
        conv.weight.mul_(gain.view(1, -1, 1, 1))
    elif conv.groups == conv.in_channels == conv.out_channels:
        conv.weight.mul_(gain.view(-1, 1, 1, 1))
    else:
        raise ValueError("Only dense or depthwise convolutions can absorb an input gain")


def _gru_step_conv(gru: nn.GRU, y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """One ``nn.GRU`` step evaluated with 1x1 Conv2d on ``[B, C, 1, N]`` tensors (no Split/Concat)."""
    hid = gru.hidden_size
    w_ih, w_hh = gru.weight_ih_l0, gru.weight_hh_l0
    b_ih, b_hh = gru.bias_ih_l0, gru.bias_hh_l0

    def gate(weight: torch.Tensor, index: int) -> torch.Tensor:
        return weight[index * hid : (index + 1) * hid].reshape(hid, -1, 1, 1)

    reset = torch.sigmoid(F.conv2d(y, gate(w_ih, 0), b_ih[:hid] + b_hh[:hid]) + F.conv2d(h, gate(w_hh, 0)))
    update = torch.sigmoid(
        F.conv2d(y, gate(w_ih, 1), b_ih[hid : 2 * hid] + b_hh[hid : 2 * hid]) + F.conv2d(h, gate(w_hh, 1))
    )
    candidate = torch.tanh(
        F.conv2d(y, gate(w_ih, 2), b_ih[2 * hid :]) + reset * F.conv2d(h, gate(w_hh, 2), b_hh[2 * hid :])
    )
    return candidate + update * (h - candidate)


def _gru_sequence(gru: nn.GRU, y: torch.Tensor, h0: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    """Run ``gru`` over time independently for every column of ``y`` ``[B, C, T, N]``.

    Returns the output ``[B, H, T, N]`` and the last hidden state ``[B, H, 1, N]``.
    """
    bsz, channels, n_frames, n_cols = y.shape
    hid = gru.hidden_size
    seq = y.permute(0, 3, 2, 1).reshape(bsz * n_cols, n_frames, channels)
    h = None
    if h0 is not None:
        h = h0.permute(2, 0, 3, 1).reshape(1, bsz * n_cols, hid).float().contiguous()
    # cuDNN GRU under bf16 autocast is not reliable; keep the recurrence in fp32.
    with autocast(device_type=y.device.type, enabled=False):
        out, h_last = gru(seq.float(), h)
    out = out.to(y.dtype).reshape(bsz, n_cols, n_frames, hid).permute(0, 3, 2, 1)
    h_last = h_last.reshape(bsz, n_cols, hid, 1).permute(0, 2, 3, 1).to(y.dtype)
    return out, h_last


class BandGRU(nn.Module):
    """Pre-norm causal GRU over time for every band (weights shared across bands)."""

    def __init__(self, channels: int, hidden: int, eps: float) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.norm = RMSNorm2d(channels, eps)
        self.gru = nn.GRU(channels, hidden, batch_first=True)
        self.proj = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        out, h_last = _gru_sequence(self.gru, self.norm(x), h0)
        return x + self.proj(out), h_last

    def step(self, x: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One frame (export path). ``x``: ``[B, C, 1, K]``, ``h``: ``[B, H, 1, K]``."""
        h_new = _gru_step_conv(self.gru, self.norm(x), h)
        return x + self.proj(h_new), h_new

    @torch.no_grad()
    def fold_norm_(self) -> None:
        self.gru.weight_ih_l0.mul_(self.norm.take_gain().view(1, -1))


class SceneGRU(nn.Module):
    """Global scene memory: GRU on the band-mean of each frame, FiLM on every band.

    ``x <- x + x * gamma(h) + beta(h)``; FiLM convs start at zero (identity).
    """

    def __init__(self, channels: int, hidden: int, eps: float) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.norm = RMSNorm2d(channels, eps)
        self.gru = nn.GRU(channels, hidden, batch_first=True)
        self.gamma = nn.Conv2d(hidden, channels, kernel_size=1)
        self.beta = nn.Conv2d(hidden, channels, kernel_size=1)
        for conv in (self.gamma, self.beta):
            nn.init.zeros_(conv.weight)
            nn.init.zeros_(conv.bias)

    def _summary(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        if self.norm.reduce != "conv":
            return y.mean(dim=3, keepdim=True)
        # MEAN-free fallback: staged AvgPool2d with legal kernels (<= 15) and strides (4, then 1).
        n_bands = y.shape[3]
        while n_bands > 15 and n_bands % 4 == 0:
            y = F.avg_pool2d(y, kernel_size=(1, 4), stride=(1, 4))
            n_bands //= 4
        if n_bands > 15:
            raise ValueError(f"Cannot average {y.shape[3]} bands with legal AvgPool2d stages")
        return F.avg_pool2d(y, kernel_size=(1, n_bands), stride=(1, 1)) if n_bands > 1 else y

    def _film(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return x + x * self.gamma(h) + self.beta(h)

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        out, h_last = _gru_sequence(self.gru, self._summary(x), h0)
        return self._film(x, out), h_last

    def step(self, x: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h_new = _gru_step_conv(self.gru, self._summary(x), h)
        return self._film(x, h_new), h_new

    @torch.no_grad()
    def fold_norm_(self) -> None:
        # mean over bands is linear, so the per-channel gain commutes with it.
        self.gru.weight_ih_l0.mul_(self.norm.take_gain().view(1, -1))


class BandAttention(nn.Module):
    """Pre-norm multi-head self-attention across the band tokens of each frame."""

    def __init__(self, channels: int, heads: int, qk_dim: int, eps: float) -> None:
        super().__init__()
        if channels % heads != 0:
            raise ValueError(f"channels={channels} must be divisible by heads={heads}")
        self.heads = int(heads)
        self.qk_dim = int(qk_dim)
        self.v_dim = channels // heads
        self.scale = self.qk_dim**-0.5
        self.norm = RMSNorm2d(channels, eps)
        self.query = nn.Conv2d(channels, heads * qk_dim, kernel_size=1)
        self.key = nn.Conv2d(channels, heads * qk_dim, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def _split(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        bsz, _, n_frames, n_bands = x.shape
        if n_frames == 1:
            return x.reshape(bsz, self.heads, dim, n_bands)
        # Training only: frames are independent attention problems.
        return x.permute(0, 2, 1, 3).reshape(bsz * n_frames, self.heads, dim, n_bands)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, channels, n_frames, n_bands = x.shape
        y = self.norm(x)
        q = self._split(self.query(y), self.qk_dim)
        k = self._split(self.key(y), self.qk_dim)
        v = self._split(self.value(y), self.v_dim)
        scores = torch.matmul(q.transpose(-1, -2), k)
        if self.scale != 1.0:
            scores = scores * self.scale
        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(v, weights.transpose(-1, -2))
        if n_frames == 1:
            out = out.reshape(bsz, channels, 1, n_bands)
        else:
            out = out.reshape(bsz, n_frames, channels, n_bands).permute(0, 2, 1, 3)
        return x + self.proj(out)

    @torch.no_grad()
    def fold_norm_(self) -> None:
        gain = self.norm.take_gain()
        for conv in (self.query, self.key, self.value):
            _fold_gain_into_conv_input(conv, gain)
        # Pre-scale the query so the exported graph has no score Mul.
        self.query.weight.mul_(self.scale)
        self.query.bias.mul_(self.scale)
        self.scale = 1.0


class BandConvGLU(nn.Module):
    """Pre-norm depthwise band conv followed by a gated 1x1 FFN."""

    def __init__(self, channels: int, hidden: int, kernel_size: int, eps: float) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError(f"freq kernel must be odd, got {kernel_size}")
        _check_span(kernel_size, name="band conv")
        self.norm = RMSNorm2d(channels, eps)
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size=(1, kernel_size), padding=(0, kernel_size // 2), groups=channels
        )
        self.value = nn.Conv2d(channels, hidden, kernel_size=1)
        self.gate = nn.Conv2d(channels, hidden, kernel_size=1)
        self.proj = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.depthwise(self.norm(x))
        return x + self.proj(self.value(y) * torch.sigmoid(self.gate(y)))

    @torch.no_grad()
    def fold_norm_(self) -> None:
        _fold_gain_into_conv_input(self.depthwise, self.norm.take_gain())


class DualPathBlock(nn.Module):
    """Band GRU (time) -> optional scene FiLM -> optional band attention -> band ConvGLU."""

    def __init__(
        self,
        channels: int,
        *,
        gru_hidden: int,
        scene_hidden: int,
        use_attention: bool,
        attn_heads: int,
        attn_qk_dim: int,
        ffn_hidden: int,
        freq_kernel: int,
        eps: float,
    ) -> None:
        super().__init__()
        self.time = BandGRU(channels, gru_hidden, eps)
        self.scene = SceneGRU(channels, scene_hidden, eps) if scene_hidden > 0 else None
        self.attention = BandAttention(channels, attn_heads, attn_qk_dim, eps) if use_attention else None
        self.freq = BandConvGLU(channels, ffn_hidden, freq_kernel, eps)

    def _frequency(self, x: torch.Tensor) -> torch.Tensor:
        if self.attention is not None:
            x = self.attention(x)
        return self.freq(x)

    def forward(
        self, x: torch.Tensor, h_band: torch.Tensor | None, h_scene: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        x, h_band = self.time(x, h_band)
        if self.scene is not None:
            x, h_scene = self.scene(x, h_scene)
        return self._frequency(x), h_band, h_scene

    def step(
        self, x: torch.Tensor, h_band: torch.Tensor, h_scene: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        x, h_band = self.time.step(x, h_band)
        if self.scene is not None:
            x, h_scene = self.scene.step(x, h_scene)
        return self._frequency(x), h_band, h_scene


class RegionMaskHead(nn.Module):
    """Band token -> complex-mask values for every bin of the band (GLU MLP, tanh-bounded)."""

    def __init__(self, channels: int, hidden: int, n_out: int, mask_bound: float) -> None:
        super().__init__()
        self.value = nn.Conv2d(channels, hidden, kernel_size=1)
        self.gate = nn.Conv2d(channels, hidden, kernel_size=1)
        self.out = nn.Conv2d(hidden, n_out, kernel_size=1)
        self.mask_bound = float(mask_bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = torch.tanh(self.out(self.value(x) * torch.sigmoid(self.gate(x))))
        return mask if self.mask_bound == 1.0 else self.mask_bound * mask


class BandDualPathNPUCore(nn.Module):
    """Deployable core: band-region tensors in, band-region complex masks out.

    ``forward`` processes a whole causal sequence (training); ``forward_stream``
    processes exactly one frame and is the ONNX/ONE export path.
    """

    def __init__(
        self,
        *,
        n_freq: int = 1025,
        regions: Sequence[Sequence[int]] = DEFAULT_REGIONS,
        n_src: int = 3,
        n_chan: int = 1,
        channels: int = 96,
        gru_hidden: int = 80,
        n_blocks: int = 6,
        attn_every: int = 3,
        attn_heads: int = 4,
        attn_qk_dim: int = 16,
        ffn_hidden: int = 96,
        freq_kernel: int = 5,
        scene_hidden: int = 384,
        scene_every: int = 2,
        mask_hidden: int = 128,
        mask_bound: float = 1.0,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if n_blocks <= 0:
            raise ValueError("n_blocks must be positive")
        self.layout = BandLayout(n_freq, regions)
        self.n_freq = int(n_freq)
        self.n_src = int(n_src)
        self.n_chan = int(n_chan)
        self.channels = int(channels)
        self.gru_hidden = int(gru_hidden)
        self.scene_hidden = int(scene_hidden)
        self.n_blocks = int(n_blocks)
        self.n_bands = self.layout.n_bands
        self.in_channels = 2 * self.n_chan
        self.out_channels = 2 * self.n_chan * self.n_src

        def every(idx: int, period: int) -> bool:
            return period > 0 and idx % period == period - 1

        self.embed = nn.ModuleList(
            nn.Conv2d(self.in_channels * region.width, channels, kernel_size=1) for region in self.layout.regions
        )
        self.band_pos = nn.Parameter(torch.randn(1, channels, 1, self.n_bands) * 0.02)
        self.blocks = nn.ModuleList(
            DualPathBlock(
                channels,
                gru_hidden=gru_hidden,
                # A scene GRU after the last block could not influence the masks much; keep it inside.
                scene_hidden=scene_hidden if every(idx, scene_every) and idx < n_blocks - 1 else 0,
                use_attention=every(idx, attn_every),
                attn_heads=attn_heads,
                attn_qk_dim=attn_qk_dim,
                ffn_hidden=ffn_hidden,
                freq_kernel=freq_kernel,
                eps=eps,
            )
            for idx in range(n_blocks)
        )
        self.n_scene = sum(block.scene is not None for block in self.blocks)
        self.out_norm = RMSNorm2d(channels, eps)
        self.heads = nn.ModuleList(
            RegionMaskHead(channels, mask_hidden, self.out_channels * region.width, mask_bound)
            for region in self.layout.regions
        )

    # ------------------------------------------------------------------ shared
    def _embed(self, parts: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(parts) != len(self.embed):
            raise ValueError(f"Expected {len(self.embed)} band-region inputs, got {len(parts)}")
        tokens = [conv(part) for conv, part in zip(self.embed, parts)]
        x = tokens[0] if len(tokens) == 1 else torch.cat(tokens, dim=3)
        return x + self.band_pos

    def _decode(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = self.out_norm(x)
        counts = self.layout.band_counts
        chunks = (x,) if len(counts) == 1 else torch.split(x, list(counts), dim=3)
        return tuple(head(chunk) for head, chunk in zip(self.heads, chunks))

    def _run(self, x: torch.Tensor, band_state, scene_state, *, stream: bool):
        band_states = self._rows(band_state, self.n_blocks)
        scene_states = self._rows(scene_state, self.n_scene)
        next_band, next_scene = [], []
        scene_idx = 0
        for block, h_band in zip(self.blocks, band_states):
            h_scene = None
            if block.scene is not None:
                h_scene = scene_states[scene_idx]
                scene_idx += 1
            if stream:
                x, h_band, h_scene = block.step(x, h_band, h_scene)
            else:
                x, h_band, h_scene = block(x, h_band, h_scene)
            next_band.append(h_band)
            if block.scene is not None:
                next_scene.append(h_scene)
        return x, self._pack_rows(next_band), self._pack_rows(next_scene)

    @staticmethod
    def _rows(state: torch.Tensor | None, count: int) -> list[torch.Tensor | None]:
        if state is None:
            return [None] * count
        return [state[:, :, idx : idx + 1, :] for idx in range(count)]

    @staticmethod
    def _pack_rows(rows: list[torch.Tensor]) -> torch.Tensor | None:
        if not rows:
            return None
        return rows[0] if len(rows) == 1 else torch.cat(rows, dim=2)

    # ---------------------------------------------------------------- training
    def forward(
        self,
        parts: Sequence[torch.Tensor],
        state: tuple[torch.Tensor, torch.Tensor | None] | None = None,
        return_state: bool = False,
    ):
        band_state, scene_state = state if state is not None else (None, None)
        x, next_band, next_scene = self._run(self._embed(parts), band_state, scene_state, stream=False)
        masks = self._decode(x)
        if return_state:
            return masks, (next_band, next_scene)
        return masks

    # --------------------------------------------------------------- streaming
    def init_stream_state(
        self,
        batch_size: int = 1,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        band = torch.zeros(batch_size, self.gru_hidden, self.n_blocks, self.n_bands, device=device, dtype=dtype)
        scene = None
        if self.n_scene:
            scene = torch.zeros(batch_size, self.scene_hidden, self.n_scene, 1, device=device, dtype=dtype)
        return band, scene

    def forward_stream(
        self, parts: Sequence[torch.Tensor], state: tuple[torch.Tensor, torch.Tensor | None]
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, torch.Tensor | None]]:
        if any(part.shape[2] != 1 for part in parts):
            raise ValueError("forward_stream processes exactly one frame per call")
        band_state, scene_state = state
        x, next_band, next_scene = self._run(self._embed(parts), band_state, scene_state, stream=True)
        return self._decode(x), (next_band, next_scene)

    # ----------------------------------------------------------------- export
    @torch.no_grad()
    def prepare_for_export_(self) -> BandDualPathNPUCore:
        """Fold RMSNorm gains / attention scale into convs (numerically identical, fewer nodes)."""
        for block in self.blocks:
            block.time.fold_norm_()
            if block.scene is not None:
                block.scene.fold_norm_()
            if block.attention is not None:
                block.attention.fold_norm_()
            block.freq.fold_norm_()
        gain = self.out_norm.take_gain()
        for head in self.heads:
            _fold_gain_into_conv_input(head.value, gain)
            _fold_gain_into_conv_input(head.gate, gain)
        return self

    def set_rms_reduce_(self, mode: str) -> BandDualPathNPUCore:
        """Select how RMSNorm computes the channel mean in the exported graph ("mean" or "conv")."""
        if mode not in ("mean", "conv"):
            raise ValueError(f"Unsupported RMSNorm reduce mode {mode!r}")
        for module in self.modules():
            if isinstance(module, RMSNorm2d):
                module.reduce = mode
        return self

    def state_size_bytes(self, *, batch_size: int = 1, dtype: torch.dtype = torch.float16) -> int:
        itemsize = torch.empty((), dtype=dtype).element_size()
        return sum(s.numel() for s in self.init_stream_state(batch_size) if s is not None) * itemsize

    def io_size_bytes(self, *, batch_size: int = 1, dtype: torch.dtype = torch.float16) -> dict[str, int]:
        itemsize = torch.empty((), dtype=dtype).element_size()
        regions = self.layout.regions
        inputs = sum(self.in_channels * r.width * r.n_bands for r in regions) * batch_size * itemsize
        outputs = sum(self.out_channels * r.width * r.n_bands for r in regions) * batch_size * itemsize
        state = self.state_size_bytes(batch_size=batch_size, dtype=dtype)
        return {
            "frame_inputs": inputs,
            "mask_outputs": outputs,
            "state_in": state,
            "state_out": state,
            "total": inputs + outputs + 2 * state,
        }

    def macs_per_frame(self) -> int:
        """Analytic multiply-accumulate count for one streamed frame (convs + attention matmuls)."""
        k, c = self.n_bands, self.channels
        total = sum(conv.in_channels * conv.out_channels * r.n_bands for conv, r in zip(self.embed, self.layout.regions))
        for block in self.blocks:
            h = block.time.hidden
            total += k * (3 * c * h + 3 * h * h + h * c)
            if block.scene is not None:
                hs = block.scene.hidden
                total += 3 * c * hs + 3 * hs * hs + 2 * hs * c
            if block.attention is not None:
                att = block.attention
                qk = att.heads * att.qk_dim
                total += k * (2 * c * qk + 2 * c * c)
                total += att.heads * k * k * (att.qk_dim + att.v_dim)
            ffn = block.freq
            total += k * c * ffn.depthwise.kernel_size[1]
            total += k * 3 * c * ffn.value.out_channels
        for head, r in zip(self.heads, self.layout.regions):
            hid = head.value.out_channels
            total += r.n_bands * (2 * c * hid + hid * head.out.out_channels)
        return int(total)


class BandDualPathNPUModel(nn.Module):
    """Training wrapper: complex STFT ``[B, M, F, T]`` -> separated STFT ``[B, S, M, F, T]``.

    It reproduces the host-side contract exactly: pack real/imag, power-law
    compress, band-pack, run the core, unpack masks, apply complex masks to the
    *uncompressed* mixture STFT.
    """

    def __init__(
        self,
        *,
        compress_exponent: float = 0.3,
        mixture_consistency: bool = False,
        **core_kwargs,
    ) -> None:
        super().__init__()
        self.core = BandDualPathNPUCore(**core_kwargs)
        self.compress_exponent = float(compress_exponent)
        self.mixture_consistency = bool(mixture_consistency)
        self.n_src = self.core.n_src
        self.n_chan = self.core.n_chan

    @staticmethod
    def pack_complex(x: torch.Tensor) -> torch.Tensor:
        """``[B, M, F, T]`` complex -> ``[B, 2M, T, F]`` real, channel order (m0.re, m0.im, m1.re, ...)."""
        x = x.transpose(-2, -1)
        return torch.stack((x.real, x.imag), dim=2).reshape(x.shape[0], 2 * x.shape[1], x.shape[2], x.shape[3])

    def host_features(self, spec: torch.Tensor) -> tuple[torch.Tensor, ...]:
        packed = self.pack_complex(spec)
        return self.core.layout.pack(compress_packed_spectrum(packed, self.compress_exponent))

    def apply_masks(self, spec: torch.Tensor, masks: Sequence[torch.Tensor]) -> torch.Tensor:
        """Host post-processing: complex masks (order src, mic, re/im) times the mixture STFT."""
        bsz, n_chan, n_freq, n_frames = spec.shape
        full = self.core.layout.unpack(masks, self.core.out_channels)  # [B, S*M*2, T, F]
        full = full.reshape(bsz, self.n_src, n_chan, 2, n_frames, n_freq).transpose(-1, -2)
        mask = torch.complex(full[:, :, :, 0].float(), full[:, :, :, 1].float())
        est = mask * spec.unsqueeze(1)
        if self.mixture_consistency:
            est = est + (spec.unsqueeze(1) - est.sum(dim=1, keepdim=True)) / self.n_src
        return est

    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        kwargs.pop("ref", None)
        masks = self.core(self.host_features(input))
        return self.apply_masks(input, masks)


class BandDualPathNPUExportWrapper(nn.Module):
    """ONNX signature ``(x_band0..N, band_state[, scene_state]) -> (mask_band0..N, next_band_state[, next_scene_state])``."""

    def __init__(self, core: BandDualPathNPUCore) -> None:
        super().__init__()
        self.core = core
        self.n_regions = len(core.layout.regions)
        self.has_scene = core.n_scene > 0

    def forward(self, *inputs: torch.Tensor):
        parts = inputs[: self.n_regions]
        band_state = inputs[self.n_regions]
        scene_state = inputs[self.n_regions + 1] if self.has_scene else None
        masks, (next_band, next_scene) = self.core.forward_stream(parts, (band_state, scene_state))
        return (*masks, next_band) + ((next_scene,) if self.has_scene else ())

    def example_inputs(self, batch_size: int = 1) -> tuple[torch.Tensor, ...]:
        parts = tuple(
            torch.randn(batch_size, self.core.in_channels * r.width, 1, r.n_bands) for r in self.core.layout.regions
        )
        return (*parts, *(s for s in self.core.init_stream_state(batch_size) if s is not None))

    def io_names(self) -> tuple[list[str], list[str]]:
        states = ["band_state"] + (["scene_state"] if self.has_scene else [])
        inputs = [f"x_band{idx}" for idx in range(self.n_regions)] + states
        outputs = [f"mask_band{idx}" for idx in range(self.n_regions)] + [f"next_{name}" for name in states]
        return inputs, outputs


PRESETS: dict[str, dict] = {
    # Main candidate: band GRU in every block, band attention every 3rd block,
    # scene GRU after blocks 2 and 4.
    "medium": dict(
        channels=96, gru_hidden=80, n_blocks=6, attn_every=3, ffn_hidden=96,
        scene_hidden=384, scene_every=2, mask_hidden=128,
    ),
    # No BatchMatMul/Softmax/Transpose at all: lowest compile risk and node count.
    "medium_noattn": dict(
        channels=96, gru_hidden=80, n_blocks=6, attn_every=0, ffn_hidden=128,
        scene_hidden=384, scene_every=2, mask_hidden=128,
    ),
    # Wider, shallower trunk with a scene GRU after every block but the last (~4M params).
    "wide": dict(
        channels=128, gru_hidden=96, n_blocks=4, attn_every=2, ffn_hidden=96,
        scene_hidden=512, scene_every=1, mask_hidden=96,
    ),
    # Low-power fallback.
    "small": dict(
        channels=64, gru_hidden=64, n_blocks=5, attn_every=2, ffn_hidden=96,
        scene_hidden=256, scene_every=2, mask_hidden=96,
    ),
}


def build_band_dualpath_npu_system(
    *,
    n_fft: int,
    hop_length: int,
    fs: int,
    preset: str | None = None,
    n_src: int = 3,
    n_chan: int = 1,
    regions: Sequence[Sequence[int]] = DEFAULT_REGIONS,
    compress_exponent: float = 0.3,
    mixture_consistency: bool = False,
    scaling: bool = False,
    css_segment_size: int = 12,
    css_shift_size: int = 6,
    css_batch_size: int = 1,
    **core_overrides,
):
    from spectral_feature_compression.core.model.model_wrapper import ModelWrapper

    core_kwargs = dict(PRESETS[preset]) if preset is not None else {}
    core_kwargs.update(core_overrides)
    model = BandDualPathNPUModel(
        n_freq=n_fft // 2 + 1,
        regions=[tuple(region) for region in regions],
        n_src=n_src,
        n_chan=n_chan,
        compress_exponent=compress_exponent,
        mixture_consistency=mixture_consistency,
        **core_kwargs,
    )
    return ModelWrapper(
        model=model,
        n_fft=n_fft,
        hop_length=hop_length,
        fs=fs,
        scaling=scaling,
        css_segment_size=css_segment_size,
        css_shift_size=css_shift_size,
        css_batch_size=css_batch_size,
    )


__all__ = [
    "DEFAULT_REGIONS",
    "PRESETS",
    "BandDualPathNPUCore",
    "BandDualPathNPUExportWrapper",
    "BandDualPathNPUModel",
    "BandLayout",
    "build_band_dualpath_npu_system",
    "compress_packed_spectrum",
]
