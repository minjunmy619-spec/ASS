"""
Performance-first source-aware MelBand RoFormer separator.

This module is intentionally *not* a strict-NPU model.  It is a high-capacity
teacher/reference architecture for DnR-style speech/music/effects separation.
The design puts parameters in the separation path first:

* overlapped adaptive mel-band SFC tokenization preserves perceptual frequency
  structure before expensive modeling;
* axial rotary attention models long time context and inter-band harmonic
  relations;
* explicit source tokens model speech/music/effects streams before full-band
  reconstruction;
* source-axis attention and mixture/source fusion model competition between
  stems;
* complex mask + complex residual reconstruction and mixture consistency target
  high-quality separation rather than a minimal deployment graph.

After this model proves useful as a teacher, a separate distillation step can
approximate it with strict online/NPU primitives.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectral_feature_compression.core.model.adaptive_mel_sfc_2d import (
    AdaptiveMelBandSpec2d,
    _apply_packed_complex_mask_no_repeat,
)
from spectral_feature_compression.core.model.model_wrapper import ModelWrapper
from spectral_feature_compression.core.model.online_model_wrapper import OnlineModelWrapper
from spectral_feature_compression.core.model.online_sfc_2d import (
    RMSNorm2d,
    _runtime_assert,
    pack_complex_stft_as_2d,
    unpack_2d_to_complex_stft,
)
from spectral_feature_compression.core.model.online_soft_band_query_sfc_2d import (
    SoftBandQueryCompressor2d,
    SoftBandQueryExpander2d,
)


def _as_pair(value: Sequence[int] | int, *, name: str) -> tuple[int, int]:
    pair = (value, value) if isinstance(value, int) else tuple(int(v) for v in value)
    if len(pair) != 2:
        raise ValueError(f"{name} must contain exactly two values, got {value}.")
    return pair


def _packed_complex_features(x: torch.Tensor, *, n_chan: int, include_logmag: bool) -> torch.Tensor:
    """Augment packed real/imag channels with magnitude features."""

    _runtime_assert(x.shape[1] == 2 * n_chan, f"Expected {2 * n_chan} packed channels, got {x.shape}")
    feats: list[torch.Tensor] = [x]
    mags: list[torch.Tensor] = []
    for chan_idx in range(n_chan):
        real = x[:, 2 * chan_idx : 2 * chan_idx + 1]
        imag = x[:, 2 * chan_idx + 1 : 2 * chan_idx + 2]
        mags.append(torch.sqrt(real * real + imag * imag + 1e-8))
    mag = torch.cat(mags, dim=1)
    feats.append(mag)
    if include_logmag:
        feats.append(torch.log1p(mag))
    return torch.cat(feats, dim=1)


def _reshape_source_tokens(source_tokens: torch.Tensor) -> tuple[torch.Tensor, int, int, int, int, int]:
    _runtime_assert(source_tokens.ndim == 5, f"Expected [B,N,C,T,K], got {source_tokens.shape}")
    batch, n_src, channels, n_frames, n_bands = source_tokens.shape
    flat = source_tokens.reshape(batch * n_src, channels, n_frames, n_bands)
    return flat, batch, n_src, channels, n_frames, n_bands


class RotarySelfAttention(nn.Module):
    """Multi-head self-attention with rotary position embedding."""

    def __init__(self, channels: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if channels % n_heads != 0:
            raise ValueError(f"channels={channels} must be divisible by n_heads={n_heads}")
        self.channels = int(channels)
        self.n_heads = int(n_heads)
        self.head_dim = channels // n_heads
        self.qkv = nn.Linear(channels, 3 * channels, bias=True)
        self.out = nn.Linear(channels, channels, bias=True)
        self.dropout = float(dropout)

    def _rotary_tables(
        self, seq_len: int, *, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rotary_dim = (self.head_dim // 2) * 2
        if rotary_dim == 0:
            empty = torch.empty(1, 1, seq_len, 0, device=device, dtype=dtype)
            return empty, empty
        pos = torch.arange(seq_len, device=device, dtype=torch.float32)
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32) / rotary_dim))
        freqs = torch.outer(pos, inv_freq)
        cos = freqs.cos().to(dtype=dtype).view(1, 1, seq_len, rotary_dim // 2)
        sin = freqs.sin().to(dtype=dtype).view(1, 1, seq_len, rotary_dim // 2)
        return cos, sin

    @staticmethod
    def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        rotary_dim = cos.shape[-1] * 2
        if rotary_dim == 0:
            return x
        x_rot = x[..., :rotary_dim]
        x_pass = x[..., rotary_dim:]
        x_pair = x_rot.reshape(*x_rot.shape[:-1], rotary_dim // 2, 2)
        x_even = x_pair[..., 0]
        x_odd = x_pair[..., 1]
        rotated = torch.stack([x_even * cos - x_odd * sin, x_even * sin + x_odd * cos], dim=-1).flatten(-2)
        if x_pass.numel() == 0:
            return rotated
        return torch.cat([rotated, x_pass], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        batch, seq_len, channels = x.shape
        qkv = self.qkv(x).reshape(batch, seq_len, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        cos, sin = self._rotary_tables(seq_len, device=x.device, dtype=x.dtype)
        q = self._apply_rotary(q, cos, sin)
        k = self._apply_rotary(k, cos, sin)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.permute(0, 2, 1, 3).reshape(batch, seq_len, channels)
        return self.out(y)


class ChannelFFN2d(nn.Module):
    """Channel-last FFN applied independently per time/band token."""

    def __init__(self, channels: int, hidden_channels: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.net = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, channels),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.permute(0, 2, 3, 1)
        y = self.net(self.norm(y))
        return y.permute(0, 3, 1, 2)


class LocalTFConv2d(nn.Module):
    """Conformer-style local time-frequency convolution branch."""

    def __init__(self, channels: int, *, conv_kernel_size: tuple[int, int], dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            RMSNorm2d(channels),
            nn.Conv2d(channels, 2 * channels, kernel_size=1, bias=True),
            nn.GLU(dim=1),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=conv_kernel_size,
                padding=(conv_kernel_size[0] // 2, conv_kernel_size[1] // 2),
                groups=channels,
                bias=True,
            ),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MelBandRoformerBlock2d(nn.Module):
    """Local TF conv + time RoPE attention + band RoPE attention + FFN."""

    def __init__(
        self,
        channels: int,
        *,
        n_heads: int,
        ffn_mult: int,
        dropout: float,
        conv_kernel_size: tuple[int, int],
        layer_scale_init: float = 0.1,
    ):
        super().__init__()
        self.local = LocalTFConv2d(channels, conv_kernel_size=conv_kernel_size, dropout=dropout)
        self.time_norm = nn.LayerNorm(channels)
        self.band_norm = nn.LayerNorm(channels)
        self.time_attn = RotarySelfAttention(channels, n_heads=n_heads, dropout=dropout)
        self.band_attn = RotarySelfAttention(channels, n_heads=n_heads, dropout=dropout)
        self.ffn = ChannelFFN2d(channels, hidden_channels=ffn_mult * channels, dropout=dropout)
        self.local_scale = nn.Parameter(torch.full((1,), float(layer_scale_init)))
        self.time_scale = nn.Parameter(torch.full((1,), float(layer_scale_init)))
        self.band_scale = nn.Parameter(torch.full((1,), float(layer_scale_init)))
        self.ffn_scale = nn.Parameter(torch.full((1,), float(layer_scale_init)))

    def _time_attention(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, n_frames, n_bands = x.shape
        seq = x.permute(0, 3, 2, 1).reshape(batch * n_bands, n_frames, channels)
        out = self.time_attn(self.time_norm(seq))
        return out.reshape(batch, n_bands, n_frames, channels).permute(0, 3, 2, 1)

    def _band_attention(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, n_frames, n_bands = x.shape
        seq = x.permute(0, 2, 3, 1).reshape(batch * n_frames, n_bands, channels)
        out = self.band_attn(self.band_norm(seq))
        return out.reshape(batch, n_frames, n_bands, channels).permute(0, 3, 1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.local(x) * self.local_scale
        x = x + self._time_attention(x) * self.time_scale
        x = x + self._band_attention(x) * self.band_scale
        x = x + self.ffn(x) * self.ffn_scale
        return x


class SourceAxisAttention2d(nn.Module):
    """Attention across source streams at each time/band coordinate."""

    def __init__(self, channels: int, *, n_heads: int, dropout: float, layer_scale_init: float = 0.1):
        super().__init__()
        if channels % n_heads != 0:
            raise ValueError(f"channels={channels} must be divisible by n_heads={n_heads}")
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, n_heads, dropout=dropout, batch_first=True)
        self.scale = nn.Parameter(torch.full((1,), float(layer_scale_init)))

    def forward(self, source_tokens: torch.Tensor) -> torch.Tensor:
        batch, n_src, channels, n_frames, n_bands = source_tokens.shape
        seq = source_tokens.permute(0, 3, 4, 1, 2).reshape(batch * n_frames * n_bands, n_src, channels)
        normed = self.norm(seq)
        out, _ = self.attn(normed, normed, normed, need_weights=False)
        out = out.reshape(batch, n_frames, n_bands, n_src, channels).permute(0, 3, 4, 1, 2)
        return source_tokens + out * self.scale


class MixtureSourceFusion2d(nn.Module):
    """Fuse each source stream with mixture and other-source context."""

    def __init__(self, channels: int, *, dropout: float, layer_scale_init: float = 0.1):
        super().__init__()
        self.norm = RMSNorm2d(3 * channels)
        self.net = nn.Sequential(
            nn.Conv2d(3 * channels, 2 * channels, kernel_size=1, bias=True),
            nn.GLU(dim=1),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Dropout(dropout),
        )
        self.scale = nn.Parameter(torch.full((1,), float(layer_scale_init)))

    def forward(self, source_tokens: torch.Tensor, mixture_tokens: torch.Tensor) -> torch.Tensor:
        batch, n_src, channels, n_frames, n_bands = source_tokens.shape
        source_mean = source_tokens.mean(dim=1, keepdim=True)
        if n_src > 1:
            other_mean = (source_mean * float(n_src) - source_tokens) / float(n_src - 1)
        else:
            other_mean = source_mean.expand_as(source_tokens)
        mixture = mixture_tokens.unsqueeze(1).expand(-1, n_src, -1, -1, -1)
        fused = torch.cat([source_tokens, other_mean, mixture], dim=2).reshape(
            batch * n_src,
            3 * channels,
            n_frames,
            n_bands,
        )
        delta = self.net(self.norm(fused)).reshape(batch, n_src, channels, n_frames, n_bands)
        return source_tokens + delta * self.scale


class SourceAwareDecoderBlock2d(nn.Module):
    """Per-source axial modeling, source competition, and mixture fusion."""

    def __init__(
        self,
        channels: int,
        *,
        n_heads: int,
        source_attention_heads: int,
        ffn_mult: int,
        dropout: float,
        conv_kernel_size: tuple[int, int],
        layer_scale_init: float = 0.1,
    ):
        super().__init__()
        self.self_block = MelBandRoformerBlock2d(
            channels,
            n_heads=n_heads,
            ffn_mult=ffn_mult,
            dropout=dropout,
            conv_kernel_size=conv_kernel_size,
            layer_scale_init=layer_scale_init,
        )
        self.source_attn = SourceAxisAttention2d(
            channels,
            n_heads=source_attention_heads,
            dropout=dropout,
            layer_scale_init=layer_scale_init,
        )
        self.fusion = MixtureSourceFusion2d(channels, dropout=dropout, layer_scale_init=layer_scale_init)

    def forward(self, source_tokens: torch.Tensor, mixture_tokens: torch.Tensor) -> torch.Tensor:
        flat, batch, n_src, channels, n_frames, n_bands = _reshape_source_tokens(source_tokens)
        flat = self.self_block(flat)
        source_tokens = flat.reshape(batch, n_src, channels, n_frames, n_bands)
        source_tokens = self.source_attn(source_tokens)
        return self.fusion(source_tokens, mixture_tokens)


class ReconstructionHead2d(nn.Module):
    """Source-shared full-band decoder that emits complex masks and residuals."""

    def __init__(self, channels: int, *, n_chan: int, dropout: float):
        super().__init__()
        out_ch = 2 * n_chan
        self.mask = nn.Sequential(
            RMSNorm2d(channels),
            nn.Conv2d(channels, 2 * channels, kernel_size=1, bias=True),
            nn.GLU(dim=1),
            nn.Dropout(dropout),
            nn.Conv2d(channels, out_ch, kernel_size=1, bias=True),
        )
        self.residual = nn.Sequential(
            RMSNorm2d(channels),
            nn.Conv2d(channels, 2 * channels, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(2 * channels, out_ch, kernel_size=1, bias=True),
        )

    def mask_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.mask(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # The RoFormer mask head has no post-projection activation. Its final
        # projection is both the raw mask logit and the complex mask used below.
        mask_logits = self.mask_logits(x)
        return mask_logits, mask_logits, self.residual(x)


class SourceAwareMelBandRoformer2D(nn.Module):
    """High-performance packed-STFT separator core."""

    def __init__(
        self,
        n_freq: int,
        *,
        n_fft: int | None = None,
        sample_rate: int = 44100,
        n_src: int = 3,
        n_chan: int = 1,
        n_bands: int = 128,
        d_model: int = 128,
        n_heads: int = 8,
        source_attention_heads: int = 1,
        n_encoder_layers: int = 6,
        n_decoder_layers: int = 4,
        ffn_mult: int = 4,
        dropout: float = 0.1,
        conv_kernel_size: Sequence[int] | int = (5, 5),
        routing_kernel_size: Sequence[int] | int = (3, 5),
        low_freq_hz: float = 1000.0,
        low_freq_band_fraction: float = 0.45,
        overlap_factor: float = 1.5,
        low_freq_overlap_factor: float = 2.0,
        include_logmag_features: bool = True,
        masking: bool = True,
        residual_output: bool = True,
        residual_scale_init: float = 0.05,
        mixture_consistency: bool = True,
        routing_normalization: str = "softmax",
        layer_scale_init: float = 0.1,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        if source_attention_heads <= 0:
            raise ValueError("source_attention_heads must be positive")
        if d_model % source_attention_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by source_attention_heads={source_attention_heads}")
        conv_kernel_size = _as_pair(conv_kernel_size, name="conv_kernel_size")
        routing_kernel_size = _as_pair(routing_kernel_size, name="routing_kernel_size")
        if conv_kernel_size[0] % 2 == 0 or conv_kernel_size[1] % 2 == 0:
            raise ValueError(f"conv_kernel_size must be odd, got {conv_kernel_size}")

        self.n_freq = int(n_freq)
        self.n_src = int(n_src)
        self.n_chan = int(n_chan)
        self.n_bands = int(n_bands)
        self.d_model = int(d_model)
        self.masking = bool(masking)
        self.residual_output = bool(residual_output)
        self.mixture_consistency = bool(mixture_consistency)
        self.include_logmag_features = bool(include_logmag_features)

        feature_channels = (4 if include_logmag_features else 3) * n_chan
        band_spec = AdaptiveMelBandSpec2d(
            n_freq=n_freq,
            n_bands=n_bands,
            sample_rate=sample_rate,
            low_freq_hz=low_freq_hz,
            low_freq_band_fraction=low_freq_band_fraction,
            overlap_factor=overlap_factor,
            low_freq_overlap_factor=low_freq_overlap_factor,
        )
        self.band_spec = band_spec
        self.input_frontend = nn.Sequential(
            nn.Conv2d(feature_channels, d_model, kernel_size=1, bias=True),
            RMSNorm2d(d_model),
            nn.Conv2d(d_model, d_model, kernel_size=(3, 3), padding=(1, 1), bias=True),
            nn.GELU(),
            RMSNorm2d(d_model),
        )
        self.compressor = SoftBandQueryCompressor2d(
            channels=d_model,
            band_spec=band_spec,
            kernel_size=routing_kernel_size,
            causal=False,
            normalization=routing_normalization,
        )
        self.encoder = nn.ModuleList(
            [
                MelBandRoformerBlock2d(
                    d_model,
                    n_heads=n_heads,
                    ffn_mult=ffn_mult,
                    dropout=dropout,
                    conv_kernel_size=conv_kernel_size,
                    layer_scale_init=layer_scale_init,
                )
                for _ in range(n_encoder_layers)
            ]
        )
        self.source_embedding = nn.Parameter(torch.randn(n_src, d_model) * 0.02)
        self.source_seed = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )
        self.decoder = nn.ModuleList(
            [
                SourceAwareDecoderBlock2d(
                    d_model,
                    n_heads=n_heads,
                    source_attention_heads=source_attention_heads,
                    ffn_mult=ffn_mult,
                    dropout=dropout,
                    conv_kernel_size=conv_kernel_size,
                    layer_scale_init=layer_scale_init,
                )
                for _ in range(n_decoder_layers)
            ]
        )
        self.expander = SoftBandQueryExpander2d(channels=d_model, band_spec=band_spec)
        self.reconstruction = ReconstructionHead2d(d_model, n_chan=n_chan, dropout=dropout)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))

    def _apply_mixture_consistency(self, estimates: torch.Tensor, mixture: torch.Tensor) -> torch.Tensor:
        if not self.mixture_consistency:
            return estimates
        batch, _, n_frames, n_freq = mixture.shape
        est = estimates.reshape(batch, self.n_src, 2 * self.n_chan, n_frames, n_freq)
        correction = (mixture - est.sum(dim=1)) / float(self.n_src)
        est = est + correction.unsqueeze(1)
        return est.reshape(batch, 2 * self.n_src * self.n_chan, n_frames, n_freq)

    def forward(self, x: torch.Tensor, *, return_aux: bool = False):
        _runtime_assert(x.ndim == 4, f"Expected [B,2M,T,F], got {x.shape}")
        _runtime_assert(x.shape[1] == 2 * self.n_chan, f"Expected {2 * self.n_chan} packed channels, got {x.shape}")
        _runtime_assert(x.shape[-1] == self.n_freq, f"Expected F={self.n_freq}, got {x.shape}")

        features = _packed_complex_features(x, n_chan=self.n_chan, include_logmag=self.include_logmag_features)
        h = self.input_frontend(features)
        mixture_tokens, query_tokens = self.compressor(h)
        for block in self.encoder:
            mixture_tokens = block(mixture_tokens)

        batch, channels, n_frames, n_bands = mixture_tokens.shape
        source_bias = self.source_seed(self.source_embedding).view(1, self.n_src, channels, 1, 1)
        source_tokens = mixture_tokens.unsqueeze(1) + source_bias
        for block in self.decoder:
            source_tokens = block(source_tokens, mixture_tokens)

        flat, _, _, _, _, _ = _reshape_source_tokens(source_tokens)
        query_rep = query_tokens.repeat_interleave(self.n_src, dim=0)
        fullband = self.expander(flat, query_rep)
        masks_flat, mask_logits_flat, residual_flat = self.reconstruction(fullband)
        masks = masks_flat.reshape(batch, self.n_src * 2 * self.n_chan, n_frames, self.n_freq)
        mask_logits = mask_logits_flat.reshape(batch, self.n_src * 2 * self.n_chan, n_frames, self.n_freq)
        if not self.masking:
            if return_aux:
                return masks, {
                    "mask": masks,
                    "mask_domain": "packed_complex_mask",
                    "mask_logits": mask_logits,
                    "mask_logits_domain": "source_aware_melband_roformer_complex_mask_logits",
                }
            return masks

        estimates = _apply_packed_complex_mask_no_repeat(x=x, y=masks, n_src=self.n_src, n_chan=self.n_chan)
        if self.residual_output:
            residual = residual_flat.reshape(batch, self.n_src * 2 * self.n_chan, n_frames, self.n_freq)
            estimates = estimates + residual * self.residual_scale
        estimates = self._apply_mixture_consistency(estimates, x)
        if return_aux:
            return estimates, {
                "mask": masks,
                "mask_domain": "packed_complex_mask",
                "mask_logits": mask_logits,
                "mask_logits_domain": "source_aware_melband_roformer_complex_mask_logits",
            }
        return estimates


class SourceAwareMelBandRoformerModel(nn.Module):
    """Complex-STFT wrapper around SourceAwareMelBandRoformer2D."""

    def __init__(self, *, n_freq: int, n_src: int = 3, n_chan: int = 1, **kwargs):
        super().__init__()
        self.core = SourceAwareMelBandRoformer2D(n_freq=n_freq, n_src=n_src, n_chan=n_chan, **kwargs)
        self.n_src = int(n_src)
        self.n_chan = int(n_chan)

    def forward(self, x: torch.Tensor, **kwargs):
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


def build_source_aware_melband_roformer_system(
    *,
    n_fft: int,
    hop_length: int,
    fs: int,
    n_src: int = 3,
    n_chan: int = 1,
    n_bands: int = 128,
    d_model: int = 128,
    n_heads: int = 8,
    source_attention_heads: int = 1,
    n_encoder_layers: int = 6,
    n_decoder_layers: int = 4,
    ffn_mult: int = 4,
    dropout: float = 0.1,
    conv_kernel_size: Sequence[int] | int = (5, 5),
    routing_kernel_size: Sequence[int] | int = (3, 5),
    low_freq_hz: float = 1000.0,
    low_freq_band_fraction: float = 0.45,
    overlap_factor: float = 1.5,
    low_freq_overlap_factor: float = 2.0,
    include_logmag_features: bool = True,
    masking: bool = True,
    residual_output: bool = True,
    residual_scale_init: float = 0.05,
    mixture_consistency: bool = True,
    routing_normalization: str = "softmax",
    layer_scale_init: float = 0.1,
    scaling: bool = True,
    css_segment_size: int = 12,
    css_shift_size: int = 6,
    css_batch_size: int = 1,
    online_wrapper: bool = False,
):
    n_freq = (n_fft // 2) + 1
    model = SourceAwareMelBandRoformerModel(
        n_freq=n_freq,
        n_src=n_src,
        n_chan=n_chan,
        n_fft=n_fft,
        sample_rate=fs,
        n_bands=n_bands,
        d_model=d_model,
        n_heads=n_heads,
        source_attention_heads=source_attention_heads,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers,
        ffn_mult=ffn_mult,
        dropout=dropout,
        conv_kernel_size=conv_kernel_size,
        routing_kernel_size=routing_kernel_size,
        low_freq_hz=low_freq_hz,
        low_freq_band_fraction=low_freq_band_fraction,
        overlap_factor=overlap_factor,
        low_freq_overlap_factor=low_freq_overlap_factor,
        include_logmag_features=include_logmag_features,
        masking=masking,
        residual_output=residual_output,
        residual_scale_init=residual_scale_init,
        mixture_consistency=mixture_consistency,
        routing_normalization=routing_normalization,
        layer_scale_init=layer_scale_init,
    )
    wrapper_cls = OnlineModelWrapper if online_wrapper else ModelWrapper
    return wrapper_cls(
        model=model,
        n_fft=n_fft,
        hop_length=hop_length,
        fs=fs,
        scaling=False if online_wrapper else scaling,
        css_segment_size=css_segment_size,
        css_shift_size=css_shift_size,
        css_batch_size=css_batch_size,
    )


def estimate_source_aware_melband_roformer_params(
    *,
    n_fft: int = 2048,
    fs: int = 44100,
    n_src: int = 3,
    n_chan: int = 1,
    n_bands: int = 128,
    d_model: int = 128,
    n_heads: int = 8,
    source_attention_heads: int = 1,
    n_encoder_layers: int = 6,
    n_decoder_layers: int = 4,
) -> int:
    """Convenience helper for recipe-size sanity checks in tests/tools."""

    model = SourceAwareMelBandRoformer2D(
        n_freq=(n_fft // 2) + 1,
        n_fft=n_fft,
        sample_rate=fs,
        n_src=n_src,
        n_chan=n_chan,
        n_bands=n_bands,
        d_model=d_model,
        n_heads=n_heads,
        source_attention_heads=source_attention_heads,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers,
    )
    return sum(int(p.numel()) for p in model.parameters())
