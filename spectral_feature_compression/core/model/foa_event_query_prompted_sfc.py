"""
High-capacity FOA event-query Prompted Asymmetric SFC.

This is the non-NPU Proposal-D branch: it is intentionally offline/non-causal and
uses axial Transformer attention.  Query prompts come from one-hot or soft sound
event class vectors.  The waveform wrapper expects 4-channel FOA input and
returns one mono estimate per query.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectral_feature_compression.core.model.model_wrapper import ModelWrapper
from spectral_feature_compression.core.model.online_crossattn_query_sfc_2d import (
    NPUSafeCrossAttnDecoder2d,
    NPUSafeCrossAttnEncoder2d,
)
from spectral_feature_compression.core.model.online_sfc_2d import (
    RMSNorm2d,
    _runtime_assert,
    pack_complex_stft_as_2d,
    unpack_2d_to_complex_stft,
)
from spectral_feature_compression.core.model.online_soft_band_sfc_2d import SoftBandSpec2d


def _as_pair(value: Sequence[int] | int, *, name: str) -> tuple[int, int]:
    pair = (value, value) if isinstance(value, int) else tuple(int(v) for v in value)
    if len(pair) != 2:
        raise ValueError(f"{name} must contain exactly two values, got {value}.")
    return pair


def _default_event_labels(n_event_classes: int) -> tuple[str, ...]:
    return tuple(f"event_{idx:02d}" for idx in range(int(n_event_classes)))


def _apply_mono_mask_from_foa_w_channel(*, x: torch.Tensor, y: torch.Tensor, n_src: int) -> torch.Tensor:
    """Apply per-query mono complex masks to the FOA W-channel mixture."""

    _runtime_assert(x.shape[1] >= 2, f"Expected packed FOA input channels, got {x.shape}")
    _runtime_assert(y.shape[1] == 2 * n_src, f"Expected {2 * n_src} output channels, got {y.shape}")
    in_r = x[:, 0:1, :, :]
    in_i = x[:, 1:2, :, :]
    outputs: list[torch.Tensor] = []
    for src_idx in range(n_src):
        mask_base = 2 * src_idx
        mask_r = y[:, mask_base : mask_base + 1, :, :]
        mask_i = y[:, mask_base + 1 : mask_base + 2, :, :]
        outputs.append(in_r * mask_r - in_i * mask_i)
        outputs.append(in_r * mask_i + in_i * mask_r)
    return torch.cat(outputs, dim=1)


class ChannelFFN2d(nn.Module):
    """Channel-last Transformer FFN for 4D spectrogram tokens."""

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


class AxialTransformerBlock2d(nn.Module):
    """Non-causal time-axis + band-axis attention with a Conformer-style conv branch."""

    def __init__(
        self,
        channels: int,
        *,
        n_heads: int = 8,
        ffn_mult: int = 4,
        dropout: float = 0.1,
        conv_kernel_size: Sequence[int] | int = (5, 5),
        layer_scale_init: float = 0.1,
    ):
        super().__init__()
        if channels % n_heads != 0:
            raise ValueError(f"channels={channels} must be divisible by n_heads={n_heads}.")
        conv_kernel_size = _as_pair(conv_kernel_size, name="conv_kernel_size")
        if conv_kernel_size[0] % 2 == 0 or conv_kernel_size[1] % 2 == 0:
            raise ValueError(f"conv_kernel_size must be odd for same padding, got {conv_kernel_size}.")

        self.channels = int(channels)
        self.time_norm = nn.LayerNorm(channels)
        self.band_norm = nn.LayerNorm(channels)
        self.time_attn = nn.MultiheadAttention(channels, n_heads, dropout=dropout, batch_first=True)
        self.band_attn = nn.MultiheadAttention(channels, n_heads, dropout=dropout, batch_first=True)
        self.conv = nn.Sequential(
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
        self.ffn = ChannelFFN2d(channels, hidden_channels=ffn_mult * channels, dropout=dropout)
        self.time_scale = nn.Parameter(torch.full((1,), float(layer_scale_init)))
        self.band_scale = nn.Parameter(torch.full((1,), float(layer_scale_init)))
        self.conv_scale = nn.Parameter(torch.full((1,), float(layer_scale_init)))
        self.ffn_scale = nn.Parameter(torch.full((1,), float(layer_scale_init)))

    def _time_attention(self, x: torch.Tensor) -> torch.Tensor:
        bsz, channels, n_frames, n_bands = x.shape
        seq = x.permute(0, 3, 2, 1).reshape(bsz * n_bands, n_frames, channels)
        out, _ = self.time_attn(self.time_norm(seq), self.time_norm(seq), self.time_norm(seq), need_weights=False)
        return out.reshape(bsz, n_bands, n_frames, channels).permute(0, 3, 2, 1)

    def _band_attention(self, x: torch.Tensor) -> torch.Tensor:
        bsz, channels, n_frames, n_bands = x.shape
        seq = x.permute(0, 2, 3, 1).reshape(bsz * n_frames, n_bands, channels)
        out, _ = self.band_attn(self.band_norm(seq), self.band_norm(seq), self.band_norm(seq), need_weights=False)
        return out.reshape(bsz, n_frames, n_bands, channels).permute(0, 3, 1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self._time_attention(x) * self.time_scale
        x = x + self._band_attention(x) * self.band_scale
        x = x + self.conv(x) * self.conv_scale
        x = x + self.ffn(x) * self.ffn_scale
        return x


class EventClassQueryEmbedding(nn.Module):
    """Map one-hot, softmax, or logits class queries into dense prompt embeddings."""

    def __init__(
        self,
        *,
        n_event_classes: int,
        n_queries: int,
        channels: int,
        hidden_channels: int | None = None,
        dropout: float = 0.1,
        condition_mode: str = "probability",
    ):
        super().__init__()
        if n_event_classes <= 0:
            raise ValueError(f"n_event_classes must be positive, got {n_event_classes}.")
        if n_queries <= 0:
            raise ValueError(f"n_queries must be positive, got {n_queries}.")
        hidden_channels = hidden_channels or max(4 * channels, 2 * n_event_classes)
        self.n_event_classes = int(n_event_classes)
        self.n_queries = int(n_queries)
        self.channels = int(channels)
        self.condition_mode = condition_mode
        self.default_query_logits = nn.Parameter(torch.zeros(n_queries, n_event_classes))
        self.query_bias = nn.Parameter(torch.randn(n_queries, channels) * 0.02)
        self.net = nn.Sequential(
            nn.Linear(n_event_classes, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, channels),
            nn.LayerNorm(channels),
        )

    def _convert_condition(self, event_condition: torch.Tensor, mode: str) -> torch.Tensor:
        if mode in {"probability", "soft", "softmax_probability"}:
            return event_condition
        if mode in {"softmax", "logits"}:
            return torch.softmax(event_condition, dim=-1)
        if mode == "onehot":
            indices = event_condition.argmax(dim=-1)
            return F.one_hot(indices, num_classes=self.n_event_classes).to(dtype=event_condition.dtype)
        raise ValueError(
            "condition_mode must be one of probability, softmax, logits, onehot, "
            f"got {mode!r}."
        )

    def forward(
        self,
        event_condition: torch.Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        condition_mode: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mode = condition_mode or self.condition_mode
        if event_condition is None:
            probs = torch.softmax(self.default_query_logits, dim=-1).to(device=device, dtype=dtype)
            probs = probs.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            probs = event_condition.to(device=device, dtype=dtype)
            if probs.ndim == 2:
                probs = probs.unsqueeze(1)
            _runtime_assert(probs.ndim == 3, f"event_condition must be [B,Q,C] or [B,C], got {probs.shape}")
            if probs.shape[0] == 1 and batch_size != 1:
                probs = probs.expand(batch_size, -1, -1)
            _runtime_assert(probs.shape[0] == batch_size, f"event_condition batch {probs.shape[0]} vs {batch_size}")
            _runtime_assert(
                probs.shape[1] == self.n_queries,
                f"event_condition queries {probs.shape[1]} vs {self.n_queries}",
            )
            _runtime_assert(
                probs.shape[2] == self.n_event_classes,
                f"event_condition classes {probs.shape[2]} vs {self.n_event_classes}",
            )
            probs = self._convert_condition(probs, mode)

        embeddings = self.net(probs) + self.query_bias.to(device=device, dtype=dtype).unsqueeze(0)
        return probs, embeddings


class EventFiLM2d(nn.Module):
    """Apply event-query FiLM to a flattened [B*Q, C, T, K/F] tensor."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.proj = nn.Linear(channels, 2 * channels)
        self.scale_strength = nn.Parameter(torch.tensor(0.5))
        self.bias_strength = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor, event_embedding: torch.Tensor) -> torch.Tensor:
        _runtime_assert(event_embedding.ndim == 2, f"Expected [B*Q,C] event embedding, got {event_embedding.shape}")
        scale, bias = self.proj(self.norm(event_embedding)).chunk(2, dim=-1)
        scale = torch.tanh(scale).unsqueeze(-1).unsqueeze(-1) * self.scale_strength
        bias = bias.unsqueeze(-1).unsqueeze(-1) * self.bias_strength
        return x * (1.0 + scale) + bias


class EventConditionedDecoderBlock2d(nn.Module):
    """Shared decoder block: event FiLM, source self-attention, and mixture cross-attention."""

    def __init__(
        self,
        channels: int,
        *,
        n_heads: int = 8,
        ffn_mult: int = 4,
        dropout: float = 0.1,
        conv_kernel_size: Sequence[int] | int = (5, 5),
        layer_scale_init: float = 0.1,
    ):
        super().__init__()
        self.film = EventFiLM2d(channels)
        self.self_block = AxialTransformerBlock2d(
            channels,
            n_heads=n_heads,
            ffn_mult=ffn_mult,
            dropout=dropout,
            conv_kernel_size=conv_kernel_size,
            layer_scale_init=layer_scale_init,
        )
        self.query_norm = nn.LayerNorm(channels)
        self.memory_norm = nn.LayerNorm(channels)
        self.cross_attn = nn.MultiheadAttention(channels, n_heads, dropout=dropout, batch_first=True)
        self.cross_scale = nn.Parameter(torch.full((1,), float(layer_scale_init)))
        self.ffn = ChannelFFN2d(channels, hidden_channels=ffn_mult * channels, dropout=dropout)
        self.ffn_scale = nn.Parameter(torch.full((1,), float(layer_scale_init)))

    def forward(
        self,
        source_tokens: torch.Tensor,
        mixture_tokens: torch.Tensor,
        event_embedding: torch.Tensor,
        *,
        n_queries: int,
    ) -> torch.Tensor:
        source_tokens = self.film(source_tokens, event_embedding)
        source_tokens = self.self_block(source_tokens)
        bq, channels, n_frames, n_bands = source_tokens.shape
        memory = mixture_tokens.repeat_interleave(n_queries, dim=0)
        _runtime_assert(memory.shape[0] == bq, f"memory batch {memory.shape[0]} vs source batch {bq}")

        q = source_tokens.permute(0, 2, 3, 1).reshape(bq, n_frames * n_bands, channels)
        kv = memory.permute(0, 2, 3, 1).reshape(bq, n_frames * n_bands, channels)
        cross, _ = self.cross_attn(self.query_norm(q), self.memory_norm(kv), self.memory_norm(kv), need_weights=False)
        source_tokens = (q + cross * self.cross_scale).reshape(bq, n_frames, n_bands, channels).permute(0, 3, 1, 2)
        source_tokens = source_tokens + self.ffn(source_tokens) * self.ffn_scale
        return source_tokens


class FOAEventQueryPromptedAsymmetricSFC2D(nn.Module):
    """Strong non-causal Proposal-D core for FOA-conditioned mono event separation."""

    def __init__(
        self,
        n_freq: int,
        *,
        n_bands: int = 128,
        n_fft: int | None = None,
        sample_rate: int | None = None,
        band_config: str = "musical",
        n_queries: int = 4,
        n_event_classes: int = 18,
        n_chan: int = 4,
        output_n_chan: int = 1,
        d_model: int = 192,
        n_heads: int = 8,
        n_encoder_layers: int = 8,
        n_decoder_layers: int = 4,
        ffn_mult: int = 4,
        dropout: float = 0.1,
        conv_kernel_size: Sequence[int] | int = (5, 5),
        routing_kernel_size: Sequence[int] | tuple[int, int] = (3, 5),
        sfc_query_type: str = "adaptive",
        event_condition_hidden: int | None = None,
        event_condition_mode: str = "probability",
        event_class_labels: Sequence[str] | None = None,
        masking: bool = True,
        residual_output: bool = True,
        routing_normalization: str = "softmax",
    ):
        super().__init__()
        if n_queries <= 0:
            raise ValueError(f"n_queries must be positive, got {n_queries}.")
        if n_chan != 4:
            raise ValueError(f"FOAEventQueryPromptedAsymmetricSFC2D expects n_chan=4 FOA input, got {n_chan}.")
        if output_n_chan != 1:
            raise ValueError(
                "FOAEventQueryPromptedAsymmetricSFC2D currently emits one mono channel per query; "
                f"got output_n_chan={output_n_chan}."
            )
        routing_kernel_size = _as_pair(routing_kernel_size, name="routing_kernel_size")
        self.n_freq = int(n_freq)
        self.n_bands = int(n_bands)
        self.n_queries = int(n_queries)
        self.n_src = int(n_queries)
        self.n_event_classes = int(n_event_classes)
        self.n_chan = int(n_chan)
        self.output_n_chan = int(output_n_chan)
        self.d_model = int(d_model)
        self.masking = bool(masking)
        self.residual_output = bool(residual_output)
        self.event_class_labels = (
            tuple(event_class_labels) if event_class_labels is not None else _default_event_labels(n_event_classes)
        )
        if len(self.event_class_labels) != n_event_classes:
            raise ValueError(f"Expected {n_event_classes} event labels, got {len(self.event_class_labels)}.")

        band_spec = SoftBandSpec2d(
            n_freq=n_freq,
            n_bands=n_bands,
            n_fft=n_fft,
            sample_rate=sample_rate,
            band_config=band_config,
        )
        self.band_spec = band_spec
        self.input_frontend = nn.Sequential(
            nn.Conv2d(2 * n_chan, d_model, kernel_size=1, bias=True),
            RMSNorm2d(d_model),
            nn.Conv2d(d_model, d_model, kernel_size=(3, 3), padding=(1, 1), bias=True),
            nn.GELU(),
            RMSNorm2d(d_model),
        )
        self.compressor = NPUSafeCrossAttnEncoder2d(
            channels=d_model,
            band_spec=band_spec,
            kernel_size=routing_kernel_size,
            causal=False,
            query_type=sfc_query_type,
            routing_normalization=routing_normalization,
        )
        self.event_embed = EventClassQueryEmbedding(
            n_event_classes=n_event_classes,
            n_queries=n_queries,
            channels=d_model,
            hidden_channels=event_condition_hidden,
            dropout=dropout,
            condition_mode=event_condition_mode,
        )
        self.event_seed = nn.Linear(d_model, d_model)
        self.encoder = nn.ModuleList(
            [
                AxialTransformerBlock2d(
                    d_model,
                    n_heads=n_heads,
                    ffn_mult=ffn_mult,
                    dropout=dropout,
                    conv_kernel_size=conv_kernel_size,
                )
                for _ in range(n_encoder_layers)
            ]
        )
        self.decoder = nn.ModuleList(
            [
                EventConditionedDecoderBlock2d(
                    d_model,
                    n_heads=n_heads,
                    ffn_mult=ffn_mult,
                    dropout=dropout,
                    conv_kernel_size=conv_kernel_size,
                )
                for _ in range(n_decoder_layers)
            ]
        )
        self.expander = NPUSafeCrossAttnDecoder2d(
            channels=d_model,
            band_spec=band_spec,
            query_type=sfc_query_type,
            routing_normalization=routing_normalization,
        )
        self.mask_head = nn.Sequential(
            RMSNorm2d(d_model),
            nn.Conv2d(d_model, 2 * d_model, kernel_size=1, bias=True),
            nn.GLU(dim=1),
            nn.Conv2d(d_model, 2 * output_n_chan, kernel_size=1, bias=True),
        )
        self.residual_head = nn.Sequential(
            RMSNorm2d(d_model),
            nn.Conv2d(d_model, 2 * d_model, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(2 * d_model, 2 * output_n_chan, kernel_size=1, bias=True),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.05))

    def event_query_manifest(self) -> dict[str, object]:
        return {
            "type": "foa_event_class_query_prompted_asymmetric_sfc",
            "n_queries": self.n_queries,
            "n_event_classes": self.n_event_classes,
            "event_class_labels": list(self.event_class_labels),
            "query_condition_shape": ["batch", self.n_queries, self.n_event_classes],
            "accepted_condition_modes": ["probability", "softmax", "logits", "onehot"],
            "foa_channels": self.n_chan,
            "output_channels": self.output_n_chan,
            "sfc_query_compression": "cross_attention",
            "static_export_prompts": False,
            "npu_target": False,
        }

    def prompt_manifest(self) -> dict[str, object]:
        return self.event_query_manifest()

    def forward(
        self,
        x: torch.Tensor,
        *,
        event_condition: torch.Tensor | None = None,
        condition_mode: str | None = None,
    ) -> torch.Tensor:
        _runtime_assert(x.ndim == 4, f"Expected 4D input (B,C,T,F), got {x.shape}")
        _runtime_assert(x.shape[1] == 2 * self.n_chan, f"Expected packed FOA channels {2 * self.n_chan}, got {x.shape}")
        _runtime_assert(x.shape[-1] == self.n_freq, f"{x.shape} vs n_freq={self.n_freq}")

        batch = x.shape[0]
        _, event_emb = self.event_embed(
            event_condition,
            batch_size=batch,
            device=x.device,
            dtype=x.dtype,
            condition_mode=condition_mode,
        )
        h = self.input_frontend(x)
        z, side_tokens = self.compressor(h)
        for block in self.encoder:
            z = block(z)

        n_queries = self.n_queries
        event_flat = event_emb.reshape(batch * n_queries, self.d_model)
        source_tokens = z.unsqueeze(1).expand(-1, n_queries, -1, -1, -1).reshape(
            batch * n_queries,
            self.d_model,
            z.shape[2],
            z.shape[3],
        )
        source_tokens = source_tokens + self.event_seed(event_flat).unsqueeze(-1).unsqueeze(-1)
        for block in self.decoder:
            source_tokens = block(source_tokens, z, event_flat, n_queries=n_queries)

        side_rep = side_tokens.repeat_interleave(n_queries, dim=0)
        expanded = self.expander(source_tokens, side_rep)
        masks = self.mask_head(expanded).reshape(
            batch,
            n_queries * 2 * self.output_n_chan,
            expanded.shape[-2],
            self.n_freq,
        )
        if not self.masking:
            return masks

        estimates = _apply_mono_mask_from_foa_w_channel(x=x, y=masks, n_src=n_queries)
        if self.residual_output:
            residual = self.residual_head(expanded).reshape(
                batch,
                n_queries * 2 * self.output_n_chan,
                expanded.shape[-2],
                self.n_freq,
            )
            estimates = estimates + residual * self.residual_scale
        return estimates


class FOAEventQueryPromptedAsymmetricSFCModel(nn.Module):
    """Complex-STFT wrapper for FOAEventQueryPromptedAsymmetricSFC2D."""

    def __init__(
        self,
        *,
        n_freq: int,
        n_queries: int = 4,
        n_chan: int = 4,
        output_n_chan: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.core = FOAEventQueryPromptedAsymmetricSFC2D(
            n_freq=n_freq,
            n_queries=n_queries,
            n_chan=n_chan,
            output_n_chan=output_n_chan,
            **kwargs,
        )
        self.n_src = int(n_queries)
        self.n_queries = int(n_queries)
        self.input_n_chan = int(n_chan)
        self.n_chan = int(output_n_chan)

    def forward(
        self,
        x: torch.Tensor,
        *,
        event_condition: torch.Tensor | None = None,
        condition_mode: str | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        x2d = pack_complex_stft_as_2d(x)
        y2d = self.core(x2d, event_condition=event_condition, condition_mode=condition_mode)
        return unpack_2d_to_complex_stft(y2d, n_src=self.n_queries, n_chan=self.n_chan)


def build_foa_event_query_prompted_asymmetric_sfc_system(
    *,
    n_fft: int,
    hop_length: int,
    fs: int,
    n_queries: int = 4,
    n_src: int | None = None,
    n_event_classes: int = 18,
    n_chan: int = 4,
    output_n_chan: int = 1,
    n_bands: int = 128,
    band_config: str = "musical",
    d_model: int = 192,
    n_heads: int = 8,
    n_encoder_layers: int = 8,
    n_decoder_layers: int = 4,
    ffn_mult: int = 4,
    dropout: float = 0.1,
    conv_kernel_size: Sequence[int] | int = (5, 5),
    routing_kernel_size: Sequence[int] | tuple[int, int] = (3, 5),
    sfc_query_type: str = "adaptive",
    event_condition_hidden: int | None = None,
    event_condition_mode: str = "probability",
    event_class_labels: Sequence[str] | None = None,
    masking: bool = True,
    residual_output: bool = True,
    routing_normalization: str = "softmax",
    scaling: bool = True,
    css_segment_size: int = 12,
    css_shift_size: int = 6,
    css_batch_size: int = 1,
) -> ModelWrapper:
    if n_src is not None and int(n_src) != int(n_queries):
        raise ValueError(f"For event-query Proposal D, n_src must equal n_queries: {n_src} vs {n_queries}.")
    n_freq = (n_fft // 2) + 1
    core = FOAEventQueryPromptedAsymmetricSFCModel(
        n_freq=n_freq,
        n_queries=n_queries,
        n_bands=n_bands,
        n_fft=n_fft,
        sample_rate=fs,
        band_config=band_config,
        n_event_classes=n_event_classes,
        n_chan=n_chan,
        output_n_chan=output_n_chan,
        d_model=d_model,
        n_heads=n_heads,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers,
        ffn_mult=ffn_mult,
        dropout=dropout,
        conv_kernel_size=conv_kernel_size,
        routing_kernel_size=routing_kernel_size,
        sfc_query_type=sfc_query_type,
        event_condition_hidden=event_condition_hidden,
        event_condition_mode=event_condition_mode,
        event_class_labels=event_class_labels,
        masking=masking,
        residual_output=residual_output,
        routing_normalization=routing_normalization,
    )
    return ModelWrapper(
        model=core,
        n_fft=n_fft,
        hop_length=hop_length,
        fs=fs,
        scaling=scaling,
        css_segment_size=css_segment_size,
        css_shift_size=css_shift_size,
        css_batch_size=css_batch_size,
    )
