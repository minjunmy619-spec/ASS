"""NPU-friendly SFC-small rewrite with faithful SFC frequency transport.

This module keeps the original SFC-small topology:

    complex STFT -> SFC encoder -> TF separator -> SFC decoder -> complex mask

The NPU rewrite keeps the SFC encoder/decoder idea: learned band/full-frequency
queries cross-attend over full-bin or compressed-band embeddings with musical
band position bias. Projections and FFNs are Conv2D + BatchNorm2D, and temporal
modeling remains causal Conv2D on the compressed band axis.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectral_feature_compression.core.model.bandit_split import get_band_specs
from spectral_feature_compression.core.model.model_wrapper import ModelWrapper


def _validate_kernel_span(kernel_size: int, dilation: int, *, name: str) -> None:
    if kernel_size <= 0:
        raise ValueError(f"{name} kernel_size must be positive, got {kernel_size}")
    if dilation <= 0:
        raise ValueError(f"{name} dilation must be positive, got {dilation}")
    span = (kernel_size - 1) * dilation
    if span > 14:
        raise ValueError(f"{name} violates NPU kernel/dilation span: ({kernel_size} - 1) * {dilation} = {span}")


def _validate_odd_kernel(kernel_size: int, *, name: str) -> None:
    if kernel_size % 2 != 1:
        raise ValueError(f"{name} must be odd for same-frequency padding, got {kernel_size}")


def _normalize_dilation_cycle(n_layers: int, dilation_cycle: Sequence[int] | None) -> tuple[int, ...]:
    if dilation_cycle is None:
        dilation_cycle = (1,)
    cycle = tuple(int(value) for value in dilation_cycle)
    if not cycle:
        raise ValueError("dilation_cycle must not be empty")
    if any(value <= 0 for value in cycle):
        raise ValueError(f"dilation_cycle values must be positive, got {cycle}")
    return tuple(cycle[idx % len(cycle)] for idx in range(n_layers))


def _resolve_n_fft(n_freq: int, n_fft: int | None) -> int:
    inferred = 2 * (int(n_freq) - 1)
    if n_fft is None:
        return inferred
    if int(n_fft) // 2 + 1 != int(n_freq):
        raise ValueError(f"n_fft={n_fft} is inconsistent with n_freq={n_freq}")
    return int(n_fft)


def _resolve_band_indices(
    *,
    n_freq: int,
    n_bands: int,
    n_fft: int | None,
    sample_rate: int,
    band_config: str,
) -> list[tuple[int, int]]:
    resolved_n_fft = _resolve_n_fft(n_freq, n_fft)
    band_indices, _freq_weights, _overlap = get_band_specs(
        band_config,
        resolved_n_fft,
        sample_rate,
        n_bands=n_bands,
    )
    if len(band_indices) != n_bands:
        raise ValueError(f"Expected {n_bands} bands from {band_config}, got {len(band_indices)}")
    clipped = [(max(0, int(start)), min(int(n_freq), int(end))) for start, end in band_indices]
    if any(end <= start for start, end in clipped):
        raise ValueError(f"{band_config}{n_bands} produced zero-width bands for n_freq={n_freq}")
    covered = torch.zeros(n_freq)
    for start, end in clipped:
        covered[start:end] += 1
    if torch.any(covered == 0):
        raise ValueError(f"{band_config}{n_bands} does not cover all {n_freq} frequency bins")
    return clipped


def _build_encoder_position_bias(band_indices: list[tuple[int, int]], n_freq: int) -> torch.Tensor:
    bias = torch.zeros(len(band_indices), n_freq)
    for band_idx, (start, end) in enumerate(band_indices):
        center = (start + end - 1) / 2.0
        denom = max((end - start) / 2.0, 1.0)
        for freq_idx in range(n_freq):
            if freq_idx < start:
                bias[band_idx, freq_idx] = float(freq_idx - start)
            elif freq_idx >= end:
                bias[band_idx, freq_idx] = float(end - 1 - freq_idx)
            else:
                bias[band_idx, freq_idx] = -abs(float(freq_idx) - center) / denom
    return bias


class CausalConv2dBNAct(nn.Module):
    """Causal temporal Conv2D followed by BatchNorm2D and optional ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: tuple[int, int] = (1, 3),
        dilation: tuple[int, int] = (1, 1),
        activation: bool = True,
    ) -> None:
        super().__init__()
        kt, kf = (int(kernel_size[0]), int(kernel_size[1]))
        dt, df = (int(dilation[0]), int(dilation[1]))
        _validate_kernel_span(kt, dt, name="time")
        _validate_kernel_span(kf, df, name="frequency")
        _validate_odd_kernel(kf, name="frequency kernel")
        self.context_frames = (kt - 1) * dt
        self.freq_pad = ((kf - 1) * df) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kt, kf),
            dilation=(dt, df),
            bias=True,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=False) if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.freq_pad, self.freq_pad, self.context_frames, 0))
        return self.act(self.bn(self.conv(x)))

    def init_stream_state(
        self,
        batch_size: int,
        *,
        freq_bins: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        in_channels = self.conv.in_channels
        return torch.zeros(batch_size, in_channels, self.context_frames, freq_bins, device=device, dtype=dtype)

    def forward_stream(self, x: torch.Tensor, state: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        if self.context_frames == 0:
            state = self.init_stream_state(x.shape[0], freq_bins=x.shape[-1], device=x.device, dtype=x.dtype)
            padded = F.pad(x, (self.freq_pad, self.freq_pad, 0, 0)) if self.freq_pad > 0 else x
            return self.act(self.bn(self.conv(padded))), state
        if state is None:
            state = self.init_stream_state(x.shape[0], freq_bins=x.shape[-1], device=x.device, dtype=x.dtype)
        joined = torch.cat((state, x), dim=2)
        padded = F.pad(joined, (self.freq_pad, self.freq_pad, 0, 0)) if self.freq_pad > 0 else joined
        y = self.act(self.bn(self.conv(padded)))
        if self.context_frames == 1 and (torch.jit.is_tracing() or x.shape[2] == 1):
            next_state = x
        else:
            next_state = joined[:, :, -self.context_frames :, :]
        return y, next_state


class Conv2dBNAct(nn.Module):
    """Conv2D followed by BatchNorm2D and optional ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: tuple[int, int] = (1, 1),
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (0, 0),
        activation: bool = True,
    ) -> None:
        super().__init__()
        kt, kf = (int(kernel_size[0]), int(kernel_size[1]))
        _validate_kernel_span(kt, 1, name="time")
        _validate_kernel_span(kf, 1, name="frequency")
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kt, kf),
            stride=stride,
            padding=padding,
            bias=True,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=False) if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class SFCSmallConv2DBNEncoder(nn.Module):
    """SFC encoder: full-frequency embeddings queried by musical band tokens."""

    def __init__(
        self,
        *,
        in_channels: int,
        d_inner: int,
        d_model: int,
        n_freq: int,
        n_bands: int,
        n_fft: int | None,
        sample_rate: int,
        band_config: str,
        n_heads: int,
        learnable_pos_bias: bool,
        use_learnable_query: bool,
    ) -> None:
        super().__init__()
        if d_inner % n_heads != 0:
            raise ValueError(f"d_inner={d_inner} must be divisible by n_heads={n_heads}")
        self.n_freq = int(n_freq)
        self.n_bands = int(n_bands)
        self.n_heads = int(n_heads)
        self.head_dim = int(d_inner) // int(n_heads)
        self.scale = self.head_dim**-0.5
        self.query_type = "learnable" if use_learnable_query else "adaptive"
        self.input_channels = int(in_channels)
        self.input = Conv2dBNAct(in_channels, d_inner, kernel_size=(1, 3), padding=(0, 1))
        self.kv_proj = nn.Conv2d(d_inner, 2 * d_inner, kernel_size=1, bias=False)
        self.aggregate = Conv2dBNAct(d_inner, d_inner, activation=False)
        self.ffn = nn.Sequential(
            Conv2dBNAct(d_inner, 2 * d_inner, activation=True),
            Conv2dBNAct(2 * d_inner, d_inner, activation=False),
        )
        self.output = Conv2dBNAct(d_inner, d_model, kernel_size=(1, 3), padding=(0, 1))

        band_indices = _resolve_band_indices(
            n_freq=n_freq,
            n_bands=n_bands,
            n_fft=n_fft,
            sample_rate=sample_rate,
            band_config=band_config,
        )
        pos_bias = _build_encoder_position_bias(band_indices, n_freq).unsqueeze(0).repeat(n_heads, 1, 1)
        if learnable_pos_bias:
            self.pos_bias = nn.Parameter(pos_bias)
        else:
            self.register_buffer("pos_bias", pos_bias)

        query = torch.randn(n_heads, n_bands, self.head_dim) * 0.02
        if self.query_type == "learnable":
            self.query = nn.Parameter(query)
        else:
            self.register_buffer("query", query)
        weights = torch.softmax(_build_encoder_position_bias(band_indices, n_freq), dim=-1)
        self.register_buffer("adaptive_pool", weights)

    def _prepare_query(self, emb_flat: torch.Tensor) -> list[torch.Tensor]:
        if self.query_type == "learnable":
            return [self.query[head_idx].unsqueeze(0).to(dtype=emb_flat.dtype) for head_idx in range(self.n_heads)]

        # Adaptive mode mirrors the official weighted band-mean query path with a
        # static musical-band pooling prior to avoid gather/index_add in export.
        pooled = torch.matmul(self.adaptive_pool.to(dtype=emb_flat.dtype).unsqueeze(0), emb_flat.transpose(1, 2))
        return [
            pooled[:, :, head_idx * self.head_dim : (head_idx + 1) * self.head_dim]
            for head_idx in range(self.n_heads)
        ]

    def _attend(self, h: torch.Tensor) -> torch.Tensor:
        bsz, _channels, n_frames, n_freq = h.shape
        kv = self.kv_proj(h)
        key, value = kv.chunk(2, dim=1)
        batch_frames = bsz * n_frames
        key_flat = key.permute(0, 2, 1, 3).reshape(batch_frames, -1, n_freq)
        value_flat = value.permute(0, 2, 3, 1).reshape(batch_frames, n_freq, -1)
        emb_flat = h.permute(0, 2, 1, 3).reshape(batch_frames, -1, n_freq)
        queries = self._prepare_query(emb_flat)

        head_outputs: list[torch.Tensor] = []
        pos_bias = self.pos_bias.to(dtype=h.dtype)
        for head_idx in range(self.n_heads):
            start = head_idx * self.head_dim
            end = start + self.head_dim
            key_h = key_flat[:, start:end, :]
            value_h = value_flat[:, :, start:end]
            score = torch.matmul(queries[head_idx], key_h) * self.scale
            score = score + pos_bias[head_idx : head_idx + 1]
            weight = torch.softmax(score, dim=-1)
            head_outputs.append(torch.matmul(weight, value_h))

        attended = torch.cat(head_outputs, dim=-1)
        attended = attended.transpose(1, 2).reshape(bsz, n_frames, -1, self.n_bands).permute(0, 2, 1, 3)
        attended = self.aggregate(attended)
        return attended + self.ffn(attended)

    def _attend_stream_frame(self, h: torch.Tensor) -> torch.Tensor:
        bsz, _channels, _n_frames, n_freq = h.shape
        kv = self.kv_proj(h)
        key, value = kv.chunk(2, dim=1)
        key = key.reshape(bsz, self.n_heads, self.head_dim, n_freq)
        value = value.reshape(bsz, self.n_heads, self.head_dim, n_freq).transpose(2, 3)

        if self.query_type == "learnable":
            query = self.query.unsqueeze(0).to(dtype=h.dtype)
        else:
            emb = h.reshape(bsz, self.n_heads, self.head_dim, n_freq).transpose(2, 3)
            pool = self.adaptive_pool.reshape(1, 1, self.n_bands, n_freq).to(dtype=h.dtype)
            query = torch.matmul(pool, emb)

        score = torch.matmul(query, key) * self.scale
        score = score + self.pos_bias.unsqueeze(0).to(dtype=h.dtype)
        weight = torch.softmax(score, dim=-1)
        attended = torch.matmul(weight, value)
        attended = attended.transpose(2, 3).reshape(bsz, -1, 1, self.n_bands)
        attended = self.aggregate(attended)
        return attended + self.ffn(attended)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self._attend(self.input(x)))

    def init_stream_state(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        return torch.zeros(batch_size, self.input_channels, 0, self.n_freq, device=device, dtype=dtype)

    def forward_stream(self, x: torch.Tensor, state: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.input(x)
        if state is None:
            state = self.init_stream_state(x.shape[0], device=x.device, dtype=x.dtype)
        h = self._attend(h) if not torch.jit.is_tracing() and h.shape[2] != 1 else self._attend_stream_frame(h)
        return self.output(h), state


class Conv2DLocoBNBlock(nn.Module):
    """Conv2D replacement for one TF-Locoformer block."""

    def __init__(
        self,
        channels: int,
        *,
        time_kernel_size: int,
        time_dilation: int,
        freq_kernel_size: int,
        ffn_expansion: int,
    ) -> None:
        super().__init__()
        _validate_odd_kernel(freq_kernel_size, name="freq_kernel_size")
        hidden = int(channels) * int(ffn_expansion)
        self.freq_mix = nn.Sequential(
            Conv2dBNAct(
                channels,
                channels,
                kernel_size=(1, freq_kernel_size),
                padding=(0, freq_kernel_size // 2),
            ),
            Conv2dBNAct(channels, channels, activation=False),
        )
        self.time_mix = CausalConv2dBNAct(
            channels,
            channels,
            kernel_size=(time_kernel_size, 1),
            dilation=(time_dilation, 1),
        )
        self.time_proj = Conv2dBNAct(channels, channels, activation=False)
        self.ffn = nn.Sequential(
            Conv2dBNAct(channels, hidden, activation=True),
            Conv2dBNAct(hidden, channels, activation=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.freq_mix(x)
        y = self.time_mix(x)
        x = x + self.time_proj(y)
        return x + self.ffn(x)

    def init_stream_state(
        self,
        batch_size: int,
        *,
        freq_bins: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        return self.time_mix.init_stream_state(batch_size, freq_bins=freq_bins, device=device, dtype=dtype)

    def forward_stream(self, x: torch.Tensor, state: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.freq_mix(x)
        y, state = self.time_mix.forward_stream(x, state)
        x = x + self.time_proj(y)
        return x + self.ffn(x), state


class SFCSmallConv2DBNDecoder(nn.Module):
    """SFC decoder: full-frequency queries read compressed band tokens."""

    def __init__(
        self,
        *,
        d_model: int,
        d_inner: int,
        out_channels: int,
        n_freq: int,
        n_bands: int,
        n_fft: int | None,
        sample_rate: int,
        band_config: str,
        n_heads: int,
        learnable_pos_bias: bool,
        use_learnable_query: bool,
    ) -> None:
        super().__init__()
        if d_inner % n_heads != 0:
            raise ValueError(f"d_inner={d_inner} must be divisible by n_heads={n_heads}")
        self.n_freq = int(n_freq)
        self.n_bands = int(n_bands)
        self.n_heads = int(n_heads)
        self.head_dim = int(d_inner) // int(n_heads)
        self.scale = self.head_dim**-0.5
        self.query_type = "learnable" if use_learnable_query else "adaptive"
        self.input = Conv2dBNAct(d_model, d_inner, kernel_size=(1, 3), padding=(0, 1))
        self.kv_proj = nn.Conv2d(d_inner, 2 * d_inner, kernel_size=1, bias=False)
        self.aggregate = Conv2dBNAct(d_inner, d_inner, activation=False)
        self.ffn = nn.Sequential(
            Conv2dBNAct(d_inner, 2 * d_inner, activation=True),
            Conv2dBNAct(2 * d_inner, d_inner, activation=False),
        )
        self.output = nn.Conv2d(d_inner, out_channels, kernel_size=(1, 3), padding=(0, 1), bias=True)

        band_indices = _resolve_band_indices(
            n_freq=n_freq,
            n_bands=n_bands,
            n_fft=n_fft,
            sample_rate=sample_rate,
            band_config=band_config,
        )
        enc_bias = _build_encoder_position_bias(band_indices, n_freq)
        pos_bias = enc_bias.transpose(0, 1).unsqueeze(0).repeat(n_heads, 1, 1)
        if learnable_pos_bias:
            self.pos_bias = nn.Parameter(pos_bias)
        else:
            self.register_buffer("pos_bias", pos_bias)

        query = torch.randn(n_heads, n_freq, self.head_dim) * 0.02
        if self.query_type == "learnable":
            self.query = nn.Parameter(query)
        else:
            self.register_buffer("query", query)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input(x)
        bsz, _channels, n_frames, n_bands = h.shape
        kv = self.kv_proj(h)
        key, value = kv.chunk(2, dim=1)
        batch_frames = bsz * n_frames
        key_flat = key.permute(0, 2, 1, 3).reshape(batch_frames, -1, n_bands)
        value_flat = value.permute(0, 2, 3, 1).reshape(batch_frames, n_bands, -1)

        head_outputs: list[torch.Tensor] = []
        pos_bias = self.pos_bias.to(dtype=h.dtype)
        for head_idx in range(self.n_heads):
            start = head_idx * self.head_dim
            end = start + self.head_dim
            key_h = key_flat[:, start:end, :]
            value_h = value_flat[:, :, start:end]
            query_h = self.query[head_idx].unsqueeze(0).to(dtype=h.dtype)
            score = torch.matmul(query_h, key_h) * self.scale
            score = score + pos_bias[head_idx : head_idx + 1]
            weight = torch.softmax(score, dim=-1)
            head_outputs.append(torch.matmul(weight, value_h))

        expanded = torch.cat(head_outputs, dim=-1)
        expanded = expanded.transpose(1, 2).reshape(bsz, n_frames, -1, self.n_freq).permute(0, 2, 1, 3)
        expanded = self.aggregate(expanded)
        expanded = expanded + self.ffn(expanded)
        return self.output(expanded)

    def forward_stream(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input(x)
        if not torch.jit.is_tracing() and h.shape[2] != 1:
            return self.forward(x)
        bsz, _channels, _n_frames, n_bands = h.shape
        kv = self.kv_proj(h)
        key, value = kv.chunk(2, dim=1)
        key = key.reshape(bsz, self.n_heads, self.head_dim, n_bands)
        value = value.reshape(bsz, self.n_heads, self.head_dim, n_bands).transpose(2, 3)

        query = self.query.unsqueeze(0).to(dtype=h.dtype)
        score = torch.matmul(query, key) * self.scale
        score = score + self.pos_bias.unsqueeze(0).to(dtype=h.dtype)
        weight = torch.softmax(score, dim=-1)
        expanded = torch.matmul(weight, value)
        expanded = expanded.transpose(2, 3).reshape(bsz, -1, 1, self.n_freq)
        expanded = self.aggregate(expanded)
        expanded = expanded + self.ffn(expanded)
        return self.output(expanded)


def _apply_packed_complex_mask(x: torch.Tensor, mask: torch.Tensor, *, n_src: int, n_chan: int) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    for src_idx in range(n_src):
        for chan_idx in range(n_chan):
            in_base = 2 * chan_idx
            mask_base = 2 * (src_idx * n_chan + chan_idx)
            in_r = x[:, in_base : in_base + 1, :, :]
            in_i = x[:, in_base + 1 : in_base + 2, :, :]
            mask_r = mask[:, mask_base : mask_base + 1, :, :]
            mask_i = mask[:, mask_base + 1 : mask_base + 2, :, :]
            outputs.append(in_r * mask_r - in_i * mask_i)
            outputs.append(in_r * mask_i + in_i * mask_r)
    return torch.cat(outputs, dim=1)


class SFCSmallConv2DBNNPUCore(nn.Module):
    """Packed-real SFC-small NPU core operating on ``[B, 2*M, T, F]``."""

    def __init__(
        self,
        *,
        n_freq: int,
        n_fft: int | None = None,
        sample_rate: int = 44100,
        n_bands: int = 64,
        band_config: str = "musical",
        n_src: int = 3,
        n_chan: int = 1,
        d_inner: int = 64,
        d_model: int = 160,
        n_separator_layers: int = 8,
        n_sfc_heads: int = 4,
        learnable_pos_bias: bool = True,
        time_kernel_size: int = 2,
        freq_kernel_size: int = 3,
        ffn_expansion: int = 4,
        dilation_cycle: Sequence[int] | None = None,
        masking: bool = True,
        use_learnable_query: bool = True,
    ) -> None:
        super().__init__()
        if n_src <= 0:
            raise ValueError(f"n_src must be positive, got {n_src}")
        if n_chan <= 0:
            raise ValueError(f"n_chan must be positive, got {n_chan}")
        if d_inner <= 0 or d_model <= 0:
            raise ValueError(f"d_inner and d_model must be positive, got {d_inner}, {d_model}")
        self.n_freq = int(n_freq)
        self.n_fft = _resolve_n_fft(n_freq, n_fft)
        self.sample_rate = int(sample_rate)
        self.n_bands = int(n_bands)
        self.band_config = str(band_config)
        self.n_src = int(n_src)
        self.n_chan = int(n_chan)
        self.d_inner = int(d_inner)
        self.d_model = int(d_model)
        self.n_sfc_heads = int(n_sfc_heads)
        self.n_separator_layers = int(n_separator_layers)
        self.masking = bool(masking)
        self.dilation_schedule = _normalize_dilation_cycle(self.n_separator_layers, dilation_cycle)

        in_channels = 2 * self.n_chan
        out_channels = 2 * self.n_src * self.n_chan
        self.encoder = SFCSmallConv2DBNEncoder(
            in_channels=in_channels,
            d_inner=d_inner,
            d_model=d_model,
            n_freq=n_freq,
            n_bands=n_bands,
            n_fft=self.n_fft,
            sample_rate=sample_rate,
            band_config=band_config,
            n_heads=n_sfc_heads,
            learnable_pos_bias=learnable_pos_bias,
            use_learnable_query=use_learnable_query,
        )
        self.separator = nn.ModuleList(
            [
                Conv2DLocoBNBlock(
                    d_model,
                    time_kernel_size=time_kernel_size,
                    time_dilation=dilation,
                    freq_kernel_size=freq_kernel_size,
                    ffn_expansion=ffn_expansion,
                )
                for dilation in self.dilation_schedule
            ]
        )
        self.decoder = SFCSmallConv2DBNDecoder(
            d_model=d_model,
            d_inner=d_inner,
            out_channels=out_channels,
            n_freq=n_freq,
            n_bands=n_bands,
            n_fft=self.n_fft,
            sample_rate=sample_rate,
            band_config=band_config,
            n_heads=n_sfc_heads,
            learnable_pos_bias=learnable_pos_bias,
            use_learnable_query=use_learnable_query,
        )
        self._init_mask_bias()

    def _init_mask_bias(self) -> None:
        if self.decoder.output.bias is None:
            return
        with torch.no_grad():
            self.decoder.output.bias.zero_()
            for src_idx in range(self.n_src):
                for chan_idx in range(self.n_chan):
                    self.decoder.output.bias[2 * (src_idx * self.n_chan + chan_idx)] = 1.0 / self.n_src

    def forward(self, x: torch.Tensor, return_mask: bool = False):
        if not torch.jit.is_tracing():
            if x.ndim != 4:
                raise RuntimeError(f"Expected packed STFT [B,C,T,F], got {tuple(x.shape)}")
            if x.shape[1] != 2 * self.n_chan:
                raise RuntimeError(f"Expected {2 * self.n_chan} channels, got {x.shape[1]}")
            if x.shape[-1] != self.n_freq:
                raise RuntimeError(f"Expected {self.n_freq} frequency bins, got {x.shape[-1]}")
        h = self.encoder(x)
        for block in self.separator:
            h = block(h)
        mask = self.decoder(h)
        y = _apply_packed_complex_mask(x, mask, n_src=self.n_src, n_chan=self.n_chan) if self.masking else mask
        if not torch.jit.is_tracing() and return_mask:
            return y, mask
        return y

    def init_stream_state(
        self,
        batch_size: int = 1,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, ...]:
        state: list[torch.Tensor] = [
            block.init_stream_state(batch_size, freq_bins=self.n_bands, device=device, dtype=dtype)
            for block in self.separator
        ]
        return tuple(state)

    def forward_stream(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, ...] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if state is None:
            state = self.init_stream_state(x.shape[0], device=x.device, dtype=x.dtype)
        if len(state) != len(self.separator):
            raise RuntimeError(f"Expected {len(self.separator)} state tensors, got {len(state)}")
        h, _ = self.encoder.forward_stream(x, None)
        next_state: list[torch.Tensor] = []
        for block, block_state in zip(self.separator, state):
            h, new_block_state = block.forward_stream(h, block_state)
            next_state.append(new_block_state)
        mask = self.decoder.forward_stream(h)
        y = _apply_packed_complex_mask(x, mask, n_src=self.n_src, n_chan=self.n_chan) if self.masking else mask
        return y, tuple(next_state)

    def state_size_bytes(self, *, batch_size: int = 1, dtype: torch.dtype = torch.float16) -> int:
        itemsize = torch.empty((), dtype=dtype).element_size()
        return sum(state.numel() * itemsize for state in self.init_stream_state(batch_size=batch_size, dtype=dtype))


class SFCSmallConv2DBNNPUModel(nn.Module):
    """Complex-STFT wrapper compatible with ``ModelWrapper``."""

    def __init__(self, **core_kwargs) -> None:
        super().__init__()
        self.core = SFCSmallConv2DBNNPUCore(**core_kwargs)
        self.n_src = self.core.n_src
        self.n_chan = self.core.n_chan

    @staticmethod
    def _pack_complex(x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(-2, -1)
        parts: list[torch.Tensor] = []
        for chan_idx in range(x.shape[1]):
            parts.append(x[:, chan_idx : chan_idx + 1].real)
            parts.append(x[:, chan_idx : chan_idx + 1].imag)
        return torch.cat(parts, dim=1)

    def _unpack_complex(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, n_frames, n_freq = x.shape
        x = x.reshape(bsz, self.n_src, self.n_chan, 2, n_frames, n_freq)
        y = torch.complex(x[:, :, :, 0], x[:, :, :, 1])
        return y.transpose(-1, -2)

    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        kwargs.pop("ref", None)
        packed = self._pack_complex(input)
        output = self.core(packed)
        return self._unpack_complex(output)


def build_sfc_small_conv2d_bn_npu_system(
    *,
    n_fft: int,
    hop_length: int,
    fs: int,
    n_src: int = 3,
    n_chan: int = 1,
    n_bands: int = 64,
    band_config: str = "musical",
    d_inner: int = 64,
    d_model: int = 160,
    n_separator_layers: int = 8,
    n_sfc_heads: int = 4,
    learnable_pos_bias: bool = True,
    time_kernel_size: int = 2,
    freq_kernel_size: int = 3,
    ffn_expansion: int = 4,
    dilation_cycle: Sequence[int] | None = None,
    masking: bool = True,
    use_learnable_query: bool = True,
    scaling: bool = False,
    css_segment_size: int = 12,
    css_shift_size: int = 6,
    css_batch_size: int = 1,
) -> ModelWrapper:
    core_model = SFCSmallConv2DBNNPUModel(
        n_freq=n_fft // 2 + 1,
        n_fft=n_fft,
        sample_rate=fs,
        n_bands=n_bands,
        band_config=band_config,
        n_src=n_src,
        n_chan=n_chan,
        d_inner=d_inner,
        d_model=d_model,
        n_separator_layers=n_separator_layers,
        n_sfc_heads=n_sfc_heads,
        learnable_pos_bias=learnable_pos_bias,
        time_kernel_size=time_kernel_size,
        freq_kernel_size=freq_kernel_size,
        ffn_expansion=ffn_expansion,
        dilation_cycle=dilation_cycle,
        masking=masking,
        use_learnable_query=use_learnable_query,
    )
    return ModelWrapper(
        model=core_model,
        n_fft=n_fft,
        hop_length=hop_length,
        fs=fs,
        scaling=scaling,
        css_segment_size=css_segment_size,
        css_shift_size=css_shift_size,
        css_batch_size=css_batch_size,
    )
