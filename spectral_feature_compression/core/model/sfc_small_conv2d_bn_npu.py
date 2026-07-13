"""NPU-friendly SFC-small rewrite with Conv2D and BatchNorm2D.

This module keeps the original SFC-small topology:

    complex STFT -> SFC encoder -> TF separator -> SFC decoder -> complex mask

The NPU rewrite removes cross-attention and Locoformer attention from the hot
path. Frequency compression/expansion uses Conv2D / stride-2 TransposedConv2D
pyramids, and temporal modeling uses causal Conv2D blocks on the compressed
band axis.
"""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

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


def _round_to_multiple(value: float, multiple: int = 8) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


def _interpolate_channels(start: int, end: int, n_stages: int) -> list[int]:
    if n_stages <= 0:
        raise ValueError(f"n_stages must be positive, got {n_stages}")
    channels = []
    for stage_idx in range(1, n_stages + 1):
        frac = stage_idx / n_stages
        channels.append(_round_to_multiple(start + (end - start) * frac))
    channels[-1] = int(end)
    return channels


def _compute_frequency_pyramid(n_freq: int, n_bands: int) -> tuple[list[int], list[int], list[int], list[int]]:
    """Return encoder and decoder frequency widths for stride-2 SFC transport."""

    if n_freq <= 1:
        raise ValueError(f"n_freq must be larger than one, got {n_freq}")
    if n_bands <= 0:
        raise ValueError(f"n_bands must be positive, got {n_bands}")
    if (n_freq - 1) % n_bands != 0:
        raise ValueError(f"Expected n_freq = n_bands * 2**N + 1, got n_freq={n_freq}, n_bands={n_bands}")
    scale = (n_freq - 1) // n_bands
    if scale <= 0 or scale & (scale - 1):
        raise ValueError(f"Expected power-of-two frequency scale, got {(n_freq - 1)} / {n_bands} = {scale}")
    n_stages = int(math.log2(scale))
    widths = [int(n_freq)]
    down_kernels: list[int] = []
    width = int(n_freq)
    for _ in range(n_stages):
        kernel_f = 3 if width % 2 == 1 else 2
        down_kernels.append(kernel_f)
        width = (width - kernel_f) // 2 + 1
        widths.append(width)
    if widths[-1] != n_bands:
        raise ValueError(f"Frequency pyramid ended at {widths[-1]} bands, expected {n_bands}")
    return widths, list(reversed(widths)), down_kernels, list(reversed(down_kernels))


def _normalize_dilation_cycle(n_layers: int, dilation_cycle: Sequence[int] | None) -> tuple[int, ...]:
    if dilation_cycle is None:
        dilation_cycle = (1,)
    cycle = tuple(int(value) for value in dilation_cycle)
    if not cycle:
        raise ValueError("dilation_cycle must not be empty")
    if any(value <= 0 for value in cycle):
        raise ValueError(f"dilation_cycle values must be positive, got {cycle}")
    return tuple(cycle[idx % len(cycle)] for idx in range(n_layers))


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
        return y, joined[:, :, -self.context_frames :, :]


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


class ConvTranspose2dBNAct(nn.Module):
    """TransposedConv2D followed by BatchNorm2D and optional ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_f: int,
        activation: bool = True,
    ) -> None:
        super().__init__()
        if kernel_f not in (2, 3):
            raise ValueError(f"Only valid stride-2 frequency kernels 2/3 are supported, got {kernel_f}")
        self.tconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=(1, int(kernel_f)),
            stride=(1, 2),
            padding=(0, 0),
            output_padding=(0, 0),
            bias=True,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=False) if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.tconv(x)))


class SFCSmallConv2DBNEncoder(nn.Module):
    """SFC encoder replacement: local input conv then stride-2 frequency transport."""

    def __init__(
        self,
        *,
        in_channels: int,
        d_inner: int,
        d_model: int,
        n_freq: int,
        n_bands: int,
        use_learnable_query: bool,
    ) -> None:
        super().__init__()
        enc_widths, _, down_kernels, _ = _compute_frequency_pyramid(n_freq, n_bands)
        out_channels = _interpolate_channels(d_inner, d_model, len(enc_widths) - 1)
        self.n_freq = int(n_freq)
        self.n_bands = int(n_bands)
        self.input_channels = int(in_channels)
        self.input = Conv2dBNAct(in_channels, d_inner, kernel_size=(1, 3), padding=(0, 1))
        stages: list[nn.Module] = []
        channels_in = d_inner
        for channels_out, kernel_f in zip(out_channels, down_kernels):
            stages.append(
                Conv2dBNAct(
                    channels_in,
                    channels_out,
                    kernel_size=(1, kernel_f),
                    stride=(1, 2),
                    padding=(0, 0),
                )
            )
            channels_in = channels_out
        self.down = nn.Sequential(*stages)
        self.band_query = (
            nn.Parameter(torch.zeros(1, d_model, 1, n_bands)) if use_learnable_query else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.down(self.input(x))
        if self.band_query is not None:
            h = h + self.band_query.to(dtype=h.dtype)
        return h

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
        h = self.down(h)
        if self.band_query is not None:
            h = h + self.band_query.to(dtype=h.dtype)
        return h, state


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
    """SFC decoder replacement: stride-2 frequency expansion then mask head."""

    def __init__(
        self,
        *,
        d_model: int,
        d_inner: int,
        out_channels: int,
        n_freq: int,
        n_bands: int,
        use_learnable_query: bool,
    ) -> None:
        super().__init__()
        _, dec_widths, _, up_kernels = _compute_frequency_pyramid(n_freq, n_bands)
        stage_count = len(dec_widths) - 1
        stage_channels = list(reversed(_interpolate_channels(d_inner, d_model, stage_count)))
        target_channels = stage_channels[1:] + [d_inner]
        stages: list[nn.Module] = []
        in_channels = d_model
        for out_ch, kernel_f in zip(target_channels, up_kernels):
            stages.append(
                ConvTranspose2dBNAct(
                    in_channels,
                    out_ch,
                    kernel_f=kernel_f,
                )
            )
            in_channels = out_ch
        self.up = nn.Sequential(*stages)
        self.freq_query = (
            nn.Parameter(torch.zeros(1, d_inner, 1, n_freq)) if use_learnable_query else None
        )
        self.output = nn.Conv2d(d_inner, out_channels, kernel_size=(1, 3), padding=(0, 1), bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.up(x)
        if self.freq_query is not None:
            h = h + self.freq_query.to(dtype=h.dtype)
        return self.output(h)


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
        n_bands: int = 64,
        n_src: int = 3,
        n_chan: int = 1,
        d_inner: int = 64,
        d_model: int = 160,
        n_separator_layers: int = 8,
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
        self.n_bands = int(n_bands)
        self.n_src = int(n_src)
        self.n_chan = int(n_chan)
        self.d_inner = int(d_inner)
        self.d_model = int(d_model)
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
        mask = self.decoder(h)
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
    d_inner: int = 64,
    d_model: int = 160,
    n_separator_layers: int = 8,
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
        n_bands=n_bands,
        n_src=n_src,
        n_chan=n_chan,
        d_inner=d_inner,
        d_model=d_model,
        n_separator_layers=n_separator_layers,
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
