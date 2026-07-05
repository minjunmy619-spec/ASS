"""Conv2D frequency-pyramid NPU separator for TV speech/music/effects.

This core intentionally does not use the repo's SFC/BandSplit primitives.  It
tests a different deployment hypothesis: keep the graph mostly on Conv2D and
stride-2 ConvTranspose2D fast paths, spend parameters in large 1x1 matrices at
the compressed bottleneck, and avoid attention/softmax/BMM in the student.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectral_feature_compression.core.model.frequency_preprocessing import (
    FrequencyPreprocessedOnlineModel,
    build_frequency_preprocessor,
    build_pcen_preprocessor,
    resolve_preprocessed_n_freq,
)
from spectral_feature_compression.core.model.online_model_wrapper import OnlineModelWrapper
from spectral_feature_compression.core.model.source_separation_postprocess import (
    build_misi_phase_consistency,
    build_source_separation_postprocessor,
)


def _validate_kernel(kernel_size: int, dilation: int, *, axis: str, limit: int = 14) -> None:
    span = (int(kernel_size) - 1) * int(dilation)
    if span > limit:
        raise ValueError(
            f"NPU constraint violated on {axis} axis: "
            f"(kernel_size - 1) * dilation = ({kernel_size} - 1) * {dilation} = {span} > {limit}"
        )


def _as_int_tuple(value: Sequence[int] | None, *, default: Sequence[int], name: str) -> tuple[int, ...]:
    result = tuple(int(v) for v in (default if value is None else value))
    if not result:
        raise ValueError(f"{name} must not be empty")
    if any(v <= 0 for v in result):
        raise ValueError(f"{name} values must be positive, got {result}")
    return result


class ChannelAffine2d(nn.Module):
    """Per-channel affine scale without runtime reductions."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class CausalDepthwiseConv2d(nn.Module):
    """Depthwise Conv2D with causal time padding and optional band padding."""

    def __init__(
        self,
        channels: int,
        *,
        time_kernel: int,
        time_dilation: int,
        band_kernel: int = 1,
    ) -> None:
        super().__init__()
        if band_kernel <= 0 or band_kernel % 2 != 1:
            raise ValueError(f"band_kernel must be a positive odd integer, got {band_kernel}")
        _validate_kernel(time_kernel, time_dilation, axis="time")
        _validate_kernel(band_kernel, 1, axis="band")
        self.pad_t = (int(time_kernel) - 1) * int(time_dilation)
        self.pad_f = int(band_kernel) // 2
        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=(int(time_kernel), int(band_kernel)),
            dilation=(int(time_dilation), 1),
            padding=(0, self.pad_f),
            groups=channels,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad_t > 0:
            x = F.pad(x, (0, 0, self.pad_t, 0))
        return self.conv(x)

    def init_stream_state(self, batch_size: int, *, freq_bins: int, device=None, dtype=None) -> torch.Tensor:
        return torch.zeros(batch_size, self.conv.in_channels, self.pad_t, freq_bins, device=device, dtype=dtype)

    def forward_stream(self, x: torch.Tensor, state: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        if self.pad_t == 0:
            return self.conv(x), x[:, :, :0, :]
        if state is None:
            state = self.init_stream_state(x.shape[0], freq_bins=x.shape[-1], device=x.device, dtype=x.dtype)
        full = torch.cat([state, x], dim=2)
        new_state = full[:, :, -self.pad_t :, :]
        return self.conv(full), new_state


class ConvDownsample2d(nn.Module):
    """Frequency stride-2 Conv2D downsampler."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, 3),
            stride=(1, 2),
            padding=(0, 1),
            bias=True,
        )
        self.affine = ChannelAffine2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.affine(self.conv(x)))


class ConvUpsample2d(nn.Module):
    """Frequency stride-2 ConvTranspose2D upsampler with exact target width."""

    def __init__(self, in_channels: int, out_channels: int, *, output_padding: int) -> None:
        super().__init__()
        if output_padding not in {0, 1}:
            raise ValueError(f"output_padding must be 0 or 1, got {output_padding}")
        self.deconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=(1, 3),
            stride=(1, 2),
            padding=(0, 1),
            output_padding=(0, output_padding),
            bias=True,
        )
        self.affine = ChannelAffine2d(out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.affine(self.deconv(x))
        if x.shape[-1] != skip.shape[-1]:
            raise ValueError(f"Upsample width mismatch: {x.shape[-1]} vs {skip.shape[-1]}")
        return F.relu(x + skip)


class ResizeConvUpsample2d(nn.Module):
    """Frequency nearest-resize upsampler followed by regular Conv2D."""

    def __init__(self, in_channels: int, out_channels: int, *, target_freq: int) -> None:
        super().__init__()
        if target_freq <= 0:
            raise ValueError(f"target_freq must be positive, got {target_freq}")
        self.target_freq = int(target_freq)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, 3),
            padding=(0, 1),
            bias=True,
        )
        self.affine = ChannelAffine2d(out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=(1.0, 2.0), mode="nearest")
        if x.shape[-1] > self.target_freq:
            x = x[:, :, :, : self.target_freq]
        if x.shape[-1] != skip.shape[-1]:
            raise ValueError(f"Upsample width mismatch: {x.shape[-1]} vs {skip.shape[-1]}")
        x = self.affine(self.conv(x))
        return F.relu(x + skip)


class BottleneckCapacity2d(nn.Module):
    """Large 1x1 bottleneck MLP at compressed frequency width."""

    def __init__(self, channels: int, hidden_channels: int) -> None:
        super().__init__()
        if hidden_channels <= 0:
            raise ValueError(f"hidden_channels must be positive, got {hidden_channels}")
        self.up = nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=True)
        self.down = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=3, keepdim=True)
        return x + self.down(F.relu(self.up(pooled))) * self.scale


class ConvPyramidTemporalBlock2d(nn.Module):
    """Causal bottleneck block with depthwise temporal-band mixing."""

    def __init__(
        self,
        channels: int,
        *,
        expansion: int,
        time_kernel: int,
        time_dilation: int,
        band_kernel: int,
        capacity_hidden: int,
    ) -> None:
        super().__init__()
        hidden = int(channels * expansion)
        self.hidden = hidden
        self.affine = ChannelAffine2d(channels)
        self.dw = CausalDepthwiseConv2d(
            channels,
            time_kernel=time_kernel,
            time_dilation=time_dilation,
            band_kernel=band_kernel,
        )
        self.expand = nn.Conv2d(channels, 2 * hidden, kernel_size=1, bias=True)
        self.project = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
        self.capacity = BottleneckCapacity2d(channels, capacity_hidden) if capacity_hidden > 0 else None
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dw(self.affine(x))
        value, gate = torch.split(self.expand(F.relu(y)), self.hidden, dim=1)
        y = value * torch.sigmoid(gate)
        x = x + self.project(F.relu(y)) * self.scale
        if self.capacity is not None:
            x = self.capacity(x)
        return x

    def stream_context_frames(self) -> int:
        return self.dw.pad_t

    def init_stream_state(self, batch_size: int, *, freq_bins: int, device=None, dtype=None) -> torch.Tensor:
        return self.dw.init_stream_state(batch_size, freq_bins=freq_bins, device=device, dtype=dtype)

    def forward_stream(self, x: torch.Tensor, state: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        y, new_state = self.dw.forward_stream(self.affine(x), state)
        value, gate = torch.split(self.expand(F.relu(y)), self.hidden, dim=1)
        y = value * torch.sigmoid(gate)
        x = x + self.project(F.relu(y)) * self.scale
        if self.capacity is not None:
            x = self.capacity(x)
        return x, new_state


class BottleneckConvGRU2d(nn.Module):
    """Single-frame ConvGRU bottleneck cell using Conv2D/elementwise ops only."""

    def __init__(self, channels: int, *, band_kernel: int = 3) -> None:
        super().__init__()
        if band_kernel <= 0 or band_kernel % 2 != 1:
            raise ValueError(f"band_kernel must be a positive odd integer, got {band_kernel}")
        _validate_kernel(band_kernel, 1, axis="recurrent frequency")
        self.channels = int(channels)
        self.x_affine = ChannelAffine2d(channels)
        self.h_affine = ChannelAffine2d(channels)
        self.h_mix = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, int(band_kernel)),
            padding=(0, int(band_kernel) // 2),
            groups=channels,
            bias=True,
        )
        self.x_proj = nn.Conv2d(channels, 3 * channels, kernel_size=1, bias=True)
        self.h_proj = nn.Conv2d(channels, 3 * channels, kernel_size=1, bias=True)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def init_stream_state(self, batch_size: int, *, freq_bins: int, device=None, dtype=None) -> torch.Tensor:
        return torch.zeros(batch_size, self.channels, 1, freq_bins, device=device, dtype=dtype)

    def forward_frame(self, x: torch.Tensor, state: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4 or x.shape[2] != 1:
            raise ValueError(f"BottleneckConvGRU2d expects [B,C,1,F], got {tuple(x.shape)}")
        if state is None:
            state = self.init_stream_state(x.shape[0], freq_bins=x.shape[-1], device=x.device, dtype=x.dtype)
        if state.shape != (x.shape[0], self.channels, 1, x.shape[-1]):
            raise ValueError(f"Invalid GRU state shape {tuple(state.shape)} for input {tuple(x.shape)}")

        x_gates = self.x_proj(F.relu(self.x_affine(x)))
        h_gates = self.h_proj(self.h_mix(self.h_affine(state)))
        xr, xz, xn = torch.split(x_gates, self.channels, dim=1)
        hr, hz, hn = torch.split(h_gates, self.channels, dim=1)
        reset = torch.sigmoid(xr + hr)
        update = torch.sigmoid(xz + hz)
        candidate = torch.tanh(xn + reset * hn)
        new_state = (1.0 - update) * candidate + update * state
        y = x + self.out_proj(F.relu(new_state)) * self.scale
        return y, new_state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = None
        frames = []
        for frame_idx in range(x.shape[2]):
            y, state = self.forward_frame(x[:, :, frame_idx : frame_idx + 1, :], state)
            frames.append(y)
        return torch.cat(frames, dim=2)

    def forward_stream(self, x: torch.Tensor, state: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        state_t = state
        frames = []
        for frame_idx in range(x.shape[2]):
            y, state_t = self.forward_frame(x[:, :, frame_idx : frame_idx + 1, :], state_t)
            frames.append(y)
        return torch.cat(frames, dim=2), state_t


class BottleneckConvLSTM2d(nn.Module):
    """Single-frame ConvLSTM bottleneck cell using Conv2D/elementwise ops only."""

    def __init__(self, channels: int, *, band_kernel: int = 3) -> None:
        super().__init__()
        if band_kernel <= 0 or band_kernel % 2 != 1:
            raise ValueError(f"band_kernel must be a positive odd integer, got {band_kernel}")
        _validate_kernel(band_kernel, 1, axis="recurrent frequency")
        self.channels = int(channels)
        self.x_affine = ChannelAffine2d(channels)
        self.h_affine = ChannelAffine2d(channels)
        self.h_mix = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, int(band_kernel)),
            padding=(0, int(band_kernel) // 2),
            groups=channels,
            bias=True,
        )
        self.x_proj = nn.Conv2d(channels, 4 * channels, kernel_size=1, bias=True)
        self.h_proj = nn.Conv2d(channels, 4 * channels, kernel_size=1, bias=True)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def init_stream_state(
        self,
        batch_size: int,
        *,
        freq_bins: int,
        device=None,
        dtype=None,
    ) -> tuple[torch.Tensor, ...]:
        hidden = torch.zeros(batch_size, self.channels, 1, freq_bins, device=device, dtype=dtype)
        cell = torch.zeros_like(hidden)
        return hidden, cell

    def forward_frame(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if x.ndim != 4 or x.shape[2] != 1:
            raise ValueError(f"BottleneckConvLSTM2d expects [B,C,1,F], got {tuple(x.shape)}")
        if state is None:
            state = self.init_stream_state(x.shape[0], freq_bins=x.shape[-1], device=x.device, dtype=x.dtype)
        hidden, cell = state
        expected_shape = (x.shape[0], self.channels, 1, x.shape[-1])
        if hidden.shape != expected_shape or cell.shape != expected_shape:
            raise ValueError(
                f"Invalid LSTM state shapes {tuple(hidden.shape)}, {tuple(cell.shape)} for input {tuple(x.shape)}"
            )

        x_gates = self.x_proj(F.relu(self.x_affine(x)))
        h_gates = self.h_proj(self.h_mix(self.h_affine(hidden)))
        xi, xf, xg, xo = torch.split(x_gates, self.channels, dim=1)
        hi, hf, hg, ho = torch.split(h_gates, self.channels, dim=1)
        in_gate = torch.sigmoid(xi + hi)
        forget_gate = torch.sigmoid(xf + hf)
        candidate = torch.tanh(xg + hg)
        out_gate = torch.sigmoid(xo + ho)
        new_cell = forget_gate * cell + in_gate * candidate
        new_hidden = out_gate * torch.tanh(new_cell)
        y = x + self.out_proj(F.relu(new_hidden)) * self.scale
        return y, (new_hidden, new_cell)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = None
        frames = []
        for frame_idx in range(x.shape[2]):
            y, state = self.forward_frame(x[:, :, frame_idx : frame_idx + 1, :], state)
            frames.append(y)
        return torch.cat(frames, dim=2)

    def forward_stream(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        state_t = state
        frames = []
        for frame_idx in range(x.shape[2]):
            y, state_t = self.forward_frame(x[:, :, frame_idx : frame_idx + 1, :], state_t)
            frames.append(y)
        return torch.cat(frames, dim=2), state_t


def _state_numel(state) -> int:
    if isinstance(state, torch.Tensor):
        return int(state.numel())
    if isinstance(state, (tuple, list)):
        return sum(_state_numel(item) for item in state)
    raise TypeError(f"Unsupported state leaf type: {type(state)!r}")


class MaskLogitTemporalSmoother2d(nn.Module):
    """Causal moving-average blend for mask logits."""

    def __init__(self, channels: int, *, kernel_size: int = 1, blend: float = 0.0) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if kernel_size <= 0:
            raise ValueError(f"mask_logit_smoothing_kernel must be positive, got {kernel_size}")
        if not 0.0 <= float(blend) <= 1.0:
            raise ValueError(f"mask_logit_smoothing_blend must be in [0, 1], got {blend}")
        _validate_kernel(kernel_size, 1, axis="mask logit smoothing time")

        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.blend = float(blend)
        self.pad_t = self.kernel_size - 1
        self.enabled = self.kernel_size > 1 and self.blend > 0.0
        if self.enabled:
            self.conv = nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=(self.kernel_size, 1),
                groups=1,
                bias=False,
            )
            self._init_moving_average()
        else:
            self.conv = None

    def _init_moving_average(self) -> None:
        if self.conv is None:
            return
        with torch.no_grad():
            self.conv.weight.zero_()
            value = 1.0 / float(self.kernel_size)
            for idx in range(self.channels):
                self.conv.weight[idx, idx, :, 0].fill_(value)

    def stream_context_frames(self) -> int:
        return self.pad_t if self.enabled else 0

    def init_stream_state(self, batch_size: int, *, freq_bins: int, device=None, dtype=None) -> torch.Tensor:
        return torch.zeros(
            batch_size,
            self.channels,
            self.stream_context_frames(),
            freq_bins,
            device=device,
            dtype=dtype,
        )

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return logits
        assert self.conv is not None
        smooth = self.conv(F.pad(logits, (0, 0, self.pad_t, 0)))
        return logits * (1.0 - self.blend) + smooth * self.blend

    def forward_stream(
        self,
        logits: torch.Tensor,
        state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.enabled:
            return logits, logits[:, :, :0, :]
        assert self.conv is not None
        if state is None:
            state = self.init_stream_state(
                logits.shape[0],
                freq_bins=logits.shape[-1],
                device=logits.device,
                dtype=logits.dtype,
            )
        full = torch.cat([state, logits], dim=2)
        new_state = full[:, :, -self.pad_t :, :]
        smooth = self.conv(full)
        return logits * (1.0 - self.blend) + smooth * self.blend, new_state


class ConvPyramidSourceHead2d(nn.Module):
    """Complex-mask source head with source-folded channels."""

    def __init__(
        self,
        *,
        in_channels: int,
        n_src: int,
        n_chan: int,
        hidden_channels: int,
        source_kernel: int,
        real_mask_scale: float,
        imag_mask_scale: float,
        mask_logit_smoothing_kernel: int = 1,
        mask_logit_smoothing_blend: float = 0.0,
    ) -> None:
        super().__init__()
        if source_kernel <= 0 or source_kernel % 2 != 1:
            raise ValueError(f"source_kernel must be a positive odd integer, got {source_kernel}")
        if real_mask_scale <= 0.0:
            raise ValueError(f"real_mask_scale must be positive, got {real_mask_scale}")
        if imag_mask_scale <= 0.0:
            raise ValueError(f"imag_mask_scale must be positive, got {imag_mask_scale}")
        _validate_kernel(source_kernel, 1, axis="source frequency")
        self.n_src = int(n_src)
        self.n_chan = int(n_chan)
        self.real_mask_scale = float(real_mask_scale)
        self.imag_mask_scale = float(imag_mask_scale)
        out_channels = 2 * n_src * n_chan
        folded = int(n_src * hidden_channels)
        self.seed = nn.Conv2d(in_channels, folded, kernel_size=1, bias=True)
        self.refine = nn.Conv2d(
            folded,
            folded,
            kernel_size=(1, source_kernel),
            padding=(0, source_kernel // 2),
            groups=n_src,
            bias=True,
        )
        self.mask = nn.Conv2d(folded, out_channels, kernel_size=1, groups=n_src, bias=True)
        self.logit_smoother = MaskLogitTemporalSmoother2d(
            out_channels,
            kernel_size=mask_logit_smoothing_kernel,
            blend=mask_logit_smoothing_blend,
        )

    @property
    def has_stream_state(self) -> bool:
        return self.logit_smoother.enabled

    def stream_context_frames(self) -> int:
        return self.logit_smoother.stream_context_frames()

    def init_stream_state(self, batch_size: int, *, freq_bins: int, device=None, dtype=None) -> torch.Tensor:
        return self.logit_smoother.init_stream_state(batch_size, freq_bins=freq_bins, device=device, dtype=dtype)

    def _source_logits(self, x: torch.Tensor) -> torch.Tensor:
        source = F.relu(self.seed(x))
        source = source + F.relu(self.refine(source))
        return self.mask(source)

    def _decode_logits(
        self,
        logits: torch.Tensor,
        mixture2d: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        real = torch.sigmoid(logits[:, 0::2, :, :]) * self.real_mask_scale
        imag = torch.tanh(logits[:, 1::2, :, :]) * self.imag_mask_scale
        mask_chunks = []
        for idx in range(self.n_src * self.n_chan):
            mask_chunks.append(real[:, idx : idx + 1, :, :])
            mask_chunks.append(imag[:, idx : idx + 1, :, :])
        mask = torch.cat(mask_chunks, dim=1)
        y = _complex_mask_multiply(mixture2d, mask, n_src=self.n_src, n_chan=self.n_chan)
        aux = {
            "mask": mask,
            "mask_domain": "packed_complex_mask",
            "mask_logits": logits,
            "mask_logits_domain": "tvconv_pyramid_complex_mask_logits",
            "mask_logits_transform": "sigmoid_tanh_complex_mask",
            "mask_logits_real_scale": self.real_mask_scale,
            "mask_logits_imag_scale": self.imag_mask_scale,
        }
        if self.logit_smoother.enabled:
            aux["mask_logits_smoothing_kernel"] = self.logit_smoother.kernel_size
            aux["mask_logits_smoothing_blend"] = self.logit_smoother.blend
        return y, aux

    def forward(self, x: torch.Tensor, mixture2d: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits = self.logit_smoother(self._source_logits(x))
        return self._decode_logits(logits, mixture2d)

    def forward_stream(
        self,
        x: torch.Tensor,
        mixture2d: torch.Tensor,
        state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        logits, new_state = self.logit_smoother.forward_stream(self._source_logits(x), state)
        y, aux = self._decode_logits(logits, mixture2d)
        return y, aux, new_state


class FoldedSourceAwareBlock2d(nn.Module):
    """Source-folded local/source-competition block using Conv2D only."""

    def __init__(self, folded_channels: int, *, n_src: int, source_kernel: int) -> None:
        super().__init__()
        self.local = nn.Conv2d(
            folded_channels,
            folded_channels,
            kernel_size=(1, source_kernel),
            padding=(0, source_kernel // 2),
            groups=n_src,
            bias=True,
        )
        self.mix = nn.Conv2d(folded_channels, folded_channels, kernel_size=1, bias=True)
        self.local_scale = nn.Parameter(torch.tensor(0.1))
        self.mix_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + F.relu(self.local(x)) * self.local_scale
        return x + F.relu(self.mix(x)) * self.mix_scale


class FoldedSourceAwareSourceHead2d(nn.Module):
    """Stronger source-aware mask head without source loops or norm reductions."""

    def __init__(
        self,
        *,
        in_channels: int,
        n_src: int,
        n_chan: int,
        hidden_channels: int,
        source_kernel: int,
        mixer_layers: int,
        real_mask_scale: float,
        imag_mask_scale: float,
        mask_logit_smoothing_kernel: int = 1,
        mask_logit_smoothing_blend: float = 0.0,
    ) -> None:
        super().__init__()
        if source_kernel <= 0 or source_kernel % 2 != 1:
            raise ValueError(f"source_kernel must be a positive odd integer, got {source_kernel}")
        if mixer_layers <= 0:
            raise ValueError(f"mixer_layers must be positive for folded_source_aware head, got {mixer_layers}")
        if real_mask_scale <= 0.0:
            raise ValueError(f"real_mask_scale must be positive, got {real_mask_scale}")
        if imag_mask_scale <= 0.0:
            raise ValueError(f"imag_mask_scale must be positive, got {imag_mask_scale}")
        _validate_kernel(source_kernel, 1, axis="source frequency")
        self.n_src = int(n_src)
        self.n_chan = int(n_chan)
        self.real_mask_scale = float(real_mask_scale)
        self.imag_mask_scale = float(imag_mask_scale)
        out_channels = 2 * n_src * n_chan
        folded = int(n_src * hidden_channels)
        self.seed = nn.Conv2d(in_channels, folded, kernel_size=1, bias=True)
        self.source_bias = nn.Parameter(torch.zeros(1, folded, 1, 1))
        self.blocks = nn.ModuleList(
            [
                FoldedSourceAwareBlock2d(folded, n_src=n_src, source_kernel=source_kernel)
                for _ in range(int(mixer_layers))
            ]
        )
        self.mask = nn.Conv2d(folded, out_channels, kernel_size=1, groups=n_src, bias=True)
        self.logit_smoother = MaskLogitTemporalSmoother2d(
            out_channels,
            kernel_size=mask_logit_smoothing_kernel,
            blend=mask_logit_smoothing_blend,
        )

    @property
    def has_stream_state(self) -> bool:
        return self.logit_smoother.enabled

    def stream_context_frames(self) -> int:
        return self.logit_smoother.stream_context_frames()

    def init_stream_state(self, batch_size: int, *, freq_bins: int, device=None, dtype=None) -> torch.Tensor:
        return self.logit_smoother.init_stream_state(batch_size, freq_bins=freq_bins, device=device, dtype=dtype)

    def _source_logits(self, x: torch.Tensor) -> torch.Tensor:
        source = F.relu(self.seed(x) + self.source_bias)
        for block in self.blocks:
            source = block(source)
        return self.mask(source)

    def _decode_logits(
        self,
        logits: torch.Tensor,
        mixture2d: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        real = torch.sigmoid(logits[:, 0::2, :, :]) * self.real_mask_scale
        imag = torch.tanh(logits[:, 1::2, :, :]) * self.imag_mask_scale
        mask_chunks = []
        for idx in range(self.n_src * self.n_chan):
            mask_chunks.append(real[:, idx : idx + 1, :, :])
            mask_chunks.append(imag[:, idx : idx + 1, :, :])
        mask = torch.cat(mask_chunks, dim=1)
        y = _complex_mask_multiply(mixture2d, mask, n_src=self.n_src, n_chan=self.n_chan)
        aux = {
            "mask": mask,
            "mask_domain": "packed_complex_mask",
            "mask_logits": logits,
            "mask_logits_domain": "tvconv_pyramid_folded_source_aware_complex_mask_logits",
            "mask_logits_transform": "sigmoid_tanh_complex_mask",
            "mask_logits_real_scale": self.real_mask_scale,
            "mask_logits_imag_scale": self.imag_mask_scale,
        }
        if self.logit_smoother.enabled:
            aux["mask_logits_smoothing_kernel"] = self.logit_smoother.kernel_size
            aux["mask_logits_smoothing_blend"] = self.logit_smoother.blend
        return y, aux

    def forward(self, x: torch.Tensor, mixture2d: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits = self.logit_smoother(self._source_logits(x))
        return self._decode_logits(logits, mixture2d)

    def forward_stream(
        self,
        x: torch.Tensor,
        mixture2d: torch.Tensor,
        state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        logits, new_state = self.logit_smoother.forward_stream(self._source_logits(x), state)
        y, aux = self._decode_logits(logits, mixture2d)
        return y, aux, new_state


def _complex_mask_multiply(x: torch.Tensor, mask: torch.Tensor, *, n_src: int, n_chan: int) -> torch.Tensor:
    batch, in_channels, frames, n_freq = x.shape
    if in_channels != 2 * n_chan:
        raise ValueError(f"Expected {2 * n_chan} mixture channels, got {in_channels}")
    if mask.shape != (batch, 2 * n_src * n_chan, frames, n_freq):
        raise ValueError(f"Invalid mask shape {tuple(mask.shape)}")

    chunks = []
    for src_idx in range(n_src):
        for chan_idx in range(n_chan):
            mix_base = 2 * chan_idx
            mask_base = 2 * (src_idx * n_chan + chan_idx)
            xr = x[:, mix_base : mix_base + 1, :, :]
            xi = x[:, mix_base + 1 : mix_base + 2, :, :]
            mr = mask[:, mask_base : mask_base + 1, :, :]
            mi = mask[:, mask_base + 1 : mask_base + 2, :, :]
            chunks.append(xr * mr - xi * mi)
            chunks.append(xr * mi + xi * mr)
    return torch.cat(chunks, dim=1)


class TVConvPyramidNPUSeparator2D(nn.Module):
    """Conv-only frequency-pyramid student core."""

    def __init__(
        self,
        n_freq: int,
        *,
        n_src: int = 2,
        n_chan: int = 1,
        base_channels: int = 48,
        bottleneck_channels: int = 160,
        n_down: int = 4,
        n_blocks: int = 6,
        expansion: int = 3,
        time_kernel: int = 3,
        time_dilation_cycle: Sequence[int] | None = (1, 1, 2, 2, 2, 1),
        band_kernel: int = 3,
        capacity_hidden: int = 0,
        capacity_hidden_schedule: Sequence[int] | None = None,
        recurrent_type: str = "none",
        recurrent_layers: int = 0,
        recurrent_band_kernel: int = 3,
        recurrent_replace_blocks: int = 0,
        upsample_mode: str = "convtranspose",
        source_head_type: str = "basic",
        source_mixer_layers: int = 1,
        mask_hidden: int = 96,
        source_kernel: int = 5,
        real_mask_scale: float = 1.0,
        imag_mask_scale: float = 0.12,
        mask_logit_smoothing_kernel: int = 1,
        mask_logit_smoothing_blend: float = 0.0,
    ) -> None:
        super().__init__()
        if n_freq <= 0:
            raise ValueError(f"n_freq must be positive, got {n_freq}")
        if n_down <= 0:
            raise ValueError(f"n_down must be positive, got {n_down}")
        if n_blocks <= 0:
            raise ValueError(f"n_blocks must be positive, got {n_blocks}")
        self.n_freq = int(n_freq)
        self.n_src = int(n_src)
        self.n_chan = int(n_chan)
        self.n_down = int(n_down)
        self.n_blocks = int(n_blocks)
        if source_head_type not in {"basic", "folded_source_aware"}:
            raise ValueError(
                f"source_head_type must be one of ['basic', 'folded_source_aware'], got {source_head_type!r}"
            )
        self.source_head_type = str(source_head_type)
        if recurrent_type not in {"none", "gru", "lstm"}:
            raise ValueError(f"recurrent_type must be one of ['none', 'gru', 'lstm'], got {recurrent_type}")
        self.recurrent_type = str(recurrent_type)
        if recurrent_layers < 0:
            raise ValueError(f"recurrent_layers must be non-negative, got {recurrent_layers}")
        if recurrent_replace_blocks < 0:
            raise ValueError(f"recurrent_replace_blocks must be non-negative, got {recurrent_replace_blocks}")
        if self.recurrent_type == "none":
            if recurrent_layers != 0:
                raise ValueError("recurrent_layers must be 0 when recurrent_type='none'.")
            if recurrent_replace_blocks != 0:
                raise ValueError("recurrent_replace_blocks must be 0 when recurrent_type='none'.")
        if self.recurrent_type != "none" and recurrent_layers <= 0:
            raise ValueError(f"recurrent_layers must be positive when recurrent_type={recurrent_type!r}.")
        if recurrent_replace_blocks > n_blocks:
            raise ValueError(f"recurrent_replace_blocks={recurrent_replace_blocks} exceeds n_blocks={n_blocks}.")
        self.recurrent_layers = int(recurrent_layers)
        self.recurrent_replace_blocks = int(recurrent_replace_blocks)
        self.upsample_mode = str(upsample_mode).lower().replace("-", "_")
        if self.upsample_mode not in {"convtranspose", "resize_conv"}:
            raise ValueError(
                f"upsample_mode must be one of ['convtranspose', 'resize_conv'], got {upsample_mode!r}"
            )

        dilation_cycle = _as_int_tuple(time_dilation_cycle, default=(1, 1, 2, 2, 2, 1), name="time_dilation_cycle")
        conv_block_count = int(n_blocks) - int(recurrent_replace_blocks)
        if capacity_hidden_schedule is None:
            hidden_schedule = tuple(int(capacity_hidden) for _ in range(conv_block_count))
        else:
            hidden_schedule = tuple(int(v) for v in capacity_hidden_schedule)
            if len(hidden_schedule) != conv_block_count:
                raise ValueError(
                    f"capacity_hidden_schedule must have {conv_block_count} conv-block entries, got {hidden_schedule}"
                )

        sizes = [int(n_freq)]
        for _ in range(n_down):
            sizes.append((sizes[-1] + 1) // 2)
        if sizes[-1] <= 1:
            raise ValueError(f"n_down={n_down} compresses n_freq={n_freq} too far: {sizes}")
        self.freq_sizes = tuple(sizes)

        channels = [int(base_channels)]
        while len(channels) < n_down:
            channels.append(min(int(bottleneck_channels), channels[-1] * 2))
        channels.append(int(bottleneck_channels))

        self.input = nn.Sequential(
            nn.Conv2d(2 * n_chan, channels[0], kernel_size=1, bias=True),
            ChannelAffine2d(channels[0]),
            nn.ReLU(),
        )
        self.down_blocks = nn.ModuleList(
            [ConvDownsample2d(channels[idx], channels[idx + 1]) for idx in range(n_down)]
        )
        self.temporal_blocks = nn.ModuleList(
            [
                ConvPyramidTemporalBlock2d(
                    channels[-1],
                    expansion=expansion,
                    time_kernel=time_kernel,
                    time_dilation=dilation_cycle[idx % len(dilation_cycle)],
                    band_kernel=band_kernel,
                    capacity_hidden=hidden_schedule[idx],
                )
                for idx in range(conv_block_count)
            ]
        )
        recurrent_cls = {"gru": BottleneckConvGRU2d, "lstm": BottleneckConvLSTM2d}.get(self.recurrent_type)
        self.recurrent_blocks = nn.ModuleList(
            [
                recurrent_cls(channels[-1], band_kernel=recurrent_band_kernel)
                for _ in range(self.recurrent_layers)
            ]
            if recurrent_cls is not None
            else []
        )
        up_blocks = []
        for rev_idx in range(n_down - 1, -1, -1):
            in_ch = channels[rev_idx + 1]
            out_ch = channels[rev_idx]
            in_size = sizes[rev_idx + 1]
            target_size = sizes[rev_idx]
            output_padding = target_size - (2 * in_size - 1)
            if self.upsample_mode == "resize_conv":
                up_blocks.append(ResizeConvUpsample2d(in_ch, out_ch, target_freq=target_size))
            else:
                up_blocks.append(ConvUpsample2d(in_ch, out_ch, output_padding=output_padding))
        self.up_blocks = nn.ModuleList(up_blocks)
        self.output_affine = ChannelAffine2d(channels[0])
        source_head_cls = (
            FoldedSourceAwareSourceHead2d
            if self.source_head_type == "folded_source_aware"
            else ConvPyramidSourceHead2d
        )
        source_head_kwargs = {
            "in_channels": channels[0],
            "n_src": n_src,
            "n_chan": n_chan,
            "hidden_channels": mask_hidden,
            "source_kernel": source_kernel,
            "real_mask_scale": real_mask_scale,
            "imag_mask_scale": imag_mask_scale,
            "mask_logit_smoothing_kernel": mask_logit_smoothing_kernel,
            "mask_logit_smoothing_blend": mask_logit_smoothing_blend,
        }
        if self.source_head_type == "folded_source_aware":
            source_head_kwargs["mixer_layers"] = int(source_mixer_layers)
        self.source_head = source_head_cls(**source_head_kwargs)
        self._last_aux: dict[str, torch.Tensor] = {}

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        if x.ndim != 4:
            raise ValueError(f"Expected 4D packed STFT input, got {tuple(x.shape)}")
        if x.shape[1] != 2 * self.n_chan:
            raise ValueError(f"Expected {2 * self.n_chan} channels, got {x.shape[1]}")
        if x.shape[-1] != self.n_freq:
            raise ValueError(f"Expected {self.n_freq} frequency bins, got {x.shape[-1]}")

        skips = []
        h = self.input(x)
        skips.append(h)
        for down in self.down_blocks:
            h = down(h)
            skips.append(h)
        for block in self.temporal_blocks:
            h = block(h)
        for block in self.recurrent_blocks:
            h = block(h)
        for up, skip in zip(self.up_blocks, reversed(skips[:-1]), strict=True):
            h = up(h, skip)
        h = self.output_affine(h)
        y, aux = self.source_head(h, x)
        self._last_aux = aux
        if return_aux:
            return y, aux
        return y

    def stream_context_frames(self) -> int:
        source_context = self.source_head.stream_context_frames()
        return sum(block.stream_context_frames() for block in self.temporal_blocks) + source_context

    def init_stream_state(self, batch_size: int = 1, *, device=None, dtype=None):
        freq_bins = self.freq_sizes[-1]
        temporal_states = tuple(
            block.init_stream_state(batch_size, freq_bins=freq_bins, device=device, dtype=dtype)
            for block in self.temporal_blocks
        )
        recurrent_states = tuple(
            block.init_stream_state(batch_size, freq_bins=freq_bins, device=device, dtype=dtype)
            for block in self.recurrent_blocks
        )
        source_state = (
            (self.source_head.init_stream_state(batch_size, freq_bins=self.n_freq, device=device, dtype=dtype),)
            if self.source_head.has_stream_state
            else ()
        )
        return (*temporal_states, *recurrent_states, *source_state)

    def forward_stream(self, x: torch.Tensor, state=None):
        if state is None:
            state = self.init_stream_state(batch_size=x.shape[0], device=x.device, dtype=x.dtype)
        expected_states = len(self.temporal_blocks) + len(self.recurrent_blocks)
        if self.source_head.has_stream_state:
            expected_states += 1
        if len(state) != expected_states:
            raise ValueError(f"Expected {expected_states} streaming states, got {len(state)}")
        skips = []
        h = self.input(x)
        skips.append(h)
        for down in self.down_blocks:
            h = down(h)
            skips.append(h)
        new_states = []
        temporal_state = state[: len(self.temporal_blocks)]
        recurrent_state_end = len(self.temporal_blocks) + len(self.recurrent_blocks)
        recurrent_state = state[len(self.temporal_blocks) : recurrent_state_end]
        source_state = state[-1] if self.source_head.has_stream_state else None
        for block, block_state in zip(self.temporal_blocks, temporal_state, strict=True):
            h, new_state = block.forward_stream(h, block_state)
            new_states.append(new_state)
        for block, block_state in zip(self.recurrent_blocks, recurrent_state, strict=True):
            h, new_state = block.forward_stream(h, block_state)
            new_states.append(new_state)
        for up, skip in zip(self.up_blocks, reversed(skips[:-1]), strict=True):
            h = up(h, skip)
        h = self.output_affine(h)
        if self.source_head.has_stream_state:
            y, aux, new_source_state = self.source_head.forward_stream(h, x, source_state)
            new_states.append(new_source_state)
        else:
            y, aux = self.source_head(h, x)
        self._last_aux = aux
        return y, tuple(new_states)

    def forward_stream_recompute(self, x: torch.Tensor, history: torch.Tensor | None = None):
        if self.recurrent_type != "none":
            raise RuntimeError("forward_stream_recompute is not exact for recurrent TVConv variants.")
        context = self.stream_context_frames()
        if history is None:
            history = x.new_zeros(x.shape[0], x.shape[1], context, x.shape[-1])
        full = torch.cat([history, x], dim=2)
        y = self(full)[:, :, -x.shape[2] :, :]
        new_history = full[:, :, -context:, :] if context > 0 else full[:, :, :0, :]
        return y, new_history

    def layer_cache_numel(self, batch_size: int = 1) -> int:
        return _state_numel(self.init_stream_state(batch_size=batch_size))

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


def build_tvconv_pyramid_npu_separator_system(
    *,
    n_fft: int,
    hop_length: int,
    fs: int,
    n_src: int = 3,
    n_chan: int = 1,
    core_n_src: int | None = None,
    base_channels: int = 48,
    bottleneck_channels: int = 160,
    n_down: int = 4,
    n_blocks: int = 6,
    expansion: int = 3,
    time_kernel: int = 3,
    time_dilation_cycle: Sequence[int] | None = (1, 1, 2, 2, 2, 1),
    band_kernel: int = 3,
    capacity_hidden: int = 0,
    capacity_hidden_schedule: Sequence[int] | None = None,
    recurrent_type: str = "none",
    recurrent_layers: int = 0,
    recurrent_band_kernel: int = 3,
    recurrent_replace_blocks: int = 0,
    upsample_mode: str = "convtranspose",
    source_head_type: str = "basic",
    source_mixer_layers: int = 1,
    mask_hidden: int = 96,
    source_kernel: int = 5,
    real_mask_scale: float = 1.0,
    imag_mask_scale: float = 0.12,
    mask_logit_smoothing_kernel: int = 1,
    mask_logit_smoothing_blend: float = 0.0,
    residual_source_enabled: bool = True,
    residual_source_index: int | None = None,
    scaling: bool = False,
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
    css_segment_size: int = 12,
    css_shift_size: int = 6,
    css_batch_size: int = 1,
    postprocess_enabled: bool = False,
    postprocess_mixture_consistency: str = "none",
    postprocess_final_mixture_consistency: str = "none",
    postprocess_power_beta: float = 1.0,
    postprocess_power_smoothing: float = 0.0,
    postprocess_wiener_blend: float = 0.0,
    postprocess_wiener_alpha: float = 1.0,
    postprocess_leakage_gate_enabled: bool = False,
    postprocess_leakage_gate_threshold_db: float = 12.0,
    postprocess_leakage_gate_attenuation_db: float = 6.0,
    postprocess_residual_source_index: int | None = None,
    postprocess_misi_iterations: int = 0,
    postprocess_misi_eps: float = 1.0e-8,
    postprocess_eps: float = 1.0e-8,
) -> OnlineModelWrapper:
    explicit_n_src = int(core_n_src) if core_n_src is not None else int(n_src) - 1
    if residual_source_enabled:
        expected = int(n_src) - 1
        if explicit_n_src != expected:
            raise ValueError(f"residual_source_enabled expects core_n_src={expected}, got {explicit_n_src}")
    elif explicit_n_src != int(n_src):
        raise ValueError(f"core_n_src={explicit_n_src} requires residual_source_enabled=true when n_src={int(n_src)}")

    full_n_freq = (n_fft // 2) + 1
    core_n_freq = resolve_preprocessed_n_freq(
        full_n_freq,
        enabled=freq_preprocess_enabled,
        keep_bins=freq_preprocess_keep_bins,
        target_bins=freq_preprocess_target_bins,
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
    core = TVConvPyramidNPUSeparator2D(
        n_freq=core_n_freq,
        n_src=explicit_n_src,
        n_chan=n_chan,
        base_channels=base_channels,
        bottleneck_channels=bottleneck_channels,
        n_down=n_down,
        n_blocks=n_blocks,
        expansion=expansion,
        time_kernel=time_kernel,
        time_dilation_cycle=time_dilation_cycle,
        band_kernel=band_kernel,
        capacity_hidden=capacity_hidden,
        capacity_hidden_schedule=capacity_hidden_schedule,
        recurrent_type=recurrent_type,
        recurrent_layers=recurrent_layers,
        recurrent_band_kernel=recurrent_band_kernel,
        recurrent_replace_blocks=recurrent_replace_blocks,
        upsample_mode=upsample_mode,
        source_head_type=source_head_type,
        source_mixer_layers=source_mixer_layers,
        mask_hidden=mask_hidden,
        source_kernel=source_kernel,
        real_mask_scale=real_mask_scale,
        imag_mask_scale=imag_mask_scale,
        mask_logit_smoothing_kernel=mask_logit_smoothing_kernel,
        mask_logit_smoothing_blend=mask_logit_smoothing_blend,
    )
    model = FrequencyPreprocessedOnlineModel(
        core=core,
        n_src=n_src,
        n_chan=n_chan,
        freq_preprocessor=freq_preprocessor,
        pcen_preprocessor=pcen_preprocessor,
        dc_bypass_enabled=dc_bypass_enabled,
        dc_policy=dc_policy,
        residual_source_enabled=residual_source_enabled,
        residual_source_index=residual_source_index,
    )
    postprocessor = build_source_separation_postprocessor(
        enabled=postprocess_enabled,
        mixture_consistency=postprocess_mixture_consistency,
        final_mixture_consistency=postprocess_final_mixture_consistency,
        power_beta=postprocess_power_beta,
        power_smoothing=postprocess_power_smoothing,
        wiener_blend=postprocess_wiener_blend,
        wiener_alpha=postprocess_wiener_alpha,
        leakage_gate_enabled=postprocess_leakage_gate_enabled,
        leakage_gate_threshold_db=postprocess_leakage_gate_threshold_db,
        leakage_gate_attenuation_db=postprocess_leakage_gate_attenuation_db,
        residual_source_index=postprocess_residual_source_index,
        eps=postprocess_eps,
    )
    phase_consistency = build_misi_phase_consistency(
        iterations=postprocess_misi_iterations,
        eps=postprocess_misi_eps,
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
        postprocessor=postprocessor,
        phase_consistency=phase_consistency,
    )


def build_tvconv_pyramid_convgru_npu_separator_system(
    *,
    recurrent_layers: int = 1,
    recurrent_band_kernel: int = 3,
    recurrent_replace_blocks: int = 2,
    **kwargs,
) -> OnlineModelWrapper:
    requested = kwargs.pop("recurrent_type", "gru")
    if requested != "gru":
        raise ValueError(f"ConvGRU TVConv builder requires recurrent_type='gru', got {requested!r}")
    return build_tvconv_pyramid_npu_separator_system(
        recurrent_type="gru",
        recurrent_layers=recurrent_layers,
        recurrent_band_kernel=recurrent_band_kernel,
        recurrent_replace_blocks=recurrent_replace_blocks,
        **kwargs,
    )


def build_tvconv_pyramid_convlstm_npu_separator_system(
    *,
    recurrent_layers: int = 1,
    recurrent_band_kernel: int = 3,
    recurrent_replace_blocks: int = 2,
    **kwargs,
) -> OnlineModelWrapper:
    requested = kwargs.pop("recurrent_type", "lstm")
    if requested != "lstm":
        raise ValueError(f"ConvLSTM TVConv builder requires recurrent_type='lstm', got {requested!r}")
    return build_tvconv_pyramid_npu_separator_system(
        recurrent_type="lstm",
        recurrent_layers=recurrent_layers,
        recurrent_band_kernel=recurrent_band_kernel,
        recurrent_replace_blocks=recurrent_replace_blocks,
        **kwargs,
    )


def build_tvconv_pyramid_sfclite_query_npu_separator_system(**kwargs) -> OnlineModelWrapper:
    requested = kwargs.pop("freq_preprocess_mode", "learnable_query")
    if requested not in {"learnable_query", "sfclite_query"}:
        raise ValueError(f"SFC-lite query TVConv builder requires learnable-query preprocessing, got {requested!r}")
    return build_tvconv_pyramid_npu_separator_system(
        freq_preprocess_mode="learnable_query",
        **kwargs,
    )


def build_tvconv_pyramid_sourceaware_sfclite_convgru_npu_separator_system(
    *,
    recurrent_layers: int = 1,
    recurrent_band_kernel: int = 3,
    recurrent_replace_blocks: int = 2,
    source_mixer_layers: int = 1,
    **kwargs,
) -> OnlineModelWrapper:
    requested_recurrent = kwargs.pop("recurrent_type", "gru")
    if requested_recurrent != "gru":
        raise ValueError(
            "Source-aware SFC-lite TVConv builder requires recurrent_type='gru', "
            f"got {requested_recurrent!r}"
        )
    requested_freq = kwargs.pop("freq_preprocess_mode", "learnable_query")
    if requested_freq not in {"learnable_query", "sfclite_query"}:
        raise ValueError(
            "Source-aware SFC-lite TVConv builder requires learnable-query preprocessing, "
            f"got {requested_freq!r}"
        )
    requested_head = kwargs.pop("source_head_type", "folded_source_aware")
    if requested_head != "folded_source_aware":
        raise ValueError(
            "Source-aware SFC-lite TVConv builder requires source_head_type='folded_source_aware', "
            f"got {requested_head!r}"
        )
    return build_tvconv_pyramid_npu_separator_system(
        recurrent_type="gru",
        recurrent_layers=recurrent_layers,
        recurrent_band_kernel=recurrent_band_kernel,
        recurrent_replace_blocks=recurrent_replace_blocks,
        source_head_type="folded_source_aware",
        source_mixer_layers=source_mixer_layers,
        freq_preprocess_mode="learnable_query",
        **kwargs,
    )
