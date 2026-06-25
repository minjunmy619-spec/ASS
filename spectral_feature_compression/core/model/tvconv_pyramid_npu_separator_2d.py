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
        self.mask = nn.Conv2d(folded, 2 * n_src * n_chan, kernel_size=1, groups=n_src, bias=True)

    def forward(self, x: torch.Tensor, mixture2d: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        source = F.relu(self.seed(x))
        source = source + F.relu(self.refine(source))
        logits = self.mask(source)
        real = torch.sigmoid(logits[:, 0::2, :, :]) * self.real_mask_scale
        imag = torch.tanh(logits[:, 1::2, :, :]) * self.imag_mask_scale
        mask_chunks = []
        for idx in range(self.n_src * self.n_chan):
            mask_chunks.append(real[:, idx : idx + 1, :, :])
            mask_chunks.append(imag[:, idx : idx + 1, :, :])
        mask = torch.cat(mask_chunks, dim=1)
        y = _complex_mask_multiply(mixture2d, mask, n_src=self.n_src, n_chan=self.n_chan)
        return y, {
            "mask": mask,
            "mask_domain": "packed_complex_mask",
            "mask_logits": logits,
            "mask_logits_domain": "tvconv_pyramid_complex_mask_logits",
            "mask_logits_transform": "sigmoid_tanh_complex_mask",
            "mask_logits_real_scale": self.real_mask_scale,
            "mask_logits_imag_scale": self.imag_mask_scale,
        }


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
        mask_hidden: int = 96,
        source_kernel: int = 5,
        real_mask_scale: float = 1.0,
        imag_mask_scale: float = 0.12,
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

        dilation_cycle = _as_int_tuple(time_dilation_cycle, default=(1, 1, 2, 2, 2, 1), name="time_dilation_cycle")
        if capacity_hidden_schedule is None:
            hidden_schedule = tuple(int(capacity_hidden) for _ in range(n_blocks))
        else:
            hidden_schedule = tuple(int(v) for v in capacity_hidden_schedule)
            if len(hidden_schedule) != n_blocks:
                raise ValueError(f"capacity_hidden_schedule must have {n_blocks} entries, got {hidden_schedule}")

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
                for idx in range(n_blocks)
            ]
        )
        up_blocks = []
        for rev_idx in range(n_down - 1, -1, -1):
            in_ch = channels[rev_idx + 1]
            out_ch = channels[rev_idx]
            in_size = sizes[rev_idx + 1]
            target_size = sizes[rev_idx]
            output_padding = target_size - (2 * in_size - 1)
            up_blocks.append(ConvUpsample2d(in_ch, out_ch, output_padding=output_padding))
        self.up_blocks = nn.ModuleList(up_blocks)
        self.output_affine = ChannelAffine2d(channels[0])
        self.source_head = ConvPyramidSourceHead2d(
            in_channels=channels[0],
            n_src=n_src,
            n_chan=n_chan,
            hidden_channels=mask_hidden,
            source_kernel=source_kernel,
            real_mask_scale=real_mask_scale,
            imag_mask_scale=imag_mask_scale,
        )
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
        for up, skip in zip(self.up_blocks, reversed(skips[:-1]), strict=True):
            h = up(h, skip)
        h = self.output_affine(h)
        y, aux = self.source_head(h, x)
        self._last_aux = aux
        if return_aux:
            return y, aux
        return y

    def stream_context_frames(self) -> int:
        return sum(block.stream_context_frames() for block in self.temporal_blocks)

    def init_stream_state(self, batch_size: int = 1, *, device=None, dtype=None):
        freq_bins = self.freq_sizes[-1]
        return tuple(
            block.init_stream_state(batch_size, freq_bins=freq_bins, device=device, dtype=dtype)
            for block in self.temporal_blocks
        )

    def forward_stream(self, x: torch.Tensor, state=None):
        if state is None:
            state = self.init_stream_state(batch_size=x.shape[0], device=x.device, dtype=x.dtype)
        skips = []
        h = self.input(x)
        skips.append(h)
        for down in self.down_blocks:
            h = down(h)
            skips.append(h)
        new_states = []
        for block, block_state in zip(self.temporal_blocks, state, strict=True):
            h, new_state = block.forward_stream(h, block_state)
            new_states.append(new_state)
        for up, skip in zip(self.up_blocks, reversed(skips[:-1]), strict=True):
            h = up(h, skip)
        h = self.output_affine(h)
        y, aux = self.source_head(h, x)
        self._last_aux = aux
        return y, tuple(new_states)

    def forward_stream_recompute(self, x: torch.Tensor, history: torch.Tensor | None = None):
        context = self.stream_context_frames()
        if history is None:
            history = x.new_zeros(x.shape[0], x.shape[1], context, x.shape[-1])
        full = torch.cat([history, x], dim=2)
        y = self(full)[:, :, -x.shape[2] :, :]
        new_history = full[:, :, -context:, :] if context > 0 else full[:, :, :0, :]
        return y, new_history

    def layer_cache_numel(self, batch_size: int = 1) -> int:
        return sum(int(state.numel()) for state in self.init_stream_state(batch_size=batch_size))

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
    mask_hidden: int = 96,
    source_kernel: int = 5,
    real_mask_scale: float = 1.0,
    imag_mask_scale: float = 0.12,
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
        mask_hidden=mask_hidden,
        source_kernel=source_kernel,
        real_mask_scale=real_mask_scale,
        imag_mask_scale=imag_mask_scale,
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
