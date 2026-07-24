"""Cumulative-LayerNorm separator variant of the NPU SFC Macaron model.

The encoder and decoder keep the exact SFC cross-attention implementation from
the BatchNorm variant. The separator restores causal pre-normalization at the
official Macaron sublayer boundaries and removes BatchNorm from its Conv2D
branches.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectral_feature_compression.core.model.online_model_wrapper import OnlineModelWrapper
from spectral_feature_compression.core.model.sfc_small_macaron_conv2d_bn_npu import (
    AxisConvMixer2D,
    FactorizedAxisSwiGLUFFN2D,
    SFCSmallMacaronConv2DBNNPUCore,
    SFCSmallMacaronConv2DBNNPUModel,
)


def _remove_batchnorm(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.Identity())
        else:
            _remove_batchnorm(child)


class CumulativeLayerNorm2D(nn.Module):
    """Causal cumulative LayerNorm for fixed 36-band separator features.

    Statistics cover channels, bands, and all frames observed up to the current
    frame. Streaming state contains the running first and second moments. A
    shared ``alpha`` state supplied by the separator is ``1 / (frame + 1)``.
    """

    def __init__(self, channels: int, *, n_bands: int = 36, eps: float = 1.0e-5) -> None:
        super().__init__()
        if n_bands != 36:
            raise ValueError(f"CumulativeLayerNorm2D currently requires 36 bands, got {n_bands}")
        self.channels = int(channels)
        self.n_bands = int(n_bands)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.register_buffer(
            "channel_average",
            torch.full((1, channels, 1, 1), 1.0 / channels),
        )

    def _frame_moment(self, x: torch.Tensor) -> torch.Tensor:
        # 36 -> 9 -> 1. Both pooling kernels and strides satisfy the NPU limits.
        moment = F.conv2d(x, self.channel_average)
        moment = F.avg_pool2d(moment, kernel_size=(1, 4), stride=(1, 4))
        return F.avg_pool2d(moment, kernel_size=(1, 9), stride=(1, 1))

    def _normalize(self, x: torch.Tensor, mean: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        variance = F.relu(second - mean * mean)
        normalized = (x - mean) * torch.rsqrt(variance + self.eps)
        return normalized * self.weight + self.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frame_mean = self._frame_moment(x)
        frame_second = self._frame_moment(x * x)
        count = torch.cumsum(torch.ones_like(frame_mean), dim=2)
        mean = torch.cumsum(frame_mean, dim=2) / count
        second = torch.cumsum(frame_second, dim=2) / count
        return self._normalize(x, mean, second)

    def init_stream_state(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = torch.zeros(batch_size, 1, 1, 1, device=device, dtype=dtype)
        return state, state.clone()

    def forward_stream(
        self,
        x: torch.Tensor,
        mean: torch.Tensor,
        second: torch.Tensor,
        alpha: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not torch.jit.is_tracing() and x.shape[2] != 1:
            raise RuntimeError(f"CumulativeLayerNorm2D streaming expects one frame, got {x.shape[2]}")
        frame_mean = self._frame_moment(x)
        frame_second = self._frame_moment(x * x)
        next_mean = mean + (frame_mean - mean) * alpha
        next_second = second + (frame_second - second) * alpha
        return self._normalize(x, next_mean, next_second), next_mean, next_second


class CumulativeMacaronAxisPath2D(nn.Module):
    """Pre-cLN Macaron ``FFN -> mixer -> FFN`` path."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        *,
        n_bands: int,
        axis: str,
        frequency_kernel_size: int,
        time_kernel_size: int,
        time_dilation: int,
        norm_eps: float,
    ) -> None:
        super().__init__()
        kwargs = {
            "axis": axis,
            "frequency_kernel_size": frequency_kernel_size,
            "time_kernel_size": time_kernel_size,
            "time_dilation": time_dilation,
        }
        self.axis = axis
        self.pre_norm = CumulativeLayerNorm2D(channels, n_bands=n_bands, eps=norm_eps)
        self.mixer_norm = CumulativeLayerNorm2D(channels, n_bands=n_bands, eps=norm_eps)
        self.post_norm = CumulativeLayerNorm2D(channels, n_bands=n_bands, eps=norm_eps)
        self.pre_ffn = FactorizedAxisSwiGLUFFN2D(channels, hidden_channels, **kwargs)
        self.mixer = AxisConvMixer2D(channels, **kwargs)
        self.post_ffn = FactorizedAxisSwiGLUFFN2D(channels, hidden_channels, **kwargs)
        _remove_batchnorm(self.pre_ffn)
        _remove_batchnorm(self.mixer)
        _remove_batchnorm(self.post_ffn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pre_ffn(self.pre_norm(x))
        x = x + self.mixer(self.mixer_norm(x))
        return x + self.post_ffn(self.post_norm(x))

    def init_stream_state(
        self,
        batch_size: int,
        *,
        n_bands: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, ...]:
        state: list[torch.Tensor] = []
        for norm in (self.pre_norm, self.mixer_norm, self.post_norm):
            state.extend(norm.init_stream_state(batch_size, device=device, dtype=dtype))
        if self.axis == "time":
            kwargs = {
                "batch_size": batch_size,
                "n_bands": n_bands,
                "device": device,
                "dtype": dtype,
            }
            state.extend(
                (
                    self.pre_ffn.init_stream_state(**kwargs),
                    self.mixer.init_stream_state(**kwargs),
                    self.post_ffn.init_stream_state(**kwargs),
                )
            )
        return tuple(state)

    def forward_stream(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, ...],
        alpha: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        expected = 9 if self.axis == "time" else 6
        if len(state) != expected:
            raise RuntimeError(f"Expected {expected} {self.axis} path states, got {len(state)}")

        pre, mean0, second0 = self.pre_norm.forward_stream(x, state[0], state[1], alpha)
        if self.axis == "time":
            pre, cache0 = self.pre_ffn.forward_stream(pre, state[6])
        else:
            pre = self.pre_ffn(pre)
        x = x + pre

        mixed, mean1, second1 = self.mixer_norm.forward_stream(x, state[2], state[3], alpha)
        if self.axis == "time":
            mixed, cache1 = self.mixer.forward_stream(mixed, state[7])
        else:
            mixed = self.mixer(mixed)
        x = x + mixed

        post, mean2, second2 = self.post_norm.forward_stream(x, state[4], state[5], alpha)
        if self.axis == "time":
            post, cache2 = self.post_ffn.forward_stream(post, state[8])
            next_state = (mean0, second0, mean1, second1, mean2, second2, cache0, cache1, cache2)
        else:
            post = self.post_ffn(post)
            next_state = (mean0, second0, mean1, second1, mean2, second2)
        return x + post, next_state


class CumulativeNPUTFLocoformerBlock2D(nn.Module):
    """Frequency-then-time TF block with causal cumulative pre-normalization."""

    STATE_COUNT = 15

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        *,
        n_bands: int,
        frequency_kernel_size: int,
        time_kernel_size: int,
        time_dilation: int,
        norm_eps: float,
    ) -> None:
        super().__init__()
        kwargs = {
            "n_bands": n_bands,
            "frequency_kernel_size": frequency_kernel_size,
            "time_kernel_size": time_kernel_size,
            "time_dilation": time_dilation,
            "norm_eps": norm_eps,
        }
        self.n_bands = int(n_bands)
        self.freq_path = CumulativeMacaronAxisPath2D(
            channels, hidden_channels, axis="frequency", **kwargs
        )
        self.frame_path = CumulativeMacaronAxisPath2D(
            channels, hidden_channels, axis="time", **kwargs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.frame_path(self.freq_path(x))

    def init_stream_state(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, ...]:
        return (
            *self.freq_path.init_stream_state(
                batch_size, n_bands=self.n_bands, device=device, dtype=dtype
            ),
            *self.frame_path.init_stream_state(
                batch_size, n_bands=self.n_bands, device=device, dtype=dtype
            ),
        )

    def forward_stream(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, ...],
        alpha: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if len(state) != self.STATE_COUNT:
            raise RuntimeError(f"Expected {self.STATE_COUNT} block states, got {len(state)}")
        x, freq_state = self.freq_path.forward_stream(x, state[:6], alpha)
        x, frame_state = self.frame_path.forward_stream(x, state[6:], alpha)
        return x, (*freq_state, *frame_state)


class CumulativeMacaronConv2DSeparator(nn.Module):
    """Same-band Macaron stack with one shared reciprocal frame-count state."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        *,
        n_bands: int,
        n_blocks: int,
        frequency_kernel_size: int,
        time_kernel_size: int,
        dilation_cycle: Sequence[int],
        norm_eps: float,
    ) -> None:
        super().__init__()
        if not dilation_cycle:
            raise ValueError("dilation_cycle must not be empty")
        self.blocks = nn.ModuleList(
            [
                CumulativeNPUTFLocoformerBlock2D(
                    channels,
                    hidden_channels,
                    n_bands=n_bands,
                    frequency_kernel_size=frequency_kernel_size,
                    time_kernel_size=time_kernel_size,
                    time_dilation=int(dilation_cycle[idx % len(dilation_cycle)]),
                    norm_eps=norm_eps,
                )
                for idx in range(int(n_blocks))
            ]
        )

    @property
    def state_count(self) -> int:
        return 1 + CumulativeNPUTFLocoformerBlock2D.STATE_COUNT * len(self.blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x

    def init_stream_state(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, ...]:
        alpha = torch.ones(batch_size, 1, 1, 1, device=device, dtype=dtype)
        state: list[torch.Tensor] = [alpha]
        for block in self.blocks:
            state.extend(block.init_stream_state(batch_size, device=device, dtype=dtype))
        return tuple(state)

    def forward_stream(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if len(state) != self.state_count:
            raise RuntimeError(f"Expected {self.state_count} separator states, got {len(state)}")
        alpha = state[0]
        next_state: list[torch.Tensor] = [alpha / (1.0 + alpha)]
        offset = 1
        for block in self.blocks:
            end = offset + block.STATE_COUNT
            x, block_state = block.forward_stream(x, state[offset:end], alpha)
            next_state.extend(block_state)
            offset = end
        return x, tuple(next_state)


class SFCSmallMacaronConv2DCLNNPUCore(SFCSmallMacaronConv2DBNNPUCore):
    """SFC core with cumulative pre-LayerNorm in the separator."""

    def __init__(self, *, norm_eps: float = 1.0e-5, **kwargs) -> None:
        super().__init__(**kwargs)
        self.separator = CumulativeMacaronConv2DSeparator(
            kwargs["d_model"],
            kwargs["ffn_hidden"],
            n_bands=kwargs["n_bands"],
            n_blocks=kwargs["n_separator_layers"],
            frequency_kernel_size=kwargs["frequency_kernel_size"],
            time_kernel_size=kwargs["time_kernel_size"],
            dilation_cycle=kwargs["dilation_cycle"],
            norm_eps=norm_eps,
        )


class SFCSmallMacaronConv2DCLNNPUModel(SFCSmallMacaronConv2DBNNPUModel):
    def __init__(self, **core_kwargs) -> None:
        nn.Module.__init__(self)
        self.core = SFCSmallMacaronConv2DCLNNPUCore(**core_kwargs)
        self.n_src = self.core.n_src
        self.n_chan = self.core.n_chan


def build_sfc_small_macaron_conv2d_cln_npu_system(
    *,
    n_fft: int,
    hop_length: int,
    fs: int,
    n_src: int = 3,
    n_chan: int = 1,
    n_bands: int = 36,
    band_config: str = "musical",
    d_inner: int = 32,
    d_model: int = 128,
    ffn_hidden: int = 176,
    n_separator_layers: int = 2,
    n_sfc_heads: int = 4,
    learnable_pos_bias: bool = True,
    frequency_kernel_size: int = 15,
    time_kernel_size: int = 2,
    dilation_cycle: Sequence[int] = (1,),
    norm_eps: float = 1.0e-5,
    freq_kernel_size: int | None = None,
    ffn_expansion: int | None = None,
    encoder_ffn_expansion: int = 2,
    decoder_ffn_hidden: int = 16,
    masking: bool = True,
    use_learnable_query: bool = True,
    scaling: bool = False,
    css_segment_size: int = 12,
    css_shift_size: int = 6,
    css_batch_size: int = 1,
) -> OnlineModelWrapper:
    if freq_kernel_size is not None or ffn_expansion is not None:
        raise ValueError("Use frequency_kernel_size and ffn_hidden for the cumulative-LN variant")
    model = SFCSmallMacaronConv2DCLNNPUModel(
        n_freq=n_fft // 2 + 1,
        n_fft=n_fft,
        sample_rate=fs,
        n_bands=n_bands,
        band_config=band_config,
        n_src=n_src,
        n_chan=n_chan,
        d_inner=d_inner,
        d_model=d_model,
        ffn_hidden=ffn_hidden,
        n_separator_layers=n_separator_layers,
        n_sfc_heads=n_sfc_heads,
        learnable_pos_bias=learnable_pos_bias,
        frequency_kernel_size=frequency_kernel_size,
        time_kernel_size=time_kernel_size,
        dilation_cycle=dilation_cycle,
        norm_eps=norm_eps,
        encoder_ffn_expansion=encoder_ffn_expansion,
        decoder_ffn_hidden=decoder_ffn_hidden,
        masking=masking,
        use_learnable_query=use_learnable_query,
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


__all__ = [
    "CumulativeLayerNorm2D",
    "CumulativeMacaronConv2DSeparator",
    "CumulativeNPUTFLocoformerBlock2D",
    "SFCSmallMacaronConv2DCLNNPUCore",
    "SFCSmallMacaronConv2DCLNNPUModel",
    "build_sfc_small_macaron_conv2d_cln_npu_system",
]
