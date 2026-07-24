"""Lower-node cumulative-LayerNorm variant of the NPU SFC Macaron model.

Each frequency or temporal axis path computes cumulative statistics once and
reuses them at its three pre-normalization sites. Learned affine transforms
remain independent for the pre-FFN, mixer, and post-FFN branches.
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
from spectral_feature_compression.core.model.sfc_small_macaron_conv2d_cln_npu import (
    _remove_batchnorm,
)


class SharedCumulativeStatistics2D(nn.Module):
    """One causal cumulative moment tracker for a fixed 36-band axis path."""

    def __init__(self, channels: int, *, n_bands: int = 36, eps: float = 1.0e-5) -> None:
        super().__init__()
        if n_bands != 36:
            raise ValueError(f"SharedCumulativeStatistics2D requires 36 bands, got {n_bands}")
        self.eps = float(eps)
        self.register_buffer(
            "channel_average",
            torch.full((1, channels, 1, 1), 1.0 / channels),
        )

    def _frame_moment(self, x: torch.Tensor) -> torch.Tensor:
        moment = F.conv2d(x, self.channel_average)
        moment = F.avg_pool2d(moment, kernel_size=(1, 4), stride=(1, 4))
        return F.avg_pool2d(moment, kernel_size=(1, 9), stride=(1, 1))

    def _inverse_std(self, mean: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        return torch.rsqrt(F.relu(second - mean * mean) + self.eps)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        frame_mean = self._frame_moment(x)
        frame_second = self._frame_moment(x * x)
        count = torch.cumsum(torch.ones_like(frame_mean), dim=2)
        mean = torch.cumsum(frame_mean, dim=2) / count
        second = torch.cumsum(frame_second, dim=2) / count
        return mean, self._inverse_std(mean, second)

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not torch.jit.is_tracing() and x.shape[2] != 1:
            raise RuntimeError(f"Shared cLN streaming expects one frame, got {x.shape[2]}")
        frame_mean = self._frame_moment(x)
        frame_second = self._frame_moment(x * x)
        next_mean = mean + (frame_mean - mean) * alpha
        next_second = second + (frame_second - second) * alpha
        inv_std = self._inverse_std(next_mean, next_second)
        return next_mean, next_second, next_mean, inv_std


class CumulativeAffine2D(nn.Module):
    """Independent cLN affine transform represented as depthwise 1x1 Conv2D."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.affine = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            groups=channels,
            bias=True,
        )
        with torch.no_grad():
            self.affine.weight.fill_(1.0)
            self.affine.bias.zero_()

    def forward(
        self,
        x: torch.Tensor,
        mean: torch.Tensor,
        inv_std: torch.Tensor,
    ) -> torch.Tensor:
        return self.affine((x - mean) * inv_std)


class SharedStatsMacaronAxisPath2D(nn.Module):
    """Macaron path with one moment tracker and three independent affine sites."""

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
        self.statistics = SharedCumulativeStatistics2D(channels, n_bands=n_bands, eps=norm_eps)
        self.pre_affine = CumulativeAffine2D(channels)
        self.mixer_affine = CumulativeAffine2D(channels)
        self.post_affine = CumulativeAffine2D(channels)
        self.pre_ffn = FactorizedAxisSwiGLUFFN2D(channels, hidden_channels, **kwargs)
        self.mixer = AxisConvMixer2D(channels, **kwargs)
        self.post_ffn = FactorizedAxisSwiGLUFFN2D(channels, hidden_channels, **kwargs)
        _remove_batchnorm(self.pre_ffn)
        _remove_batchnorm(self.mixer)
        _remove_batchnorm(self.post_ffn)

    def _run(
        self,
        x: torch.Tensor,
        mean: torch.Tensor,
        inv_std: torch.Tensor,
        conv_state: tuple[torch.Tensor, ...] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        pre = self.pre_affine(x, mean, inv_std)
        if self.axis == "time":
            assert conv_state is not None
            pre, cache0 = self.pre_ffn.forward_stream(pre, conv_state[0])
        else:
            pre = self.pre_ffn(pre)
        x = x + pre

        mixed = self.mixer_affine(x, mean, inv_std)
        if self.axis == "time":
            mixed, cache1 = self.mixer.forward_stream(mixed, conv_state[1])
        else:
            mixed = self.mixer(mixed)
        x = x + mixed

        post = self.post_affine(x, mean, inv_std)
        if self.axis == "time":
            post, cache2 = self.post_ffn.forward_stream(post, conv_state[2])
            next_cache = (cache0, cache1, cache2)
        else:
            post = self.post_ffn(post)
            next_cache = ()
        return x + post, next_cache

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean, inv_std = self.statistics(x)
        # Full-sequence temporal convolutions use their causal padding path.
        pre = self.pre_ffn(self.pre_affine(x, mean, inv_std))
        x = x + pre
        x = x + self.mixer(self.mixer_affine(x, mean, inv_std))
        return x + self.post_ffn(self.post_affine(x, mean, inv_std))

    def init_stream_state(
        self,
        batch_size: int,
        *,
        n_bands: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, ...]:
        state: list[torch.Tensor] = list(
            self.statistics.init_stream_state(batch_size, device=device, dtype=dtype)
        )
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
        expected = 5 if self.axis == "time" else 2
        if len(state) != expected:
            raise RuntimeError(f"Expected {expected} {self.axis} path states, got {len(state)}")
        next_mean, next_second, mean, inv_std = self.statistics.forward_stream(
            x, state[0], state[1], alpha
        )
        x, cache = self._run(x, mean, inv_std, state[2:] if self.axis == "time" else None)
        return x, (next_mean, next_second, *cache)


class SharedStatsNPUTFLocoformerBlock2D(nn.Module):
    STATE_COUNT = 7

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
        self.freq_path = SharedStatsMacaronAxisPath2D(
            channels, hidden_channels, axis="frequency", **kwargs
        )
        self.frame_path = SharedStatsMacaronAxisPath2D(
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
        x, freq_state = self.freq_path.forward_stream(x, state[:2], alpha)
        x, frame_state = self.frame_path.forward_stream(x, state[2:], alpha)
        return x, (*freq_state, *frame_state)


class SharedStatsMacaronConv2DSeparator(nn.Module):
    """Two-axis Macaron stack with four cLN trackers for the default model."""

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
                SharedStatsNPUTFLocoformerBlock2D(
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
        return 1 + SharedStatsNPUTFLocoformerBlock2D.STATE_COUNT * len(self.blocks)

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
        state: list[torch.Tensor] = [
            torch.ones(batch_size, 1, 1, 1, device=device, dtype=dtype)
        ]
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


class SFCSmallMacaronConv2DCLNLiteNPUCore(SFCSmallMacaronConv2DBNNPUCore):
    def __init__(self, *, norm_eps: float = 1.0e-5, **kwargs) -> None:
        super().__init__(**kwargs)
        self.separator = SharedStatsMacaronConv2DSeparator(
            kwargs["d_model"],
            kwargs["ffn_hidden"],
            n_bands=kwargs["n_bands"],
            n_blocks=kwargs["n_separator_layers"],
            frequency_kernel_size=kwargs["frequency_kernel_size"],
            time_kernel_size=kwargs["time_kernel_size"],
            dilation_cycle=kwargs["dilation_cycle"],
            norm_eps=norm_eps,
        )


class SFCSmallMacaronConv2DCLNLiteNPUModel(SFCSmallMacaronConv2DBNNPUModel):
    def __init__(self, **core_kwargs) -> None:
        nn.Module.__init__(self)
        self.core = SFCSmallMacaronConv2DCLNLiteNPUCore(**core_kwargs)
        self.n_src = self.core.n_src
        self.n_chan = self.core.n_chan


def build_sfc_small_macaron_conv2d_cln_lite_npu_system(
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
        raise ValueError("Use frequency_kernel_size and ffn_hidden for the cLN-lite variant")
    model = SFCSmallMacaronConv2DCLNLiteNPUModel(
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
    "SharedCumulativeStatistics2D",
    "SharedStatsMacaronConv2DSeparator",
    "SFCSmallMacaronConv2DCLNLiteNPUCore",
    "SFCSmallMacaronConv2DCLNLiteNPUModel",
    "build_sfc_small_macaron_conv2d_cln_lite_npu_system",
]
