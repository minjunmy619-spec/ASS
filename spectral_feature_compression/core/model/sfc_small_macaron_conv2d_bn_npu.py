"""Causal NPU rewrite of the official SFC TF-Locoformer block skeleton.

Each block preserves the official frequency-then-time ordering, and each axis
path preserves the Macaron ``FFN -> mixer -> FFN`` residual structure. Conv1D,
ConvTranspose1D, self-attention, and RMSGroupNorm are replaced by factorized
Conv2D, causal depthwise Conv2D, and foldable BatchNorm2D.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectral_feature_compression.core.model.online_model_wrapper import OnlineModelWrapper
from spectral_feature_compression.core.model.sfc_small_conv2d_bn_npu import (
    Conv2dBNAct,
    SFCSmallConv2DBNNPUModel,
    _apply_packed_complex_mask,
    _resolve_n_fft,
)
from spectral_feature_compression.core.model.sfc_small_pyramid_dw_bn_npu import (
    CausalDepthwiseConv2dBNAct,
    SFCSmallExactDecoder,
    SFCSmallExactEncoder,
)


class FactorizedAxisSwiGLUFFN2D(nn.Module):
    """NPU factorization of one axis-specific Conv-SwiGLU-Deconv FFN."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        *,
        axis: str,
        frequency_kernel_size: int,
        time_kernel_size: int,
        time_dilation: int,
    ) -> None:
        super().__init__()
        if axis not in {"frequency", "time"}:
            raise ValueError(f"Unsupported axis: {axis}")
        self.axis = axis
        if axis == "frequency":
            kernel_size = (1, frequency_kernel_size)
            dilation = (1, 1)
        else:
            kernel_size = (time_kernel_size, 1)
            dilation = (time_dilation, 1)

        self.axis_mix = CausalDepthwiseConv2dBNAct(
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            activation=False,
        )
        # Separate projections avoid Split/Slice in the exported SwiGLU.
        self.value = Conv2dBNAct(channels, hidden_channels, activation=False)
        self.gate = Conv2dBNAct(channels, hidden_channels, activation=False)
        self.output = Conv2dBNAct(hidden_channels, channels, activation=False)

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        value = self.value(x)
        gate = self.gate(x)
        return self.output(value * F.silu(gate))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._project(self.axis_mix(x))

    def init_stream_state(
        self,
        batch_size: int,
        *,
        n_bands: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if self.axis != "time":
            raise RuntimeError("Only temporal FFNs have streaming state")
        return self.axis_mix.init_stream_state(
            batch_size,
            freq_bins=n_bands,
            device=device,
            dtype=dtype,
        )

    def forward_stream(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.axis != "time":
            raise RuntimeError("Only temporal FFNs use forward_stream")
        x, next_state = self.axis_mix.forward_stream(x, state)
        return self._project(x), next_state


class AxisConvMixer2D(nn.Module):
    """Depthwise axis mixer plus pointwise channel aggregation."""

    def __init__(
        self,
        channels: int,
        *,
        axis: str,
        frequency_kernel_size: int,
        time_kernel_size: int,
        time_dilation: int,
    ) -> None:
        super().__init__()
        if axis not in {"frequency", "time"}:
            raise ValueError(f"Unsupported axis: {axis}")
        self.axis = axis
        if axis == "frequency":
            kernel_size = (1, frequency_kernel_size)
            dilation = (1, 1)
        else:
            kernel_size = (time_kernel_size, 1)
            dilation = (time_dilation, 1)
        self.axis_mix = CausalDepthwiseConv2dBNAct(
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            activation=True,
        )
        self.output = Conv2dBNAct(channels, channels, activation=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.axis_mix(x))

    def init_stream_state(
        self,
        batch_size: int,
        *,
        n_bands: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if self.axis != "time":
            raise RuntimeError("Only temporal mixers have streaming state")
        return self.axis_mix.init_stream_state(
            batch_size,
            freq_bins=n_bands,
            device=device,
            dtype=dtype,
        )

    def forward_stream(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.axis != "time":
            raise RuntimeError("Only temporal mixers use forward_stream")
        x, next_state = self.axis_mix.forward_stream(x, state)
        return self.output(x), next_state


class MacaronAxisPath2D(nn.Module):
    """One official-shaped ``FFN -> mixer -> FFN`` axis path."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        *,
        axis: str,
        frequency_kernel_size: int,
        time_kernel_size: int,
        time_dilation: int,
    ) -> None:
        super().__init__()
        ffn_kwargs = {
            "axis": axis,
            "frequency_kernel_size": frequency_kernel_size,
            "time_kernel_size": time_kernel_size,
            "time_dilation": time_dilation,
        }
        self.axis = axis
        self.pre_ffn = FactorizedAxisSwiGLUFFN2D(
            channels,
            hidden_channels,
            **ffn_kwargs,
        )
        self.mixer = AxisConvMixer2D(
            channels,
            **ffn_kwargs,
        )
        self.post_ffn = FactorizedAxisSwiGLUFFN2D(
            channels,
            hidden_channels,
            **ffn_kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pre_ffn(x)
        x = x + self.mixer(x)
        return x + self.post_ffn(x)

    def init_stream_state(
        self,
        batch_size: int,
        *,
        n_bands: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.axis != "time":
            raise RuntimeError("Only the temporal path has streaming state")
        kwargs = {
            "batch_size": batch_size,
            "n_bands": n_bands,
            "device": device,
            "dtype": dtype,
        }
        return (
            self.pre_ffn.init_stream_state(**kwargs),
            self.mixer.init_stream_state(**kwargs),
            self.post_ffn.init_stream_state(**kwargs),
        )

    def forward_stream(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if self.axis != "time":
            raise RuntimeError("Only the temporal path uses forward_stream")
        pre, state0 = self.pre_ffn.forward_stream(x, state[0])
        x = x + pre
        mixed, state1 = self.mixer.forward_stream(x, state[1])
        x = x + mixed
        post, state2 = self.post_ffn.forward_stream(x, state[2])
        return x + post, (state0, state1, state2)


class NPUTFLocoformerBlock2D(nn.Module):
    """Official frequency-then-time TF block without layout transformations."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        *,
        frequency_kernel_size: int,
        time_kernel_size: int,
        time_dilation: int,
    ) -> None:
        super().__init__()
        kwargs = {
            "frequency_kernel_size": frequency_kernel_size,
            "time_kernel_size": time_kernel_size,
            "time_dilation": time_dilation,
        }
        self.freq_path = MacaronAxisPath2D(
            channels,
            hidden_channels,
            axis="frequency",
            **kwargs,
        )
        self.frame_path = MacaronAxisPath2D(
            channels,
            hidden_channels,
            axis="time",
            **kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.frame_path(self.freq_path(x))

    def init_stream_state(
        self,
        batch_size: int,
        *,
        n_bands: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.frame_path.init_stream_state(
            batch_size,
            n_bands=n_bands,
            device=device,
            dtype=dtype,
        )

    def forward_stream(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        x = self.freq_path(x)
        return self.frame_path.forward_stream(x, state)


class FaithfulMacaronConv2DSeparator(nn.Module):
    """Stack of same-band NPU TF-Locoformer skeletons."""

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
    ) -> None:
        super().__init__()
        if not dilation_cycle:
            raise ValueError("dilation_cycle must not be empty")
        self.n_bands = int(n_bands)
        self.blocks = nn.ModuleList(
            [
                NPUTFLocoformerBlock2D(
                    channels,
                    hidden_channels,
                    frequency_kernel_size=frequency_kernel_size,
                    time_kernel_size=time_kernel_size,
                    time_dilation=int(dilation_cycle[idx % len(dilation_cycle)]),
                )
                for idx in range(int(n_blocks))
            ]
        )

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
        state = []
        for block in self.blocks:
            state.extend(
                block.init_stream_state(
                    batch_size,
                    n_bands=self.n_bands,
                    device=device,
                    dtype=dtype,
                )
            )
        return tuple(state)

    def forward_stream(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        next_state = []
        for block_idx, block in enumerate(self.blocks):
            offset = 3 * block_idx
            x, block_state = block.forward_stream(x, state[offset : offset + 3])
            next_state.extend(block_state)
        return x, tuple(next_state)


class SFCSmallMacaronConv2DBNNPUCore(nn.Module):
    """Packed-real causal SFC core with faithful NPU Macaron block ordering."""

    def __init__(
        self,
        *,
        n_freq: int,
        n_fft: int | None = None,
        sample_rate: int = 44100,
        n_bands: int = 36,
        band_config: str = "musical",
        n_src: int = 3,
        n_chan: int = 1,
        d_inner: int = 32,
        d_model: int = 128,
        ffn_hidden: int = 176,
        n_separator_layers: int = 2,
        n_sfc_heads: int = 4,
        learnable_pos_bias: bool = True,
        frequency_kernel_size: int = 15,
        time_kernel_size: int = 2,
        dilation_cycle: Sequence[int] = (1,),
        encoder_ffn_expansion: int = 2,
        decoder_ffn_hidden: int = 16,
        masking: bool = True,
        use_learnable_query: bool = True,
    ) -> None:
        super().__init__()
        self.n_freq = int(n_freq)
        self.n_fft = _resolve_n_fft(n_freq, n_fft)
        self.n_bands = int(n_bands)
        self.n_src = int(n_src)
        self.n_chan = int(n_chan)
        self.d_inner = int(d_inner)
        self.d_model = int(d_model)
        self.n_sfc_heads = int(n_sfc_heads)
        self.n_separator_layers = int(n_separator_layers)
        self.masking = bool(masking)

        self.encoder = SFCSmallExactEncoder(
            in_channels=2 * n_chan,
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
            encoder_ffn_expansion=encoder_ffn_expansion,
        )
        self.separator = FaithfulMacaronConv2DSeparator(
            d_model,
            ffn_hidden,
            n_bands=n_bands,
            n_blocks=n_separator_layers,
            frequency_kernel_size=frequency_kernel_size,
            time_kernel_size=time_kernel_size,
            dilation_cycle=dilation_cycle,
        )
        self.decoder = SFCSmallExactDecoder(
            d_model=d_model,
            d_inner=d_inner,
            out_channels=2 * n_src * n_chan,
            n_freq=n_freq,
            n_bands=n_bands,
            n_fft=self.n_fft,
            sample_rate=sample_rate,
            band_config=band_config,
            n_heads=n_sfc_heads,
            learnable_pos_bias=learnable_pos_bias,
            use_learnable_query=use_learnable_query,
            decoder_ffn_hidden=decoder_ffn_hidden,
        )
        self._init_mask_bias()

    def _init_mask_bias(self) -> None:
        with torch.no_grad():
            self.decoder.output.bias.zero_()
            for src_idx in range(self.n_src):
                for chan_idx in range(self.n_chan):
                    self.decoder.output.bias[2 * (src_idx * self.n_chan + chan_idx)] = 1.0 / self.n_src

    def forward(self, x: torch.Tensor, return_mask: bool = False):
        h = self.encoder(x)
        h = self.separator(h)
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
        return self.separator.init_stream_state(batch_size, device=device, dtype=dtype)

    def forward_stream(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, ...] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if state is None:
            state = self.init_stream_state(x.shape[0], device=x.device, dtype=x.dtype)
        h, _ = self.encoder.forward_stream(x, None)
        h, next_state = self.separator.forward_stream(h, state)
        mask = self.decoder.forward_stream(h)
        y = _apply_packed_complex_mask(x, mask, n_src=self.n_src, n_chan=self.n_chan) if self.masking else mask
        return y, next_state

    def state_size_bytes(
        self,
        *,
        batch_size: int = 1,
        dtype: torch.dtype = torch.float16,
    ) -> int:
        itemsize = torch.empty((), dtype=dtype).element_size()
        return sum(
            tensor.numel() * itemsize
            for tensor in self.init_stream_state(batch_size=batch_size, dtype=dtype)
        )


class SFCSmallMacaronConv2DBNNPUModel(SFCSmallConv2DBNNPUModel):
    def __init__(self, **core_kwargs) -> None:
        nn.Module.__init__(self)
        self.core = SFCSmallMacaronConv2DBNNPUCore(**core_kwargs)
        self.n_src = self.core.n_src
        self.n_chan = self.core.n_chan


def build_sfc_small_macaron_conv2d_bn_npu_system(
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
    model = SFCSmallMacaronConv2DBNNPUModel(
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
    "FaithfulMacaronConv2DSeparator",
    "NPUTFLocoformerBlock2D",
    "SFCSmallMacaronConv2DBNNPUCore",
    "SFCSmallMacaronConv2DBNNPUModel",
    "build_sfc_small_macaron_conv2d_bn_npu_system",
]
