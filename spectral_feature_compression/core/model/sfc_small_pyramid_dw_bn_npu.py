"""Low-compute causal SFC-small variant for ONE/NPU deployment.

The encoder and decoder retain learnable-query SFC cross-attention with the
official musical-band position bias. The separator keeps tensors in
``[B, C, T, F]`` and moves its parameter-heavy blocks to four frequency cells
through an additive Conv2D pyramid.
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
    _resolve_band_indices,
    _resolve_n_fft,
    _validate_kernel_span,
    _validate_odd_kernel,
)
from spectral_feature_compression.core.model.sfc_small_conv2d_bn_npu_kvsplit import (
    SFCSmallConv2DBNKvSplitDecoder,
    SFCSmallConv2DBNKvSplitEncoder,
)


def _official_encoder_position_bias(
    band_indices: list[tuple[int, int]],
    n_freq: int,
    n_heads: int,
) -> torch.Tensor:
    """Build the exact ``gentle_slope`` bias used by official SFC."""

    bias = torch.zeros(len(band_indices), n_freq)
    for band_idx, (start, end) in enumerate(band_indices):
        center = (start + end) // 2
        denominator = (end - start) // 2 + 1
        for freq_idx in range(n_freq):
            if freq_idx < start:
                bias[band_idx, freq_idx] = freq_idx - start
            elif freq_idx > end - 1:
                bias[band_idx, freq_idx] = end - 1 - freq_idx
            else:
                bias[band_idx, freq_idx] = -abs(center - freq_idx) / denominator
    return bias.unsqueeze(0).repeat(n_heads, 1, 1)


def _replace_position_bias(module: nn.Module, value: torch.Tensor, learnable: bool) -> None:
    if "pos_bias" in module._parameters:
        del module._parameters["pos_bias"]
    if "pos_bias" in module._buffers:
        del module._buffers["pos_bias"]
    if learnable:
        module.register_parameter("pos_bias", nn.Parameter(value))
    else:
        module.register_buffer("pos_bias", value)


class SFCSmallExactEncoder(SFCSmallConv2DBNKvSplitEncoder):
    """KV-split SFC encoder with exact bias and signed BN projections."""

    def __init__(self, *, encoder_ffn_expansion: int = 2, **kwargs) -> None:
        if not kwargs.get("use_learnable_query", True):
            raise ValueError("The low-compute deployment variant supports learnable SFC queries only")
        super().__init__(**kwargs)
        d_inner = self.n_heads * self.head_dim
        d_model = self.output.conv.out_channels
        hidden = d_inner * int(encoder_ffn_expansion)
        self.input = Conv2dBNAct(
            kwargs["in_channels"], d_inner, kernel_size=(1, 3), padding=(0, 1), activation=False
        )
        self.ffn = nn.Sequential(
            Conv2dBNAct(d_inner, hidden, activation=True),
            Conv2dBNAct(hidden, d_inner, activation=False),
        )
        self.output = Conv2dBNAct(
            d_inner, d_model, kernel_size=(1, 3), padding=(0, 1), activation=False
        )

        band_indices = _resolve_band_indices(
            n_freq=self.n_freq,
            n_bands=self.n_bands,
            n_fft=kwargs["n_fft"],
            sample_rate=kwargs["sample_rate"],
            band_config=kwargs["band_config"],
        )
        bias = _official_encoder_position_bias(band_indices, self.n_freq, self.n_heads)
        _replace_position_bias(self, bias, kwargs["learnable_pos_bias"])

    def _attend_stream_frame(self, h: torch.Tensor) -> torch.Tensor:
        bsz, _channels, _frames, n_freq = h.shape
        key = self.key_proj(h).reshape(bsz, self.n_heads, self.head_dim, n_freq)
        value = self.value_proj(h).reshape(bsz, self.n_heads, self.head_dim, n_freq)
        query = self.query.unsqueeze(0).to(dtype=h.dtype)
        score = torch.matmul(query, key)
        score = score + self.pos_bias.unsqueeze(0).to(dtype=h.dtype)
        weight = torch.softmax(score, dim=-1)

        # (W @ V.T).T == V @ W.T. This removes one attention transpose.
        attended = torch.matmul(value, weight.transpose(2, 3))
        attended = attended.reshape(bsz, -1, 1, self.n_bands)
        attended = self.aggregate(attended)
        return attended + self.ffn(attended)


class SFCSmallExactDecoder(SFCSmallConv2DBNKvSplitDecoder):
    """KV-split SFC decoder with exact transposed bias and a narrow full-bin FFN."""

    def __init__(self, *, decoder_ffn_hidden: int = 16, **kwargs) -> None:
        if not kwargs.get("use_learnable_query", True):
            raise ValueError("The low-compute deployment variant supports learnable SFC queries only")
        super().__init__(**kwargs)
        d_inner = self.n_heads * self.head_dim
        self.input = Conv2dBNAct(
            kwargs["d_model"], d_inner, kernel_size=(1, 3), padding=(0, 1), activation=False
        )
        self.ffn = nn.Sequential(
            Conv2dBNAct(d_inner, int(decoder_ffn_hidden), activation=True),
            Conv2dBNAct(int(decoder_ffn_hidden), d_inner, activation=False),
        )
        self.output = nn.Conv2d(
            d_inner,
            kwargs["out_channels"],
            kernel_size=(1, 3),
            padding=(0, 1),
            bias=True,
        )

        band_indices = _resolve_band_indices(
            n_freq=self.n_freq,
            n_bands=self.n_bands,
            n_fft=kwargs["n_fft"],
            sample_rate=kwargs["sample_rate"],
            band_config=kwargs["band_config"],
        )
        bias = _official_encoder_position_bias(band_indices, self.n_freq, self.n_heads).transpose(1, 2)
        _replace_position_bias(self, bias, kwargs["learnable_pos_bias"])

    def forward_stream(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input(x)
        if not torch.jit.is_tracing() and h.shape[2] != 1:
            return self.forward(x)
        bsz, _channels, _frames, n_bands = h.shape
        key = self.key_proj(h).reshape(bsz, self.n_heads, self.head_dim, n_bands)
        value = self.value_proj(h).reshape(bsz, self.n_heads, self.head_dim, n_bands)
        query = self.query.unsqueeze(0).to(dtype=h.dtype)
        score = torch.matmul(query, key)
        score = score + self.pos_bias.unsqueeze(0).to(dtype=h.dtype)
        weight = torch.softmax(score, dim=-1)
        expanded = torch.matmul(value, weight.transpose(2, 3))
        expanded = expanded.reshape(bsz, -1, 1, self.n_freq)
        expanded = self.aggregate(expanded)
        expanded = expanded + self.ffn(expanded)
        return self.output(expanded)


# Backward-compatible names for the earlier pyramid ablation.
SFCSmallPyramidEncoder = SFCSmallExactEncoder
SFCSmallPyramidDecoder = SFCSmallExactDecoder


class CausalDepthwiseConv2dBNAct(nn.Module):
    """Depthwise Conv2D with explicit causal temporal state."""

    def __init__(
        self,
        channels: int,
        *,
        kernel_size: tuple[int, int],
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
            channels,
            channels,
            kernel_size=(kt, kf),
            dilation=(dt, df),
            padding=(0, self.freq_pad),
            groups=channels,
            bias=True,
        )
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=False) if activation else nn.Identity()

    def _run(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.context_frames:
            x = F.pad(x, (0, 0, self.context_frames, 0))
        return self._run(x)

    def init_stream_state(
        self,
        batch_size: int,
        *,
        freq_bins: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        return torch.zeros(
            batch_size,
            self.conv.in_channels,
            self.context_frames,
            freq_bins,
            device=device,
            dtype=dtype,
        )

    def forward_stream(self, x: torch.Tensor, state: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        if state is None:
            state = self.init_stream_state(
                x.shape[0], freq_bins=x.shape[-1], device=x.device, dtype=x.dtype
            )
        joined = torch.cat((state, x), dim=2)
        y = self._run(joined)
        if self.context_frames == 1 and (torch.jit.is_tracing() or x.shape[2] == 1):
            next_state = x
        else:
            next_state = joined[:, :, -self.context_frames :, :]
        return y, next_state


class LowRateTFConvBlock(nn.Module):
    """Locoformer-shaped frequency, time, and FFN residual paths at low rate."""

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
        hidden = channels * int(ffn_expansion)
        self.freq_mix = CausalDepthwiseConv2dBNAct(
            channels, kernel_size=(1, freq_kernel_size), activation=True
        )
        self.time_mix = CausalDepthwiseConv2dBNAct(
            channels,
            kernel_size=(time_kernel_size, 1),
            dilation=(time_dilation, 1),
            activation=True,
        )
        self.ffn = nn.Sequential(
            Conv2dBNAct(channels, hidden, activation=True),
            Conv2dBNAct(hidden, channels, activation=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.freq_mix(x)
        x = x + self.time_mix(x)
        return x + self.ffn(x)

    def init_stream_state(
        self,
        batch_size: int,
        *,
        freq_bins: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        return self.time_mix.init_stream_state(
            batch_size, freq_bins=freq_bins, device=device, dtype=dtype
        )

    def forward_stream(self, x: torch.Tensor, state: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.freq_mix(x)
        y, next_state = self.time_mix.forward_stream(x, state)
        x = x + y
        return x + self.ffn(x), next_state


class BandDownsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = Conv2dBNAct(
            in_channels,
            out_channels,
            kernel_size=(1, 3),
            stride=(1, 2),
            padding=(0, 1),
            activation=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class BandUpsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = Conv2dBNAct(
            in_channels,
            out_channels,
            kernel_size=(1, 3),
            padding=(0, 1),
            activation=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=(1.0, 2.0), mode="nearest")
        return self.proj(x)


class AdditiveBandPyramidSeparator(nn.Module):
    """Four-level additive pyramid with all heavy blocks at ``n_bands / 16``."""

    def __init__(
        self,
        channels: int,
        *,
        n_bands: int,
        pyramid_channels: Sequence[int],
        n_blocks: int,
        time_kernel_size: int,
        freq_kernel_size: int,
        ffn_expansion: int,
        dilation_cycle: Sequence[int],
    ) -> None:
        super().__init__()
        levels = tuple(int(value) for value in pyramid_channels)
        if len(levels) != 4:
            raise ValueError(f"pyramid_channels must contain four levels, got {levels}")
        if n_bands % 16 != 0:
            raise ValueError(f"n_bands must be divisible by 16, got {n_bands}")
        if not dilation_cycle:
            raise ValueError("dilation_cycle must not be empty")
        self.deep_freq_bins = n_bands // 16
        c0, c1, c2, c3, c4 = channels, *levels
        self.down = nn.ModuleList(
            [
                BandDownsample(c0, c1),
                BandDownsample(c1, c2),
                BandDownsample(c2, c3),
                BandDownsample(c3, c4),
            ]
        )
        self.blocks = nn.ModuleList(
            [
                LowRateTFConvBlock(
                    c4,
                    time_kernel_size=time_kernel_size,
                    time_dilation=int(dilation_cycle[idx % len(dilation_cycle)]),
                    freq_kernel_size=freq_kernel_size,
                    ffn_expansion=ffn_expansion,
                )
                for idx in range(n_blocks)
            ]
        )
        self.up = nn.ModuleList(
            [
                BandUpsample(c4, c3),
                BandUpsample(c3, c2),
                BandUpsample(c2, c1),
                BandUpsample(c1, c0),
            ]
        )

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        skips = [x]
        for layer in self.down:
            x = layer(x)
            skips.append(x)
        return x, skips

    def _decode(self, x: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        for idx, layer in enumerate(self.up):
            x = layer(x)
            x = x + skips[-2 - idx]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, skips = self._encode(x)
        for block in self.blocks:
            x = block(x)
        return self._decode(x, skips)

    def init_stream_state(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, ...]:
        return tuple(
            block.init_stream_state(
                batch_size,
                freq_bins=self.deep_freq_bins,
                device=device,
                dtype=dtype,
            )
            for block in self.blocks
        )

    def forward_stream(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        x, skips = self._encode(x)
        next_state = []
        for block, block_state in zip(self.blocks, state):
            x, block_state = block.forward_stream(x, block_state)
            next_state.append(block_state)
        return self._decode(x, skips), tuple(next_state)


class SFCSmallPyramidDWBNNPUCore(nn.Module):
    """Packed-real streaming core operating on ``[B, 2*M, T, F]``."""

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
        d_inner: int = 32,
        d_model: int = 64,
        n_separator_layers: int = 8,
        n_sfc_heads: int = 4,
        learnable_pos_bias: bool = True,
        time_kernel_size: int = 2,
        freq_kernel_size: int = 3,
        ffn_expansion: int = 2,
        dilation_cycle: Sequence[int] = (1,),
        pyramid_channels: Sequence[int] = (96, 128, 192, 256),
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
        self.separator = AdditiveBandPyramidSeparator(
            d_model,
            n_bands=n_bands,
            pyramid_channels=pyramid_channels,
            n_blocks=n_separator_layers,
            time_kernel_size=time_kernel_size,
            freq_kernel_size=freq_kernel_size,
            ffn_expansion=ffn_expansion,
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

    def state_size_bytes(self, *, batch_size: int = 1, dtype: torch.dtype = torch.float16) -> int:
        itemsize = torch.empty((), dtype=dtype).element_size()
        return sum(
            tensor.numel() * itemsize
            for tensor in self.init_stream_state(batch_size=batch_size, dtype=dtype)
        )


class SFCSmallPyramidDWBNNPUModel(SFCSmallConv2DBNNPUModel):
    def __init__(self, **core_kwargs) -> None:
        nn.Module.__init__(self)
        self.core = SFCSmallPyramidDWBNNPUCore(**core_kwargs)
        self.n_src = self.core.n_src
        self.n_chan = self.core.n_chan


def build_sfc_small_pyramid_dw_bn_npu_system(
    *,
    n_fft: int,
    hop_length: int,
    fs: int,
    n_src: int = 3,
    n_chan: int = 1,
    n_bands: int = 64,
    band_config: str = "musical",
    d_inner: int = 32,
    d_model: int = 64,
    n_separator_layers: int = 8,
    n_sfc_heads: int = 4,
    learnable_pos_bias: bool = True,
    time_kernel_size: int = 2,
    freq_kernel_size: int = 3,
    ffn_expansion: int = 2,
    dilation_cycle: Sequence[int] = (1,),
    pyramid_channels: Sequence[int] = (96, 128, 192, 256),
    encoder_ffn_expansion: int = 2,
    decoder_ffn_hidden: int = 16,
    masking: bool = True,
    use_learnable_query: bool = True,
    scaling: bool = False,
    css_segment_size: int = 12,
    css_shift_size: int = 6,
    css_batch_size: int = 1,
) -> OnlineModelWrapper:
    model = SFCSmallPyramidDWBNNPUModel(
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
        pyramid_channels=pyramid_channels,
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
    "AdditiveBandPyramidSeparator",
    "SFCSmallExactDecoder",
    "SFCSmallExactEncoder",
    "SFCSmallPyramidDWBNNPUCore",
    "SFCSmallPyramidDWBNNPUModel",
    "build_sfc_small_pyramid_dw_bn_npu_system",
]
