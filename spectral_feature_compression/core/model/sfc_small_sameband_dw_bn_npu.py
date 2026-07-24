"""Causal SFC-small with a fixed 64-band Conv2D separator.

The official SFC encoder performs the only full-bin to band compression, every
separator block preserves the learned band axis, and the official SFC decoder
performs the only band to full-bin expansion.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

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


class SameBandTFConvBlock(nn.Module):
    """Frequency, causal-time, and channel residual paths at fixed band count."""

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
            channels,
            kernel_size=(1, freq_kernel_size),
            activation=True,
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
        n_bands: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        return self.time_mix.init_stream_state(
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
        x = x + self.freq_mix(x)
        y, next_state = self.time_mix.forward_stream(x, state)
        x = x + y
        return x + self.ffn(x), next_state


class SameBandConv2DSeparator(nn.Module):
    """Conv2D replacement for Locoformer that never changes the SFC band axis."""

    def __init__(
        self,
        channels: int,
        *,
        n_bands: int,
        n_blocks: int,
        time_kernel_size: int,
        freq_kernel_size: int,
        ffn_expansion: int,
        dilation_cycle: Sequence[int],
    ) -> None:
        super().__init__()
        if not dilation_cycle:
            raise ValueError("dilation_cycle must not be empty")
        self.n_bands = int(n_bands)
        self.blocks = nn.ModuleList(
            [
                SameBandTFConvBlock(
                    channels,
                    time_kernel_size=time_kernel_size,
                    time_dilation=int(dilation_cycle[idx % len(dilation_cycle)]),
                    freq_kernel_size=freq_kernel_size,
                    ffn_expansion=ffn_expansion,
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
        return tuple(
            block.init_stream_state(
                batch_size,
                n_bands=self.n_bands,
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
        next_state = []
        for block, block_state in zip(self.blocks, state):
            x, block_state = block.forward_stream(x, block_state)
            next_state.append(block_state)
        return x, tuple(next_state)


class SFCSmallSameBandDWBNNPUCore(nn.Module):
    """Packed-real streaming SFC core operating on ``[B, 2*M, T, F]``."""

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
        d_model: int = 80,
        n_separator_layers: int = 8,
        n_sfc_heads: int = 4,
        learnable_pos_bias: bool = True,
        time_kernel_size: int = 2,
        freq_kernel_size: int = 3,
        ffn_expansion: int = 3,
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
        self.separator = SameBandConv2DSeparator(
            d_model,
            n_bands=n_bands,
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


class SFCSmallSameBandDWBNNPUModel(SFCSmallConv2DBNNPUModel):
    def __init__(self, **core_kwargs) -> None:
        nn.Module.__init__(self)
        self.core = SFCSmallSameBandDWBNNPUCore(**core_kwargs)
        self.n_src = self.core.n_src
        self.n_chan = self.core.n_chan


def build_sfc_small_sameband_dw_bn_npu_system(
    *,
    n_fft: int,
    hop_length: int,
    fs: int,
    n_src: int = 3,
    n_chan: int = 1,
    n_bands: int = 64,
    band_config: str = "musical",
    d_inner: int = 32,
    d_model: int = 80,
    n_separator_layers: int = 8,
    n_sfc_heads: int = 4,
    learnable_pos_bias: bool = True,
    time_kernel_size: int = 2,
    freq_kernel_size: int = 3,
    ffn_expansion: int = 3,
    dilation_cycle: Sequence[int] = (1,),
    encoder_ffn_expansion: int = 2,
    decoder_ffn_hidden: int = 16,
    masking: bool = True,
    use_learnable_query: bool = True,
    scaling: bool = False,
    css_segment_size: int = 12,
    css_shift_size: int = 6,
    css_batch_size: int = 1,
) -> OnlineModelWrapper:
    model = SFCSmallSameBandDWBNNPUModel(
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
    "SameBandConv2DSeparator",
    "SFCSmallSameBandDWBNNPUCore",
    "SFCSmallSameBandDWBNNPUModel",
    "build_sfc_small_sameband_dw_bn_npu_system",
]
