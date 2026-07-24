"""NPU SFC Macaron model with softmax-free low-rank axial attention.

The separator keeps the official frequency-then-time and
``FFN -> attention -> FFN`` residual skeleton. Attention is implemented as a
rank-factorized, content-dependent global mixer using only Conv2D, sigmoid,
elementwise arithmetic, and fixed-axis average pooling or bounded causal
running contexts in the streaming path.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from spectral_feature_compression.core.model.online_model_wrapper import OnlineModelWrapper
from spectral_feature_compression.core.model.sfc_small_conv2d_bn_npu import (
    Conv2dBNAct,
    SFCSmallConv2DBNNPUModel,
)
from spectral_feature_compression.core.model.sfc_small_macaron_conv2d_bn_npu import (
    FactorizedAxisSwiGLUFFN2D,
    SFCSmallMacaronConv2DBNNPUCore,
)


class FixedFrequencyAverage2D(nn.Module):
    """Reduce a fixed frequency axis with legal ONE AvgPool2D stages."""

    def __init__(self, n_bands: int) -> None:
        super().__init__()
        remaining = int(n_bands)
        if remaining <= 0:
            raise ValueError(f"n_bands must be positive, got {n_bands}")

        pools: list[nn.Module] = []
        while remaining > 15:
            stride = 4 if remaining % 4 == 0 else 2 if remaining % 2 == 0 else 0
            if stride == 0:
                raise ValueError(
                    f"Cannot reduce {n_bands} bands with legal stride-2/4 AvgPool2D stages"
                )
            pools.append(
                nn.AvgPool2d(
                    kernel_size=(1, stride),
                    stride=(1, stride),
                )
            )
            remaining //= stride
        if remaining > 1:
            pools.append(
                nn.AvgPool2d(
                    kernel_size=(1, remaining),
                    stride=(1, 1),
                )
            )
        self.pools = nn.ModuleList(pools)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for pool in self.pools:
            x = pool(x)
        return x


class LowRankAxialAttention2D(nn.Module):
    """Content-dependent separable axial attention without softmax or BMM.

    For rank ``r``, each branch computes

    ``q_r(i) * mean_j(k_r(j) * v(j))``.

    Frequency means cover every compressed band directly. Temporal contexts
    are exponentially decayed and causal; their sufficient statistics are
    explicit streaming state.
    """

    def __init__(
        self,
        channels: int,
        value_channels: int,
        *,
        axis: str,
        rank: int,
        n_bands: int,
        temporal_decay: float,
        frequency_context_hidden_channels: int,
    ) -> None:
        super().__init__()
        if axis not in {"frequency", "time"}:
            raise ValueError(f"Unsupported axis: {axis}")
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if value_channels <= 0:
            raise ValueError(f"value_channels must be positive, got {value_channels}")
        if frequency_context_hidden_channels <= 0:
            raise ValueError(
                "frequency_context_hidden_channels must be positive, "
                f"got {frequency_context_hidden_channels}"
            )
        if not 0.0 < temporal_decay < 1.0:
            raise ValueError(f"temporal_decay must be in (0, 1), got {temporal_decay}")

        self.axis = axis
        self.rank = int(rank)
        self.value_channels = int(value_channels)
        self.temporal_decay = float(temporal_decay)
        self.query = nn.ModuleList(
            Conv2dBNAct(channels, 1, activation=False) for _ in range(self.rank)
        )
        self.key = nn.ModuleList(
            Conv2dBNAct(channels, 1, activation=False) for _ in range(self.rank)
        )
        self.value = Conv2dBNAct(channels, value_channels, activation=False)
        self.output = Conv2dBNAct(value_channels, channels, activation=False)
        self.frequency_average = (
            FixedFrequencyAverage2D(n_bands) if axis == "frequency" else None
        )
        self.frequency_context = nn.ModuleList(
            nn.Sequential(
                Conv2dBNAct(
                    value_channels,
                    frequency_context_hidden_channels,
                    activation=True,
                ),
                Conv2dBNAct(
                    frequency_context_hidden_channels,
                    frequency_context_hidden_channels,
                    activation=True,
                ),
                Conv2dBNAct(
                    frequency_context_hidden_channels,
                    value_channels,
                    activation=False,
                ),
            )
            for _ in range(self.rank if axis == "frequency" else 0)
        )

    @staticmethod
    def _positive(x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)

    def _project(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        value = self.value(x)
        query = [self._positive(projection(x)) for projection in self.query]
        key = [self._positive(projection(x)) for projection in self.key]
        return value, query, key

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, query, key = self._project(x)
        output: torch.Tensor | None = None

        for rank_index, (query_rank, key_rank) in enumerate(zip(query, key)):
            weighted_value = key_rank * value
            if self.axis == "frequency":
                assert self.frequency_average is not None
                context = self.frequency_average(weighted_value)
                context = self.frequency_context[rank_index](context)
            else:
                steps = torch.arange(
                    x.shape[2],
                    device=x.device,
                    dtype=x.dtype,
                ).view(1, 1, -1, 1)
                powers = self.temporal_decay**steps
                context = (
                    (1.0 - self.temporal_decay)
                    * torch.cumsum(weighted_value / powers, dim=2)
                    * powers
                )
            branch = query_rank * context
            output = branch if output is None else output + branch

        assert output is not None
        return self.output(output)

    def init_stream_state(
        self,
        batch_size: int,
        *,
        n_bands: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, ...]:
        if self.axis != "time":
            raise RuntimeError("Only temporal attention has streaming state")
        return tuple(
            torch.zeros(
                batch_size,
                self.value_channels,
                1,
                n_bands,
                device=device,
                dtype=dtype,
            )
            for _ in range(self.rank)
        )

    def forward_stream(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if self.axis != "time":
            raise RuntimeError("Only temporal attention uses forward_stream")
        if len(state) != self.rank:
            raise RuntimeError(f"Expected {self.rank} attention states, got {len(state)}")
        if not torch.jit.is_tracing() and x.shape[2] != 1:
            raise RuntimeError("Streaming attention expects exactly one frame")

        value, query, key = self._project(x)
        output: torch.Tensor | None = None
        next_state: list[torch.Tensor] = []
        for query_rank, key_rank, previous in zip(query, key, state):
            current = key_rank * value
            context = current + self.temporal_decay * (previous - current)
            branch = query_rank * context
            output = branch if output is None else output + branch
            next_state.append(context)

        assert output is not None
        return self.output(output), tuple(next_state)


class LowRankMacaronAxisPath2D(nn.Module):
    """One ``FFN -> low-rank attention -> FFN`` residual axis path."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        *,
        axis: str,
        attention_rank: int,
        attention_value_channels: int,
        n_bands: int,
        temporal_decay: float,
        frequency_context_hidden_channels: int,
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
        self.attention = LowRankAxialAttention2D(
            channels,
            attention_value_channels,
            axis=axis,
            rank=attention_rank,
            n_bands=n_bands,
            temporal_decay=temporal_decay,
            frequency_context_hidden_channels=frequency_context_hidden_channels,
        )
        self.post_ffn = FactorizedAxisSwiGLUFFN2D(
            channels,
            hidden_channels,
            **ffn_kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pre_ffn(x)
        x = x + self.attention(x)
        return x + self.post_ffn(x)

    def init_stream_state(
        self,
        batch_size: int,
        *,
        n_bands: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, ...]:
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
            *self.attention.init_stream_state(**kwargs),
            self.post_ffn.init_stream_state(**kwargs),
        )

    def forward_stream(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if self.axis != "time":
            raise RuntimeError("Only the temporal path uses forward_stream")
        rank = self.attention.rank
        if len(state) != rank + 2:
            raise RuntimeError(f"Expected {rank + 2} path states, got {len(state)}")

        pre, pre_state = self.pre_ffn.forward_stream(x, state[0])
        x = x + pre
        mixed, attention_state = self.attention.forward_stream(
            x,
            state[1 : 1 + rank],
        )
        x = x + mixed
        post, post_state = self.post_ffn.forward_stream(x, state[-1])
        return x + post, (pre_state, *attention_state, post_state)


class LowRankNPUTFLocoformerBlock2D(nn.Module):
    """Frequency-then-time TF-Locoformer block with low-rank attention."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        *,
        attention_rank: int,
        attention_value_channels: int,
        n_bands: int,
        temporal_decay: float,
        frequency_context_hidden_channels: int,
        frequency_kernel_size: int,
        time_kernel_size: int,
        time_dilation: int,
    ) -> None:
        super().__init__()
        kwargs = {
            "attention_rank": attention_rank,
            "attention_value_channels": attention_value_channels,
            "n_bands": n_bands,
            "temporal_decay": temporal_decay,
            "frequency_context_hidden_channels": frequency_context_hidden_channels,
            "frequency_kernel_size": frequency_kernel_size,
            "time_kernel_size": time_kernel_size,
            "time_dilation": time_dilation,
        }
        self.freq_path = LowRankMacaronAxisPath2D(
            channels,
            hidden_channels,
            axis="frequency",
            **kwargs,
        )
        self.frame_path = LowRankMacaronAxisPath2D(
            channels,
            hidden_channels,
            axis="time",
            **kwargs,
        )

    @property
    def state_count(self) -> int:
        return self.frame_path.attention.rank + 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.frame_path(self.freq_path(x))

    def init_stream_state(
        self,
        batch_size: int,
        *,
        n_bands: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, ...]:
        return self.frame_path.init_stream_state(
            batch_size,
            n_bands=n_bands,
            device=device,
            dtype=dtype,
        )

    def forward_stream(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        x = self.freq_path(x)
        return self.frame_path.forward_stream(x, state)


class LowRankAttentionMacaronSeparator(nn.Module):
    """Same-band Macaron blocks with one shared causal averaging coefficient."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        *,
        attention_rank: int,
        attention_value_channels: int,
        n_bands: int,
        temporal_decay: float,
        frequency_context_hidden_channels: int,
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
            LowRankNPUTFLocoformerBlock2D(
                channels,
                hidden_channels,
                attention_rank=attention_rank,
                attention_value_channels=attention_value_channels,
                n_bands=n_bands,
                temporal_decay=temporal_decay,
                frequency_context_hidden_channels=frequency_context_hidden_channels,
                frequency_kernel_size=frequency_kernel_size,
                time_kernel_size=time_kernel_size,
                time_dilation=int(dilation_cycle[index % len(dilation_cycle)]),
            )
            for index in range(int(n_blocks))
        )

    @property
    def state_count(self) -> int:
        return sum(block.state_count for block in self.blocks)

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
        state: list[torch.Tensor] = []
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
        if len(state) != self.state_count:
            raise RuntimeError(f"Expected {self.state_count} separator states, got {len(state)}")

        next_state: list[torch.Tensor] = []
        offset = 0
        for block in self.blocks:
            end = offset + block.state_count
            x, block_state = block.forward_stream(x, state[offset:end])
            next_state.extend(block_state)
            offset = end
        return x, tuple(next_state)


class SFCSmallMacaronLRAttnBNNPUCore(SFCSmallMacaronConv2DBNNPUCore):
    """Exact SFC encoder/decoder around low-rank global Macaron attention."""

    def __init__(
        self,
        *,
        attention_rank: int = 2,
        attention_value_channels: int = 64,
        temporal_decay: float = 0.995,
        frequency_context_hidden_channels: int = 560,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        frequency_kernel_size = int(kwargs.get("frequency_kernel_size", 15))
        time_kernel_size = int(kwargs.get("time_kernel_size", 2))
        dilation_cycle = kwargs.get("dilation_cycle", (1,))
        ffn_hidden = int(kwargs.get("ffn_hidden", 176))
        self.attention_rank = int(attention_rank)
        self.attention_value_channels = int(attention_value_channels)
        self.separator = LowRankAttentionMacaronSeparator(
            self.d_model,
            ffn_hidden,
            attention_rank=self.attention_rank,
            attention_value_channels=self.attention_value_channels,
            n_bands=self.n_bands,
            temporal_decay=temporal_decay,
            frequency_context_hidden_channels=frequency_context_hidden_channels,
            n_blocks=self.n_separator_layers,
            frequency_kernel_size=frequency_kernel_size,
            time_kernel_size=time_kernel_size,
            dilation_cycle=dilation_cycle,
        )


class SFCSmallMacaronLRAttnBNNPUModel(SFCSmallConv2DBNNPUModel):
    def __init__(self, **core_kwargs) -> None:
        nn.Module.__init__(self)
        self.core = SFCSmallMacaronLRAttnBNNPUCore(**core_kwargs)
        self.n_src = self.core.n_src
        self.n_chan = self.core.n_chan


def build_sfc_small_macaron_lrattn_bn_npu_system(
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
    attention_rank: int = 2,
    attention_value_channels: int = 64,
    temporal_decay: float = 0.995,
    frequency_context_hidden_channels: int = 560,
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
    model = SFCSmallMacaronLRAttnBNNPUModel(
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
        attention_rank=attention_rank,
        attention_value_channels=attention_value_channels,
        temporal_decay=temporal_decay,
        frequency_context_hidden_channels=frequency_context_hidden_channels,
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
    "LowRankAttentionMacaronSeparator",
    "LowRankAxialAttention2D",
    "LowRankMacaronAxisPath2D",
    "LowRankNPUTFLocoformerBlock2D",
    "SFCSmallMacaronLRAttnBNNPUCore",
    "SFCSmallMacaronLRAttnBNNPUModel",
    "build_sfc_small_macaron_lrattn_bn_npu_system",
]
