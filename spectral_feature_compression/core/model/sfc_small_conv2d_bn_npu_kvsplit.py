"""KV-split low-latency variant of the NPU-friendly SFC-small model.

This variant is semantically equivalent to
``sfc_small_conv2d_bn_npu.SFCSmallConv2DBNNPUCore`` but changes two export-time
details:

* the shared ``kv_proj`` Conv2D is split into separate key/value Conv2D layers;
* the attention scale is absorbed into the learnable/adaptive query path.

The goal is to keep the SFC encoder/decoder transport while removing the
remaining key/value StridedSlice ops and score-scale Mul ops from the streaming
Circle graph.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn

from spectral_feature_compression.core.model.model_wrapper import ModelWrapper
from spectral_feature_compression.core.model.sfc_small_conv2d_bn_npu import (
    SFCSmallConv2DBNDecoder,
    SFCSmallConv2DBNEncoder,
    SFCSmallConv2DBNNPUCore,
    SFCSmallConv2DBNNPUModel,
    _apply_packed_complex_mask,
)


def _head_scale_from_query(query: torch.Tensor) -> float:
    return float(query.shape[-1]) ** -0.5


def convert_sfc_small_conv2d_bn_npu_state_dict_to_kvsplit(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Map a base SFC-small state dict into the KV-split parameterization.

    ``Conv2d(d, 2d)`` weights are split into two ``Conv2d(d, d)`` weights.  The
    base model applies ``head_dim ** -0.5`` at runtime, while this variant stores
    scaled query/adaptive-pool tensors and omits the runtime Mul.
    """

    converted: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.endswith(".kv_proj.weight"):
            out_channels = value.shape[0]
            if out_channels % 2 != 0:
                raise ValueError(f"Expected even kv_proj output channels for {key}, got {out_channels}")
            half = out_channels // 2
            prefix = key[: -len("kv_proj.weight")]
            converted[f"{prefix}key_proj.weight"] = value[:half].clone()
            converted[f"{prefix}value_proj.weight"] = value[half:].clone()
            continue

        if key.endswith(".query") and (".encoder." in key or ".decoder." in key):
            converted[key] = value.clone() * _head_scale_from_query(value)
            continue

        if key.endswith(".adaptive_pool") and ".encoder." in key:
            # Adaptive encoder queries are computed as adaptive_pool @ emb.
            # Scaling the pool is equivalent to scaling the resulting query.
            query_key = key[: -len("adaptive_pool")] + "query"
            query = state_dict.get(query_key)
            scale = _head_scale_from_query(query) if query is not None else 1.0
            converted[key] = value.clone() * scale
            continue

        converted[key] = value.clone()
    return converted


class SFCSmallConv2DBNKvSplitEncoder(SFCSmallConv2DBNEncoder):
    """Encoder with separate key/value projections and pre-scaled queries."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        d_inner = self.n_heads * self.head_dim
        del self.kv_proj
        self.key_proj = nn.Conv2d(d_inner, d_inner, kernel_size=1, bias=False)
        self.value_proj = nn.Conv2d(d_inner, d_inner, kernel_size=1, bias=False)
        if self.query_type == "learnable":
            with torch.no_grad():
                self.query.mul_(self.scale)
        else:
            self.adaptive_pool.mul_(self.scale)

    def _attend(self, h: torch.Tensor) -> torch.Tensor:
        bsz, _channels, n_frames, n_freq = h.shape
        key = self.key_proj(h)
        value = self.value_proj(h)
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
            score = torch.matmul(queries[head_idx], key_h)
            score = score + pos_bias[head_idx : head_idx + 1]
            weight = torch.softmax(score, dim=-1)
            head_outputs.append(torch.matmul(weight, value_h))

        attended = torch.cat(head_outputs, dim=-1)
        attended = attended.transpose(1, 2).reshape(bsz, n_frames, -1, self.n_bands).permute(0, 2, 1, 3)
        attended = self.aggregate(attended)
        return attended + self.ffn(attended)

    def _attend_stream_frame(self, h: torch.Tensor) -> torch.Tensor:
        bsz, _channels, _n_frames, n_freq = h.shape
        key = self.key_proj(h).reshape(bsz, self.n_heads, self.head_dim, n_freq)
        value = self.value_proj(h).reshape(bsz, self.n_heads, self.head_dim, n_freq).transpose(2, 3)

        if self.query_type == "learnable":
            query = self.query.unsqueeze(0).to(dtype=h.dtype)
        else:
            emb = h.reshape(bsz, self.n_heads, self.head_dim, n_freq).transpose(2, 3)
            pool = self.adaptive_pool.reshape(1, 1, self.n_bands, n_freq).to(dtype=h.dtype)
            query = torch.matmul(pool, emb)

        score = torch.matmul(query, key)
        score = score + self.pos_bias.unsqueeze(0).to(dtype=h.dtype)
        weight = torch.softmax(score, dim=-1)
        attended = torch.matmul(weight, value)
        attended = attended.transpose(2, 3).reshape(bsz, -1, 1, self.n_bands)
        attended = self.aggregate(attended)
        return attended + self.ffn(attended)


class SFCSmallConv2DBNKvSplitDecoder(SFCSmallConv2DBNDecoder):
    """Decoder with separate key/value projections and pre-scaled queries."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        d_inner = self.n_heads * self.head_dim
        del self.kv_proj
        self.key_proj = nn.Conv2d(d_inner, d_inner, kernel_size=1, bias=False)
        self.value_proj = nn.Conv2d(d_inner, d_inner, kernel_size=1, bias=False)
        with torch.no_grad():
            self.query.mul_(self.scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input(x)
        bsz, _channels, n_frames, n_bands = h.shape
        key = self.key_proj(h)
        value = self.value_proj(h)
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
            score = torch.matmul(query_h, key_h)
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
        key = self.key_proj(h).reshape(bsz, self.n_heads, self.head_dim, n_bands)
        value = self.value_proj(h).reshape(bsz, self.n_heads, self.head_dim, n_bands).transpose(2, 3)

        query = self.query.unsqueeze(0).to(dtype=h.dtype)
        score = torch.matmul(query, key)
        score = score + self.pos_bias.unsqueeze(0).to(dtype=h.dtype)
        weight = torch.softmax(score, dim=-1)
        expanded = torch.matmul(weight, value)
        expanded = expanded.transpose(2, 3).reshape(bsz, -1, 1, self.n_freq)
        expanded = self.aggregate(expanded)
        expanded = expanded + self.ffn(expanded)
        return self.output(expanded)


class SFCSmallConv2DBNNPUKvSplitCore(SFCSmallConv2DBNNPUCore):
    """SFC-small NPU core with KV-split SFC transport."""

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
        super().__init__(
            n_freq=n_freq,
            n_fft=n_fft,
            sample_rate=sample_rate,
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
        in_channels = 2 * self.n_chan
        out_channels = 2 * self.n_src * self.n_chan
        self.encoder = SFCSmallConv2DBNKvSplitEncoder(
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
        self.decoder = SFCSmallConv2DBNKvSplitDecoder(
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


class SFCSmallConv2DBNNPUKvSplitModel(SFCSmallConv2DBNNPUModel):
    """Complex-STFT wrapper for ``SFCSmallConv2DBNNPUKvSplitCore``."""

    def __init__(self, **core_kwargs) -> None:
        nn.Module.__init__(self)
        self.core = SFCSmallConv2DBNNPUKvSplitCore(**core_kwargs)
        self.n_src = self.core.n_src
        self.n_chan = self.core.n_chan


def build_sfc_small_conv2d_bn_npu_kvsplit_system(
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
    core_model = SFCSmallConv2DBNNPUKvSplitModel(
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


__all__ = [
    "SFCSmallConv2DBNKvSplitDecoder",
    "SFCSmallConv2DBNKvSplitEncoder",
    "SFCSmallConv2DBNNPUKvSplitCore",
    "SFCSmallConv2DBNNPUKvSplitModel",
    "build_sfc_small_conv2d_bn_npu_kvsplit_system",
    "convert_sfc_small_conv2d_bn_npu_state_dict_to_kvsplit",
]
