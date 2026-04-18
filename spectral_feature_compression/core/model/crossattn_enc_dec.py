# Copyright (c) 2026 National Institute of Advanced Industrial Science and Technology (AIST), Japan
#
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel
import torch.nn.functional as F

from spectral_feature_compression.core.model.enc_dec_base import DecoderBase, EncoderBase, RMSNorm, SwiGLUMLP


class CrossAttnEncDecForward:
    input_conv: nn.Module
    d_inner: int
    block: nn.Module
    _output_proj: nn.Module

    def forward(self, input: torch.Tensor, query: torch.Tensor | None = None):
        """
        input: torch.Tensor (n_batch, n_hidden, n_frames, n_freq)
            Input spectrogram
        query: torch.Tensor (n_batch * n_frames, n_bands, n_hidden)
            Query tensor, used when self.query_type == "input".

        hidden_states: (n_bands, n_hidden)
        """
        if input.is_complex():
            input = torch.cat((input.real, input.imag), dim=1)

        n_batch, _, n_frames, n_freq = input.shape

        emb_orig = self.input_conv(input)  # (n_batch, d_inner, n_frames, n-freq)
        emb = emb_orig.permute(0, 2, 3, 1).contiguous().view(n_batch * n_frames, n_freq, self.d_inner)

        query = self._prepare_query(emb, query_orig=query)
        f_new = query.shape[-2]

        query = self.block(emb, query)

        query = query.reshape(n_batch, n_frames, f_new, -1)
        query = self._output_proj(query)

        return query, emb


class CrossAttnEncoder(CrossAttnEncDecForward, EncoderBase):
    def __init__(
        self,
        # general setup
        d_inner: int,
        d_model: int,
        n_chan: int,
        sample_rate: int,
        n_fft: int,
        n_bands: int,
        band_config: str = "musical",
        query_type: str = "learnable",
        # cross-attention related
        n_heads: int = 4,
        slope: list[float] | None = None,
        learnable_slope: bool = False,
        learnable_pos_bias: bool = True,
        mask_outside_bands: bool = False,
        use_ffn: bool = True,
    ):
        super().__init__(
            d_inner=d_inner,
            d_model=d_model,
            n_chan=n_chan,
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_bands=n_bands,
            band_config=band_config,
            query_type=query_type,
        )

        pos_bias_matrix = prepare_bandit_position_bias(
            self.band_indices, q_len=n_bands, kv_len=n_fft // 2 + 1, n_heads=n_heads
        ).contiguous()

        if slope is None:
            slope = [1.0] * n_heads

        pos_bias = PosBias(
            pos_bias_matrix,
            self.band_indices,
            q_len=n_bands,
            kv_len=n_fft // 2 + 1,
            n_heads=n_heads,
            slope=slope,
            learnable_slope=learnable_slope,
            learnable_pos_bias=learnable_pos_bias,
            mask_outside_bands=mask_outside_bands,
        )
        self.block = CrossAttnBlock(
            d_inner, d_inner * 2, pos_bias, use_ffn=use_ffn, use_res_attn=False, use_res_ffn=True, n_heads=n_heads
        )


class CrossAttnDecoder(CrossAttnEncDecForward, DecoderBase):
    def __init__(
        self,
        d_inner: int,
        d_model: int,
        n_src: int,
        n_chan: int,
        sample_rate: int,
        n_fft: int,
        n_bands: int,
        band_config: str = "musical",
        query_type: str = "learnable",
        n_heads: int = 4,
        slope: list[float] | None = None,
        learnable_slope: bool = False,
        learnable_pos_bias: bool = True,
        mask_outside_bands: bool = False,
        use_ffn: bool = True,
    ):
        super().__init__(
            d_inner=d_inner,
            d_model=d_model,
            n_chan=n_chan,
            n_src=n_src,
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_bands=n_bands,
            query_type=query_type,
            band_config=band_config,
        )

        # initialize decoder's positional bias using the transpose of the encoder's positional bias
        pos_bias_matrix = (
            prepare_bandit_position_bias(self.band_indices, q_len=n_bands, kv_len=n_fft // 2 + 1, n_heads=n_heads)
            .transpose(-1, -2)
            .contiguous()
        )
        if slope is None:
            slope = [1.0] * n_heads

        pos_bias = PosBias(
            pos_bias_matrix,
            self.band_indices,
            q_len=n_fft // 2 + 1,
            kv_len=n_bands,
            n_heads=n_heads,
            slope=slope,
            learnable_slope=learnable_slope,
            learnable_pos_bias=learnable_pos_bias,
            mask_outside_bands=mask_outside_bands,
        )
        self.block = CrossAttnBlock(
            d_inner, d_inner * 2, pos_bias, use_ffn=use_ffn, use_res_attn=False, use_res_ffn=True, n_heads=n_heads
        )


class CrossAttnBlock(nn.Module):
    def __init__(self, d_model, mlp_dim, pos_bias, use_ffn=True, use_res_attn=False, use_res_ffn=False, n_heads=4):
        super().__init__()
        self.use_res_attn = use_res_attn
        self.use_res_ffn = use_res_ffn

        self.mixer = MultiHeadCrossAttention(emb_dim=d_model, attention_dim=d_model, pos_bias=pos_bias, n_heads=n_heads)

        self.norm_attn_kv = RMSNorm(d_model)
        self.norm_attn_q = RMSNorm(d_model)

        self.use_ffn = use_ffn
        if use_ffn:
            self.feed_forward = SwiGLUMLP(d_model=d_model, d_inner=mlp_dim)
            self.norm_ffn = RMSNorm(d_model)

    # @torch.autocast("cuda", enabled=True, dtype=torch.bfloat16)
    def forward(self, hidden_states, query):
        # attention
        residual = query

        # normalization -> cross-attention
        hidden_states = self.norm_attn_kv(hidden_states)
        query = self.norm_attn_q(query)
        query = self.mixer(hidden_states, query)

        if self.use_res_attn:
            query = residual + query

        if self.use_ffn:
            if self.use_res_ffn:
                residual = query

            # normalization -> feed-forward
            query = self.norm_ffn(query)
            query = self.feed_forward(query)

            if self.use_res_ffn:
                query = residual + query

        return query


class MultiHeadCrossAttention(nn.Module):
    def __init__(
        self, emb_dim: int, attention_dim: int, pos_bias: torch.Tensor | None, n_heads: int = 4, dropout: float = 0.0
    ):
        super().__init__()

        assert attention_dim % n_heads == 0

        self.n_heads = n_heads
        self.head_dim = attention_dim // n_heads
        self.dropout = dropout

        self.kv_proj = nn.Linear(emb_dim, attention_dim * 2, bias=False)
        self.q_proj = nn.Linear(emb_dim, attention_dim, bias=False)
        self.aggregate_heads = nn.Sequential(nn.Linear(attention_dim, emb_dim, bias=False), nn.Dropout(dropout))

        if pos_bias is not None:
            # self.register_buffer("pos_bias", pos_bias)
            self.pos_bias = pos_bias
            self.sdpa_kernel = SDPBackend.EFFICIENT_ATTENTION  # flash attention is not available with position bias
        else:
            self.pos_bias = None
            self.sdpa_kernel = SDPBackend.FLASH_ATTENTION

    @torch.autocast("cuda", enabled=True, dtype=torch.bfloat16)
    def forward(self, x_kv: torch.Tensor, x_q: torch.Tensor):
        n_batch, kv_len = x_kv.shape[:2]
        q_len = x_q.shape[1]

        # query
        q = self.q_proj(x_q)
        if q.shape[0] == 1:
            q = q.expand(n_batch, -1, -1)  # for learnable query
        q = q.view(n_batch, q_len, self.n_heads, self.head_dim).transpose(1, 2).contiguous()

        # key and value
        kv = self.kv_proj(x_kv)
        kv = kv.view(n_batch, kv_len, self.n_heads, 2, self.head_dim).transpose(1, 2).contiguous()
        k, v = kv.unbind(dim=-2)

        pos_bias = self.pos_bias(dtype=k.dtype) if self.pos_bias is not None else None

        with sdpa_kernel(self.sdpa_kernel):
            output = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=pos_bias,
                dropout_p=self.dropout if self.training else 0.0,
            )  # (batch, head, seq_len, -1)

        output = output.transpose(1, 2).reshape(n_batch, q_len, self.n_heads * self.head_dim)
        return self.aggregate_heads(output)


class PosBias(nn.Module):
    def __init__(
        self,
        pos_bias_matrix: torch.Tensor,
        band_indices: list[tuple[int, int]],
        q_len: int,
        kv_len: int,
        n_heads: int,
        slope: list[float],
        learnable_slope: bool = False,
        learnable_pos_bias: bool = False,
        mask_outside_bands: bool = False,
    ):
        super().__init__()

        # (1, n_heads, q_len, kv_len)
        self.register_buffer("pos_bias", pos_bias_matrix)

        assert len(slope) == n_heads, (n_heads, slope)
        slope = torch.Tensor(slope)[:, None, None]
        if learnable_slope:
            self.slope = nn.Parameter(slope)
        else:
            self.register_buffer("slope", slope)

        if learnable_pos_bias:
            self.pos_bias = nn.Parameter(self.pos_bias)

        self.mask_outside_bands = mask_outside_bands
        if self.mask_outside_bands:
            attention_mask = prepare_attention_mask(band_indices, q_len, kv_len, n_heads)
            self.register_buffer("attention_mask", attention_mask)

    def forward(self, dtype=None):
        pos_bias = self.pos_bias + self.attention_mask if self.mask_outside_bands else self.pos_bias
        if dtype is not None:
            pos_bias = pos_bias.to(dtype=dtype)
        return pos_bias * self.slope


def prepare_bandit_position_bias(
    band_indices: list[tuple[int, int]], q_len: int, kv_len: int, n_heads: int, middle_part: str = "gentle_slope"
) -> torch.Tensor:
    """
    NOTE: This function is implemented for the encoder part.
    The position bias for the decoder is assumed to be obtained by transposing the encoder's position bias.
    """
    position_bias = torch.zeros((q_len, kv_len))

    for q_idx in range(q_len):
        for kv_idx in range(kv_len):
            if kv_idx < band_indices[q_idx][0]:
                position_bias[q_idx][kv_idx] = kv_idx - band_indices[q_idx][0]
            elif kv_idx > band_indices[q_idx][1] - 1:
                position_bias[q_idx][kv_idx] = band_indices[q_idx][1] - 1 - kv_idx
            else:
                if middle_part == "gentle_slope":
                    position_bias[q_idx][kv_idx] = -abs(
                        (band_indices[q_idx][0] + band_indices[q_idx][1]) // 2 - kv_idx
                    ) / ((band_indices[q_idx][1] - band_indices[q_idx][0]) // 2 + 1)
                elif middle_part == "zero":
                    pass
                else:
                    raise NotImplementedError()

    position_bias = position_bias[None].repeat(n_heads, 1, 1)  # (n_heads, q_len, kv_len)
    return position_bias.unsqueeze(0)  # batch dim


def prepare_attention_mask(band_indices: list[tuple[int, int]], q_len: int, kv_len: int, n_heads: int) -> torch.Tensor:
    attention_mask = torch.zeros((q_len, kv_len))

    # for encoder
    if q_len < kv_len:
        for q_idx in range(q_len):
            for kv_idx in range(kv_len):
                if kv_idx < band_indices[q_idx][0] or kv_idx > band_indices[q_idx][1] - 1:
                    attention_mask[q_idx][kv_idx] = float("-inf")
    # for decoder
    else:
        for q_idx in range(q_len):
            for kv_idx in range(kv_len):
                if q_idx < band_indices[kv_idx][0] or q_idx > band_indices[kv_idx][1] - 1:
                    attention_mask[q_idx][kv_idx] = float("-inf")

    # repeat position bias n_head times and multiply by slope inspired by ALiBi
    attention_mask = attention_mask[None].repeat(n_heads, 1, 1)  # (n_heads, q_len, kv_len)

    return attention_mask.unsqueeze(0)  # batch dim
