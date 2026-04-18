# Copyright (c) 2026 National Institute of Advanced Industrial Science and Technology (AIST), Japan
#
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn

from mamba_ssm import Mamba

from spectral_feature_compression.core.model.enc_dec_base import DecoderBase, EncoderBase, RMSNorm


class MambaEncDecForward:
    forward_block: nn.Module
    backward_block: nn.Module
    input_conv: nn.Module
    emb_indices: list
    query_indices: list
    bidirectional: bool

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

        emb_orig = self.input_conv(input).permute(0, 2, 3, 1).reshape(n_batch * n_frames, n_freq, self.d_inner)
        emb = emb_orig.clone()

        query = self._prepare_query(emb, query_orig=query)
        f_new = query.shape[-2]

        query_forward, emb_forward = self._process_input(
            self.forward_block, emb, query, self.emb_indices, self.query_indices, forward=True
        )
        if self.bidirectional:
            query_backward, emb_backward = self._process_input(
                self.backward_block, emb, query, self.emb_indices, self.query_indices, forward=False
            )
            query = torch.cat((query_forward, query_backward), dim=-1)
            emb = torch.cat((emb_forward, emb_backward), dim=-1)
        else:
            query = query_forward
            emb = emb_forward

        query = query.reshape(n_batch, n_frames, f_new, -1)
        query = self._output_proj(query)

        return query, emb

    def _process_input(self, block, emb, query, emb_indices, query_indices, forward=True):
        combined_tokens = self._prepare_combined_tokens(emb, query, emb_indices, query_indices)

        if not forward:
            combined_tokens = combined_tokens.flip([1])

        combined_tokens = block(combined_tokens)

        if not forward:
            combined_tokens = combined_tokens.flip([1])

        query = combined_tokens[:, query_indices]
        emb = combined_tokens[:, emb_indices].contiguous()

        return query, emb

    def _prepare_combined_tokens(self, emb, query, emb_indices, query_indices):
        n_batch, n_freq = emb.shape[:2]
        n_bands = query.shape[1]

        combined_tokens = torch.empty(n_batch, n_freq + n_bands, self.d_inner, dtype=emb.dtype, device=emb.device)

        if query.dtype != emb.dtype:
            query = query.to(emb.dtype)

        if query.shape[0] == 1:
            query = query.expand(n_batch, -1, -1)  # for learnable query

        combined_tokens[:, emb_indices] = emb
        combined_tokens[:, query_indices] = query.to(emb.dtype)

        return combined_tokens


class MambaEncoder(MambaEncDecForward, EncoderBase):
    def __init__(
        self,
        d_inner: int,
        d_model: int,
        n_chan: int,
        sample_rate: int,
        n_fft: int,
        n_bands: int,
        band_config: str,
        query_type: str,
        bidirectional: bool = True,
        d_state: int = 8,
        d_conv: int = 4,
        expand: int = 1,
    ):
        super().__init__(
            d_inner=d_inner,
            d_model=d_model,
            n_chan=n_chan,
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_bands=n_bands,
            query_type=query_type,
            band_config=band_config,
        )

        self.bidirectional = bidirectional
        if self.bidirectional:
            self._instantiate_conv_layers(d_inner_output_conv=self.d_inner * 2)

        self.forward_block = MambaBlock(d_inner, d_state=d_state, d_conv=d_conv, expand=expand)
        if self.bidirectional:
            self.backward_block = MambaBlock(d_inner, d_state=d_state, d_conv=d_conv, expand=expand)

        self.emb_indices, self.query_indices = prepare_query_indices(self.band_indices)


class MambaDecoder(MambaEncDecForward, DecoderBase):
    def __init__(
        self,
        d_inner: int,
        d_model: int,
        n_src: int,
        n_chan: int,
        sample_rate: int,
        n_fft: int,
        n_bands: int,
        band_config: str,
        query_type: str,
        bidirectional: bool = True,
        d_state: int = 8,
        d_conv: int = 4,
        expand: int = 1,
    ):
        super().__init__(
            d_inner=d_inner,
            d_model=d_model,
            n_chan=n_chan,
            n_src=n_src,
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_bands=n_bands,
            band_config=band_config,
            query_type=query_type,
        )

        self.bidirectional = bidirectional
        if self.bidirectional:
            self._instantiate_conv_layers(d_inner_output_conv=self.d_inner * 2)

        self.forward_block = MambaBlock(d_inner, d_state=d_state, d_conv=d_conv, expand=expand)
        if self.bidirectional:
            self.backward_block = MambaBlock(d_inner, d_state=d_state, d_conv=d_conv, expand=expand)

        self.query_indices, self.emb_indices = prepare_query_indices(self.band_indices)


class MambaBlock(nn.Module):
    def __init__(self, d_model, use_norm=True, use_res=False, d_state=8, d_conv=4, expand=1):
        super().__init__()
        self.use_norm = use_norm
        self.use_res = use_res
        if use_norm:
            self.norm = RMSNorm(d_model)
        self.mixer = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

    # @torch.autocast("cuda", enabled=True, dtype=torch.bfloat16)
    def forward(self, hidden_states):
        residual = hidden_states
        if self.use_norm:
            hidden_states = self.norm(hidden_states)
        hidden_states = self.mixer(hidden_states)
        if self.use_res:
            hidden_states = residual + hidden_states
        return hidden_states


def prepare_query_indices(band_indices):
    """
    Prepares query indices to interleave two features, 'emb' and 'query`, based on band_indices.

    Args:


    Returns:
        torch.Tensor: Combined tensor with shape (bf, f + f_new, d).
    """

    seq_len = band_indices[-1][1]

    band_indices = [(b[0] + b[1]) // 2 for b in band_indices]

    seq_len_new = len(band_indices)

    band_indices = torch.tensor(band_indices, dtype=torch.long)

    # we need to add (1, 2, ..., B) to band_indices to insert querys
    query_indices = band_indices + torch.arange(len(band_indices))

    # Create a boolean mask for query positions
    is_query_position = torch.zeros(seq_len + seq_len_new, dtype=torch.bool)

    is_query_position[query_indices] = True

    # Create an index for emb positions
    emb_indices = torch.where(~is_query_position)[0]

    return emb_indices, query_indices
