# Copyright (c) 2026 National Institute of Advanced Industrial Science and Technology (AIST), Japan
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from collections.abc import Sequence
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from spectral_feature_compression.core.model.bslocoformer import TFLocoformerBlock
from spectral_feature_compression.core.model.crossattn_enc_dec import CrossAttnDecoder, CrossAttnEncoder
from spectral_feature_compression.core.model.model_wrapper import ModelWrapper
from spectral_feature_compression.core.model.source_aware_melband_roformer import (
    MixtureSourceFusion2d,
    SourceAxisAttention2d,
)


class SourceAwareLocoformerBlock(nn.Module):
    """Shared per-source TF modeling followed by source competition and fusion."""

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        source_attention_heads: int,
        attention_dim: int,
        ffn_type: str | Sequence[str],
        ffn_hidden_dim: int | Sequence[int],
        conv1d_kernel: int,
        conv1d_shift: int,
        num_groups: int,
        dropout: float,
        tf_order: str,
        flash_attention: bool,
        layer_scale_init: float,
    ) -> None:
        super().__init__()
        self.tf_block = TFLocoformerBlock(
            emb_dim=d_model,
            norm_type="rmsgroupnorm",
            num_groups=num_groups,
            tf_order=tf_order,
            n_heads=n_heads,
            flash_attention=flash_attention,
            attention_dim=attention_dim,
            ffn_type=ffn_type,
            ffn_hidden_dim=ffn_hidden_dim,
            conv1d_kernel=conv1d_kernel,
            conv1d_shift=conv1d_shift,
            dropout=dropout,
        )
        self.source_attention = SourceAxisAttention2d(
            d_model,
            n_heads=source_attention_heads,
            dropout=dropout,
            layer_scale_init=layer_scale_init,
        )
        self.mixture_fusion = MixtureSourceFusion2d(
            d_model,
            dropout=dropout,
            layer_scale_init=layer_scale_init,
        )
        self.tf_scale = nn.Parameter(torch.full((1,), float(layer_scale_init)))

    def forward(self, source_tokens: torch.Tensor, mixture_tokens: torch.Tensor) -> torch.Tensor:
        batch, n_src, channels, n_frames, n_bands = source_tokens.shape
        flat_sources = source_tokens.reshape(batch * n_src, channels, n_frames, n_bands)
        modeled_sources = self.tf_block(flat_sources)
        source_tokens = (flat_sources + self.tf_scale * (modeled_sources - flat_sources)).reshape(
            batch, n_src, channels, n_frames, n_bands
        )
        source_tokens = self.source_attention(source_tokens)
        return self.mixture_fusion(source_tokens, mixture_tokens)


class SourceAwareSFCLocoformerTeacher(nn.Module):
    """Early-split SFC/TF-Locoformer teacher with exact mixture consistency."""

    def __init__(
        self,
        *,
        encoder: CrossAttnEncoder,
        decoder: CrossAttnDecoder,
        n_src: int = 3,
        n_chan: int = 1,
        d_model: int = 144,
        n_shared_layers: int = 4,
        n_source_layers: int = 4,
        n_heads: int = 8,
        source_attention_heads: int = 4,
        attention_dim: int = 144,
        ffn_type: str | Sequence[str] = ("swiglu_conv1d", "swiglu_conv1d"),
        ffn_hidden_dim: int | Sequence[int] = (224, 224),
        conv1d_kernel: int = 8,
        conv1d_shift: int = 1,
        num_groups: int = 8,
        dropout: float = 0.1,
        tf_order: str = "ft",
        flash_attention: bool = True,
        decoder_feature_channels: int = 16,
        residual_scale_init: float = 0.05,
        layer_scale_init: float = 0.1,
        checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if n_src < 1:
            raise ValueError(f"n_src must be positive, got {n_src}")
        if residual_scale_init <= 0:
            raise ValueError(f"residual_scale_init must be positive, got {residual_scale_init}")

        self.encoder = encoder
        self.decoder = decoder
        self.n_src = n_src
        self.n_chan = n_chan
        self.n_shared_layers = n_shared_layers
        self.n_source_layers = n_source_layers
        self.checkpointing = checkpointing

        block_kwargs = dict(
            emb_dim=d_model,
            norm_type="rmsgroupnorm",
            num_groups=num_groups,
            tf_order=tf_order,
            n_heads=n_heads,
            flash_attention=flash_attention,
            attention_dim=attention_dim,
            ffn_type=ffn_type,
            ffn_hidden_dim=ffn_hidden_dim,
            conv1d_kernel=conv1d_kernel,
            conv1d_shift=conv1d_shift,
            dropout=dropout,
        )
        self.shared_blocks = nn.ModuleList(TFLocoformerBlock(**block_kwargs) for _ in range(n_shared_layers))
        self.source_seed = nn.Conv2d(d_model, d_model, kernel_size=1)
        self.source_embeddings = nn.Parameter(torch.empty(n_src, d_model))
        nn.init.normal_(self.source_embeddings, std=d_model**-0.5)

        source_block_kwargs = dict(
            d_model=d_model,
            n_heads=n_heads,
            source_attention_heads=source_attention_heads,
            attention_dim=attention_dim,
            ffn_type=ffn_type,
            ffn_hidden_dim=ffn_hidden_dim,
            conv1d_kernel=conv1d_kernel,
            conv1d_shift=conv1d_shift,
            num_groups=num_groups,
            dropout=dropout,
            tf_order=tf_order,
            flash_attention=flash_attention,
            layer_scale_init=layer_scale_init,
        )
        self.source_blocks = nn.ModuleList(
            SourceAwareLocoformerBlock(**source_block_kwargs) for _ in range(n_source_layers)
        )

        decoder_channels = 2 * decoder_feature_channels
        reconstruction_channels = 4 * n_chan + 1
        self.reconstruction_head = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(decoder_channels, reconstruction_channels, kernel_size=1),
        )
        inverse_softplus = math.log(math.expm1(residual_scale_init))
        self.residual_scale_unconstrained = nn.Parameter(torch.tensor(inverse_softplus))

    @property
    def residual_scale(self) -> torch.Tensor:
        return F.softplus(self.residual_scale_unconstrained)

    def _run_block(self, block: nn.Module, *args: torch.Tensor) -> torch.Tensor:
        if self.training and self.checkpointing:
            return checkpoint(block, *args, use_reentrant=False)
        return block(*args)

    def forward(self, input: torch.Tensor, *, return_aux: bool = False):
        if input.ndim != 4:
            raise ValueError(f"input must have shape [B, C, F, T], got {tuple(input.shape)}")
        if input.shape[1] != self.n_chan:
            raise ValueError(f"expected {self.n_chan} input channels, got {input.shape[1]}")

        batch, _, n_freq, n_frames = input.shape
        mixture = input.transpose(-2, -1)
        mixture_tokens, fullband_embeddings = self._run_block(self.encoder, mixture)

        for block in self.shared_blocks:
            mixture_tokens = self._run_block(block, mixture_tokens)

        source_tokens = self.source_seed(mixture_tokens).unsqueeze(1)
        source_tokens = source_tokens + self.source_embeddings[None, :, :, None, None]
        for block in self.source_blocks:
            source_tokens = self._run_block(block, source_tokens, mixture_tokens)

        source_tokens = source_tokens.reshape(batch * self.n_src, -1, n_frames, source_tokens.shape[-1])
        fullband_embeddings = fullband_embeddings.reshape(batch, n_frames, n_freq, -1)
        fullband_embeddings = (
            fullband_embeddings[:, None]
            .expand(-1, self.n_src, -1, -1, -1)
            .reshape(batch * self.n_src * n_frames, n_freq, -1)
        )
        decoder_features, _ = self._run_block(self.decoder, source_tokens, fullband_embeddings)
        reconstruction = self.reconstruction_head(decoder_features).float()
        reconstruction = reconstruction.reshape(batch, self.n_src, -1, n_frames, n_freq)

        mask_end = 2 * self.n_chan
        residual_end = 4 * self.n_chan
        raw_mask = reconstruction[:, :, :mask_end].reshape(
            batch, self.n_src, self.n_chan, 2, n_frames, n_freq
        )
        raw_residual = reconstruction[:, :, mask_end:residual_end].reshape(
            batch, self.n_src, self.n_chan, 2, n_frames, n_freq
        )
        confidence = reconstruction[:, :, residual_end:]

        bounded_mask = 2.0 * torch.tanh(raw_mask / 2.0)
        complex_mask = torch.complex(bounded_mask[:, :, :, 0], bounded_mask[:, :, :, 1])
        complex_residual = torch.complex(raw_residual[:, :, :, 0], raw_residual[:, :, :, 1])
        raw_estimates = mixture.unsqueeze(1) * complex_mask + self.residual_scale * complex_residual

        confidence_weights = confidence.softmax(dim=1)
        correction = mixture - raw_estimates.sum(dim=1)
        estimates = raw_estimates + confidence_weights * correction.unsqueeze(1)
        estimates = estimates.transpose(-1, -2)

        if not return_aux:
            return estimates
        return estimates, {
            "bounded_mask": bounded_mask.transpose(-1, -2),
            "confidence_weights": confidence_weights.transpose(-1, -2),
            "raw_estimates": raw_estimates.transpose(-1, -2),
            "residual_scale": self.residual_scale,
        }


def build_source_aware_sfc_locoformer_teacher_system(
    *,
    n_fft: int,
    hop_length: int,
    fs: int,
    n_src: int = 3,
    n_chan: int = 1,
    n_bands: int = 80,
    band_config: str = "musical",
    d_inner: int = 112,
    d_model: int = 144,
    encoder_heads: int = 4,
    n_shared_layers: int = 4,
    n_source_layers: int = 4,
    n_heads: int = 8,
    source_attention_heads: int = 4,
    attention_dim: int | None = None,
    ffn_type: str | Sequence[str] = ("swiglu_conv1d", "swiglu_conv1d"),
    ffn_hidden_dim: int | Sequence[int] = (224, 224),
    conv1d_kernel: int = 8,
    conv1d_shift: int = 1,
    num_groups: int = 8,
    dropout: float = 0.1,
    tf_order: str = "ft",
    flash_attention: bool = True,
    decoder_feature_channels: int = 16,
    residual_scale_init: float = 0.05,
    layer_scale_init: float = 0.1,
    checkpointing: bool = False,
    scaling: bool = False,
    css_segment_size: int = 12,
    css_shift_size: int = 6,
    css_batch_size: int = 1,
) -> ModelWrapper:
    """Build the offline source-aware SFC teacher described by the model-first plan."""

    attention_dim = d_model if attention_dim is None else attention_dim
    encoder = CrossAttnEncoder(
        d_inner=d_inner,
        d_model=d_model,
        n_chan=n_chan,
        sample_rate=fs,
        n_fft=n_fft,
        n_bands=n_bands,
        band_config=band_config,
        query_type="learnable",
        n_heads=encoder_heads,
        slope=[1.0] * encoder_heads,
        learnable_slope=False,
        learnable_pos_bias=True,
        mask_outside_bands=False,
    )
    decoder = CrossAttnDecoder(
        d_inner=d_inner,
        d_model=d_model,
        n_src=1,
        n_chan=decoder_feature_channels,
        sample_rate=fs,
        n_fft=n_fft,
        n_bands=n_bands,
        band_config=band_config,
        query_type="adaptive",
        n_heads=encoder_heads,
        slope=[1.0] * encoder_heads,
        learnable_slope=False,
        learnable_pos_bias=True,
        mask_outside_bands=False,
    )
    model = SourceAwareSFCLocoformerTeacher(
        encoder=encoder,
        decoder=decoder,
        n_src=n_src,
        n_chan=n_chan,
        d_model=d_model,
        n_shared_layers=n_shared_layers,
        n_source_layers=n_source_layers,
        n_heads=n_heads,
        source_attention_heads=source_attention_heads,
        attention_dim=attention_dim,
        ffn_type=ffn_type,
        ffn_hidden_dim=ffn_hidden_dim,
        conv1d_kernel=conv1d_kernel,
        conv1d_shift=conv1d_shift,
        num_groups=num_groups,
        dropout=dropout,
        tf_order=tf_order,
        flash_attention=flash_attention,
        decoder_feature_channels=decoder_feature_channels,
        residual_scale_init=residual_scale_init,
        layer_scale_init=layer_scale_init,
        checkpointing=checkpointing,
    )
    return ModelWrapper(
        model=model,
        n_fft=n_fft,
        hop_length=hop_length,
        fs=fs,
        scaling=scaling,
        css_segment_size=css_segment_size,
        css_shift_size=css_shift_size,
        css_batch_size=css_batch_size,
    )
