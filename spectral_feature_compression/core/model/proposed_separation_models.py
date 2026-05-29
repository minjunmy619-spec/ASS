"""Factory entry points for the 2026 edge-separation proposal set.

The research notes point to four repo-local implementation tracks.  This file
keeps them as thin, explicit builders over existing model families so configs
can select the proposal by name without silently changing older recipes.
"""

from __future__ import annotations

from typing import Sequence

from BandSFCNetNPU.training_wrapper import build_band_sfc_net_npu_system
from EdgeFusionNPU.training_wrapper import build_edge_fusion_npu_system

from spectral_feature_compression.core.model.bslocoformer import BSLocoformer
from spectral_feature_compression.core.model.crossattn_enc_dec import CrossAttnDecoder, CrossAttnEncoder
from spectral_feature_compression.core.model.model_wrapper import ModelWrapper
from spectral_feature_compression.core.model.online_hierarchical_soft_band_parallel_ffi_sfc_2d import (
    build_online_hierarchical_soft_band_parallel_ffi_sfc_system,
)


def build_sfc_locoformer_lite_plus_system(
    *,
    n_fft: int,
    hop_length: int,
    fs: int,
    n_src: int = 4,
    n_chan: int = 2,
    n_bands: int = 64,
    band_config: str = "musical",
    query_type: str = "learnable",
    d_inner: int = 64,
    d_model: int = 96,
    n_layers: int = 4,
    n_heads: int = 4,
    attention_dim: int | None = None,
    ffn_type: str | Sequence[str] = ("swiglu_conv1d", "swiglu_conv1d"),
    ffn_hidden_dim: int | Sequence[int] | None = None,
    conv1d_kernel: int = 8,
    conv1d_shift: int = 1,
    dropout: float = 0.1,
    flash_attention: bool = True,
    masking: bool = True,
    scaling: bool = True,
    css_segment_size: int = 12,
    css_shift_size: int = 6,
    css_batch_size: int = 1,
    checkpointing: bool = False,
) -> ModelWrapper:
    """Proposal A: SFC-CA encoder/decoder with a compact TF-Locoformer core."""

    attention_dim = d_model if attention_dim is None else attention_dim
    ffn_type_arg: str | Sequence[str]
    if isinstance(ffn_type, str):
        ffn_type_arg = [ffn_type]
        ffn_hidden_dim = [2 * d_model] if ffn_hidden_dim is None else ffn_hidden_dim
        if isinstance(ffn_hidden_dim, int):
            ffn_hidden_dim = [ffn_hidden_dim]
    else:
        ffn_type_arg = ffn_type
        ffn_hidden_dim = tuple(2 * d_model for _ in ffn_type_arg) if ffn_hidden_dim is None else ffn_hidden_dim
        if isinstance(ffn_hidden_dim, int):
            ffn_hidden_dim = tuple(ffn_hidden_dim for _ in ffn_type_arg)
    encoder = CrossAttnEncoder(
        d_inner=d_inner,
        d_model=d_model,
        n_chan=n_chan,
        sample_rate=fs,
        n_fft=n_fft,
        n_bands=n_bands,
        band_config=band_config,
        query_type=query_type,
        n_heads=n_heads,
        slope=[1] * n_heads,
        learnable_slope=False,
        learnable_pos_bias=True,
        mask_outside_bands=False,
    )
    decoder = CrossAttnDecoder(
        d_inner=d_inner,
        d_model=d_model,
        n_src=n_src,
        n_chan=n_chan,
        sample_rate=fs,
        n_fft=n_fft,
        n_bands=n_bands,
        band_config=band_config,
        query_type=query_type,
        n_heads=n_heads,
        slope=[1] * n_heads,
        learnable_slope=False,
        learnable_pos_bias=True,
        mask_outside_bands=False,
    )
    model = BSLocoformer(
        encoder=encoder,
        decoder=decoder,
        n_src=n_src,
        n_chan=n_chan,
        n_layers=n_layers,
        emb_dim=d_model,
        norm_type="rmsgroupnorm",
        num_groups=4,
        tf_order="ft",
        n_heads=n_heads,
        flash_attention=flash_attention,
        attention_dim=attention_dim,
        ffn_type=ffn_type_arg,
        ffn_hidden_dim=ffn_hidden_dim,
        conv1d_kernel=conv1d_kernel,
        conv1d_shift=conv1d_shift,
        dropout=dropout,
        masking=masking,
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


def build_band_sfc_net_rt_plus_system(
    *,
    n_fft: int,
    hop_length: int,
    fs: int,
    n_src: int = 3,
    n_chan: int = 1,
    scaling: bool = False,
    freq_preprocess_enabled: bool = True,
    freq_preprocess_keep_bins: int | None = 475,
    freq_preprocess_target_bins: int | None = 512,
    freq_preprocess_mode: str = "triangular",
    css_segment_size: int = 12,
    css_shift_size: int = 6,
    css_batch_size: int = 1,
):
    """Proposal B: BandSFCNet-RT+ edge student with residual correction head."""

    return build_band_sfc_net_npu_system(
        n_fft=n_fft,
        hop_length=hop_length,
        fs=fs,
        n_src=n_src,
        n_chan=n_chan,
        preset="rt_plus",
        scaling=scaling,
        freq_preprocess_enabled=freq_preprocess_enabled,
        freq_preprocess_keep_bins=freq_preprocess_keep_bins,
        freq_preprocess_target_bins=freq_preprocess_target_bins,
        freq_preprocess_mode=freq_preprocess_mode,
        css_segment_size=css_segment_size,
        css_shift_size=css_shift_size,
        css_batch_size=css_batch_size,
    )


def build_hierarchical_sfc_ffi_lite_system(
    *,
    n_fft: int,
    hop_length: int,
    fs: int,
    n_src: int = 3,
    n_chan: int = 1,
    pre_bands: int = 128,
    mid_bands: int = 96,
    bottleneck_bands: int = 48,
    d_model: int = 20,
    pre_layers: int = 0,
    mid_layers: int = 1,
    bottleneck_layers: int = 1,
    time_branch_kernel_sizes: Sequence[int] = (3, 3),
    time_branch_dilations: Sequence[int] = (1, 6),
    band_config: str = "musical",
    freq_preprocess_enabled: bool = True,
    freq_preprocess_keep_bins: int | None = 475,
    freq_preprocess_target_bins: int | None = 512,
    freq_preprocess_mode: str = "triangular",
    scaling: bool = False,
    css_segment_size: int = 12,
    css_shift_size: int = 6,
    css_batch_size: int = 1,
):
    """Proposal C: hierarchical SFC-FFI middle-tier student."""

    return build_online_hierarchical_soft_band_parallel_ffi_sfc_system(
        n_fft=n_fft,
        hop_length=hop_length,
        fs=fs,
        pre_bands=pre_bands,
        mid_bands=mid_bands,
        bottleneck_bands=bottleneck_bands,
        band_config=band_config,
        n_src=n_src,
        n_chan=n_chan,
        d_model=d_model,
        pre_layers=pre_layers,
        mid_layers=mid_layers,
        bottleneck_layers=bottleneck_layers,
        time_branch_kernel_sizes=tuple(time_branch_kernel_sizes),
        time_branch_dilations=tuple(time_branch_dilations),
        freq_preprocess_enabled=freq_preprocess_enabled,
        freq_preprocess_keep_bins=freq_preprocess_keep_bins,
        freq_preprocess_target_bins=freq_preprocess_target_bins,
        freq_preprocess_mode=freq_preprocess_mode,
        scaling=scaling,
        css_segment_size=css_segment_size,
        css_shift_size=css_shift_size,
        css_batch_size=css_batch_size,
    )


def build_edgefusion_sfc_distilled_system(
    *,
    n_fft: int,
    hop_length: int,
    fs: int,
    n_src: int = 3,
    n_chan: int = 1,
    preset: str = "large-v2-hybrid-5m",
    scaling: bool = False,
    freq_preprocess_enabled: bool = True,
    freq_preprocess_keep_bins: int | None = 192,
    freq_preprocess_target_bins: int | None = 257,
    freq_preprocess_mode: str = "triangular",
    chunk_frames: int | None = None,
    detach_state_between_chunks: bool = False,
    css_segment_size: int = 12,
    css_shift_size: int = 6,
    css_batch_size: int = 1,
):
    """Proposal E: strict EdgeFusion student intended for teacher distillation."""

    return build_edge_fusion_npu_system(
        n_fft=n_fft,
        hop_length=hop_length,
        fs=fs,
        n_src=n_src,
        n_chan=n_chan,
        preset=preset,
        scaling=scaling,
        freq_preprocess_enabled=freq_preprocess_enabled,
        freq_preprocess_keep_bins=freq_preprocess_keep_bins,
        freq_preprocess_target_bins=freq_preprocess_target_bins,
        freq_preprocess_mode=freq_preprocess_mode,
        chunk_frames=chunk_frames,
        detach_state_between_chunks=detach_state_between_chunks,
        css_segment_size=css_segment_size,
        css_shift_size=css_shift_size,
        css_batch_size=css_batch_size,
    )
