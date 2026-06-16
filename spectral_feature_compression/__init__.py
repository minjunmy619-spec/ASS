# Copyright (c) 2026 National Institute of Advanced Industrial Science and Technology (AIST), Japan
#
# SPDX-License-Identifier: MIT

"""Top-level lazy exports for spectral_feature_compression.

The online/NPU smoke tests should not require optional offline dependencies
such as librosa or mamba_ssm at import time.  Keep top-level names available,
but import their implementation only when the name is actually requested.
"""

from __future__ import annotations

from importlib import import_module

_MODEL = "spectral_feature_compression.core.model"

_EXPORTS = {
    "BanditDecoder": f"{_MODEL}.bandit_split",
    "BanditEncoder": f"{_MODEL}.bandit_split",
    "AdaptiveMelBandSpec2d": f"{_MODEL}.adaptive_mel_sfc_2d",
    "BSLocoformer": f"{_MODEL}.bslocoformer",
    "CrossAttnDecoder": f"{_MODEL}.crossattn_enc_dec",
    "CrossAttnEncoder": f"{_MODEL}.crossattn_enc_dec",
    "FrequencyPreprocessedOnlineModel": f"{_MODEL}.frequency_preprocessing",
    "FOAEventQueryPromptedAsymmetricSFC2D": f"{_MODEL}.foa_event_query_prompted_sfc",
    "FOAEventQueryPromptedAsymmetricSFCModel": f"{_MODEL}.foa_event_query_prompted_sfc",
    "HybridFrequencyProjector2d": f"{_MODEL}.frequency_preprocessing",
    "PCENGainNormalizer2d": f"{_MODEL}.frequency_preprocessing",
    "PooledChannelCapacityMixer2d": f"{_MODEL}.npu_capacity_blocks_2d",
    "AxialTransformerBlock2d": f"{_MODEL}.foa_event_query_prompted_sfc",
    "EventClassQueryEmbedding": f"{_MODEL}.foa_event_query_prompted_sfc",
    "EventConditionedDecoderBlock2d": f"{_MODEL}.foa_event_query_prompted_sfc",
    "MambaDecoder": f"{_MODEL}.mamba_enc_dec",
    "MambaEncoder": f"{_MODEL}.mamba_enc_dec",
    "AdaptiveMelLocoformerLiteBlock2d": f"{_MODEL}.adaptive_mel_locoformer_lite_sfc_2d",
    "OnlineAdaptiveMelLocoformerLiteSFC2D": f"{_MODEL}.adaptive_mel_locoformer_lite_sfc_2d",
    "OnlineAdaptiveMelLocoformerLiteSFCModel": f"{_MODEL}.adaptive_mel_locoformer_lite_sfc_2d",
    "OnlineAdaptiveMelSFC2D": f"{_MODEL}.adaptive_mel_sfc_2d",
    "OnlineAdaptiveMelSFCModel": f"{_MODEL}.adaptive_mel_sfc_2d",
    "OnlineCrossAttnQuerySFC2D": f"{_MODEL}.online_crossattn_query_sfc_2d",
    "OnlineCrossAttnQuerySFCModel": f"{_MODEL}.online_crossattn_query_sfc_2d",
    "OnlineHardBandSFC2D": f"{_MODEL}.online_hard_band_sfc_2d",
    "OnlineHardBandSFCModel": f"{_MODEL}.online_hard_band_sfc_2d",
    "OnlineHierarchicalSoftBandSFC2D": f"{_MODEL}.online_hierarchical_soft_band_sfc_2d",
    "OnlineHierarchicalSoftBandSFCModel": f"{_MODEL}.online_hierarchical_soft_band_sfc_2d",
    "OnlineHierarchicalSoftBandFFISFC2D": f"{_MODEL}.online_hierarchical_soft_band_ffi_sfc_2d",
    "OnlineHierarchicalSoftBandFFISFCModel": f"{_MODEL}.online_hierarchical_soft_band_ffi_sfc_2d",
    "OnlineHierarchicalSoftBandParallelFFISFC2D": (f"{_MODEL}.online_hierarchical_soft_band_parallel_ffi_sfc_2d"),
    "OnlineHierarchicalSoftBandParallelFFISFCModel": (f"{_MODEL}.online_hierarchical_soft_band_parallel_ffi_sfc_2d"),
    "OnlineModelWrapper": f"{_MODEL}.online_model_wrapper",
    "OnlineSFC2D": f"{_MODEL}.online_sfc_2d",
    "OnlineSFCModel": f"{_MODEL}.online_wrapper",
    "OnlineSoftBandDilatedSFC2D": f"{_MODEL}.online_soft_band_dilated_sfc_2d",
    "OnlineSoftBandDilatedSFCModel": f"{_MODEL}.online_soft_band_dilated_sfc_2d",
    "OnlineSoftBandGRUSFC2D": f"{_MODEL}.online_soft_band_gru_sfc_2d",
    "OnlineSoftBandGRUSFCModel": f"{_MODEL}.online_soft_band_gru_sfc_2d",
    "OnlineSoftBandQuerySFC2D": f"{_MODEL}.online_soft_band_query_sfc_2d",
    "OnlineSoftBandQuerySFCModel": f"{_MODEL}.online_soft_band_query_sfc_2d",
    "OnlineSoftBandSFC2D": f"{_MODEL}.online_soft_band_sfc_2d",
    "OnlineSoftBandSFCModel": f"{_MODEL}.online_soft_band_sfc_2d",
    "OnlinePromptedAsymmetricSFC2D": f"{_MODEL}.prompted_asymmetric_sfc_2d",
    "OnlinePromptedAsymmetricSFCModel": f"{_MODEL}.prompted_asymmetric_sfc_2d",
    "PromptConditioner2d": f"{_MODEL}.prompted_asymmetric_sfc_2d",
    "PromptedSharedDecoder2d": f"{_MODEL}.prompted_asymmetric_sfc_2d",
    "PromptedTokenSplitter2d": f"{_MODEL}.prompted_asymmetric_sfc_2d",
    "LowRankResidualCorrectionHead2d": f"{_MODEL}.source_aware_residual_sfc_2d",
    "Mamba2LiteTemporalBranch2d": f"{_MODEL}.residual_refinement_sfc_2d",
    "OnlineResidualRefinementSFC2D": f"{_MODEL}.residual_refinement_sfc_2d",
    "OnlineResidualRefinementSFCModel": f"{_MODEL}.residual_refinement_sfc_2d",
    "OnlineSourceAwareResidualSFC2D": f"{_MODEL}.source_aware_residual_sfc_2d",
    "OnlineSourceAwareResidualSFCModel": f"{_MODEL}.source_aware_residual_sfc_2d",
    "OnlineSourceAwareMelBandLocoCNBStudentSFC2D": f"{_MODEL}.source_aware_melband_loco_cnb_student_sfc_2d",
    "OnlineSourceAwareMelBandLocoCNBStudentSFCModel": f"{_MODEL}.source_aware_melband_loco_cnb_student_sfc_2d",
    "OnlineSourceAwareMelBandStrongStudentSFC2D": f"{_MODEL}.source_aware_melband_strong_student_sfc_2d",
    "OnlineSourceAwareMelBandStrongStudentSFCModel": f"{_MODEL}.source_aware_melband_strong_student_sfc_2d",
    "OnlineSourceAwareMelBandStudentSFC2D": f"{_MODEL}.source_aware_melband_student_sfc_2d",
    "OnlineSourceAwareMelBandStudentSFCModel": f"{_MODEL}.source_aware_melband_student_sfc_2d",
    "RotarySelfAttention": f"{_MODEL}.source_aware_melband_roformer",
    "SourceAwareMelBandRoformer2D": f"{_MODEL}.source_aware_melband_roformer",
    "SourceAwareMelBandRoformerModel": f"{_MODEL}.source_aware_melband_roformer",
    "SourceAwareLocoCNBBlock2d": f"{_MODEL}.source_aware_melband_loco_cnb_student_sfc_2d",
    "SourceCompetitiveDecoder2d": f"{_MODEL}.source_aware_melband_student_sfc_2d",
    "SourceCompetitiveDecoderBlock2d": f"{_MODEL}.source_aware_melband_student_sfc_2d",
    "SourceCompetitionFusion2d": f"{_MODEL}.source_aware_melband_student_sfc_2d",
    "SourceSeedSplitter2d": f"{_MODEL}.source_aware_melband_student_sfc_2d",
    "SourceSharedMelBandReconstructionDecoder2d": f"{_MODEL}.source_aware_melband_student_sfc_2d",
    "LocoCompressedBandAttentionFusion2d": f"{_MODEL}.source_aware_melband_loco_cnb_student_sfc_2d",
    "LocoFSMNBandMixer2d": f"{_MODEL}.source_aware_melband_loco_cnb_student_sfc_2d",
    "LocoLocalTFMixer2d": f"{_MODEL}.source_aware_melband_loco_cnb_student_sfc_2d",
    "StrongAdaptiveMelRouter2d": f"{_MODEL}.source_aware_melband_strong_student_sfc_2d",
    "StrongMaskCorrectionHead2d": f"{_MODEL}.source_aware_melband_strong_student_sfc_2d",
    "StrongMelBandExpander2d": f"{_MODEL}.source_aware_melband_strong_student_sfc_2d",
    "StrongSourceCompetitionBlock2d": f"{_MODEL}.source_aware_melband_strong_student_sfc_2d",
    "StrongSourceDecoder2d": f"{_MODEL}.source_aware_melband_strong_student_sfc_2d",
    "StrongSourceMaskHead2d": f"{_MODEL}.source_aware_melband_strong_student_sfc_2d",
    "StrongSourceSeed2d": f"{_MODEL}.source_aware_melband_strong_student_sfc_2d",
    "StrongTemporalBandBlock2d": f"{_MODEL}.source_aware_melband_strong_student_sfc_2d",
    "StrongTokenFFN2d": f"{_MODEL}.source_aware_melband_strong_student_sfc_2d",
    "OnlineSourceSplitSFC2D": f"{_MODEL}.source_split_sfc_2d",
    "OnlineSourceSplitSFCModel": f"{_MODEL}.source_split_sfc_2d",
    "ResidualCorrectionHead2d": f"{_MODEL}.residual_refinement_sfc_2d",
    "SharedSourceRefiner2d": f"{_MODEL}.source_split_sfc_2d",
    "SourceSharedReconstructionDecoder2d": f"{_MODEL}.source_split_sfc_2d",
    "SourceTokenSplitter2d": f"{_MODEL}.source_split_sfc_2d",
    "SparseBandUNetDecoder": f"{_MODEL}.sparse_unet_mel_sfc_2d",
    "SparseBandUNetEncoder": f"{_MODEL}.sparse_unet_mel_sfc_2d",
    "SparseUNetMelSFC2D": f"{_MODEL}.sparse_unet_mel_sfc_2d",
    "SparseUNetMelSFCModel": f"{_MODEL}.sparse_unet_mel_sfc_2d",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
