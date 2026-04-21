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


_EXPORTS = {
    "BanditDecoder": "spectral_feature_compression.core.model.bandit_split",
    "BanditEncoder": "spectral_feature_compression.core.model.bandit_split",
    "BSLocoformer": "spectral_feature_compression.core.model.bslocoformer",
    "CrossAttnDecoder": "spectral_feature_compression.core.model.crossattn_enc_dec",
    "CrossAttnEncoder": "spectral_feature_compression.core.model.crossattn_enc_dec",
    "FrequencyPreprocessedOnlineModel": "spectral_feature_compression.core.model.frequency_preprocessing",
    "HybridFrequencyProjector2d": "spectral_feature_compression.core.model.frequency_preprocessing",
    "MambaDecoder": "spectral_feature_compression.core.model.mamba_enc_dec",
    "MambaEncoder": "spectral_feature_compression.core.model.mamba_enc_dec",
    "OnlineCrossAttnQuerySFC2D": "spectral_feature_compression.core.model.online_crossattn_query_sfc_2d",
    "OnlineCrossAttnQuerySFCModel": "spectral_feature_compression.core.model.online_crossattn_query_sfc_2d",
    "OnlineHardBandSFC2D": "spectral_feature_compression.core.model.online_hard_band_sfc_2d",
    "OnlineHardBandSFCModel": "spectral_feature_compression.core.model.online_hard_band_sfc_2d",
    "OnlineHierarchicalSoftBandSFC2D": "spectral_feature_compression.core.model.online_hierarchical_soft_band_sfc_2d",
    "OnlineHierarchicalSoftBandSFCModel": "spectral_feature_compression.core.model.online_hierarchical_soft_band_sfc_2d",
    "OnlineHierarchicalSoftBandFFISFC2D": "spectral_feature_compression.core.model.online_hierarchical_soft_band_ffi_sfc_2d",
    "OnlineHierarchicalSoftBandFFISFCModel": "spectral_feature_compression.core.model.online_hierarchical_soft_band_ffi_sfc_2d",
    "OnlineHierarchicalSoftBandParallelFFISFC2D": "spectral_feature_compression.core.model.online_hierarchical_soft_band_parallel_ffi_sfc_2d",
    "OnlineHierarchicalSoftBandParallelFFISFCModel": "spectral_feature_compression.core.model.online_hierarchical_soft_band_parallel_ffi_sfc_2d",
    "OnlineModelWrapper": "spectral_feature_compression.core.model.online_model_wrapper",
    "OnlineSFC2D": "spectral_feature_compression.core.model.online_sfc_2d",
    "OnlineSFCModel": "spectral_feature_compression.core.model.online_wrapper",
    "OnlineSoftBandDilatedSFC2D": "spectral_feature_compression.core.model.online_soft_band_dilated_sfc_2d",
    "OnlineSoftBandDilatedSFCModel": "spectral_feature_compression.core.model.online_soft_band_dilated_sfc_2d",
    "OnlineSoftBandGRUSFC2D": "spectral_feature_compression.core.model.online_soft_band_gru_sfc_2d",
    "OnlineSoftBandGRUSFCModel": "spectral_feature_compression.core.model.online_soft_band_gru_sfc_2d",
    "OnlineSoftBandQuerySFC2D": "spectral_feature_compression.core.model.online_soft_band_query_sfc_2d",
    "OnlineSoftBandQuerySFCModel": "spectral_feature_compression.core.model.online_soft_band_query_sfc_2d",
    "OnlineSoftBandSFC2D": "spectral_feature_compression.core.model.online_soft_band_sfc_2d",
    "OnlineSoftBandSFCModel": "spectral_feature_compression.core.model.online_soft_band_sfc_2d",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
