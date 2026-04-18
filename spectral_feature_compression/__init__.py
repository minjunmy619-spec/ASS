# Copyright (c) 2026 National Institute of Advanced Industrial Science and Technology (AIST), Japan
#
# SPDX-License-Identifier: MIT

from spectral_feature_compression.core.model.bandit_split import BanditDecoder, BanditEncoder
from spectral_feature_compression.core.model.bslocoformer import BSLocoformer
from spectral_feature_compression.core.model.crossattn_enc_dec import CrossAttnDecoder, CrossAttnEncoder
from spectral_feature_compression.core.model.frequency_preprocessing import (
    FrequencyPreprocessedOnlineModel,
    HybridFrequencyProjector2d,
)
from spectral_feature_compression.core.model.online_crossattn_query_sfc_2d import (
    OnlineCrossAttnQuerySFC2D,
    OnlineCrossAttnQuerySFCModel,
)
from spectral_feature_compression.core.model.online_model_wrapper import OnlineModelWrapper
from spectral_feature_compression.core.model.online_hard_band_sfc_2d import OnlineHardBandSFC2D, OnlineHardBandSFCModel
from spectral_feature_compression.core.model.online_hierarchical_soft_band_sfc_2d import (
    OnlineHierarchicalSoftBandSFC2D,
    OnlineHierarchicalSoftBandSFCModel,
)
from spectral_feature_compression.core.model.online_hierarchical_soft_band_ffi_sfc_2d import (
    OnlineHierarchicalSoftBandFFISFC2D,
    OnlineHierarchicalSoftBandFFISFCModel,
)
from spectral_feature_compression.core.model.online_hierarchical_soft_band_parallel_ffi_sfc_2d import (
    OnlineHierarchicalSoftBandParallelFFISFC2D,
    OnlineHierarchicalSoftBandParallelFFISFCModel,
)
from spectral_feature_compression.core.model.online_soft_band_dilated_sfc_2d import (
    OnlineSoftBandDilatedSFC2D,
    OnlineSoftBandDilatedSFCModel,
)
from spectral_feature_compression.core.model.online_soft_band_gru_sfc_2d import (
    OnlineSoftBandGRUSFC2D,
    OnlineSoftBandGRUSFCModel,
)
from spectral_feature_compression.core.model.online_soft_band_query_sfc_2d import (
    OnlineSoftBandQuerySFC2D,
    OnlineSoftBandQuerySFCModel,
)
from spectral_feature_compression.core.model.online_soft_band_sfc_2d import OnlineSoftBandSFC2D, OnlineSoftBandSFCModel
from spectral_feature_compression.core.model.online_sfc_2d import OnlineSFC2D
from spectral_feature_compression.core.model.online_wrapper import OnlineSFCModel

try:
    from spectral_feature_compression.core.model.mamba_enc_dec import MambaDecoder, MambaEncoder
except ModuleNotFoundError as exc:
    if exc.name != "mamba_ssm":
        raise
    MambaDecoder = None
    MambaEncoder = None
