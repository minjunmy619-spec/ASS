"""Band-SCNet-NPU: NPU-native causal 3-stem audio source separator.

See `.kiro/specs/band-scnet-npu/design.md` for full design rationale.
"""
from __future__ import annotations

from .blocks import BoundedCausalAttn, CrossBandBlock, GatedAct, NarrowBandBlock
from .band_scnet_npu import BandSCNetNPU, BandSCNetNPUStreamingExportWrapper
from .presets import build_band_scnet_npu_preset, edge_small, rt192k, rt192k_plus
from .sparse_io import (
    SparseDownsampleEncoder,
    SparseUpsampleDecoder,
    pad_n_freq_for_split,
    split_bands,
)

__all__ = [
    "BandSCNetNPU",
    "BandSCNetNPUStreamingExportWrapper",
    "BoundedCausalAttn",
    "CrossBandBlock",
    "GatedAct",
    "NarrowBandBlock",
    "SparseDownsampleEncoder",
    "SparseUpsampleDecoder",
    "build_band_scnet_npu_preset",
    "edge_small",
    "pad_n_freq_for_split",
    "rt192k",
    "rt192k_plus",
    "split_bands",
]
