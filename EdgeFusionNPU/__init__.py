"""NPU-friendly online audio separation candidates."""

from .edge_fusion_npu import (
    EdgeFusionNPU,
    EdgeFusionNPUConfig,
    build_edge_fusion_npu_preset,
)
from .training_wrapper import build_edge_fusion_npu_system

__all__ = [
    "EdgeFusionNPU",
    "EdgeFusionNPUConfig",
    "build_edge_fusion_npu_preset",
    "build_edge_fusion_npu_system",
]
