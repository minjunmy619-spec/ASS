"""Thin wrappers around ``spectral_feature_compression.utils.onnx_streaming``.

These helpers bridge the ``BandSCNetNPUState`` NamedTuple to the flat tuple
form expected by ``torch.onnx.export``.
"""
from __future__ import annotations

import torch

from spectral_feature_compression.utils.onnx_streaming import (
    TensorTreeSpec,
    flatten_tensor_tree,
    unflatten_tensor_tree,
)

from .band_scnet_npu import BandSCNetNPU, BandSCNetNPUState


def build_example_state_and_spec(
    model: BandSCNetNPU,
    *,
    batch_size: int = 1,
    dtype: torch.dtype = torch.float32,
) -> tuple[tuple[torch.Tensor, ...], TensorTreeSpec]:
    """Return ``(flat_state, spec)`` suitable for ``torch.onnx.export``."""
    example = tuple(model.init_stream_state(batch_size=batch_size, dtype=dtype))
    flat, spec = flatten_tensor_tree(example)
    return flat, spec


def restore_state_from_flat(
    flat_state: tuple[torch.Tensor, ...],
    spec: TensorTreeSpec,
) -> BandSCNetNPUState:
    """Convert the flat tuple back into a ``BandSCNetNPUState``."""
    tree = unflatten_tensor_tree(flat_state, spec)
    return BandSCNetNPUState(*tree)


__all__ = [
    "build_example_state_and_spec",
    "restore_state_from_flat",
]
