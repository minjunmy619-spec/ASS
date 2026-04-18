from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn as nn


@dataclass(frozen=True)
class TensorTreeSpec:
    kind: str
    children: tuple["TensorTreeSpec", ...] = ()


@dataclass(frozen=True)
class ModuleTensorBinding:
    module_path: str
    attr_name: str

    @property
    def qualified_name(self) -> str:
        if self.module_path:
            return f"{self.module_path}.{self.attr_name}"
        return self.attr_name


def flatten_tensor_tree(tree) -> tuple[tuple[torch.Tensor, ...], TensorTreeSpec]:
    if isinstance(tree, torch.Tensor):
        return (tree,), TensorTreeSpec(kind="tensor")
    if isinstance(tree, (tuple, list)):
        flat: list[torch.Tensor] = []
        children: list[TensorTreeSpec] = []
        for item in tree:
            child_flat, child_spec = flatten_tensor_tree(item)
            flat.extend(child_flat)
            children.append(child_spec)
        return tuple(flat), TensorTreeSpec(kind="tuple", children=tuple(children))
    raise TypeError(f"Unsupported tensor tree leaf type: {type(tree)!r}")


def _unflatten_tensor_tree_from_iter(flat_iter: Iterator[torch.Tensor], spec: TensorTreeSpec):
    if spec.kind == "tensor":
        return next(flat_iter)
    if spec.kind == "tuple":
        return tuple(_unflatten_tensor_tree_from_iter(flat_iter, child) for child in spec.children)
    raise ValueError(f"Unsupported tensor tree spec kind: {spec.kind!r}")


def unflatten_tensor_tree(flat: tuple[torch.Tensor, ...] | list[torch.Tensor], spec: TensorTreeSpec):
    flat_iter = iter(flat)
    tree = _unflatten_tensor_tree_from_iter(flat_iter, spec)
    try:
        next(flat_iter)
    except StopIteration:
        return tree
    raise ValueError("Flat tensor list contains more elements than the tensor tree spec expects.")


def tensor_tree_shapes(tree) -> list[list[int]]:
    flat, _ = flatten_tensor_tree(tree)
    return [list(t.shape) for t in flat]


def _get_submodule(root: nn.Module, module_path: str) -> nn.Module:
    module = root
    if not module_path:
        return module
    for part in module_path.split("."):
        module = getattr(module, part)
    return module


def collect_external_constant_bindings(core: nn.Module) -> tuple[ModuleTensorBinding, ...]:
    bindings: list[ModuleTensorBinding] = []
    for module_path, module in core.named_modules():
        if hasattr(module, "band_bias") and isinstance(getattr(module, "band_bias"), torch.Tensor):
            bindings.append(ModuleTensorBinding(module_path=module_path, attr_name="band_bias"))
        if hasattr(module, "decode_basis") and isinstance(getattr(module, "decode_basis"), torch.Tensor):
            bindings.append(ModuleTensorBinding(module_path=module_path, attr_name="decode_basis"))
        if hasattr(module, "routing_bias") and isinstance(getattr(module, "routing_bias"), torch.Tensor):
            bindings.append(ModuleTensorBinding(module_path=module_path, attr_name="routing_bias"))
        if hasattr(module, "expansion_basis") and isinstance(getattr(module, "expansion_basis"), torch.Tensor):
            bindings.append(ModuleTensorBinding(module_path=module_path, attr_name="expansion_basis"))
        if module.__class__.__name__ == "HardBandCompressor2d" and hasattr(module, "weights"):
            bindings.append(ModuleTensorBinding(module_path=module_path, attr_name="weights"))
        if module.__class__.__name__ == "HardBandExpander2d" and hasattr(module, "basis"):
            bindings.append(ModuleTensorBinding(module_path=module_path, attr_name="basis"))
    return tuple(bindings)


def get_external_constant_tensors(core: nn.Module, bindings: tuple[ModuleTensorBinding, ...]) -> tuple[torch.Tensor, ...]:
    tensors = []
    for binding in bindings:
        module = _get_submodule(core, binding.module_path)
        tensors.append(getattr(module, binding.attr_name))
    return tuple(tensors)


@contextmanager
def override_module_tensors(
    core: nn.Module,
    bindings: tuple[ModuleTensorBinding, ...],
    tensors: tuple[torch.Tensor, ...] | list[torch.Tensor],
):
    if len(bindings) != len(tensors):
        raise ValueError(f"Expected {len(bindings)} tensors, got {len(tensors)}")

    originals: list[tuple[nn.Module, str, torch.Tensor]] = []
    try:
        for binding, tensor in zip(bindings, tensors):
            module = _get_submodule(core, binding.module_path)
            originals.append((module, binding.attr_name, getattr(module, binding.attr_name)))
            setattr(module, binding.attr_name, tensor)
        yield
    finally:
        for module, attr_name, tensor in originals:
            setattr(module, attr_name, tensor)


class ExternalizedConstantsWrapper(nn.Module):
    """
    Export helper that exposes selected band/basis constants as explicit graph
    inputs for stateless `forward(...)`.
    """

    def __init__(self, core: nn.Module):
        super().__init__()
        self.core = core
        self.constant_bindings = collect_external_constant_bindings(core)

    def forward(self, x: torch.Tensor, *flat_constants: torch.Tensor):
        with override_module_tensors(self.core, self.constant_bindings, flat_constants):
            return self.core(x)


class StreamingStateIOWrapper(nn.Module):
    """
    Export helper that flattens nested streaming state tuples into ONNX-friendly
    tensor inputs / outputs.
    """

    def __init__(
        self,
        core: nn.Module,
        *,
        batch_size: int = 1,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        externalize_constants: bool = False,
    ):
        super().__init__()
        self.core = core
        example_state = core.init_stream_state(batch_size=batch_size, device=device, dtype=dtype)
        flat_state, state_spec = flatten_tensor_tree(example_state)
        self.state_spec = state_spec
        self.state_tensor_count = len(flat_state)
        self.constant_bindings = collect_external_constant_bindings(core) if externalize_constants else ()

    def forward(self, x: torch.Tensor, *flat_inputs: torch.Tensor):
        flat_state = flat_inputs[: self.state_tensor_count]
        flat_constants = flat_inputs[self.state_tensor_count :]
        state = unflatten_tensor_tree(flat_state, self.state_spec)
        with override_module_tensors(self.core, self.constant_bindings, flat_constants):
            y, new_state = self.core.forward_stream(x, state)
        flat_new_state, _ = flatten_tensor_tree(new_state)
        return (y, *flat_new_state)
