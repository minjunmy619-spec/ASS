"""Shared internal helpers. Not part of the public API."""

import copy
import itertools

import torch
from torch import nn


def tree_copy(value, device):
    """Recursively detach, move, and clone tensors inside common containers."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device).clone()
    if isinstance(value, tuple):
        values = [tree_copy(v, device) for v in value]
        return type(value)(*values) if hasattr(value, "_fields") else tuple(values)
    if isinstance(value, list):
        return [tree_copy(v, device) for v in value]
    if isinstance(value, dict):
        return {k: tree_copy(v, device) for k, v in value.items()}
    return copy.deepcopy(value)


def single_device(model, description="model"):
    if not isinstance(model, nn.Module):
        raise TypeError(f"{description} must be a torch.nn.Module")
    devices = {t.device for t in itertools.chain(model.parameters(), model.buffers())}
    if len(devices) > 1:
        raise ValueError(f"{description} must be on a single device")
    return next(iter(devices), torch.device("cpu"))


def is_excluded(name, exclude):
    """Match an exact module name or any of its descendants ('' is the root)."""
    return any(ex == "" or name == ex or name.startswith(ex + ".") for ex in exclude)


def storage_interval(tensor):
    """Return (device, start, end) bytes, or None when addresses are meaningless.

    Sparse/meta tensors have no usable strided address range; callers must not
    infer aliasing from them.
    """
    if tensor.layout != torch.strided:
        return None
    storage = tensor.untyped_storage()
    start = storage.data_ptr()
    if start == 0:  # Meta/fake tensors all report a null address.
        return None
    return tensor.device, start, start + storage.nbytes()


def storages_overlap(left, right):
    """Detect shared bytes, including distinct storages over one allocation."""
    first, second = storage_interval(left), storage_interval(right)
    if first is None or second is None:
        return False
    device, start, end = first
    other_device, other_start, other_end = second
    return device == other_device and start < other_end and other_start < end


def find_aliased_tensors(model):
    """Return ids of parameters/buffers that share identity or storage bytes."""
    tensors = (list(model.named_parameters(remove_duplicate=False))
               + list(model.named_buffers(remove_duplicate=False)))
    aliased = {}
    for index, (name, tensor) in enumerate(tensors):
        for other_name, other in tensors[:index]:
            if tensor is other or storages_overlap(tensor, other):
                aliased[id(tensor)] = (name, other_name)
                aliased[id(other)] = (name, other_name)
    return aliased
