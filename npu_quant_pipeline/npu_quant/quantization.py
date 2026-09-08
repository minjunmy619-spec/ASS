"""Observer-based fake quantization using only PyTorch.

Only exact Linear and Conv1d/2d/3d module boundaries are quantized. Functional
operations (including add and normalization) are not quantized. Bias stays float.
Exclusions match a module name and its descendants; overrides match exact names.
The root module's name is the empty string. This is inference simulation, not QAT.
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from ._utils import find_aliased_tensors, is_excluded, storage_interval


def _validate_bits(bits):
    if isinstance(bits, bool) or not isinstance(bits, int) or not 2 <= bits <= 16:
        raise ValueError("quantization bits must be an integer in 2..16")


@dataclass(frozen=True)
class QuantConfig:
    weight_bits: int = 8
    activation_bits: int = 8
    symmetric_activations: bool = False
    exclude: tuple[str, ...] = ()
    overrides: Optional[dict[str, tuple[int, int]]] = None

    def __post_init__(self):
        _validate_bits(self.weight_bits)
        _validate_bits(self.activation_bits)
        if not isinstance(self.symmetric_activations, bool):
            raise ValueError("symmetric_activations must be bool")
        if not isinstance(self.exclude, tuple) or not all(isinstance(n, str) for n in self.exclude):
            raise ValueError("exclude must be a tuple of module names")
        if self.overrides is not None:
            if not isinstance(self.overrides, dict):
                raise ValueError("overrides must map module names to bit pairs")
            for name, pair in self.overrides.items():
                if not isinstance(name, str) or not isinstance(pair, tuple) or len(pair) != 2:
                    raise ValueError("overrides must map module names to (weight_bits, activation_bits)")
                for bits in pair:
                    _validate_bits(bits)
            object.__setattr__(self, "overrides", dict(self.overrides))


class FakeQuantizer(nn.Module):
    """Explicit observe -> freeze -> forward lifecycle; no implicit calibration.

    Ranges and quantization parameters are FP32 buffers. Per-channel parameters
    are vectors, broadcast along channel_axis (negative axes are supported).
    Load state into a quantizer constructed with the same configuration.
    """

    def __init__(self, bits=8, symmetric=True, channel_axis=None):
        super().__init__()
        _validate_bits(bits)
        if not isinstance(symmetric, bool):
            raise ValueError("symmetric must be bool")
        if channel_axis is not None and (isinstance(channel_axis, bool) or not isinstance(channel_axis, int)):
            raise ValueError("channel_axis must be an integer or None")
        self.bits = bits
        self.symmetric = symmetric
        self.channel_axis = channel_axis
        self.qmax = (1 << (bits - 1)) - 1 if symmetric else (1 << bits) - 1
        self.qmin = -self.qmax if symmetric else 0
        for name in ("min_val", "max_val", "scale", "zero_point"):
            self.register_buffer(name, torch.empty(0, dtype=torch.float32))
        self.register_buffer("calibrated", torch.tensor(False))

    def _validate_tensor(self, x):
        if not isinstance(x, torch.Tensor) or not x.is_floating_point():
            raise ValueError("quantization requires a floating-point tensor")
        if x.numel() == 0 or not torch.isfinite(x).all():
            raise ValueError("quantization rejects empty or nonfinite tensors")
        if self.channel_axis is not None and not -x.ndim <= self.channel_axis < x.ndim:
            raise ValueError("channel_axis is out of range")

    @torch.no_grad()
    def observe(self, x):
        if self.calibrated.item():
            raise RuntimeError("observer is frozen")
        self._validate_tensor(x)
        values = x.detach().float()
        if not torch.isfinite(values).all():
            raise ValueError("observations must be representable in FP32")
        if self.channel_axis is None:
            low, high = values.amin(), values.amax()
        else:
            values = values.movedim(self.channel_axis, 0).reshape(x.shape[self.channel_axis], -1)
            low, high = values.amin(dim=1), values.amax(dim=1)
        if not self.symmetric:
            low = low.clamp(max=0)
            high = high.clamp(min=0)
        if self.min_val.numel():
            if low.shape != self.min_val.shape:
                raise ValueError("observed channel count changed")
            low = torch.minimum(self.min_val, low)
            high = torch.maximum(self.max_val, high)
        self.min_val, self.max_val = low, high

    @torch.no_grad()
    def freeze(self):
        if not self.min_val.numel():
            raise RuntimeError("cannot freeze without observations")
        low, high = self.min_val.double(), self.max_val.double()
        span = torch.maximum(low.abs(), high.abs()) if self.symmetric else high - low
        scale = span / (self.qmax if self.symmetric else self.qmax - self.qmin)
        scale = torch.where(span == 0, torch.ones_like(scale), scale)
        self.scale = scale.clamp(min=torch.finfo(torch.float32).tiny).float()
        self.zero_point = (torch.zeros_like(self.scale) if self.symmetric else
                           (self.qmin - low / self.scale.double()).round().clamp(self.qmin, self.qmax).float())
        self.calibrated = torch.tensor(True, device=self.scale.device)

    def forward(self, x):
        if not self.calibrated.item():
            raise RuntimeError("quantizer must be observed and frozen before forward")
        self._validate_tensor(x)
        scale, zero = self.scale, self.zero_point
        if self.channel_axis is not None:
            if x.shape[self.channel_axis] != scale.numel():
                raise ValueError("input channel count differs from calibration")
            shape = [1] * x.ndim
            shape[self.channel_axis] = -1
            scale, zero = scale.reshape(shape), zero.reshape(shape)
        quantized = (x.float() / scale + zero).round().clamp(self.qmin, self.qmax)
        return ((quantized - zero) * scale).to(x.dtype)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        for name in ("min_val", "max_val", "scale", "zero_point", "calibrated"):
            value = state_dict.get(prefix + name)
            if value is not None:
                buffer = getattr(self, name)
                setattr(self, name, torch.empty(value.shape, dtype=buffer.dtype, device=buffer.device))
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)


class QuantizedLayer(nn.Module):
    """A copied float layer with boundary and per-output-channel weight quantizers."""

    def __init__(self, layer, weight_bits=8, activation_bits=8, symmetric_activations=False):
        super().__init__()
        self.layer = layer
        self.input_quantizer = FakeQuantizer(activation_bits, symmetric_activations)
        self.output_quantizer = FakeQuantizer(activation_bits, symmetric_activations)
        self.weight_quantizer = FakeQuantizer(weight_bits, True, channel_axis=0)
        self.weight_quantizer.observe(layer.weight)
        self.weight_quantizer.freeze()
        self.enabled = True
        self.training = layer.training

    def forward(self, input):
        if not self.enabled:
            return self.layer(input)
        output = torch.func.functional_call(
            self.layer, {"weight": self.weight_quantizer(self.layer.weight)},
            (self.input_quantizer(input),))
        return self.output_quantizer(output)


def calibrate_simulation(model, calibration_data, config=None, *, weight_encodings=None):
    """Copy and calibrate from an iterable of tuples of positional model arguments.

    Calibration runs in eval mode without quantization or gradients, restoring
    each module's training flag afterwards. Shared modules and shared parameters
    (including storage shared with buffers) are rejected rather than silently
    losing aliases. Sparse and meta parameters/buffers cannot be calibrated and are
    rejected. Already quantized models are rejected. Unexecuted wrappers fail
    calibration. Hooks are always removed; observers are frozen before return.

    weight_encodings optionally maps exact layer names to AdaRound encodings:
    bits, qmin, qmax, channel_axis=0, and positive FP32 scales of shape [out_channels].
    Supplied scales are copied, not recomputed from rounded weights; zero points
    are zero and bounds must be narrow signed. Additional report fields are ignored.
    """
    config = QuantConfig() if config is None else config
    if not isinstance(config, QuantConfig):
        raise TypeError("config must be QuantConfig")
    # Revalidate mutable override mappings before consuming them.
    config = QuantConfig(config.weight_bits, config.activation_bits,
                         config.symmetric_activations, config.exclude, config.overrides)
    seen = {}
    modules = {}
    for name, module in model.named_modules(remove_duplicate=False):
        if isinstance(module, QuantizedLayer):
            raise ValueError(f"already contains QuantizedLayer at {name!r}; double quantization is unsupported")
        if id(module) in seen:
            raise ValueError(f"shared module aliases are unsupported: {seen[id(module)]!r} and {name!r}")
        seen[id(module)] = name
        modules[name] = module
    for name in (*config.exclude, *(config.overrides or {})):
        if name not in modules:
            raise ValueError(f"unknown module name in exclusions/overrides: {name!r}")
    seen = {}
    tensors = list(model.named_parameters(remove_duplicate=False)) + list(model.named_buffers(remove_duplicate=False))
    for name, parameter in tensors:
        if id(parameter) in seen:
            raise ValueError(f"shared parameter aliases are unsupported: {seen[id(parameter)]!r} and {name!r}")
        seen[id(parameter)] = name
        # Sparse and meta/fake tensors carry no usable values or addresses.
        if parameter.numel() and storage_interval(parameter) is None:
            raise ValueError(f"cannot calibrate non-strided or meta parameter/buffer {name!r}; "
                             f"materialize dense tensors on a real device first")
    # Distinct frombuffer storages can overlap despite different base pointers.
    aliased = find_aliased_tensors(model)
    for _, parameter in tensors:
        if id(parameter) in aliased:
            later, earlier = aliased[id(parameter)]
            raise ValueError("shared parameter/buffer storage aliases are unsupported: "
                             f"{earlier!r} and {later!r}")
    supported = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)
    if weight_encodings is not None and not isinstance(weight_encodings, dict):
        raise ValueError("weight_encodings must map layer names to encoding dictionaries")
    scales = {}
    for name, encoding in (weight_encodings or {}).items():
        if name not in modules:
            raise ValueError(f"unknown weight encoding layer: {name!r}")
        if type(modules[name]) not in supported:
            raise ValueError(f"unsupported weight encoding layer: {name!r}")
        if is_excluded(name, config.exclude):
            raise ValueError(f"excluded weight encoding layer: {name!r}")
        bits = (config.overrides or {}).get(name, (config.weight_bits, config.activation_bits))[0]
        qmax = (1 << (bits - 1)) - 1
        expected = {"bits": bits, "qmin": -qmax, "qmax": qmax, "channel_axis": 0}
        if not isinstance(encoding, dict) or any(
            type(encoding.get(field)) is not int or encoding[field] != value
            for field, value in expected.items()
        ):
            raise ValueError(f"mismatched weight encoding for {name!r}: expected {expected}")
        try:
            scale = torch.as_tensor(encoding["scales"], dtype=torch.float32,
                                    device=modules[name].weight.device).detach().clone()
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(f"invalid weight encoding scales for {name!r}") from exc
        if (scale.shape != (modules[name].weight.shape[0],)
                or not torch.isfinite(scale).all() or not (scale > 0).all()):
            raise ValueError(f"weight encoding scales for {name!r} must have shape [out_channels] and be finite positive FP32")
        if "zero_points" in encoding:
            try:
                zero = torch.as_tensor(encoding["zero_points"])
            except (TypeError, ValueError, RuntimeError) as exc:
                raise ValueError(f"invalid weight encoding zero_points for {name!r}") from exc
            if zero.shape != scale.shape or not (zero == 0).all():
                raise ValueError(f"weight encoding zero_points for {name!r} must be zero with shape [out_channels]")
        scales[name] = scale
    result = deepcopy(model)
    wrappers = []

    def wrap(module, name):
        if is_excluded(name, config.exclude):
            return module
        if type(module) in supported:
            wb, ab = (config.overrides or {}).get(name, (config.weight_bits, config.activation_bits))
            wrapped = QuantizedLayer(module, wb, ab, config.symmetric_activations)
            if name in scales:
                wrapped.weight_quantizer.scale = scales[name]
                wrapped.weight_quantizer.zero_point = torch.zeros_like(scales[name])
            wrapped.enabled = False
            wrappers.append(wrapped)
            return wrapped
        for child_name, child in list(module.named_children()):
            module._modules[child_name] = wrap(child, f"{name}.{child_name}" if name else child_name)
        return module

    result = wrap(result, "")
    training = [(module, module.training) for module in result.modules()]
    hooks = []

    def observe(module, args, kwargs, output):
        if args:
            value = args[0]
        elif "input" in kwargs:
            value = kwargs["input"]
        else:
            raise TypeError(
                "calibration requires the quantized layer input as the first positional "
                f"argument or as the keyword 'input'; got keywords {sorted(kwargs)}")
        module.input_quantizer.observe(value)
        module.output_quantizer.observe(output)

    try:
        result.eval()
        for wrapper in wrappers:
            hooks.append(wrapper.register_forward_hook(observe, with_kwargs=True))
        count = 0
        with torch.no_grad():
            for args in calibration_data:
                if not isinstance(args, tuple):
                    raise TypeError("each calibration batch must be a tuple of positional model arguments")
                result(*args)
                count += 1
        if count == 0:
            raise ValueError("calibration_data must contain at least one batch")
        for wrapper in wrappers:
            wrapper.input_quantizer.freeze()
            wrapper.output_quantizer.freeze()
            wrapper.enabled = True
    finally:
        for hook in hooks:
            hook.remove()
        for module, was_training in training:
            module.training = was_training
    return result


def quantization_report(model):
    """JSON-compatible wrapper-name -> weight/input/output parameters.

    Excluded and unsupported modules are omitted, not reported as quantized.
    Per-tensor scales/zero_points are numbers; per-channel values are lists.
    """
    report = {}
    for name, module in model.named_modules():
        if isinstance(module, QuantizedLayer):
            entry = {}
            for kind in ("weight", "input", "output"):
                quantizer = getattr(module, kind + "_quantizer")
                entry[kind] = {
                    "bits": quantizer.bits,
                    "channel_axis": quantizer.channel_axis,
                    "scales": quantizer.scale.detach().cpu().tolist(),
                    "zero_points": quantizer.zero_point.detach().cpu().tolist(),
                    "qmin": quantizer.qmin,
                    "qmax": quantizer.qmax,
                }
            report[name] = entry
    return report
