"""Torch-only local weight reconstruction and bias correction.

Calibration data is an iterable of tuples of positional model inputs (including
for single-input models). Returned models are independent, eval-mode copies.
Reports describe local calibration errors, not an end-to-end accuracy guarantee.
"""

import copy
import itertools
import math
from collections.abc import Mapping

import torch
from torch import nn

from . import _utils


_SUPPORTED = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)


def _check_bits(bits):
    if isinstance(bits, bool) or not isinstance(bits, int) or not 2 <= bits <= 16:
        raise ValueError("bits must be an integer in [2, 16]")


def _check_finite(value, context):
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise ValueError(f"nonfinite tensor in {context}")
    elif isinstance(value, (tuple, list)):
        for item in value:
            _check_finite(item, context)
    elif isinstance(value, dict):
        for item in value.values():
            _check_finite(item, context)


def _check_model(model):
    """Reject shared modules and any parameter/buffer identity or storage alias.

    Sparse and meta tensors have no meaningful address range, so they are never
    treated as aliases here; unusable supported weights are rejected later by
    _check_dense_weight instead of raising a RuntimeError from storage access.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    seen = set()
    for _, module in model.named_modules(remove_duplicate=False):
        if id(module) in seen:
            raise ValueError("shared modules are ambiguous for reconstruction")
        seen.add(id(module))
    # Covers parameter/parameter, parameter/buffer, and buffer/buffer overlaps.
    aliased = _utils.find_aliased_tensors(model)
    if aliased:
        first, second = next(iter(aliased.values()))
        raise ValueError("shared parameters or storage aliases are ambiguous: "
                         f"{first!r} and {second!r}")


def _check_dense_weight(name, layer):
    """A supported layer weight must have real, dense storage to be calibrated."""
    weight = layer.weight
    if weight.layout != torch.strided:
        raise ValueError(f"layer {name!r} weight must be dense (strided); "
                         "sparse weights cannot be rounded or corrected")
    if _utils.storage_interval(weight) is None:
        raise ValueError(f"layer {name!r} weight has no materialized storage; "
                         "meta or fake tensors cannot be calibrated")


_tree_copy = _utils.tree_copy


def _batches(calibration_data, limit):
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("max_cached_batches must be a positive integer")
    batches = []
    for batch in itertools.islice(calibration_data, limit):
        if not isinstance(batch, tuple):
            raise TypeError("each calibration batch must be a tuple of positional inputs")
        _check_finite(batch, "calibration inputs")
        batches.append(_tree_copy(batch, "cpu"))
    if not batches:
        raise ValueError("calibration_data must not be empty")
    return batches


def _device(model):
    return _utils.single_device(model, "each model in reconstruction")


def _capture(model, layer, batches, retain_output=True):
    """Clone in the hook so downstream in-place operations cannot alter targets.

    Each captured call is (args, kwargs, output_or_None, output_shape). With
    retain_output=False the output is still validated but never retained, which
    halves the cache footprint for callers that only need the shape.
    """
    captured = []

    def hook(module, args, kwargs, output):
        if not isinstance(output, torch.Tensor):
            raise ValueError("supported layer must return a tensor")
        _check_finite((args, kwargs), "captured inputs")
        _check_finite(output, "captured output/target")
        captured[-1].append((_tree_copy(args, "cpu"),
                             _tree_copy(kwargs, "cpu"),
                             _tree_copy(output, "cpu") if retain_output else None,
                             tuple(output.shape)))

    handle = layer.register_forward_hook(hook, with_kwargs=True)
    try:
        with torch.no_grad():
            for batch in batches:
                captured.append([])
                model(*_tree_copy(batch, _device(model)))
    finally:
        handle.remove()
    return captured


def _samples(reference, candidate, name, batches, capture_outputs=True):
    """Pair reference targets with candidate inputs.

    Returns (args, kwargs, target, output) where output is None when
    capture_outputs is False; shapes are validated either way.
    """
    ref = _capture(reference, reference.get_submodule(name), batches)
    cand = _capture(candidate, candidate.get_submodule(name), batches,
                    retain_output=capture_outputs)
    samples = []
    for ref_calls, cand_calls in zip(ref, cand):
        if len(ref_calls) != len(cand_calls):
            raise ValueError(f"invocation pairing mismatch for layer {name!r}")
        for (_, _, target, _), (args, kwargs, output, shape) in zip(ref_calls, cand_calls):
            if tuple(target.shape) != shape or target.numel() == 0:
                raise ValueError(f"output shape mismatch or empty output for layer {name!r}")
            samples.append((args, kwargs, target, output))
    if not samples:
        raise ValueError(f"layer {name!r} was not invoked by calibration data")
    return samples


def _local_output(layer, weight, args, kwargs):
    parameters = {"weight": weight}
    if layer.bias is not None:
        parameters["bias"] = layer.bias.detach()
    return torch.func.functional_call(layer, parameters, args, kwargs)


def adaround(model, calibration_data, *, bits=8, iterations=200,
             learning_rate=0.01, regularization=0.01, max_cached_batches=16,
             layer_bits=None):
    """Optimize per-channel symmetric narrow-signed rounding, layer by layer.

    The fixed reference is a copy of the supplied, currently optimized model.
    Each report includes baseline_mse (nearest), candidate_mse (hard learned),
    accepted_mse, accepted ("adaround" or "nearest"), and CPU encoding scales.
    The returned floating weights are snapped to that encoding; a backend must
    preserve the reported scales rather than recompute them from snapped weights.
    layer_bits optionally maps exact supported module names to bit widths; only
    those layers are rounded. An empty mapping validates batches but rounds none.
    Nonfinite captured tensors or reconstruction computations raise ValueError.
    """
    _check_bits(bits)
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 0:
        raise ValueError("iterations must be a nonnegative integer")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(regularization) or regularization < 0:
        raise ValueError("regularization must be finite and nonnegative")
    _check_model(model)
    if layer_bits is not None:
        if not isinstance(layer_bits, Mapping):
            raise TypeError("layer_bits must be a mapping of exact layer names to bits")
        modules = dict(model.named_modules())
        layer_bits = dict(layer_bits)
        for name, width in layer_bits.items():
            if not isinstance(name, str) or name not in modules:
                raise ValueError(f"unknown layer name: {name!r}")
            if type(modules[name]) not in _SUPPORTED:
                raise ValueError(f"layer {name!r} is not an exact supported type")
            _check_bits(width)
    batches = _batches(calibration_data, max_cached_batches)
    # Reject unusable weights before copying, since sparse/meta tensors are not
    # copyable and must not surface as RuntimeError.
    for name, layer in model.named_modules():
        if type(layer) in _SUPPORTED and (layer_bits is None or name in layer_bits):
            _check_dense_weight(name, layer)
    reference, candidate = copy.deepcopy(model).eval(), copy.deepcopy(model).eval()
    report = []
    for name, layer in candidate.named_modules():
        if type(layer) not in _SUPPORTED:
            continue
        if layer_bits is not None and name not in layer_bits:
            continue
        width = bits if layer_bits is None else layer_bits[name]
        qmax = 2 ** (width - 1) - 1
        _check_dense_weight(name, layer)
        # Candidate outputs are unused here, so they are validated but not cached.
        samples = _samples(reference, candidate, name, batches, capture_outputs=False)
        weight = layer.weight.detach()
        if not weight.is_floating_point() or not torch.isfinite(weight).all():
            raise ValueError("weights must be finite floating point tensors")
        # Use at least float32 for the encoding and rounding optimization.
        work = weight.to(torch.float64 if weight.dtype == torch.float64 else torch.float32)
        axes = tuple(range(1, work.ndim))
        span = work.abs().amax(dim=axes, keepdim=True)
        scale = torch.where(span == 0, torch.ones_like(span),
                            (span / qmax).clamp_min(torch.finfo(work.dtype).tiny))
        normalized = work / scale
        floor = normalized.floor()
        fraction = normalized - floor
        alpha = nn.Parameter(torch.logit((fraction + 0.1) / 1.2))
        optimizer = torch.optim.Adam([alpha], lr=learning_rate)
        count = sum(target.numel() for _, _, target, _ in samples)
        max_gradient = 0.0
        # Hoisted once per layer: functional_call on Linear/Conv never mutates
        # its inputs, so the same device copies are reused by every iteration.
        local = [(_tree_copy(args, weight.device),
                  _tree_copy(kwargs, weight.device),
                  target.to(device=weight.device, dtype=work.dtype))
                 for args, kwargs, target, _ in samples]

        def snapped(rounding):
            return ((floor + rounding).clamp(-qmax, qmax) * scale).to(weight.dtype)

        def error(quantized, backward=False):
            total = 0.0
            for args, kwargs, target in local:
                output = _local_output(layer, quantized, args, kwargs)
                loss = (output.to(work.dtype) - target).square().sum() / count
                _check_finite(loss, f"reconstruction loss for layer {name!r}")
                if backward:
                    # Each sample shares the rounding graph, not an activation graph.
                    loss.backward(retain_graph=True)
                total += loss.detach().item()
            if not math.isfinite(total):
                raise ValueError(f"nonfinite reconstruction error for layer {name!r}")
            return total

        for step in range(iterations):
            optimizer.zero_grad()
            soft = (alpha.sigmoid() * 1.2 - 0.1).clamp(0, 1)
            error(snapped(soft), backward=True)
            warmup = iterations * 0.2
            if step >= warmup and regularization:
                progress = (step - warmup) / max(iterations - 1 - warmup, 1)
                beta = 20 + (2 - 20) * progress
                penalty = regularization * (1 - (2 * soft - 1).abs().pow(beta)).mean()
                _check_finite(penalty, "rounding regularization")
                penalty.backward()
            if alpha.grad is None:
                raise ValueError(f"no rounding gradient for layer {name!r}; "
                                 "the layer output does not depend on its weight")
            _check_finite(alpha.grad, "rounding gradient")
            max_gradient = max(max_gradient, alpha.grad.detach().abs().max().item())
            optimizer.step()
            _check_finite(alpha, "rounding parameters")

        with torch.no_grad():
            nearest = (normalized.round().clamp(-qmax, qmax) * scale).to(weight.dtype)
            hard = snapped((alpha >= 0).to(work.dtype))
            baseline, learned = error(nearest), error(hard)
            accept = learned <= baseline
            layer.weight.copy_(hard if accept else nearest)
        report.append({"layer": name, "op": "adaround", "bits": width,
                       "qmin": -qmax, "qmax": qmax, "channel_axis": 0,
                       "scales": scale.flatten().detach().cpu().clone(),
                       "encoding_warning": "Backend must preserve these scales; do not recompute.",
                       "baseline_mse": baseline, "candidate_mse": learned,
                       "accepted_mse": learned if accept else baseline,
                       "accepted": "adaround" if accept else "nearest",
                       "max_abs_alpha_gradient": max_gradient,
                       "local_invocations": len(samples)})
    return candidate, report


def bias_correct(reference, candidate, calibration_data, *, max_cached_batches=16,
                 layers=None):
    """Correct matching exact Linear/Conv layer biases using channel residuals.

    Each full-model pass observes reference and candidate at their respective
    inputs. Layers are corrected sequentially; weights are never changed.
    Reports contain CPU correction tensors and element counts per channel.
    layers optionally restricts correction to a tuple of exact names, each of
    which must have the same supported type in both models. Nonfinite captured
    tensors or correction computations raise ValueError.
    """
    _check_model(reference)
    _check_model(candidate)
    if layers is not None:
        if not isinstance(layers, tuple):
            raise TypeError("layers must be a tuple of exact layer names")
        ref_modules, cand_modules = dict(reference.named_modules()), dict(candidate.named_modules())
        for name in layers:
            if not isinstance(name, str) or name not in ref_modules or name not in cand_modules:
                raise ValueError(f"unknown layer name: {name!r}")
            if (type(cand_modules[name]) not in _SUPPORTED or
                    type(ref_modules[name]) is not type(cand_modules[name])):
                raise ValueError(f"layer {name!r} must have matching exact supported types")
        if len(set(layers)) != len(layers):
            raise ValueError("layers must not contain duplicate names")
    batches = _batches(calibration_data, max_cached_batches)
    originals = dict(reference.named_modules())
    for name, layer in candidate.named_modules():
        if layers is not None and name not in layers:
            continue
        if type(layer) in _SUPPORTED and type(originals.get(name)) is type(layer):
            _check_dense_weight(name, layer)
            _check_dense_weight(name, originals[name])
    reference, candidate = copy.deepcopy(reference).eval(), copy.deepcopy(candidate).eval()
    reference_layers = dict(reference.named_modules())
    report = []
    for name, layer in candidate.named_modules():
        if layers is not None and name not in layers:
            continue
        if type(layer) not in _SUPPORTED or name not in reference_layers:
            continue
        if type(reference_layers[name]) is not type(layer):
            continue
        _check_dense_weight(name, layer)
        _check_dense_weight(name, reference_layers[name])
        samples = _samples(reference, candidate, name, batches)
        channel_sum, count, squared = None, 0, 0.0
        for _, _, target, output in samples:
            axis = output.ndim - 1 if type(layer) is nn.Linear else 1
            if type(layer) is not nn.Linear and output.ndim != layer.weight.ndim:
                raise ValueError("bias correction requires batched convolution inputs")
            channels = layer.out_features if type(layer) is nn.Linear else layer.out_channels
            if output.shape[axis] != channels:
                raise ValueError(f"channel shape mismatch for layer {name!r}")
            residual = (target.to(torch.float64) - output.to(torch.float64)).movedim(axis, 0)
            residual = residual.reshape(channels, -1)
            value = residual.sum(dim=1)
            channel_sum = value if channel_sum is None else channel_sum + value
            count += residual.shape[1]
            squared += residual.square().sum().item()
        correction = (channel_sum / count).to(device=layer.weight.device, dtype=layer.weight.dtype)
        _check_finite(correction, f"bias correction for layer {name!r}")
        baseline = squared / (count * correction.numel())
        if not math.isfinite(baseline):
            raise ValueError(f"nonfinite bias correction error for layer {name!r}")
        with torch.no_grad():
            if layer.bias is None:
                layer.bias = nn.Parameter(correction.clone(), requires_grad=layer.weight.requires_grad)
            else:
                layer.bias.add_(correction)
            _check_finite(layer.bias, f"corrected bias for layer {name!r}")
        report.append({"layer": name, "op": "bias_correct",
                       "correction": correction.detach().cpu().clone(),
                       "elements_per_channel": count,
                       "baseline_mse": baseline,
                       "local_invocations": len(samples)})
    return candidate, report
