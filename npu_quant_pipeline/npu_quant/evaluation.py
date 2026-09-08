"""Evaluation of PyTorch outputs, not a guarantee of deployed backend accuracy."""

import copy
import itertools
import math

import torch
from torch import nn

from .quantization import QuantizedLayer


def _clone(value, device):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device).clone()
    if isinstance(value, tuple):
        values = [_clone(v, device) for v in value]
        return type(value)(*values) if hasattr(value, "_fields") else tuple(values)
    if isinstance(value, list):
        return [_clone(v, device) for v in value]
    if isinstance(value, dict):
        return {k: _clone(v, device) for k, v in value.items()}
    return copy.deepcopy(value)


def _device(model):
    if not isinstance(model, nn.Module):
        raise TypeError("models must be torch.nn.Module instances")
    devices = {t.device for t in itertools.chain(model.parameters(), model.buffers())}
    if len(devices) > 1:
        raise ValueError("each model must be on a single device")
    return next(iter(devices), torch.device("cpu"))


def _pairs(reference, candidate):
    if isinstance(reference, torch.Tensor) and isinstance(candidate, torch.Tensor):
        if reference.shape != candidate.shape:
            raise ValueError("output tensor shapes must match exactly")
        for value in (reference, candidate):
            if value.numel() == 0 or not torch.isfinite(value).all():
                raise ValueError("outputs must be nonempty and finite")
        dtype = torch.complex128 if reference.is_complex() or candidate.is_complex() else torch.float64
        yield reference.to(device="cpu", dtype=dtype), candidate.to(device="cpu", dtype=dtype)
    elif type(reference) is not type(candidate):
        raise ValueError("output structure must match exactly")
    elif isinstance(reference, (tuple, list)):
        if len(reference) != len(candidate):
            raise ValueError("output structure lengths must match")
        for left, right in zip(reference, candidate):
            yield from _pairs(left, right)
    elif isinstance(reference, dict):
        if reference.keys() != candidate.keys():
            raise ValueError("output structure keys must match")
        for key in reference:
            yield from _pairs(reference[key], candidate[key])
    elif reference != candidate:
        raise ValueError("output structure metadata must match")


def compare_models(reference, candidate, data):
    """Compare tuples of positional inputs, weighted by all output elements.

    Models are not copied or moved. Each receives independent recursive input
    clones on its single device (CPU for stateless models). Eval/inference mode
    is temporary; every module's original training flag is restored on failure.
    Custom forwards must not mutate model state. Tensor/list/tuple/dict outputs
    must match exactly in structure and shape; metadata must compare equal.

    CPU double totals define MSE and NMSE (reference mean power floored at
    1e-12). SQNR is clipped to [-120, 120] dB; perfect agreement is 120 dB,
    including two zero outputs. Empty data and tensor-free outputs are errors.
    """
    devices = _device(reference), _device(candidate)
    modes = {m: m.training for model in (reference, candidate) for m in model.modules()}
    squared = power = maximum = 0.0
    count = 0
    try:
        reference.eval()
        candidate.eval()
        with torch.inference_mode():
            for args in data:
                if not isinstance(args, tuple):
                    raise TypeError("each batch must be a tuple of positional model arguments")
                # Snapshot before the next forward, even if outputs alias buffers.
                target = _clone(reference(*_clone(args, devices[0])), "cpu")
                output = candidate(*_clone(args, devices[1]))
                batch_count = 0
                for left, right in _pairs(target, output):
                    error = (left - right).abs()
                    squared += error.square().sum().item()
                    power += left.abs().square().sum().item()
                    maximum = max(maximum, error.max().item())
                    batch_count += left.numel()
                if not batch_count:
                    raise ValueError("each output must contain at least one tensor")
                count += batch_count
    finally:
        for module, training in modes.items():
            module.training = training
    if not count:
        raise ValueError("data must not be empty")
    if not all(math.isfinite(v) for v in (squared, power, maximum)):
        raise ValueError("output totals overflowed double precision")
    mse = squared / count
    nmse = mse / max(power / count, 1e-12)
    sqnr = 120.0 if squared == 0 else (-120.0 if power == 0 else
            max(-120.0, min(120.0, 10 * (math.log10(power) - math.log10(squared)))))
    return {"mse": mse, "nmse": nmse, "max_abs_error": maximum, "sqnr_db": sqnr}


def layer_sensitivity(reference, simulation, data):
    """Rank single-layer bypasses by baseline NMSE minus bypass NMSE.

    Cache input clones on CPU so generators (including reused buffers) can be
    replayed. Existing enabled flags are respected and always restored. A
    disabled layer therefore has zero improvement. Shared reference/simulation
    quantized layers are rejected since toggling would change the reference.
    """
    _device(reference)
    _device(simulation)
    layers = [(name, m) for name, m in simulation.named_modules()
              if isinstance(m, QuantizedLayer)]
    if any(m in set(reference.modules()) for _, m in layers):
        raise ValueError("reference and simulation must not share quantized layers")
    batches = []
    for args in data:
        if not isinstance(args, tuple):
            raise TypeError("each batch must be a tuple of positional model arguments")
        batches.append(_clone(args, "cpu"))
    flags = [(m, m.enabled) for _, m in layers]
    rows = []
    try:
        baseline = compare_models(reference, simulation, batches)["nmse"]
        for name, layer in layers:
            enabled = layer.enabled
            try:
                layer.enabled = False
                nmse = compare_models(reference, simulation, batches)["nmse"]
                rows.append({"layer": name, "nmse": nmse, "improvement": baseline - nmse})
            finally:
                layer.enabled = enabled
    finally:
        for layer, enabled in flags:
            layer.enabled = enabled
    return sorted(rows, key=lambda row: row["improvement"], reverse=True)


def _waveforms(reference, estimate):
    if not isinstance(reference, torch.Tensor) or not isinstance(estimate, torch.Tensor):
        raise TypeError("waveforms must be tensors")
    if reference.shape != estimate.shape or reference.ndim < 1 or reference.numel() == 0:
        raise ValueError("waveforms require exactly equal, nonempty shapes with a time axis")
    for value in (reference, estimate):
        if value.is_complex() or not torch.isfinite(value).all():
            raise ValueError("waveforms must be real and finite")
    reference = reference.to(dtype=torch.float64)
    estimate = estimate.to(device=reference.device, dtype=torch.float64)
    if (reference.abs().amax(dim=-1) == 0).any():
        raise ValueError("silent reference signals are undefined; select nonzero signals explicitly")
    return reference, estimate


def waveform_metrics(reference, estimate, *, eps=1e-8):
    """Per-signal metrics along time only, returned in double on reference device.

    SI-SDR projects onto the reference without mean subtraction. Gain is
    20*log10(estimate RMS / reference RMS), not scale-invariant; negative gain
    detects too-quiet estimates. DC error is mean(estimate - reference).
    eps floors powers in dB ratios, keeping perfect/silent estimates finite.
    Any silent reference signal is rejected rather than silently averaged out.
    """
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    reference, estimate = _waveforms(reference, estimate)
    power = reference.square().mean(dim=-1)
    estimated_power = estimate.square().mean(dim=-1)
    error = estimate - reference
    projection = ((estimate * reference).mean(dim=-1) / power).unsqueeze(-1) * reference

    def db_ratio(numerator, denominator):
        return 10 * (numerator.clamp_min(eps).log10() - denominator.clamp_min(eps).log10())

    return {"snr_db": db_ratio(power, error.square().mean(dim=-1)),
            "si_sdr_db": db_ratio(projection.square().mean(dim=-1),
                                   (estimate - projection).square().mean(dim=-1)),
            "gain_db": db_ratio(estimated_power, power),
            "dc_error": error.mean(dim=-1)}


def bss_metrics(reference, estimate):
    """Optional mir_eval BSS scores for [sources, time], with fixed ordering.

    Returns SDR/SIR/SAR lists in dB. These filter-based scores are not backend
    accuracy guarantees or replacements for gain-sensitive waveform metrics.
    """
    reference, estimate = _waveforms(reference, estimate)
    if reference.ndim != 2:
        raise ValueError("BSS metrics require [sources, time]")
    if (estimate.abs().amax(dim=-1) == 0).any():
        raise ValueError("BSS metrics require nonzero estimated stems")
    try:
        from mir_eval.separation import bss_eval_sources
        arrays = (reference.detach().cpu().numpy(), estimate.detach().cpu().numpy())
    except (ImportError, RuntimeError) as exc:
        raise ImportError("bss_metrics requires optional extras: install mir_eval and numpy") from exc
    sdr, sir, sar, _ = bss_eval_sources(*arrays, compute_permutation=False)
    return {"SDR": sdr.tolist(), "SIR": sir.tolist(), "SAR": sar.tolist()}
