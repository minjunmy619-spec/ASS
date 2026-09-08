"""Quality-gated orchestration, not a vendor NPU compiler."""

import copy
import math
from dataclasses import dataclass

import torch
from torch import nn

from .evaluation import compare_models
from .export import export_onnx, save_encodings
from .graph import optimize_graph
from .quantization import QuantConfig, calibrate_simulation, quantization_report
from .reconstruction import _batches, _check_model, _device, _tree_copy, adaround, bias_correct


@dataclass
class OptimizationResult:
    model: nn.Module
    simulation: nn.Module
    report: dict

    def export_onnx(self, example_args, path, **kwargs):
        """Export optimized float operators; backend quantization is a later step."""
        return export_onnx(self.model, example_args, path, **kwargs)

    def save_encodings(self, path):
        save_encodings(self.simulation, path, metadata=self.report)


class NPUQuantizer:
    """Optimize independent eval-mode copies of a pretrained model.

    Data items must be tuples of positional model inputs, NOT (inputs, labels).
    Supply a separate representative held-out validation set. The default gate
    minimizes output NMSE against the original floating-point model. An optional
    score_fn(model) can instead compute a higher-is-better task metric using its
    own held-out labeled data. Scores must be finite scalars.
    """

    def __init__(self, model, config=None):
        _check_model(model)
        self.model = copy.deepcopy(model).eval()
        self.config = QuantConfig() if config is None else config
        if not isinstance(self.config, QuantConfig):
            raise TypeError("config must be QuantConfig")

    def optimize(self, calibration_data, validation_data, *, max_batches=16,
                 fold_bn=True, equalize=True, smooth=False,
                 adaround_iterations=0, bias_correction=False, score_fn=None):
        """Run opt-in reconstruction and retain only non-regressing stages.

        Calibration and validation are each bounded to max_batches CPU-cached
        batches. Each candidate is recalibrated. The simulator covers Conv/Linear
        boundaries, not internal RMSNorm/add arithmetic or integer accumulators.
        A passing simulation gate is not a guarantee of NPU accuracy.
        """
        if (isinstance(adaround_iterations, bool) or not isinstance(adaround_iterations, int)
                or adaround_iterations < 0):
            raise ValueError("adaround_iterations must be a nonnegative integer")
        if bias_correction and not adaround_iterations:
            raise ValueError("pipeline bias correction requires AdaRound; use bias_correct directly otherwise")
        calibration = _batches(calibration_data, max_batches)
        validation = _batches(validation_data, max_batches)
        reference = copy.deepcopy(self.model).eval()
        device = _device(reference)

        def calibration_on_device():
            for batch in calibration:
                yield _tree_copy(batch, device)

        def evaluate(model, encodings):
            simulation = calibrate_simulation(
                model, calibration_on_device(), self.config, weight_encodings=encodings).eval()
            if not quantization_report(simulation):
                raise ValueError("no supported quantized layers remain; check model types and exclusions")
            metrics = compare_models(reference, simulation, validation)
            if score_fn is None:
                score = -metrics["nmse"]
            else:
                with torch.no_grad():
                    score = float(score_fn(simulation))
                if not math.isfinite(score):
                    raise ValueError("score_fn must return a finite higher-is-better scalar")
            return simulation, metrics, score

        best = copy.deepcopy(reference)
        encodings = {}
        simulation, baseline, best_score = evaluate(best, encodings)
        report = {"baseline": baseline, "baseline_score": best_score, "stages": [],
                  "calibration_batches": len(calibration), "validation_batches": len(validation),
                  "selection_metric": "negative_output_nmse" if score_fn is None else "custom_score",
                  "scope": "Conv/Linear boundary simulation only; validate the compiled NPU model."}

        def consider(name, candidate, candidate_encodings, operations):
            nonlocal best, encodings, simulation, best_score
            proposed_sim, metrics, score = evaluate(candidate, candidate_encodings)
            if set(quantization_report(proposed_sim)) != set(quantization_report(simulation)):
                raise RuntimeError("candidate changed quantization coverage; refusing an invalid comparison")
            accepted = score >= best_score
            report["stages"].append({"stage": name, "accepted": accepted,
                                     "metrics": metrics, "score": score,
                                     "operations": operations})
            if accepted:
                best, encodings, simulation, best_score = candidate, candidate_encodings, proposed_sim, score
            return accepted

        if fold_bn or equalize or smooth:
            maxima, hooks = {}, []
            if smooth:
                norm_types = (nn.LayerNorm,) + ((nn.RMSNorm,) if hasattr(nn, "RMSNorm") else ())

                def observer(name):
                    def hook(module, args, output):
                        value = output.detach().float()
                        if not torch.isfinite(value).all() or not value.numel():
                            raise ValueError("normalization calibration must be finite and nonempty")
                        current = value.reshape(-1, value.shape[-1]).abs().amax(0)
                        maxima[name] = current if name not in maxima else torch.maximum(maxima[name], current)
                    return hook

                try:
                    for name, module in best.named_modules():
                        if type(module) in norm_types and len(module.normalized_shape) == 1:
                            hooks.append(module.register_forward_hook(observer(name)))
                    with torch.no_grad():
                        for batch in calibration_on_device():
                            best(*batch)
                finally:
                    for hook in hooks:
                        hook.remove()
            candidate, operations = optimize_graph(
                best, _tree_copy(validation[0], device), fold_bn=fold_bn,
                equalize=equalize, smooth=smooth, activation_max=maxima)
            # A graph rewrite must preserve float behavior before its quantized
            # quality is considered. Check held-out batches, not calibration only.
            equivalence = compare_models(best, candidate, validation)
            if equivalence["nmse"] > 1e-10:
                raise RuntimeError("graph rewrite failed held-out floating-point equivalence")
            consider("graph", candidate, {}, operations)

        if adaround_iterations:
            layer_bits = {}
            for name, module in best.named_modules():
                excluded = any(ex == "" or name == ex or name.startswith(ex + ".")
                               for ex in self.config.exclude)
                if type(module) in (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d) and not excluded:
                    layer_bits[name] = (self.config.overrides or {}).get(
                        name, (self.config.weight_bits, self.config.activation_bits))[0]
            rounding_reference = copy.deepcopy(best)
            candidate, operations = adaround(
                best, calibration, iterations=adaround_iterations,
                layer_bits=layer_bits, max_cached_batches=max_batches)
            proposed_encodings = {entry["layer"]: entry for entry in operations}
            accepted = consider("adaround", candidate, proposed_encodings, operations)
            if bias_correction and accepted and proposed_encodings:
                candidate, operations = bias_correct(
                    rounding_reference, best, calibration, layers=tuple(proposed_encodings),
                    max_cached_batches=max_batches)
                consider("bias_correction", candidate, encodings, operations)
            elif bias_correction:
                report["stages"].append({"stage": "bias_correction", "accepted": False,
                                         "reason": "no accepted rounded layers"})

        report["selected_score"] = best_score
        report["selected"] = compare_models(reference, simulation, validation)
        report["weight_encodings"] = encodings
        return OptimizationResult(best, simulation, report)
