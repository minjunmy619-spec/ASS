import argparse
import copy
import json
import os
from itertools import combinations, permutations

import soundfile as sf
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import initialize_config
from .metrics import label_metric, s5capi_metric
from .metrics.s5_validation_breakdown import S5ValidationBreakdownMetric


_TSE_CONDITION_KEYS = (
    "query_condition",
    "tse_condition",
    "bridge_condition",
    "proposal_condition",
    "temporal_conditioning",
)


def _load_yaml(path, base_dir=None):
    candidate = path
    if not os.path.isabs(candidate) and not os.path.exists(candidate) and base_dir:
        candidate = os.path.join(base_dir, candidate)
    with open(candidate) as f:
        return yaml.safe_load(f), candidate


def _get_by_path(obj, dotted_path):
    cur = obj
    for part in dotted_path.split("."):
        if isinstance(cur, dict):
            cur = cur[part]
        elif isinstance(cur, list):
            cur = cur[int(part)]
        else:
            raise KeyError(f"Cannot descend into {type(cur)} at '{part}' for path '{dotted_path}'")
    return cur


def _deep_update(base, updates):
    base = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _resolve_config_references(obj, base_dir=None):
    if isinstance(obj, list):
        return [_resolve_config_references(x, base_dir=base_dir) for x in obj]
    if not isinstance(obj, dict):
        return obj
    ref_path = obj.get("from_training_config", obj.get("config_ref", None))
    if ref_path is not None:
        key = obj.get("key", "lightning_module.args.model")
        ref_cfg, resolved_path = _load_yaml(ref_path, base_dir=base_dir)
        value = copy.deepcopy(_get_by_path(ref_cfg, key))
        value = _resolve_config_references(value, base_dir=os.path.dirname(resolved_path))
        overrides = obj.get("overrides", obj.get("override", None))
        if overrides:
            if not isinstance(value, dict) or not isinstance(overrides, dict):
                raise TypeError("override(s) can only be applied when both referenced value and override are dictionaries")
            value = _deep_update(value, overrides)
        return value
    return {k: _resolve_config_references(v, base_dir=base_dir) for k, v in obj.items()}


def _as_list(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _resolve_existing_path(path, config_dir):
    if path is None:
        return None
    if os.path.exists(path):
        return path
    candidate = os.path.join(config_dir, path)
    if os.path.exists(candidate):
        return candidate
    return path


def _select_model_state_dict(state_dict, model_state, preferred_prefixes=()):
    if set(state_dict.keys()) == set(model_state.keys()):
        return state_dict
    for prefix in preferred_prefixes:
        stripped = {
            k[len(prefix):]: v
            for k, v in state_dict.items()
            if isinstance(k, str) and k.startswith(prefix)
        }
        if stripped and any(k in model_state for k in stripped):
            return stripped
    one_model_key = next(iter(model_state.keys()))
    suffix_matches = [k for k in state_dict if isinstance(k, str) and k.endswith(one_model_key)]
    if suffix_matches:
        prefix = suffix_matches[0][:-len(one_model_key)]
        return {k[len(prefix):]: v for k, v in state_dict.items() if isinstance(k, str) and k.startswith(prefix)}
    for prefix in ("model.", "module.", "net."):
        stripped = {k[len(prefix):]: v for k, v in state_dict.items() if isinstance(k, str) and k.startswith(prefix)}
        if stripped and any(k in model_state for k in stripped):
            return stripped
    return state_dict


def _stage_checkpoint_prefixes(stage):
    return {
        "uss": ("uss_model.", "model.", "module.", "net."),
        "sc": ("sc_model.", "model.", "module.", "net."),
        "tse": ("tse_model.", "model.", "module.", "net."),
    }.get(stage, ("model.", "module.", "net."))


def _load_checkpoint(model, checkpoint_path, preferred_prefixes=()):
    if checkpoint_path is None:
        return
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model_state = model.state_dict()
    state_dict = _select_model_state_dict(state_dict, model_state, preferred_prefixes=preferred_prefixes)
    model.load_state_dict(state_dict)


def _stage_model_config(config, stage):
    if "model" in config and f"{stage}_config" in config["model"].get("args", {}):
        args = config["model"]["args"]
        return copy.deepcopy(args[f"{stage}_config"]), args.get(f"{stage}_ckpt")
    if "lightning_module" in config:
        return copy.deepcopy(config["lightning_module"]["args"]["model"]), None
    if "model" in config:
        return copy.deepcopy(config["model"]), None
    raise KeyError("Could not find a model config")


def _dataset_config(config, split):
    if "dataset" in config:
        return copy.deepcopy(config["dataset"])
    if "datamodule" in config:
        dm_args = config["datamodule"]["args"]
        key = f"{split}_dataloader"
        if key not in dm_args:
            raise KeyError(f"Training config has no datamodule.args.{key}")
        return copy.deepcopy(dm_args[key]["dataset"])
    raise KeyError("Could not find a dataset config")


def _label_from_vectors(label_vectors, labels):
    batch_labels = []
    for sample_vectors in label_vectors.detach().cpu():
        sample_labels = []
        for vector in sample_vectors:
            if float(vector.abs().sum()) == 0.0:
                sample_labels.append("silence")
            else:
                sample_labels.append(labels[int(torch.argmax(vector).item())])
        batch_labels.append(sample_labels)
    return batch_labels


def _labels_from_indices(indices, labels):
    return [[labels[int(idx)]] for idx in indices.detach().cpu().tolist()]


def _uss_labels(output, labels):
    class_probs = torch.softmax(output["class_logits"], dim=-1)
    probabilities, indices = class_probs.max(dim=-1)
    active_logits = output.get("silence_logits")
    if active_logits is None:
        active = torch.ones_like(indices, dtype=torch.bool)
        active_probs = torch.ones_like(probabilities)
    else:
        active_probs = torch.sigmoid(active_logits)
        active = active_logits > 0.0
    batch_labels = []
    for sample_indices, sample_active in zip(indices.detach().cpu(), active.detach().cpu()):
        sample_labels = []
        for idx, is_active in zip(sample_indices.tolist(), sample_active.tolist()):
            sample_labels.append(labels[int(idx)] if is_active else "silence")
        batch_labels.append(sample_labels)
    return batch_labels, probabilities * active_probs


def _pairwise_sdr(est_waveforms, ref_waveforms, eps=1e-8):
    est = est_waveforms.float()
    ref = ref_waveforms.float()
    noise = est[:, None, :] - ref[None, :, :]
    signal_power = ref.pow(2).sum(dim=-1).clamp_min(eps)
    noise_power = noise.pow(2).sum(dim=-1).clamp_min(eps)
    return 10.0 * torch.log10(signal_power[None, :] / noise_power)


def _pit_oracle_labels_for_sample(est_waveforms, ref_waveforms, ref_labels):
    n_est = int(est_waveforms.shape[0])
    est_labels = ["silence"] * n_est
    probabilities = torch.zeros(n_est, dtype=torch.float32)
    active_ref_indices = [idx for idx, label in enumerate(ref_labels) if label != "silence"]
    if n_est == 0 or len(active_ref_indices) == 0:
        return est_labels, probabilities
    n_match = min(n_est, len(active_ref_indices))
    active_ref_waveforms = ref_waveforms[active_ref_indices]
    pair_scores = _pairwise_sdr(est_waveforms, active_ref_waveforms)
    best_score = None
    best_assignment = None
    for ref_subset in combinations(range(len(active_ref_indices)), n_match):
        ref_subset = list(ref_subset)
        for est_perm in permutations(range(n_est), n_match):
            scores = [pair_scores[est_idx, ref_idx] for est_idx, ref_idx in zip(est_perm, ref_subset)]
            score = torch.stack(scores).mean()
            if best_score is None or score > best_score:
                best_score = score
                best_assignment = list(zip(est_perm, ref_subset))
    if best_assignment is None:
        return est_labels, probabilities
    for est_idx, ref_local_idx in best_assignment:
        ref_idx = active_ref_indices[ref_local_idx]
        est_labels[int(est_idx)] = ref_labels[int(ref_idx)]
        probabilities[int(est_idx)] = 1.0
    return est_labels, probabilities


def _pit_oracle_labels(est_waveforms, ref_waveforms, ref_labels):
    batch_labels = []
    batch_probabilities = []
    for sample_est, sample_ref, sample_labels in zip(est_waveforms, ref_waveforms, ref_labels):
        labels, probabilities = _pit_oracle_labels_for_sample(sample_est, sample_ref, sample_labels)
        batch_labels.append(labels)
        batch_probabilities.append(probabilities)
    return batch_labels, torch.stack(batch_probabilities, dim=0)


def _classification_counts(est_labels, ref_labels):
    total = 0
    correct = 0
    active_total = 0
    active_correct = 0
    silence_total = 0
    silence_correct = 0
    tp = 0
    fp = 0
    fn = 0
    for sample_est, sample_ref in zip(est_labels, ref_labels):
        for est, ref in zip(sample_est, sample_ref):
            total += 1
            correct += int(est == ref)
            if ref == "silence":
                silence_total += 1
                silence_correct += int(est == ref)
            else:
                active_total += 1
                active_correct += int(est == ref)
            if est == ref and ref != "silence":
                tp += 1
            else:
                fp += int(est != "silence")
                fn += int(ref != "silence")
    precision = 100.0 * tp / max(tp + fp, 1)
    recall = 100.0 * tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "source_top1_accuracy": 100.0 * correct / max(total, 1),
        "active_source_top1_accuracy": 100.0 * active_correct / max(active_total, 1),
        "silence_accuracy": 100.0 * silence_correct / max(silence_total, 1),
        "source_precision": precision,
        "source_recall": recall,
        "source_f1": f1,
        "num_sources": total,
        "num_active_sources": active_total,
        "num_silence_sources": silence_total,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def _compute_metric(metric_func, is_print=True):
    summary = metric_func.compute(is_print=is_print)
    if summary is not None:
        return summary
    values = [v for v in getattr(metric_func, "metric_values", []) if v is not None]
    if values:
        return {"mean": sum(values) / len(values)}
    return None


class StageEvaluator:
    def __init__(
        self,
        config_path,
        stage,
        checkpoint=None,
        split="val",
        waveform_output_dir="",
        result_dir="",
        batch_size=2,
        use_cpu=False,
        num_workers=None,
        uss_oracle_labels=False,
        sc_prediction_mode="raw",
        compare_assignment=False,
        validation_breakdown=False,
    ):
        self.config_path = config_path
        self.config_dir = os.path.dirname(os.path.abspath(config_path))
        raw_config, resolved_config_path = _load_yaml(config_path)
        self.config = _resolve_config_references(raw_config, base_dir=os.path.dirname(os.path.abspath(resolved_config_path)))
        self.stage = stage
        self.filename = os.path.basename(config_path)[:-5]
        self.batch_size = batch_size
        self.result_dir = result_dir
        self.uss_oracle_labels = bool(uss_oracle_labels)
        self.sc_prediction_mode = sc_prediction_mode
        self.compare_assignment = bool(compare_assignment)
        self.validation_breakdown = bool(validation_breakdown)
        self.waveform_output_dir = os.path.join(waveform_output_dir, self.filename, stage) if waveform_output_dir else waveform_output_dir
        self.device = torch.device("cpu" if use_cpu or not torch.cuda.is_available() else "cuda")
        if self.waveform_output_dir:
            os.makedirs(self.waveform_output_dir, exist_ok=True)

        dataset_config = _dataset_config(self.config, split)
        self.dataset = initialize_config(dataset_config, reload=True)
        self.labels = getattr(self.dataset, "labels", None)
        if self.labels is None and hasattr(self.dataset, "base_dataset"):
            self.labels = self.dataset.base_dataset.labels
        self.sr = getattr(self.dataset, "sr", None)
        if self.sr is None and hasattr(self.dataset, "base_dataset"):
            self.sr = self.dataset.base_dataset.sr

        workers = batch_size * 2 if num_workers is None else num_workers
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=getattr(self.dataset, "collate_fn", None),
            num_workers=workers,
        )

        model_config, config_checkpoint = _stage_model_config(self.config, stage)
        checkpoint = checkpoint or config_checkpoint
        checkpoint = _resolve_existing_path(checkpoint, self.config_dir)
        self.model = initialize_config(model_config, reload=True)
        _load_checkpoint(self.model, checkpoint, preferred_prefixes=_stage_checkpoint_prefixes(stage))
        self.model.eval()
        self.model.to(self.device)
        self.tse_conditioning_configured = bool(
            getattr(self.model, "query_conditioner", None) is not None
            or getattr(self.model, "bridge_to_label", None) is not None
        )
        self.tse_condition_keys_seen = set()

    def _write_waveforms(self, batch, labels, waveforms):
        if not self.waveform_output_dir:
            return
        for sample_labels, sample_waveforms, soundscape_name in zip(labels, waveforms, batch["soundscape"]):
            for source_idx, (label, waveform) in enumerate(zip(sample_labels, sample_waveforms)):
                if label == "silence":
                    continue
                wavpath = os.path.join(self.waveform_output_dir, f"{soundscape_name}_{source_idx}_{label}.wav")
                sf.write(wavpath, waveform.detach().cpu().numpy(), self.sr)

    def _evaluate_uss_batch(self, batch):
        mixture = batch["mixture"].to(self.device)
        with torch.no_grad():
            output = self.model({"mixture": mixture})
        est_waveforms = output["foreground_waveform"][:, :, 0, :].detach().cpu()
        if self.uss_oracle_labels:
            ref_waveforms = batch["dry_sources"][:, :, 0, :].detach().cpu()
            est_labels, probabilities = _pit_oracle_labels(est_waveforms, ref_waveforms, batch["label"])
        else:
            est_labels, probabilities = _uss_labels(output, self.labels)
            probabilities = probabilities.detach().cpu()
        return est_labels, est_waveforms, probabilities, batch["label"]

    def _evaluate_sc_batch(self, batch):
        if "dry_sources" in batch:
            ref_waveforms = batch["dry_sources"][:, :, 0, :]
            batch_size, n_sources, samples = ref_waveforms.shape
            flat = ref_waveforms.reshape(batch_size * n_sources, samples)
            ref_labels = batch["label"]
        else:
            flat = batch["waveform"]
            if flat.dim() == 3 and flat.shape[1] == 1:
                flat = flat[:, 0]
            batch_size, n_sources = flat.shape[0], 1
            ref_labels = []
            for class_index, is_silence in zip(batch["class_index"].tolist(), batch["is_silence"].tolist()):
                ref_labels.append(["silence" if is_silence else self.labels[int(class_index)]])

        flat = flat.to(self.device)
        with torch.no_grad():
            output = self.model.predict({"waveform": flat})

        raw_label_vector = output.get("raw_label_vector", output["label_vector"])
        gated_label_vector = output["label_vector"]
        if self.sc_prediction_mode == "raw":
            selected_vector = raw_label_vector
        elif self.sc_prediction_mode == "gated":
            selected_vector = gated_label_vector
        else:
            raise ValueError(f"Unsupported sc_prediction_mode: {self.sc_prediction_mode}")

        label_vector = selected_vector.view(batch_size, n_sources, -1)
        probabilities = output["probabilities"].view(batch_size, n_sources)
        est_labels = _label_from_vectors(label_vector, self.labels)
        return est_labels, None, probabilities.detach().cpu(), ref_labels

    def _evaluate_tse_batch(self, batch):
        mixture = batch["mixture"].to(self.device)
        enrollment = batch["enrollment"] if "enrollment" in batch else batch["dry_sources"]
        enrollment = enrollment.to(self.device)
        label_vector = batch["label_vector"].to(self.device)
        label_dim = getattr(self.model, "label_dim", None)
        if label_dim is not None and label_vector.shape[-1] > label_dim:
            label_vector = label_vector[..., :label_dim]
        input_dict = {
            "mixture": mixture,
            "enrollment": enrollment,
            "label_vector": label_vector,
        }
        for key in _TSE_CONDITION_KEYS:
            if key in batch:
                input_dict[key] = batch[key].to(self.device) if torch.is_tensor(batch[key]) else batch[key]
                self.tse_condition_keys_seen.add(key)
        with torch.no_grad():
            output = self.model(input_dict)
        est_waveforms = output["waveform"][:, :, 0, :].detach().cpu()
        probabilities = torch.ones(est_waveforms.shape[:2], dtype=torch.float32)
        return batch["label"], est_waveforms, probabilities, batch["label"]

    def evaluate(self):
        results = []
        label_metrics = label_metric.LabelMetric()
        separation_metrics = [s5capi_metric.S5ClassAwareMetric(metricfunc="sdr")]
        if self.compare_assignment:
            separation_metrics.append(s5capi_metric.S5ClassAwareMetricAssignmentComparison(metricfunc="sdr"))
        if self.validation_breakdown:
            separation_metrics.append(S5ValidationBreakdownMetric(metricfunc="sdr", prefix="valid"))
        metric_funcs = [label_metrics] if self.stage == "sc" else separation_metrics + [label_metrics]
        for metric_func in metric_funcs:
            metric_func.reset()

        all_est_labels = []
        all_ref_labels = []
        for batch in tqdm(self.dataloader):
            if self.stage == "uss":
                est_labels, est_waveforms, probabilities, ref_labels = self._evaluate_uss_batch(batch)
            elif self.stage == "sc":
                est_labels, est_waveforms, probabilities, ref_labels = self._evaluate_sc_batch(batch)
            elif self.stage == "tse":
                est_labels, est_waveforms, probabilities, ref_labels = self._evaluate_tse_batch(batch)
            else:
                raise ValueError(f"Unsupported stage: {self.stage}")

            all_est_labels.extend(est_labels)
            all_ref_labels.extend(ref_labels)
            ref_waveforms = batch.get("dry_sources")
            mixture = batch.get("mixture")
            metric_values = []
            for metric_func in metric_funcs:
                kwargs = {"batch_est_labels": est_labels, "batch_ref_labels": ref_labels}
                if metric_func in separation_metrics:
                    kwargs.update({
                        "batch_est_waveforms": est_waveforms,
                        "batch_ref_waveforms": ref_waveforms[:, :, 0, :],
                        "batch_mixture": mixture[:, 0, :],
                    })
                metric_values.append(metric_func.update(**kwargs))

            if est_waveforms is not None:
                self._write_waveforms(batch, est_labels, est_waveforms)

            if self.result_dir:
                soundscapes = batch.get("soundscape", [None] * len(est_labels))
                for i, soundscape in enumerate(soundscapes):
                    reobj = {
                        "soundscape": soundscape,
                        "stage": self.stage,
                        "sc_prediction_mode": self.sc_prediction_mode if self.stage == "sc" else None,
                        "oracle_labels": bool(self.uss_oracle_labels and self.stage == "uss"),
                        "ref_labels": ref_labels[i],
                        "est_labels": est_labels[i],
                        "probabilities": _as_list(probabilities[i]),
                        "metrics": [],
                    }
                    for mval, mfunc in zip(metric_values, metric_funcs):
                        reobj["metrics"].append({"metric": getattr(mfunc, "metric_name", None), "value": mval[i]})
                    results.append(reobj)

        summary = {}
        for metric_func in metric_funcs:
            summary[getattr(metric_func, "metric_name", metric_func.__class__.__name__)] = _compute_metric(metric_func, is_print=True)
        if self.stage == "sc":
            summary["sc_prediction_mode"] = self.sc_prediction_mode
            summary["source_classification"] = _classification_counts(all_est_labels, all_ref_labels)
            print(f"SC prediction mode: {self.sc_prediction_mode}")
            print("Source top-1 accuracy: %.3f" % summary["source_classification"]["source_top1_accuracy"])
            print("Active source top-1 accuracy: %.3f" % summary["source_classification"]["active_source_top1_accuracy"])
            print("Silence accuracy: %.3f" % summary["source_classification"]["silence_accuracy"])
            print("Source F1: %.3f" % summary["source_classification"]["source_f1"])
        if self.stage == "tse":
            summary["tse_conditioning_configured"] = self.tse_conditioning_configured
            summary["tse_condition_keys_seen"] = sorted(self.tse_condition_keys_seen)
            if self.tse_conditioning_configured and not self.tse_condition_keys_seen:
                print(
                    "WARNING: TSE has query conditioning layers, but this stage dataset "
                    "did not provide query/bridge/proposal conditions. Full S5 evaluation "
                    "is required to test live USS-conditioned handoff."
                )

        if self.result_dir:
            os.makedirs(self.result_dir, exist_ok=True)
            stem = f"{self.filename}_{self.stage}"
            if self.stage == "sc":
                stem = f"{stem}_{self.sc_prediction_mode}"
            if self.uss_oracle_labels and self.stage == "uss":
                stem = f"{stem}_oracle_labels"
            with open(os.path.join(self.result_dir, f"{stem}_results.json"), "w") as outfile:
                json.dump(results, outfile, indent=4)
            with open(os.path.join(self.result_dir, f"{stem}_summary.json"), "w") as outfile:
                json.dump(summary, outfile, indent=4)


def main(args):
    evaluator = StageEvaluator(
        args.config,
        args.stage,
        checkpoint=args.checkpoint,
        split=args.split,
        waveform_output_dir=args.waveform_output_dir,
        result_dir=args.result_dir,
        batch_size=args.batchsize,
        use_cpu=args.cpu,
        num_workers=args.num_workers,
        uss_oracle_labels=args.uss_oracle_labels,
        sc_prediction_mode=args.sc_prediction_mode,
        compare_assignment=args.compare_assignment,
        validation_breakdown=args.validation_breakdown,
    )
    evaluator.evaluate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True)
    parser.add_argument("--stage", choices=["uss", "sc", "tse"], required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--waveform_output_dir", type=str, default="")
    parser.add_argument("--result_dir", type=str, default="")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--batchsize", "-b", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--sc_prediction_mode", choices=["raw", "gated"], default="gated", help="For --stage sc: gated matches final S5 predict() energy/silence thresholds; raw reports plain-logit top-1 diagnostics.")
    parser.add_argument("--compare_assignment", action="store_true", help="For --stage uss/tse: also log official raw-SDR assignment vs paper SDRi-assignment CAPI-SDRi diagnostics.")
    parser.add_argument("--validation_breakdown", action="store_true", help="For --stage uss/tse: also log CAPI-SDRi, zero-target FP, silence, leakage, and scene-bucket diagnostics.")
    parser.add_argument(
        "--uss_oracle_labels",
        action="store_true",
        help="For --stage uss only: assign oracle class labels to estimated USS slots after waveform-level PIT against reference dry sources.",
    )
    args = parser.parse_args()
    print("START")
    main(args)
