import argparse
import csv
import copy
import os
import shutil

import soundfile as sf
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import initialize_config
from src.tools.estimated_source_matching import match_batch


_CONDITION_KEYS = ("query_condition", "tse_condition", "bridge_condition", "proposal_condition")


def _as_mono_or_multichannel(waveform):
    waveform = waveform.detach().cpu()
    if waveform.dim() == 1:
        return waveform.numpy()
    return waveform.transpose(0, 1).numpy()


def _write_wav(path, waveform, sample_rate, overwrite=False):
    if os.path.exists(path) and not overwrite:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, _as_mono_or_multichannel(waveform), sample_rate)


def _copy_if_available(src, dst, overwrite=False):
    if src is None or not os.path.exists(src):
        return False
    if os.path.exists(dst) and not overwrite:
        return True
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _label_vectors(batch, labels_num, device):
    label_vector = batch["label_vector"].to(device)
    if label_vector.shape[-1] > labels_num:
        label_vector = label_vector[..., :labels_num]
    return label_vector


def _save_reference_files(batch, dataset, output_root, sample_rate, overwrite=False):
    soundscape_dir = os.path.join(output_root, "soundscape")
    oracle_dir = os.path.join(output_root, "oracle_target")

    for i, soundscape in enumerate(batch["soundscape"]):
        src_path = dataset.data[dataset.data_index[soundscape]]["mixture_path"] if hasattr(dataset, "data_index") else None
        dst_path = os.path.join(soundscape_dir, f"{soundscape}.wav")
        if not _copy_if_available(src_path, dst_path, overwrite=overwrite):
            _write_wav(dst_path, batch["mixture"][i], sample_rate, overwrite=overwrite)

        for slot, label in enumerate(batch["label"][i]):
            if label == "silence":
                continue
            oracle_path = os.path.join(oracle_dir, f"{soundscape}_{slot}_{label}.wav")
            _write_wav(oracle_path, batch["dry_sources"][i, slot, 0], sample_rate, overwrite=overwrite)


def _save_estimates(batch, waveforms, labels, output_root, sample_rate, overwrite=False):
    estimate_dir = os.path.join(output_root, "estimate_target")
    waveforms = waveforms.detach().cpu()
    for i, soundscape in enumerate(batch["soundscape"]):
        for slot, label in enumerate(labels[i]):
            if label == "silence":
                continue
            estimate_path = os.path.join(estimate_dir, f"{soundscape}_{slot}_{label}.wav")
            _write_wav(estimate_path, waveforms[i, slot, 0], sample_rate, overwrite=overwrite)


def _write_manifest(rows, path):
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = [
        "soundscape",
        "oracle_slot",
        "estimate_slot",
        "label",
        "metric",
        "match_score",
        "second_best_score",
        "match_margin",
        "energy_db",
        "quality_group",
        "sample_weight",
        "saved",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


def _ensure_soundscape_ids(batch, split, offset):
    if "soundscape" in batch:
        return batch
    batch_size = int(batch["mixture"].shape[0])
    batch = dict(batch)
    batch["soundscape"] = [
        f"{split}_generated_{offset + idx:08d}"
        for idx in range(batch_size)
    ]
    return batch


def _condition_from_batch(batch, device):
    for key in _CONDITION_KEYS:
        if key in batch:
            value = batch[key]
            return value.to(device) if torch.is_tensor(value) else value
    return None


def _save_query_conditions(batch, output, output_root, overwrite=False):
    condition = None
    for key in _CONDITION_KEYS:
        if key in output:
            condition = output[key]
            break
    if condition is None:
        return
    condition = condition.detach().cpu()
    feature_dir = os.path.join(output_root, "uss_bridge_features")
    os.makedirs(feature_dir, exist_ok=True)
    for i, soundscape in enumerate(batch["soundscape"]):
        path = os.path.join(feature_dir, f"{soundscape}.pt")
        if os.path.exists(path) and not overwrite:
            obj = torch.load(path, map_location="cpu")
        else:
            obj = {}
        obj["query_condition"] = condition[i]
        obj["tse_condition"] = condition[i]
        torch.save(obj, path)


def _dataset_config(config, split):
    if "dataset" in config:
        return copy.deepcopy(config["dataset"])
    if "datamodule" in config:
        dm_args = config["datamodule"]["args"]
        key = f"{split}_dataloader"
        if key not in dm_args:
            raise KeyError(f"Config has no datamodule.args.{key}")
        return copy.deepcopy(dm_args[key]["dataset"])
    raise KeyError("Config must contain either dataset or datamodule")


def _model_config(config):
    if "model" in config:
        return copy.deepcopy(config["model"])
    if "lightning_module" in config:
        return copy.deepcopy(config["lightning_module"]["args"]["model"])
    raise KeyError("Config must contain either model or lightning_module")


def _dataset_sample_rate(dataset):
    if hasattr(dataset, "sr"):
        return dataset.sr
    if hasattr(dataset, "base_dataset") and hasattr(dataset.base_dataset, "sr"):
        return dataset.base_dataset.sr
    raise AttributeError("Dataset does not expose sr or base_dataset.sr")


def _load_checkpoint(model, checkpoint_path):
    if not checkpoint_path:
        return
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model_state = model.state_dict()
    if set(state_dict.keys()) != set(model_state.keys()):
        one_model_key = next(iter(model_state.keys()))
        suffix_matches = [key for key in state_dict if isinstance(key, str) and key.endswith(one_model_key)]
        if suffix_matches:
            prefix = suffix_matches[0][:-len(one_model_key)]
            state_dict = {
                key[len(prefix):]: value
                for key, value in state_dict.items()
                if isinstance(key, str) and key.startswith(prefix)
            }
        else:
            for prefix in ("model.", "module.", "net."):
                stripped = {
                    key[len(prefix):]: value
                    for key, value in state_dict.items()
                    if isinstance(key, str) and key.startswith(prefix)
                }
                if stripped and any(key in model_state for key in stripped):
                    state_dict = stripped
                    break
    model.load_state_dict(state_dict)


def _get_uss_model(model):
    return getattr(model, "uss", model)


def _uss_labels(output, labels):
    class_probs = torch.softmax(output["class_logits"], dim=-1)
    _, indices = class_probs.max(dim=-1)
    active_logits = output.get("silence_logits")
    if active_logits is None:
        active = torch.ones_like(indices, dtype=torch.bool)
    else:
        active = active_logits > 0.0
    batch_labels = []
    for sample_indices, sample_active in zip(indices.detach().cpu(), active.detach().cpu()):
        sample_labels = []
        for index, is_active in zip(sample_indices.tolist(), sample_active.tolist()):
            sample_labels.append(labels[int(index)] if is_active else "silence")
        batch_labels.append(sample_labels)
    return batch_labels


def _labels_from_match_rows(rows, n_est):
    labels = ["silence"] * int(n_est)
    for row in rows:
        if row.saved and row.label != "silence" and row.estimate_slot >= 0:
            labels[int(row.estimate_slot)] = row.label
    return labels


def _pit_oracle_labels_with_manifest(
    est_waveforms,
    ref_waveforms,
    ref_labels,
    soundscapes=None,
    match_metric="sa_sdr",
    min_match_score=-10.0,
    min_match_margin=-1.0e9,
    min_energy_db=-60.0,
    clean_match_score=0.0,
    clean_match_margin=2.0,
    uncertain_weight=0.35,
    save_uncertain=False,
    include_unmatched=False,
):
    if soundscapes is None:
        soundscapes = [f"sample_{idx:08d}" for idx in range(len(ref_labels))]
    batch_results = match_batch(
        est_sources=est_waveforms,
        ref_sources=ref_waveforms,
        labels=ref_labels,
        soundscapes=soundscapes,
        metric=match_metric,
        min_match_score=min_match_score,
        min_match_margin=min_match_margin,
        min_energy_db=min_energy_db,
        clean_match_score=clean_match_score,
        clean_match_margin=clean_match_margin,
        uncertain_weight=uncertain_weight,
        save_uncertain=save_uncertain,
        include_unmatched=include_unmatched,
    )
    labels = [
        _labels_from_match_rows(rows, est_waveforms.shape[1])
        for rows in batch_results
    ]
    manifest_rows = [row for rows in batch_results for row in rows]
    return labels, manifest_rows


def _pit_oracle_labels(est_waveforms, ref_waveforms, ref_labels, **kwargs):
    labels, _ = _pit_oracle_labels_with_manifest(est_waveforms, ref_waveforms, ref_labels, **kwargs)
    return labels


def _build_dataset_index(dataset):
    if hasattr(dataset, "data"):
        dataset.data_index = {item["soundscape"]: idx for idx, item in enumerate(dataset.data)}
    if hasattr(dataset, "base_dataset") and hasattr(dataset.base_dataset, "data"):
        dataset.base_dataset.data_index = {
            item["soundscape"]: idx
            for idx, item in enumerate(dataset.base_dataset.data)
        }


def export_cache(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)

    dataset = initialize_config(_dataset_config(config, args.split), reload=True)
    _build_dataset_index(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batchsize,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        num_workers=args.num_workers,
    )

    model = initialize_config(_model_config(config), reload=True)
    _load_checkpoint(model, args.checkpoint)
    model.eval()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model = model.to(device)
    sample_rate = _dataset_sample_rate(dataset)
    dataset_labels = getattr(dataset, "labels", None)
    if dataset_labels is None and hasattr(dataset, "base_dataset"):
        dataset_labels = dataset.base_dataset.labels

    os.makedirs(args.output_root, exist_ok=True)

    next_soundscape_idx = 0
    manifest_rows = []
    for batch in tqdm(dataloader):
        batch = _ensure_soundscape_ids(batch, args.split, next_soundscape_idx)
        next_soundscape_idx += len(batch["soundscape"])
        _save_reference_files(
            batch,
            dataset,
            args.output_root,
            sample_rate,
            overwrite=args.overwrite,
        )

        mixture = batch["mixture"].to(device)
        with torch.no_grad():
            if args.mode == "oracle_tse":
                if not hasattr(model, "_run_tse"):
                    raise TypeError("oracle_tse mode requires a Kwon2025S5-like model with _run_tse")
                enroll = batch["dry_sources"].to(device)
                label_vector = _label_vectors(batch, len(model.labels), device)
                query_condition = _condition_from_batch(batch, device)
                waveforms = model._run_tse(mixture, enroll, label_vector, query_condition)
                labels = batch["label"]
                output = {"waveform": waveforms, "label": labels}
                if query_condition is not None:
                    output["query_condition"] = query_condition
            elif args.mode == "pseudo_s5":
                output = model.predict_label_separate(mixture)
                waveforms = output["waveform"]
                labels = output["label"]
            elif args.mode in {"uss_oracle", "uss_pseudo"}:
                uss = _get_uss_model(model)
                uss_output = uss({"mixture": mixture})
                waveforms = uss_output["foreground_waveform"]
                if args.mode == "uss_oracle":
                    ref_waveforms = batch["dry_sources"][:, :, 0, :].detach().cpu()
                    est_waveforms = waveforms[:, :, 0, :].detach().cpu()
                    labels, rows = _pit_oracle_labels_with_manifest(
                        est_waveforms,
                        ref_waveforms,
                        batch["label"],
                        soundscapes=batch["soundscape"],
                        match_metric=args.match_metric,
                        min_match_score=args.min_match_score,
                        min_match_margin=args.min_match_margin,
                        min_energy_db=args.min_energy_db,
                        clean_match_score=args.clean_match_score,
                        clean_match_margin=args.clean_match_margin,
                        uncertain_weight=args.uncertain_weight,
                        save_uncertain=args.save_uncertain,
                        include_unmatched=args.save_unmatched_manifest,
                    )
                    manifest_rows.extend(rows)
                else:
                    labels = _uss_labels(uss_output, getattr(model, "labels", dataset_labels))
                output = {
                    "waveform": waveforms,
                    "label": labels,
                }
                for key in _CONDITION_KEYS:
                    if key in uss_output:
                        output[key] = uss_output[key]
            else:
                raise ValueError(f"Unknown mode: {args.mode}")

        _save_estimates(
            batch,
            waveforms,
            labels,
            args.output_root,
            sample_rate,
            overwrite=args.overwrite,
        )
        _save_query_conditions(batch, output, args.output_root, overwrite=args.overwrite)
    if args.manifest_path:
        _write_manifest(manifest_rows, args.manifest_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--output_root", default="workspace/sc_finetune")
    parser.add_argument("--mode", choices=["oracle_tse", "pseudo_s5", "uss_oracle", "uss_pseudo"], default="oracle_tse")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--batchsize", "-b", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--manifest_path", default=None, help="Optional CSV path for USS-oracle match quality rows")
    parser.add_argument("--match_metric", default="sa_sdr", choices=["sa_sdr", "si_sdr"], help="Metric used for oracle-estimate matching")
    parser.add_argument("--min_match_score", type=float, default=-10.0, help="Reject oracle matches below this score")
    parser.add_argument("--min_match_margin", type=float, default=-1000000000.0, help="Reject oracle matches with smaller best-vs-second margin")
    parser.add_argument("--min_energy_db", type=float, default=-60.0, help="Reject estimated slots below this RMS dB")
    parser.add_argument("--clean_match_score", type=float, default=0.0, help="Score threshold for clean oracle-label cache samples")
    parser.add_argument("--clean_match_margin", type=float, default=2.0, help="Margin threshold for clean oracle-label cache samples")
    parser.add_argument("--uncertain_weight", type=float, default=0.35, help="Manifest weight assigned to uncertain matches")
    parser.add_argument("--save_uncertain", action="store_true", help="Also save uncertain matches as labeled SC training examples")
    parser.add_argument("--save_unmatched_manifest", action="store_true", help="Record unmatched estimate slots in the manifest")
    args = parser.parse_args()
    export_cache(args)


if __name__ == "__main__":
    main()
