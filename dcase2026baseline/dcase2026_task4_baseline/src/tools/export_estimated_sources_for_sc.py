from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import soundfile as sf
except Exception:  # pragma: no cover
    sf = None

from scipy.io import wavfile

from src.datamodules.dataset import DatasetS3
from src.tools.estimated_source_matching import MatchResult, match_batch
from src.utils import initialize_config, parse_yaml


def _load_model(config_path: str, checkpoint_path: str, device: torch.device):
    cfg = parse_yaml(config_path)
    if "lightning_module" in cfg:
        model_cfg = cfg["lightning_module"]["args"]["model"]
    elif "model" in cfg:
        model_cfg = cfg["model"]
    else:
        raise KeyError("Config must contain either lightning_module.args.model or model")
    model = initialize_config(model_cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    clean_state = {}
    for key, value in state.items():
        if key.startswith("model."):
            key = key[len("model.") :]
        clean_state[key] = value
    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    if missing:
        print(f"[export_estimated_sources_for_sc] missing model keys: {len(missing)}")
    if unexpected:
        print(f"[export_estimated_sources_for_sc] unexpected model keys: {len(unexpected)}")
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint/config mismatch while exporting estimated sources: "
            f"missing={list(missing)[:10]}, unexpected={list(unexpected)[:10]}"
        )
    model.to(device)
    model.eval()
    return model


def _build_dataset(args):
    return DatasetS3(
        config={
            "mode": "waveform",
            "soundscape_dir": args.soundscape_dir,
            "oracle_target_dir": args.oracle_target_dir,
            "sr": args.sample_rate,
        },
        n_sources=args.n_sources,
        label_set=args.label_set,
        return_source=True,
        label_vector_mode="stack",
        silence_label_mode="zeros",
        return_meta=False,
    )


def _to_device(batch: Dict, device: torch.device) -> Dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _write_wav(path: str, waveform: np.ndarray, sample_rate: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    waveform = np.asarray(waveform, dtype=np.float32)
    waveform = np.clip(waveform, -1.0, 1.0)
    if sf is not None:
        sf.write(path, waveform, sample_rate)
    else:
        wavfile.write(path, sample_rate, waveform)


def _safe_label(label: str) -> str:
    return str(label).replace("/", "_").replace(" ", "")


def _write_manifest(rows: List[MatchResult], path: str) -> None:
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


def _export_batch(output, batch, args, manifest_rows: List[MatchResult]):
    est = output[args.output_key].detach().cpu().float()
    if est.dim() == 4 and est.shape[2] != 1:
        est = est[:, :, :1]
    ref = batch["dry_sources"].detach().cpu().float()
    batch_results = match_batch(
        est_sources=est,
        ref_sources=ref,
        labels=batch["label"],
        soundscapes=batch["soundscape"],
        metric=args.match_metric,
        min_match_score=args.min_match_score,
        min_match_margin=args.min_match_margin,
        min_energy_db=args.min_energy_db,
        clean_match_score=args.clean_match_score,
        clean_match_margin=args.clean_match_margin,
        uncertain_weight=args.uncertain_weight,
        save_uncertain=args.save_uncertain,
        include_unmatched=args.save_unmatched_manifest,
    )
    for batch_idx, rows in enumerate(batch_results):
        output_slot = 0
        for row in rows:
            if row.saved and row.label != "silence":
                filename = f"{row.soundscape}_{output_slot:02d}_{_safe_label(row.label)}.wav"
                wav_path = os.path.join(args.output_dir, filename)
                _write_wav(wav_path, est[batch_idx, row.estimate_slot, 0].numpy(), args.sample_rate)
                output_slot += 1
            manifest_rows.append(row)


def main():
    parser = argparse.ArgumentParser(description="Export USS estimated sources with oracle-matched labels for SC training.")
    parser.add_argument("--config", required=True, help="USS model training config containing lightning_module.args.model")
    parser.add_argument("--checkpoint", required=True, help="USS checkpoint path")
    parser.add_argument("--soundscape_dir", required=True, help="Directory of mixture wav files")
    parser.add_argument("--oracle_target_dir", required=True, help="Directory of oracle target wav files with labels in filenames")
    parser.add_argument("--output_dir", required=True, help="Output estimate_target_dir for EstimatedSourceClassifierDataset")
    parser.add_argument("--manifest_path", default=None, help="Optional CSV manifest for match quality and sample weights")
    parser.add_argument("--sample_rate", type=int, default=32000)
    parser.add_argument("--n_sources", type=int, default=3)
    parser.add_argument("--label_set", default="dcase2026t4")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_key", default="foreground_waveform", help="Model output key to export")
    parser.add_argument("--match_metric", default="sa_sdr", choices=["sa_sdr", "si_sdr"], help="Metric used for oracle-estimate matching")
    parser.add_argument("--min_match_score", type=float, default=-10.0, help="Reject matches below this score")
    parser.add_argument("--min_match_margin", type=float, default=-1000000000.0, help="Reject ambiguous matches below this margin")
    parser.add_argument("--min_energy_db", type=float, default=-60.0, help="Reject estimates below this RMS dB")
    parser.add_argument("--clean_match_score", type=float, default=0.0, help="Clean label threshold for match score")
    parser.add_argument("--clean_match_margin", type=float, default=2.0, help="Clean label threshold for match margin")
    parser.add_argument("--uncertain_weight", type=float, default=0.35, help="Sample weight for uncertain saved matches")
    parser.add_argument("--save_uncertain", action="store_true", help="Also write uncertain matches for robust SC fine-tuning")
    parser.add_argument("--save_unmatched_manifest", action="store_true", help="Record unmatched estimate slots in manifest")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    model = _load_model(args.config, args.checkpoint, device)
    dataset = _build_dataset(args)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=dataset.collate_fn)

    rows: List[MatchResult] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="export estimated sources for SC"):
            device_batch = _to_device(batch, device)
            output = model({"mixture": device_batch["mixture"]})
            _export_batch(output, batch, args, rows)
    if args.manifest_path:
        _write_manifest(rows, args.manifest_path)
    print(f"Exported estimated sources to: {args.output_dir}")
    if args.manifest_path:
        print(f"Wrote match manifest to: {args.manifest_path}")


if __name__ == "__main__":
    main()
