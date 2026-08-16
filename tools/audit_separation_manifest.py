#!/usr/bin/env python3
from __future__ import annotations

from typing import Any

from argparse import ArgumentParser
from collections import defaultdict
from collections.abc import Sequence
import csv
import json
import math
from pathlib import Path

import numpy as np

import torch

import torchaudio.functional as AF

import soundfile as sf

_TIERS = {"ood_synthetic", "real_multitrack", "real_unlabeled"}


def _resolve_path(value: str, *, manifest_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else manifest_dir / path


def _load_mono(path: Path, *, sample_rate: int | None) -> tuple[np.ndarray, int]:
    audio, source_rate = sf.read(path, always_2d=True, dtype="float32")
    mono = audio.mean(axis=1, dtype=np.float32)
    target_rate = int(source_rate if sample_rate is None else sample_rate)
    if target_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {target_rate}")
    if int(source_rate) != target_rate:
        mono = (
            AF.resample(torch.from_numpy(mono).unsqueeze(0), orig_freq=int(source_rate), new_freq=target_rate)
            .squeeze(0)
            .numpy()
        )
    return mono, target_rate


def _frame_power(audio: np.ndarray, frame_samples: int) -> np.ndarray:
    frame_samples = max(1, int(frame_samples))
    n_frames = max(1, math.ceil(audio.size / frame_samples))
    padded = np.pad(audio.astype(np.float64), (0, n_frames * frame_samples - audio.size))
    return np.mean(np.square(padded.reshape(n_frames, frame_samples)), axis=1)


def _audio_stats(audio: np.ndarray, *, sample_rate: int, frame_ms: float, silence_db: float) -> dict[str, float]:
    power = float(np.mean(np.square(audio.astype(np.float64)))) if audio.size else 0.0
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    rms = math.sqrt(power)
    frame_samples = max(1, int(round(frame_ms * sample_rate / 1000.0)))
    frames = _frame_power(audio, frame_samples)
    threshold = 10.0 ** (silence_db / 10.0)
    return {
        "rms_dbfs": 20.0 * math.log10(max(rms, 1.0e-12)),
        "peak_dbfs": 20.0 * math.log10(max(peak, 1.0e-12)),
        "crest_db": 20.0 * math.log10(max(peak, 1.0e-12) / max(rms, 1.0e-12)),
        "active_fraction": float(np.mean(frames > threshold)),
    }


def _si_sdr(estimate: np.ndarray, reference: np.ndarray) -> float:
    reference_energy = float(np.dot(reference, reference))
    if reference_energy <= 1.0e-12:
        return float("nan")
    scale = float(np.dot(estimate, reference)) / reference_energy
    target = scale * reference
    error = estimate - target
    return 10.0 * math.log10(float(np.dot(target, target)) / max(float(np.dot(error, error)), 1.0e-12))


def _inactive_leakage_db(
    estimate: np.ndarray,
    reference: np.ndarray,
    mixture: np.ndarray,
    *,
    frame_samples: int,
    silence_db: float,
) -> float:
    ref_power = _frame_power(reference, frame_samples)
    est_power = _frame_power(estimate, frame_samples)
    mix_power = _frame_power(mixture, frame_samples)
    inactive = ref_power <= 10.0 ** (silence_db / 10.0)
    if not np.any(inactive):
        return float("nan")
    leakage = float(np.mean(est_power[inactive]))
    context = float(np.mean(mix_power[inactive]))
    return 10.0 * math.log10(max(leakage, 1.0e-12) / max(context, 1.0e-12))


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return {"count": 0, "mean": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def audit_manifest(
    manifest_path: str | Path,
    *,
    tier: str,
    source_order: Sequence[str] = ("speech", "music", "effects"),
    sample_rate: int | None = None,
    frame_ms: float = 80.0,
    silence_db: float = -50.0,
) -> dict[str, Any]:
    """Audit synthetic, real multitrack, or unlabelled real-audio manifests.

    All tiers require ``mixture_filepath``. Labelled tiers additionally use one
    ``{stem}_filepath`` column per source. Optional ``pred_{stem}_filepath``
    columns enable SI-SDR and reference-inactive leakage measurements.
    """

    if tier not in _TIERS:
        raise ValueError(f"tier must be one of {sorted(_TIERS)}, got {tier!r}")
    if frame_ms <= 0.0:
        raise ValueError(f"frame_ms must be positive, got {frame_ms}")
    manifest_path = Path(manifest_path).expanduser()
    metrics: dict[str, list[float]] = defaultdict(list)
    reference_metrics: dict[str, dict[str, list[float]]] = {stem: defaultdict(list) for stem in source_order}
    prediction_metrics: dict[str, dict[str, list[float]]] = {stem: defaultdict(list) for stem in source_order}

    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        rows = list(reader)
    required_columns = {"mixture_filepath"}
    if tier != "real_unlabeled":
        required_columns.update(f"{stem}_filepath" for stem in source_order)
    missing_columns = sorted(required_columns - fieldnames)
    if missing_columns:
        raise ValueError(f"Manifest {manifest_path} is missing columns: {missing_columns}")
    if not rows:
        raise ValueError(f"Manifest contains no rows: {manifest_path}")

    for row_idx, row in enumerate(rows, start=2):
        mixture_value = str(row.get("mixture_filepath", "")).strip()
        if not mixture_value:
            raise ValueError(f"Missing mixture_filepath in {manifest_path} row {row_idx}")
        mixture, row_rate = _load_mono(
            _resolve_path(mixture_value, manifest_dir=manifest_path.parent),
            sample_rate=sample_rate,
        )
        for name, value in _audio_stats(
            mixture,
            sample_rate=row_rate,
            frame_ms=frame_ms,
            silence_db=silence_db,
        ).items():
            metrics[f"mixture_{name}"].append(value)

        references: dict[str, np.ndarray] = {}
        if tier != "real_unlabeled":
            for stem in source_order:
                value = str(row.get(f"{stem}_filepath", "")).strip()
                if value:
                    reference, _ = _load_mono(
                        _resolve_path(value, manifest_dir=manifest_path.parent),
                        sample_rate=row_rate,
                    )
                    if reference.size != mixture.size:
                        raise ValueError(
                            f"{stem}_filepath length differs from mixture in {manifest_path} row {row_idx}"
                        )
                else:
                    reference = np.zeros_like(mixture)
                references[stem] = reference
                for name, stat in _audio_stats(
                    reference,
                    sample_rate=row_rate,
                    frame_ms=frame_ms,
                    silence_db=silence_db,
                ).items():
                    reference_metrics[stem][name].append(stat)

            reconstructed = np.sum(np.stack(list(references.values()), axis=0), axis=0)
            error_power = float(np.mean(np.square(mixture - reconstructed)))
            mixture_power = float(np.mean(np.square(mixture)))
            metrics["additivity_error_db"].append(
                10.0 * math.log10(max(error_power, 1.0e-12) / max(mixture_power, 1.0e-12))
            )

        for stem in source_order:
            prediction_value = str(row.get(f"pred_{stem}_filepath", "")).strip()
            if not prediction_value:
                continue
            if tier == "real_unlabeled":
                raise ValueError("Prediction leakage metrics require reference stems, not tier='real_unlabeled'")
            prediction, _ = _load_mono(
                _resolve_path(prediction_value, manifest_dir=manifest_path.parent),
                sample_rate=row_rate,
            )
            if prediction.size != mixture.size:
                raise ValueError(f"pred_{stem}_filepath length differs from mixture in row {row_idx}")
            prediction_metrics[stem]["si_sdr_db"].append(_si_sdr(prediction, references[stem]))
            frame_samples = max(1, int(round(frame_ms * row_rate / 1000.0)))
            prediction_metrics[stem]["inactive_leakage_db"].append(
                _inactive_leakage_db(
                    prediction,
                    references[stem],
                    mixture,
                    frame_samples=frame_samples,
                    silence_db=silence_db,
                )
            )

    report: dict[str, Any] = {
        "tier": tier,
        "manifest": str(manifest_path),
        "rows": len(rows),
    }
    for name, values in metrics.items():
        report[name] = _summary(values)
    if tier != "real_unlabeled":
        report["references"] = {
            stem: {name: _summary(values) for name, values in stem_metrics.items()}
            for stem, stem_metrics in reference_metrics.items()
        }
        report["predictions"] = {
            stem: {name: _summary(values) for name, values in stem_metrics.items()}
            for stem, stem_metrics in prediction_metrics.items()
            if stem_metrics
        }
    return report


def _parse_args() -> Any:
    parser = ArgumentParser(description="Audit separation manifests for synthesis-to-real domain gaps")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--tier", required=True, choices=sorted(_TIERS))
    parser.add_argument("--source-order", default="speech,music,effects")
    parser.add_argument("--sample-rate", type=int)
    parser.add_argument("--frame-ms", type=float, default=80.0)
    parser.add_argument("--silence-db", type=float, default=-50.0)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    source_order = tuple(item.strip() for item in args.source_order.split(",") if item.strip())
    report = audit_manifest(
        args.manifest,
        tier=args.tier,
        source_order=source_order,
        sample_rate=args.sample_rate,
        frame_ms=args.frame_ms,
        silence_db=args.silence_db,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
