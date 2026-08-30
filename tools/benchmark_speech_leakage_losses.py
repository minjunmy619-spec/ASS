#!/usr/bin/env python3
"""Probe GT-gated Speech leakage metrics with controlled reference injections."""

from __future__ import annotations

from argparse import ArgumentParser
from collections.abc import Sequence
import csv
import json
from pathlib import Path

import torch

from spectral_feature_compression.core.loss.composite_separation import CompositeSeparationSpectralLoss
from tools.evaluate_clap_whisper_stems import _load_mono, _resolve_path


def inject_speech_leakage(
    reference_stems: torch.Tensor,
    *,
    source_order: Sequence[str],
    target_source: str,
    leakage_db: float,
    start_sample: int = 0,
    duration_samples: int | None = None,
) -> torch.Tensor:
    """Add a known Speech excerpt to one clean target stem without changing references."""

    if reference_stems.ndim != 4:
        raise ValueError(
            "reference_stems must have shape [batch, source, channel, samples], "
            f"got {tuple(reference_stems.shape)}"
        )
    source_order = tuple(str(source) for source in source_order)
    if "speech" not in source_order or target_source not in source_order:
        raise ValueError("source_order must contain speech and target_source")
    if target_source == "speech":
        raise ValueError("target_source must not be speech")
    samples = reference_stems.shape[-1]
    if start_sample < 0 or start_sample >= samples:
        raise ValueError(f"start_sample must be in [0, {samples}), got {start_sample}")
    if duration_samples is None:
        duration_samples = samples - start_sample
    if duration_samples <= 0:
        raise ValueError(f"duration_samples must be positive, got {duration_samples}")
    end_sample = min(samples, start_sample + duration_samples)
    leaked = reference_stems.clone()
    gain = float(10.0 ** (float(leakage_db) / 20.0))
    speech_index = source_order.index("speech")
    target_index = source_order.index(target_source)
    leaked[:, target_index, :, start_sample:end_sample] += gain * reference_stems[
        :, speech_index, :, start_sample:end_sample
    ]
    return leaked


def probe_speech_leakage_levels(
    reference_stems: torch.Tensor,
    *,
    scorer: CompositeSeparationSpectralLoss,
    source_order: Sequence[str],
    target_source: str,
    leakage_db_values: Sequence[float],
    start_sample: int = 0,
    duration_samples: int | None = None,
) -> list[dict[str, float]]:
    """Return metric response for increasing controlled Speech leakage levels."""

    results = []
    target_key = f"speech_leakage_tf_{target_source}"
    with torch.no_grad():
        for leakage_db in leakage_db_values:
            estimate = inject_speech_leakage(
                reference_stems,
                source_order=source_order,
                target_source=target_source,
                leakage_db=float(leakage_db),
                start_sample=start_sample,
                duration_samples=duration_samples,
            )
            components = scorer.speech_leakage_components(estimate, reference_stems)
            results.append(
                {
                    "leakage_db": float(leakage_db),
                    "speech_leakage_tf": float(components["speech_leakage_tf"]),
                    target_key: float(components[target_key]),
                }
            )
    return results


def _parse_sources(value: str) -> tuple[str, ...]:
    sources = tuple(item.strip() for item in value.split(",") if item.strip())
    if not sources:
        raise ValueError("Source list must not be empty")
    return sources


def _parse_levels(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("Leakage dB list must not be empty")
    return values


def _is_non_decreasing(values: Sequence[float], *, atol: float = 1.0e-8) -> bool:
    return all(next_value + atol >= value for value, next_value in zip(values, values[1:], strict=False))


def _parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Benchmark GT-gated Speech leakage metrics using controlled injections")
    parser.add_argument("manifest", type=Path, help="Paired manifest with reference_{stem}_filepath columns")
    parser.add_argument("--source-order", default="speech,music,effects")
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--target-sources", default="music,effects")
    parser.add_argument("--leakage-db", default="-40,-30,-20,-10,-5")
    parser.add_argument("--duration-seconds", type=float, default=1.0)
    parser.add_argument("--speech-active-db", type=float, default=-45.0)
    parser.add_argument("--target-relative-db", type=float, default=12.0)
    parser.add_argument("--mask-softness-db", type=float, default=3.0)
    parser.add_argument("--tolerance-ratio", type=float, default=0.0)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--output-json", type=Path)
    return parser


def main() -> None:
    args = _parse_args().parse_args()
    source_order = _parse_sources(args.source_order)
    target_sources = _parse_sources(args.target_sources)
    leakage_db_values = tuple(sorted(_parse_levels(args.leakage_db)))
    if args.sample_rate <= 0 or args.duration_seconds <= 0.0:
        raise ValueError("sample-rate and duration-seconds must be positive")
    manifest_path = args.manifest.expanduser()
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {f"reference_{source}_filepath" for source in source_order}
    if not rows:
        raise ValueError(f"Manifest contains no rows: {manifest_path}")
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"Paired manifest is missing columns: {missing}")

    scorer = CompositeSeparationSpectralLoss(
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        source_order=source_order,
        speech_leakage_weight=1.0,
        speech_leakage_target_sources=target_sources,
        speech_leakage_n_fft=args.n_fft,
        speech_leakage_hop_length=args.hop_length,
        speech_leakage_speech_active_db=args.speech_active_db,
        speech_leakage_target_relative_db=args.target_relative_db,
        speech_leakage_mask_softness_db=args.mask_softness_db,
        speech_leakage_tolerance_ratio=args.tolerance_ratio,
    )
    duration_samples = int(round(args.duration_seconds * args.sample_rate))
    records = []
    for row_index, row in enumerate(rows, start=2):
        audio = []
        for source in source_order:
            value = str(row.get(f"reference_{source}_filepath", "")).strip()
            if not value:
                raise ValueError(f"Empty reference_{source}_filepath in row {row_index}")
            path = _resolve_path(value, manifest_dir=manifest_path.parent)
            audio.append(_load_mono(path, sample_rate=args.sample_rate))
        if len({item.numel() for item in audio}) != 1:
            raise ValueError(f"Reference stems must have equal lengths in row {row_index}")
        references = torch.stack(audio)[None, :, None, :]
        start_sample = max(0, (references.shape[-1] - duration_samples) // 2)
        actual_duration = min(duration_samples, references.shape[-1] - start_sample)
        target_results = {
            target: probe_speech_leakage_levels(
                references,
                scorer=scorer,
                source_order=source_order,
                target_source=target,
                leakage_db_values=leakage_db_values,
                start_sample=start_sample,
                duration_samples=actual_duration,
            )
            for target in target_sources
        }
        records.append(
            {
                "row": row_index,
                "recording_id": str(row.get("recording_id", row.get("mixture_id", row_index - 2))),
                "target_results": target_results,
            }
        )

    monotonic = {
        target: all(
            _is_non_decreasing([record[target_key] for record in row["target_results"][target]])
            for row in records
        )
        for target in target_sources
        for target_key in (f"speech_leakage_tf_{target}",)
    }
    report = {
        "manifest": str(manifest_path),
        "sample_rate": args.sample_rate,
        "leakage_db": list(leakage_db_values),
        "duration_seconds": args.duration_seconds,
        "target_sources": list(target_sources),
        "per_row_monotonic": monotonic,
        "rows": records,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.output_json is not None:
        args.output_json.expanduser().parent.mkdir(parents=True, exist_ok=True)
        args.output_json.expanduser().write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
