#!/usr/bin/env python3
from __future__ import annotations

from typing import Any

from argparse import ArgumentParser
from collections.abc import Mapping, Sequence
import csv
import json
from pathlib import Path
import sys

import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spectral_feature_compression.common.datasets.on_the_fly_stem_dataset import OnTheFlyStemDataset  # noqa: E402

_DEFAULT_SOURCE_ORDER = ("speech", "music", "effects")


def _parse_source_order(value: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        stems = tuple(item.strip() for item in value.split(",") if item.strip())
    else:
        stems = tuple(str(item) for item in value)
    if not stems:
        raise ValueError("source_order must contain at least one stem")
    if len(set(stems)) != len(stems):
        raise ValueError(f"source_order contains duplicates: {stems}")
    return stems


def _load_synthesis_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.expanduser().open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("synthesis JSON must contain an object/mapping")
    return data


def _parse_source_pool_specs(specs: Sequence[str]) -> dict[str, list[str]]:
    pools: dict[str, list[str]] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"source pool must be STEM=PATH, got {spec!r}")
        stem, path = spec.split("=", 1)
        stem = stem.strip()
        path = path.strip()
        if not stem or not path:
            raise ValueError(f"source pool must be STEM=PATH, got {spec!r}")
        pools.setdefault(stem, []).append(path)
    return pools


def _csv_paths(value: Sequence[str | Path] | str | Path | None) -> list[str | Path] | None:
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        return [value]
    return list(value)


def export_fixed_stem_mixes(
    *,
    output_csv: str | Path,
    output_audio_dir: str | Path | None = None,
    output_split: str,
    num_examples: int,
    sr: int,
    duration: float,
    seed: int,
    source_order: Sequence[str] = _DEFAULT_SOURCE_ORDER,
    source_pools: Mapping[str, Sequence[str | Path] | str | Path] | None = None,
    source_manifest_csv: Sequence[str | Path] | str | Path | None = None,
    source_manifest_split: str | Sequence[str] | None = None,
    synthesis: Mapping[str, Any] | None = None,
    export_mixtures: bool = False,
    audio_subtype: str = "FLOAT",
) -> Path:
    """Export a deterministic fixed split manifest and rendered reference stems.

    The output CSV is directly usable by ``OnTheFlyStemDataModule`` through
    ``val_fixed_mix_manifest_csv`` or ``test_fixed_mix_manifest_csv``.  Reference
    stem WAVs are always rendered because the separation losses need fixed
    targets.  Mixture WAVs are optional; when omitted, the fixed dataset
    reconstructs ``wav`` as ``ref.sum(dim=0)``.
    """

    if num_examples <= 0:
        raise ValueError(f"num_examples must be positive, got {num_examples}")
    if sr <= 0:
        raise ValueError(f"sr must be positive, got {sr}")
    if duration <= 0.0:
        raise ValueError(f"duration must be positive, got {duration}")
    stems = tuple(str(stem) for stem in source_order)
    if sum(item is not None for item in (source_pools, source_manifest_csv)) != 1:
        raise ValueError("Provide exactly one of source_pools or source_manifest_csv")

    output_csv = Path(output_csv).expanduser()
    if output_audio_dir is None:
        output_audio_dir = output_csv.parent / output_csv.stem
    output_audio_dir = Path(output_audio_dir).expanduser()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_audio_dir.mkdir(parents=True, exist_ok=True)

    dataset_kwargs = dict(synthesis or {})
    dataset_kwargs.pop("return_metadata", None)
    mixture_duration = dataset_kwargs.pop("mixture_duration", None)
    if mixture_duration is not None:
        duration = float(mixture_duration)

    dataset = OnTheFlyStemDataset(
        source_pools=source_pools,
        source_manifest_csv=source_manifest_csv,
        manifest_split=source_manifest_split,
        source_order=stems,
        sr=sr,
        duration=duration,
        dataset_length=num_examples,
        seed=seed,
        return_metadata=True,
        **dataset_kwargs,
    )

    fieldnames = [
        "mixture_id",
        "split",
        "sample_rate",
        "duration",
        "n_samples",
        "mixture_filepath",
        *[f"{stem}_filepath" for stem in stems],
        "active_stems",
        "source_paths_json",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(num_examples):
            wav, ref, metadata = dataset[index]
            mixture_id = f"{output_split}_{index:06d}"
            split_dir = output_audio_dir / output_split
            stem_paths: dict[str, str] = {}
            for stem_idx, stem_name in enumerate(stems):
                stem_dir = split_dir / "refs" / stem_name
                stem_dir.mkdir(parents=True, exist_ok=True)
                stem_path = stem_dir / f"{mixture_id}_{stem_name}.wav"
                sf.write(stem_path, ref[stem_idx, 0].cpu().numpy(), sr, subtype=audio_subtype)
                stem_paths[stem_name] = str(stem_path.resolve())

            mixture_path_str = ""
            if export_mixtures:
                mixture_dir = split_dir / "mixtures"
                mixture_dir.mkdir(parents=True, exist_ok=True)
                mixture_path = mixture_dir / f"{mixture_id}_mix.wav"
                sf.write(mixture_path, wav[0].cpu().numpy(), sr, subtype=audio_subtype)
                mixture_path_str = str(mixture_path.resolve())

            row = {
                "mixture_id": mixture_id,
                "split": output_split,
                "sample_rate": str(sr),
                "duration": f"{duration:.9g}",
                "n_samples": str(wav.shape[-1]),
                "mixture_filepath": mixture_path_str,
                "active_stems": "|".join(metadata.get("active_stems", [])),
                "source_paths_json": json.dumps(metadata.get("source_paths", {}), ensure_ascii=False, sort_keys=True),
            }
            for stem_name in stems:
                row[f"{stem_name}_filepath"] = stem_paths[stem_name]
            writer.writerow(row)
    return output_csv


def build_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Export fixed validation/test stem-mixture manifests for DnR training.")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--source-manifest-csv",
        action="append",
        default=None,
        help="Curated source CSV. Repeat to combine multiple CSVs.",
    )
    source_group.add_argument(
        "--source-pool",
        action="append",
        default=None,
        help="Source pool in STEM=PATH form. Repeat for multiple stems/roots.",
    )
    parser.add_argument("--source-manifest-split", default=None, help="Optional split filter for source CSV rows.")
    parser.add_argument("--output-csv", required=True, help="Fixed mixture manifest CSV to write.")
    parser.add_argument("--output-audio-dir", default=None, help="Directory for rendered reference stems and mixtures.")
    parser.add_argument(
        "--output-split", required=True, help="Split label written to the fixed manifest, e.g. validation/test."
    )
    parser.add_argument("--num-examples", type=int, required=True, help="Number of fixed mixtures to export.")
    parser.add_argument("--sr", type=int, default=44100, help="Target sample rate.")
    parser.add_argument("--duration", type=float, default=6.0, help="Mixture duration in seconds.")
    parser.add_argument("--seed", type=int, default=0, help="Base deterministic seed for exported mixture recipes.")
    parser.add_argument("--source-order", default=",".join(_DEFAULT_SOURCE_ORDER), help="Comma-separated stem order.")
    parser.add_argument(
        "--synthesis-json",
        type=Path,
        default=None,
        help="Optional JSON mapping of OnTheFlyStemDataset synthesis controls.",
    )
    parser.add_argument(
        "--export-mixtures", action="store_true", help="Also render mixture WAVs. Ref stem WAVs are always rendered."
    )
    parser.add_argument(
        "--audio-subtype",
        default="FLOAT",
        help="soundfile subtype for exported WAVs; FLOAT preserves consistency best.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    source_pools = _parse_source_pool_specs(args.source_pool or []) if args.source_pool else None
    output_csv = export_fixed_stem_mixes(
        output_csv=args.output_csv,
        output_audio_dir=args.output_audio_dir,
        output_split=args.output_split,
        num_examples=args.num_examples,
        sr=args.sr,
        duration=args.duration,
        seed=args.seed,
        source_order=_parse_source_order(args.source_order),
        source_pools=source_pools,
        source_manifest_csv=_csv_paths(args.source_manifest_csv),
        source_manifest_split=args.source_manifest_split,
        synthesis=_load_synthesis_json(args.synthesis_json),
        export_mixtures=args.export_mixtures,
        audio_subtype=args.audio_subtype,
    )
    print(f"Wrote fixed mixture manifest: {output_csv}")


if __name__ == "__main__":
    main()
