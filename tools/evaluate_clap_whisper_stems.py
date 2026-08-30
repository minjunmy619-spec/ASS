#!/usr/bin/env python3
"""Evaluate real separated stem waveforms with frozen CLAP and Whisper."""

from __future__ import annotations

from typing import Any, Callable

from argparse import ArgumentParser
from collections.abc import Mapping, Sequence
import csv
import json
import math
from pathlib import Path

import numpy as np

import torch

import torchaudio.functional as AF

import soundfile as sf

from spectral_feature_compression.core.loss.composite_separation import CompositeSeparationSpectralLoss
from spectral_feature_compression.core.loss.frozen_audio_perceptual import ClapSemanticLoss, WhisperStemPerceptualLoss

_DEFAULT_CLAP_PROMPT_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "recipes/dnr/models"
    / "tvconv-pyramid-sourceaware-sfclite-convgru-smoothup-smoothlogit-npu.speech-music-residual-sfx."
    "robust-distill.rt192k.fp512keep475.broadcast-v1.clap-a2a-bank-whisper-ft"
    / "clap_prompts.yaml"
)


def _resolve_path(value: str, *, manifest_dir: Path) -> Path:
    path = Path(value).expanduser()
    path = path if path.is_absolute() else manifest_dir / path
    if not path.is_file():
        raise FileNotFoundError(f"Stem audio does not exist: {path}")
    return path


def _load_mono(path: Path, *, sample_rate: int) -> torch.Tensor:
    audio_np, source_rate = sf.read(path, always_2d=True, dtype="float32")
    if audio_np.size == 0:
        raise ValueError(f"Could not read audio from {path}")
    audio = torch.from_numpy(audio_np.T.copy()).float().mean(dim=0, keepdim=True)
    if int(source_rate) != sample_rate:
        audio = AF.resample(audio, orig_freq=int(source_rate), new_freq=sample_rate)
    return audio.squeeze(0)


def _whisper_metrics(result: Mapping[str, Any]) -> dict[str, Any]:
    records = result.get("windows", result.get("segments", ()))
    segments = [segment for segment in records if isinstance(segment, Mapping)]
    durations = np.asarray(
        [max(0.0, float(segment.get("end", 0.0)) - float(segment.get("start", 0.0))) for segment in segments],
        dtype=np.float64,
    )

    def duration_weighted(field: str) -> float | None:
        values = np.asarray([float(segment.get(field, math.nan)) for segment in segments], dtype=np.float64)
        valid = np.isfinite(values) & (durations > 0.0)
        if not valid.any():
            return None
        weights = durations[valid] / durations[valid].sum()
        return float(np.sum(weights * values[valid]))

    avg_logprob = duration_weighted("avg_logprob")
    no_speech_probability = duration_weighted("no_speech_prob")
    text = str(result.get("text", "")).strip()
    return {
        "avg_logprob": avg_logprob,
        "no_speech_probability": no_speech_probability,
        "window_count": len(segments),
        "transcript": text,
        "transcript_characters": len(text),
        "windows": segments,
    }


def _summary(values: Sequence[float | None]) -> dict[str, float | int | None]:
    finite = np.asarray(
        [float(value) for value in values if value is not None and math.isfinite(float(value))],
        dtype=np.float64,
    )
    if finite.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }
    return {
        "count": int(finite.size),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "min": float(finite.min()),
        "max": float(finite.max()),
    }


def _masked_weighted_mean(
    values: torch.Tensor,
    valid: torch.Tensor,
    weights: torch.Tensor,
) -> float | None:
    masked_weights = valid.to(weights.dtype) * weights
    denominator = masked_weights.sum()
    if not bool(denominator > 0):
        return None
    return float((values * masked_weights).sum() / denominator)


def evaluate_manifest(
    manifest_path: str | Path,
    *,
    clap_scorer: Any,
    whisper_transcriber: Callable[[Path], Mapping[str, Any]] | None = None,
    whisper_stems: Sequence[str] = ("speech",),
    speech_leakage_scorer: Any | None = None,
    whisper_leakage_scorer: Any | None = None,
    source_order: Sequence[str] = ("speech", "music", "effects"),
    sample_rate: int = 44100,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Evaluate already-separated real waveforms without modifying their content."""

    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    source_order = tuple(str(source) for source in source_order)
    unknown_whisper_stems = sorted(set(whisper_stems) - set(source_order))
    if unknown_whisper_stems:
        raise ValueError(f"whisper_stems contains unknown sources: {unknown_whisper_stems}")
    manifest_path = Path(manifest_path).expanduser()
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        rows = list(reader)
    required = {f"{source}_filepath" for source in source_order}
    missing = sorted(required - fieldnames)
    if missing:
        raise ValueError(f"Manifest {manifest_path} is missing columns: {missing}")
    reference_fields = {f"reference_{source}_filepath" for source in source_order}
    present_reference_fields = reference_fields & fieldnames
    if present_reference_fields and present_reference_fields != reference_fields:
        missing_reference_fields = sorted(reference_fields - present_reference_fields)
        raise ValueError(
            f"Manifest {manifest_path} has partial reference stems; missing columns: {missing_reference_fields}"
        )
    has_references = bool(present_reference_fields)
    if has_references and not hasattr(clap_scorer, "reference_window_metrics"):
        raise ValueError("Reference stem columns require a CLAP scorer with reference_window_metrics()")
    if not has_references and (speech_leakage_scorer is not None or whisper_leakage_scorer is not None):
        raise ValueError("Speech leakage metrics require all reference stem columns")
    if not rows:
        raise ValueError(f"Manifest contains no rows: {manifest_path}")

    output_rows = []
    clap_metric_names = ["positive_similarity", "negative_similarity", "purity_margin"]
    has_prompt_banks = bool(getattr(clap_scorer, "has_prompt_banks", False))
    if has_prompt_banks:
        clap_metric_names.extend(("prompt_bank_probability", "prompt_bank_margin"))
    if has_references:
        clap_metric_names.extend(("reference_similarity", "relative_cross_stem_excess"))
    clap_values = {
        source: {metric: [] for metric in clap_metric_names}
        for source in source_order
    }
    whisper_values = {
        source: {"avg_logprob": [], "no_speech_probability": [], "transcript_characters": []}
        for source in whisper_stems
    }
    speech_leakage_values: dict[str, list[float]] = {}
    whisper_leakage_values: dict[str, list[float]] = {}
    target_device = torch.device(device)
    for row_idx, row in enumerate(rows, start=2):
        paths = {}
        audio = []
        for source in source_order:
            value = str(row.get(f"{source}_filepath", "")).strip()
            if not value:
                raise ValueError(f"Empty {source}_filepath in {manifest_path} row {row_idx}")
            path = _resolve_path(value, manifest_dir=manifest_path.parent)
            paths[source] = path
            audio.append(_load_mono(path, sample_rate=sample_rate))
        stem_lengths = {source: item.numel() for source, item in zip(source_order, audio, strict=True)}
        if len(set(stem_lengths.values())) != 1:
            raise ValueError(f"Separated stems must have equal lengths in row {row_idx}, got {stem_lengths}")
        stacked = torch.stack(audio)
        stems = stacked[None, :, None, :].to(target_device)
        reference_stems = None
        if has_references:
            reference_audio = []
            for source in source_order:
                value = str(row.get(f"reference_{source}_filepath", "")).strip()
                if not value:
                    raise ValueError(f"Empty reference_{source}_filepath in {manifest_path} row {row_idx}")
                path = _resolve_path(value, manifest_dir=manifest_path.parent)
                reference_audio.append(_load_mono(path, sample_rate=sample_rate))
            all_lengths = [item.numel() for item in audio + reference_audio]
            if len(set(all_lengths)) != 1:
                raise ValueError(
                    f"Separated and reference stems must have equal lengths in row {row_idx}, got {all_lengths}"
                )
            reference_stems = torch.stack(reference_audio)[None, :, None, :].to(target_device)
        with torch.no_grad():
            if hasattr(clap_scorer, "window_metrics"):
                combined_metrics = clap_scorer.window_metrics(stems, reference_stems)
                positive = combined_metrics["positive_similarity"]
                negative = combined_metrics["negative_similarity"]
                bank_scores = combined_metrics.get("prompt_bank_scores")
                reference_metrics = (
                    {
                        name: combined_metrics[name]
                        for name in (
                            "same_stem_similarity",
                            "relative_cross_stem_excess",
                            "reference_active",
                            "cross_stem_valid",
                        )
                    }
                    if reference_stems is not None
                    else None
                )
            elif hasattr(clap_scorer, "semantic_window_scores"):
                positive, negative = clap_scorer.semantic_window_scores(stems)
                bank_scores = clap_scorer.prompt_bank_window_scores(stems) if has_prompt_banks else None
                reference_metrics = (
                    clap_scorer.reference_window_metrics(stems, reference_stems)
                    if reference_stems is not None
                    else None
                )
            else:
                positive, negative = clap_scorer.semantic_scores(stems)
                positive = positive[..., None]
                negative = negative[..., None]
                bank_scores = clap_scorer.prompt_bank_window_scores(stems) if has_prompt_banks else None
                reference_metrics = (
                    clap_scorer.reference_window_metrics(stems, reference_stems)
                    if reference_stems is not None
                    else None
                )
            if speech_leakage_scorer is not None:
                if reference_stems is None:
                    raise RuntimeError("Reference stems unexpectedly missing for speech leakage metrics")
                raw_components = speech_leakage_scorer.speech_leakage_components(stems, reference_stems)
                speech_leakage_row = {name: float(value) for name, value in raw_components.items()}
                for name, value in speech_leakage_row.items():
                    speech_leakage_values.setdefault(name, []).append(value)
            else:
                speech_leakage_row = {}
            if whisper_leakage_scorer is not None:
                if reference_stems is None:
                    raise RuntimeError("Reference stems unexpectedly missing for Whisper leakage metrics")
                _, raw_components = whisper_leakage_scorer.forward_with_components(stems, reference_stems)
                whisper_leakage_row = {name: float(value) for name, value in raw_components.items()}
                for name, value in whisper_leakage_row.items():
                    whisper_leakage_values.setdefault(name, []).append(value)
            else:
                whisper_leakage_row = {}
        positive = positive[0].detach().cpu()
        negative = negative[0].detach().cpu()
        if bank_scores is not None:
            bank_scores = bank_scores[0].detach().cpu()
            bank_probabilities = bank_scores.softmax(dim=-1)
        else:
            bank_probabilities = None
        if reference_metrics is not None:
            reference_metrics = {name: value[0].detach().cpu() for name, value in reference_metrics.items()}
        if hasattr(clap_scorer, "window_bounds"):
            window_bounds = clap_scorer.window_bounds(stacked.shape[-1])
        else:
            window_bounds = ((0, stacked.shape[-1]),)
        if positive.shape[-1] != len(window_bounds):
            raise ValueError(
                f"CLAP returned {positive.shape[-1]} windows but reported {len(window_bounds)} bounds"
            )
        window_durations = positive.new_tensor([end - start for start, end in window_bounds])
        window_weights = window_durations / window_durations.sum()

        clap_row = {}
        for source_idx, source in enumerate(source_order):
            has_negative = (
                clap_scorer.source_has_negative_prompt(source)
                if hasattr(clap_scorer, "source_has_negative_prompt")
                else True
            )
            positive_values = positive[source_idx]
            negative_values = negative[source_idx]
            positive_mean = float((positive_values * window_weights).sum())
            negative_mean = float((negative_values * window_weights).sum()) if has_negative else None
            purity_margin = positive_mean - negative_mean if negative_mean is not None else None
            prompt_probability_values = None
            prompt_margin_values = None
            if bank_scores is not None and bank_probabilities is not None:
                prompt_probability_values = bank_probabilities[source_idx, :, source_idx]
                other_source_mask = torch.arange(len(source_order)) != source_idx
                prompt_margin_values = (
                    bank_scores[source_idx, :, source_idx]
                    - bank_scores[source_idx, :, other_source_mask].max(dim=-1).values
                )
            reference_similarity_values = None
            reference_active_values = None
            relative_excess_values = None
            relative_excess_valid = None
            if reference_metrics is not None:
                reference_similarity_values = reference_metrics["same_stem_similarity"][source_idx]
                reference_active_values = reference_metrics["reference_active"][source_idx]
                relative_excess_values = reference_metrics["relative_cross_stem_excess"][source_idx]
                relative_excess_valid = reference_metrics["cross_stem_valid"][source_idx]
            source_scores = {
                "positive_similarity": positive_mean,
                "negative_similarity": negative_mean,
                "purity_margin": purity_margin,
                "window_count": len(window_bounds),
            }
            if prompt_probability_values is not None and prompt_margin_values is not None:
                source_scores["prompt_bank_probability"] = float(
                    (prompt_probability_values * window_weights).sum()
                )
                source_scores["prompt_bank_margin"] = float((prompt_margin_values * window_weights).sum())
            if (
                reference_similarity_values is not None
                and reference_active_values is not None
                and relative_excess_values is not None
                and relative_excess_valid is not None
            ):
                source_scores["reference_similarity"] = _masked_weighted_mean(
                    reference_similarity_values,
                    reference_active_values,
                    window_weights,
                )
                source_scores["relative_cross_stem_excess"] = _masked_weighted_mean(
                    relative_excess_values,
                    relative_excess_valid,
                    window_weights,
                )
            windows = []
            for window_idx, (start, end) in enumerate(window_bounds):
                window = {
                    "start_seconds": start / sample_rate,
                    "end_seconds": end / sample_rate,
                    "positive_similarity": float(positive_values[window_idx]),
                    "negative_similarity": float(negative_values[window_idx]) if has_negative else None,
                    "purity_margin": (
                        float(positive_values[window_idx] - negative_values[window_idx])
                        if has_negative
                        else None
                    ),
                }
                if prompt_probability_values is not None and prompt_margin_values is not None:
                    window["prompt_bank_probability"] = float(prompt_probability_values[window_idx])
                    window["prompt_bank_margin"] = float(prompt_margin_values[window_idx])
                if (
                    reference_similarity_values is not None
                    and reference_active_values is not None
                    and relative_excess_values is not None
                    and relative_excess_valid is not None
                ):
                    window["reference_similarity"] = (
                        float(reference_similarity_values[window_idx])
                        if bool(reference_active_values[window_idx])
                        else None
                    )
                    window["relative_cross_stem_excess"] = (
                        float(relative_excess_values[window_idx])
                        if bool(relative_excess_valid[window_idx])
                        else None
                    )
                windows.append(window)
            source_scores["windows"] = windows
            clap_row[source] = source_scores
            for metric in clap_values[source]:
                clap_values[source][metric].append(source_scores[metric])

        whisper_row = {}
        if whisper_transcriber is not None:
            for source in whisper_stems:
                metrics = _whisper_metrics(whisper_transcriber(paths[source]))
                whisper_row[source] = metrics
                for metric in whisper_values[source]:
                    whisper_values[source][metric].append(metrics[metric])

        output_rows.append(
            {
                "row": row_idx,
                "recording_id": str(row.get("recording_id", row.get("mixture_id", row_idx - 2))),
                "clap": clap_row,
                "whisper": whisper_row,
                "speech_leakage": speech_leakage_row,
                "whisper_relative_bleed": whisper_leakage_row,
            }
        )

    summary: dict[str, Any] = {
        "clap": {
            source: {metric: _summary(values) for metric, values in metrics.items()}
            for source, metrics in clap_values.items()
        }
    }
    if whisper_transcriber is not None:
        summary["whisper"] = {
            source: {metric: _summary(values) for metric, values in metrics.items()}
            for source, metrics in whisper_values.items()
        }
    if speech_leakage_scorer is not None:
        summary["speech_leakage"] = {
            name: _summary(values) for name, values in speech_leakage_values.items()
        }
    if whisper_leakage_scorer is not None:
        summary["whisper_relative_bleed"] = {
            name: _summary(values) for name, values in whisper_leakage_values.items()
        }
    return {
        "manifest": str(manifest_path),
        "sample_rate": sample_rate,
        "has_reference_stems": has_references,
        "clap_config": {
            "audio_model": getattr(clap_scorer, "amodel", None),
            "audio_antibleed_margin": getattr(clap_scorer, "audio_antibleed_margin", None),
        },
        "row_count": len(output_rows),
        "rows": output_rows,
        "summary": summary,
    }


def _parse_sources(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result:
        raise ValueError("Source list must not be empty")
    return result


def _parse_args(argv: Sequence[str] | None = None) -> Any:
    parser = ArgumentParser(description="Evaluate real separated stems with frozen CLAP and Whisper")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--source-order", default="speech,music,effects")
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--clap-checkpoint", type=Path)
    parser.add_argument("--clap-model-id", type=int, default=1)
    parser.add_argument("--clap-audio-model", default="HTSAT-tiny")
    parser.add_argument("--clap-audio-antibleed-margin", type=float, default=0.02)
    parser.add_argument("--clap-prompt-config", type=Path, default=_DEFAULT_CLAP_PROMPT_CONFIG)
    parser.add_argument("--allow-clap-download", action="store_true")
    parser.add_argument("--enable-speech-leakage-metrics", action="store_true")
    parser.add_argument("--speech-leakage-source", default="speech")
    parser.add_argument("--speech-leakage-target-sources", default="music,effects")
    parser.add_argument("--speech-leakage-n-fft", type=int, default=1024)
    parser.add_argument("--speech-leakage-hop-length", type=int, default=256)
    parser.add_argument("--speech-leakage-speech-active-db", type=float, default=-45.0)
    parser.add_argument("--speech-leakage-target-relative-db", type=float, default=12.0)
    parser.add_argument("--speech-leakage-mask-softness-db", type=float, default=3.0)
    parser.add_argument("--speech-leakage-tolerance-ratio", type=float, default=0.0)
    parser.add_argument("--whisper-model", default="base")
    parser.add_argument("--whisper-download-root", type=Path)
    parser.add_argument("--whisper-stems", default="speech,music,effects")
    parser.add_argument("--whisper-language")
    parser.add_argument("--disable-whisper", action="store_true")
    parser.add_argument("--enable-whisper-relative-bleed", action="store_true")
    parser.add_argument("--whisper-relative-bleed-margin", type=float, default=0.0)
    parser.add_argument("--whisper-relative-bleed-speech-active-db", type=float, default=-45.0)
    parser.add_argument("--whisper-relative-bleed-target-relative-db", type=float, default=12.0)
    parser.add_argument("--whisper-relative-bleed-mask-softness-db", type=float, default=3.0)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name if args.device == "auto" else args.device)
    source_order = _parse_sources(args.source_order)
    clap_scorer = ClapSemanticLoss(
        sample_rate=args.sample_rate,
        source_order=source_order,
        prompt_config_path=args.clap_prompt_config,
        checkpoint_path=args.clap_checkpoint,
        model_id=args.clap_model_id,
        amodel=args.clap_audio_model,
        audio_antibleed_margin=args.clap_audio_antibleed_margin,
        allow_download=args.allow_clap_download,
    ).to(device)
    speech_leakage_scorer = None
    if args.enable_speech_leakage_metrics:
        speech_leakage_scorer = CompositeSeparationSpectralLoss(
            n_fft=args.speech_leakage_n_fft,
            hop_length=args.speech_leakage_hop_length,
            source_order=source_order,
            speech_leakage_weight=1.0,
            speech_leakage_source=args.speech_leakage_source,
            speech_leakage_target_sources=_parse_sources(args.speech_leakage_target_sources),
            speech_leakage_n_fft=args.speech_leakage_n_fft,
            speech_leakage_hop_length=args.speech_leakage_hop_length,
            speech_leakage_speech_active_db=args.speech_leakage_speech_active_db,
            speech_leakage_target_relative_db=args.speech_leakage_target_relative_db,
            speech_leakage_mask_softness_db=args.speech_leakage_mask_softness_db,
            speech_leakage_tolerance_ratio=args.speech_leakage_tolerance_ratio,
        ).to(device)

    transcriber = None
    whisper_leakage_scorer = None
    whisper_stems: tuple[str, ...] = ()
    if args.enable_whisper_relative_bleed and args.disable_whisper:
        raise ValueError("--enable-whisper-relative-bleed requires Whisper; omit --disable-whisper")
    if not args.disable_whisper:
        try:
            import whisper
        except ModuleNotFoundError as exc:
            raise ImportError(
                "Whisper evaluation requires 'openai-whisper'. Install requirements-perceptual.txt."
            ) from exc
        whisper_model = whisper.load_model(
            args.whisper_model,
            device=str(device),
            download_root=(
                None if args.whisper_download_root is None else str(args.whisper_download_root.expanduser())
            ),
        )
        whisper_model.eval()
        whisper_stems = _parse_sources(args.whisper_stems)
        if args.enable_whisper_relative_bleed:
            whisper_leakage_scorer = WhisperStemPerceptualLoss(
                sample_rate=args.sample_rate,
                source_order=source_order,
                speech_feature_match_weight=0.0,
                cross_stem_bleed_weight=1.0,
                speech_active_db=args.whisper_relative_bleed_speech_active_db,
                target_quiet_relative_db=args.whisper_relative_bleed_target_relative_db,
                mask_softness_db=args.whisper_relative_bleed_mask_softness_db,
                relative_bleed_margin=args.whisper_relative_bleed_margin,
                selected_layer_policy="middle",
                whisper_model=whisper_model,
                mel_filters=whisper.audio.mel_filters("cpu", int(whisper_model.dims.n_mels)),
            ).to(device)

        def transcriber(path: Path) -> Mapping[str, Any]:
            audio_16k = _load_mono(path, sample_rate=16000)
            window_samples = int(30 * 16000)
            windows = []
            texts = []
            for start in range(0, audio_16k.numel(), window_samples):
                end = min(audio_16k.numel(), start + window_samples)
                audio_window = audio_16k[start:end]
                mel = whisper.log_mel_spectrogram(audio_window, n_mels=int(whisper_model.dims.n_mels))
                mel = whisper.pad_or_trim(mel, whisper.audio.N_FRAMES).to(device)
                options = whisper.DecodingOptions(
                    language=args.whisper_language,
                    task="transcribe",
                    temperature=0.0,
                    fp16=device.type == "cuda",
                    without_timestamps=True,
                )
                result = whisper.decode(whisper_model, mel, options)
                text = str(result.text).strip()
                if text:
                    texts.append(text)
                windows.append(
                    {
                        "start": start / 16000.0,
                        "end": end / 16000.0,
                        "avg_logprob": float(result.avg_logprob),
                        "no_speech_prob": float(result.no_speech_prob),
                        "text": text,
                    }
                )
            return {"text": " ".join(texts), "windows": windows}

    report = evaluate_manifest(
        args.manifest,
        clap_scorer=clap_scorer,
        whisper_transcriber=transcriber,
        whisper_stems=whisper_stems,
        speech_leakage_scorer=speech_leakage_scorer,
        whisper_leakage_scorer=whisper_leakage_scorer,
        source_order=source_order,
        sample_rate=args.sample_rate,
        device=device,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
