from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

import torch

import pytest

import soundfile as sf

from tools.evaluate_clap_whisper_stems import _parse_args, _whisper_metrics, evaluate_manifest


class _FakeClapScorer:
    source_order = ("speech", "music", "effects")
    amodel = "fake-clap"
    audio_antibleed_margin = 0.02

    def semantic_scores(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert tuple(audio.shape[:3]) == (1, 3, 1)
        positive = audio.new_tensor([[0.8, 0.7, 0.6]])
        negative = audio.new_tensor([[0.1, 0.2, 0.3]])
        return positive, negative

    def semantic_window_scores(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        positive, negative = self.semantic_scores(audio)
        return positive[..., None], negative[..., None]

    def window_bounds(self, n_samples: int) -> tuple[tuple[int, int], ...]:
        return ((0, n_samples),)

    def source_has_negative_prompt(self, source: str) -> bool:
        return source != "speech"


class _ReferenceAwareFakeClapScorer(_FakeClapScorer):
    has_prompt_banks = True

    def prompt_bank_window_scores(self, audio: torch.Tensor) -> torch.Tensor:
        return audio.new_tensor(
            [
                [
                    [[2.0, 0.0, 0.0]],
                    [[0.0, 2.0, 0.0]],
                    [[0.0, 0.0, 2.0]],
                ]
            ]
        )

    def reference_window_metrics(
        self,
        estimate: torch.Tensor,
        reference: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        assert estimate.shape == reference.shape
        return {
            "same_stem_similarity": estimate.new_tensor([[[0.9], [0.8], [0.7]]]),
            "relative_cross_stem_excess": estimate.new_tensor([[[0.1], [0.2], [0.3]]]),
            "reference_active": torch.tensor([[[True], [True], [False]]], device=estimate.device),
            "cross_stem_valid": torch.tensor([[[True], [True], [False]]], device=estimate.device),
        }


class _FakeSpeechLeakageScorer:
    def speech_leakage_components(
        self,
        estimate: torch.Tensor,
        reference: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        assert estimate.shape == reference.shape
        return {
            "speech_leakage_tf": estimate.new_tensor(0.25),
            "speech_leakage_tf_music": estimate.new_tensor(0.5),
            "speech_leakage_tf_effects": estimate.new_tensor(0.0),
        }


def test_evaluate_manifest_scores_real_rendered_stem_paths(tmp_path: Path) -> None:
    paths = {}
    for stem, value in zip(("speech", "music", "effects"), (0.2, 0.1, 0.05), strict=True):
        path = tmp_path / f"{stem}.wav"
        sf.write(path, np.full(800, value, dtype=np.float32), 8000, subtype="FLOAT")
        paths[stem] = path
    manifest = tmp_path / "separated.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["recording_id", "speech_filepath", "music_filepath", "effects_filepath"],
        )
        writer.writeheader()
        writer.writerow({"recording_id": "real_001", **{f"{stem}_filepath": path for stem, path in paths.items()}})

    def fake_transcriber(path: Path) -> dict:
        assert path.is_file()
        return {
            "text": "hello world",
            "windows": [
                {"start": 0.0, "end": 0.4, "avg_logprob": -0.2, "no_speech_prob": 0.1},
                {"start": 0.4, "end": 1.0, "avg_logprob": -0.4, "no_speech_prob": 0.3},
            ],
        }

    report = evaluate_manifest(
        manifest,
        clap_scorer=_FakeClapScorer(),
        whisper_transcriber=fake_transcriber,
        whisper_stems=("speech",),
        sample_rate=16000,
    )

    assert report["row_count"] == 1
    assert report["clap_config"] == {
        "audio_model": "fake-clap",
        "audio_antibleed_margin": 0.02,
    }
    row = report["rows"][0]
    assert row["recording_id"] == "real_001"
    assert row["clap"]["music"]["purity_margin"] == pytest.approx(0.5)
    assert row["clap"]["speech"]["negative_similarity"] is None
    assert row["clap"]["speech"]["purity_margin"] is None
    assert row["clap"]["effects"]["window_count"] == 1
    assert row["whisper"]["speech"]["avg_logprob"] == pytest.approx(-0.32)
    assert row["whisper"]["speech"]["no_speech_probability"] == pytest.approx(0.22)
    assert row["whisper"]["speech"]["window_count"] == 2
    assert report["summary"]["clap"]["effects"]["positive_similarity"]["count"] == 1
    assert report["summary"]["clap"]["effects"]["positive_similarity"]["std"] == 0.0
    json.dumps(report, allow_nan=False)


def test_evaluator_parses_clap_audio_model_and_antibleed_margin() -> None:
    args = _parse_args(
        [
            "separated.csv",
            "--clap-audio-model",
            "HTSAT-base",
            "--clap-audio-antibleed-margin",
            "0.03",
        ]
    )

    assert args.clap_audio_model == "HTSAT-base"
    assert args.clap_audio_antibleed_margin == pytest.approx(0.03)


def test_evaluate_manifest_rejects_misaligned_separated_stems(tmp_path: Path) -> None:
    paths = {}
    for stem, length in zip(("speech", "music", "effects"), (800, 799, 800), strict=True):
        path = tmp_path / f"{stem}.wav"
        sf.write(path, np.zeros(length, dtype=np.float32), 8000, subtype="FLOAT")
        paths[stem] = path
    manifest = tmp_path / "misaligned.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["speech_filepath", "music_filepath", "effects_filepath"],
        )
        writer.writeheader()
        writer.writerow({f"{stem}_filepath": path for stem, path in paths.items()})

    with pytest.raises(ValueError, match="equal lengths"):
        evaluate_manifest(manifest, clap_scorer=_FakeClapScorer(), sample_rate=8000)


def test_whisper_metrics_use_null_instead_of_nonstandard_nan() -> None:
    metrics = _whisper_metrics({"text": "", "segments": []})

    assert metrics["avg_logprob"] is None
    assert metrics["no_speech_probability"] is None
    json.dumps(metrics, allow_nan=False)


def test_clap_complete_file_score_is_duration_weighted(tmp_path: Path) -> None:
    class _TwoWindowScorer(_FakeClapScorer):
        def semantic_window_scores(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            positive = audio.new_tensor([[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]])
            return positive, torch.zeros_like(positive)

        def window_bounds(self, n_samples: int) -> tuple[tuple[int, int], ...]:
            assert n_samples == 1200
            return ((0, 1000), (1000, 1200))

        def source_has_negative_prompt(self, source: str) -> bool:
            return True

    paths = {}
    for stem in ("speech", "music", "effects"):
        path = tmp_path / f"{stem}.wav"
        sf.write(path, np.zeros(1200, dtype=np.float32), 100, subtype="FLOAT")
        paths[stem] = path
    manifest = tmp_path / "two_windows.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["speech_filepath", "music_filepath", "effects_filepath"],
        )
        writer.writeheader()
        writer.writerow({f"{stem}_filepath": path for stem, path in paths.items()})

    report = evaluate_manifest(manifest, clap_scorer=_TwoWindowScorer(), sample_rate=100)

    assert report["rows"][0]["clap"]["music"]["positive_similarity"] == pytest.approx(1.0 / 6.0)


def test_evaluate_manifest_reports_prompt_bank_and_reference_audio_metrics(tmp_path: Path) -> None:
    row = {"recording_id": "paired_001"}
    fieldnames = ["recording_id"]
    for prefix in ("", "reference_"):
        for stem, value in zip(("speech", "music", "effects"), (0.2, 0.1, 0.0), strict=True):
            path = tmp_path / f"{prefix}{stem}.wav"
            sf.write(path, np.full(800, value, dtype=np.float32), 8000, subtype="FLOAT")
            field = f"{prefix}{stem}_filepath"
            fieldnames.append(field)
            row[field] = path
    manifest = tmp_path / "paired.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)

    report = evaluate_manifest(
        manifest,
        clap_scorer=_ReferenceAwareFakeClapScorer(),
        sample_rate=8000,
    )

    assert report["has_reference_stems"] is True
    speech = report["rows"][0]["clap"]["speech"]
    assert speech["prompt_bank_probability"] == pytest.approx(0.786986)
    assert speech["prompt_bank_margin"] == pytest.approx(2.0)
    assert speech["reference_similarity"] == pytest.approx(0.9)
    assert speech["relative_cross_stem_excess"] == pytest.approx(0.1)
    effects = report["rows"][0]["clap"]["effects"]
    assert effects["reference_similarity"] is None
    assert effects["relative_cross_stem_excess"] is None
    assert report["summary"]["clap"]["effects"]["reference_similarity"]["count"] == 0
    json.dumps(report, allow_nan=False)


def test_evaluate_manifest_reports_reference_gated_speech_leakage_metrics(tmp_path: Path) -> None:
    row = {}
    fieldnames = []
    for prefix in ("", "reference_"):
        for stem, value in zip(("speech", "music", "effects"), (0.2, 0.1, 0.0), strict=True):
            path = tmp_path / f"{prefix}{stem}.wav"
            sf.write(path, np.full(800, value, dtype=np.float32), 8000, subtype="FLOAT")
            field = f"{prefix}{stem}_filepath"
            fieldnames.append(field)
            row[field] = path
    manifest = tmp_path / "paired_leakage.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)

    report = evaluate_manifest(
        manifest,
        clap_scorer=_ReferenceAwareFakeClapScorer(),
        speech_leakage_scorer=_FakeSpeechLeakageScorer(),
        sample_rate=8000,
    )

    assert report["rows"][0]["speech_leakage"]["speech_leakage_tf_music"] == pytest.approx(0.5)
    assert report["summary"]["speech_leakage"]["speech_leakage_tf"]["mean"] == pytest.approx(0.25)
    json.dumps(report, allow_nan=False)


def test_evaluate_manifest_rejects_partial_reference_columns(tmp_path: Path) -> None:
    manifest = tmp_path / "partial_reference.csv"
    fieldnames = [
        "speech_filepath",
        "music_filepath",
        "effects_filepath",
        "reference_speech_filepath",
    ]
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({field: "missing.wav" for field in fieldnames})

    with pytest.raises(ValueError, match="partial reference stems"):
        evaluate_manifest(manifest, clap_scorer=_ReferenceAwareFakeClapScorer(), sample_rate=8000)


def test_evaluate_manifest_rejects_speech_leakage_metrics_without_references(tmp_path: Path) -> None:
    paths = {}
    for stem in ("speech", "music", "effects"):
        path = tmp_path / f"{stem}.wav"
        sf.write(path, np.zeros(800, dtype=np.float32), 8000, subtype="FLOAT")
        paths[stem] = path
    manifest = tmp_path / "unpaired.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[f"{stem}_filepath" for stem in paths])
        writer.writeheader()
        writer.writerow({f"{stem}_filepath": path for stem, path in paths.items()})

    with pytest.raises(ValueError, match="require all reference"):
        evaluate_manifest(
            manifest,
            clap_scorer=_FakeClapScorer(),
            speech_leakage_scorer=_FakeSpeechLeakageScorer(),
            sample_rate=8000,
        )
