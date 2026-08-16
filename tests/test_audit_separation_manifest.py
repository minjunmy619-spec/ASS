from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

import pytest

import soundfile as sf

from tools.audit_separation_manifest import audit_manifest


def _write(path: Path, audio: np.ndarray, sr: int = 1000) -> None:
    sf.write(path, audio.astype(np.float32), sr, subtype="FLOAT")


def test_audit_manifest_checks_additivity_and_frame_local_leakage(tmp_path: Path) -> None:
    speech = np.zeros(1000, dtype=np.float32)
    speech[:500] = 0.5
    music = np.full(1000, 0.1, dtype=np.float32)
    effects = np.zeros(1000, dtype=np.float32)
    mixture = speech + music + effects
    predicted_speech = speech.copy()
    predicted_speech[500:] = 0.05

    paths = {}
    for name, audio in {
        "speech": speech,
        "music": music,
        "effects": effects,
        "mixture": mixture,
        "pred_speech": predicted_speech,
    }.items():
        paths[name] = tmp_path / f"{name}.wav"
        _write(paths[name], audio)

    manifest = tmp_path / "real_multitrack.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "mixture_filepath",
                "speech_filepath",
                "music_filepath",
                "effects_filepath",
                "pred_speech_filepath",
            ],
        )
        writer.writeheader()
        writer.writerow({f"{name}_filepath": str(path) for name, path in paths.items()})

    report = audit_manifest(
        manifest,
        tier="real_multitrack",
        source_order=("speech", "music", "effects"),
        sample_rate=1000,
        frame_ms=100.0,
        silence_db=-50.0,
    )

    assert report["rows"] == 1
    assert report["additivity_error_db"]["mean"] < -100.0
    assert report["references"]["speech"]["active_fraction"]["mean"] == 0.5
    assert report["predictions"]["speech"]["inactive_leakage_db"]["mean"] > -30.0


def test_audit_manifest_requires_all_labeled_stem_columns(tmp_path: Path) -> None:
    mixture = tmp_path / "mixture.wav"
    _write(mixture, np.zeros(1000, dtype=np.float32))
    manifest = tmp_path / "missing_effects_column.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["mixture_filepath", "speech_filepath", "music_filepath"],
        )
        writer.writeheader()
        writer.writerow({"mixture_filepath": str(mixture)})

    with pytest.raises(ValueError, match="effects_filepath"):
        audit_manifest(manifest, tier="real_multitrack")


def test_audit_manifest_resamples_mixture_and_stems_consistently(tmp_path: Path) -> None:
    paths = {}
    for name, audio in {
        "speech": np.full(400, 0.2, dtype=np.float32),
        "music": np.full(400, 0.1, dtype=np.float32),
        "effects": np.zeros(400, dtype=np.float32),
        "mixture": np.full(400, 0.3, dtype=np.float32),
    }.items():
        paths[name] = tmp_path / f"{name}.wav"
        _write(paths[name], audio, sr=800)
    manifest = tmp_path / "resampled.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["mixture_filepath", "speech_filepath", "music_filepath", "effects_filepath"],
        )
        writer.writeheader()
        writer.writerow({f"{name}_filepath": str(path) for name, path in paths.items()})

    report = audit_manifest(manifest, tier="ood_synthetic", sample_rate=1000)

    assert report["rows"] == 1
    assert report["additivity_error_db"]["mean"] < -100.0
