from __future__ import annotations

import csv
from pathlib import Path
import random

import numpy as np

import torch

import pytest

import soundfile as sf

from spectral_feature_compression.common.datasets.broadcast_stem_synthesis import BroadcastStemRenderer
from spectral_feature_compression.common.datasets.on_the_fly_stem_dataset import OnTheFlyStemDataset

_STEMS = ("speech", "music", "effects")


def test_dialog_ducking_tracks_speech_activity() -> None:
    stems = torch.zeros(3, 1, 2000)
    stems[0, 0, 500:1000] = 0.5
    stems[1, 0] = 0.25
    renderer = BroadcastStemRenderer(
        sr=1000,
        source_order=_STEMS,
        config={
            "ducking": {
                "probability": 1.0,
                "speech_stem": "speech",
                "target_stems": ["music"],
                "attenuation_db": [-12.0, -12.0],
                "activity_threshold_db": -40.0,
                "frame_ms": 20.0,
                "attack_ms": 20.0,
                "release_ms": 40.0,
            }
        },
    )

    rendered, metadata = renderer.render(stems, rng=random.Random(0))

    assert rendered[1, 0, 700:900].abs().mean() < rendered[1, 0, :300].abs().mean() * 0.4
    torch.testing.assert_close(rendered[0], stems[0])
    assert metadata["ducking_applied"] is True


def test_shared_bus_dynamics_preserve_source_ratio_and_peak_limit() -> None:
    stems = torch.zeros(3, 1, 1000)
    stems[0] = 0.8
    stems[1] = 0.4
    renderer = BroadcastStemRenderer(
        sr=1000,
        source_order=_STEMS,
        config={
            "bus_compression": {
                "probability": 1.0,
                "threshold_db": [-20.0, -20.0],
                "ratio": [4.0, 4.0],
                "frame_ms": 20.0,
                "attack_ms": 10.0,
                "release_ms": 50.0,
            },
            "bus_peak_limit_db": -6.0,
        },
    )

    rendered, metadata = renderer.render(stems, rng=random.Random(0))
    mixture = rendered.sum(dim=0)

    torch.testing.assert_close(rendered[0], rendered[1] * 2.0, rtol=1e-5, atol=1e-5)
    assert float(mixture.abs().max()) <= 10.0 ** (-6.0 / 20.0) + 1e-6
    assert metadata["bus_compression_applied"] is True


def test_room_rendering_requires_dry_metadata_and_uses_shared_rir(tmp_path: Path) -> None:
    rir_path = tmp_path / "room.wav"
    sf.write(rir_path, np.array([1.0, 0.5, 0.25], dtype=np.float32), 1000)
    stems = torch.zeros(3, 1, 64)
    stems[0, 0, 8] = 1.0
    stems[2, 0, 8] = 0.5
    paths = {"speech": ["speech.wav"], "music": [], "effects": ["effects.wav"]}
    renderer = BroadcastStemRenderer(
        sr=1000,
        source_order=_STEMS,
        config={
            "room": {
                "probability": 1.0,
                "rir_paths": [str(rir_path)],
                "shared_stems": ["speech", "effects"],
                "wet_mix": [1.0, 1.0],
                "unknown_wet_policy": "assume_wet",
                "preserve_rms": False,
            }
        },
    )

    skipped, skipped_meta = renderer.render(stems, rng=random.Random(0), source_paths=paths)
    torch.testing.assert_close(skipped, stems)
    assert skipped_meta["room_applied_stems"] == []

    metadata = {
        str(Path("speech.wav")): {"is_wet": "false"},
        str(Path("effects.wav")): {"is_wet": "false"},
    }
    rendered, rendered_meta = renderer.render(
        stems,
        rng=random.Random(0),
        source_paths=paths,
        source_metadata=metadata,
    )

    assert torch.count_nonzero(rendered[0]) > torch.count_nonzero(stems[0])
    torch.testing.assert_close(rendered[0], rendered[2] * 2.0, rtol=1e-5, atol=1e-5)
    assert rendered_meta["room_applied_stems"] == ["speech", "effects"]


def test_broadcast_dataset_reads_wet_metadata_and_keeps_exact_targets(tmp_path: Path) -> None:
    rows = []
    for stem, value in zip(_STEMS, (0.2, 0.1, 0.05), strict=True):
        path = tmp_path / f"{stem}.wav"
        sf.write(path, np.full(1000, value, dtype=np.float32), 1000, subtype="FLOAT")
        rows.append({"filepath": str(path), "type": stem, "is_wet": "false"})
    manifest = tmp_path / "sources.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("filepath", "type", "is_wet"))
        writer.writeheader()
        writer.writerows(rows)
    rir_path = tmp_path / "rir.wav"
    sf.write(rir_path, np.array([1.0, 0.25], dtype=np.float32), 1000, subtype="FLOAT")
    dataset = OnTheFlyStemDataset(
        source_manifest_csv=manifest,
        source_order=_STEMS,
        sr=1000,
        duration=1.0,
        dataset_length=1,
        backend="broadcast_mix",
        active_stem_count={"mode": "fixed", "value": 3},
        clips_per_active_stem=1,
        short_clip_policy="pad",
        peak_norm_db=None,
        broadcast={
            "room": {
                "probability": 1.0,
                "rir_paths": [str(rir_path)],
                "shared_stems": ["speech", "effects"],
                "wet_mix": 1.0,
                "preserve_rms": True,
            }
        },
        seed=0,
        return_metadata=True,
    )

    mixture, references, metadata = dataset[0]

    torch.testing.assert_close(mixture, references.sum(dim=0), rtol=0.0, atol=0.0)
    assert metadata["broadcast"]["room_applied_stems"] == ["speech", "effects"]


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("source_compression", "frame_ms", -20.0),
        ("source_compression", "attack_ms", 0.0),
        ("ducking", "release_ms", -1.0),
    ],
)
def test_renderer_rejects_nonpositive_dynamics_timing(section: str, field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        BroadcastStemRenderer(
            sr=1000,
            source_order=_STEMS,
            config={section: {"probability": 1.0, field: value}},
        )


def test_renderer_rejects_zero_energy_rir(tmp_path: Path) -> None:
    rir_path = tmp_path / "silent_rir.wav"
    sf.write(rir_path, np.zeros(32, dtype=np.float32), 1000, subtype="FLOAT")
    renderer = BroadcastStemRenderer(
        sr=1000,
        source_order=_STEMS,
        config={
            "room": {
                "probability": 1.0,
                "rir_paths": [str(rir_path)],
                "shared_stems": ["speech"],
                "wet_mix": 1.0,
                "unknown_wet_policy": "assume_dry",
            }
        },
    )
    stems = torch.zeros(3, 1, 100)
    stems[0, 0, 0] = 1.0

    with pytest.raises(ValueError, match="no energy"):
        renderer.render(stems, rng=random.Random(0), source_paths={"speech": ["speech.wav"]})
