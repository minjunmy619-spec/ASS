from __future__ import annotations

from typing import Any, cast

import csv
from pathlib import Path

import numpy as np

import torch

import pytest

import soundfile as sf

from spectral_feature_compression.common.datamodules.on_the_fly_stem_datamodule import OnTheFlyStemDataModule
from spectral_feature_compression.common.datasets.on_the_fly_stem_dataset import (
    FixedStemMixDataset,
    OnTheFlyStemDataset,
    ProbabilisticInterleaveDataset,
)
from tools.export_fixed_stem_mixes import export_fixed_stem_mixes

_STEMS = ("speech", "music", "effects")


class _IndexDataset:
    def __init__(self, prefix: str, length: int) -> None:
        self.prefix = prefix
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> str:
        return f"{self.prefix}:{index}"


def _write_wav(path: Path, value: float, *, sr: int = 8000, n_samples: int = 1600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.full(n_samples, value, dtype=np.float32), sr)


def _make_source_pools(tmp_path: Path, *, sr: int = 8000) -> dict[str, list[str]]:
    values = {"speech": 0.1, "music": 0.2, "effects": 0.3}
    pools: dict[str, list[str]] = {}
    for stem in _STEMS:
        stem_dir = tmp_path / stem
        pools[stem] = [str(stem_dir)]
        _write_wav(stem_dir / f"{stem}_a.wav", values[stem], sr=sr, n_samples=1200)
        _write_wav(stem_dir / f"{stem}_b.wav", values[stem] * 0.5, sr=sr, n_samples=1000)
    return pools


def _write_source_manifest(path: Path, rows: list[dict[str, str]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "split", "type", "filepath", "sample_rate", "channels"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def _manifest_rows(tmp_path: Path, split: str, suffix: str) -> list[dict[str, str]]:
    rows = []
    for stem in _STEMS:
        wav_path = tmp_path / stem / f"{stem}_{suffix}.wav"
        rows.append(
            {
                "filename": wav_path.name,
                "split": split,
                "type": stem,
                "filepath": str(wav_path),
                "sample_rate": "8000",
                "channels": "1",
            }
        )
    return rows


def _base_synthesis() -> dict[str, Any]:
    return {
        "backend": "dry_mix",
        "active_stem_count": {"mode": "fixed", "value": 3},
        "clips_per_active_stem": {"speech": [2, 2], "music": [1, 1], "effects": [2, 2]},
        "short_clip_policy": "concatenate",
        "same_stem_placement": {
            "mode": "sequential",
            "initial_offset_sec_range": [0.0, 0.0],
            "gap_sec_range": [0.0, 0.0],
            "overlap_sec_range": [0.0, 0.0],
            "allow_self_overlap": False,
        },
        "stem_gain_db": {"speech": [0.0, 0.0], "music": [0.0, 0.0], "effects": [0.0, 0.0]},
        "peak_norm_db": None,
        "return_metadata": True,
    }


def test_probabilistic_interleave_dataset_selects_supplemental_examples() -> None:
    primary = _IndexDataset("synthetic", 10)
    supplemental = _IndexDataset("real", 3)

    only_synthetic = ProbabilisticInterleaveDataset(primary, supplemental, probability=0.0, seed=7)
    only_real = ProbabilisticInterleaveDataset(primary, supplemental, probability=1.0, seed=7)

    assert [only_synthetic[index] for index in range(4)] == [f"synthetic:{index}" for index in range(4)]
    assert [only_real[index] for index in range(4)] == ["real:0", "real:1", "real:2", "real:0"]


def test_on_the_fly_stem_dataset_returns_fixed_duration_and_consistent_mix(tmp_path: Path) -> None:
    pools = _make_source_pools(tmp_path)
    dataset = OnTheFlyStemDataset(
        source_pools=pools,
        source_order=_STEMS,
        sr=8000,
        duration=1.0,
        dataset_length=8,
        seed=123,
        **_base_synthesis(),
    )

    wav, ref, metadata = cast(tuple[torch.Tensor, torch.Tensor, dict[str, Any]], dataset[0])

    assert tuple(wav.shape) == (1, 8000)
    assert tuple(ref.shape) == (3, 1, 8000)
    torch.testing.assert_close(wav, ref.sum(dim=0), rtol=0.0, atol=0.0)
    assert metadata["active_stems"] == list(_STEMS)
    assert len(metadata["source_paths"]["speech"]) == 2
    assert len(metadata["source_paths"]["effects"]) == 2
    assert torch.count_nonzero(ref[:, 0].abs().sum(dim=-1)).item() == 3


def test_on_the_fly_stem_dataset_samples_source_clip_duration_range(tmp_path: Path) -> None:
    long_pools = {}
    short_pools = {}
    for stem in _STEMS:
        long_path = tmp_path / "long" / stem / f"{stem}.wav"
        short_path = tmp_path / "short" / stem / f"{stem}.wav"
        _write_wav(long_path, 0.1, sr=8000, n_samples=8000)
        _write_wav(short_path, 0.1, sr=8000, n_samples=1000)
        long_pools[stem] = str(long_path)
        short_pools[stem] = str(short_path)

    long_dataset = OnTheFlyStemDataset(
        source_pools=long_pools,
        source_order=_STEMS,
        sr=8000,
        duration=0.5,
        source_clip_duration_range=[0.25, 0.25],
        dataset_length=1,
        seed=123,
        **_base_synthesis(),
    )
    short_dataset = OnTheFlyStemDataset(
        source_pools=short_pools,
        source_order=_STEMS,
        sr=8000,
        duration=0.5,
        source_clip_duration_range=[0.25, 0.25],
        dataset_length=1,
        seed=123,
        **_base_synthesis(),
    )

    long_clip, _ = long_dataset._sample_audio("speech", long_dataset._rng_for_index(0))
    short_clip, _ = short_dataset._sample_audio("speech", short_dataset._rng_for_index(0))

    assert long_clip.numel() == 2000
    assert short_clip.numel() == 1000


def test_on_the_fly_stem_dataset_supports_per_stem_source_clip_duration_ranges(tmp_path: Path) -> None:
    pools = {}
    for stem in _STEMS:
        path = tmp_path / stem / f"{stem}.wav"
        _write_wav(path, 0.1, sr=8000, n_samples=8000)
        pools[stem] = str(path)
    dataset = OnTheFlyStemDataset(
        source_pools=pools,
        source_order=_STEMS,
        sr=8000,
        duration=1.0,
        dataset_length=1,
        source_clip_duration_range={
            "speech": [0.10, 0.10],
            "music": [0.15, 0.15],
            "effects": [0.05, 0.05],
        },
        seed=0,
    )

    speech, _ = dataset._sample_audio("speech", dataset._rng_for_index(0))
    music, _ = dataset._sample_audio("music", dataset._rng_for_index(0))
    effects, _ = dataset._sample_audio("effects", dataset._rng_for_index(0))

    assert speech.numel() == 800
    assert music.numel() == 1200
    assert effects.numel() == 400


def test_on_the_fly_stem_dataset_normalizes_active_sources_before_gain(tmp_path: Path) -> None:
    pools = _make_source_pools(tmp_path)
    synthesis = _base_synthesis()
    synthesis["normalize_sources"] = True
    dataset = OnTheFlyStemDataset(
        source_pools=pools,
        source_order=_STEMS,
        sr=8000,
        duration=1.0,
        dataset_length=1,
        seed=123,
        **synthesis,
    )

    wav, ref, _ = cast(tuple[torch.Tensor, torch.Tensor, dict[str, Any]], dataset[0])
    source_rms = ref[:, 0].square().mean(dim=-1).sqrt()

    torch.testing.assert_close(source_rms, torch.ones_like(source_rms), rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(wav, ref.sum(dim=0), rtol=0.0, atol=0.0)


def test_on_the_fly_stem_dataset_retries_inactive_crops(tmp_path: Path) -> None:
    pools = _make_source_pools(tmp_path)
    path = tmp_path / "speech" / "speech_activity.wav"
    samples = np.concatenate((np.zeros(2000, dtype=np.float32), np.full(2000, 0.5, dtype=np.float32)))
    sf.write(path, samples, 8000)
    dataset = OnTheFlyStemDataset(
        source_pools=pools,
        source_order=_STEMS,
        sr=8000,
        duration=0.125,
        dataset_length=1,
        source_activity_threshold=0.1,
        crop_retry=2,
    )

    class SequenceRng:
        def __init__(self) -> None:
            self.starts = iter((0, 3000))

        def randint(self, _lo: int, _hi: int) -> int:
            return next(self.starts)

    audio = dataset._load_audio(path, SequenceRng(), max_samples=1000)  # type: ignore[arg-type]

    assert float(audio.square().mean().sqrt().item()) >= 0.49

    dataset.source_activity_threshold = 1.0

    class FallbackRng:
        def __init__(self) -> None:
            self.starts = iter((0, 1500))

        def randint(self, _lo: int, _hi: int) -> int:
            return next(self.starts)

    fallback = dataset._load_audio(path, FallbackRng(), max_samples=1000)  # type: ignore[arg-type]

    assert float(fallback.square().mean().sqrt().item()) >= 0.3


def test_on_the_fly_stem_dataset_loads_sources_from_manifest_csv(tmp_path: Path) -> None:
    _make_source_pools(tmp_path)
    for stem in _STEMS:
        _write_wav(tmp_path / stem / f"{stem}_bad.wav", 0.9, n_samples=1200)
    manifest_path = _write_source_manifest(
        tmp_path / "sources.csv",
        _manifest_rows(tmp_path, "train", "a")
        + _manifest_rows(tmp_path, "validation", "b")
        + [
            {
                "filename": "ignored.wav",
                "split": "train",
                "type": "ignored_type",
                "filepath": str(tmp_path / "ignored.wav"),
                "sample_rate": "8000",
                "channels": "1",
            }
        ],
    )
    dataset = OnTheFlyStemDataset(
        source_manifest_csv=manifest_path,
        manifest_split="train",
        source_order=_STEMS,
        sr=8000,
        duration=0.5,
        dataset_length=2,
        active_stem_count={"mode": "fixed", "value": 3},
        clips_per_active_stem=1,
        stem_gain_db={"speech": [0.0, 0.0], "music": [0.0, 0.0], "effects": [0.0, 0.0]},
        peak_norm_db=None,
        seed=123,
        return_metadata=True,
    )

    wav, ref, metadata = cast(tuple[torch.Tensor, torch.Tensor, dict[str, Any]], dataset[0])

    assert tuple(wav.shape) == (1, 4000)
    assert tuple(ref.shape) == (3, 1, 4000)
    torch.testing.assert_close(wav, ref.sum(dim=0), rtol=0.0, atol=0.0)
    for stem in _STEMS:
        assert len(metadata["source_paths"][stem]) == 1
        assert Path(metadata["source_paths"][stem][0]).name == f"{stem}_a.wav"


def test_on_the_fly_stem_dataset_pad_policy_randomly_places_one_clip(tmp_path: Path) -> None:
    pools = _make_source_pools(tmp_path)
    dataset = OnTheFlyStemDataset(
        source_pools=pools,
        source_order=_STEMS,
        sr=8000,
        duration=0.5,
        dataset_length=2,
        active_stem_count={"mode": "fixed", "value": 3},
        clips_per_active_stem={"speech": [2, 2], "music": [2, 2], "effects": [2, 2]},
        short_clip_policy="pad",
        stem_gain_db={"speech": [0.0, 0.0], "music": [0.0, 0.0], "effects": [0.0, 0.0]},
        peak_norm_db=None,
        seed=123,
        return_metadata=True,
    )

    wav, ref, metadata = cast(tuple[torch.Tensor, torch.Tensor, dict[str, Any]], dataset[0])

    assert tuple(wav.shape) == (1, 4000)
    assert tuple(ref.shape) == (3, 1, 4000)
    torch.testing.assert_close(wav, ref.sum(dim=0), rtol=0.0, atol=0.0)
    starts = []
    for stem_idx, stem in enumerate(_STEMS):
        paths = metadata["source_paths"][stem]
        assert len(paths) == 1
        expected_clip_len = 1200 if Path(paths[0]).name.endswith("_a.wav") else 1000
        active_indices = torch.nonzero(ref[stem_idx, 0], as_tuple=False).flatten()
        assert active_indices.numel() == expected_clip_len
        start = int(active_indices[0].item())
        end = int(active_indices[-1].item()) + 1
        starts.append(start)
        assert end - start == expected_clip_len
        torch.testing.assert_close(active_indices, torch.arange(start, end), rtol=0.0, atol=0.0)
        assert torch.count_nonzero(ref[stem_idx, 0, :start]) == 0
        assert torch.count_nonzero(ref[stem_idx, 0, end:]) == 0
    assert any(start > 0 for start in starts)


def test_on_the_fly_stem_dataset_pad_or_concatenate_policy_uses_configured_probability(tmp_path: Path) -> None:
    pools = _make_source_pools(tmp_path)
    dataset = OnTheFlyStemDataset(
        source_pools=pools,
        source_order=_STEMS,
        sr=8000,
        duration=0.5,
        dataset_length=2,
        active_stem_count={"mode": "fixed", "value": 3},
        clips_per_active_stem={"speech": [2, 2], "music": [2, 2], "effects": [2, 2]},
        short_clip_policy="pad_or_concatenate",
        short_clip_pad_probability={"speech": 1.0, "music": 0.0, "effects": 1.0},
        stem_gain_db={"speech": [0.0, 0.0], "music": [0.0, 0.0], "effects": [0.0, 0.0]},
        peak_norm_db=None,
        seed=123,
        return_metadata=True,
    )

    wav, ref, metadata = cast(tuple[torch.Tensor, torch.Tensor, dict[str, Any]], dataset[0])

    assert tuple(wav.shape) == (1, 4000)
    assert tuple(ref.shape) == (3, 1, 4000)
    torch.testing.assert_close(wav, ref.sum(dim=0), rtol=0.0, atol=0.0)
    assert len(metadata["source_paths"]["speech"]) == 1
    assert len(metadata["source_paths"]["music"]) == 2
    assert len(metadata["source_paths"]["effects"]) == 1


def test_on_the_fly_stem_dataset_pad_or_concatenate_rejects_bad_probability_config(tmp_path: Path) -> None:
    pools = _make_source_pools(tmp_path)
    base_kwargs = dict(
        source_pools=pools,
        source_order=_STEMS,
        sr=8000,
        duration=0.5,
        dataset_length=2,
        active_stem_count={"mode": "fixed", "value": 3},
        short_clip_policy="pad_or_concatenate",
        peak_norm_db=None,
        seed=123,
    )

    with pytest.raises(ValueError, match="unknown stem names"):
        OnTheFlyStemDataset(**base_kwargs, short_clip_pad_probability={"sfx": 0.5})

    with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
        OnTheFlyStemDataset(**base_kwargs, short_clip_pad_probability={"effects": 1.5})


def test_on_the_fly_stem_dataset_allows_zero_or_single_active_stem(tmp_path: Path) -> None:
    pools = _make_source_pools(tmp_path)
    zero_dataset = OnTheFlyStemDataset(
        source_pools=pools,
        source_order=_STEMS,
        sr=8000,
        duration=0.5,
        dataset_length=2,
        active_stem_count={"mode": "fixed", "value": 0},
        peak_norm_db=None,
        seed=1,
    )
    wav, ref = cast(tuple[torch.Tensor, torch.Tensor], zero_dataset[0])
    assert torch.count_nonzero(wav) == 0
    assert torch.count_nonzero(ref) == 0

    single_dataset = OnTheFlyStemDataset(
        source_pools=pools,
        source_order=_STEMS,
        sr=8000,
        duration=0.5,
        dataset_length=2,
        active_stem_count={"mode": "fixed", "value": 1},
        clips_per_active_stem=1,
        peak_norm_db=None,
        seed=2,
    )
    wav, ref = cast(tuple[torch.Tensor, torch.Tensor], single_dataset[0])
    active = ref[:, 0].abs().sum(dim=-1) > 0
    assert int(active.sum().item()) == 1
    torch.testing.assert_close(wav, ref.sum(dim=0), rtol=0.0, atol=0.0)


def test_on_the_fly_stem_datamodule_rejects_metadata_batches(tmp_path: Path) -> None:
    pools = _make_source_pools(tmp_path)
    datamodule = OnTheFlyStemDataModule(
        source_pools=pools,
        source_order=_STEMS,
        sr=8000,
        duration=0.5,
        dataset_length=4,
        val_dataset_length=2,
        batch_size=2,
        num_workers=0,
        synthesis={"return_metadata": True},
    )

    with pytest.raises(ValueError, match="return_metadata=True"):
        datamodule.setup("fit")


def test_on_the_fly_stem_datamodule_builds_csv_train_val_test_batches(tmp_path: Path) -> None:
    _make_source_pools(tmp_path)
    train_csv = _write_source_manifest(tmp_path / "train.csv", _manifest_rows(tmp_path, "train", "a"))
    val_csv = _write_source_manifest(tmp_path / "val.csv", _manifest_rows(tmp_path, "validation", "b"))
    test_csv = _write_source_manifest(tmp_path / "test.csv", _manifest_rows(tmp_path, "test", "b"))
    datamodule = OnTheFlyStemDataModule(
        source_manifest_csv=train_csv,
        val_source_manifest_csv=val_csv,
        test_source_manifest_csv=test_csv,
        source_order=_STEMS,
        sr=8000,
        duration=0.5,
        dataset_length=4,
        val_dataset_length=2,
        test_dataset_length=2,
        batch_size=2,
        val_batch_size=1,
        test_batch_size=1,
        num_workers=0,
        train_seed=10,
        val_seed=20,
        test_seed=30,
        train_drop_last=False,
        synthesis={
            "active_stem_count": {"mode": "fixed", "value": 3},
            "clips_per_active_stem": 1,
            "peak_norm_db": None,
        },
    )

    datamodule.setup("fit")
    wav, ref = next(iter(datamodule.train_dataloader()))
    val_wav, val_ref = next(iter(datamodule.val_dataloader()))
    datamodule.setup("test")
    test_wav, test_ref = next(iter(datamodule.test_dataloader()))

    assert tuple(wav.shape) == (2, 1, 4000)
    assert tuple(ref.shape) == (2, 3, 1, 4000)
    assert tuple(val_wav.shape) == (1, 1, 4000)
    assert tuple(val_ref.shape) == (1, 3, 1, 4000)
    assert tuple(test_wav.shape) == (1, 1, 4000)
    assert tuple(test_ref.shape) == (1, 3, 1, 4000)
    torch.testing.assert_close(wav, ref.sum(dim=1), rtol=0.0, atol=0.0)
    torch.testing.assert_close(val_wav, val_ref.sum(dim=1), rtol=0.0, atol=0.0)
    torch.testing.assert_close(test_wav, test_ref.sum(dim=1), rtol=0.0, atol=0.0)


def test_fixed_stem_mix_exporter_outputs_datamodule_manifest(tmp_path: Path) -> None:
    _make_source_pools(tmp_path)
    source_csv = _write_source_manifest(
        tmp_path / "sources.csv",
        _manifest_rows(tmp_path, "train", "a") + _manifest_rows(tmp_path, "validation", "b"),
    )
    fixed_csv = export_fixed_stem_mixes(
        output_csv=tmp_path / "fixed_validation.csv",
        output_audio_dir=tmp_path / "fixed_audio",
        output_split="validation",
        num_examples=3,
        sr=8000,
        duration=0.5,
        seed=123,
        source_order=_STEMS,
        source_manifest_csv=source_csv,
        source_manifest_split="validation",
        synthesis={
            "active_stem_count": {"mode": "fixed", "value": 3},
            "clips_per_active_stem": 1,
            "stem_gain_db": {"speech": [0.0, 0.0], "music": [0.0, 0.0], "effects": [0.0, 0.0]},
            "peak_norm_db": None,
        },
        export_mixtures=True,
    )

    with fixed_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert all(Path(row["mixture_filepath"]).is_file() for row in rows)
    assert all(Path(row[f"{stem}_filepath"]).is_file() for row in rows for stem in _STEMS)

    dataset = FixedStemMixDataset(fixed_mix_manifest_csv=fixed_csv, source_order=_STEMS, sr=8000, duration=0.5)
    wav, ref = cast(tuple[torch.Tensor, torch.Tensor], dataset[0])
    assert tuple(wav.shape) == (1, 4000)
    assert tuple(ref.shape) == (3, 1, 4000)
    torch.testing.assert_close(wav, ref.sum(dim=0), rtol=0.0, atol=0.0)

    datamodule = OnTheFlyStemDataModule(
        source_manifest_csv=source_csv,
        val_fixed_mix_manifest_csv=fixed_csv,
        source_order=_STEMS,
        sr=8000,
        duration=0.5,
        dataset_length=2,
        batch_size=1,
        val_batch_size=1,
        num_workers=0,
        synthesis={
            "active_stem_count": {"mode": "fixed", "value": 3},
            "clips_per_active_stem": 1,
            "peak_norm_db": None,
        },
    )
    datamodule.setup("fit")
    val_wav, val_ref = next(iter(datamodule.val_dataloader()))
    assert tuple(val_wav.shape) == (1, 1, 4000)
    assert tuple(val_ref.shape) == (1, 3, 1, 4000)
    torch.testing.assert_close(val_wav, val_ref.sum(dim=1), rtol=0.0, atol=0.0)

    blended_datamodule = OnTheFlyStemDataModule(
        source_manifest_csv=source_csv,
        supplemental_fixed_mix_manifest_csv=fixed_csv,
        supplemental_fixed_mix_probability=1.0,
        source_order=_STEMS,
        sr=8000,
        duration=0.5,
        dataset_length=2,
        val_dataset_length=1,
        batch_size=1,
        num_workers=0,
        synthesis={
            "active_stem_count": {"mode": "fixed", "value": 3},
            "clips_per_active_stem": 1,
            "peak_norm_db": None,
        },
    )
    blended_datamodule.setup("fit")
    assert isinstance(blended_datamodule.train_dataset, ProbabilisticInterleaveDataset)
    blended_wav, blended_ref = next(iter(blended_datamodule.train_dataloader()))
    torch.testing.assert_close(blended_wav, blended_ref.sum(dim=1), rtol=0.0, atol=0.0)


def test_fixed_stem_mix_dataset_strict_shape_rejects_mismatches(tmp_path: Path) -> None:
    _make_source_pools(tmp_path)
    source_csv = _write_source_manifest(tmp_path / "sources.csv", _manifest_rows(tmp_path, "validation", "b"))
    fixed_csv = export_fixed_stem_mixes(
        output_csv=tmp_path / "fixed_validation.csv",
        output_audio_dir=tmp_path / "fixed_audio",
        output_split="validation",
        num_examples=1,
        sr=8000,
        duration=0.5,
        seed=123,
        source_order=_STEMS,
        source_manifest_csv=source_csv,
        source_manifest_split="validation",
        synthesis={
            "active_stem_count": {"mode": "fixed", "value": 3},
            "clips_per_active_stem": 1,
            "peak_norm_db": None,
        },
        export_mixtures=True,
    )
    with fixed_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    bad_sr_csv = tmp_path / "fixed_bad_sr.csv"
    bad_sr_rows = [dict(row) for row in rows]
    bad_sr_rows[0]["sample_rate"] = "16000"
    with bad_sr_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(bad_sr_rows)
    with pytest.raises(ValueError, match="sample_rate mismatch"):
        FixedStemMixDataset(fixed_mix_manifest_csv=bad_sr_csv, source_order=_STEMS, sr=8000, duration=0.5)

    bad_len_csv = tmp_path / "fixed_bad_len.csv"
    bad_len_rows = [dict(row) for row in rows]
    bad_len_rows[0]["n_samples"] = "3999"
    with bad_len_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(bad_len_rows)
    with pytest.raises(ValueError, match="manifest shape is inconsistent"):
        FixedStemMixDataset(fixed_mix_manifest_csv=bad_len_csv, source_order=_STEMS, sr=8000, duration=0.5)

    datamodule = OnTheFlyStemDataModule(
        source_manifest_csv=source_csv,
        val_fixed_mix_manifest_csv=bad_len_csv,
        source_order=_STEMS,
        sr=8000,
        duration=0.5,
        dataset_length=1,
        batch_size=1,
        num_workers=0,
        synthesis={"active_stem_count": {"mode": "fixed", "value": 3}, "peak_norm_db": None},
    )
    with pytest.raises(ValueError, match="manifest shape is inconsistent"):
        datamodule.setup("fit")


def test_fixed_stem_mix_dataset_rejects_nonadditive_rendered_mixture(tmp_path: Path) -> None:
    n_samples = 4000
    paths: dict[str, Path] = {}
    for stem, value in zip(_STEMS, (0.1, 0.2, 0.05), strict=True):
        paths[stem] = tmp_path / f"{stem}.wav"
        sf.write(paths[stem], np.full(n_samples, value, dtype=np.float32), 8000, subtype="FLOAT")
    paths["mixture"] = tmp_path / "mixture.wav"
    sf.write(paths["mixture"], np.full(n_samples, 0.9, dtype=np.float32), 8000, subtype="FLOAT")

    manifest = tmp_path / "nonadditive.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "mixture_filepath",
                "speech_filepath",
                "music_filepath",
                "effects_filepath",
                "sample_rate",
                "n_samples",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "mixture_filepath": str(paths["mixture"]),
                **{f"{stem}_filepath": str(paths[stem]) for stem in _STEMS},
                "sample_rate": "8000",
                "n_samples": str(n_samples),
            }
        )

    dataset = FixedStemMixDataset(
        fixed_mix_manifest_csv=manifest,
        source_order=_STEMS,
        sr=8000,
        duration=0.5,
        max_additivity_error_db=-40.0,
    )
    with pytest.raises(ValueError, match="additivity error"):
        dataset[0]

    datamodule = OnTheFlyStemDataModule(
        fixed_mix_manifest_csv=manifest,
        source_order=_STEMS,
        sr=8000,
        duration=0.5,
        dataset_length=1,
        batch_size=1,
        num_workers=0,
        synthesis={"fixed_mix_max_additivity_error_db": -40.0},
    )
    datamodule.setup("fit")
    with pytest.raises(ValueError, match="additivity error"):
        datamodule.train_dataset[0]


def test_on_the_fly_stem_datamodule_builds_tuple_batches(tmp_path: Path) -> None:
    pools = _make_source_pools(tmp_path)
    datamodule = OnTheFlyStemDataModule(
        source_pools=pools,
        source_order=_STEMS,
        sr=8000,
        duration=0.5,
        dataset_length=4,
        val_dataset_length=2,
        batch_size=2,
        val_batch_size=1,
        num_workers=0,
        train_seed=10,
        val_seed=20,
        train_drop_last=False,
        synthesis={
            "active_stem_count": {"mode": "fixed", "value": 3},
            "clips_per_active_stem": 1,
            "peak_norm_db": None,
        },
    )

    datamodule.setup("fit")
    wav, ref = next(iter(datamodule.train_dataloader()))
    val_wav, val_ref = next(iter(datamodule.val_dataloader()))

    assert tuple(wav.shape) == (2, 1, 4000)
    assert tuple(ref.shape) == (2, 3, 1, 4000)
    assert tuple(val_wav.shape) == (1, 1, 4000)
    assert tuple(val_ref.shape) == (1, 3, 1, 4000)
    torch.testing.assert_close(wav, ref.sum(dim=1), rtol=0.0, atol=0.0)
    torch.testing.assert_close(val_wav, val_ref.sum(dim=1), rtol=0.0, atol=0.0)
