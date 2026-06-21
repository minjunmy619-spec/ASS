import torch

import pytest

from spectral_feature_compression.common.datasets.on_the_fly_stem_dataset import OnTheFlyStemDataset


def _dataset(source_normalization):
    ds = OnTheFlyStemDataset.__new__(OnTheFlyStemDataset)
    ds.source_order = ("speech", "music", "effects")
    ds.sr = 24000
    ds.normalize_sources = True
    ds.source_normalization = source_normalization
    return ds


def test_percentile_rms_normalization_uses_active_frames_and_clamps_gain():
    ds = _dataset(
        {
            "target_rms": 1.0,
            "mode": {"speech": "percentile_rms"},
            "top_percent": {"speech": 50.0},
            "frame_ms": 100.0,
            "hop_ms": 100.0,
            "max_gain_db": {"speech": 6.0},
        }
    )
    stems = torch.zeros(3, 24000, dtype=torch.float32)
    stems[0, :12000] = 0.1  # active half; full-window RMS would be lower than active RMS.
    before = stems[0].clone()

    ds._normalize_active_sources(stems, ["speech"])

    applied_gain = float(stems[0, 0] / before[0])
    assert 1.9 < applied_gain < 2.1  # clamped by max_gain_db=6 dB, not boosted to 10x.


def test_min_rms_db_skips_boost_for_too_quiet_source():
    ds = _dataset(
        {
            "target_rms": 1.0,
            "mode": {"speech": "full_rms"},
            "min_rms_db": {"speech": -40.0},
            "max_gain_db": {"speech": 20.0},
        }
    )
    stems = torch.zeros(3, 1000, dtype=torch.float32)
    stems[0] = 1.0e-3  # -60 dB RMS, below min_rms_db=-40 dB.
    before = stems[0].clone()

    ds._normalize_active_sources(stems, ["speech"])

    assert torch.equal(stems[0], before)


def test_source_normalization_rejects_unknown_per_source_key():
    ds = _dataset({"mode": {"spech": "full_rms"}})
    with pytest.raises(ValueError, match="unknown stem"):
        ds._validate_source_normalization_config()


def test_source_normalization_rejects_unknown_scalar_field():
    ds = _dataset({"max_gian_db": 12.0})
    with pytest.raises(ValueError, match="unknown fields"):
        ds._validate_source_normalization_config()


def test_quiet_normalization_rejection_cannot_be_undone_by_random_gain():
    ds = _dataset(
        {
            "target_rms": 1.0,
            "mode": {"speech": "full_rms"},
            "min_rms_db": {"speech": -45.0},
            "max_gain_db": {"speech": 12.0},
        }
    )
    ds.stem_gain_db = {"speech": [6.0, 6.0], "music": [0.0, 0.0]}
    ds.stem_snr_db = {
        "enabled": True,
        "anchor": "speech",
        "anchor_min_rms_db": -45.0,
        "range": {"music": [0.0, 0.0]},
    }
    stems = torch.zeros(3, 1000, dtype=torch.float32)
    stems[0] = 0.004
    stems[1] = 1.0

    snr_ineligible = ds._normalize_active_sources(stems, ["speech", "music"])
    ds._apply_independent_gain(stems, ["speech", "music"], rng=None)
    ds._apply_relative_snr(stems, ["speech", "music"], rng=None, snr_ineligible_stems=snr_ineligible)

    assert "speech" in snr_ineligible
    torch.testing.assert_close(stems[1], torch.ones_like(stems[1]))


def test_relative_snr_skips_when_anchor_is_too_quiet():
    ds = _dataset({"target_rms": 1.0, "mode": "full_rms"})
    ds.stem_snr_db = {
        "enabled": True,
        "anchor": "speech",
        "anchor_min_rms_db": -40.0,
        "range": {"music": [6, 6], "effects": [0, 0]},
    }
    ds.source_order = ("speech", "music", "effects")
    stems = torch.ones(3, 1000, dtype=torch.float32)
    stems[0] = 1.0e-3  # -60 dB, below anchor_min_rms_db.
    before = stems.clone()

    ds._apply_relative_snr(stems, ["speech", "music", "effects"], rng=None)

    assert torch.equal(stems, before)


def test_relative_snr_uses_speech_anchor_after_normalization():
    ds = _dataset({"target_rms": 1.0, "mode": "full_rms"})
    ds.stem_snr_db = {"enabled": True, "anchor": "speech", "range": {"music": [6, 6], "effects": [0, 0]}}
    ds.source_order = ("speech", "music", "effects")
    stems = torch.ones(3, 1000, dtype=torch.float32)

    ds._apply_relative_snr(stems, ["speech", "music", "effects"], rng=None)

    speech_rms = stems[0].square().mean().sqrt()
    music_rms = stems[1].square().mean().sqrt()
    effects_rms = stems[2].square().mean().sqrt()
    assert torch.isclose(music_rms / speech_rms, torch.tensor(10.0 ** (-6.0 / 20.0)), atol=1e-5)
    assert torch.isclose(effects_rms / speech_rms, torch.tensor(1.0), atol=1e-5)
