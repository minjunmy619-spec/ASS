from __future__ import annotations

import torch

from spectral_feature_compression.common.datamodules.hdf5_wav_datamodule import DataModule
from spectral_feature_compression.common.datasets.waveform_augmentations import SourceSeparationAugmenter


def test_source_separation_augmenter_preserves_shape_and_changes_sources() -> None:
    torch.manual_seed(0)
    augmenter = SourceSeparationAugmenter(
        p_gain=1.0,
        gain_db_min=3.0,
        gain_db_max=3.0,
        p_polarity=1.0,
        p_channel_swap=1.0,
        p_time_shift=1.0,
        max_time_shift_samples=4,
        p_pitch_time=1.0,
        pitch_time_scale_min=0.99,
        pitch_time_scale_max=1.01,
        p_random_eq=1.0,
        eq_bands=4,
        eq_gain_db_min=-1.0,
        eq_gain_db_max=1.0,
        p_band_dropout=1.0,
        band_dropout_width=0.1,
    )
    ref = torch.randn(3, 2, 256)

    augmented = augmenter(ref)

    assert augmented.shape == ref.shape
    assert not torch.allclose(augmented, ref)
    assert torch.isfinite(augmented).all()


def test_datamodule_builds_train_augmenter_from_config() -> None:
    datamodule = DataModule(
        train_dataset_path=["speech.hdf5", "music.hdf5", "effects.hdf5"],
        val_dataset_path="cv.hdf5",
        batch_size=2,
        num_workers=0,
        duration=1,
        sr=8000,
        return_ref=True,
        use_scatter_dataset=False,
        use_dm_dataset=True,
        p_source_dropout=0.1,
        remix_sources=True,
        source_gain_db_min=-12.0,
        source_gain_db_max=12.0,
        crop_retry=8,
        source_activity_threshold=1.0e-4,
        train_augmentations={
            "p_gain": 0.5,
            "p_polarity": 0.25,
            "p_channel_swap": 0.25,
        },
    )

    kwargs = datamodule.default_train_dataset_kwargs
    assert isinstance(kwargs["augmenter"], SourceSeparationAugmenter)
    assert kwargs["p_source_dropout"] == 0.1
    assert kwargs["source_gain_db_min"] == -12.0
    assert kwargs["source_gain_db_max"] == 12.0
    assert kwargs["crop_retry"] == 8
    assert kwargs["source_activity_threshold"] == 1.0e-4
