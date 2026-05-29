# Copyright (c) 2026 National Institute of Advanced Industrial Science and Technology (AIST), Japan
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Any

from collections.abc import Mapping
from pathlib import Path
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader

import lightning as lt

_LOCAL_AIACCEL = Path(__file__).resolve().parents[3] / "aiaccel"
if _LOCAL_AIACCEL.is_dir() and str(_LOCAL_AIACCEL) not in sys.path:
    sys.path.insert(0, str(_LOCAL_AIACCEL))

from aiaccel.torch.datasets import scatter_dataset  # noqa: E402

from spectral_feature_compression.common.datasets.hdf5_wav_dataset import HDF5WavDataset  # noqa: E402
from spectral_feature_compression.common.datasets.hdf5_wav_dataset_dm import HDF5WavDMDataset  # noqa: E402
from spectral_feature_compression.common.datasets.waveform_augmentations import SourceSeparationAugmenter  # noqa: E402


def _build_augmenter(config: Mapping[str, Any] | nn.Module | None) -> nn.Module | None:
    if config is None or isinstance(config, nn.Module):
        return config
    config_dict = dict(config)
    config_dict.pop("_target_", None)
    return SourceSeparationAugmenter(**config_dict)


class DataModule(lt.LightningDataModule):
    def __init__(
        self,
        train_dataset_path: str | Path,
        val_dataset_path: str | Path,
        batch_size: int,
        num_workers: int = 10,
        duration: int | None = None,
        sr: int | None = None,
        return_ref: bool = False,
        use_scatter_dataset: bool = True,
        use_dm_dataset: bool = False,
        p_source_dropout: float = 0.0,
        train_augmentations: Mapping[str, Any] | nn.Module | None = None,
        remix_sources: bool = True,
        normalize_sources: bool = True,
        source_gain_db_min: float = -10.0,
        source_gain_db_max: float = 10.0,
        crop_retry: int = 16,
        source_activity_threshold: float = 0.0,
        min_active_sources: int = 1,
        # validation configurations
        val_batch_size: int | None = None,
        val_duration: int | None = None,
        val_drop_last: bool = False,
    ):
        super().__init__()

        self.train_dataset_path = train_dataset_path
        self.val_dataset_path = val_dataset_path

        self.default_train_dataloader_kwargs: dict[str, Any] = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=True,
            drop_last=True,
            shuffle=True,
        )
        self.default_val_dataloader_kwargs: dict[str, Any] = dict(
            batch_size=batch_size if val_batch_size is None else val_batch_size,
            num_workers=num_workers,
            persistent_workers=True,
            drop_last=val_drop_last,
            shuffle=False,
            collate_fn=None,
        )
        self.default_train_dataset_kwargs: dict[str, Any] = dict(
            duration=duration,
            sr=sr,
            return_ref=return_ref,
            augmenter=_build_augmenter(train_augmentations),
            crop_retry=crop_retry,
            source_activity_threshold=source_activity_threshold,
            min_active_sources=min_active_sources,
        )
        self.default_val_dataset_kwargs: dict[str, Any] = dict(
            duration=val_duration,
            sr=sr,
            return_ref=return_ref,
        )

        self.use_scatter_dataset = use_scatter_dataset
        self.use_dm_dataset = use_dm_dataset

        if self.use_dm_dataset:
            self.default_train_dataset_kwargs["p_source_dropout"] = p_source_dropout
            self.default_train_dataset_kwargs["remix_sources"] = remix_sources
            self.default_train_dataset_kwargs["normalize_sources"] = normalize_sources
            self.default_train_dataset_kwargs["source_gain_db_min"] = source_gain_db_min
            self.default_train_dataset_kwargs["source_gain_db_max"] = source_gain_db_max
            self.default_train_dataset_kwargs.pop("min_active_sources")

    def setup(self, stage: str | None):
        train_dataset_class = HDF5WavDMDataset if self.use_dm_dataset else HDF5WavDataset
        if stage == "fit":
            if self.use_scatter_dataset and torch.cuda.device_count() > 1:
                self.train_dataset = scatter_dataset(
                    train_dataset_class(self.train_dataset_path, **self.default_train_dataset_kwargs)
                )
                self.val_dataset = scatter_dataset(
                    HDF5WavDataset(self.val_dataset_path, **self.default_val_dataset_kwargs)
                )
            else:
                self.train_dataset = train_dataset_class(self.train_dataset_path, **self.default_train_dataset_kwargs)

                if torch.cuda.device_count() > 1:
                    self.val_dataset = scatter_dataset(
                        HDF5WavDataset(self.val_dataset_path, **self.default_val_dataset_kwargs)
                    )
                else:
                    self.val_dataset = HDF5WavDataset(self.val_dataset_path, **self.default_val_dataset_kwargs)

            print(f"Dataset size: {len(self.train_dataset)=},  {len(self.val_dataset)=}")
        else:
            raise ValueError("`stage` is not 'fit'.")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            **self.default_train_dataloader_kwargs,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            **self.default_val_dataloader_kwargs,
        )
