# Copyright (c) 2026 National Institute of Advanced Industrial Science and Technology (AIST), Japan
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Any

from pathlib import Path

import torch
from torch.utils.data import DataLoader

import lightning as lt

from aiaccel.torch.datasets import scatter_dataset
from spectral_feature_compression.common.datasets.hdf5_wav_dataset import HDF5WavDataset
from spectral_feature_compression.common.datasets.hdf5_wav_dataset_dm import HDF5WavDMDataset


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
