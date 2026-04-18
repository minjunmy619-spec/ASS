# Copyright (c) 2026 National Institute of Advanced Industrial Science and Technology (AIST), Japan
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

import numpy as np

import torch

from aiaccel.torch.datasets import CachedDataset, HDF5Dataset


class HDF5WavDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path: Path | str,
        duration: int | None = None,
        sr: int | None = None,
        return_ref: bool = False,
    ) -> None:
        super().__init__()

        self._dataset = CachedDataset(HDF5Dataset(dataset_path))
        # self._dataset = HDF5Dataset(dataset_path)

        self.duration = duration
        self.sr = sr

        self.return_ref = return_ref

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int):
        item = self._dataset[index]
        wav = item["wav"]

        if self.duration is not None:
            duration = self.sr * self.duration

            flag = True
            while flag:
                try:
                    t_start = np.random.randint(0, wav.shape[1] - duration + 1)
                except Exception as e:
                    print(wav.shape, duration, e, flush=True)
                t_end = t_start + duration
                if abs(wav[:, t_start:t_end]).sum() > 0.0:
                    flag = False

            wav = wav[:, t_start:t_end]

        if self.return_ref:
            assert "ref" in item, list(item.keys())
            ref = item["ref"]
            ref = ref if self.duration is None else ref[..., t_start:t_end]

            assert wav.shape[-1] == ref.shape[-1], (wav.shape, ref.shape)

            return wav, ref
        else:
            return wav
