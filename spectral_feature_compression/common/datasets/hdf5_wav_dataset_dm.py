# Copyright (c) 2026 National Institute of Advanced Industrial Science and Technology (AIST), Japan
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

import numpy as np

import torch

from aiaccel.torch.datasets import CachedDataset, HDF5Dataset


class HDF5WavDMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path: list[Path] | list[str],
        duration: int | None = None,
        sr: int | None = None,
        return_ref: bool = False,
        p_source_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self._dataset = []
        for path in dataset_path:
            self._dataset.append(CachedDataset(HDF5Dataset(path)))
            # self._dataset.append(HDF5Dataset(path))
        self._dataset_len = [len(d) for d in self._dataset]
        self.min_num_data = min(self._dataset_len)

        self.duration = duration
        self.sr = sr

        self.return_ref = return_ref

        self.p_source_dropout = p_source_dropout
        assert 0.0 <= self.p_source_dropout < 1.0

    def __len__(self) -> int:
        return self.min_num_data

    def __getitem__(self, index: int):
        ref_tensors = []
        for dataset in self._dataset:
            idx = np.random.randint(len(dataset))
            x = dataset[idx]["wav"]  # torch.Tensor で受ける
            if self.duration is not None:
                T = self.duration * self.sr
                t0 = np.random.randint(0, x.shape[-1] - T + 1)
                x = x[..., t0 : t0 + T]
            ref_tensors.append(x)

        ref = torch.stack(ref_tensors, dim=0)  # (n_src, n_chan, n_samples)

        if self.p_source_dropout > 0.0:
            while True:
                drop = torch.rand(ref.size(0)) < self.p_source_dropout
                if (~drop).any():
                    break
            ref[drop] = 0

        # rms normalization and gain adjustment
        coef = ref.pow(2).mean(dim=(-1, -2), keepdim=True).add_(1e-12).sqrt_()
        ref = ref / coef
        gain_db = torch.rand(ref.size(0), 1, 1) * 20.0 - 10.0
        gain = 10.0 ** (gain_db / 20.0)
        ref = ref * gain

        wav = ref.sum(dim=0)
        return (wav, ref) if self.return_ref else wav
