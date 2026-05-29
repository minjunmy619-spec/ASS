# Copyright (c) 2026 National Institute of Advanced Industrial Science and Technology (AIST), Japan
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

import torch
from torch import nn

_LOCAL_AIACCEL = Path(__file__).resolve().parents[3] / "aiaccel"
if _LOCAL_AIACCEL.is_dir() and str(_LOCAL_AIACCEL) not in sys.path:
    sys.path.insert(0, str(_LOCAL_AIACCEL))

from aiaccel.torch.datasets import CachedDataset, HDF5Dataset  # noqa: E402


class HDF5WavDMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path: list[Path] | list[str],
        duration: int | None = None,
        sr: int | None = None,
        return_ref: bool = False,
        p_source_dropout: float = 0.0,
        augmenter: nn.Module | None = None,
        remix_sources: bool = True,
        normalize_sources: bool = True,
        source_gain_db_min: float = -10.0,
        source_gain_db_max: float = 10.0,
        crop_retry: int = 16,
        source_activity_threshold: float = 0.0,
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
        self.augmenter = augmenter
        self.remix_sources = remix_sources
        self.normalize_sources = normalize_sources
        self.source_gain_db_min = float(source_gain_db_min)
        self.source_gain_db_max = float(source_gain_db_max)
        self.crop_retry = int(crop_retry)
        self.source_activity_threshold = float(source_activity_threshold)

    def _crop_source(self, x: torch.Tensor) -> torch.Tensor:
        if self.duration is None:
            return x
        T = self.duration * self.sr
        t0 = 0
        for _ in range(max(1, self.crop_retry)):
            t0 = np.random.randint(0, x.shape[-1] - T + 1)
            crop = x[..., t0 : t0 + T]
            if crop.float().square().mean().sqrt().item() > self.source_activity_threshold:
                return crop
        return x[..., t0 : t0 + T]

    def __len__(self) -> int:
        return self.min_num_data

    def __getitem__(self, index: int):
        ref_tensors = []
        for dataset_idx, dataset in enumerate(self._dataset):
            idx = np.random.randint(len(dataset)) if self.remix_sources else index % self._dataset_len[dataset_idx]
            x = torch.as_tensor(dataset[idx]["wav"])
            x = self._crop_source(x)
            ref_tensors.append(x)

        ref = torch.stack(ref_tensors, dim=0)  # (n_src, n_chan, n_samples)

        if self.p_source_dropout > 0.0:
            while True:
                drop = torch.rand(ref.size(0)) < self.p_source_dropout
                if (~drop).any():
                    break
            ref[drop] = 0

        if self.normalize_sources:
            coef = ref.pow(2).mean(dim=(-1, -2), keepdim=True).add_(1e-12).sqrt_()
            ref = ref / coef
        gain_db = torch.empty(ref.size(0), 1, 1, device=ref.device, dtype=ref.dtype).uniform_(
            self.source_gain_db_min,
            self.source_gain_db_max,
        )
        gain = 10.0 ** (gain_db / 20.0)
        ref = ref * gain

        if self.augmenter is not None:
            ref = self.augmenter(ref)

        wav = ref.sum(dim=0)
        return (wav, ref) if self.return_ref else wav
