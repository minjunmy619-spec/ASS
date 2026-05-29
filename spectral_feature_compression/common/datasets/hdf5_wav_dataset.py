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


class HDF5WavDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path: Path | str,
        duration: int | None = None,
        sr: int | None = None,
        return_ref: bool = False,
        augmenter: nn.Module | None = None,
        crop_retry: int = 16,
        source_activity_threshold: float = 0.0,
        min_active_sources: int = 1,
    ) -> None:
        super().__init__()

        self._dataset = CachedDataset(HDF5Dataset(dataset_path))
        # self._dataset = HDF5Dataset(dataset_path)

        self.duration = duration
        self.sr = sr

        self.return_ref = return_ref
        self.augmenter = augmenter
        self.crop_retry = int(crop_retry)
        self.source_activity_threshold = float(source_activity_threshold)
        self.min_active_sources = int(min_active_sources)

    def _is_active_crop(self, wav: torch.Tensor, ref: torch.Tensor | None, t_start: int, t_end: int) -> bool:
        if ref is not None:
            ref_crop = ref[..., t_start:t_end]
            source_rms = ref_crop.float().square().mean(dim=(-1, -2)).sqrt()
            return int((source_rms > self.source_activity_threshold).sum().item()) >= self.min_active_sources
        wav_crop = wav[..., t_start:t_end]
        return wav_crop.float().square().mean().sqrt().item() > self.source_activity_threshold

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int):
        item = self._dataset[index]
        wav = torch.as_tensor(item["wav"])
        ref = torch.as_tensor(item["ref"]) if self.return_ref and "ref" in item else None

        if self.duration is not None:
            duration = self.sr * self.duration

            t_start = 0
            for _ in range(max(1, self.crop_retry)):
                try:
                    t_start = np.random.randint(0, wav.shape[1] - duration + 1)
                except Exception as e:
                    print(wav.shape, duration, e, flush=True)
                t_end = t_start + duration
                if self._is_active_crop(wav, ref, t_start, t_end):
                    break

            wav = wav[:, t_start:t_end]

        if self.return_ref:
            assert ref is not None, list(item.keys())
            ref = ref if self.duration is None else ref[..., t_start:t_end]
            if self.augmenter is not None:
                ref = self.augmenter(ref)
                wav = ref.sum(dim=0)

            assert wav.shape[-1] == ref.shape[-1], (wav.shape, ref.shape)

            return wav, ref
        else:
            return wav
