# Copyright (c) 2026 National Institute of Advanced Industrial Science and Technology (AIST), Japan
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Any

from collections.abc import Mapping, Sequence
from pathlib import Path

from torch.utils.data import DataLoader

import lightning as lt

from spectral_feature_compression.common.datasets.on_the_fly_stem_dataset import (
    FixedStemMixDataset,
    OnTheFlyStemDataset,
    ProbabilisticInterleaveDataset,
)


class OnTheFlyStemDataModule(lt.LightningDataModule):
    """Opt-in dry on-the-fly stem mixer datamodule for 3-stem separation.

    The dataloaders return ``(wav, ref)`` batches compatible with the existing
    ``SupTask``/``TeacherStudentDistillationTask`` contract.
    """

    def __init__(
        self,
        *,
        source_pools: Mapping[str, Sequence[str | Path] | str | Path] | None = None,
        source_manifest_csv: Sequence[str | Path] | str | Path | None = None,
        fixed_mix_manifest_csv: Sequence[str | Path] | str | Path | None = None,
        supplemental_fixed_mix_manifest_csv: Sequence[str | Path] | str | Path | None = None,
        supplemental_fixed_mix_probability: float = 0.0,
        batch_size: int,
        sr: int = 44100,
        duration: float = 6.0,
        dataset_length: int = 100_000,
        val_dataset_length: int = 1_000,
        test_dataset_length: int | None = None,
        source_order: Sequence[str] = ("speech", "music", "effects"),
        val_source_pools: Mapping[str, Sequence[str | Path] | str | Path] | None = None,
        val_source_manifest_csv: Sequence[str | Path] | str | Path | None = None,
        val_fixed_mix_manifest_csv: Sequence[str | Path] | str | Path | None = None,
        test_source_pools: Mapping[str, Sequence[str | Path] | str | Path] | None = None,
        test_source_manifest_csv: Sequence[str | Path] | str | Path | None = None,
        test_fixed_mix_manifest_csv: Sequence[str | Path] | str | Path | None = None,
        train_manifest_split: str | Sequence[str] | None = None,
        val_manifest_split: str | Sequence[str] | None = None,
        test_manifest_split: str | Sequence[str] | None = None,
        synthesis: Mapping[str, Any] | None = None,
        val_synthesis: Mapping[str, Any] | None = None,
        test_synthesis: Mapping[str, Any] | None = None,
        num_workers: int = 4,
        val_batch_size: int | None = None,
        test_batch_size: int | None = None,
        train_seed: int | None = None,
        val_seed: int | None = 0,
        test_seed: int | None = 0,
        train_drop_last: bool = True,
        val_drop_last: bool = False,
        test_drop_last: bool = False,
        pin_memory: bool = False,
        persistent_workers: bool | None = None,
    ) -> None:
        super().__init__()
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if val_batch_size is not None and val_batch_size <= 0:
            raise ValueError(f"val_batch_size must be positive, got {val_batch_size}")
        if test_batch_size is not None and test_batch_size <= 0:
            raise ValueError(f"test_batch_size must be positive, got {test_batch_size}")
        if num_workers < 0:
            raise ValueError(f"num_workers must be non-negative, got {num_workers}")
        if not 0.0 <= supplemental_fixed_mix_probability <= 1.0:
            raise ValueError(
                "supplemental_fixed_mix_probability must be in [0, 1], "
                f"got {supplemental_fixed_mix_probability}"
            )
        if supplemental_fixed_mix_probability > 0.0 and supplemental_fixed_mix_manifest_csv is None:
            raise ValueError(
                "supplemental_fixed_mix_probability is positive but supplemental_fixed_mix_manifest_csv is not set"
            )

        if sum(item is not None for item in (source_pools, source_manifest_csv, fixed_mix_manifest_csv)) != 1:
            raise ValueError("Provide exactly one of source_pools, source_manifest_csv, or fixed_mix_manifest_csv")
        if (
            sum(item is not None for item in (val_source_pools, val_source_manifest_csv, val_fixed_mix_manifest_csv))
            > 1
        ):
            raise ValueError(
                "Provide only one validation source: val_source_pools, "
                "val_source_manifest_csv, or val_fixed_mix_manifest_csv"
            )
        if (
            sum(item is not None for item in (test_source_pools, test_source_manifest_csv, test_fixed_mix_manifest_csv))
            > 1
        ):
            raise ValueError(
                "Provide only one test source: test_source_pools, "
                "test_source_manifest_csv, or test_fixed_mix_manifest_csv"
            )

        self.source_pools = source_pools
        self.source_manifest_csv = source_manifest_csv
        self.fixed_mix_manifest_csv = fixed_mix_manifest_csv
        self.supplemental_fixed_mix_manifest_csv = supplemental_fixed_mix_manifest_csv
        self.supplemental_fixed_mix_probability = float(supplemental_fixed_mix_probability)
        self.val_source_pools = val_source_pools
        self.val_source_manifest_csv = val_source_manifest_csv
        self.val_fixed_mix_manifest_csv = val_fixed_mix_manifest_csv
        if (
            self.val_source_pools is None
            and self.val_source_manifest_csv is None
            and self.val_fixed_mix_manifest_csv is None
        ):
            self.val_source_pools = source_pools
            self.val_source_manifest_csv = source_manifest_csv
            self.val_fixed_mix_manifest_csv = fixed_mix_manifest_csv
        self.test_source_pools = test_source_pools
        self.test_source_manifest_csv = test_source_manifest_csv
        self.test_fixed_mix_manifest_csv = test_fixed_mix_manifest_csv
        if (
            self.test_source_pools is None
            and self.test_source_manifest_csv is None
            and self.test_fixed_mix_manifest_csv is None
            and test_manifest_split is not None
        ):
            self.test_source_manifest_csv = source_manifest_csv
            self.test_fixed_mix_manifest_csv = fixed_mix_manifest_csv
        self.train_manifest_split = train_manifest_split
        self.val_manifest_split = val_manifest_split
        self.test_manifest_split = test_manifest_split
        self.source_order = tuple(source_order)
        self.sr = int(sr)
        self.duration = float(duration)
        self.dataset_length = int(dataset_length)
        self.val_dataset_length = int(val_dataset_length)
        self.test_dataset_length = int(val_dataset_length if test_dataset_length is None else test_dataset_length)
        self.synthesis = dict(synthesis or {})
        self.val_synthesis = dict(val_synthesis or {})
        self.test_synthesis = dict(test_synthesis or {})
        self.train_seed = train_seed
        self.val_seed = val_seed
        self.test_seed = test_seed

        use_persistent_workers = (
            num_workers > 0 if persistent_workers is None else bool(persistent_workers and num_workers > 0)
        )
        self.train_dataloader_kwargs: dict[str, Any] = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=train_drop_last,
            shuffle=True,
            pin_memory=pin_memory,
            persistent_workers=use_persistent_workers,
        )
        self.val_dataloader_kwargs: dict[str, Any] = dict(
            batch_size=batch_size if val_batch_size is None else val_batch_size,
            num_workers=num_workers,
            drop_last=val_drop_last,
            shuffle=False,
            pin_memory=pin_memory,
            persistent_workers=use_persistent_workers,
        )
        self.test_dataloader_kwargs: dict[str, Any] = dict(
            batch_size=batch_size if test_batch_size is None else test_batch_size,
            num_workers=num_workers,
            drop_last=test_drop_last,
            shuffle=False,
            pin_memory=pin_memory,
            persistent_workers=use_persistent_workers,
        )

    def _common_dataset_kwargs(self, synthesis: Mapping[str, Any]) -> dict[str, Any]:
        kwargs = dict(synthesis)
        if bool(kwargs.pop("return_metadata", False)):
            raise ValueError(
                "OnTheFlyStemDataModule dataloaders must return (wav, ref) batches for the existing "
                "training tasks. Use the dataset classes directly with return_metadata=True for debugging."
            )
        kwargs["return_metadata"] = False
        mixture_duration = kwargs.pop("mixture_duration", None)
        if mixture_duration is not None:
            kwargs["duration"] = float(mixture_duration)
        else:
            kwargs.setdefault("duration", self.duration)
        kwargs.setdefault("sr", self.sr)
        kwargs.setdefault("source_order", self.source_order)
        return kwargs

    def _build_dataset(
        self,
        *,
        source_pools: Mapping[str, Sequence[str | Path] | str | Path] | None,
        source_manifest_csv: Sequence[str | Path] | str | Path | None,
        fixed_mix_manifest_csv: Sequence[str | Path] | str | Path | None,
        manifest_split: str | Sequence[str] | None,
        dataset_length: int,
        seed: int | None,
        synthesis: Mapping[str, Any],
        supplemental_fixed_mix_manifest_csv: Sequence[str | Path] | str | Path | None = None,
        supplemental_fixed_mix_probability: float = 0.0,
    ):
        kwargs = self._common_dataset_kwargs(synthesis)
        use_rendered_mixture = bool(kwargs.pop("use_rendered_mixture", True))
        fixed_mix_strict_shape = bool(kwargs.pop("fixed_mix_strict_shape", True))
        fixed_mix_max_additivity_error_db = kwargs.pop("fixed_mix_max_additivity_error_db", None)
        if fixed_mix_manifest_csv is not None:
            if supplemental_fixed_mix_manifest_csv is not None:
                raise ValueError("supplemental fixed mixes require an on-the-fly primary training dataset")
            return FixedStemMixDataset(
                fixed_mix_manifest_csv=fixed_mix_manifest_csv,
                source_order=kwargs["source_order"],
                sr=kwargs["sr"],
                duration=kwargs["duration"],
                manifest_split=manifest_split,
                use_rendered_mixture=use_rendered_mixture,
                strict_shape=fixed_mix_strict_shape,
                max_additivity_error_db=fixed_mix_max_additivity_error_db,
                return_metadata=False,
            )
        kwargs["dataset_length"] = dataset_length
        kwargs["seed"] = seed
        kwargs["source_pools"] = source_pools
        kwargs["source_manifest_csv"] = source_manifest_csv
        kwargs["manifest_split"] = manifest_split
        primary_dataset = OnTheFlyStemDataset(**kwargs)
        if supplemental_fixed_mix_manifest_csv is None or supplemental_fixed_mix_probability <= 0.0:
            return primary_dataset
        supplemental_dataset = FixedStemMixDataset(
            fixed_mix_manifest_csv=supplemental_fixed_mix_manifest_csv,
            source_order=kwargs["source_order"],
            sr=kwargs["sr"],
            duration=kwargs["duration"],
            use_rendered_mixture=use_rendered_mixture,
            strict_shape=fixed_mix_strict_shape,
            max_additivity_error_db=fixed_mix_max_additivity_error_db,
            return_metadata=False,
        )
        return ProbabilisticInterleaveDataset(
            primary_dataset,
            supplemental_dataset,
            probability=supplemental_fixed_mix_probability,
            seed=0 if seed is None else seed,
        )

    def _has_test_sources(self) -> bool:
        return (
            self.test_source_pools is not None
            or self.test_source_manifest_csv is not None
            or self.test_fixed_mix_manifest_csv is not None
        )

    def setup(self, stage: str | None = None) -> None:
        if stage not in {None, "fit", "validate", "test"}:
            raise ValueError(f"Unsupported stage for OnTheFlyStemDataModule: {stage!r}")

        if stage in {None, "fit"}:
            self.train_dataset = self._build_dataset(
                source_pools=self.source_pools,
                source_manifest_csv=self.source_manifest_csv,
                fixed_mix_manifest_csv=self.fixed_mix_manifest_csv,
                manifest_split=self.train_manifest_split,
                dataset_length=self.dataset_length,
                seed=self.train_seed,
                synthesis=self.synthesis,
                supplemental_fixed_mix_manifest_csv=self.supplemental_fixed_mix_manifest_csv,
                supplemental_fixed_mix_probability=self.supplemental_fixed_mix_probability,
            )

        if stage in {None, "fit", "validate"}:
            val_synthesis = {**self.synthesis, **self.val_synthesis}
            self.val_dataset = self._build_dataset(
                source_pools=self.val_source_pools,
                source_manifest_csv=self.val_source_manifest_csv,
                fixed_mix_manifest_csv=self.val_fixed_mix_manifest_csv,
                manifest_split=self.val_manifest_split,
                dataset_length=self.val_dataset_length,
                seed=self.val_seed,
                synthesis=val_synthesis,
            )

        if stage in {None, "test"} and self._has_test_sources():
            test_synthesis = {**self.synthesis, **self.val_synthesis, **self.test_synthesis}
            self.test_dataset = self._build_dataset(
                source_pools=self.test_source_pools,
                source_manifest_csv=self.test_source_manifest_csv,
                fixed_mix_manifest_csv=self.test_fixed_mix_manifest_csv,
                manifest_split=self.test_manifest_split,
                dataset_length=self.test_dataset_length,
                seed=self.test_seed,
                synthesis=test_synthesis,
            )
        elif stage == "test":
            raise ValueError(
                "test requested, but no test_source_pools/test_source_manifest_csv/test_manifest_split was provided"
            )

        sizes = []
        if hasattr(self, "train_dataset"):
            sizes.append(f"len(train_dataset)={len(self.train_dataset)}")
        if hasattr(self, "val_dataset"):
            sizes.append(f"len(val_dataset)={len(self.val_dataset)}")
        if hasattr(self, "test_dataset"):
            sizes.append(f"len(test_dataset)={len(self.test_dataset)}")
        print(f"Dataset size: {', '.join(sizes)}")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.train_dataloader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.val_dataloader_kwargs)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self.test_dataloader_kwargs)

    @property
    def example_batch_shapes(self) -> dict[str, tuple[int, ...]]:
        n_samples = int(round(self.sr * self.duration))
        return {
            "wav": (1, n_samples),
            "ref": (len(self.source_order), 1, n_samples),
        }
