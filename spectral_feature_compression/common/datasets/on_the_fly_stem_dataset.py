# Copyright (c) 2026 National Institute of Advanced Industrial Science and Technology (AIST), Japan
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Any

from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import torchaudio.functional as AF

import soundfile as sf

_DEFAULT_SOURCE_ORDER = ("speech", "music", "effects")
_DEFAULT_EXTENSIONS = (".wav", ".flac")


@dataclass(frozen=True)
class _PlacementConfig:
    mode: str = "random_sequential"
    gap_sec_range: tuple[float, float] = (0.0, 0.5)
    overlap_sec_range: tuple[float, float] = (0.0, 0.25)
    initial_offset_sec_range: tuple[float, float] = (0.0, 0.0)
    allow_self_overlap: bool = True
    max_self_overlap: int = 2
    placement_retry: int = 16


def _as_float_range(value: Any, *, default: tuple[float, float]) -> tuple[float, float]:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        val = float(value)
        return (val, val)
    if isinstance(value, Sequence) and len(value) == 2:
        lo, hi = float(value[0]), float(value[1])
        if hi < lo:
            raise ValueError(f"Invalid range {value}: upper bound is smaller than lower bound")
        return (lo, hi)
    raise ValueError(f"Expected scalar or 2-value range, got {value!r}")


def _as_int_range(value: Any, *, default: tuple[int, int]) -> tuple[int, int]:
    if value is None:
        return default
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, Sequence) and len(value) == 2:
        lo, hi = int(value[0]), int(value[1])
        if lo < 0 or hi < lo:
            raise ValueError(f"Invalid integer range {value}")
        return (lo, hi)
    raise ValueError(f"Expected integer or 2-value integer range, got {value!r}")


def _range_for_key(mapping: Mapping[str, Any] | None, key: str, default: tuple[float, float]) -> tuple[float, float]:
    if mapping is None:
        return default
    return _as_float_range(mapping.get(key, default), default=default)


def _as_probability(value: Any, *, name: str) -> float:
    probability = float(value)
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {probability}")
    return probability


def _probability_for_key(value: Any, key: str, default: float) -> float:
    if isinstance(value, Mapping):
        return _as_probability(value.get(key, default), name=f"probability for {key!r}")
    return _as_probability(default if value is None else value, name="probability")


def _validate_mapping_keys(mapping: Mapping[str, Any], valid_keys: Sequence[str], *, name: str) -> None:
    unknown = sorted(set(mapping) - set(valid_keys))
    if unknown:
        raise ValueError(f"{name} contains unknown stem names: {unknown}. Valid stems are: {list(valid_keys)}")


def _sample_uniform(rng: random.Random, value_range: tuple[float, float]) -> float:
    lo, hi = value_range
    if lo == hi:
        return lo
    return rng.uniform(lo, hi)


def _weighted_sample_without_replacement(
    items: Sequence[str],
    weights: Mapping[str, float] | None,
    k: int,
    rng: random.Random,
) -> list[str]:
    remaining = list(items)
    selected: list[str] = []
    for _ in range(min(k, len(remaining))):
        item_weights = [max(0.0, float(weights.get(item, 1.0) if weights is not None else 1.0)) for item in remaining]
        if sum(item_weights) <= 0.0:
            item_weights = [1.0] * len(remaining)
        chosen = rng.choices(remaining, weights=item_weights, k=1)[0]
        selected.append(chosen)
        remaining.remove(chosen)
    return selected


class OnTheFlyStemDataset(Dataset):
    """Config-driven dry 3-stem on-the-fly source-separation mixer.

    The dataset samples mono source clips from stem-specific pools, places one or
    more clips into each active stem timeline, applies gain/SNR controls, and
    returns the existing separation-task contract: ``(wav, ref)`` where
    ``wav == ref.sum(dim=0)``.

    This first backend deliberately does not add room impulse responses. Input
    WAVs are treated as already-rendered source stems, which is safe when it is
    unknown whether the files are dry or wet.
    """

    def __init__(
        self,
        *,
        source_pools: Mapping[str, Sequence[str | Path] | str | Path] | None = None,
        source_manifest_csv: Sequence[str | Path] | str | Path | None = None,
        manifest_split: str | Sequence[str] | None = None,
        manifest_filepath_column: str = "filepath",
        manifest_type_column: str = "type",
        manifest_split_column: str = "split",
        source_order: Sequence[str] = _DEFAULT_SOURCE_ORDER,
        sr: int = 44100,
        duration: float = 6.0,
        dataset_length: int = 100_000,
        backend: str = "dry_mix",
        file_extensions: Sequence[str] = _DEFAULT_EXTENSIONS,
        active_stem_count: Mapping[str, Any] | Sequence[int] | None = None,
        stem_sampling_weights: Mapping[str, float] | None = None,
        clips_per_active_stem: Mapping[str, Any] | int | Sequence[int] | None = None,
        short_clip_policy: str = "concatenate",
        short_clip_pad_probability: float | Mapping[str, float] = 0.5,
        same_stem_placement: Mapping[str, Any] | None = None,
        stem_gain_db: Mapping[str, Any] | None = None,
        stem_snr_db: Mapping[str, Any] | None = None,
        normalize_sources: bool = False,
        source_normalization: Mapping[str, Any] | None = None,
        source_activity_threshold: float = 0.0,
        crop_retry: int = 1,
        peak_norm_db: float | None = -1.0,
        peak_norm_mode: str = "scale_down",
        seed: int | None = None,
        return_metadata: bool = False,
    ) -> None:
        super().__init__()
        if backend != "dry_mix":
            raise ValueError(f"OnTheFlyStemDataset currently supports backend='dry_mix' only, got {backend!r}")
        if sr <= 0:
            raise ValueError(f"sr must be positive, got {sr}")
        if duration <= 0.0:
            raise ValueError(f"duration must be positive, got {duration}")
        if dataset_length <= 0:
            raise ValueError(f"dataset_length must be positive, got {dataset_length}")

        self.source_order = tuple(str(stem) for stem in source_order)
        if len(self.source_order) == 0:
            raise ValueError("source_order must contain at least one stem")
        if len(set(self.source_order)) != len(self.source_order):
            raise ValueError(f"source_order contains duplicates: {self.source_order}")

        self.sr = int(sr)
        self.duration = float(duration)
        self.n_samples = int(round(self.sr * self.duration))
        self.dataset_length = int(dataset_length)
        self.seed = seed
        self.return_metadata = bool(return_metadata)
        self.short_clip_policy = str(short_clip_policy)
        valid_short_clip_policies = {"pad", "loop", "concatenate", "random_place", "pad_or_concatenate"}
        if self.short_clip_policy not in valid_short_clip_policies:
            raise ValueError(
                "short_clip_policy must be one of 'pad', 'loop', 'concatenate', 'random_place', "
                f"or 'pad_or_concatenate', got {short_clip_policy!r}"
            )
        self.short_clip_pad_probability = short_clip_pad_probability
        if self.short_clip_policy == "pad_or_concatenate":
            if isinstance(self.short_clip_pad_probability, Mapping):
                _validate_mapping_keys(
                    self.short_clip_pad_probability,
                    self.source_order,
                    name="short_clip_pad_probability",
                )
            for stem in self.source_order:
                _probability_for_key(self.short_clip_pad_probability, stem, 0.5)
        self.peak_norm_db = None if peak_norm_db is None else float(peak_norm_db)
        self.peak_norm_mode = str(peak_norm_mode)
        if self.peak_norm_mode not in {"scale_down", "normalize"}:
            raise ValueError(f"peak_norm_mode must be 'scale_down' or 'normalize', got {peak_norm_mode!r}")
        self.normalize_sources = bool(normalize_sources)
        self.source_normalization = dict(source_normalization or {})
        self.source_activity_threshold = float(source_activity_threshold)
        self.crop_retry = int(crop_retry)
        if self.source_activity_threshold < 0.0:
            raise ValueError(
                f"source_activity_threshold must be non-negative, got {source_activity_threshold}"
            )
        if self.crop_retry <= 0:
            raise ValueError(f"crop_retry must be positive, got {crop_retry}")

        extensions = tuple(
            ext.lower() if str(ext).startswith(".") else f".{str(ext).lower()}" for ext in file_extensions
        )
        if (source_pools is None) == (source_manifest_csv is None):
            raise ValueError("Provide exactly one of source_pools or source_manifest_csv")
        if source_manifest_csv is not None:
            self.source_files = self._load_source_manifest(
                source_manifest_csv,
                manifest_split=manifest_split,
                filepath_column=manifest_filepath_column,
                type_column=manifest_type_column,
                split_column=manifest_split_column,
                extensions=extensions,
            )
        else:
            if source_pools is None:
                raise RuntimeError("source_pools unexpectedly missing after source config validation")
            self.source_files = self._scan_source_pools(source_pools, extensions)
        self.active_stem_count = active_stem_count or {"mode": "weighted", "weights": {1: 0.2, 2: 0.35, 3: 0.45}}
        self.stem_sampling_weights = dict(stem_sampling_weights or {})
        self.clips_per_active_stem = clips_per_active_stem or {stem: 1 for stem in self.source_order}
        self.placement = self._parse_placement(same_stem_placement)
        self.stem_gain_db = dict(stem_gain_db or {})
        self.stem_snr_db = dict(stem_snr_db or {})
        self._validate_source_normalization_config()

    def _scan_source_pools(
        self,
        source_pools: Mapping[str, Sequence[str | Path] | str | Path],
        extensions: tuple[str, ...],
    ) -> dict[str, list[Path]]:
        source_files: dict[str, list[Path]] = {}
        missing = sorted(set(self.source_order) - set(source_pools))
        if missing:
            raise ValueError(f"source_pools is missing entries for stems: {missing}")
        for stem in self.source_order:
            files: list[Path] = []
            raw_roots = source_pools[stem]
            if isinstance(raw_roots, (str, Path)):
                raw_roots = [raw_roots]
            for raw_root in raw_roots:
                root = Path(raw_root).expanduser()
                if not root.exists():
                    raise FileNotFoundError(f"Source pool for stem '{stem}' does not exist: {root}")
                if root.is_file():
                    if root.suffix.lower() in extensions:
                        files.append(root)
                    continue
                for path in root.rglob("*"):
                    if path.is_file() and path.suffix.lower() in extensions:
                        files.append(path)
            files = sorted(set(files))
            if not files:
                raise ValueError(f"No audio files found for stem '{stem}' in {source_pools[stem]}")
            source_files[stem] = files
        return source_files

    def _load_source_manifest(
        self,
        source_manifest_csv: Sequence[str | Path] | str | Path,
        *,
        manifest_split: str | Sequence[str] | None,
        filepath_column: str,
        type_column: str,
        split_column: str,
        extensions: tuple[str, ...],
    ) -> dict[str, list[Path]]:
        manifest_paths = [source_manifest_csv] if isinstance(source_manifest_csv, (str, Path)) else source_manifest_csv
        split_filter: set[str] | None
        if manifest_split is None:
            split_filter = None
        elif isinstance(manifest_split, str):
            split_filter = {manifest_split}
        else:
            split_filter = {str(split) for split in manifest_split}

        source_files: dict[str, list[Path]] = {stem: [] for stem in self.source_order}
        for raw_manifest_path in manifest_paths:
            manifest_path = Path(raw_manifest_path).expanduser()
            if not manifest_path.exists():
                raise FileNotFoundError(f"Source manifest CSV does not exist: {manifest_path}")
            with manifest_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fieldnames = set(reader.fieldnames or [])
                required_columns = {filepath_column, type_column}
                if split_filter is not None:
                    required_columns.add(split_column)
                missing_columns = sorted(required_columns - fieldnames)
                if missing_columns:
                    raise ValueError(f"Source manifest {manifest_path} is missing columns: {missing_columns}")

                for row_idx, row in enumerate(reader, start=2):
                    if split_filter is not None and str(row.get(split_column, "")).strip() not in split_filter:
                        continue
                    stem = str(row.get(type_column, "")).strip()
                    if stem not in source_files:
                        continue
                    raw_filepath = str(row.get(filepath_column, "")).strip()
                    if not raw_filepath:
                        raise ValueError(f"Empty {filepath_column!r} in {manifest_path} row {row_idx}")
                    path = Path(raw_filepath).expanduser()
                    if not path.is_absolute():
                        path = manifest_path.parent / path
                    if path.suffix.lower() not in extensions:
                        continue
                    if not path.is_file():
                        raise FileNotFoundError(f"Manifest row points to a missing audio file: {path}")
                    source_files[stem].append(path)

        for stem, files in source_files.items():
            unique_files = sorted(set(files))
            if not unique_files:
                split_msg = "" if split_filter is None else f" for split(s) {sorted(split_filter)}"
                raise ValueError(f"No manifest audio files found for stem '{stem}'{split_msg}")
            source_files[stem] = unique_files
        return source_files

    def _parse_placement(self, config: Mapping[str, Any] | None) -> _PlacementConfig:
        cfg = dict(config or {})
        return _PlacementConfig(
            mode=str(cfg.get("mode", "random_sequential")),
            gap_sec_range=_as_float_range(cfg.get("gap_sec_range"), default=(0.0, 0.5)),
            overlap_sec_range=_as_float_range(cfg.get("overlap_sec_range"), default=(0.0, 0.25)),
            initial_offset_sec_range=_as_float_range(cfg.get("initial_offset_sec_range"), default=(0.0, 0.0)),
            allow_self_overlap=bool(cfg.get("allow_self_overlap", True)),
            max_self_overlap=max(1, int(cfg.get("max_self_overlap", 2))),
            placement_retry=max(1, int(cfg.get("placement_retry", 16))),
        )

    def __len__(self) -> int:
        return self.dataset_length

    def _rng_for_index(self, index: int) -> random.Random:
        if self.seed is None:
            return random.Random()
        return random.Random(int(self.seed) + int(index))

    def _sample_active_count(self, rng: random.Random) -> int:
        cfg = self.active_stem_count
        n_stems = len(self.source_order)
        if isinstance(cfg, Mapping):
            mode = str(cfg.get("mode", "weighted"))
            if mode == "weighted":
                weights = cfg.get("weights", {1: 1.0})
                if not isinstance(weights, Mapping):
                    raise ValueError("active_stem_count.weights must be a mapping")
                keys = [int(k) for k in weights]
                vals = [max(0.0, float(v)) for v in weights.values()]
                if not keys or sum(vals) <= 0.0:
                    raise ValueError("active_stem_count.weights must contain at least one positive weight")
                return max(0, min(n_stems, rng.choices(keys, weights=vals, k=1)[0]))
            if mode == "range":
                lo, hi = _as_int_range(cfg.get("range", [1, n_stems]), default=(1, n_stems))
                return max(0, min(n_stems, rng.randint(lo, hi)))
            if mode == "fixed":
                return max(0, min(n_stems, int(cfg.get("value", n_stems))))
            raise ValueError(f"Unsupported active_stem_count mode: {mode!r}")
        if isinstance(cfg, Sequence):
            values = [int(v) for v in cfg]
            if not values:
                raise ValueError("active_stem_count sequence must not be empty")
            return max(0, min(n_stems, rng.choice(values)))
        return max(0, min(n_stems, int(cfg)))

    def _sample_clip_count(self, stem: str, rng: random.Random) -> int:
        cfg = self.clips_per_active_stem
        value = cfg.get(stem, 1) if isinstance(cfg, Mapping) else cfg
        lo, hi = _as_int_range(value, default=(1, 1))
        return max(1, rng.randint(lo, hi))

    def _load_audio(self, path: Path, rng: random.Random, *, max_samples: int) -> torch.Tensor:
        info = sf.info(path)
        source_sr = int(info.samplerate)
        if source_sr <= 0 or info.frames <= 0:
            raise ValueError(f"Invalid audio file: {path}")

        # Read only a needed crop when possible.  If resampling is needed, read a
        # proportional crop and trim/pad after resampling.
        read_frames = min(info.frames, max(1, int(math.ceil(max_samples * source_sr / self.sr))))
        attempts = self.crop_retry if self.source_activity_threshold > 0.0 and info.frames > read_frames else 1
        best_audio: torch.Tensor | None = None
        best_rms = -1.0
        for _ in range(attempts):
            start = rng.randint(0, info.frames - read_frames) if info.frames > read_frames else 0
            audio_np, _ = sf.read(path, start=start, frames=read_frames, always_2d=True, dtype="float32")
            if audio_np.size == 0:
                continue
            audio = torch.from_numpy(audio_np.T.copy()).float().mean(dim=0, keepdim=True)
            if source_sr != self.sr:
                audio = AF.resample(audio, orig_freq=source_sr, new_freq=self.sr)
            if audio.shape[-1] > max_samples:
                audio = audio[..., :max_samples]
            audio = audio.squeeze(0)
            rms = float(audio.square().mean().sqrt().item()) if audio.numel() > 0 else 0.0
            if rms > best_rms:
                best_audio = audio
                best_rms = rms
            if rms >= self.source_activity_threshold:
                return audio

        if best_audio is None:
            raise ValueError(f"Could not read audio from {path}")
        return best_audio

    def _sample_audio(self, stem: str, rng: random.Random) -> tuple[torch.Tensor, Path]:
        path = rng.choice(self.source_files[stem])
        audio = self._load_audio(path, rng, max_samples=self.n_samples)
        return audio, path

    def _try_add_clip(
        self,
        stem: torch.Tensor,
        occupancy: torch.Tensor,
        clip: torch.Tensor,
        start: int,
    ) -> bool:
        if start >= self.n_samples:
            return False
        start = max(0, int(start))
        length = min(int(clip.numel()), self.n_samples - start)
        if length <= 0:
            return False
        end = start + length
        if not self.placement.allow_self_overlap and torch.any(occupancy[start:end] > 0):
            return False
        if self.placement.max_self_overlap > 0 and torch.any(occupancy[start:end] >= self.placement.max_self_overlap):
            return False
        stem[start:end] += clip[:length]
        occupancy[start:end] += 1
        return True

    def _place_random(self, clips: list[torch.Tensor], rng: random.Random) -> torch.Tensor:
        stem = torch.zeros(self.n_samples, dtype=torch.float32)
        occupancy = torch.zeros(self.n_samples, dtype=torch.int16)
        for clip in clips:
            max_start = max(0, self.n_samples - min(int(clip.numel()), self.n_samples))
            placed = False
            for _ in range(self.placement.placement_retry):
                start = rng.randint(0, max_start) if max_start > 0 else 0
                if self._try_add_clip(stem, occupancy, clip, start):
                    placed = True
                    break
            if not placed:
                self._try_add_clip(stem, occupancy, clip, 0)
        return stem

    def _place_sequential(self, clips: list[torch.Tensor], rng: random.Random) -> torch.Tensor:
        stem = torch.zeros(self.n_samples, dtype=torch.float32)
        occupancy = torch.zeros(self.n_samples, dtype=torch.int16)
        current_sec = _sample_uniform(rng, self.placement.initial_offset_sec_range)
        for clip in clips:
            start = int(round(current_sec * self.sr))
            if start >= self.n_samples:
                break
            if not self._try_add_clip(stem, occupancy, clip, start):
                # Fall back to no-overlap placement when a configured overlap
                # would exceed max_self_overlap.
                start = (
                    int(torch.nonzero(occupancy == 0, as_tuple=False).flatten()[0].item())
                    if torch.any(occupancy == 0)
                    else self.n_samples
                )
                if not self._try_add_clip(stem, occupancy, clip, start):
                    break
            clip_sec = min(int(clip.numel()), self.n_samples - start) / float(self.sr)
            gap = _sample_uniform(rng, self.placement.gap_sec_range)
            overlap = (
                _sample_uniform(rng, self.placement.overlap_sec_range) if self.placement.allow_self_overlap else 0.0
            )
            current_sec = max(0.0, current_sec + clip_sec + gap - overlap)
        return stem

    def _build_padded_stem(self, stem_name: str, rng: random.Random) -> tuple[torch.Tensor, list[str]]:
        clip, path = self._sample_audio(stem_name, rng)
        stem = torch.zeros(self.n_samples, dtype=torch.float32)
        length = min(int(clip.numel()), self.n_samples)
        if length > 0:
            max_start = self.n_samples - length
            start = rng.randint(0, max_start) if max_start > 0 else 0
            stem[start : start + length] = clip[:length].float()
        return stem, [str(path)]

    def _build_stem(self, stem_name: str, rng: random.Random) -> tuple[torch.Tensor, list[str]]:
        if self.short_clip_policy == "pad":
            return self._build_padded_stem(stem_name, rng)
        if self.short_clip_policy == "pad_or_concatenate":
            pad_probability = _probability_for_key(self.short_clip_pad_probability, stem_name, 0.5)
            if rng.random() < pad_probability:
                return self._build_padded_stem(stem_name, rng)
        if self.short_clip_policy == "loop":
            clip, path = self._sample_audio(stem_name, rng)
            if clip.numel() == 0:
                return torch.zeros(self.n_samples, dtype=torch.float32), [str(path)]
            repeats = math.ceil(self.n_samples / int(clip.numel()))
            return clip.repeat(repeats)[: self.n_samples].float(), [str(path)]

        n_clips = self._sample_clip_count(stem_name, rng)
        clips = []
        paths = []
        for _ in range(n_clips):
            clip, path = self._sample_audio(stem_name, rng)
            clips.append(clip.float())
            paths.append(str(path))
        if self.short_clip_policy == "random_place" or self.placement.mode == "random":
            return self._place_random(clips, rng), paths
        if self.placement.mode not in {"random_sequential", "sequential"}:
            raise ValueError(f"Unsupported same_stem_placement mode: {self.placement.mode!r}")
        return self._place_sequential(clips, rng), paths

    def _apply_independent_gain(self, stems: torch.Tensor, active_stems: list[str], rng: random.Random) -> None:
        for stem_idx, stem_name in enumerate(self.source_order):
            if stem_name not in active_stems:
                continue
            gain_range = _range_for_key(self.stem_gain_db, stem_name, (0.0, 0.0))
            gain_db = _sample_uniform(rng, gain_range)
            stems[stem_idx] *= float(10.0 ** (gain_db / 20.0))

    def _normalization_value_for_key(self, key: str, stem_name: str, default: Any) -> Any:
        value = self.source_normalization.get(key, default)
        if isinstance(value, Mapping):
            return value.get(stem_name, default)
        return value

    def _validate_source_normalization_config(self) -> None:
        if not self.source_normalization:
            return
        valid_mapping_fields = {
            "mode",
            "target_rms",
            "frame_ms",
            "hop_ms",
            "activity_threshold_db",
            "top_percent",
            "max_gain_db",
            "min_gain_db",
            "min_rms_db",
        }
        for field, value in self.source_normalization.items():
            if isinstance(value, Mapping):
                if field not in valid_mapping_fields:
                    raise ValueError(
                        f"source_normalization.{field} is a per-source mapping, but only "
                        f"{sorted(valid_mapping_fields)} support per-source mappings"
                    )
                _validate_mapping_keys(value, self.source_order, name=f"source_normalization.{field}")

        valid_modes = {"full_rms", "active_rms", "percentile_rms", "none"}
        mode_cfg = self.source_normalization.get("mode", "full_rms")
        modes = mode_cfg.values() if isinstance(mode_cfg, Mapping) else [mode_cfg]
        for mode in modes:
            if str(mode) not in valid_modes:
                raise ValueError(f"source_normalization.mode must be one of {sorted(valid_modes)}, got {mode!r}")

        for stem_name in self.source_order:
            target_rms = float(self._normalization_value_for_key("target_rms", stem_name, 1.0))
            if target_rms <= 0.0:
                raise ValueError(f"source_normalization.target_rms for {stem_name!r} must be positive")
            frame_ms = float(self._normalization_value_for_key("frame_ms", stem_name, 40.0))
            hop_ms = float(self._normalization_value_for_key("hop_ms", stem_name, 20.0))
            if frame_ms <= 0.0 or hop_ms <= 0.0:
                raise ValueError(f"source_normalization frame_ms/hop_ms for {stem_name!r} must be positive")
            top_percent = float(self._normalization_value_for_key("top_percent", stem_name, 50.0))
            if not 0.0 < top_percent <= 100.0:
                raise ValueError(f"source_normalization.top_percent for {stem_name!r} must be in (0, 100]")

    def _frame_rms(self, stem: torch.Tensor, frame_size: int, hop_size: int) -> torch.Tensor:
        if stem.numel() < frame_size:
            padded = F.pad(stem, (0, frame_size - stem.numel()))
            frames = padded.unfold(0, frame_size, hop_size)
        else:
            frames = stem.unfold(0, frame_size, hop_size)
            tail_start = max(0, stem.numel() - frame_size)
            if frames.shape[0] == 0 or tail_start > (frames.shape[0] - 1) * hop_size:
                frames = torch.cat([frames, stem[tail_start:].unfold(0, frame_size, hop_size)], dim=0)
        return frames.float().square().mean(dim=-1).sqrt()

    def _source_rms_for_normalization(self, stem: torch.Tensor, stem_name: str) -> torch.Tensor:
        mode = str(self._normalization_value_for_key("mode", stem_name, "full_rms"))
        eps = stem.new_tensor(1.0e-8)
        if mode == "none":
            return stem.new_zeros(())
        if mode == "full_rms":
            return stem.float().square().mean().sqrt()

        frame_ms = float(self._normalization_value_for_key("frame_ms", stem_name, 40.0))
        hop_ms = float(self._normalization_value_for_key("hop_ms", stem_name, 20.0))
        frame_size = max(1, int(round(frame_ms * self.sr / 1000.0)))
        hop_size = max(1, int(round(hop_ms * self.sr / 1000.0)))
        frame_rms = self._frame_rms(stem, frame_size=frame_size, hop_size=hop_size)
        if frame_rms.numel() == 0:
            return stem.float().square().mean().sqrt()

        if mode == "active_rms":
            threshold_db = float(self._normalization_value_for_key("activity_threshold_db", stem_name, -45.0))
            threshold = float(10.0 ** (threshold_db / 20.0))
            selected = frame_rms[frame_rms >= threshold]
            if selected.numel() == 0:
                selected = frame_rms.topk(k=1).values
            return selected.square().mean().sqrt()

        if mode == "percentile_rms":
            top_percent = float(self._normalization_value_for_key("top_percent", stem_name, 50.0))
            if not 0.0 < top_percent <= 100.0:
                raise ValueError(f"source_normalization.top_percent for {stem_name!r} must be in (0, 100]")
            k = max(1, int(math.ceil(frame_rms.numel() * top_percent / 100.0)))
            selected = frame_rms.topk(k=k, largest=True).values
            return selected.square().mean().sqrt()

        raise ValueError(f"Unsupported source normalization mode: {mode!r}")

    def _normalize_active_sources(self, stems: torch.Tensor, active_stems: list[str]) -> None:
        if not self.normalize_sources:
            return
        for stem_idx, stem_name in enumerate(self.source_order):
            if stem_name not in active_stems:
                continue
            rms = self._source_rms_for_normalization(stems[stem_idx], stem_name)
            if float(rms.item()) <= 0.0:
                continue
            rms_value = float(rms.clamp_min(1.0e-8).item())
            target_rms = float(self._normalization_value_for_key("target_rms", stem_name, 1.0))
            gain = target_rms / rms_value

            # If the measured normalization RMS is extremely low, the source is
            # often mostly silence or pseudo-label residue.  In that case do not
            # boost it toward target_rms; otherwise low-level vocal-gap noise can
            # become a strong supervised target.  Attenuation is still allowed.
            min_rms_db = self._normalization_value_for_key("min_rms_db", stem_name, None)
            if min_rms_db is not None:
                min_rms = float(10.0 ** (float(min_rms_db) / 20.0))
                if rms_value < min_rms and gain > 1.0:
                    continue

            max_gain_db = self._normalization_value_for_key("max_gain_db", stem_name, None)
            if max_gain_db is not None:
                gain = min(gain, float(10.0 ** (float(max_gain_db) / 20.0)))
            min_gain_db = self._normalization_value_for_key("min_gain_db", stem_name, None)
            if min_gain_db is not None:
                gain = max(gain, float(10.0 ** (float(min_gain_db) / 20.0)))
            stems[stem_idx] *= gain

    def _apply_relative_snr(self, stems: torch.Tensor, active_stems: list[str], rng: random.Random) -> None:
        cfg = self.stem_snr_db
        if not cfg or not bool(cfg.get("enabled", False)) or len(active_stems) < 2:
            return
        anchor_cfg = str(cfg.get("anchor", "random_active"))
        if anchor_cfg == "random_active" or anchor_cfg not in active_stems:
            anchor_name = rng.choice(active_stems)
        else:
            anchor_name = anchor_cfg
        anchor_idx = self.source_order.index(anchor_name)
        anchor_rms_raw = stems[anchor_idx].square().mean().sqrt()
        anchor_min_rms_db = cfg.get("anchor_min_rms_db", None)
        if anchor_min_rms_db is not None:
            anchor_min_rms = float(10.0 ** (float(anchor_min_rms_db) / 20.0))
            if float(anchor_rms_raw.item()) < anchor_min_rms:
                return
        anchor_rms = anchor_rms_raw.clamp_min(1.0e-8)
        ranges = cfg.get("range", {})
        if not isinstance(ranges, Mapping):
            raise ValueError("stem_snr_db.range must be a mapping from stem name to dB range")
        for stem_name in active_stems:
            if stem_name == anchor_name:
                continue
            stem_idx = self.source_order.index(stem_name)
            stem_rms = stems[stem_idx].square().mean().sqrt().clamp_min(1.0e-8)
            snr_db = _sample_uniform(rng, _range_for_key(ranges, stem_name, (0.0, 0.0)))
            # Positive SNR means the anchor is louder than this stem.
            target_rms = anchor_rms * float(10.0 ** (-snr_db / 20.0))
            stems[stem_idx] *= target_rms / stem_rms

    def _apply_peak_norm(self, stems: torch.Tensor) -> torch.Tensor:
        if self.peak_norm_db is None:
            return stems
        mixture = stems.sum(dim=0)
        peak = mixture.abs().max()
        if float(peak.item()) <= 0.0:
            return stems
        target = float(10.0 ** (self.peak_norm_db / 20.0))
        if self.peak_norm_mode == "scale_down" and float(peak.item()) <= target:
            return stems
        return stems * (target / peak)

    def __getitem__(self, index: int):
        rng = self._rng_for_index(index)
        active_count = self._sample_active_count(rng)
        active_stems = _weighted_sample_without_replacement(
            self.source_order,
            self.stem_sampling_weights,
            active_count,
            rng,
        )

        stems = torch.zeros(len(self.source_order), self.n_samples, dtype=torch.float32)
        metadata: dict[str, Any] = {
            "active_stems": list(active_stems),
            "source_paths": {stem: [] for stem in self.source_order},
        }
        for stem_name in active_stems:
            stem_idx = self.source_order.index(stem_name)
            stem_audio, paths = self._build_stem(stem_name, rng)
            stems[stem_idx] = stem_audio
            metadata["source_paths"][stem_name] = paths

        self._normalize_active_sources(stems, active_stems)
        self._apply_independent_gain(stems, active_stems, rng)
        self._apply_relative_snr(stems, active_stems, rng)
        stems = self._apply_peak_norm(stems)

        ref = stems[:, None, :].contiguous()
        wav = ref.sum(dim=0).contiguous()
        if self.return_metadata:
            metadata["index"] = int(index)
            return wav, ref, metadata
        return wav, ref


class FixedStemMixDataset(Dataset):
    """Replay fixed, pre-rendered stem-mixture examples from a CSV manifest.

    The manifest is intentionally wide: one row is one mixture.  It must contain
    one ``{stem}_filepath`` column per configured stem in ``source_order``.  Empty
    stem file paths are treated as inactive/silence.  If ``mixture_filepath`` is
    present and ``use_rendered_mixture`` is true, the input wav is loaded from
    disk; otherwise it is reconstructed as ``ref.sum(dim=0)``.
    """

    def __init__(
        self,
        *,
        fixed_mix_manifest_csv: Sequence[str | Path] | str | Path,
        source_order: Sequence[str] = _DEFAULT_SOURCE_ORDER,
        sr: int | None = None,
        duration: float | None = None,
        manifest_split: str | Sequence[str] | None = None,
        manifest_split_column: str = "split",
        mixture_filepath_column: str = "mixture_filepath",
        use_rendered_mixture: bool = True,
        strict_shape: bool = True,
        return_metadata: bool = False,
    ) -> None:
        super().__init__()
        self.source_order = tuple(str(stem) for stem in source_order)
        if len(self.source_order) == 0:
            raise ValueError("source_order must contain at least one stem")
        if len(set(self.source_order)) != len(self.source_order):
            raise ValueError(f"source_order contains duplicates: {self.source_order}")
        if sr is not None and sr <= 0:
            raise ValueError(f"sr must be positive, got {sr}")
        if duration is not None and duration <= 0.0:
            raise ValueError(f"duration must be positive, got {duration}")
        self.sr = None if sr is None else int(sr)
        self.duration = None if duration is None else float(duration)
        self.manifest_split_column = str(manifest_split_column)
        self.mixture_filepath_column = str(mixture_filepath_column)
        self.use_rendered_mixture = bool(use_rendered_mixture)
        self.strict_shape = bool(strict_shape)
        self.return_metadata = bool(return_metadata)
        self.rows = self._load_fixed_manifest(fixed_mix_manifest_csv, manifest_split=manifest_split)
        if self.strict_shape:
            for row in self.rows:
                self._target_sr_and_samples(row)

    def _load_fixed_manifest(
        self,
        fixed_mix_manifest_csv: Sequence[str | Path] | str | Path,
        *,
        manifest_split: str | Sequence[str] | None,
    ) -> list[dict[str, str]]:
        manifest_paths = (
            [fixed_mix_manifest_csv] if isinstance(fixed_mix_manifest_csv, (str, Path)) else fixed_mix_manifest_csv
        )
        if manifest_split is None:
            split_filter = None
        elif isinstance(manifest_split, str):
            split_filter = {manifest_split}
        else:
            split_filter = {str(split) for split in manifest_split}

        rows: list[dict[str, str]] = []
        required_columns = {f"{stem}_filepath" for stem in self.source_order}
        for raw_manifest_path in manifest_paths:
            manifest_path = Path(raw_manifest_path).expanduser()
            if not manifest_path.exists():
                raise FileNotFoundError(f"Fixed mixture manifest CSV does not exist: {manifest_path}")
            with manifest_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fieldnames = set(reader.fieldnames or [])
                missing_columns = sorted(required_columns - fieldnames)
                if split_filter is not None and self.manifest_split_column not in fieldnames:
                    missing_columns.append(self.manifest_split_column)
                if missing_columns:
                    raise ValueError(f"Fixed mixture manifest {manifest_path} is missing columns: {missing_columns}")
                for row in reader:
                    if (
                        split_filter is not None
                        and str(row.get(self.manifest_split_column, "")).strip() not in split_filter
                    ):
                        continue
                    row = dict(row)
                    row["__manifest_dir"] = str(manifest_path.parent)
                    rows.append(row)
        if not rows:
            split_msg = "" if split_filter is None else f" for split(s) {sorted(split_filter)}"
            raise ValueError(f"No fixed mixture rows found{split_msg}")
        return rows

    def __len__(self) -> int:
        return len(self.rows)

    def _target_sr_and_samples(self, row: Mapping[str, str]) -> tuple[int, int]:
        row_label = str(row.get("mixture_id", row.get("index", "<unknown>")))
        raw_sr = str(row.get("sample_rate", "")).strip()
        if self.sr is None:
            if not raw_sr:
                raise ValueError("Fixed mixture manifest must contain sample_rate when sr is not provided")
            target_sr = int(raw_sr)
        else:
            target_sr = self.sr
            if self.strict_shape:
                if not raw_sr:
                    raise ValueError(
                        f"Fixed mixture {row_label} is missing sample_rate; "
                        f"strict_shape requires it to match sr={target_sr}"
                    )
                row_sr = int(raw_sr)
                if row_sr != target_sr:
                    raise ValueError(
                        f"Fixed mixture {row_label} sample_rate mismatch: manifest has {row_sr}, "
                        f"datamodule/config has {target_sr}"
                    )

        raw_n_samples = str(row.get("n_samples", "")).strip()
        raw_duration = str(row.get("duration", "")).strip()
        manifest_n_samples: int | None = int(raw_n_samples) if raw_n_samples else None
        duration_n_samples: int | None = None
        if raw_duration:
            duration_n_samples = int(round(target_sr * float(raw_duration)))
        if (
            self.strict_shape
            and manifest_n_samples is not None
            and duration_n_samples is not None
            and manifest_n_samples != duration_n_samples
        ):
            raise ValueError(
                f"Fixed mixture {row_label} manifest shape is inconsistent: n_samples={manifest_n_samples}, "
                f"duration={raw_duration} at sample_rate={target_sr} implies {duration_n_samples} samples"
            )

        if self.duration is not None:
            n_samples = int(round(target_sr * self.duration))
            if self.strict_shape:
                if manifest_n_samples is None and duration_n_samples is None:
                    raise ValueError(
                        f"Fixed mixture {row_label} is missing n_samples/duration; strict_shape requires one of them "
                        f"to match configured duration={self.duration}"
                    )
                if manifest_n_samples is not None and manifest_n_samples != n_samples:
                    raise ValueError(
                        f"Fixed mixture {row_label} n_samples mismatch: manifest has {manifest_n_samples}, "
                        f"datamodule/config duration={self.duration} at sample_rate={target_sr} implies {n_samples}"
                    )
                if duration_n_samples is not None and duration_n_samples != n_samples:
                    raise ValueError(
                        f"Fixed mixture {row_label} duration mismatch: manifest duration={raw_duration} at "
                        f"sample_rate={target_sr} implies {duration_n_samples} samples, datamodule/config expects "
                        f"{n_samples}"
                    )
        else:
            if manifest_n_samples is not None:
                n_samples = manifest_n_samples
            elif duration_n_samples is not None:
                n_samples = duration_n_samples
            else:
                raise ValueError(
                    "Fixed mixture manifest must contain n_samples or duration when duration is not provided"
                )
        if n_samples <= 0:
            raise ValueError(f"Invalid fixed mixture sample count: {n_samples}")
        return target_sr, n_samples

    @staticmethod
    def _resolve_manifest_path(row: Mapping[str, str], column: str) -> Path | None:
        raw_path = str(row.get(column, "")).strip()
        if not raw_path:
            return None
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = Path(str(row["__manifest_dir"])) / path
        return path

    @staticmethod
    def _load_rendered_audio(path: Path, *, target_sr: int, n_samples: int) -> torch.Tensor:
        if not path.is_file():
            raise FileNotFoundError(f"Fixed mixture manifest points to a missing audio file: {path}")
        audio_np, source_sr = sf.read(path, always_2d=True, dtype="float32")
        if audio_np.size == 0:
            raise ValueError(f"Could not read audio from {path}")
        audio = torch.from_numpy(audio_np.T.copy()).float().mean(dim=0, keepdim=True)
        if int(source_sr) != target_sr:
            audio = AF.resample(audio, orig_freq=int(source_sr), new_freq=target_sr)
        audio = audio.squeeze(0)
        if audio.numel() < n_samples:
            padded = torch.zeros(n_samples, dtype=torch.float32)
            padded[: audio.numel()] = audio
            return padded
        return audio[:n_samples].float()

    def __getitem__(self, index: int):
        row = self.rows[int(index)]
        target_sr, n_samples = self._target_sr_and_samples(row)
        stems = torch.zeros(len(self.source_order), n_samples, dtype=torch.float32)
        metadata: dict[str, Any] = {
            "index": int(index),
            "mixture_id": row.get("mixture_id", str(index)),
            "source_paths": {stem: [] for stem in self.source_order},
        }
        for stem_idx, stem_name in enumerate(self.source_order):
            stem_path = self._resolve_manifest_path(row, f"{stem_name}_filepath")
            if stem_path is None:
                continue
            stems[stem_idx] = self._load_rendered_audio(stem_path, target_sr=target_sr, n_samples=n_samples)
            metadata["source_paths"][stem_name] = [str(stem_path)]

        ref = stems[:, None, :].contiguous()
        mixture_path = self._resolve_manifest_path(row, self.mixture_filepath_column)
        if self.use_rendered_mixture and mixture_path is not None:
            wav = self._load_rendered_audio(mixture_path, target_sr=target_sr, n_samples=n_samples)[
                None, :
            ].contiguous()
        else:
            wav = ref.sum(dim=0).contiguous()
        if self.return_metadata:
            metadata["mixture_path"] = "" if mixture_path is None else str(mixture_path)
            return wav, ref, metadata
        return wav, ref
