import torch
from src.datamodules.dataset import DatasetS3


class SourceClassifierDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = DatasetS3(**base_dataset["args"]) if isinstance(base_dataset, dict) else base_dataset
        self.labels = self.base_dataset.labels
        self.collate_fn = self._collate_fn

    def __len__(self):
        return len(self.base_dataset) * self.base_dataset.n_sources

    def __getitem__(self, idx):
        mixture_idx = idx // self.base_dataset.n_sources
        source_idx = idx % self.base_dataset.n_sources
        item = self.base_dataset[mixture_idx]
        label = item["label"][source_idx]
        is_silence = label == "silence"
        class_index = 0 if is_silence else self.labels.index(label)
        return {
            "waveform": item["dry_sources"][source_idx],
            "class_index": torch.tensor(class_index, dtype=torch.long),
            "is_silence": torch.tensor(is_silence, dtype=torch.bool),
        }

    def _collate_fn(self, items):
        return {
            "waveform": torch.stack([x["waveform"] for x in items], dim=0),
            "class_index": torch.stack([x["class_index"] for x in items], dim=0),
            "is_silence": torch.stack([x["is_silence"] for x in items], dim=0),
        }


class EstimatedSourceClassifierDataset(torch.utils.data.Dataset):
    """Single-source classifier dataset backed by cached separator estimates.

    The wrapped ``DatasetS3`` must be in ``waveform`` mode and provide an
    ``estimate_target_dir``. Estimated files should follow the repository
    convention:

    ``<soundscape>_<slot>_<label>.wav`` or ``<soundscape>_<label>.wav``.

    Labels are read from the estimate filenames. For supervised SC fine-tuning,
    generate the cache with oracle labels in the filenames. If the filenames use
    model-predicted labels, this becomes pseudo-label fine-tuning.
    """

    def __init__(self, base_dataset, source_prefix="est"):
        self.base_dataset = DatasetS3(**base_dataset["args"]) if isinstance(base_dataset, dict) else base_dataset
        self.source_prefix = source_prefix
        self.labels = self.base_dataset.labels
        self.collate_fn = self._collate_fn

        source_key = f"{self.source_prefix}_dry_sources"
        label_key = f"{self.source_prefix}_label"
        if not self.base_dataset.config.get("mode") == "waveform":
            raise ValueError("EstimatedSourceClassifierDataset requires DatasetS3 waveform mode")
        if source_key != "dry_sources" and self.base_dataset.config.get("estimate_target_dir") is None:
            raise ValueError("DatasetS3 config must provide estimate_target_dir for estimated-source training")
        self.source_key = source_key
        self.label_key = label_key

    def __len__(self):
        return len(self.base_dataset) * self.base_dataset.n_sources

    def __getitem__(self, idx):
        mixture_idx = idx // self.base_dataset.n_sources
        source_idx = idx % self.base_dataset.n_sources
        item = self.base_dataset[mixture_idx]
        label = item[self.label_key][source_idx]
        is_silence = label == "silence"
        class_index = 0 if is_silence else self.labels.index(label)
        return {
            "waveform": item[self.source_key][source_idx],
            "class_index": torch.tensor(class_index, dtype=torch.long),
            "is_silence": torch.tensor(is_silence, dtype=torch.bool),
        }

    def _collate_fn(self, items):
        return {
            "waveform": torch.stack([x["waveform"] for x in items], dim=0),
            "class_index": torch.stack([x["class_index"] for x in items], dim=0),
            "is_silence": torch.stack([x["is_silence"] for x in items], dim=0),
        }
