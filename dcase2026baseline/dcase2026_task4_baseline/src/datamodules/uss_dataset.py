import numpy as np
import torch

from src.datamodules.dataset import DatasetS3


def _extract_waveforms(events, n_expected, length):
    waveforms = []
    for event in events[:n_expected]:
        wav = event.get("waveform_dry", event.get("waveform", None))
        if wav is None:
            continue
        wav = np.asarray(wav)
        if wav.ndim == 1:
            wav = wav[None, :]
        waveforms.append(wav.astype(np.float32))
    while len(waveforms) < n_expected:
        waveforms.append(np.zeros((1, length), dtype=np.float32))
    return torch.from_numpy(np.stack(waveforms, axis=0))


class USSDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        args = base_dataset["args"].copy()
        args["return_meta"] = True
        self.base_dataset = DatasetS3(**args)
        self.labels = self.base_dataset.labels
        self.collate_fn = self.base_dataset.collate_fn

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        metadata = item.get("metadata", {})
        length = item["mixture"].shape[-1]

        int_events = metadata.get("int_events", metadata.get("interference_events", []))
        background = metadata.get("bg_events", metadata.get("background", metadata.get("bg_event", {})))
        background_events = [background] if isinstance(background, dict) else background

        labels = item["label"]
        class_index = []
        is_silence = []
        for label in labels:
            is_silence.append(label == "silence")
            class_index.append(0 if label == "silence" else self.labels.index(label))

        item["foreground_waveform"] = item["dry_sources"]
        item["interference_waveform"] = _extract_waveforms(int_events, 2, length)
        item["noise_waveform"] = _extract_waveforms(background_events, 1, length)
        item["class_index"] = torch.tensor(class_index, dtype=torch.long)
        item["is_silence"] = torch.tensor(is_silence, dtype=torch.bool)
        return item
