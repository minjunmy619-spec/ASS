import torch

from src.datamodules.dataset import DatasetS3


def _zero_inactive(label_vector, active_mask):
    label_vector = label_vector.clone()
    label_vector[~active_mask] = 0.0
    return label_vector


def _crop_or_pad_tensor(tensor, start, length):
    end = start + length
    if tensor.shape[-1] >= end:
        return tensor[..., start:end]
    cropped = tensor[..., start:]
    return torch.nn.functional.pad(cropped, (0, length - cropped.shape[-1]))


def _crop_spans(span_sec, start_sample, length, sample_rate):
    if span_sec is None:
        return None
    start_sec = float(start_sample) / float(sample_rate)
    duration_sec = float(length) / float(sample_rate)
    out = span_sec.clone()
    valid = (out[..., 0] >= 0.0) & (out[..., 1] > out[..., 0])
    shifted_start = torch.clamp(out[..., 0] - start_sec, min=0.0, max=duration_sec)
    shifted_end = torch.clamp(out[..., 1] - start_sec, min=0.0, max=duration_sec)
    keep = valid & (shifted_end > shifted_start)
    out[..., 0] = shifted_start
    out[..., 1] = shifted_end
    out[~keep] = -1.0
    return out


def _span_active_mask(span_sec):
    if span_sec is None:
        return None
    return (span_sec[..., 0] >= 0.0) & (span_sec[..., 1] > span_sec[..., 0])


def _waveform_active_mask(waveform, energy_eps):
    flat = waveform.flatten(start_dim=1).float()
    return flat.pow(2).sum(dim=-1) > float(energy_eps)


class TSEDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        args = base_dataset["args"].copy()
        args["return_source"] = True
        self.base_dataset = DatasetS3(**args)
        self.collate_fn = self.base_dataset.collate_fn

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        labels = item["label"]
        label_stack = []
        active_mask = []
        for source_vec, label in zip(item["label_vector"].view(self.base_dataset.n_sources, -1), labels):
            if label == "silence":
                label_stack.append(torch.zeros_like(source_vec))
                active_mask.append(False)
            else:
                label_stack.append(source_vec)
                active_mask.append(True)
        item["label_vector"] = torch.stack(label_stack, dim=0)
        item["enrollment"] = item["dry_sources"].clone()
        item["waveform"] = item["dry_sources"]
        item["active_mask"] = torch.tensor(active_mask, dtype=torch.bool)
        if "span_sec" not in item:
            item["span_sec"] = torch.full((self.base_dataset.n_sources, 2), -1.0, dtype=torch.float32)
        return item


class EstimatedEnrollmentTSEDataset(torch.utils.data.Dataset):
    """TSE fine-tuning dataset that uses separator estimates as enrollment.

    This matches S5 inference more closely than ``TSEDataset``: the target is
    still the oracle source waveform, but the enrollment is the corresponding
    USS/S5 estimated slot loaded from ``estimate_target_dir``.
    """

    def __init__(
        self,
        base_dataset,
        label_source="oracle",
        crop_seconds=None,
        random_crop=True,
        require_estimate_for_active=True,
        active_energy_eps=1e-8,
    ):
        args = base_dataset["args"].copy() if isinstance(base_dataset, dict) else None
        if args is not None:
            args["return_source"] = True
            self.base_dataset = DatasetS3(**args)
        else:
            self.base_dataset = base_dataset
        self.collate_fn = getattr(self.base_dataset, "collate_fn", None)
        self.label_source = label_source
        self.crop_seconds = crop_seconds
        self.random_crop = bool(random_crop)
        self.require_estimate_for_active = bool(require_estimate_for_active)
        self.active_energy_eps = float(active_energy_eps)

        if self.label_source not in {"oracle", "estimate"}:
            raise ValueError("label_source must be 'oracle' or 'estimate'")
        if not hasattr(self.base_dataset, "n_sources"):
            raise ValueError("base_dataset must expose n_sources")
        if args is not None and self.base_dataset.config.get("mode") != "waveform":
            raise ValueError("EstimatedEnrollmentTSEDataset requires DatasetS3 waveform mode")
        if args is not None and self.base_dataset.config.get("estimate_target_dir") is None:
            raise ValueError("DatasetS3 config must provide estimate_target_dir")

    def __len__(self):
        return len(self.base_dataset)

    def _crop_item(self, item):
        if self.crop_seconds is None:
            return item
        sample_rate = getattr(self.base_dataset, "sr", None)
        if sample_rate is None:
            raise ValueError("base_dataset must expose sr when crop_seconds is set")
        crop_samples = int(round(float(self.crop_seconds) * float(sample_rate)))
        if crop_samples <= 0:
            raise ValueError("crop_seconds must be positive")

        samples = item["mixture"].shape[-1]
        if samples > crop_samples:
            if self.random_crop:
                start = int(torch.randint(0, samples - crop_samples + 1, ()).item())
            else:
                start = max(0, (samples - crop_samples) // 2)
        else:
            start = 0

        for key in ("mixture", "dry_sources", "est_dry_sources"):
            if key in item:
                item[key] = _crop_or_pad_tensor(item[key], start, crop_samples)
        for key in ("span_sec", "est_span_sec"):
            if key in item:
                item[key] = _crop_spans(item[key], start, crop_samples, sample_rate)
        return item

    def __getitem__(self, idx):
        item = self._crop_item(self.base_dataset[idx])
        labels = item["label"] if self.label_source == "oracle" else item["est_label"]
        label_vector = item["label_vector"] if self.label_source == "oracle" else item["est_label_vector"]
        label_vector = label_vector.view(self.base_dataset.n_sources, -1).clone()

        oracle_active = torch.tensor([label != "silence" for label in item["label"]], dtype=torch.bool)
        estimate_present = torch.tensor([label != "silence" for label in item["est_label"]], dtype=torch.bool)
        label_active = torch.tensor([label != "silence" for label in labels], dtype=torch.bool)
        active_mask = label_active & oracle_active
        target_span_active = _span_active_mask(item.get("span_sec"))
        if target_span_active is not None:
            active_mask = active_mask & target_span_active
        else:
            active_mask = active_mask & _waveform_active_mask(item["dry_sources"], self.active_energy_eps)
        if self.require_estimate_for_active:
            estimate_span_active = _span_active_mask(item.get("est_span_sec"))
            if estimate_span_active is not None:
                estimate_present = estimate_present & estimate_span_active
            elif "est_dry_sources" in item:
                estimate_present = estimate_present & _waveform_active_mask(item["est_dry_sources"], self.active_energy_eps)
            active_mask = active_mask & estimate_present

        waveform = item["dry_sources"].clone()
        waveform[~active_mask] = 0.0

        return {
            "mixture": item["mixture"],
            "enrollment": item["est_dry_sources"].clone(),
            "waveform": waveform,
            "label_vector": _zero_inactive(label_vector, active_mask),
            "active_mask": active_mask,
            "span_sec": item.get("span_sec", torch.full((self.base_dataset.n_sources, 2), -1.0, dtype=torch.float32)),
        }
