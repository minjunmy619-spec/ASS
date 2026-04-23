import argparse
import os
import shutil

import soundfile as sf
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import initialize_config


def _as_mono_or_multichannel(waveform):
    waveform = waveform.detach().cpu()
    if waveform.dim() == 1:
        return waveform.numpy()
    return waveform.transpose(0, 1).numpy()


def _write_wav(path, waveform, sample_rate, overwrite=False):
    if os.path.exists(path) and not overwrite:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, _as_mono_or_multichannel(waveform), sample_rate)


def _copy_if_available(src, dst, overwrite=False):
    if src is None or not os.path.exists(src):
        return False
    if os.path.exists(dst) and not overwrite:
        return True
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _label_vectors(batch, labels_num, device):
    label_vector = batch["label_vector"].to(device)
    if label_vector.shape[-1] > labels_num:
        label_vector = label_vector[..., :labels_num]
    return label_vector


def _save_reference_files(batch, dataset, output_root, sample_rate, overwrite=False):
    soundscape_dir = os.path.join(output_root, "soundscape")
    oracle_dir = os.path.join(output_root, "oracle_target")

    for i, soundscape in enumerate(batch["soundscape"]):
        src_path = dataset.data[dataset.data_index[soundscape]]["mixture_path"] if hasattr(dataset, "data_index") else None
        dst_path = os.path.join(soundscape_dir, f"{soundscape}.wav")
        if not _copy_if_available(src_path, dst_path, overwrite=overwrite):
            _write_wav(dst_path, batch["mixture"][i], sample_rate, overwrite=overwrite)

        for slot, label in enumerate(batch["label"][i]):
            if label == "silence":
                continue
            oracle_path = os.path.join(oracle_dir, f"{soundscape}_{slot}_{label}.wav")
            _write_wav(oracle_path, batch["dry_sources"][i, slot, 0], sample_rate, overwrite=overwrite)


def _save_estimates(batch, waveforms, labels, output_root, sample_rate, overwrite=False):
    estimate_dir = os.path.join(output_root, "estimate_target")
    waveforms = waveforms.detach().cpu()
    for i, soundscape in enumerate(batch["soundscape"]):
        for slot, label in enumerate(labels[i]):
            if label == "silence":
                continue
            estimate_path = os.path.join(estimate_dir, f"{soundscape}_{slot}_{label}.wav")
            _write_wav(estimate_path, waveforms[i, slot, 0], sample_rate, overwrite=overwrite)


def _build_dataset_index(dataset):
    if hasattr(dataset, "data"):
        dataset.data_index = {item["soundscape"]: idx for idx, item in enumerate(dataset.data)}


def export_cache(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)

    dataset = initialize_config(config["dataset"], reload=True)
    _build_dataset_index(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batchsize,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        num_workers=args.num_workers,
    )

    model = initialize_config(config["model"], reload=True)
    model.eval()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model = model.to(device)

    os.makedirs(args.output_root, exist_ok=True)

    for batch in tqdm(dataloader):
        _save_reference_files(
            batch,
            dataset,
            args.output_root,
            dataset.sr,
            overwrite=args.overwrite,
        )

        mixture = batch["mixture"].to(device)
        with torch.no_grad():
            if args.mode == "oracle_tse":
                if not hasattr(model, "_run_tse"):
                    raise TypeError("oracle_tse mode requires a Kwon2025S5-like model with _run_tse")
                enroll = batch["dry_sources"].to(device)
                label_vector = _label_vectors(batch, len(model.labels), device)
                waveforms = model._run_tse(mixture, enroll, label_vector)
                labels = batch["label"]
            elif args.mode == "pseudo_s5":
                output = model.predict_label_separate(mixture)
                waveforms = output["waveform"]
                labels = output["label"]
            else:
                raise ValueError(f"Unknown mode: {args.mode}")

        _save_estimates(
            batch,
            waveforms,
            labels,
            args.output_root,
            dataset.sr,
            overwrite=args.overwrite,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--output_root", default="workspace/sc_finetune")
    parser.add_argument("--mode", choices=["oracle_tse", "pseudo_s5"], default="oracle_tse")
    parser.add_argument("--batchsize", "-b", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    export_cache(args)


if __name__ == "__main__":
    main()
