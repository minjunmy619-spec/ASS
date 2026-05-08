from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datamodules.dataset import DatasetS3
from src.utils import initialize_config, parse_yaml


def load_bridge_model(config_path, checkpoint_path, device):
    cfg = parse_yaml(config_path)
    model = initialize_config(cfg["lightning_module"]["args"]["model"])
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    model_state = {}
    for key, value in state.items():
        if key.startswith("model."):
            key = key[len("model.") :]
        model_state[key] = value
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if missing:
        print(f"missing model keys: {len(missing)}")
    if unexpected:
        print(f"unexpected model keys: {len(unexpected)}")
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint/config mismatch while exporting USS bridge features: "
            f"missing={list(missing)[:10]}, unexpected={list(unexpected)[:10]}"
        )
    model.to(device)
    model.eval()
    return model


def build_waveform_dataset(args):
    cfg = {"mode": "waveform", "soundscape_dir": args.soundscape_dir, "sr": args.sample_rate}
    if args.oracle_target_dir:
        cfg["oracle_target_dir"] = args.oracle_target_dir
    if args.estimate_target_dir:
        cfg["estimate_target_dir"] = args.estimate_target_dir
    return DatasetS3(
        config=cfg,
        n_sources=args.n_sources,
        label_set=args.label_set,
        label_vector_mode="stack",
        silence_label_mode="zeros",
        return_source=args.oracle_target_dir is not None,
        return_meta=False,
    )


def to_device(batch, device):
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


def save_features(output, batch, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    keys = (
        "tse_condition",
        "foreground_embedding",
        "foreground_audio_embedding",
        "pred_doa_vector",
        "used_spatial_vector",
        "prototype_logits",
        "class_logits",
        "silence_logits",
    )
    for idx, soundscape in enumerate(batch["soundscape"]):
        obj = {}
        for key in keys:
            if key in output:
                obj[key] = output[key][idx].detach().cpu()
        torch.save(obj, os.path.join(output_dir, f"{soundscape}.pt"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--soundscape_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--oracle_target_dir", default=None)
    parser.add_argument("--estimate_target_dir", default=None)
    parser.add_argument("--sample_rate", type=int, default=32000)
    parser.add_argument("--n_sources", type=int, default=3)
    parser.add_argument("--label_set", default="dcase2026t4")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_bridge_model(args.config, args.checkpoint, device)
    dataset = build_waveform_dataset(args)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=dataset.collate_fn)

    with torch.no_grad():
        for batch in tqdm(loader, desc="export USS bridge features"):
            device_batch = to_device(batch, device)
            output = model({"mixture": device_batch["mixture"]})
            save_features(output, batch, args.output_dir)


if __name__ == "__main__":
    main()
