#!/usr/bin/env python3
"""Prepare multi-input NHWC calibration data from real sequential stem mixtures."""

from __future__ import annotations

from argparse import ArgumentParser
import json
import math
from pathlib import Path
import sys

import h5py
import torch
import torch.nn.functional as F
from hydra.utils import instantiate

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.online.export_onnx_online_model import load_export_core  # noqa: E402


BASE_CONFIG_PATH = REPO_ROOT / "aiaccel" / "aiaccel" / "torch" / "apps" / "config"


def _load_data_config(recipe: Path):
    from aiaccel.config import load_config, resolve_inherit

    config = load_config(
        recipe,
        {
            "config_path": str(recipe),
            "working_directory": str(recipe.parent.resolve()),
            "base_config_path": str(BASE_CONFIG_PATH.resolve()),
        },
    )
    return resolve_inherit(config)


def _to_nhwc(tensor: torch.Tensor):
    return tensor.detach().cpu().permute(0, 2, 3, 1).contiguous().numpy().astype("float32")


def _causal_stft(wav: torch.Tensor, *, n_fft: int, hop_length: int) -> torch.Tensor:
    wav = F.pad(wav, (n_fft - hop_length, 0))
    window = torch.hann_window(n_fft, device=wav.device, dtype=wav.dtype)
    return torch.stft(
        wav,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        center=False,
        return_complex=True,
    )


def _packed_frame(stft: torch.Tensor, frame_idx: int) -> torch.Tensor:
    frame = stft[..., frame_idx : frame_idx + 1]
    return torch.cat((frame.real, frame.imag), dim=0).unsqueeze(0).transpose(2, 3).contiguous()


@torch.inference_mode()
def main() -> int:
    parser = ArgumentParser()
    parser.add_argument("model_path", type=Path, help="Recipe, trained directory, or checkpoint")
    parser.add_argument("--data-recipe", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--records", type=int, default=64)
    parser.add_argument("--mixtures", type=int, default=4)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--warmup-frames", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.records <= 0 or args.mixtures <= 0:
        raise ValueError("--records and --mixtures must be positive")
    if not args.source_manifest.is_file():
        raise FileNotFoundError(args.source_manifest)

    torch.manual_seed(args.seed)
    core, source_mode = load_export_core(args.model_path, args.device)
    core = core.to(args.device).eval()
    if hasattr(core, "masking"):
        core.masking = False

    config = _load_data_config(args.data_recipe)
    manifest = str(args.source_manifest.resolve())
    config.datamodule.source_manifest_csv = [manifest]
    config.datamodule.val_source_manifest_csv = [manifest]
    config.datamodule.batch_size = 1
    config.datamodule.val_batch_size = 1
    config.datamodule.num_workers = 0
    config.datamodule.dataset_length = args.mixtures
    config.datamodule.val_dataset_length = 1
    config.datamodule.train_seed = args.seed
    config.datamodule.duration = args.duration
    config.datamodule.synthesis.mixture_duration = args.duration
    datamodule = instantiate(config.datamodule)
    datamodule.setup("fit")

    per_mixture = math.ceil(args.records / args.mixtures)
    records: list[list[torch.Tensor]] = []
    selected_frames: list[dict[str, int]] = []
    for mixture_idx in range(args.mixtures):
        wav, _reference = datamodule.train_dataset[mixture_idx]
        wav = wav.to(device=args.device, dtype=torch.float32)
        stft = _causal_stft(wav, n_fft=args.n_fft, hop_length=args.hop_length)
        available = stft.shape[-1] - args.warmup_frames
        if available <= 0:
            raise ValueError(
                f"Mixture {mixture_idx} has only {stft.shape[-1]} frames, "
                f"not enough for warmup={args.warmup_frames}"
            )
        stride = max(1, available // per_mixture)
        wanted = set(
            min(args.warmup_frames + idx * stride, stft.shape[-1] - 1)
            for idx in range(per_mixture)
        )
        state = core.init_stream_state(batch_size=1, device=torch.device(args.device), dtype=torch.float32)
        for frame_idx in range(stft.shape[-1]):
            x = _packed_frame(stft, frame_idx)
            if frame_idx in wanted and len(records) < args.records:
                records.append([x.clone(), *[tensor.clone() for tensor in state]])
                selected_frames.append({"mixture": mixture_idx, "frame": frame_idx})
            _output, state = core.forward_stream(x, state)
        if len(records) >= args.records:
            break

    if len(records) != args.records:
        raise RuntimeError(f"Prepared {len(records)} records, expected {args.records}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out, "w") as h5:
        value = h5.create_group("value")
        for record_idx, tensors in enumerate(records):
            group = value.create_group(str(record_idx))
            for input_idx, tensor in enumerate(tensors):
                group.create_dataset(str(input_idx), data=_to_nhwc(tensor))

    metadata = {
        "model_path": str(args.model_path.resolve()),
        "source_mode": source_mode,
        "data_recipe": str(args.data_recipe.resolve()),
        "source_manifest": str(args.source_manifest.resolve()),
        "records": args.records,
        "mixtures": args.mixtures,
        "duration": args.duration,
        "seed": args.seed,
        "n_fft": args.n_fft,
        "hop_length": args.hop_length,
        "input_names": ["x", *[f"state_{idx}" for idx in range(len(records[0]) - 1)]],
        "nhwc_shapes": [list(_to_nhwc(tensor).shape) for tensor in records[0]],
        "selected_frames": selected_frames,
        "quality_note": (
            "Config-only models provide structural quantization calibration only. "
            "Use a trained directory/checkpoint for separation-quality validation."
        ),
    }
    metadata_path = args.out.with_suffix(args.out.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
