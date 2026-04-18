#!/usr/bin/env python3

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import torch

from aiaccel.config import load_config
from hydra.utils import instantiate


def load_trained_task(model_path: Path, device: str):
    if model_path.is_dir():
        checkpoint_path = model_path / "checkpoints"
        config_path = model_path
    elif model_path.suffix in [".ckpt", ".pth", ".pt"]:
        checkpoint_path = model_path
        config_path = checkpoint_path.parent.parent
    else:
        raise ValueError("model_path must be either a model directory or a checkpoint file")

    config = load_config(config_path / "merged_config.yaml")
    task = instantiate(config.task).to(device)
    task.load_average_checkpoint(checkpoint_path)
    task.eval()
    return task


@torch.inference_mode()
def main():
    p = ArgumentParser()
    p.add_argument("model_path", type=Path, help="Model directory (with merged_config.yaml) or a checkpoint file")
    p.add_argument("--out", type=Path, required=True, help="Output ONNX path")
    p.add_argument("--device", type=str, default="cpu")

    p.add_argument("--n_chan", type=int, required=True, help="Number of audio channels M")
    p.add_argument("--n_src", type=int, required=True, help="Number of sources N")
    p.add_argument("--frames", type=int, default=64, help="Fixed T for export")
    p.add_argument("--freqs", type=int, default=1025, help="Fixed F for export (n_fft//2+1)")

    p.add_argument("--opset", type=int, default=17)

    args = p.parse_args()

    task = load_trained_task(args.model_path, args.device)

    # task.model is a ModelWrapper; the inner model is OnlineSFCModel; core is OnlineSFC2D
    model_wrapper = task.ema_model.module if getattr(task, "use_ema_model", False) else task.model
    online_model = model_wrapper.model  # OnlineSFCModel
    core = online_model.core

    core.eval()

    # NPU core expects packed real/imag on channel: (B=1, 2*M, T, F)
    dummy = torch.randn(1, 2 * args.n_chan, args.frames, args.freqs, dtype=torch.float32, device=args.device)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        core,
        dummy,
        str(args.out),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["x"],
        output_names=["y"],
        dynamic_axes=None,  # keep fixed shapes for stricter NPUs
    )

    print(f"Exported: {args.out}")
    print("Input:  x (1, 2*M, T, F)")
    print("Output: y (1, 2*N*M, T, F)")


if __name__ == "__main__":
    main()

