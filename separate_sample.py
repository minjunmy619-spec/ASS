# Copyright (c) 2026 National Institute of Advanced Industrial Science and Technology (AIST), Japan
#
# SPDX-License-Identifier: MIT

from argparse import ArgumentParser, Namespace
from pathlib import Path

from hydra.utils import instantiate

from einops import rearrange
import numpy as np

import torch
from torch import nn

import soundfile as sf

from aiaccel.config import load_config


def initialize_model(args: Namespace):
    if args.model_path.is_dir():
        checkpoint_path = args.model_path / "checkpoints"
        config_path = args.model_path
    elif args.model_path.suffix in [".ckpt", ".pth", ".pt"]:
        checkpoint_path = args.model_path
        config_path = checkpoint_path.parent.parent
    else:
        raise ValueError("model_path has to be either path to a directory or .ckpt")

    config = load_config(config_path / "merged_config.yaml")

    model = instantiate(config.task).to(args.device)

    # weight averaging using all .ckpt files except for last.ckpt
    model.load_average_checkpoint(checkpoint_path)

    model.eval()

    # overwrite CSS parameters
    if args.css_segment_size is not None:
        model.model.css_segment_size = args.css_segment_size
        if model.use_ema_model:
            model.ema_model.module.css_segment_size = args.css_segment_size
    if args.css_shift_size is not None:
        model.model.css_shift_size = args.css_shift_size
        if model.use_ema_model:
            model.ema_model.module.css_shift_size = args.css_shift_size

    return model


@torch.inference_mode()
def separate(src_filename: Path, dst_dir: Path, model: nn.Module):
    # load wav
    wav, sr = sf.read(src_filename, dtype=np.float32)
    wav = rearrange(torch.from_numpy(wav).to(model.device), "t m -> 1 m t")

    # est: (n_batch, n_src, n_chan, n_samples)
    est = model.model.css(wav, ref=None) if model.css_validation else model.model(wav)
    est = est[0]  # remove batch dimension

    # save separated signal
    dst_dir.mkdir(exist_ok=True, parents=True)
    for n in range(est.shape[0]):
        dst_dir_n = dst_dir / f"est{n}.wav"
        sf.write(dst_dir_n, est[n].T.cpu().numpy(), sr, "PCM_24")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("model_path", type=Path)
    parser.add_argument("src_filename", type=Path)
    parser.add_argument("dst_dir", type=Path)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--css_segment_size", type=int, default=None)
    parser.add_argument("--css_shift_size", type=int, default=None)
    args, unk_args = parser.parse_known_args()

    model = initialize_model(args)

    separate(args.src_filename, args.dst_dir, model)
