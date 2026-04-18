# Copyright (c) 2026 National Institute of Advanced Industrial Science and Technology (AIST), Japan
#
# SPDX-License-Identifier: MIT

from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig

from einops import rearrange
import numpy as np

import torch
from torch import nn

from torchaudio.transforms import InverseSpectrogram

import soundfile as sf

from aiaccel.config import load_config
from spectral_feature_compression.utils.separator import main


@dataclass
class Context:
    model: nn.Module
    istft: nn.Module
    config: ListConfig | DictConfig


def add_common_args(parser):
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--css_segment_size", type=int, default=None)
    parser.add_argument("--css_shift_size", type=int, default=None)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)


def initialize(args: Namespace, unk_args: list[str]):
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

    istft = InverseSpectrogram(config.n_fft, hop_length=config.hop_length).to(args.device)

    # overwrite CSS parameters
    if args.css_segment_size is not None:
        model.model.css_segment_size = args.css_segment_size
        if model.use_ema_model:
            model.ema_model.module.css_segment_size = args.css_segment_size
    if args.css_shift_size is not None:
        model.model.css_shift_size = args.css_shift_size
        if model.use_ema_model:
            model.ema_model.module.css_shift_size = args.css_shift_size

    return Context(model, istft, config)


@torch.autocast("cuda", enabled=True, dtype=torch.bfloat16)
def separate(
    src_filename: Path,
    dst_filename: Path,
    ctx: Context,
    args: Namespace,
    unk_args: list[str],
):
    model = ctx.model

    # load wav
    wav, sr = sf.read(src_filename, dtype=np.float32, always_2d=True)
    wav = rearrange(torch.from_numpy(wav).to(model.device), "t m -> 1 m t")

    if args.start is not None and args.end is not None:
        wav = wav[..., args.start * sr : args.end * sr]

    # est: (n_batch, n_src, n_chan, n_samples)
    separation_model = model.ema_model.module if model.use_ema_model else model.model

    est = (
        separation_model.css(wav, ref=None)
        if model.css_validation and wav.shape[-1] >= args.css_segment_size * sr
        else separation_model(wav)
    )
    # est = model.model.css(wav, ref=None) if model.css_validation else model.model(wav)
    est = est[0]  # remove batch dimension

    # save separated signal
    for n in range(est.shape[0]):
        dst_filename_n = dst_filename.parent / f"est{n}.wav"
        sf.write(dst_filename_n, est[n].T.cpu().numpy(), sr, "PCM_24")


if __name__ == "__main__":
    main(add_common_args, initialize, separate)
