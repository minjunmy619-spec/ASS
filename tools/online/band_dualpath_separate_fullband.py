#!/usr/bin/env python3
"""Separate a full-band (e.g. 48 kHz) wav with a 24 kHz BandDualPathNPU model.

Runs exactly the host math of ``spectral_feature_compression.core.model.band_dualpath_fullband``:
one 48 kHz STFT (4096 / 1024), bins 0-1024 / 2 into the model, model masks below
12 kHz, per-stem power-share gains above 12 kHz (crossfaded), one 48 kHz iSTFT per stem.

Example::

    python tools/online/band_dualpath_separate_fullband.py \\
        --config recipes/dnr/models/band-dualpath-npu.medium-noattn.sr24k.bs62r5.slots-p16.onfly.rt192k/config.yaml \\
        --ckpt path/to/epoch=xxxx.ckpt --input tv_clip_48k.wav --out-dir out/tv_clip
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from spectral_feature_compression.core.model.band_dualpath_fullband import (  # noqa: E402
    FullBandConfig,
    separate_fullband,
)
from tools.online.export_band_dualpath_npu import REGION_CHOICES, build_model  # noqa: E402

STEM_NAMES = ("speech", "music", "effects")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", type=Path, help="recipe config.yaml (model sample rate / STFT taken from it)")
    src.add_argument("--preset", help="model preset (untrained unless --ckpt)")
    p.add_argument("--ckpt", type=Path)
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--fs", type=int, default=24000, help="model sample rate when using --preset")
    p.add_argument("--n-fft", type=int, default=2048)
    p.add_argument("--hop", type=int, default=512)
    p.add_argument("--regions", default="sr24k_r5", choices=sorted(REGION_CHOICES))
    p.add_argument("--io-layout", default="slots", choices=["regions", "slots"])
    p.add_argument("--mask-points", type=int, default=16)
    p.add_argument("--input", type=Path, required=True, help="wav at an integer multiple of the model sample rate")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--crossfade-bins", type=int, default=64)
    p.add_argument("--smoothing", type=float, default=0.0)
    p.add_argument("--ref-bins", type=int, nargs=2, default=(768, 1024), metavar=("LO", "HI"))
    p.add_argument("--band-only", action="store_true", help="also write stems without the high-band extension")
    args = p.parse_args()

    import soundfile as sf

    model = build_model(args)
    wav, sr = sf.read(args.input, dtype="float32", always_2d=True)
    if sr % args.fs != 0:
        raise ValueError(f"Input rate {sr} is not an integer multiple of the model rate {args.fs}")
    cfg = FullBandConfig(
        ratio=sr // args.fs,
        model_n_fft=args.n_fft,
        model_hop=args.hop,
        ref_bins=tuple(args.ref_bins),
        crossfade_bins=args.crossfade_bins,
        smoothing=args.smoothing,
    )
    mix = torch.from_numpy(wav.T).unsqueeze(0)  # [1, M, N]
    full, band_only = separate_fullband(model, mix, cfg)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for idx, name in enumerate(STEM_NAMES[: full.shape[1]]):
        sf.write(args.out_dir / f"{idx:02d}_{name}.wav", full[0, idx].T.numpy(), sr)
        if args.band_only:
            sf.write(args.out_dir / f"{idx:02d}_{name}.band_only.wav", band_only[0, idx].T.numpy(), sr)
    print(f"wrote {full.shape[1]} stems at {sr} Hz to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
