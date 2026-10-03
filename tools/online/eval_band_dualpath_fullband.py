#!/usr/bin/env python3
"""Evaluate the 12-24 kHz host extension of a 24 kHz BandDualPathNPU model on 48 kHz synthetic stems.

Compares, on the same held-out 48 kHz clips (``synthetic_stem_benchmark`` generator):

* ``band_only``: model masks below 12 kHz, nothing above (a 24 kHz-only pipeline);
* ``ext``: default extension (per-frame power-share gains from 9-12 kHz, 64-bin crossfade);
* ``ext_smooth``: same with causal smoothing 0.7 on the reference energies;
* ``equal``: every stem gets 1/S of the high band (no information from the model);
* ``oracle``: per-frame power share of the *true* stems above 12 kHz (upper bound
  for any per-frame, per-stem real gain).

Metrics per stem (active stems only): full-band SI-SDR improvement and high-band
(12-24 kHz) SDR.

Example::

    python tools/online/synthetic_stem_benchmark.py --model band_dualpath_medium_noattn --sr 24000 \\
        --regions sr24k_r5 --io-layout slots --steps 400 --save-state --tag hf --out logs/hf
    python tools/online/eval_band_dualpath_fullband.py --state logs/hf/band_dualpath_medium_noattn_hf.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import tools.online.synthetic_stem_benchmark as bench  # noqa: E402
from spectral_feature_compression.core.model.band_dualpath_fullband import (  # noqa: E402
    FullBandConfig,
    extend_masks,
)

NAMES = ("speech", "music", "effects")


def hf_sdr(est: torch.Tensor, ref: torch.Tensor, sr: int, cutoff: float = 12000.0) -> torch.Tensor:
    spec_e, spec_r = torch.fft.rfft(est, dim=-1), torch.fft.rfft(ref, dim=-1)
    band = torch.fft.rfftfreq(est.shape[-1], 1.0 / sr) > cutoff
    num = spec_r[..., band].abs().square().sum(-1)
    den = (spec_r[..., band] - spec_e[..., band]).abs().square().sum(-1)
    return 10 * torch.log10(num / (den + 1e-12) + 1e-12), num


@torch.no_grad()
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--state", type=Path, required=True, help="state_dict saved by synthetic_stem_benchmark --save-state")
    p.add_argument("--model", default="band_dualpath_medium_noattn")
    p.add_argument("--regions", default="sr24k_r5")
    p.add_argument("--io-layout", default="slots")
    p.add_argument("--mask-points", type=int, default=16)
    p.add_argument("--clips", type=int, default=24)
    p.add_argument("--seconds", type=float, default=3.0)
    p.add_argument("--ref-bins", type=int, nargs=2, default=(768, 1024), metavar=("LO", "HI"))
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    torch.set_num_threads(4)

    bench.SR = 24000
    net = bench.build(args.model, regions=args.regions, io_layout=args.io_layout, mask_points=args.mask_points)
    net.load_state_dict(torch.load(args.state, map_location="cpu"))
    net.eval()
    model = net.model  # BandDualPathNPUModel inside ModelWrapper

    bench.SR = 48000  # held-out clips at 48 kHz, with content above 12 kHz
    mix, src = bench.make_batch(args.clips, int(args.seconds * 48000), torch.Generator().manual_seed(999))
    cfg = FullBandConfig(ref_bins=tuple(args.ref_bins))
    window = torch.hann_window(cfg.n_fft)
    n = mix.shape[-1]

    def stft(x: torch.Tensor) -> torch.Tensor:
        return torch.stft(x, cfg.n_fft, cfg.hop, window=window, return_complex=True)

    spec = stft(mix[:, 0])[:, None]  # [B, 1, F, T]
    spec_src = stft(src[:, :, 0].reshape(-1, n)).reshape(args.clips, 3, 1, cfg.n_freq_full, -1)
    spec_lo = spec[:, :, : cfg.n_freq_model] / cfg.ratio
    masks = torch.cat(
        [model.complex_masks(model.core(model.host_features(spec_lo[i : i + 4]))) for i in range(0, args.clips, 4)]
    )

    hf = slice(cfg.n_freq_model, None)
    hf_shape = masks.shape[:3] + (cfg.n_freq_full - cfg.n_freq_model, masks.shape[-1])
    true_power = spec_src[..., hf, :].abs().square().sum(-2, keepdim=True)
    oracle_gain = true_power / true_power.sum(1, keepdim=True).clamp_min(1e-12)
    variants = {
        "band_only": torch.cat([masks, torch.zeros(hf_shape, dtype=masks.dtype)], -2),
        "ext": extend_masks(masks, spec_lo, cfg),
        "ext_smooth": extend_masks(masks, spec_lo, FullBandConfig(ref_bins=cfg.ref_bins, smoothing=0.7)),
        "equal": torch.cat([masks, torch.full(hf_shape, 1 / 3, dtype=masks.dtype)], -2),
        "oracle": torch.cat([masks, oracle_gain.expand(hf_shape).to(masks.dtype)], -2),
    }
    ref = src[:, :, 0]
    base = bench.si_sdr(mix[:, 0][:, None].expand_as(ref), ref)
    active = ref.square().mean(-1) > 1e-8
    _, ref_hf_energy = hf_sdr(ref, ref, 48000)
    hf_active = active & (ref_hf_energy > 1e-3 * ref_hf_energy.max())
    results = {}
    for name, m in variants.items():
        est = torch.istft((m * spec[:, None]).reshape(-1, cfg.n_freq_full, m.shape[-1]), cfg.n_fft, cfg.hop,
                          window=window, length=n).reshape(args.clips, 3, n)
        imp = bench.si_sdr(est, ref) - base
        hsdr, _ = hf_sdr(est, ref, 48000)
        row = {}
        for i, stem in enumerate(NAMES):
            row[f"sisdri_{stem}"] = float(imp[:, i][active[:, i]].mean())
            row[f"hf_sdr_{stem}"] = float(hsdr[:, i][hf_active[:, i]].mean()) if hf_active[:, i].any() else None
        row["sisdri_mean"] = sum(row[f"sisdri_{s}"] for s in NAMES) / 3
        results[name] = row
        hf_parts, full_parts = [], []
        for stem in NAMES:
            value = row["hf_sdr_" + stem]
            hf_parts.append(f"{stem}={value:.2f}" if value is not None else f"{stem}=n/a")
            full_parts.append(f"{stem}={row['sisdri_' + stem]:.2f}")
        print(f"{name:11s} full-band SI-SDRi mean={row['sisdri_mean']:.2f} ({' '.join(full_parts)}) "
              f"| 12-24 kHz SDR {' '.join(hf_parts)}")
    print("clips with stem energy above 12 kHz:", {s: int(hf_active[:, i].sum()) for i, s in enumerate(NAMES)})
    if args.out:
        args.out.write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
