#!/usr/bin/env python3
"""Controlled CPU sanity benchmark on synthetic speech / music / effects stems.

This is NOT a replacement for DnR / TV-profile training.  It gives every model
exactly the same on-the-fly data stream, loss, optimizer and number of steps so
that learnability and relative behaviour can be checked on a machine without
datasets or GPUs.  The stems are designed so that separating them needs both
spectral cues (harmonicity, formants, noise bands) and temporal cues (syllabic
rhythm, note decay/onsets, event envelopes):

* speech-like: glottal harmonics with a drifting F0, moving formants (vowel
  targets every ~0.2 s), a 3-6 Hz syllabic gate with pauses, and fricative
  noise bursts in the gaps;
* music-like: chords of decaying harmonic notes on a beat grid plus kick and
  hi-hat;
* effects-like: band-limited noise events, clicks and occasional chirps.

Example::

    python tools/online/synthetic_stem_benchmark.py --model band_dualpath_medium --steps 1500 --out logs/synth_bench
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SR = 44100


def _smooth_noise(n: int, knots: int, gen: torch.Generator) -> torch.Tensor:
    values = torch.rand(knots, generator=gen)
    return torch.nn.functional.interpolate(values.view(1, 1, -1), size=n, mode="linear", align_corners=True).view(-1)


def _bandpass_noise(n: int, lo: float, hi: float, gen: torch.Generator) -> torch.Tensor:
    spec = torch.fft.rfft(torch.randn(n, generator=gen))
    freqs = torch.fft.rfftfreq(n, 1.0 / SR)
    spec = spec * ((freqs >= lo) & (freqs <= hi))
    out = torch.fft.irfft(spec, n)
    return out / (out.std() + 1e-8)


def speech_like(n: int, gen: torch.Generator) -> torch.Tensor:
    t = torch.arange(n) / SR
    f0 = (90 + 150 * torch.rand(1, generator=gen)) * 2 ** (0.4 * (_smooth_noise(n, 12, gen) - 0.5))
    phase = 2 * math.pi * torch.cumsum(f0, 0) / SR
    k = torch.arange(1, 81).view(-1, 1)
    harm_freq = k * f0.view(1, -1)
    n_syl = max(2, int(n / SR / 0.2))
    formants = []
    for lo, hi, bw in ((300, 900, 90), (900, 2500, 140), (2300, 3300, 200)):
        track = lo + (hi - lo) * _smooth_noise(n, n_syl, gen)
        formants.append((track, bw))
    env = sum(torch.exp(-((harm_freq - track.view(1, -1)) ** 2) / (2 * bw**2)) for track, bw in formants)
    env = env * (1.0 / k) ** 0.5 * (harm_freq < 8000)
    voiced = (env * torch.sin(k * phase.view(1, -1))).sum(0)
    rate = 3 + 3 * torch.rand(1, generator=gen)
    syl = torch.sin(2 * math.pi * rate * t + 6.28 * torch.rand(1, generator=gen)).clamp(min=0) ** 0.7
    pause = (_smooth_noise(n, max(2, int(n / SR * 1.5)), gen) > 0.25).float()
    voiced = voiced * syl * pause
    gaps = (1 - syl.clamp(max=1)) * pause
    fric = _bandpass_noise(n, 3000, 10000, gen) * (gaps > 0.9).float() * 0.3 * torch.rand(1, generator=gen)
    return voiced / (voiced.std() + 1e-8) + fric


def music_like(n: int, gen: torch.Generator) -> torch.Tensor:
    t = torch.arange(n) / SR
    bpm = 90 + 50 * torch.rand(1, generator=gen).item()
    beat = 60.0 / bpm
    root = 110 * 2 ** (torch.randint(0, 12, (1,), generator=gen).item() / 12)
    scale = [0, 2, 4, 5, 7, 9, 11, 12, 14, 16]
    out = torch.zeros(n)
    onset = 0.0
    while onset < n / SR:
        dur = beat * float(torch.randint(1, 4, (1,), generator=gen))
        chord = torch.randint(0, len(scale), (3,), generator=gen)
        decay = 0.3 + 1.2 * torch.rand(1, generator=gen)
        mask = (t >= onset).float() * torch.exp(-(t - onset).clamp(min=0) / decay)
        mask = mask * (t < onset + dur + 0.3).float()
        for idx in chord.tolist():
            f = root * 2 ** (scale[idx] / 12) * (1 + 0.002 * torch.randn(1, generator=gen))
            k = torch.arange(1, 13).view(-1, 1)
            note = (k.float() ** -1.3 * torch.sin(2 * math.pi * k * f * t.view(1, -1)) * (k * f < min(16000, 0.45 * SR))).sum(0)
            out = out + note * mask
        onset += dur
    kick = torch.zeros(n)
    hat = torch.zeros(n)
    hat_noise = _bandpass_noise(n, 6000, min(16000, 0.45 * SR), gen)
    beat_t = 0.0
    while beat_t < n / SR:
        start = int(beat_t * SR)
        seg = torch.arange(n - start) / SR
        kick[start:] += torch.sin(2 * math.pi * (50 * seg + 70 * 0.05 * (1 - torch.exp(-seg / 0.05)))) * torch.exp(-seg / 0.12)
        hstart = int((beat_t + beat / 2) * SR)
        if hstart < n:
            hseg = torch.arange(n - hstart) / SR
            hat[hstart:] += hat_noise[hstart:] * torch.exp(-hseg / 0.03)
        beat_t += beat
    out = out / (out.std() + 1e-8)
    return out + 0.6 * kick + 0.25 * hat


def effects_like(n: int, gen: torch.Generator) -> torch.Tensor:
    t = torch.arange(n) / SR
    out = torch.zeros(n)
    for _ in range(int(torch.randint(2, 7, (1,), generator=gen))):
        start = float(torch.rand(1, generator=gen)) * n / SR
        dur = 0.1 + 0.7 * float(torch.rand(1, generator=gen))
        kind = float(torch.rand(1, generator=gen))
        rel = (t - start).clamp(min=0)
        env = (t >= start).float() * torch.clamp(rel / 0.01, max=1) * torch.exp(-rel / dur)
        if kind < 0.6:
            center = 200 * 40 ** float(torch.rand(1, generator=gen))
            ev = _bandpass_noise(n, center / 1.6, center * 1.6, gen)
        elif kind < 0.85:
            ev = torch.zeros(n)
            ev[int(start * SR) : int(start * SR) + 40] = torch.randn(min(40, n - int(start * SR)), generator=gen) * 8
            env = torch.ones(n)
        else:
            f_start = 300 + 2000 * float(torch.rand(1, generator=gen))
            f_end = 300 + 4000 * float(torch.rand(1, generator=gen))
            inst = f_start + (f_end - f_start) * (rel / dur).clamp(max=1)
            ev = torch.sin(2 * math.pi * torch.cumsum(inst, 0) / SR)
        out = out + ev * env
    return out / (out.std() + 1e-8)


def make_batch(batch: int, n: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    sources = []
    for _ in range(batch):
        stems = []
        for fn in (speech_like, music_like, effects_like):
            gain = 10 ** ((-8 + 12 * float(torch.rand(1, generator=gen))) / 20)
            active = float(torch.rand(1, generator=gen)) < 0.9
            stems.append(fn(n, gen) * 0.05 * gain * active)
        sources.append(torch.stack(stems))
    src = torch.stack(sources).unsqueeze(2)  # [B, S, 1, n]
    return src.sum(1), src  # mix [B, 1, n]


def si_sdr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    ref = ref - ref.mean(-1, keepdim=True)
    est = est - est.mean(-1, keepdim=True)
    proj = (est * ref).sum(-1, keepdim=True) / (ref.square().sum(-1, keepdim=True) + eps) * ref
    return 10 * torch.log10(proj.square().sum(-1) / ((est - proj).square().sum(-1) + eps) + eps)


def build(name: str, *, regions: str = "default", io_layout: str = "regions", mask_points: int = 16):
    if name.startswith("band_dualpath_"):
        from spectral_feature_compression.core.model.band_dualpath_npu import (
            DEFAULT_REGIONS,
            REGIONS_SR24K_R5,
            build_band_dualpath_npu_system,
        )

        return build_band_dualpath_npu_system(
            n_fft=2048,
            hop_length=512,
            fs=SR,
            preset=name[len("band_dualpath_"):],
            regions={"default": DEFAULT_REGIONS, "sr24k_r5": REGIONS_SR24K_R5}[regions],
            io_layout=io_layout,
            mask_points=mask_points,
        )
    if name == "sfc_macaron_lrattn_bn":
        # Mirrors recipes/dnr/models/sfc-small-macaron-lrattn-bn-npu.musical36.2l.r2d64g560.onfly.rt192k
        from spectral_feature_compression.core.model.sfc_small_macaron_lrattn_bn_npu import (
            build_sfc_small_macaron_lrattn_bn_npu_system,
        )

        return build_sfc_small_macaron_lrattn_bn_npu_system(
            n_fft=2048, hop_length=512, fs=SR, n_src=3, n_chan=1, n_bands=36, band_config="musical",
            d_inner=32, d_model=128, ffn_hidden=176, n_separator_layers=2, n_sfc_heads=4,
            learnable_pos_bias=True, attention_rank=2, attention_value_channels=64, temporal_decay=0.995,
            frequency_context_hidden_channels=560, frequency_kernel_size=15, time_kernel_size=2,
            dilation_cycle=[1], encoder_ffn_expansion=2, decoder_ffn_hidden=16, masking=True,
            use_learnable_query=True, scaling=False,
        )
    if name == "dolphin_slim6m_fp512":
        # Mirrors recipes/dnr/models/dolphin-sfc-npu.slim-6m.distill.rt192k.fp512keep475 (student, no teacher).
        from DolphinSFCNPU.training_wrapper import build_dolphin_sfc_npu_system

        return build_dolphin_sfc_npu_system(
            n_fft=4096, hop_length=1024, fs=SR, n_src=3, n_chan=1, preset="slim_6m", band_config="musical",
            freq_preprocess_enabled=True, freq_preprocess_keep_bins=475, freq_preprocess_target_bins=512,
            freq_preprocess_mode="triangular", scaling=False,
        )
    raise ValueError(name)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--sr", type=int, default=44100, help="sample rate of the synthetic stems and the model")
    p.add_argument("--regions", default="default", choices=["default", "sr24k_r5"], help="band_dualpath layout")
    p.add_argument("--io-layout", default="regions", choices=["regions", "slots"], help="band_dualpath I/O layout")
    p.add_argument("--mask-points", type=int, default=16, help="band_dualpath mask points (slots layout)")
    p.add_argument("--tag", default="", help="suffix for the result file name")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seconds", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eval-clips", type=int, default=24)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--save-state", action="store_true", help="also save the trained state_dict")
    args = p.parse_args()
    global SR
    SR = int(args.sr)
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    from spectral_feature_compression.core.loss.snr import ThresSNRLossWithInactiveSource

    net = build(args.model, regions=args.regions, io_layout=args.io_layout, mask_points=args.mask_points)
    run_name = args.model + (f"_{args.tag}" if args.tag else "")
    loss_fn = ThresSNRLossWithInactiveSource(solve_perm=False, n_src=3, zeroref_weight=0.1, only_denominator=False)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    n = int(args.seconds * SR)
    train_gen = torch.Generator().manual_seed(1234)
    eval_gen = torch.Generator().manual_seed(999)
    eval_mix, eval_src = make_batch(args.eval_clips, int(3.0 * SR), eval_gen)

    def evaluate() -> dict[str, float]:
        net.eval()
        with torch.no_grad():
            ests = torch.cat([net(eval_mix[i : i + 4]) for i in range(0, args.eval_clips, 4)])
        net.train()
        n_eval = min(ests.shape[-1], eval_src.shape[-1])
        ref = eval_src[..., :n_eval]
        active = ref.square().mean(-1) > 1e-8
        score = si_sdr(ests[..., :n_eval], ref)
        base = si_sdr(eval_mix[..., :n_eval].unsqueeze(1).expand_as(ref), ref)
        imp = (score - base)
        names = ("speech", "music", "effects")
        out = {f"sisdri_{nm}": float(imp[:, i][active[:, i]].mean()) for i, nm in enumerate(names)}
        out["sisdri_mean"] = sum(out[f"sisdri_{nm}"] for nm in names) / 3
        return out

    params = sum(q.numel() for q in net.parameters())
    history = []
    start = time.time()
    net.train()
    for step in range(1, args.steps + 1):
        mix, src = make_batch(args.batch, n, train_gen)
        est = net(mix)
        loss = loss_fn(est.transpose(1, 2), src.transpose(1, 2)).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        sched.step()
        if step % max(1, args.steps // 6) == 0 or step == args.steps:
            metrics = evaluate()
            metrics.update(step=step, loss=float(loss), elapsed_s=time.time() - start)
            history.append(metrics)
            print(json.dumps(metrics), flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    result = {"model": args.model, "tag": args.tag, "sr": SR, "regions": args.regions,
              "io_layout": args.io_layout, "mask_points": args.mask_points, "params": params, "steps": args.steps, "batch": args.batch,
              "seconds": args.seconds, "history": history, "final": history[-1]}
    (args.out / f"{run_name}.json").write_text(json.dumps(result, indent=2))
    if args.save_state:
        torch.save(net.state_dict(), args.out / f"{run_name}.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
