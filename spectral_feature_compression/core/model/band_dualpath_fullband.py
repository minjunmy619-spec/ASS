"""Full-band (e.g. 48 kHz) host processing around a 24 kHz BandDualPathNPU model.

The model is trained at ``sr`` (24 kHz) with ``n_fft`` / ``hop`` (2048 / 512).
A TV pipeline running at ``ratio * sr`` (48 kHz) uses one STFT with
``ratio * n_fft`` / ``ratio * hop`` (4096 / 1024): it has the same bin spacing
(11.72 Hz) and the same frame timing (21.3 ms hop, 85 ms periodic-Hann window),
and its first ``n_fft // 2 + 1`` bins equal ``ratio`` times the model's STFT of
the decimated signal (-39 dB relative error for content below 12 kHz).  So:

1. low band:  ``X_lo = X_full[:1025] / ratio`` -> the model, unchanged;
2. masks for bins 0-1024 come from the model;
3. high band (bins 1025-2048, 12-24 kHz): every stem gets a real gain from its
   share of the separated power in a reference band just below 12 kHz
   (default 9-12 kHz, model bins 768-1023):

   ``E_s = sum_ref |M_s X_lo|^2`` (optionally causally smoothed over frames),
   ``g_s = E_s / sum_s' E_s'``  (equal split when the reference band is silent).

   The gains sum to one, so the stems' high bands add up to the mixture's high
   band.  A linear crossfade over the last ``crossfade_bins`` model bins
   (default 64 bins = 750 Hz) blends the model masks into the gains.
4. one full-rate iSTFT per stem.

Everything here is host (DSP) work; the NPU graph is unchanged.
``tools/online/band_dualpath_host_reference.py`` has the per-frame NumPy version.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from spectral_feature_compression.core.model.band_dualpath_npu import BandDualPathNPUModel


@dataclass(frozen=True)
class FullBandConfig:
    ratio: int = 2  # full-band sample rate / model sample rate
    model_n_fft: int = 2048
    model_hop: int = 512
    ref_bins: tuple[int, int] = (768, 1024)  # model bins of the high-band reference (9-12 kHz at 24 kHz)
    crossfade_bins: int = 64  # model bins blended from mask to high-band gain below the model Nyquist
    smoothing: float = 0.0  # causal EMA coefficient on the reference energies (0 = per frame)
    eps: float = 1e-10

    @property
    def n_fft(self) -> int:
        return self.ratio * self.model_n_fft

    @property
    def hop(self) -> int:
        return self.ratio * self.model_hop

    @property
    def n_freq_model(self) -> int:
        return self.model_n_fft // 2 + 1

    @property
    def n_freq_full(self) -> int:
        return self.n_fft // 2 + 1


def highband_gains(masks: torch.Tensor, spec_lo: torch.Tensor, cfg: FullBandConfig) -> torch.Tensor:
    """Per-stem real gains ``[B, S, M, T]`` for the band above the model Nyquist.

    ``masks``: complex ``[B, S, M, F_lo, T]``; ``spec_lo``: complex ``[B, M, F_lo, T]`` (model input scale).
    """
    lo, hi = cfg.ref_bins
    energy = (masks[..., lo:hi, :] * spec_lo[:, None, :, lo:hi, :]).abs().square().sum(dim=-2)  # [B, S, M, T]
    if cfg.smoothing > 0.0:
        smoothed = []
        running = energy[..., 0]
        for t in range(energy.shape[-1]):
            running = energy[..., t] if t == 0 else cfg.smoothing * running + (1.0 - cfg.smoothing) * energy[..., t]
            smoothed.append(running)
        energy = torch.stack(smoothed, dim=-1)
    total = energy.sum(dim=1, keepdim=True)
    n_src = energy.shape[1]
    return torch.where(total > cfg.eps, energy / total.clamp_min(cfg.eps), torch.full_like(energy, 1.0 / n_src))


def extend_masks(masks: torch.Tensor, spec_lo: torch.Tensor, cfg: FullBandConfig) -> torch.Tensor:
    """Complex model masks ``[B, S, M, F_lo, T]`` -> full-band masks ``[B, S, M, F_full, T]``."""
    f_lo, f_full = cfg.n_freq_model, cfg.n_freq_full
    if masks.shape[-2] != f_lo:
        raise ValueError(f"Expected {f_lo} model bins, got {masks.shape[-2]}")
    gains = highband_gains(masks, spec_lo, cfg).to(masks.dtype)  # [B, S, M, T]
    low = masks
    if cfg.crossfade_bins > 0:
        n = cfg.crossfade_bins
        weight = torch.arange(1, n + 1, device=masks.device, dtype=torch.float32) / (n + 1)  # ramps towards 1
        weight = weight.view(n, 1)
        low = masks.clone()
        low[..., f_lo - n :, :] = (1.0 - weight) * masks[..., f_lo - n :, :] + weight * gains[..., None, :]
    high = gains[..., None, :].expand(*gains.shape[:-1], f_full - f_lo, gains.shape[-1])
    return torch.cat([low, high], dim=-2)


@torch.no_grad()
def separate_fullband(
    model: BandDualPathNPUModel, wav: torch.Tensor, cfg: FullBandConfig = FullBandConfig()
) -> tuple[torch.Tensor, torch.Tensor]:
    """``wav [B, M, N]`` at ``ratio * sr`` -> (full-band stems, model-band-only stems), each ``[B, S, M, N]``.

    The model runs over the whole sequence with its causal forward, which equals
    frame-by-frame streaming.  The second output drops everything above the model
    Nyquist (what a 24 kHz-only pipeline produces) for comparison.
    """
    bsz, n_chan, n_samples = wav.shape
    window = torch.hann_window(cfg.n_fft, device=wav.device, dtype=wav.dtype)
    spec = torch.stft(wav.reshape(bsz * n_chan, n_samples), cfg.n_fft, cfg.hop, window=window, return_complex=True)
    spec = spec.reshape(bsz, n_chan, cfg.n_freq_full, -1)
    spec_lo = spec[:, :, : cfg.n_freq_model] / cfg.ratio
    masks = model.complex_masks(model.core(model.host_features(spec_lo)))  # [B, S, M, F_lo, T]

    full_masks = extend_masks(masks, spec_lo, cfg)
    band_masks = torch.cat([masks, torch.zeros_like(full_masks[..., cfg.n_freq_model :, :])], dim=-2)
    outputs = []
    for mask in (full_masks, band_masks):
        est = mask * spec[:, None]
        if model.mixture_consistency:
            est = est + (spec[:, None] - est.sum(dim=1, keepdim=True)) / model.n_src
        n_src = est.shape[1]
        wave = torch.istft(
            est.reshape(bsz * n_src * n_chan, cfg.n_freq_full, -1), cfg.n_fft, cfg.hop, window=window,
            length=n_samples,
        )
        outputs.append(wave.reshape(bsz, n_src, n_chan, n_samples))
    return outputs[0], outputs[1]


__all__ = ["FullBandConfig", "extend_masks", "highband_gains", "separate_fullband"]
