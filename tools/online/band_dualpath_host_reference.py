#!/usr/bin/env python3
"""NumPy reference of the BandDualPathNPU host (DSP) side for the ``slots`` I/O layout.

Per STFT frame the host does two table-driven steps around one NPU call:

1. ``pack_frame``: power-law compress the complex frame and gather it into the
   block-sparse NPU input ``spectrum [1, 2M*sum(W), 1, K]``.
2. ``expand_masks``: turn the NPU output ``mask_points [1, 2SM*P, 1, K]`` into
   per-bin complex masks (exact copy for bands of W <= P bins, two-tap linear
   interpolation for wider bands) and apply them to the *uncompressed* frame.

All index / weight tables are precomputed once by ``HostTables.build`` and are
plain integer and float arrays, so the per-frame work is gathers and
multiply-adds that map directly to DSP code.  ``python
tools/online/band_dualpath_host_reference.py --check`` verifies the tables
against the PyTorch training wrapper (``BandDualPathNPUModel``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HostTables:
    n_freq: int
    n_chan: int  # microphones / audio channels M
    n_src: int  # stems S
    n_bands: int  # K
    slot_width: int  # sum of band widths W over regions
    mask_points: int  # P
    # pack: for NPU input channel c_in in [0, 2M*sum(W)) and band k -> (source bin, re/im/mic channel) or -1
    pack_bin: np.ndarray  # int32 [2M*sum(W), K], -1 = zero
    pack_chan: np.ndarray  # int32 [2M*sum(W), K]
    # expand: for each frequency bin f -> band k, two point indices and weights
    bin_band: np.ndarray  # int32 [F]
    bin_p0: np.ndarray  # int32 [F]
    bin_p1: np.ndarray  # int32 [F]
    bin_w1: np.ndarray  # float32 [F], mask = (1 - w1) * point[p0] + w1 * point[p1]

    @classmethod
    def build(
        cls,
        regions: Sequence[Sequence[int]],
        *,
        n_freq: int = 1025,
        n_chan: int = 1,
        n_src: int = 3,
        mask_points: int = 16,
    ) -> HostTables:
        regions = [tuple(int(v) for v in r) for r in regions]
        in_ch = 2 * n_chan
        n_bands = sum((e - s) // w for s, e, w in regions)
        slot_width = sum(w for _, _, w in regions)
        pack_bin = -np.ones((in_ch * slot_width, n_bands), dtype=np.int32)
        pack_chan = np.zeros((in_ch * slot_width, n_bands), dtype=np.int32)
        bin_band = np.zeros(n_freq, dtype=np.int32)
        bin_p0 = np.zeros(n_freq, dtype=np.int32)
        bin_p1 = np.zeros(n_freq, dtype=np.int32)
        bin_w1 = np.zeros(n_freq, dtype=np.float32)

        channel_offset = band_offset = 0
        for start, end, width in regions:
            for kb in range((end - start) // width):
                k = band_offset + kb
                for c in range(in_ch):
                    for w in range(width):
                        row = channel_offset + c * width + w  # same c * W + w order as BandLayout.pack
                        pack_bin[row, k] = start + kb * width + w
                        pack_chan[row, k] = c
                for w in range(width):
                    f = start + kb * width + w
                    bin_band[f] = k
                    if width <= mask_points:
                        bin_p0[f], bin_p1[f], bin_w1[f] = w, w, 0.0
                    else:
                        pos = w * (mask_points - 1) / (width - 1)
                        p0 = min(int(np.floor(pos)), mask_points - 2)
                        bin_p0[f], bin_p1[f], bin_w1[f] = p0, p0 + 1, pos - p0
            channel_offset += in_ch * width
            band_offset += (end - start) // width
        covered = regions[-1][1]
        for f in range(covered, n_freq):  # Nyquist tail reuses the last covered bin
            bin_band[f], bin_p0[f], bin_p1[f], bin_w1[f] = (
                bin_band[covered - 1],
                bin_p0[covered - 1],
                bin_p1[covered - 1],
                bin_w1[covered - 1],
            )
        return cls(n_freq, n_chan, n_src, n_bands, slot_width, int(mask_points), pack_bin, pack_chan, bin_band,
                   bin_p0, bin_p1, bin_w1)


def compress_frame(frame: np.ndarray, exponent: float = 0.3, eps: float = 1e-12) -> np.ndarray:
    """Complex frame ``[M, F]`` -> packed real ``[2M, F]`` with magnitude ``|X|**exponent`` and phase kept."""
    scale = (frame.real**2 + frame.imag**2 + eps) ** (0.5 * (exponent - 1.0))
    packed = np.empty((2 * frame.shape[0], frame.shape[1]), dtype=np.float32)
    packed[0::2] = frame.real * scale
    packed[1::2] = frame.imag * scale
    return packed


def pack_frame(frame: np.ndarray, tables: HostTables, exponent: float = 0.3) -> np.ndarray:
    """Complex STFT frame ``[M, F]`` -> NPU input ``spectrum [1, 2M*sum(W), 1, K]``."""
    packed = compress_frame(frame, exponent)
    valid = tables.pack_bin >= 0
    out = np.zeros(tables.pack_bin.shape, dtype=np.float32)
    out[valid] = packed[tables.pack_chan[valid], tables.pack_bin[valid]]
    return out[None, :, None, :]


def expand_masks(mask_points: np.ndarray, tables: HostTables) -> np.ndarray:
    """NPU output ``[1, 2SM*P, 1, K]`` -> complex masks ``[S, M, F]``."""
    pts = mask_points.reshape(tables.n_src * tables.n_chan * 2, tables.mask_points, tables.n_bands)
    k = tables.bin_band
    lo = pts[:, tables.bin_p0, k]
    hi = pts[:, tables.bin_p1, k]
    bins = (1.0 - tables.bin_w1) * lo + tables.bin_w1 * hi  # [2SM, F], channel order (s, m, re/im)
    bins = bins.reshape(tables.n_src, tables.n_chan, 2, tables.n_freq)
    return bins[:, :, 0] + 1j * bins[:, :, 1]


def apply_masks(frame: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Complex frame ``[M, F]`` and masks ``[S, M, F]`` -> separated frames ``[S, M, F]``."""
    return masks * frame[None]


def _check() -> None:
    import torch

    from spectral_feature_compression.core.model.band_dualpath_npu import (
        PRESETS,
        REGIONS_SR24K_R5,
        BandDualPathNPUModel,
    )

    torch.manual_seed(0)
    model = BandDualPathNPUModel(regions=REGIONS_SR24K_R5, io_layout="slots", mask_points=16,
                                 **PRESETS["medium_noattn"]).eval()
    tables = HostTables.build(REGIONS_SR24K_R5, mask_points=16)
    spec = torch.randn(1, 1, 1025, 1, dtype=torch.complex64) * 2
    ref_in = model.host_features(spec)[0].numpy()
    got_in = pack_frame(spec[0, :, :, 0].numpy(), tables)
    points = torch.randn(1, model.core.out_channels * 16, 1, model.core.n_bands)
    ref_est = model.apply_masks(spec, (points,))[..., 0].numpy()  # [1, S, M, F]
    got_est = apply_masks(spec[0, :, :, 0].numpy(), expand_masks(points.numpy(), tables))
    print("pack max err", float(np.abs(ref_in - got_in).max()), "| masked frame max err",
          float(np.abs(ref_est[0] - got_est).max()))


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    import sys

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="compare against the PyTorch training wrapper")
    if parser.parse_args().check:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        _check()
