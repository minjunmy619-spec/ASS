# BandDualPathNPU at 24 kHz with a 3-in / 3-out ABI ("slots" layout)

Date: 2026-10-03.  Follows `band_dualpath_npu_20260928_operation.md`.

## 1. Requirements

- Train at `sr=24000`, `n_fft=2048`, `hop=512` (11.72 Hz bins, Nyquist 12 kHz,
  46.875 frames/s, 85 ms window).
- A better band split for 24 kHz.
- Graph inputs/outputs reduced from 5-7 to 3-4 **without adding memory ops**
  (Slice / Split / Concat / Reshape / Transpose).

## 2. Band split for 24 kHz (`REGIONS_SR24K_R5`)

The 44.1 kHz layout reused at 24 kHz gives 47 / 188 / 984 Hz bands with 4x and
5.25x width jumps.  The new layout doubles the band width per region:

| Region | Bins | Frequency | Bands x width | Content |
|---|---|---|---|---|
| R0 | 0-95 | 0-1.125 kHz | 24 x 46.9 Hz | F0, F1, bass/kick, music fundamentals |
| R1 | 96-191 | 1.125-2.25 kHz | 12 x 93.8 Hz | F2, upper harmonics |
| R2 | 192-383 | 2.25-4.5 kHz | 12 x 187.5 Hz | F3, consonants (intelligibility) |
| R3 | 384-639 | 4.5-7.5 kHz | 8 x 375 Hz | sibilants, presence, attacks |
| R4 | 640-1023 | 7.5-12 kHz | 6 x 750 Hz | air, cymbals, noisy effects |

62 bands; bin 1024 (Nyquist) reuses the bin-1023 mask.  The 2-4.5 kHz speech
range has 12 tokens (vs 6 before).

## 3. 3-in / 3-out ABI without extra memory ops (`io_layout="slots"`)

Approaches compared:

| Approach | Graph I/O | Memory ops (`medium_noattn`) | Result |
|---|---|---|---|
| Per-region tensors (`regions`) | 7 in / 7 out | 12 | baseline |
| One permuted spectrum + in-graph Split/Reshape/Concat | 3 / 3 | 24 (+1 Split, +10 Reshape, +1 Concat) | rejected: adds memory ops |
| **Block-sparse slot input + shared P-point mask head (`slots`)** | **3 / 3** | **10** | **chosen** |

### Input: block-sparse slots

`BandLayout.pack_slots` writes the packed (re/im) compressed spectrum into one
tensor `spectrum [1, 2M * sum(W), 1, K] = [1, 248, 1, 62]`.  Region `r` owns
channels `[2M*off_r, 2M*(off_r+W_r))`; inside its slot a band uses the same
`c * W + w` channel order as the per-region layout, and all other channels of
the band are zero (13% non-zero).  One dense 1x1 Conv2d (248 -> 96) therefore
computes exactly the per-region embeddings (test
`test_slots_embedding_equals_per_region_embeddings`); per-region embedding
biases fold into the per-band position embedding.  This removes the
embedding Concat and adds no Reshape.

### Output: one mask head with P=16 points per band

A single shared GLU mask head emits `mask_points [1, 2SM*P, 1, K] = [1, 96, 1, 62]`
(channel `c * P + p`, `c` = (stem, mic, re/im)).  The host expands points to
bins with `BandLayout.unpack_points`:

- `W <= P` (R0-R2, up to 4.5 kHz): the first `W` points are the per-bin masks (exact).
- `W > P`: linear interpolation of the P points from the first to the last bin
  of the band (R3: one point per 2 bins = 23 Hz; R4: one point per 4 bins = 47 Hz).

The training wrapper uses the same expansion, so training and deployment
agree.  This removes the Split before the heads.  It is a modelling change
(shared head, interpolated masks above 4.5 kHz), measured in section 6.

### Remaining memory ops

The 10 remaining ops (8 Slice + 2 Concat) unpack/repack the 6 band-GRU and 2
scene-GRU state rows; removing them would need one state tensor per block,
i.e. more I/O tensors.  `medium` adds 4 Transpose + 8 Reshape from its two
band-attention layers (22 total).

## 4. Measured budgets (`medium_noattn`, 24 kHz)

| | 44.1 kHz layout at 24 kHz, `regions` | R5, `regions` | **R5, `slots` P=16** |
|---|---|---|---|
| Bands | 48 | 62 | 62 |
| Params | 1.97M | 2.04M | 1.86M |
| GMAC/s | 1.34 | 1.70 | 1.76 |
| Graph I/O | 5 / 5 | 7 / 7 | **3 / 3** |
| ONNX nodes | 302 | 316 | **286** |
| Memory ops | 12 | 12 | **10** |
| uint8 ABI per call | 54.5 KiB | 67.6 KiB | 80.5 KiB |
| NPU rule violations | 0 | 0 | 0 |

`medium` (with attention) in `slots`: 1.86M params, 1.83 GMAC/s, 324 nodes,
22 memory ops, 3 / 3 I/O.

ABI of the `slots` export:

```text
inputs : spectrum [1,248,1,62]  band_state [1,80,6,62]  scene_state [1,384,2,1]
outputs: mask_points [1,96,1,62]  next_band_state [1,80,6,62]  next_scene_state [1,384,2,1]
```

## 5. Host (DSP) contract

`tools/online/band_dualpath_host_reference.py` is the NumPy reference, written
as precomputed tables so the per-frame work is gathers and multiply-adds:

- `HostTables.build(regions, mask_points=16)`: `pack_bin` / `pack_chan`
  (source bin and channel of every NPU input element, -1 = zero) and
  `bin_band` / `bin_p0` / `bin_p1` / `bin_w1` (two-tap expansion per bin).
- `pack_frame(frame)`: power-law compression (|X|^0.3, phase kept) + gather.
- `expand_masks(mask_points)` and `apply_masks(frame, masks)`: masks applied to
  the uncompressed frame.

`python tools/online/band_dualpath_host_reference.py --check` compares it with
the PyTorch training wrapper (pack 2.4e-7, masked output 1.3e-6); the same check
is a unit test.  The export `report.json` now records `stft` and `host_layout`
(regions, slot width, mask points, compression exponent).

## 6. Synthetic A/B (24 kHz)

See the results appended below once the runs finish.

## 7. Files

- `spectral_feature_compression/core/model/band_dualpath_npu.py`:
  `REGIONS_SR24K_R5`, `BandLayout.slot_width / pack_slots / point_expansion /
  unpack_points`, `io_layout` / `mask_points` in the core, training wrapper and
  export wrapper.  The `regions` layout and all 44.1 kHz recipes are unchanged.
- `tools/online/export_band_dualpath_npu.py`: `--fs/--n-fft/--hop` (read from
  the recipe with `--config`), `--regions`, `--io-layout`, `--mask-points`;
  GMAC/s and calibration use the model's sample rate.
- `tools/online/band_dualpath_host_reference.py`: NumPy host reference.
- `tools/online/synthetic_stem_benchmark.py`: `--sr`, `--regions`,
  `--io-layout`, `--mask-points`, `--tag`.
- `recipes/dnr/models/band-dualpath-npu.{medium,medium-noattn}.sr24k.bs62r5.slots-p16.onfly.rt192k/config.yaml`
- `tests/test_band_dualpath_npu.py`: 11 new tests (29 total).

## 8. Commands

```bash
# tests
python -m pytest -q tests/test_band_dualpath_npu.py

# export + audit + calibration + onecc cfg (add --run-one on the ONE machine)
python tools/online/export_band_dualpath_npu.py \
  --config recipes/dnr/models/band-dualpath-npu.medium-noattn.sr24k.bs62r5.slots-p16.onfly.rt192k/config.yaml \
  --out-dir logs/band_dualpath_npu/medium_noattn_sr24k_slots
# --calib-wav must be a 24 kHz wav for these recipes

# training
PYTHONPATH=.:aiaccel .venv/bin/python -m aiaccel.torch.apps.train \
  recipes/dnr/models/band-dualpath-npu.medium-noattn.sr24k.bs62r5.slots-p16.onfly.rt192k/config.yaml

# host reference check
python tools/online/band_dualpath_host_reference.py --check
```

## 9. Notes and open items

- 24 kHz stems are band-limited to 12 kHz.  If the TV path is 48 kHz, either
  accept that or extend the R4 masks to 12-24 kHz on the host (not implemented).
- The 85 ms analysis window raises algorithmic latency from 46 ms (44.1 kHz).
- ONE import / optimize / quantize of the `slots` graph is still to be run on
  the WSL machine; whether ONE inserts layout transposes anywhere is visible in
  `circle-inspect --operators model.opt.circle`.
- Input bytes grow from 2 KB to 15 KB per call (block-sparse zeros) but the
  total uint8 ABI stays at 80.5 KiB.
