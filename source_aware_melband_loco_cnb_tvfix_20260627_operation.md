# Source-Aware MelBand Loco-CNB TV Fix Operation

Date: 2026-06-27

## Problem

The trained `source-aware-melband-loco-cnb.tv-stems.robust-lowlat.distill.rt192k.fp512keep475`
variant showed weak separation quality. The review found the recipe was only a
loss/objective overlay on the older low-latency Loco-CNB distillation recipe.

## Findings

1. The TV recipe inherited the HDF5 DnR datamodule from
   `online-soft-band-query-sfc2d.causal96dim.12l.musical64`, not the TV
   on-the-fly stem synthesis profile used by the TVConv TV recipes.
2. The inherited Loco-CNB architecture kept the pooled capacity mixer schedule
   `[2048, 4096, 4096, 2048]`. Those mixers add parameter and graph cost through
   ReduceMean/pointwise branches, but prior ablations showed this capacity is not
   an efficient quality gain.
3. The distillation task weighted mask/logit losses, but the Loco-CNB student did
   not expose `return_aux=True`. The task therefore fell back to waveform-derived
   pseudo masks instead of comparing real model mask tensors.
4. Enabling aux with the inherited `mel_centers` mapping would be wrong for
   full-frequency mask tensors: RoFormer aux masks are full FFT-bin masks while
   Loco-CNB aux masks are in the 512-bin compressed frequency domain. The fixed
   recipes use linear frequency mapping, matching the TVConv distillation setup.
5. Removing pooled mixers without reallocating capacity leaves only about 1.10M
   parameters, so that variant is useful as a diagnostic ablation but is probably
   too small as the main student.
6. A no-distillation run should not use the inherited distillation teacher
   scaffold. It should explicitly set `teacher_model: null` and all teacher
   weights to zero, otherwise the recipe name and resolved config are misleading.

## Changes

- Added `return_aux=True` support to
  `OnlineSourceAwareMelBandLocoCNBStudentSFC2D` and its complex-STFT wrapper.
  The aux contract returns:
  - `mask`
  - `mask_domain: packed_complex_mask`
  - `mask_logits`
  - `mask_logits_domain: source_aware_melband_loco_cnb_complex_mask_logits`
- Added diagnostic recipe:
  `recipes/dnr/models/source-aware-melband-loco-cnb.tvfix-nopool.robust-lowlat.distill.rt192k.fp512keep475/config.yaml`
- Added recommended retraining recipe:
  `recipes/dnr/models/source-aware-melband-loco-cnb.tvfix-strong-nopool.robust-lowlat.distill.rt192k.fp512keep475/config.yaml`
- Added supervised-only capacity probe:
  `recipes/dnr/models/source-aware-melband-loco-cnb.tvfix-capacity-nopool.sup.rt192k.fp512keep475/config.yaml`

## Recommended Recipe

Use the strong no-pool recipe for the next Loco-CNB TV retrain:

```bash
recipes/dnr/models/source-aware-melband-loco-cnb.tvfix-strong-nopool.robust-lowlat.distill.rt192k.fp512keep475/config.yaml
```

It keeps the 36-channel streaming backbone and 177 KiB fp16 state, disables the
pooled mixers, and moves capacity into source-aware decoding and mask prediction:

- `source_channels: 80`
- `n_source_layers: 5`
- `source_fusion_hidden: 320`
- `source_seed_hidden: 320`
- `expander_hidden: 192`
- `mask_hidden: 256`
- `correction_channels: 24`

## Validation Commands

```bash
.venv/bin/python -m pytest tests/test_proposed_separation_models.py \
  -k "loco_cnb_student_npu_forward_streaming_and_recipe"
```

```bash
.venv/bin/python tools/online/export_onnx_online_model.py \
  recipes/dnr/models/source-aware-melband-loco-cnb.tvfix-strong-nopool.robust-lowlat.distill.rt192k.fp512keep475/config.yaml \
  --out /tmp/source_aware_loco_cnb_tvfix_strong_stream.onnx \
  --n-chan 1 --frames 1 --freqs 512 --opset 14 --check --streaming \
  --op-preset edge_npu_recommended --externalize-band-constants \
  --state-meta-out /tmp/source_aware_loco_cnb_tvfix_strong_state.json \
  --deploy-manifest-out /tmp/source_aware_loco_cnb_tvfix_strong_manifest.json \
  --constants-out /tmp/source_aware_loco_cnb_tvfix_strong_constants.npz
```

```bash
.venv/bin/python tools/online/audit_onnx_model.py \
  /tmp/source_aware_loco_cnb_tvfix_strong_stream.onnx \
  --op-preset edge_npu_recommended \
  --state-meta /tmp/source_aware_loco_cnb_tvfix_strong_state.json \
  --budget-kib 192 --budget-dtype fp16 \
  --risk-profile tiger_one_strict_edge
```

## Export Snapshot

Strong no-pool recipe:

- Params: 3.37M
- FP16 initializer estimate: 6.42 MiB
- FP16 streaming state: 173.25 KiB
- Externalized band constants: 168 KiB
- ONNX nodes: 1718
- Disallowed ops: none
- Strict-edge risk: false

This is still much heavier than TVConv SourceAware in node count, so it should be
treated as a quality-recovery Loco-CNB experiment, not the cleanest deployment
candidate.

## Supervised-Only Capacity Probe

For testing Loco-CNB's standalone capacity without teacher distillation, use:

```bash
recipes/dnr/models/source-aware-melband-loco-cnb.tvfix-capacity-nopool.sup.rt192k.fp512keep475/config.yaml
```

This variant disables the teacher completely and spends capacity in source/mask
decoding:

- `source_channels: 112`
- `n_source_layers: 5`
- `source_fusion_hidden: 448`
- `source_seed_hidden: 448`
- `expander_hidden: 256`
- `mask_hidden: 384`
- `correction_channels: 32`

Export snapshot:

- Params: 6.47M
- FP16 initializer estimate: 12.33 MiB
- FP16 streaming state: 173.25 KiB
- Externalized band constants: 168 KiB
- ONNX nodes: 1718
- Disallowed ops: none
- Strict-edge risk: false

Interpretation: if this supervised-only capacity variant is still clearly weak,
the limitation is probably structural rather than a training-profile accident.
The strongest suspicion would then be that adaptive Mel routing plus 56-band
Loco-CNB is losing too much full-band detail for speech/music/SFX separation,
and the next model should move away from this Loco-CNB decoder rather than keep
scaling it.

## PCEN Norm-Lite Capacity Probe

New variant:

```bash
recipes/dnr/models/source-aware-melband-loco-cnb.tvfix-capacity-pcen-normlite.sup.rt192k.fp512keep475/config.yaml
```

This keeps the supervised-only capacity settings, but changes the normalization
contract:

- wrapper-side PCEN gain normalization is enabled;
- exported Loco-CNB core uses `norm_type: affine`, replacing RMSNorm reductions
  with per-channel affine calibration;
- internal magnitude feature injection is disabled, so the core avoids the
  magnitude `Sqrt` side path and relies on the PCEN-normalized RI input.

PCEN placement is `after_frequency_preprocessing_before_core`. Deployment must
apply the PCEN gain stage around the ONNX/Circle core and divide the separated
outputs by the same gain, matching `FrequencyPreprocessedOnlineModel`.

Validation commands:

```bash
.venv/bin/python -m pytest tests/test_proposed_separation_models.py -q
.venv/bin/python -m pytest tests/test_online_frequency_preprocessing.py tests/test_npu_export_audit.py -q
.venv/bin/ruff check \
  spectral_feature_compression/core/model/online_sfc_2d.py \
  spectral_feature_compression/core/model/source_aware_melband_strong_student_sfc_2d.py \
  spectral_feature_compression/core/model/source_aware_melband_loco_cnb_student_sfc_2d.py \
  spectral_feature_compression/core/model/proposed_separation_models.py \
  tools/online/export_onnx_online_model.py \
  tools/online/audit_onnx_model.py \
  tests/test_proposed_separation_models.py
```

Export command:

```bash
.venv/bin/python tools/online/export_onnx_online_model.py \
  recipes/dnr/models/source-aware-melband-loco-cnb.tvfix-capacity-pcen-normlite.sup.rt192k.fp512keep475/config.yaml \
  --out /tmp/source_aware_loco_cnb_pcen_normlite_stream.onnx \
  --n-chan 1 --frames 1 --freqs 512 --opset 14 --check --streaming \
  --op-preset edge_npu_recommended --externalize-band-constants \
  --state-meta-out /tmp/source_aware_loco_cnb_pcen_normlite_state.json \
  --deploy-manifest-out /tmp/source_aware_loco_cnb_pcen_normlite_manifest.json \
  --constants-out /tmp/source_aware_loco_cnb_pcen_normlite_constants.npz
```

Audit command:

```bash
.venv/bin/python tools/online/audit_onnx_model.py \
  /tmp/source_aware_loco_cnb_pcen_normlite_stream.onnx \
  --op-preset edge_npu_recommended \
  --state-meta /tmp/source_aware_loco_cnb_pcen_normlite_state.json \
  --budget-kib 192 --budget-dtype fp16 \
  --risk-profile tiger_one_strict_edge
```

Export snapshot:

- Params: 6,473,887
- Core RMSNorm modules: 0
- Core affine norm modules: 54
- FP16 initializer estimate: 12.33 MiB
- Core FP16 streaming state: 173.25 KiB
- PCEN FP16 state: 1.00 KiB
- Total deployment FP16 streaming state: 174.25 KiB
- Externalized band constants: 168.00 KiB
- ONNX nodes: 1265
- ONNX ops: Add, Concat, Constant, Conv, Div, Identity, MatMul, Mul,
  ReduceSum, Sigmoid, Slice, Softmax, Split, Sub, Transpose
- Disallowed ops: none
- Strict-edge risk: false

Compared with the supervised-only capacity probe, node count drops from 1718 to
1265 by removing RMSNorm reduction chains and the internal magnitude feature
path.  This is a better Loco-CNB deployment probe, but it is still a large
parameter-payload model; the 192 KiB target is satisfied for streaming state
only, not for state plus weights/constants.
