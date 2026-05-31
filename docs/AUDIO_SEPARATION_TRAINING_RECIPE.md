# Audio Separation Training Recipe

Date: 2026-05-29

This document closes the training-recipe gap for the current three-stem
`speech`, `music`, `effects` edge-NPU path.  It defines the augmentation flags
and the staged teacher-to-student curriculum now available in configs.

## Augmentation Support

The datamodule now accepts `train_augmentations` and source-activity controls.
Augmentations are applied only to training sources and the mixture is recomputed
after augmentation.

Supported source augmentations:

- Random per-source gain perturbation.
- Polarity inversion.
- Stereo/channel swap.
- Per-source time shift.
- Mild pitch/time perturbation through short resampling and crop/pad.
- Random EQ over coarse frequency bands.
- Mild frequency-band dropout.

Supported dataset controls:

- `remix_sources`: random source selection from per-source datasets.
- `normalize_sources`: per-source RMS normalization before gain.
- `source_gain_db_min` / `source_gain_db_max`: configurable base source gain.
- `crop_retry`: retry count for active training crops.
- `source_activity_threshold`: RMS threshold for activity-aware crop selection.
- `min_active_sources`: active-source count required for non-DM supervised HDF5
  crops when references are available.

## Staged Curriculum

### Stage 1: Teacher

Recipe:

```text
recipes/dnr/models/sfc-locoformer-lite-plus.teacher.stage1-augmented/config.yaml
```

Purpose:

- Train or fine-tune a strong SFC-Locoformer teacher with robust waveform
  augmentation.
- This remains an offline/chunked quality model, not the strict NPU student.

### Stage 1 Student: Supervised RT+ Warm Start

Recipe:

```text
recipes/dnr/models/band-sfc-net-npu.rt-plus.stage1-supervised-aug.rt192k.fp512/config.yaml
```

Purpose:

- Warm-start `BandSFCNet-RT+` before teacher distillation.
- Useful when there is no trained RT+ checkpoint yet.

### Optional PCEN Front-End Ablation

Recipe:

```text
recipes/dnr/models/band-sfc-net-npu.rt-plus.stage1-supervised-aug-pcen.rt192k.fp512/config.yaml
```

Purpose:

- Test wrapper-side PCEN-style gain normalization without changing the exported
  packed core contract.
- The wrapper applies PCEN gain to the complex input before the core, then
  divides the output by the same gain expanded over sources so masking remains on
  the original mixture scale.
- Treat this as an ablation only until DnR validation metrics show it helps.

### Optional DC-Bin Bypass Ablation

Recipe:

```text
recipes/dnr/models/band-sfc-net-npu.rt-plus.stage1-supervised-aug-dcbypass.rt192k.fp512/config.yaml
```

Purpose:

- Keep the STFT DC bin outside the frequency-compressed core so `n_fft=2048`
  uses body bins `1..1024`, then `1024 -> 512` frequency preprocessing.
- The wrapper restores a zero DC bin before iSTFT by default.  This avoids
  source-specific DC prediction inside the exported core and keeps core frequency
  sizes compiler-friendly.
- Treat this as an ablation until it has matched or improved validation metrics.

### Optional 2-Mask Residual-SFX Ablation

Recipe:

```text
recipes/dnr/models/band-sfc-net-npu.rt-plus.2mask-residual-sfx.rt192k.fp512/config.yaml
```

Purpose:

- Test whether the DnR student can predict only explicit Speech and Music masks
  while reconstructing SFX outside the NPU core as `mixture - speech - music`.
- The external training/evaluation contract remains 3 stems in the standard DnR
  order: `speech`, `music`, `effects`.
- The exported BandSFCNetNPU core uses `core_n_src=2`; wrapper-side residual
  reconstruction restores the third source before loss/evaluation/iSTFT.
- Train and score all 3 stems.  The SFX residual is an error bucket, so compare
  per-stem SFX metrics and listening leakage before treating this as a size or
  quality win.
- ONNX/ONE validation has passed for the config-only core export; trained quality
  metrics are still pending.

### Stage 2 Student: Chunk-Causal Distillation

Recipe:

```text
recipes/dnr/models/band-sfc-net-npu.rt-plus.stage2-distill-chunk.rt192k.fp512/config.yaml
```

Purpose:

- Distill from the trained SFC-Locoformer teacher with moderate 6-second chunks.
- Uses teacher output loss, mixture consistency, low-frequency loss,
  silent-source penalty, complex RI, log-magnitude, multi-resolution STFT, and
  transient losses.
- Also enables teacher spectral mask/logit distillation and
  source-activity-aware waveform loss.
- Latent distillation plumbing is available through `student_latent_modules` and
  `teacher_latent_modules`, but remains weight `0.0` in default recipes until a
  shape-compatible teacher/student hook pair is selected.

Required before launch:

- Set `teacher_checkpoint_path` to the trained teacher checkpoint.

### Stage 3 Student: Strict Streaming Fine-Tune

Recipe:

```text
recipes/dnr/models/band-sfc-net-npu.rt-plus.stage3-distill-strict.rt192k.fp512/config.yaml
```

Purpose:

- Fine-tune after Stage 2 with shorter two-second crops and stricter chunk
  validation settings.
- Use this before final ONNX/ONE verification.

Required before launch:

- Set `teacher_checkpoint_path`.
- Set `task.pretrained_model_path` to the Stage 2 student checkpoint.

## Example Commands

Teacher:

```bash
./.venv/bin/python -m spectral_feature_compression.train \
  --config-path recipes/dnr/models/sfc-locoformer-lite-plus.teacher.stage1-augmented \
  --config-name config
```

Student supervised warm start:

```bash
./.venv/bin/python -m spectral_feature_compression.train \
  --config-path recipes/dnr/models/band-sfc-net-npu.rt-plus.stage1-supervised-aug.rt192k.fp512 \
  --config-name config
```

Student supervised PCEN ablation:

```bash
./.venv/bin/python -m spectral_feature_compression.train \
  --config-path recipes/dnr/models/band-sfc-net-npu.rt-plus.stage1-supervised-aug-pcen.rt192k.fp512 \
  --config-name config
```

Student supervised DC-bypass ablation:

```bash
./.venv/bin/python -m spectral_feature_compression.train \
  --config-path recipes/dnr/models/band-sfc-net-npu.rt-plus.stage1-supervised-aug-dcbypass.rt192k.fp512 \
  --config-name config
```

Student supervised 2-mask residual-SFX ablation:

```bash
./.venv/bin/python -m spectral_feature_compression.train \
  --config-path recipes/dnr/models/band-sfc-net-npu.rt-plus.2mask-residual-sfx.rt192k.fp512 \
  --config-name config
```

Student distillation:

```bash
./.venv/bin/python -m spectral_feature_compression.train \
  --config-path recipes/dnr/models/band-sfc-net-npu.rt-plus.stage2-distill-chunk.rt192k.fp512 \
  --config-name config \
  teacher_checkpoint_path=/path/to/teacher.ckpt
```

Strict fine-tune:

```bash
./.venv/bin/python -m spectral_feature_compression.train \
  --config-path recipes/dnr/models/band-sfc-net-npu.rt-plus.stage3-distill-strict.rt192k.fp512 \
  --config-name config \
  teacher_checkpoint_path=/path/to/teacher.ckpt \
  task.pretrained_model_path=/path/to/stage2_student.ckpt
```

## Notes

- STFT/iSTFT remain outside the exported strict NPU core.
- Residual-source recipes keep `n_src=3` externally; only `.model.core` exports
  with `core_n_src=2`.  Device integration must reconstruct the residual stem
  outside the NPU core before final iSTFT/output packaging.
- These recipe variants intentionally do not replace the older baseline configs;
  they are sibling configs for controlled ablation and curriculum training.
- After each trained stage, fill
  `docs/templates/audio_separation_results_manifest.csv` and run the NPU export
  audit before long ONE compile/debug cycles.

## DolphinSFCNPU Distillation Track

After the main BandSFC RT+ path has a usable teacher/student baseline, train the
DolphinSFCNPU second candidate with:

```text
recipes/dnr/models/dolphin-sfc-npu.slim-6m.distill.rt192k.fp512keep475/config.yaml
recipes/dnr/models/dolphin-sfc-npu.slim-8m.distill.rt192k.fp512keep475/config.yaml
```

Use `slim_6m` first.  Treat `slim_8m` as a quality probe only after `slim_6m`
shows a useful quality/state tradeoff.  See
`docs/DOLPHIN_SFC_NPU_DISTILLATION.md` for launch commands and teacher checkpoint
compatibility notes.

## Sparse U-Net Mel-SFC Probe

The sparse low/mid/high Mel-SFC U-Net is now available as an opt-in quality
fallback, not as the primary strict RT+ deployment path.

Recipes:

```text
recipes/musdb18hq/models/sparse-unet-mel-sfc.music.rt192k.fp512keep475/config.yaml
recipes/dnr/models/sparse-unet-mel-sfc.rt192k.fp512keep475/config.yaml
```

Use the MUSDB recipe first because this architecture was added for the
music-first fallback gap.  The default packed core is about `0.54M` parameters
with about `140 KiB` fp16 layer cache at `512` preprocessed frequency bins, but
it still needs trained metrics and ONNX/ONE validation before deployment use.

Example MUSDB launch:

```bash
./.venv/bin/python -m spectral_feature_compression.train \
  --config-path recipes/musdb18hq/models/sparse-unet-mel-sfc.music.rt192k.fp512keep475 \
  --config-name config
```

## SFC-SepReformer Multi-Stem Probe

The early source-split SFC variant tests whether splitting compressed SFC tokens
before reconstruction reduces the burden on the final mask head.  It uses packed
4D source channels, shared source-wise refiner weights, and a shared SFC decoder.

Recipes:

```text
recipes/dnr/models/sfc-sepreformer-multistem.rt192k.fp512keep475/config.yaml
recipes/musdb18hq/models/sfc-sepreformer-multistem.rt192k.fp512keep475/config.yaml
```

Use the DnR recipe first because this idea should help the three-stem universal
task most.  The default DnR packed core is about `0.04M` parameters with about
`112 KiB` fp16 layer cache at `512` preprocessed bins; the MUSDB four-stem core
is about `0.04M` parameters with about `144 KiB` fp16 layer cache.  Treat both
as ablations until trained metrics and ONNX/ONE export are available.

Example DnR launch:

```bash
./.venv/bin/python -m spectral_feature_compression.train \
  --config-path recipes/dnr/models/sfc-sepreformer-multistem.rt192k.fp512keep475 \
  --config-name config
```

## SFC Residual-Refinement Probe

The residual-refinement SFC variant tests the useful part of the Mamba2 / TS-BSMamba2
recommendation without introducing unsupported Mamba2 kernels.  It adds a causal
dilated long-temporal branch after SFC compression and a second-stage full-band
correction head that predicts a packed complex residual on top of the first
masked estimate.

Recipes:

```text
recipes/dnr/models/sfc-residual-refinement.rt192k.fp512keep475/config.yaml
recipes/musdb18hq/models/sfc-residual-refinement.rt192k.fp512keep475/config.yaml
```

Use DnR first, then compare directly with `BandSFCNet-RT+` and
`online-soft-band-query-sfc2d` under the same loss stack.  The default packed
core is about `0.03M` parameters with about `144 KiB` fp16 layer cache at `512`
preprocessed bins.  Keep true Mamba2 outside the strict NPU path until a
teacher/middle-tier experiment proves it is worth the export risk.

Example DnR launch:

```bash
./.venv/bin/python -m spectral_feature_compression.train \
  --config-path recipes/dnr/models/sfc-residual-refinement.rt192k.fp512keep475 \
  --config-name config
```

## Band-Mapping Ablation Track

The adaptive mel / overlapped perceptual band-mapping track now has explicit
fixed and mel-overlap SFC recipes, plus teacher-side SFC-CA and SFC-Mamba rows
for comparison.

Deployable/online SFC ablations:

```text
recipes/dnr/models/bandmap-ablation.fixed80.rt192k.fp512keep475/config.yaml
recipes/dnr/models/bandmap-ablation.mel-overlap80.rt192k.fp512keep475/config.yaml
recipes/musdb18hq/models/bandmap-ablation.fixed80.rt192k.fp512keep475/config.yaml
recipes/musdb18hq/models/bandmap-ablation.mel-overlap80.rt192k.fp512keep475/config.yaml
```

Teacher/non-strict comparison rows:

```text
recipes/dnr/models/bandmap-ablation.sfc-ca80.teacher/config.yaml
recipes/musdb18hq/models/bandmap-ablation.sfc-ca80.teacher/config.yaml
recipes/musdb18hq/models/bandmap-ablation.sfc-mamba64.teacher/config.yaml
```

The mel-overlap builder exposes `low_freq_hz`, `low_freq_band_fraction`,
`overlap_factor`, and `low_freq_overlap_factor` for bass/music preservation.  The
default online packed core is about `0.03M` parameters with about `90 KiB` fp16
layer cache at `512` preprocessed bins.

Example DnR mel-overlap launch:

```bash
./.venv/bin/python -m spectral_feature_compression.train \
  --config-path recipes/dnr/models/bandmap-ablation.mel-overlap80.rt192k.fp512keep475 \
  --config-name config
```
