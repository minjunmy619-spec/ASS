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
