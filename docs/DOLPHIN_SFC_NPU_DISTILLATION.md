# DolphinSFCNPU Distillation

Date: 2026-05-29

This note documents the DolphinSFCNPU distillation path added for audit item 12.
It is a second deployable candidate to compare against `BandSFCNet-RT+`, not a
replacement for the current main RT+ student.

## Recipes

Primary candidate:

```text
recipes/dnr/models/dolphin-sfc-npu.slim-6m.distill.rt192k.fp512keep475/config.yaml
```

Larger quality probe:

```text
recipes/dnr/models/dolphin-sfc-npu.slim-8m.distill.rt192k.fp512keep475/config.yaml
```

Both recipes:

- use `TeacherStudentDistillationTask`;
- require `teacher_checkpoint_path` before launch;
- use an SFC-Locoformer-Lite+ teacher builder by default;
- enable teacher output loss, mixture consistency, low-frequency loss,
  silent-source penalty, complex RI loss, log-magnitude loss,
  multi-resolution STFT loss, and transient loss;
- use the same training augmentation stack as the BandSFC RT+ staged recipes;
- use `fp512keep475` preprocessing for the DolphinSFCNPU core.

Budget note:

- `slim_6m`: about 5.17M params and 162 KiB fp16 state at `n_freq=257`.
- `slim_8m`: about 6.55M params and 174 KiB fp16 state at `n_freq=257` after
  the preset was tightened to stay inside the repo's edge budget tests.

## Teacher Checkpoint Compatibility

The default teacher builder uses:

```yaml
teacher_n_fft: ${n_fft}
teacher_hop_length: ${hop_length}
```

Override these if the teacher checkpoint was trained with a different STFT
contract, for example:

```bash
teacher_n_fft=2048 teacher_hop_length=512
```

The teacher and student only need waveform output agreement for distillation;
their internal STFT contracts may differ as long as the teacher checkpoint
matches the configured teacher model.

## Example Commands

Slim 6m distilled run:

```bash
./.venv/bin/python -m spectral_feature_compression.train \
  --config-path recipes/dnr/models/dolphin-sfc-npu.slim-6m.distill.rt192k.fp512keep475 \
  --config-name config \
  teacher_checkpoint_path=/path/to/sfc_locoformer_teacher.ckpt
```

Slim 8m distilled probe:

```bash
./.venv/bin/python -m spectral_feature_compression.train \
  --config-path recipes/dnr/models/dolphin-sfc-npu.slim-8m.distill.rt192k.fp512keep475 \
  --config-name config \
  teacher_checkpoint_path=/path/to/sfc_locoformer_teacher.ckpt
```

## Required Comparison

After training, compare against BandSFC RT+ under the same manifest fields:

- DnR SNR / SI-SDR by stem and average.
- Parameters.
- GMAC/s.
- fp16/fp32 streaming state KiB.
- ONNX node count and strict-edge risk profile.
- ONE import / optimize / quantize / `circle-verify` status.
- Listening notes for bass, speech leakage, effects leakage, musical noise, and
  chunk-boundary artifacts.
