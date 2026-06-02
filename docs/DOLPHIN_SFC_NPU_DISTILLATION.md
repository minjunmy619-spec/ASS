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

Mixture-consistent repair candidate after the first `slim_6m` run plateaued
around validation SNR `6.9`:

```text
recipes/dnr/models/dolphin-sfc-npu.slim-6m.distill-mixsoftmax.rt192k.fp512keep475/config.yaml
```

Both recipes:

- use `TeacherStudentDistillationTask`;
- require `teacher_checkpoint_path` before launch;
- use an SFC-Locoformer-Lite+ teacher builder by default;
- enable teacher output loss, mixture consistency, low-frequency loss,
  silent-source penalty, complex RI loss, log-magnitude loss,
  multi-resolution STFT loss, transient loss, teacher spectral mask/logit
  distillation, and source-activity-aware waveform loss;
- use the same training augmentation stack as the BandSFC RT+ staged recipes;
- use `fp512keep475` preprocessing for the DolphinSFCNPU core.

The `distill-mixsoftmax` overlay keeps the same training setup but selects
`task.model.mask_activation: softmax`.  This uses a source-axis softmax over
the real-valued gain logits, so the reduced-frequency estimates sum to the
projected mixture before full-frequency synthesis.  The Dolphin output head now
also initializes to a conservative mixture split instead of random source
gains.

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
PYTHONPATH=$PWD/aiaccel:$PWD ./.venv/bin/python -m aiaccel.torch.apps.train \
  recipes/dnr/models/dolphin-sfc-npu.slim-6m.distill.rt192k.fp512keep475/config.yaml \
  teacher_checkpoint_path=/path/to/sfc_locoformer_teacher.ckpt
```

Slim 8m distilled probe:

```bash
PYTHONPATH=$PWD/aiaccel:$PWD ./.venv/bin/python -m aiaccel.torch.apps.train \
  recipes/dnr/models/dolphin-sfc-npu.slim-8m.distill.rt192k.fp512keep475/config.yaml \
  teacher_checkpoint_path=/path/to/sfc_locoformer_teacher.ckpt
```

Slim 6m mixture-consistent repair run:

```bash
PYTHONPATH=$PWD/aiaccel:$PWD ./.venv/bin/python -m aiaccel.torch.apps.train \
  recipes/dnr/models/dolphin-sfc-npu.slim-6m.distill-mixsoftmax.rt192k.fp512keep475/config.yaml \
  teacher_checkpoint_path=/path/to/sfc_locoformer_teacher.ckpt
```

During validation, check all three metrics:

- `validation/snr`: student vs ground truth.
- `validation/teacher_snr`: teacher checkpoint vs ground truth.
- `validation/student_teacher_snr`: student agreement with teacher.

If `validation/teacher_snr` is below the target, fix the teacher checkpoint or
`teacher_n_fft` / `teacher_hop_length` compatibility before tuning the student.

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

## Export Status

- `dolphin-sfc-npu.large-6m.fp512keep475` and
  `dolphin-sfc-npu.slim-6m.distill.rt192k.fp512keep475` both pass ONNX export,
  ONE import, ONE optimize, ONE quantize, and `circle-verify` in the focused
  2026-05-30 validation run.
- Both simplified ONNX graphs have `314` nodes and remaining conservative audit
  flags `activation_matmul_rank_le3=2` and `transpose=7`.
- Trained quality metrics are still missing.
- `dolphin-sfc-npu.slim-6m.distill-mixsoftmax.rt192k.fp512keep475` passed the
  focused 2026-06-02 verifier through ONNX export, forced onnxsim, ONE import,
  ONE optimize, and channel-wise ONE quantization.  The forced onnxsim flag is
  required for this verifier path; otherwise ONE import can fail on unsimplified
  shape ops with `invalid tensor dimension size`.
- For `--disable-masking` export, the deploy manifest now records
  `mask_postprocess.activation: softmax`; external postprocessing must use that
  activation for the mixsoftmax recipe.
