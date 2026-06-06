# Adaptive Mel Loco-CNB Low-SNR Follow-up

Date: 2026-06-05

## Trigger

A supervised BandSFCNetNPU adaptive-mel Loco-CNB run still reports roughly
`validation/snr ~= 3.5 dB` after about 30 epochs.  That is too low for a final
DnR Speech/Music/Effects candidate and means the issue is no longer just the
initial LR instability observed in the earlier run.

The local workspace does not contain the configured HDF5 files under
`recipes/dnr/hdf5`, so this pass could not inspect the exact validation samples
or reproduce the training metric locally.

## Additional diagnosis

The previous clean pointwise recipe was intended as a structural diagnostic, but
profiling showed it is not a good branch to widen:

```text
adaptive_mel_loco_cnb_stable_soft_band_query:
  params = 2,852,491
  fp16 state = 185.62 KiB
  profiled core MAC/frame = 15,664,320
  profiled core GMAC/s @ 44100/512 = 1.349

adaptive_mel_loco_cnb_stable_soft_query.residual_sfx:
  core_n_src = 2, wrapper output n_src = 3
  params = 2,852,417
  fp16 state = 185.62 KiB
  profiled core MAC/frame = 15,627,456
  profiled core GMAC/s @ 44100/512 = 1.346

adaptive_mel_loco_cnb_band56_soft_band_query:
  params = 2,210,395
  fp16 state = 168.44 KiB
  profiled core MAC/frame = 12,522,272
  profiled core GMAC/s @ 44100/512 = 1.079

adaptive_mel_loco_cnb_clean_soft_band_query:
  params = 876,604
  fp16 state = 185.62 KiB
  profiled core MAC/frame = 47,307,456
  profiled core GMAC/s @ 44100/512 = 4.075
```

The clean recipe is smaller in parameters, but its pointwise per-band mixers are
expensive because the MLP is applied at every compressed band.  Widening it to a
medium schedule `[1024, 2048, 2048, 2048, 1024]` would increase the profile to
about `5.904 GMAC/s`, so it should not be used as the next deployment-quality
branch under the current `<3 GMAC/s` target.

Conclusion:

- do **not** keep widening the clean pointwise branch;
- use the compute-safe stable pooled branch for the next controlled ablation;
- reduce the explicit prediction task with residual SFX;
- if supervised training remains near 3.5 dB, move to teacher distillation
  rather than more supervised-only training.

## New recipes

Added a supervised residual-SFX ablation:

```text
recipes/dnr/models/band-sfc-net-npu.adaptive-mel-loco-cnb.stable-soft-query.residual-sfx.rt192k.fp512keep475/config.yaml
```

This inherits the stable adaptive-mel Loco-CNB recipe, but the NPU core predicts
only Speech and Music (`core_n_src=2`).  The wrapper reconstructs Effects/SFX as:

```text
sfx = mixture - speech - music
```

The public output order remains the DnR contract:

```text
[speech, music, sfx]
```

Added a teacher-guided version:

```text
recipes/dnr/models/band-sfc-net-npu.adaptive-mel-loco-cnb.stable-soft-query.residual-sfx.distill.rt192k.fp512keep475/config.yaml
```

This keeps the residual-SFX student shape and uses the existing
`TeacherStudentDistillationTask` with the SFC-Locoformer-lite-plus teacher
builder.

## Suggested next run order

### 1. Short supervised residual-SFX check

```bash
cd /home/cmj/works/ASS
PYTHONPATH=.:aiaccel .venv/bin/python -m aiaccel.torch.apps.train \
  recipes/dnr/models/band-sfc-net-npu.adaptive-mel-loco-cnb.stable-soft-query.residual-sfx.rt192k.fp512keep475/config.yaml
```

Run only long enough to see whether validation moves above the old `3.5 dB`
plateau.  If it stays near the same value, the bottleneck is probably not only
the explicit SFX head.

### 2. Distill from a stronger teacher

Set `teacher_checkpoint_path` to a trained teacher checkpoint.  If continuing
from the supervised residual-SFX student, also set `task.pretrained_model_path`:

```bash
cd /home/cmj/works/ASS
PYTHONPATH=.:aiaccel .venv/bin/python -m aiaccel.torch.apps.train \
  recipes/dnr/models/band-sfc-net-npu.adaptive-mel-loco-cnb.stable-soft-query.residual-sfx.distill.rt192k.fp512keep475/config.yaml \
  teacher_checkpoint_path=/path/to/teacher.ckpt \
  task.pretrained_model_path=/path/to/residual_sfx_student.ckpt
```

If there is no good teacher checkpoint yet, train or select the teacher first;
otherwise the student is still learning only from the same supervised signal.

## Validation commands

Recipe/test validation:

```bash
cd /home/cmj/works/ASS
PYTHONPATH=. .venv/bin/python -m pytest tests/test_proposed_separation_models.py -q
```

NPU compile verification for the new supervised residual-SFX recipe:

```bash
cd /home/cmj/works/ASS
.venv/bin/python tools/online/verify_npu_variants.py \
  --mode recipe \
  --recipe-name-contains adaptive-mel-loco-cnb.stable-soft-query.residual-sfx \
  --run-name band_sfc_adaptive_mel_loco_cnb_stable_residual_sfx_20260605 \
  --force-onnxsim-large-shape-ops \
  --quantize-layer-fallback
```

Stateful streaming verification should also be run after stateless verification:

```bash
cd /home/cmj/works/ASS
.venv/bin/python tools/online/verify_npu_variants.py \
  --mode recipe \
  --recipe-name-contains adaptive-mel-loco-cnb.stable-soft-query.residual-sfx \
  --run-name band_sfc_adaptive_mel_loco_cnb_stable_residual_sfx_streaming_20260605 \
  --force-onnxsim-large-shape-ops \
  --quantize-layer-fallback \
  --streaming
```

Local results:

```text
pytest targeted recipe tests: PASS
BandSFCNetNPU smoke tests: PASS
ONE stateless verification: PASS for supervised and distill residual-SFX recipes
ONE streaming verification: PASS for supervised and distill residual-SFX recipes
```

The generated verifier artifact directories were removed after validation to
avoid leaving about 203 MB of untracked ONNX/Circle files in `logs/`.  Re-run
the commands above if the artifacts are needed.

Stateless export summary for the supervised residual-SFX recipe:

```text
simplified non-Constant nodes: 687
params: 2,852,417
streaming fp16 state: 185.62 KiB
ONE result: model.circle, model.opt.circle, model.q.circle PASS
```

## Metric diagnostics to check in the training run

For the current low-SNR run, compare these before deciding the architecture is
bad:

1. `validation/loss` versus `validation/snr`.
2. Per-stem SNR if available: Speech, Music, SFX.
3. Direct short validation (`model(wav)`) versus CSS validation (`model.css`).
4. Mixture-split baseline on the same validation files.
5. Activity ratio/silent-source ratio in `cv_unsegmented.hdf5`.

If direct short validation is much better than CSS validation, debug chunk
stitching/evaluation before changing the model.  If SFX is the only very bad
stem, the residual-SFX recipe is the right next ablation.  If all stems are near
3.5 dB, supervised training is likely insufficient and teacher distillation is
needed.
