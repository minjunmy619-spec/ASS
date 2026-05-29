# Audio Research Gap Fill Operation

Date: 2026-05-29

## Scope

This pass addressed implementable gaps from
`docs/AUDIO_SEPARATION_RESEARCH_GAP_AUDIT.md` without attempting long model
training.

## Changes

- Added TIGER/ONE-derived strict-edge ONNX risk checks to
  `tools/online/audit_onnx_model.py`.
- Wired the same risk profile into `tools/online/export_verify_mlir.py`.
- Documented the strict-edge checklist in `docs/NPU_VARIANT_VERIFIER_GUIDE.md`.
- Added `CompositeSeparationSpectralLoss` with complex RI, log magnitude,
  multi-resolution STFT, and transient terms.
- Wired those opt-in loss terms into `TeacherStudentDistillationTask`.
- Enabled the new spectral loss weights in the DnR BandSFC RT+ and EdgeFusion
  distillation recipes.
- Added the primary three-stem benchmark contract and result/listening CSV
  templates.
- Updated the audit doc to show which gaps are now filled and which remain.

## RT+ ONNX Audit Observation

Command:

```bash
./.venv/bin/python tools/online/audit_onnx_model.py \
  logs/npu_verify_general/band_sfc_net_rt_plus_stream.onnx \
  --op-preset edge_npu_recommended \
  --risk-profile tiger_one_strict_edge \
  --transpose-threshold 500
```

Result summary:

- Disallowed ops: none.
- `Tile`: 0.
- `ConstantOfShape`: 0.
- `Expand`: 0.
- Strict-edge risk flags remain:
  - dynamic `Slice` bounds: 53.
  - dynamic `Slice` with dynamic non-axis dims: 49.
  - rank<=3 activation `MatMul`: 14.

Interpretation: RT+ is cleaner than the old Tile/Expand/ConstantOfShape graphs,
but should not be treated as fully strict-edge clean until those Slice/MatMul
patterns are rewritten or proven safe through full ONE import/quantization.

## Verification

Commands:

```bash
./.venv/bin/python -m py_compile \
  tools/online/audit_onnx_model.py \
  tools/online/export_verify_mlir.py \
  spectral_feature_compression/core/loss/composite_separation.py \
  spectral_feature_compression/core/tasks/distillation_task.py

./.venv/bin/python -m ruff check \
  tools/online/audit_onnx_model.py \
  tools/online/export_verify_mlir.py \
  spectral_feature_compression/core/loss/composite_separation.py \
  spectral_feature_compression/core/tasks/distillation_task.py \
  tests/test_npu_export_audit.py \
  tests/test_proposed_separation_models.py

./.venv/bin/python -m pytest \
  tests/test_npu_export_audit.py \
  tests/test_proposed_separation_models.py
```

Results:

- `py_compile`: pass.
- `ruff check`: pass, with only the existing pyproject deprecation warning.
- `pytest`: 7 passed.

## Training Recipe Gap Fill

Additional changes for audit item 10:

- Added `spectral_feature_compression/common/datasets/waveform_augmentations.py`.
- Added training-only datamodule flags for waveform augmentation, source remix
  controls, source gain ranges, and activity-aware crop retry.
- Added staged DnR configs:
  - `recipes/dnr/models/sfc-locoformer-lite-plus.teacher.stage1-augmented/config.yaml`
  - `recipes/dnr/models/band-sfc-net-npu.rt-plus.stage1-supervised-aug.rt192k.fp512/config.yaml`
  - `recipes/dnr/models/band-sfc-net-npu.rt-plus.stage2-distill-chunk.rt192k.fp512/config.yaml`
  - `recipes/dnr/models/band-sfc-net-npu.rt-plus.stage3-distill-strict.rt192k.fp512/config.yaml`
- Added `docs/AUDIO_SEPARATION_TRAINING_RECIPE.md`.

Verification commands added for this section:

```bash
./.venv/bin/python -m py_compile \
  spectral_feature_compression/common/datasets/waveform_augmentations.py \
  spectral_feature_compression/common/datasets/hdf5_wav_dataset.py \
  spectral_feature_compression/common/datasets/hdf5_wav_dataset_dm.py \
  spectral_feature_compression/common/datamodules/hdf5_wav_datamodule.py

./.venv/bin/python -m ruff check \
  spectral_feature_compression/common/datasets/waveform_augmentations.py \
  spectral_feature_compression/common/datasets/hdf5_wav_dataset.py \
  spectral_feature_compression/common/datasets/hdf5_wav_dataset_dm.py \
  spectral_feature_compression/common/datamodules/hdf5_wav_datamodule.py \
  tests/test_waveform_augmentations.py

./.venv/bin/python -m pytest \
  tests/test_waveform_augmentations.py \
  tests/test_npu_export_audit.py \
  tests/test_proposed_separation_models.py
```

Results:

- `py_compile`: pass.
- `ruff check`: pass, with only the existing pyproject deprecation warning.
- `pytest`: 9 passed.

## DolphinSFCNPU Distillation Gap Fill

Changes for audit item 12:

- Added `recipes/dnr/models/dolphin-sfc-npu.slim-6m.distill.rt192k.fp512keep475/config.yaml`.
- Added `recipes/dnr/models/dolphin-sfc-npu.slim-8m.distill.rt192k.fp512keep475/config.yaml`.
- Added `docs/DOLPHIN_SFC_NPU_DISTILLATION.md`.
- Added DolphinSFCNPU distilled TODO rows to
  `docs/templates/audio_separation_results_manifest.csv`.
- Updated the audit doc to mark DolphinSFCNPU distillation recipe plumbing as
  implemented and local training/evaluation as pending.
- Tightened `DolphinSFCNPU` `slim_8m` from 8.13M params to 6.55M params so the
  quality probe stays inside the existing preset budget test and closer to the
  project's `<7M` edge target.
- Added `large_6m` and `large_8m` aliases to the current `slim_6m` and
  `slim_8m` presets because existing local Dolphin configs still use the older
  names.

The Dolphin distillation recipes use the same `TeacherStudentDistillationTask`
and composite loss stack as the BandSFC RT+ distillation path.  They explicitly
select current preset names `slim_6m` and `slim_8m`.

Verification commands:

```bash
./.venv/bin/python -c "import yaml; from pathlib import Path; paths=[Path('recipes/dnr/models/dolphin-sfc-npu.slim-6m.distill.rt192k.fp512keep475/config.yaml'), Path('recipes/dnr/models/dolphin-sfc-npu.slim-8m.distill.rt192k.fp512keep475/config.yaml')]; [yaml.safe_load(p.read_text()) for p in paths]; print('yaml-ok')"

PYTHONPYCACHEPREFIX=/tmp/opencode/pycache \
  ./.venv/bin/python -m py_compile DolphinSFCNPU/dolphin_sfc.py

./.venv/bin/python -m ruff check DolphinSFCNPU/dolphin_sfc.py

./.venv/bin/python -m pytest \
  DolphinSFCNPU/test_dolphin_sfc_npu.py::test_slim_presets_fit_param_and_state_budgets
```

Results:

- YAML parse: pass.
- `py_compile`: pass with `PYTHONPYCACHEPREFIX=/tmp/opencode/pycache` because
  the local `DolphinSFCNPU/__pycache__` path is not writable.
- `ruff check`: pass, with only the existing pyproject deprecation warning.
- Dolphin slim preset budget test: pass.
