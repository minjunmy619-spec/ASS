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

## Full Loss Stack Gap Fill

Additional changes for audit item 2:

- `TeacherStudentDistillationTask` now accepts mapping/tuple model outputs and
  can parse auxiliary tensors without changing default tensor-only models.
- Added source-activity-aware waveform loss via `source_activity_loss_weight`.
- Added teacher spectral mask distillation via `teacher_mask_loss_weight`.
- Added teacher spectral logit distillation via `teacher_logit_loss_weight`.
- Added generic latent/intermediate feature distillation via auxiliary outputs or
  forward hooks configured with `student_latent_modules` and
  `teacher_latent_modules`.
- Enabled mask/logit/source-activity terms in the BandSFC RT+, EdgeFusion-SFC,
  and DolphinSFCNPU distillation recipes.
- Left latent distillation weight at `0.0` in default recipes until a
  shape-compatible teacher/student hook pair is selected.

Verification command:

```bash
PYTHONPYCACHEPREFIX=/tmp/opencode/pycache \
  ./.venv/bin/python -m py_compile \
  spectral_feature_compression/core/tasks/distillation_task.py

./.venv/bin/python -m ruff check \
  spectral_feature_compression/core/tasks/distillation_task.py \
  tests/test_proposed_separation_models.py

./.venv/bin/python -m pytest tests/test_proposed_separation_models.py

./.venv/bin/python -c "import yaml; from pathlib import Path; paths=[Path('recipes/dnr/models/band-sfc-net-npu.rt-plus.distill.rt192k.fp512/config.yaml'), Path('recipes/dnr/models/edge-fusion-sfc-distilled.rt192k/config.yaml'), Path('recipes/dnr/models/dolphin-sfc-npu.slim-6m.distill.rt192k.fp512keep475/config.yaml'), Path('recipes/dnr/models/dolphin-sfc-npu.slim-8m.distill.rt192k.fp512keep475/config.yaml')]; [yaml.safe_load(p.read_text()) for p in paths]; print('yaml-ok')"
```

Results:

- `py_compile`: pass.
- `ruff check`: pass, with only the existing pyproject deprecation warning.
- `pytest`: 6 passed.
- Distillation YAML parse: pass.

## Sparse U-Net Mel-SFC Gap Fill

Changes for audit item 3:

- Added `spectral_feature_compression/core/model/sparse_unet_mel_sfc_2d.py`.
- Added `RegionalMelBandSpec2d` for overlapped low/mid/high mel routing.
- Added `SparseBandUNetEncoder` and `SparseBandUNetDecoder` with branch-local
  U-Net skip paths.
- Added `SparseUNetMelSFC2D`, `SparseUNetMelSFCModel`, and
  `build_sparse_unet_mel_sfc_system`.
- Exposed the proposal through
  `build_sparse_unet_mel_sfc_music_system` in
  `spectral_feature_compression/core/model/proposed_separation_models.py`.
- Added MUSDB-first and DnR sibling recipes:
  - `recipes/musdb18hq/models/sparse-unet-mel-sfc.music.rt192k.fp512keep475/config.yaml`
  - `recipes/dnr/models/sparse-unet-mel-sfc.rt192k.fp512keep475/config.yaml`
- Added a smoke test for builder waveform output, packed-core streaming shape,
  and a tiny fp16 state budget.
- Added TODO rows to `docs/templates/audio_separation_results_manifest.csv`.

Default packed-core budget observation:

```bash
./.venv/bin/python -c "from spectral_feature_compression.core.model.sparse_unet_mel_sfc_2d import SparseUNetMelSFC2D; import torch; m=SparseUNetMelSFC2D(n_freq=512, sample_rate=44100, n_src=4, n_chan=2, d_model=64, branch_bands=(24,32,24)); print(sum(p.numel() for p in m.parameters())); print(m.state_size_bytes(dtype=torch.float16)/1024)"
```

Result:

- Parameters: `535346`.
- fp16 layer-cache state: `140.0 KiB`.

Verification commands:

```bash
PYTHONPYCACHEPREFIX=/tmp/opencode/pycache \
  ./.venv/bin/python -m py_compile \
  spectral_feature_compression/core/model/sparse_unet_mel_sfc_2d.py \
  spectral_feature_compression/core/model/proposed_separation_models.py \
  spectral_feature_compression/__init__.py \
  tests/test_proposed_separation_models.py

./.venv/bin/python -m ruff check \
  spectral_feature_compression/core/model/sparse_unet_mel_sfc_2d.py \
  spectral_feature_compression/core/model/proposed_separation_models.py \
  spectral_feature_compression/__init__.py \
  tests/test_proposed_separation_models.py

./.venv/bin/python -c "import yaml; from pathlib import Path; paths=[Path('recipes/musdb18hq/models/sparse-unet-mel-sfc.music.rt192k.fp512keep475/config.yaml'), Path('recipes/dnr/models/sparse-unet-mel-sfc.rt192k.fp512keep475/config.yaml')]; [yaml.safe_load(p.read_text()) for p in paths]; print('yaml-ok')"

./.venv/bin/python -m pytest tests/test_proposed_separation_models.py
```

Results:

- `py_compile`: pass.
- `ruff check`: pass, with only the existing pyproject deprecation warning.
- YAML parse: pass.
- `pytest tests/test_proposed_separation_models.py`: 7 passed.

## SFC-SepReformer Early Source Split Gap Fill

Changes for audit item 5:

- Added `spectral_feature_compression/core/model/source_split_sfc_2d.py`.
- Added `SourceTokenSplitter2d` after SFC query compression.
- Added `SharedSourceRefiner2d`, which applies shared causal 2D refiner blocks to
  each fixed source stream.
- Added `CrossSourceReconstructionMixer2d` for other-source and mixture-token
  context before reconstruction.
- Added `SourceSharedReconstructionDecoder2d`, which reuses one SFC query
  expander and one mask head across all sources.
- Exposed the proposal through `build_sfc_sepreformer_multistem_system` in
  `spectral_feature_compression/core/model/proposed_separation_models.py`.
- Added recipes:
  - `recipes/dnr/models/sfc-sepreformer-multistem.rt192k.fp512keep475/config.yaml`
  - `recipes/musdb18hq/models/sfc-sepreformer-multistem.rt192k.fp512keep475/config.yaml`
- Added a smoke test for builder waveform output, source-split token shape,
  packed-core streaming shape, and tiny fp16 state budget.
- Added TODO rows to `docs/templates/audio_separation_results_manifest.csv`.

Default packed-core budget observation:

```bash
./.venv/bin/python -c "from spectral_feature_compression.core.model.source_split_sfc_2d import OnlineSourceSplitSFC2D; import torch; m=OnlineSourceSplitSFC2D(n_freq=512, n_bands=64, sample_rate=44100, n_src=3, n_chan=1, d_model=32, n_shared_layers=1, n_source_layers=2); print(sum(p.numel() for p in m.parameters())); print(m.state_size_bytes(dtype=torch.float16)/1024); m4=OnlineSourceSplitSFC2D(n_freq=512, n_bands=64, sample_rate=44100, n_src=4, n_chan=2, d_model=32, n_shared_layers=1, n_source_layers=2); print(sum(p.numel() for p in m4.parameters())); print(m4.state_size_bytes(dtype=torch.float16)/1024)"
```

Result:

- DnR three-stem default parameters: `40522`.
- DnR three-stem fp16 layer-cache state: `112.0 KiB`.
- MUSDB four-stem default parameters: `41708`.
- MUSDB four-stem fp16 layer-cache state: `144.0 KiB`.

Verification commands:

```bash
PYTHONPYCACHEPREFIX=/tmp/opencode/pycache \
  ./.venv/bin/python -m py_compile \
  spectral_feature_compression/core/model/source_split_sfc_2d.py \
  spectral_feature_compression/core/model/proposed_separation_models.py \
  spectral_feature_compression/__init__.py \
  tests/test_proposed_separation_models.py

./.venv/bin/python -m ruff check \
  spectral_feature_compression/core/model/source_split_sfc_2d.py \
  spectral_feature_compression/core/model/proposed_separation_models.py \
  spectral_feature_compression/__init__.py \
  tests/test_proposed_separation_models.py

./.venv/bin/python -c "import yaml; from pathlib import Path; paths=[Path('recipes/dnr/models/sfc-sepreformer-multistem.rt192k.fp512keep475/config.yaml'), Path('recipes/musdb18hq/models/sfc-sepreformer-multistem.rt192k.fp512keep475/config.yaml')]; [yaml.safe_load(p.read_text()) for p in paths]; print('yaml-ok')"

./.venv/bin/python -m pytest tests/test_proposed_separation_models.py
```

Results:

- `py_compile`: pass.
- `ruff check`: pass, with only the existing pyproject deprecation warning.
- YAML parse: pass.
- `pytest tests/test_proposed_separation_models.py`: 8 passed.

## SFC Residual-Refinement Gap Fill

Changes for audit item 6:

- Added `spectral_feature_compression/core/model/residual_refinement_sfc_2d.py`.
- Added `Mamba2LiteTemporalBranch2d`, a causal dilated latent-band branch after
  SFC query compression.  This covers the targeted Mamba2 ablation role without
  importing unsupported Mamba2 kernels into the strict path.
- Added `ResidualCorrectionHead2d`, which consumes mixture, first estimate, and
  refined SFC token context to predict a packed complex residual correction.
- Added `OnlineResidualRefinementSFC2D`, `OnlineResidualRefinementSFCModel`, and
  `build_residual_refinement_sfc_system`.
- Exposed the proposal through `build_sfc_residual_refinement_system` in
  `spectral_feature_compression/core/model/proposed_separation_models.py`.
- Added recipes:
  - `recipes/dnr/models/sfc-residual-refinement.rt192k.fp512keep475/config.yaml`
  - `recipes/musdb18hq/models/sfc-residual-refinement.rt192k.fp512keep475/config.yaml`
- Added a smoke test for builder waveform output, packed-core streaming shape,
  and trainable correction/long-branch scales.
- Added TODO rows to `docs/templates/audio_separation_results_manifest.csv`.

Default packed-core budget observation:

```bash
./.venv/bin/python -c "from spectral_feature_compression.core.model.residual_refinement_sfc_2d import OnlineResidualRefinementSFC2D; import torch; m=OnlineResidualRefinementSFC2D(n_freq=512, n_bands=64, sample_rate=44100, n_src=3, n_chan=1); print(sum(p.numel() for p in m.parameters())); print(m.state_size_bytes(dtype=torch.float16)/1024); m4=OnlineResidualRefinementSFC2D(n_freq=512, n_bands=64, sample_rate=44100, n_src=4, n_chan=2); print(sum(p.numel() for p in m4.parameters())); print(m4.state_size_bytes(dtype=torch.float16)/1024)"
```

Result:

- DnR three-stem default parameters: `26364`.
- DnR three-stem fp16 layer-cache state: `144.0 KiB`.
- MUSDB four-stem default parameters: `27200`.
- MUSDB four-stem fp16 layer-cache state: `144.0 KiB`.

Verification commands:

```bash
PYTHONPYCACHEPREFIX=/tmp/opencode/pycache \
  ./.venv/bin/python -m py_compile \
  spectral_feature_compression/core/model/residual_refinement_sfc_2d.py \
  spectral_feature_compression/core/model/proposed_separation_models.py \
  spectral_feature_compression/__init__.py \
  tests/test_proposed_separation_models.py

./.venv/bin/python -m ruff check \
  spectral_feature_compression/core/model/residual_refinement_sfc_2d.py \
  spectral_feature_compression/core/model/proposed_separation_models.py \
  spectral_feature_compression/__init__.py \
  tests/test_proposed_separation_models.py

./.venv/bin/python -c "import yaml; from pathlib import Path; paths=[Path('recipes/dnr/models/sfc-residual-refinement.rt192k.fp512keep475/config.yaml'), Path('recipes/musdb18hq/models/sfc-residual-refinement.rt192k.fp512keep475/config.yaml')]; [yaml.safe_load(p.read_text()) for p in paths]; print('yaml-ok')"

./.venv/bin/python -m pytest tests/test_proposed_separation_models.py
```

Results:

- `py_compile`: pass.
- `ruff check`: pass, with only the existing pyproject deprecation warning.
- YAML parse: pass.
- `pytest tests/test_proposed_separation_models.py`: 9 passed.

## Adaptive Mel / Band-Mapping Gap Fill

Changes for audit item 7:

- Added `spectral_feature_compression/core/model/adaptive_mel_sfc_2d.py`.
- Added `AdaptiveMelBandSpec2d` with explicit overlapped mel bands and controls:
  - `low_freq_hz`
  - `low_freq_band_fraction`
  - `overlap_factor`
  - `low_freq_overlap_factor`
- Added `OnlineAdaptiveMelSFC2D`, `OnlineAdaptiveMelSFCModel`, and
  `build_adaptive_mel_sfc_system`.
- Exposed the proposal through `build_adaptive_mel_sfc_ablation_system` in
  `spectral_feature_compression/core/model/proposed_separation_models.py`.
- Added `fixed` / `linear` / `uniform` modes to `SoftBandSpec2d` for clean
  fixed-band ablations.
- Added ablation recipes:
  - `recipes/dnr/models/bandmap-ablation.fixed80.rt192k.fp512keep475/config.yaml`
  - `recipes/musdb18hq/models/bandmap-ablation.fixed80.rt192k.fp512keep475/config.yaml`
  - `recipes/dnr/models/bandmap-ablation.mel-overlap80.rt192k.fp512keep475/config.yaml`
  - `recipes/musdb18hq/models/bandmap-ablation.mel-overlap80.rt192k.fp512keep475/config.yaml`
  - `recipes/dnr/models/bandmap-ablation.sfc-ca80.teacher/config.yaml`
  - `recipes/musdb18hq/models/bandmap-ablation.sfc-ca80.teacher/config.yaml`
  - `recipes/musdb18hq/models/bandmap-ablation.sfc-mamba64.teacher/config.yaml`
- Added a smoke test for adaptive mel basis shape, low-frequency overlap, builder
  waveform output, packed-core streaming shape, and fp16 state budget.
- Added TODO rows to `docs/templates/audio_separation_results_manifest.csv`.

Default packed-core budget observation:

```bash
./.venv/bin/python -c "from spectral_feature_compression.core.model.adaptive_mel_sfc_2d import OnlineAdaptiveMelSFC2D; import torch; m=OnlineAdaptiveMelSFC2D(n_freq=512, n_bands=80, sample_rate=44100, n_src=3, n_chan=1, d_model=24, n_layers=6); print(sum(p.numel() for p in m.parameters())); print(m.state_size_bytes(dtype=torch.float16)/1024); m4=OnlineAdaptiveMelSFC2D(n_freq=512, n_bands=80, sample_rate=44100, n_src=4, n_chan=2, d_model=24, n_layers=6); print(sum(p.numel() for p in m4.parameters())); print(m4.state_size_bytes(dtype=torch.float16)/1024)"
```

Result:

- DnR three-stem default parameters: `31244`.
- DnR three-stem fp16 layer-cache state: `90.0 KiB`.
- MUSDB four-stem default parameters: `31542`.
- MUSDB four-stem fp16 layer-cache state: `90.0 KiB`.

Verification commands:

```bash
PYTHONPYCACHEPREFIX=/tmp/opencode/pycache \
  ./.venv/bin/python -m py_compile \
  spectral_feature_compression/core/model/adaptive_mel_sfc_2d.py \
  spectral_feature_compression/core/model/online_soft_band_sfc_2d.py \
  spectral_feature_compression/core/model/proposed_separation_models.py \
  spectral_feature_compression/__init__.py \
  tests/test_proposed_separation_models.py

./.venv/bin/python -m ruff check \
  spectral_feature_compression/core/model/adaptive_mel_sfc_2d.py \
  spectral_feature_compression/core/model/online_soft_band_sfc_2d.py \
  spectral_feature_compression/core/model/proposed_separation_models.py \
  spectral_feature_compression/__init__.py \
  tests/test_proposed_separation_models.py

./.venv/bin/python -c "import yaml; from pathlib import Path; paths=[Path('recipes/dnr/models/bandmap-ablation.fixed80.rt192k.fp512keep475/config.yaml'), Path('recipes/musdb18hq/models/bandmap-ablation.fixed80.rt192k.fp512keep475/config.yaml'), Path('recipes/dnr/models/bandmap-ablation.mel-overlap80.rt192k.fp512keep475/config.yaml'), Path('recipes/musdb18hq/models/bandmap-ablation.mel-overlap80.rt192k.fp512keep475/config.yaml'), Path('recipes/dnr/models/bandmap-ablation.sfc-ca80.teacher/config.yaml'), Path('recipes/musdb18hq/models/bandmap-ablation.sfc-ca80.teacher/config.yaml'), Path('recipes/musdb18hq/models/bandmap-ablation.sfc-mamba64.teacher/config.yaml')]; [yaml.safe_load(p.read_text()) for p in paths]; print('yaml-ok')"

./.venv/bin/python -m pytest tests/test_proposed_separation_models.py
```

Results:

- `py_compile`: pass.
- `ruff check`: pass, with only the existing pyproject deprecation warning.
- YAML parse: pass.
- `pytest tests/test_proposed_separation_models.py`: 10 passed.

## 2026-05-30 Review Follow-Up

Review changes:

- Tightened `tests/test_proposed_separation_models.py` so causal streaming tests now
  assert frame-by-frame output matches full forward output within `1e-5`, instead
  of checking shape only.
- Fixed `docs/templates/audio_separation_results_manifest.csv` column alignment:
  every row now has the 44 header columns, the paper parameter counts are under
  `params`, and the paper MUSDB/DnR metrics are under the correct metric columns.

Verification commands:

```bash
PYTHONPYCACHEPREFIX=/tmp/opencode/pycache \
  ./.venv/bin/python -m py_compile \
  spectral_feature_compression/core/model/sparse_unet_mel_sfc_2d.py \
  spectral_feature_compression/core/model/source_split_sfc_2d.py \
  spectral_feature_compression/core/model/residual_refinement_sfc_2d.py \
  spectral_feature_compression/core/model/adaptive_mel_sfc_2d.py \
  spectral_feature_compression/core/model/online_soft_band_sfc_2d.py \
  spectral_feature_compression/core/model/proposed_separation_models.py \
  spectral_feature_compression/core/tasks/distillation_task.py \
  spectral_feature_compression/__init__.py \
  tests/test_proposed_separation_models.py

./.venv/bin/python -m ruff check \
  spectral_feature_compression/core/model/sparse_unet_mel_sfc_2d.py \
  spectral_feature_compression/core/model/source_split_sfc_2d.py \
  spectral_feature_compression/core/model/residual_refinement_sfc_2d.py \
  spectral_feature_compression/core/model/adaptive_mel_sfc_2d.py \
  spectral_feature_compression/core/model/online_soft_band_sfc_2d.py \
  spectral_feature_compression/core/model/proposed_separation_models.py \
  spectral_feature_compression/core/tasks/distillation_task.py \
  spectral_feature_compression/__init__.py \
  tests/test_proposed_separation_models.py

./.venv/bin/python -c "import csv, yaml; from pathlib import Path; paths=[Path('recipes/dnr/models/sparse-unet-mel-sfc.rt192k.fp512keep475/config.yaml'), Path('recipes/musdb18hq/models/sparse-unet-mel-sfc.music.rt192k.fp512keep475/config.yaml'), Path('recipes/dnr/models/sfc-sepreformer-multistem.rt192k.fp512keep475/config.yaml'), Path('recipes/musdb18hq/models/sfc-sepreformer-multistem.rt192k.fp512keep475/config.yaml'), Path('recipes/dnr/models/sfc-residual-refinement.rt192k.fp512keep475/config.yaml'), Path('recipes/musdb18hq/models/sfc-residual-refinement.rt192k.fp512keep475/config.yaml'), Path('recipes/dnr/models/bandmap-ablation.fixed80.rt192k.fp512keep475/config.yaml'), Path('recipes/musdb18hq/models/bandmap-ablation.fixed80.rt192k.fp512keep475/config.yaml'), Path('recipes/dnr/models/bandmap-ablation.mel-overlap80.rt192k.fp512keep475/config.yaml'), Path('recipes/musdb18hq/models/bandmap-ablation.mel-overlap80.rt192k.fp512keep475/config.yaml'), Path('recipes/dnr/models/bandmap-ablation.sfc-ca80.teacher/config.yaml'), Path('recipes/musdb18hq/models/bandmap-ablation.sfc-ca80.teacher/config.yaml'), Path('recipes/musdb18hq/models/bandmap-ablation.sfc-mamba64.teacher/config.yaml')]; [yaml.safe_load(p.read_text()) for p in paths]; rows=list(csv.reader(Path('docs/templates/audio_separation_results_manifest.csv').open(newline=''))); width=len(rows[0]); bad=[(i+1,len(r)) for i,r in enumerate(rows) if len(r)!=width]; assert not bad, bad; print(f'yaml-ok {len(paths)} files; csv-ok {len(rows)} rows x {width} columns')"

PYTHONPATH=/home/cmj/works/ASS:/home/cmj/works/ASS/aiaccel \
  ./.venv/bin/python -m pytest tests/test_proposed_separation_models.py
```

Results:

- `py_compile`: pass.
- `ruff check`: pass, with only the existing pyproject deprecation warning.
- YAML/CSV validation: `yaml-ok 13 files; csv-ok 21 rows x 44 columns`.
- `pytest tests/test_proposed_separation_models.py`: 10 passed.

## 2026-05-30 ONNX/ONE Validation Follow-Up

Validation changes:

- Fixed `tools/online/verify_npu_variants.py` so generated ONE configs now set
  `replace_non_const_fc_with_batch_matmul=True`.  This matches
  `OPERATION_MANUAL_PYTORCH_TO_ONE_NPU.md` and unblocks quantization failures of
  the form `Unsupported non const input ... MatMul/tr`.
- Removed the unused `onnx.numpy_helper` import from the verifier while touching
  the file.
- Made proposal builders tolerate inherited base recipe keys that are irrelevant
  for the new target builder:
  - `build_sparse_unet_mel_sfc_music_system`: ignores inherited `n_bands`,
    `band_config`, and `n_layers`.
  - `build_sfc_sepreformer_multistem_system`: ignores inherited `n_layers`.
  - `build_adaptive_mel_sfc_ablation_system`: ignores inherited `band_config`.
- Updated `docs/templates/audio_separation_results_manifest.csv` for the DnR
  rows that completed ONNX/ONE validation.

Focused validation commands:

```bash
PYTHONPATH=/home/cmj/works/ASS:/home/cmj/works/ASS/aiaccel \
  ./.venv/bin/python - <<'PY'
from tools.online.verify_npu_variants import ROOT, Variant, build_env, run_one_variant, write_summary

run_root = ROOT / 'logs' / 'npu_verify_general' / 'focused_onnx_one_20260530_fixed_opt'
run_root.mkdir(parents=True, exist_ok=True)
variant = Variant(
    kind='recipe',
    name='band-sfc-net-npu.rt-plus.rt192k.fp512',
    recipe_cfg=ROOT / 'recipes/dnr/models/band-sfc-net-npu.rt-plus.rt192k.fp512/config.yaml',
)
result = run_one_variant(
    variant,
    run_root / variant.name,
    ROOT / '.venv/bin/python',
    build_env(),
    force_onnxsim_large_shape_ops=True,
)
write_summary([result], run_root)
print(result)
PY

PYTHONPATH=/home/cmj/works/ASS:/home/cmj/works/ASS/aiaccel \
  ./.venv/bin/python - <<'PY'
from tools.online.verify_npu_variants import ROOT, Variant, build_env, run_one_variant, write_summary

names = [
    'sparse-unet-mel-sfc.rt192k.fp512keep475',
    'sfc-sepreformer-multistem.rt192k.fp512keep475',
    'sfc-residual-refinement.rt192k.fp512keep475',
    'bandmap-ablation.fixed80.rt192k.fp512keep475',
    'bandmap-ablation.mel-overlap80.rt192k.fp512keep475',
]
run_root = ROOT / 'logs' / 'npu_verify_general' / 'focused_onnx_one_20260530_new_variants_fixed'
run_root.mkdir(parents=True, exist_ok=True)
env = build_env()
results = []
for name in names:
    variant = Variant(kind='recipe', name=name, recipe_cfg=ROOT / 'recipes/dnr/models' / name / 'config.yaml')
    result = run_one_variant(
        variant,
        run_root / variant.name,
        ROOT / '.venv/bin/python',
        env,
        force_onnxsim_large_shape_ops=True,
    )
    results.append(result)
    print(f"{result['status']} {name} {result['fail_stage'] or '-'}")
write_summary(results, run_root)
PY
```

Direct Circle verification and ONNX risk summary were collected with
`/home/cmj/works/ONE/build/compiler/circle-verify/circle-verify` and written to:

- `logs/npu_verify_general/focused_onnx_one_20260530_validation_summary.json`
- `logs/npu_verify_general/focused_onnx_one_20260530_remaining_validation_summary.json`

Clean verifier summaries:

- `logs/npu_verify_general/focused_onnx_one_20260530_fixed_opt/summary.md`
- `logs/npu_verify_general/focused_onnx_one_20260530_new_variants_fixed/summary.md`
- `logs/npu_verify_general/focused_onnx_one_20260530_remaining_candidates_fixed/summary.md`

Results:

| Variant | ONE import | optimize | quantize | circle-verify | ONNX nodes | Remaining risk flags |
|---|---|---|---|---|---:|---|
| `band-sfc-net-npu.rt-plus.rt192k.fp512` | PASS | PASS | PASS | PASS | 444 | `activation_matmul_rank_le3=14`, `transpose=37` |
| `sparse-unet-mel-sfc.rt192k.fp512keep475` | PASS | PASS | PASS | PASS | 555 | `activation_matmul_rank_le3=6`, `transpose=21` |
| `sfc-sepreformer-multistem.rt192k.fp512keep475` | PASS | PASS | PASS | PASS | 375 | `activation_matmul_rank_le3=5`, `transpose=23` |
| `sfc-residual-refinement.rt192k.fp512keep475` | PASS | PASS | PASS | PASS | 222 | `activation_matmul_rank_le3=4`, `transpose=17` |
| `bandmap-ablation.fixed80.rt192k.fp512keep475` | PASS | PASS | PASS | PASS | 208 | `activation_matmul_rank_le3=3`, `transpose=11` |
| `bandmap-ablation.mel-overlap80.rt192k.fp512keep475` | PASS | PASS | PASS | PASS | 208 | `activation_matmul_rank_le3=3`, `transpose=11` |
| `dolphin-sfc-npu.large-6m.fp512keep475` | PASS | PASS | PASS | PASS | 314 | `activation_matmul_rank_le3=2`, `transpose=7` |
| `dolphin-sfc-npu.slim-6m.distill.rt192k.fp512keep475` | PASS | PASS | PASS | PASS | 314 | `activation_matmul_rank_le3=2`, `transpose=7` |
| `edge-fusion-sfc-distilled.rt192k` | PASS | PASS | PASS | PASS | 168 | `transpose=2` |
| `online-hierarchical-soft-band-parallel-ffi-sfc2d.rt192k.speech-lowfreq-narrow.causal20dim.0-1-1l.128-96-48b` | PASS | PASS | PASS | PASS | 252 | `activation_matmul_rank_le3=6`, `transpose=21` |

Notes:

- All validated ONNX graphs had zero `Tile`, `ConstantOfShape`, `Expand`,
  `PRelu`, dynamic slice risk, scalar gather risk, rank>4 value risk, and
  transpose-perm dtype risk after forced ONNX simplification and verifier
  compatibility rewrites.
- `activation_matmul_rank_le3` remains as a conservative audit flag, but the
  full ONE pipeline plus `circle-verify` succeeded for these fixed-shape exports.
- `EdgeFusionNPU2DPackedCoreAdapter` now exposes a `masking` switch so
  `--disable-masking` can export raw masks without tracing complex STFT multiply.
- `tools/online/verify_npu_variants.py` now retries the common EdgeFusion DnR
  channel mismatch (`expected 2 packed channels, got 4`) with `--n-chan 1`.

## Wrapper-Side PCEN Ablation Plumbing

Changes:

- Added `PCENGainNormalizer2d` and `build_pcen_preprocessor` in
  `spectral_feature_compression/core/model/frequency_preprocessing.py`.
- Extended `FrequencyPreprocessedOnlineModel` to apply optional PCEN-style
  complex input gain before the packed core and invert the same gain on the
  source output.
- Kept PCEN outside exported cores: `tools/online/export_onnx_online_model.py`
  still extracts `.model.core`, so the exported input contract remains packed
  complex STFT `[B, 2*M, T, F]`.
- Added PCEN builder flags to BandSFCNetNPU, EdgeFusionNPU, DolphinSFCNPU, and
  the proposal wrappers that delegate to BandSFCNetNPU/EdgeFusionNPU.
- Added an opt-in RT+ supervised warm-start ablation recipe:
  `recipes/dnr/models/band-sfc-net-npu.rt-plus.stage1-supervised-aug-pcen.rt192k.fp512/config.yaml`.
- Added PCEN manifest reporting to `tools/online/run_streaming_inference.py`.

Verification commands:

```bash
PYTHONPYCACHEPREFIX=/tmp/opencode/pycache \
  ./.venv/bin/python -m py_compile \
  spectral_feature_compression/core/model/frequency_preprocessing.py \
  BandSFCNetNPU/training_wrapper.py \
  EdgeFusionNPU/training_wrapper.py \
  DolphinSFCNPU/training_wrapper.py \
  spectral_feature_compression/core/model/proposed_separation_models.py \
  tools/online/run_streaming_inference.py \
  tests/test_online_frequency_preprocessing.py

./.venv/bin/python -m ruff check \
  spectral_feature_compression/core/model/frequency_preprocessing.py \
  BandSFCNetNPU/training_wrapper.py \
  EdgeFusionNPU/training_wrapper.py \
  DolphinSFCNPU/training_wrapper.py \
  spectral_feature_compression/core/model/proposed_separation_models.py \
  tools/online/run_streaming_inference.py \
  tests/test_online_frequency_preprocessing.py

./.venv/bin/python -m pytest \
  tests/test_online_frequency_preprocessing.py \
  tests/test_proposed_separation_models.py

./.venv/bin/python -c "from pathlib import Path; from tools.online.export_onnx_online_model import build_model_system_from_recipe_config; m=build_model_system_from_recipe_config(Path('recipes/dnr/models/band-sfc-net-npu.rt-plus.stage1-supervised-aug-pcen.rt192k.fp512/config.yaml')); print(type(m.model).__name__, m.model.pcen_preprocess_manifest()['type'])"

./.venv/bin/python -c "from pathlib import Path; from tools.online.export_onnx_online_model import build_model_system_from_recipe_config, get_export_core_from_model_system; m=build_model_system_from_recipe_config(Path('recipes/dnr/models/band-sfc-net-npu.rt-plus.stage1-supervised-aug-pcen.rt192k.fp512/config.yaml')); core=get_export_core_from_model_system(m); print(type(core).__name__, core.n_freq, hasattr(core, 'pcen_preprocessor'))"

git diff --check
```

Results:

- `py_compile`: pass.
- `ruff check`: pass, with only the existing pyproject deprecation warning.
- `pytest`: 14 passed, with existing CUDA/autocast warnings and local
  `.pytest_cache` permission warning.
- PCEN recipe instantiation: `FrequencyPreprocessedOnlineModel pcen_gain_normalizer_2d`.
- Export core check: `BandSFCNetNPU 512 False`.
- `git diff --check`: pass.

## DC-Bin Bypass Frequency-Shape Ablation

Changes:

- Added opt-in DC bypass to `FrequencyPreprocessedOnlineModel`.
- When enabled, the wrapper splits packed STFT bin `0` before frequency
  preprocessing, sends only bins `1..F-1` into the core path, and restores a DC
  bin before unpacking/iSTFT.
- Added `dc_policy` with current options:
  - `zero`: restore zero DC for every output stem.
  - `mixture_equal`: split the saved mixture DC equally across stems.
- Extended frequency preprocessing helpers so `n_fft=2048` can use body bins
  `1024 -> 512` while wrapper input/output remains full `1025` bins.
- Added DC-bypass flags to BandSFCNetNPU, EdgeFusionNPU, DolphinSFCNPU, and the
  current proposal builders.
- Added opt-in DnR recipe:
  `recipes/dnr/models/band-sfc-net-npu.rt-plus.stage1-supervised-aug-dcbypass.rt192k.fp512/config.yaml`.
- Added DC-bypass metadata to streaming/export manifests.

Validation commands:

```bash
PYTHONPYCACHEPREFIX=/tmp/opencode/pycache \
  ./.venv/bin/python -m py_compile \
  spectral_feature_compression/core/model/frequency_preprocessing.py \
  BandSFCNetNPU/training_wrapper.py \
  EdgeFusionNPU/training_wrapper.py \
  DolphinSFCNPU/training_wrapper.py \
  spectral_feature_compression/core/model/proposed_separation_models.py \
  spectral_feature_compression/core/model/adaptive_mel_sfc_2d.py \
  spectral_feature_compression/core/model/residual_refinement_sfc_2d.py \
  spectral_feature_compression/core/model/source_split_sfc_2d.py \
  spectral_feature_compression/core/model/sparse_unet_mel_sfc_2d.py \
  tools/online/export_onnx_online_model.py \
  tools/online/run_streaming_inference.py \
  tests/test_online_frequency_preprocessing.py

./.venv/bin/python -m ruff check \
  spectral_feature_compression/core/model/frequency_preprocessing.py \
  BandSFCNetNPU/training_wrapper.py \
  EdgeFusionNPU/training_wrapper.py \
  DolphinSFCNPU/training_wrapper.py \
  spectral_feature_compression/core/model/proposed_separation_models.py \
  spectral_feature_compression/core/model/adaptive_mel_sfc_2d.py \
  spectral_feature_compression/core/model/residual_refinement_sfc_2d.py \
  spectral_feature_compression/core/model/source_split_sfc_2d.py \
  spectral_feature_compression/core/model/sparse_unet_mel_sfc_2d.py \
  tools/online/export_onnx_online_model.py \
  tools/online/run_streaming_inference.py \
  tests/test_online_frequency_preprocessing.py

./.venv/bin/python -m pytest \
  tests/test_online_frequency_preprocessing.py \
  tests/test_proposed_separation_models.py

./.venv/bin/python -c "from pathlib import Path; from tools.online.export_onnx_online_model import build_model_system_from_recipe_config; m=build_model_system_from_recipe_config(Path('recipes/dnr/models/band-sfc-net-npu.rt-plus.stage1-supervised-aug-dcbypass.rt192k.fp512/config.yaml')); print(type(m.model).__name__, m.model.input_n_freq, m.model.body_input_n_freq, m.model.core.n_freq, m.model.dc_bypass_manifest()['policy'])"

./.venv/bin/python -c "from pathlib import Path; from tools.online.export_onnx_online_model import build_model_system_from_recipe_config, get_export_core_from_model_system, load_dc_bypass_metadata; p=Path('recipes/dnr/models/band-sfc-net-npu.rt-plus.stage1-supervised-aug-dcbypass.rt192k.fp512/config.yaml'); m=build_model_system_from_recipe_config(p); core=get_export_core_from_model_system(m); print(type(core).__name__, core.n_freq, hasattr(core, 'dc_bypass_enabled'), load_dc_bypass_metadata(p))"

./.venv/bin/python tools/online/export_onnx_online_model.py \
  recipes/dnr/models/band-sfc-net-npu.rt-plus.stage1-supervised-aug-dcbypass.rt192k.fp512/config.yaml \
  --out /tmp/opencode/band_sfc_rt_plus_dcbypass_core.onnx \
  --n-chan 1 \
  --frames 1 \
  --opset 14 \
  --disable-masking \
  --check \
  --deploy-manifest-out /tmp/opencode/band_sfc_rt_plus_dcbypass_manifest.json

git diff --check
```

Results:

- `py_compile`: pass.
- `ruff check`: pass, with only the existing pyproject deprecation warning.
- `pytest`: 16 passed, with existing CUDA/autocast warnings and local
  `.pytest_cache` permission warning.
- DC-bypass recipe instantiation:
  `FrequencyPreprocessedOnlineModel 1025 1024 512 zero`.
- Export core check:
  `BandSFCNetNPU 512 False {'enabled': True, 'policy': 'zero', 'full_n_freq': 1025, 'body_n_freq': 1024}`.
- ONNX config-only export/check: passed; exported core input shape is
  `(1, 2, 1, 512)` and the deploy manifest records frequency preprocessing
  `input_n_freq: 1024` plus `dc_bypass` metadata.
- The raw unsimplified ONNX audit still reports `And`, `GreaterOrEqual`, and
  `Less` outside the recommended allowlist in this quick export, matching the
  need to keep using the established forced simplification/ONE flow for final
  compilation checks.
- `git diff --check`: pass.

## Unified Wrapper Coverage Pass

Scope:

- Audited online/NPU model builder entry points that participate in the current
  training/inference/export flows.
- Kept augmentation and distillation in their existing shared locations:
  datamodule/task configuration, not model wrappers.
- Did not refactor offline teacher-only `ModelWrapper` paths; those remain
  compatible with shared datamodule/distillation tasks but are not strict NPU
  packed-core deployment paths.

Coverage changes:

- Added PCEN and DC-bypass config support to BandSCNetNPU:
  `BandSCNetNPU/training_wrapper.py` and `BandSCNetNPU/freq_preprocessed.py`.
- Added PCEN and DC-bypass config support to the online SFC family builders:
  `online_wrapper.py`, `online_soft_band_sfc_2d.py`,
  `online_soft_band_query_sfc_2d.py`, `online_soft_band_dilated_sfc_2d.py`,
  `online_soft_band_query_dilated_sfc_2d.py`,
  `online_crossattn_query_sfc_2d.py`, `online_hard_band_sfc_2d.py`,
  `online_soft_band_gru_sfc_2d.py`, and the three hierarchical SFC builders.
- Added PCEN config support to the new proposal-core builders:
  Sparse U-Net Mel-SFC, source-split SFC, residual-refinement SFC, and adaptive
  mel SFC.  DC-bypass support had already been added in the prior pass.
- Added PCEN/DC passthrough to all proposal delegator builders in
  `proposed_separation_models.py`, including the hierarchical middle-tier
  proposal.
- Aligned `TIGER.training_wrapper.TIGERWaveformSeparator` with the same
  wrapper-side semantics:
  - frequency preprocessing before TIGER core RI sequence processing,
  - optional DC split/restore around the core body bins,
  - optional PCEN gain normalization before core masking,
  - PCEN gain inversion before frequency synthesis/iSTFT.
- Added export manifest metadata for wrapper-side PCEN via
  `load_pcen_preprocess_metadata` in `tools/online/export_onnx_online_model.py`.

Validation commands:

```bash
PYTHONPYCACHEPREFIX=/tmp/opencode/pycache \
  ./.venv/bin/python -m py_compile \
  TIGER/training_wrapper.py \
  BandSCNetNPU/training_wrapper.py \
  BandSCNetNPU/freq_preprocessed.py \
  spectral_feature_compression/core/model/online_wrapper.py \
  spectral_feature_compression/core/model/online_soft_band_sfc_2d.py \
  spectral_feature_compression/core/model/online_soft_band_query_sfc_2d.py \
  spectral_feature_compression/core/model/online_soft_band_dilated_sfc_2d.py \
  spectral_feature_compression/core/model/online_soft_band_query_dilated_sfc_2d.py \
  spectral_feature_compression/core/model/online_crossattn_query_sfc_2d.py \
  spectral_feature_compression/core/model/online_hierarchical_soft_band_parallel_ffi_sfc_2d.py \
  spectral_feature_compression/core/model/online_hierarchical_soft_band_ffi_sfc_2d.py \
  spectral_feature_compression/core/model/online_hierarchical_soft_band_sfc_2d.py \
  spectral_feature_compression/core/model/online_hard_band_sfc_2d.py \
  spectral_feature_compression/core/model/online_soft_band_gru_sfc_2d.py \
  spectral_feature_compression/core/model/adaptive_mel_sfc_2d.py \
  spectral_feature_compression/core/model/residual_refinement_sfc_2d.py \
  spectral_feature_compression/core/model/source_split_sfc_2d.py \
  spectral_feature_compression/core/model/sparse_unet_mel_sfc_2d.py \
  spectral_feature_compression/core/model/proposed_separation_models.py \
  tools/online/export_onnx_online_model.py \
  tests/test_online_frequency_preprocessing.py

./.venv/bin/python -m ruff check \
  TIGER/training_wrapper.py \
  BandSCNetNPU/training_wrapper.py \
  BandSCNetNPU/freq_preprocessed.py \
  spectral_feature_compression/core/model/online_wrapper.py \
  spectral_feature_compression/core/model/online_soft_band_sfc_2d.py \
  spectral_feature_compression/core/model/online_soft_band_query_sfc_2d.py \
  spectral_feature_compression/core/model/online_soft_band_dilated_sfc_2d.py \
  spectral_feature_compression/core/model/online_soft_band_query_dilated_sfc_2d.py \
  spectral_feature_compression/core/model/online_crossattn_query_sfc_2d.py \
  spectral_feature_compression/core/model/online_hierarchical_soft_band_parallel_ffi_sfc_2d.py \
  spectral_feature_compression/core/model/online_hierarchical_soft_band_ffi_sfc_2d.py \
  spectral_feature_compression/core/model/online_hierarchical_soft_band_sfc_2d.py \
  spectral_feature_compression/core/model/online_hard_band_sfc_2d.py \
  spectral_feature_compression/core/model/online_soft_band_gru_sfc_2d.py \
  spectral_feature_compression/core/model/adaptive_mel_sfc_2d.py \
  spectral_feature_compression/core/model/residual_refinement_sfc_2d.py \
  spectral_feature_compression/core/model/source_split_sfc_2d.py \
  spectral_feature_compression/core/model/sparse_unet_mel_sfc_2d.py \
  spectral_feature_compression/core/model/proposed_separation_models.py \
  tools/online/export_onnx_online_model.py \
  tests/test_online_frequency_preprocessing.py

./.venv/bin/python -m pytest \
  tests/test_online_frequency_preprocessing.py \
  tests/test_proposed_separation_models.py

./.venv/bin/python -c "from TIGER.training_wrapper import build_tiger_system; m=build_tiger_system(n_fft=2048, hop_length=512, fs=44100, n_src=3, n_chan=1, variant='npu-edge-v2', freq_preprocess_enabled=True, freq_preprocess_keep_bins=475, freq_preprocess_target_bins=512, dc_bypass_enabled=True, pcen_preprocess_enabled=True, css_segment_size=1, css_shift_size=1); print(type(m).__name__, m.body_input_n_freq, m.core_n_freq, m.pcen_preprocess_manifest()['type'], m.dc_bypass_manifest()['body_input_n_freq'])"

./.venv/bin/python -c "from pathlib import Path; from tools.online.export_onnx_online_model import load_frequency_preprocess_metadata, load_dc_bypass_metadata, load_pcen_preprocess_metadata; p=Path('recipes/dnr/models/band-sfc-net-npu.rt-plus.stage1-supervised-aug-pcen.rt192k.fp512/config.yaml'); print(load_pcen_preprocess_metadata(p)); q=Path('recipes/dnr/models/band-sfc-net-npu.rt-plus.stage1-supervised-aug-dcbypass.rt192k.fp512/config.yaml'); print(load_frequency_preprocess_metadata(q)); print(load_dc_bypass_metadata(q))"

git diff --check
```

Results:

- `py_compile`: pass.
- `ruff check`: pass, with only the existing pyproject deprecation warning.
- `pytest`: 17 passed, with existing CUDA/autocast warnings and local
  `.pytest_cache` permission warning.
- TIGER wrapper smoke check:
  `TIGERWaveformSeparator 1024 512 pcen_gain_normalizer_2d 1024`.
- Export metadata checks report PCEN metadata for the PCEN recipe and
  `input_n_freq: 1024` plus DC metadata for the DC-bypass recipe.
- `git diff --check`: pass.

## PCEN/DC Wrapper Review Follow-Up

Scope:

- Reviewed the wrapper-side frequency preprocessing, PCEN, DC-bypass, builder,
  and export metadata paths after the unified coverage pass.
- Kept PCEN/DC outside exported cores; the ONNX core contract remains packed
  complex STFT `[B, 2*M, T, F]`.

Fixes:

- `FrequencyPreprocessedOnlineModel.forward_stream_recompute` now carries PCEN
  IIR state and normalized core history for PCEN-enabled recompute streaming,
  instead of recomputing gain from only the local context window.
- `PCENGainNormalizer2d` validates stream state shape early.
- `FrequencyPreprocessedOnlineModel` validates core frequency/source/channel
  contracts at construction time.
- Online SFC and proposal builders now pass `core_n_fft = 2 * (core_n_freq - 1)`
  to reduced-bin cores, so band specs align with the compressed core frequency
  axis instead of the original full STFT `n_fft`.
- `verify_npu_variants.py` now follows concrete `_base_` recipe inheritance for
  `n_chan` inference, avoiding the previous default `--n-chan 2` retry path for
  inherited DnR recipes.
- `export_onnx_online_model.py` now reports PCEN external state shape metadata,
  returns `target_bins` from `infer_export_freq_bins`, records actual core
  `n_chan`, and fails early on `--n-chan` mismatch.
- `run_streaming_inference.py` now tries the same Hydra/aiaccel config loader as
  the export script before falling back to the lightweight parser, and prints
  instantiated model manifests for frequency preprocessing, PCEN, and DC-bypass.

Validation commands:

```bash
PYTHONPYCACHEPREFIX=/tmp/opencode/pycache \
  ./.venv/bin/python -m py_compile \
  spectral_feature_compression/core/model/frequency_preprocessing.py \
  spectral_feature_compression/core/model/online_wrapper.py \
  spectral_feature_compression/core/model/online_soft_band_sfc_2d.py \
  spectral_feature_compression/core/model/online_soft_band_query_sfc_2d.py \
  spectral_feature_compression/core/model/online_hard_band_sfc_2d.py \
  spectral_feature_compression/core/model/online_soft_band_gru_sfc_2d.py \
  spectral_feature_compression/core/model/online_soft_band_dilated_sfc_2d.py \
  spectral_feature_compression/core/model/online_soft_band_query_dilated_sfc_2d.py \
  spectral_feature_compression/core/model/online_crossattn_query_sfc_2d.py \
  spectral_feature_compression/core/model/online_hierarchical_soft_band_sfc_2d.py \
  spectral_feature_compression/core/model/online_hierarchical_soft_band_ffi_sfc_2d.py \
  spectral_feature_compression/core/model/online_hierarchical_soft_band_parallel_ffi_sfc_2d.py \
  spectral_feature_compression/core/model/sparse_unet_mel_sfc_2d.py \
  spectral_feature_compression/core/model/source_split_sfc_2d.py \
  spectral_feature_compression/core/model/residual_refinement_sfc_2d.py \
  spectral_feature_compression/core/model/adaptive_mel_sfc_2d.py \
  tools/online/export_onnx_online_model.py \
  tools/online/run_streaming_inference.py \
  tools/online/verify_npu_variants.py \
  tests/test_online_frequency_preprocessing.py

./.venv/bin/python -m ruff check \
  spectral_feature_compression/core/model/frequency_preprocessing.py \
  spectral_feature_compression/core/model/online_wrapper.py \
  spectral_feature_compression/core/model/online_soft_band_sfc_2d.py \
  spectral_feature_compression/core/model/online_soft_band_query_sfc_2d.py \
  spectral_feature_compression/core/model/online_hard_band_sfc_2d.py \
  spectral_feature_compression/core/model/online_soft_band_gru_sfc_2d.py \
  spectral_feature_compression/core/model/online_soft_band_dilated_sfc_2d.py \
  spectral_feature_compression/core/model/online_soft_band_query_dilated_sfc_2d.py \
  spectral_feature_compression/core/model/online_crossattn_query_sfc_2d.py \
  spectral_feature_compression/core/model/online_hierarchical_soft_band_sfc_2d.py \
  spectral_feature_compression/core/model/online_hierarchical_soft_band_ffi_sfc_2d.py \
  spectral_feature_compression/core/model/online_hierarchical_soft_band_parallel_ffi_sfc_2d.py \
  spectral_feature_compression/core/model/sparse_unet_mel_sfc_2d.py \
  spectral_feature_compression/core/model/source_split_sfc_2d.py \
  spectral_feature_compression/core/model/residual_refinement_sfc_2d.py \
  spectral_feature_compression/core/model/adaptive_mel_sfc_2d.py \
  tools/online/export_onnx_online_model.py \
  tools/online/run_streaming_inference.py \
  tools/online/verify_npu_variants.py \
  tests/test_online_frequency_preprocessing.py

./.venv/bin/python -m pytest \
  tests/test_online_frequency_preprocessing.py \
  tests/test_proposed_separation_models.py

./.venv/bin/python tools/online/export_onnx_online_model.py \
  recipes/dnr/models/band-sfc-net-npu.rt-plus.stage1-supervised-aug-dcbypass.rt192k.fp512/config.yaml \
  --out /tmp/opencode/band_sfc_rt_plus_dcbypass_core.onnx \
  --n-chan 1 --frames 1 --opset 14 --disable-masking --check \
  --deploy-manifest-out /tmp/opencode/band_sfc_rt_plus_dcbypass_manifest.json

./.venv/bin/python tools/online/export_onnx_online_model.py \
  recipes/dnr/models/band-sfc-net-npu.rt-plus.stage1-supervised-aug-pcen.rt192k.fp512/config.yaml \
  --out /tmp/opencode/band_sfc_rt_plus_pcen_core.onnx \
  --n-chan 1 --frames 1 --opset 14 --disable-masking \
  --deploy-manifest-out /tmp/opencode/band_sfc_rt_plus_pcen_manifest.json

git diff --check
```

Results:

- `py_compile`: pass.
- `ruff check`: pass, with only the existing pyproject deprecation warning.
- `pytest`: 20 passed, with existing CUDA/autocast warnings and local
  `.pytest_cache` permission warning.
- `infer_n_chan` now reports `1` for the inherited PCEN/DC DnR recipes.
- PCEN manifest records external state shape `[1, 1, 1, 512]`.
- DC-bypass export/check passed; deploy manifest records core `freqs: 512`,
  `n_chan: 1`, frequency preprocessing `input_n_freq: 1024`, and DC metadata.
- PCEN export passed; deploy manifest records core `freqs: 512`, `n_chan: 1`,
  and PCEN wrapper metadata while keeping PCEN outside the exported core.
- Raw unsimplified ONNX still reports `And`, `GreaterOrEqual`, and `Less` as
  disallowed by the quick allowlist, so final NPU validation should continue to
  use the established forced simplification/ONE flow.
- `git diff --check`: pass.

## BandSFC RT+ 2-Mask Residual SFX Ablation

Scope:

- Added an opt-in DnR ablation where the NPU core predicts only Speech and
  Music.  The wrapper reconstructs SFX as `mixture - speech - music` so the
  external training/evaluation contract remains the standard 3-stem order:
  Speech, Music, SFX.

Changes:

- Added residual-source reconstruction support to
  `FrequencyPreprocessedOnlineModel`.
- Added `core_n_src`, `residual_source_enabled`, and `residual_source_index` to
  `BandSFCNetNPU.training_wrapper.build_band_sfc_net_npu_system`.
- Added recipe:
  `recipes/dnr/models/band-sfc-net-npu.rt-plus.2mask-residual-sfx.rt192k.fp512/config.yaml`.
- Added residual-source metadata to ONNX deploy manifests and streaming run
  manifests.
- Added a result-manifest TODO row in
  `docs/templates/audio_separation_results_manifest.csv` for the 2-mask
  residual-SFX ablation.
- Added tests that verify mixture consistency and the 3-stem wrapper / 2-source
  core contract.

Validation commands:

```bash
PYTHONPYCACHEPREFIX=/tmp/opencode/pycache \
  ./.venv/bin/python -m py_compile \
  spectral_feature_compression/core/model/frequency_preprocessing.py \
  BandSFCNetNPU/training_wrapper.py \
  tools/online/export_onnx_online_model.py \
  tools/online/run_streaming_inference.py \
  tests/test_online_frequency_preprocessing.py

./.venv/bin/python -m ruff check \
  spectral_feature_compression/core/model/frequency_preprocessing.py \
  BandSFCNetNPU/training_wrapper.py \
  tools/online/export_onnx_online_model.py \
  tools/online/run_streaming_inference.py \
  tests/test_online_frequency_preprocessing.py

./.venv/bin/python -m pytest \
  tests/test_online_frequency_preprocessing.py \
  tests/test_proposed_separation_models.py

./.venv/bin/python - <<'PY'
from pathlib import Path
import torch
from tools.online.export_onnx_online_model import build_model_system_from_recipe_config
p = Path('recipes/dnr/models/band-sfc-net-npu.rt-plus.2mask-residual-sfx.rt192k.fp512/config.yaml')
system = build_model_system_from_recipe_config(p).eval()
core = system.model.core.eval()
x = torch.randn(1, 2, 1, core.n_freq)
with torch.no_grad():
    y = core(x)
print('core_n_src', core.n_src, 'core_n_freq', core.n_freq, 'core_output_shape', tuple(y.shape))
PY

./.venv/bin/python tools/online/export_onnx_online_model.py \
  recipes/dnr/models/band-sfc-net-npu.rt-plus.2mask-residual-sfx.rt192k.fp512/config.yaml \
  --out /tmp/opencode/band_sfc_rt_plus_2mask_residual_sfx_core.onnx \
  --n-chan 1 --frames 1 --opset 14 --disable-masking --check \
  --deploy-manifest-out /tmp/opencode/band_sfc_rt_plus_2mask_residual_sfx_manifest.json

./.venv/bin/python tools/online/verify_npu_variants.py \
  --mode recipe \
  --recipe-name-contains band-sfc-net-npu.rt-plus.2mask-residual-sfx.rt192k.fp512 \
  --run-name focused_onnx_one_20260530_2mask_residual_sfx \
  --force-onnxsim-large-shape-ops

git diff --check
```

Results:

- `py_compile`: pass.
- `ruff check`: pass, with only the existing pyproject deprecation warning.
- `pytest`: 22 passed, with existing CUDA/autocast warnings and local
  `.pytest_cache` permission warning.
- Recipe instantiation: `FrequencyPreprocessedOnlineModel` has external
  `n_src=3`; exported core has `n_src=2` and `n_freq=512`.
- Core tensor smoke check reports output shape `(1, 4, 1, 512)` with masking
  enabled, matching two mono complex source estimates.
- Config-only ONNX export/check passed.  The deploy manifest records
  `core_n_src: 2`, frequency preprocessing `target_bins: 512`, and residual
  metadata `explicit_n_src: 2`, `output_n_src: 3`, `residual_source_index: 2`.
- Result manifest validation now reports `22 rows x 44 columns`.
- The `--disable-masking` raw-head export outputs 8 channels for RT+ because the
  residual correction head doubles the explicit 2-source complex mask channels;
  masking-enabled core output remains 4 packed channels.
- Full ONNX/ONE verifier pass:
  `logs/npu_verify_general/focused_onnx_one_20260530_2mask_residual_sfx/summary.md`
  reports 1 PASS / 0 FAIL.
- Raw unsimplified ONNX still reports `And`, `GreaterOrEqual`, and `Less` as
  disallowed by the quick allowlist, so final NPU validation should continue to
  use the established forced simplification/ONE flow.
- `git diff --check`: pass.
