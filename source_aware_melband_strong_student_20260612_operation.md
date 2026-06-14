# Source-Aware MelBand Strong NPU Student

Date: 2026-06-12

## Trigger

The previous pass introduced a performance-first `SourceAwareMelBandRoformer`
teacher and then a strict NPU student.  The conservative student was useful as a
baseline, but it was too small and too close to existing repo blocks.  This pass
continued the work by adding a stronger handcrafted deployment student that keeps
more of the teacher's source-aware separation structure while staying inside the
online/NPU constraints.

## Strong student design

New model file:

```text
spectral_feature_compression/core/model/source_aware_melband_strong_student_sfc_2d.py
```

Main classes:

- `OnlineSourceAwareMelBandStrongStudentSFC2D`
- `OnlineSourceAwareMelBandStrongStudentSFCModel`
- `StrongAdaptiveMelRouter2d`
- `StrongTemporalBandBlock2d`
- `StrongTokenFFN2d`
- `StrongSourceSeed2d`
- `StrongSourceCompetitionBlock2d`
- `StrongSourceDecoder2d`
- `StrongMelBandExpander2d`
- `StrongSourceMaskHead2d`
- `StrongMaskCorrectionHead2d`

Registered builder:

```text
spectral_feature_compression.core.model.proposed_separation_models.build_source_aware_melband_roformer_strong_student_npu_system
```

Recipes:

```text
recipes/dnr/models/source-aware-melband-roformer.student-npu-strong.rt192k.fp512keep475/config.yaml
recipes/dnr/models/source-aware-melband-roformer.student-npu-strong.distill.rt192k.fp512keep475/config.yaml
```

The strong student uses NPU-friendly primitives only:

- `Conv2d` and depthwise `Conv2d`;
- rank-4 `MatMul` for adaptive mel routing and expansion;
- `Softmax`, `Sigmoid`, reductions, and elementwise ops;
- `RMSNorm2d` implemented with basic tensor ops;
- packed 4D tensors `[B, C, T, F]` throughout the exported path.

The architecture intentionally avoids attention/rotary ops, explicit 5D source
tensors, recurrent attention caches, adaptive pooling, `Tile`, `Expand`, and
unsupported control flow.

## Architecture

```text
packed complex STFT [B, 2M, T, F]
  -> RI + optional magnitude frontend
  -> custom adaptive mel router F=512 -> K=80
  -> custom causal temporal/band encoder blocks
  -> learned source seeding for Speech/Music/SFX streams
  -> repeated stateless source competition blocks:
       source token
       other-source mean
       mixture token
       source-vs-other deltas
  -> query-conditioned mel expansion K=80 -> F=512
  -> source-shared full-band complex mask head
  -> low-rank full-band mask correction
  -> packed complex mask application
  -> 4D mixture-consistency projection
```

The source decoder is stateless in time.  Streaming state is spent only on the
router, two encoder blocks, and the final correction block.

## Deploy recipe profile

From:

```text
recipes/dnr/models/source-aware-melband-roformer.student-npu-strong.rt192k.fp512keep475/config.yaml
```

Measured with `build_model_system_from_recipe_config`:

```text
core = OnlineSourceAwareMelBandStrongStudentSFC2D
n_freq = 512
n_bands = 80
d_model = 48
n_source_layers = 5
parameters = 1,381,501
stream_context_frames = 8
state_tensors = 3
fp16 streaming state = 190,464 bytes
```

The fp16 stream state is 186 KiB, under the 192 KiB DSP state target.

## Strict ONNX/NPU export continuation

A deploy-sized streaming ONNX export initially passed the recommended op
allowlist, but the stricter repo audit reported risk patterns:

```text
has_strict_edge_risks = True
dynamic_slice_bounds = 116
dynamic_slice_with_dynamic_non_axis_dims = 104
activation_matmul_rank_le3 = 7
rank_gt4_values = 0
```

Root causes:

- `Tensor.chunk` and explicit channel slicing exported as dynamic `Slice` nodes
  in opset 14.
- `torch.bmm` exported as rank-3 activation `MatMul`, which the repo's strict
  TIGER/ONE-derived audit treats as risky even though the graph used only
  allowlisted ops.

Fixes applied in
`spectral_feature_compression/core/model/source_aware_melband_strong_student_sfc_2d.py`:

- replaced channel slicing and `chunk` with static `torch.split`;
- replaced router/expander `torch.bmm` calls with rank-4 `torch.matmul`;
- kept tensor rank at 4 or below and preserved forward/streaming equivalence.

After the fix, the deploy-sized streaming export is strict-risk-clean.

## Export command

```bash
cd /home/cmj/works/ASS
PYTHONPATH=.:aiaccel .venv/bin/python tools/online/export_onnx_online_model.py \
  recipes/dnr/models/source-aware-melband-roformer.student-npu-strong.rt192k.fp512keep475/config.yaml \
  --out /tmp/source_aware_melband_strong_student_stream.onnx \
  --n-chan 1 \
  --frames 1 \
  --streaming \
  --externalize-band-constants \
  --constants-out /tmp/source_aware_melband_strong_student_consts.npz \
  --state-meta-out /tmp/source_aware_melband_strong_student_state.json \
  --check
```

Result:

```text
Core: OnlineSourceAwareMelBandStrongStudentSFC2D
Opset: 14
Streaming export: True
Input shape: (1, 2, 1, 512)
Flattened state tensors: 3
Externalized constant tensors: 3
ONNX checker: passed
Disallowed ops: none
```

Observed ONNX ops after the fix:

```text
Add, Concat, Constant, Conv, Div, Identity, MatMul, Mul, ReduceMean,
ReduceSum, Sigmoid, Slice, Softmax, Split, Sqrt, Sub, Transpose
```

## Strict audit command

```bash
cd /home/cmj/works/ASS
PYTHONPATH=.:aiaccel .venv/bin/python tools/online/audit_onnx_model.py \
  /tmp/source_aware_melband_strong_student_stream.onnx \
  --risk-profile tiger_one_strict_edge \
  --state-meta /tmp/source_aware_melband_strong_student_state.json \
  --budget-kib 192 \
  --budget-dtype fp16
```

Result:

```text
Streaming state (fp16 estimate): 190464 B (186.00 KiB)
Externalized band constants (fp16 estimate): 245760 B (240.00 KiB)
Disallowed ops: none
Risk counts:
  tile: 0
  constant_of_shape: 0
  expand: 0
  prelu: 0
  dynamic_slice_bounds: 0
  dynamic_slice_with_dynamic_non_axis_dims: 0
  scalar_gather: 0
  activation_matmul_rank_le3: 0
  matmul_rank3_rhs2_nonconst: 0
  rank_gt4_values: 0
  transpose_perm_not_int32: 0
  high_transpose_count: 0
Strict-edge risks: False
```

Note: the audit's combined parameter-payload budget line includes ONNX weights
and externalized constants together with stream state.  The 192 KiB target for
this model is the runtime streaming state; weights/constants are packaged
separately.

## Validation

Focused strong-student tests:

```bash
cd /home/cmj/works/ASS
PYTHONPATH=.:aiaccel .venv/bin/python -m pytest \
  tests/test_proposed_separation_models.py::test_source_aware_melband_roformer_strong_student_npu_forward_streaming_and_recipe \
  tests/test_proposed_separation_models.py::test_source_aware_melband_roformer_strong_student_npu_onnx_audit_smoke \
  -q
```

Result:

```text
2 passed
```

After tightening the strong ONNX smoke to opset 14:

```bash
cd /home/cmj/works/ASS
PYTHONPATH=.:aiaccel .venv/bin/python -m pytest \
  tests/test_proposed_separation_models.py::test_source_aware_melband_roformer_strong_student_npu_onnx_audit_smoke \
  -q
```

Result:

```text
1 passed
```

Ruff:

```bash
cd /home/cmj/works/ASS
.venv/bin/python -m ruff check \
  spectral_feature_compression/core/model/source_aware_melband_strong_student_sfc_2d.py
```

Result:

```text
All checks passed
```

Project `tests/` suite:

```bash
cd /home/cmj/works/ASS
PYTHONPATH=.:aiaccel .venv/bin/python -m pytest tests -q
```

Result:

```text
63 passed
```

A full-repository `pytest -q` was also attempted, but collection enters vendored
or project-ignored folders such as `dcase2026baseline` and `hydra`, then fails on
missing optional modules (`pytest_mock`, `src`, `hydra_plugins`, `boto3`).  This
is unrelated to the source-aware MelBand student changes.

## Training commands

Supervised strong student:

```bash
cd /home/cmj/works/ASS
PYTHONPATH=.:aiaccel .venv/bin/python -m aiaccel.torch.apps.train \
  recipes/dnr/models/source-aware-melband-roformer.student-npu-strong.rt192k.fp512keep475/config.yaml
```

Distillation from the trained teacher:

```bash
cd /home/cmj/works/ASS
PYTHONPATH=.:aiaccel .venv/bin/python -m aiaccel.torch.apps.train \
  recipes/dnr/models/source-aware-melband-roformer.student-npu-strong.distill.rt192k.fp512keep475/config.yaml \
  teacher_checkpoint_path=/path/to/source_aware_melband_roformer_teacher.ckpt
```

## Code review follow-up

A follow-up code review found several issues that were fixed before training:

1. The conservative student ONNX smoke was a false positive because it used a
   tiny graph and opset 18.  The test now uses the same legacy ONNX exporter mode
   and opset 14 as `tools/online/export_onnx_online_model.py`, and it checks both
   the allowed-op preset and the strict risk audit.
2. Shared NPU/export helpers were updated to avoid strict-edge graph risks:
   channel slicing and `chunk` were replaced with static `torch.split`, and
   `torch.bmm` routing/expansion was replaced with rank-4 `torch.matmul`.
3. The conservative deploy recipe now exports strict-risk-clean as well:

   ```text
   source-aware-melband-roformer.student-npu.rt192k.fp512keep475
   Disallowed ops: none
   dynamic_slice_bounds: 0
   dynamic_slice_with_dynamic_non_axis_dims: 0
   activation_matmul_rank_le3: 0
   rank_gt4_values: 0
   Strict-edge risks: False
   Streaming state fp16 estimate: 167,936 B (164.00 KiB)
   ```

4. The strong student's correction streaming state accounting now supports
   stateless correction blocks such as `kernel_size=(1, 3)`.
5. Distillation recipes now set `teacher_css_validation: true`, so validation
   teacher outputs use chunked CSS instead of full-record teacher inference.
6. Teacher checkpoint loading now prefers `ema_model.module.*` weights over raw
   `model.*` weights when both are present.
7. Mask/logit distillation fallback behavior is now documented in code: if a
   model does not expose true mask/logit auxiliary tensors, the task intentionally
   uses waveform-derived spectral pseudo-masks.
8. Public lazy exports for the new source-aware MelBand symbols are covered by
   tests.

Updated validation after these fixes:

```bash
cd /home/cmj/works/ASS
.venv/bin/python -m ruff check \
  spectral_feature_compression/core/model/adaptive_mel_sfc_2d.py \
  spectral_feature_compression/core/model/online_sfc_2d.py \
  spectral_feature_compression/core/model/online_soft_band_dilated_sfc_2d.py \
  spectral_feature_compression/core/model/online_soft_band_query_sfc_2d.py \
  spectral_feature_compression/core/model/residual_refinement_sfc_2d.py \
  spectral_feature_compression/core/model/source_aware_residual_sfc_2d.py \
  spectral_feature_compression/core/model/source_aware_melband_student_sfc_2d.py \
  spectral_feature_compression/core/model/source_aware_melband_strong_student_sfc_2d.py \
  spectral_feature_compression/core/tasks/distillation_task.py \
  tests/test_proposed_separation_models.py \
  spectral_feature_compression/__init__.py \
  spectral_feature_compression/core/model/proposed_separation_models.py \
  spectral_feature_compression/core/tasks/composite_sup_task.py
```

Result:

```text
All checks passed
```

```bash
cd /home/cmj/works/ASS
PYTHONPATH=.:aiaccel .venv/bin/python -m pytest tests/test_proposed_separation_models.py -q
PYTHONPATH=.:aiaccel .venv/bin/python -m pytest tests -q
```

Results:

```text
tests/test_proposed_separation_models.py: 26 passed
tests/: 65 passed
```

The actual deploy-sized streaming exports for both conservative and strong
students were also rerun with `--streaming --externalize-band-constants --check`
and both passed ONNX checker, the recommended op allowlist, and the strict
`tiger_one_strict_edge` risk audit.

## Next steps

1. Train the teacher recipe and confirm it improves validation separation
   quality.
2. Train the strong student with supervised loss, then distill from the trained
   teacher checkpoint.
3. Compare teacher, conservative student, and strong student on source-wise
   SI-SDR/SNR, leakage, transient quality, and low-frequency fidelity.
4. If quality is acceptable, run the full ONE toolchain import -> optimize ->
   quantize path using the exported streaming ONNX and state/constants manifest.
