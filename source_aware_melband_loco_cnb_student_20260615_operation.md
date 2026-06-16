# Source-Aware MelBand Loco-CNB Student

Date: 2026-06-15

## Trigger

The previous code review concluded that the strongest strict-NPU alternative
should not be only a larger source-aware student or a generic BandSFCNetNPU
variant.  The desired next version should combine:

- the source-aware MelBand student's explicit Speech/Music/Effects separation
  priors;
- the adaptive-mel Loco-CNB branch's longer causal FSMN memory;
- strict 4D tensor and NPU-friendly operator constraints;
- a deploy recipe that stays below the 192 KiB fp16 runtime-state target.

## Implemented model

New model file:

```text
spectral_feature_compression/core/model/source_aware_melband_loco_cnb_student_sfc_2d.py
```

Main classes:

- `OnlineSourceAwareMelBandLocoCNBStudentSFC2D`
- `OnlineSourceAwareMelBandLocoCNBStudentSFCModel`
- `SourceAwareLocoCNBBlock2d`
- `LocoLocalTFMixer2d`
- `LocoCrossBandMixer2d`
- `LocoFSMNBandMixer2d`
- `LocoCompressedBandAttentionFusion2d`
- `LocoPooledChannelMixer2d`

Registered proposal builder:

```text
spectral_feature_compression.core.model.proposed_separation_models.build_source_aware_melband_loco_cnb_student_npu_system
```

Public lazy exports were added for the new core/model and Loco-CNB helper
classes in:

```text
spectral_feature_compression/__init__.py
```

## Architecture

```text
packed complex STFT [B, 2M, T, F=512]
  -> RI + magnitude features
  -> 1x1 frontend + token FFN
  -> adaptive overlapped Mel router F=512 -> K=56
  -> shared causal Loco-CNB backbone:
       local TF gated mixer
       grouped cross-band mixer
       causal FSMN mixer with dilation schedule [1, 2, 3]
       optional compressed-band attention branch
       frequency-pooled channel capacity mixer
  -> source-width projection
  -> learned source seeding
  -> stateless source competition decoder
  -> query-conditioned Mel expansion K=56 -> F=512
  -> source-shared full-band complex mask head
  -> stateless full-band correction head
  -> mixture consistency
```

Important design choice:

- persistent streaming state is spent only on the compressed shared Loco-CNB
  backbone;
- source decoding, source competition, Mel expansion, full-band local mask head,
  and full-band correction are stateless in time;
- `correction_kernel_size[0]` is required to be `1` so the full-band correction
  does not consume the state budget.

## Recipes

Added supervised explicit three-stem recipe:

```text
recipes/dnr/models/source-aware-melband-loco-cnb.student-npu.rt192k.fp512keep475/config.yaml
```

Added recommended residual-SFX recipe:

```text
recipes/dnr/models/source-aware-melband-loco-cnb.student-npu-residual-sfx.rt192k.fp512keep475/config.yaml
```

This explicit core predicts Speech and Music only:

```yaml
core_n_src: 2
residual_source_enabled: true
residual_source_index: 2
loco_cnb_mixture_consistency: false
```

Core-level mixture consistency is intentionally disabled for residual-SFX. If the
2-stem explicit core is forced to sum to the mixture, the wrapper residual
`effects = mixture - speech - music` collapses to zero/projector residual instead
of becoming a useful SFX stem. The builder now rejects
`residual_source_enabled=True` with `mixture_consistency=True` to prevent future
unsafe residual recipes.

The wrapper reconstructs:

```text
effects = mixture - speech - music
```

Added teacher-distillation recipe:

```text
recipes/dnr/models/source-aware-melband-loco-cnb.student-npu-residual-sfx.distill.rt192k.fp512keep475/config.yaml
```

The distillation teacher is the source-aware MelBand RoFormer teacher:

```text
build_source_aware_melband_roformer_teacher_system
```

## Deploy profile

Default deploy shape:

```yaml
n_bands: 56
state_channels: 36
source_channels: 48
n_loco_layers: 4
n_source_layers: 4
cnb_kernel: 4
cnb_dilation_schedule: [1, 2, 3]
loco_time_kernel: 3
loco_band_kernel: 3
pooled_mixer_hidden_schedule: [2048, 4096, 4096, 2048]
correction_kernel_size: [1, 5]
cnb_attention_enabled: false
```

Measured core profile:

```text
explicit 3-source params: 2,461,786
residual-SFX core params: 2,452,408
fp16 streaming state: 177,408 B / 173.25 KiB
stream context: 44 compressed-band frames
state tensors: 4 nested stage states / 8 flattened tensors
```

State formula for the deploy recipe:

```text
4 stages * (local 2 + FSMN 9) frames * 36 channels * 56 bands * 2 bytes
= 177,408 B
```

This keeps runtime state under the 192 KiB fp16 target.

## Attention branch note

`LocoCompressedBandAttentionFusion2d` is implemented and can be enabled with:

```yaml
cnb_attention_enabled: true
```

The first full ONE attempt with the attention branch enabled failed in
`one-optimize` at `CircleBatchMatMul.cpp:140`:

```text
Internal Exception. x_rhs and y_lhs should be same
```

Follow-up analysis showed this was **not an intrinsic attention-operator
failure**.  It was the same static one-frame rank-4 MatMul lowering issue that
also affects the adaptive Mel router/expander MatMuls.  With the
`matmul_unit_batch_rank4_to_2d` pre-import rewrite added to
`tools/online/verify_npu_variants.py`, a temporary attention-enabled residual-SFX
recipe passed streaming ONE import -> optimize -> quantize:

```text
MatMul: 14
Softmax: 5
matmul_unit_batch_rank4_to_2d: 14
ONE result: PASS, channel quantization, no layer fallback
```

The committed deploy recipes still keep the branch disabled by default as the
lower-node/latency conservative target.  Attention is now a valid quality
ablation candidate rather than a known compiler blocker.  If enabled, use the
same verifier/pre-import rewrite path for ONE compilation.

## Verifier update

The strict deploy ONNX intentionally uses rank-4 MatMul for SFC transport to
avoid rank-3 activation MatMul risks.  ONE can import these nodes, but the
current optimize flow can fail after layout conversion for the static streaming
shape `[1, 1, M, K] @ [1, 1, K, N]`.

Updated:

```text
tools/online/verify_npu_variants.py
```

New pre-import compatibility rewrite:

```text
rewrite_unit_batch_rank4_matmul_to_2d
```

It rewrites only the static one-frame streaming pattern:

```text
[1, 1, M, K] @ [1, 1, K, N]
  -> Reshape [M, K]
  -> Reshape [K, N]
  -> MatMul2D
  -> Reshape [1, 1, M, N]
```

This preserves the exported strict ONNX while giving ONE a simpler import/opt
pattern.

The lightweight recipe parser in `tools/online/export_onnx_online_model.py` was
also made indentation-robust for `task.model` blocks, because some YAML edits can
use four-space indentation while older recipes use two-space indentation.

## ONNX export and strict audit

Deploy-sized streaming export command:

```bash
cd /home/cmj/works/ASS
PYTHONPATH=.:aiaccel .venv/bin/python tools/online/export_onnx_online_model.py \
  recipes/dnr/models/source-aware-melband-loco-cnb.student-npu-residual-sfx.rt192k.fp512keep475/config.yaml \
  --out /tmp/source_aware_melband_loco_cnb_residual_sfx_stream.onnx \
  --n-chan 1 \
  --frames 1 \
  --streaming \
  --externalize-band-constants \
  --constants-out /tmp/source_aware_melband_loco_cnb_residual_sfx_consts.npz \
  --state-meta-out /tmp/source_aware_melband_loco_cnb_residual_sfx_state.json \
  --check
```

Result:

```text
Core: OnlineSourceAwareMelBandLocoCNBStudentSFC2D
Opset: 14
Streaming export: True
Input shape: (1, 2, 1, 512)
Flattened state tensors: 8
Externalized constant tensors: 3
ONNX checker: passed
Disallowed ops: none
```

Observed ONNX ops:

```text
Add, Concat, Constant, Conv, Div, Identity, MatMul, Mul, ReduceMean,
ReduceSum, Sigmoid, Slice, Softmax, Split, Sqrt, Sub, Transpose
```

Strict audit command:

```bash
cd /home/cmj/works/ASS
PYTHONPATH=.:aiaccel .venv/bin/python tools/online/audit_onnx_model.py \
  /tmp/source_aware_melband_loco_cnb_residual_sfx_stream.onnx \
  --risk-profile tiger_one_strict_edge \
  --state-meta /tmp/source_aware_melband_loco_cnb_residual_sfx_state.json \
  --budget-kib 192 \
  --budget-dtype fp16
```

Result:

```text
Streaming state: 177408 B (173.25 KiB)
Externalized band constants: 172032 B (168.00 KiB)
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
Strict-edge risks: False
```

Note: the audit's combined parameter-payload budget line includes model weights
and externalized constants.  The 192 KiB target here is the runtime streaming
state budget.

## Full ONE verification

Streaming ONE verification command:

```bash
cd /home/cmj/works/ASS
.venv/bin/python tools/online/verify_npu_variants.py \
  --mode recipe \
  --recipe-name-contains source-aware-melband-loco-cnb.student-npu-residual-sfx.rt192k.fp512keep475 \
  --run-name source_aware_melband_loco_cnb_residual_sfx_noattn_rank4fix_20260615 \
  --force-onnxsim-large-shape-ops \
  --quantize-layer-fallback \
  --streaming
```

Result:

```text
[PASS] recipe:source-aware-melband-loco-cnb.student-npu-residual-sfx.rt192k.fp512keep475
```

Artifact-verified outputs were produced during the run:

```text
model.circle
model.opt.circle
model.q.circle
```

Pre-import rewrite count from the run log:

```text
matmul_unit_batch_rank4_to_2d: 6
```

Quantization used channel granularity and did not require layer fallback.  The
large generated verifier artifact directory was removed after validation to
avoid leaving about 46 MB of untracked files; rerun the command above to
recreate the artifacts.

## How to compile through the verifier/pre-import rewrite path

Use the project verifier script for actual ONE compilation, not raw `onecc`
directly.  The verifier applies the ONNX simplification and the pre-import
compatibility rewrites before calling ONE:

```text
PyTorch/config -> ONNX export
  -> onnxsim
  -> NPU_ONNX_IMPORT_PREP rewrites
  -> calibration H5 generation
  -> one-import-onnx
  -> one-optimize
  -> one-quantize
```

Default residual-SFX compile command:

```bash
cd /home/cmj/works/ASS
.venv/bin/python tools/online/verify_npu_variants.py \
  --mode recipe \
  --recipe-name-contains source-aware-melband-loco-cnb.student-npu-residual-sfx.rt192k.fp512keep475 \
  --run-name source_aware_melband_loco_cnb_residual_sfx_compile \
  --force-onnxsim-large-shape-ops \
  --quantize-layer-fallback \
  --streaming
```

Expected result:

```text
[PASS] recipe:source-aware-melband-loco-cnb.student-npu-residual-sfx.rt192k.fp512keep475
```

Generated files are written under:

```text
logs/npu_verify_general/source_aware_melband_loco_cnb_residual_sfx_compile/
```

Key artifacts inside the recipe subdirectory:

```text
model.onnx
model.sim.onnx
model.circle
model.opt.circle
model.q.circle
run.log
summary.md
summary.json
```

To confirm the pre-import rewrite ran, inspect `run.log` and find:

```text
=== NPU_ONNX_IMPORT_PREP ===
```

For the default attention-disabled recipe, expect:

```text
matmul_unit_batch_rank4_to_2d: 6
```

That count corresponds to the adaptive Mel router/expander transport MatMuls.
For the attention-enabled variant below, expect:

```text
matmul_unit_batch_rank4_to_2d: 14
```

That includes the additional compressed-band attention MatMuls.

### Attention-enabled compile overlay

The committed deploy recipe keeps attention disabled by default:

```yaml
loco_cnb_attention_enabled: false
```

To compile the attention ablation without editing the committed recipe, create a
temporary overlay recipe:

```bash
cd /home/cmj/works/ASS
mkdir -p tmp_attention_recipe/source-aware-melband-loco-cnb.student-npu-residual-sfx-attn.rt192k.fp512keep475
cat > tmp_attention_recipe/source-aware-melband-loco-cnb.student-npu-residual-sfx-attn.rt192k.fp512keep475/config.yaml <<'YAML'
_base_: /home/cmj/works/ASS/recipes/dnr/models/source-aware-melband-loco-cnb.student-npu-residual-sfx.rt192k.fp512keep475/config.yaml

loco_cnb_attention_enabled: true
YAML
```

Compile the overlay through the verifier:

```bash
cd /home/cmj/works/ASS
.venv/bin/python tools/online/verify_npu_variants.py \
  --mode recipe \
  --recipe-root tmp_attention_recipe \
  --recipe-name-contains attn \
  --run-name source_aware_melband_loco_cnb_residual_sfx_attn_compile \
  --force-onnxsim-large-shape-ops \
  --quantize-layer-fallback \
  --streaming
```

Expected pre-import rewrite count in `run.log`:

```text
matmul_unit_batch_rank4_to_2d: 14
```

Expected ONE result:

```text
PASS, channel quantization, no layer fallback
```

After collecting artifacts, remove temporary files if they are no longer needed:

```bash
rm -rf tmp_attention_recipe
rm -rf logs/npu_verify_general/source_aware_melband_loco_cnb_residual_sfx_attn_compile
```

### ONNX export/audit is not the same as ONE compile

The standalone export/audit path is useful for graph inspection:

```bash
cd /home/cmj/works/ASS
PYTHONPATH=.:aiaccel .venv/bin/python tools/online/export_onnx_online_model.py \
  recipes/dnr/models/source-aware-melband-loco-cnb.student-npu-residual-sfx.rt192k.fp512keep475/config.yaml \
  --out /tmp/source_aware_melband_loco_cnb_residual_sfx_stream.onnx \
  --n-chan 1 \
  --frames 1 \
  --streaming \
  --externalize-band-constants \
  --constants-out /tmp/source_aware_melband_loco_cnb_residual_sfx_consts.npz \
  --state-meta-out /tmp/source_aware_melband_loco_cnb_residual_sfx_state.json \
  --check

PYTHONPATH=.:aiaccel .venv/bin/python tools/online/audit_onnx_model.py \
  /tmp/source_aware_melband_loco_cnb_residual_sfx_stream.onnx \
  --risk-profile tiger_one_strict_edge \
  --state-meta /tmp/source_aware_melband_loco_cnb_residual_sfx_state.json \
  --budget-kib 192 \
  --budget-dtype fp16
```

However, this path does **not** run the ONE pre-import rewrite.  For actual ONE
artifacts (`model.circle`, `model.opt.circle`, `model.q.circle`), use
`tools/online/verify_npu_variants.py` as shown above.

## Tests and lint

Targeted tests:

```bash
cd /home/cmj/works/ASS
PYTHONPATH=.:aiaccel .venv/bin/python -m pytest \
  tests/test_proposed_separation_models.py::test_source_aware_melband_loco_cnb_student_npu_forward_streaming_and_recipe \
  tests/test_proposed_separation_models.py::test_source_aware_melband_loco_cnb_student_npu_onnx_audit_smoke \
  -q
```

Result:

```text
2 passed
```

Full proposal test file:

```bash
cd /home/cmj/works/ASS
PYTHONPATH=.:aiaccel .venv/bin/python -m pytest tests/test_proposed_separation_models.py -q
```

Result:

```text
29 passed
```

Ruff:

```bash
cd /home/cmj/works/ASS
.venv/bin/python -m ruff check \
  spectral_feature_compression/core/model/source_aware_melband_loco_cnb_student_sfc_2d.py \
  spectral_feature_compression/core/model/proposed_separation_models.py \
  spectral_feature_compression/__init__.py \
  tools/online/export_onnx_online_model.py \
  tools/online/verify_npu_variants.py \
  tests/test_proposed_separation_models.py
```

Result:

```text
All checks passed
```

## Training commands

Supervised explicit three-stem:

```bash
cd /home/cmj/works/ASS
PYTHONPATH=.:aiaccel .venv/bin/python -m aiaccel.torch.apps.train \
  recipes/dnr/models/source-aware-melband-loco-cnb.student-npu.rt192k.fp512keep475/config.yaml
```

Recommended first supervised residual-SFX run:

```bash
cd /home/cmj/works/ASS
PYTHONPATH=.:aiaccel .venv/bin/python -m aiaccel.torch.apps.train \
  recipes/dnr/models/source-aware-melband-loco-cnb.student-npu-residual-sfx.rt192k.fp512keep475/config.yaml
```

Distillation from a trained source-aware MelBand RoFormer teacher:

```bash
cd /home/cmj/works/ASS
PYTHONPATH=.:aiaccel .venv/bin/python -m aiaccel.torch.apps.train \
  recipes/dnr/models/source-aware-melband-loco-cnb.student-npu-residual-sfx.distill.rt192k.fp512keep475/config.yaml \
  teacher_checkpoint_path=/path/to/source_aware_melband_roformer_teacher.ckpt
```

## Next steps

1. Train the residual-SFX recipe long enough to compare against the old
   adaptive-mel Loco-CNB `~3.5 dB` plateau.
2. If supervised training remains low, distill from the source-aware MelBand
   RoFormer teacher before changing the architecture again.
3. Track per-stem metrics separately.  If SFX quality is poor but speech/music
   improve, keep residual-SFX and tune the training objective.  If speech/music
   leak strongly into SFX, compare against the explicit three-stem recipe.
4. Revisit `cnb_attention_enabled: true` only if ONE's BatchMatMul optimize issue
   is solved or if a safer attention-lowering rewrite is added.
