# DNR SFC NPU Import Root Cause Analysis

## Scope

- Target set: `recipes/dnr/models/*sfc*/config.yaml` (22 variants)
- Export constraint: ONNX opset `11~14`, `dynamo=False`
- ONE flow tested: `one-import-onnx -> one-optimize -> one-quantize`

## Corrected Result

- ONNX export succeeded for all 22 variants.
- `model.circle` generation failed for all 22 variants.
- Therefore overall compilation status is `FAIL` for all 22 (blocked at import).

## Failure Classes

### Class A (19 models): Conv legalization failure

- Error signature:
  - `loc("/compressor/dw/conv/Conv"): error: failed to legalize operation 'onnx.Conv'`
  - or same pattern on `/pre_compressor/dw/conv/Conv`, `/separator.0/dw/conv/Conv`
- ONNX pattern at failing node (representative):
  - `Conv(group=96, kernel_shape=[3,3], pads=[0,1,0,1])`
  - input is from an explicit `Pad` node with runtime-computed `pads` tensor
  - weight shape: `[96,1,3,3]` (depthwise conv)
- Interpretation:
  - The importer fails on this depthwise-conv + dynamic-pad legalization pattern.
  - This is consistent across most SFC variants.

### Class B (3 models): invalid tensor dimension size

- Error signature:
  - `error: invalid tensor dimension size`
  - followed by `onnx2circle ... StorageUniquerSupport.h ... RankedTensorType ... verifyInvariants ... failed`
- Additional signal from exported ONNX:
  - shape-driving constants in the same conv/pad subgraph include very large sentinel values like `-9223372036854775807` in `Slice` paths.
- Interpretation:
  - `onnx2circle` shape/rank materialization fails for this dynamic shape construction path.

## Quick Experiments

- `frames=4` export for representative Class A model:
  - still fails with `failed to legalize operation 'onnx.Conv'`.
- `opset=11` export for representative Class B model:
  - still fails with `invalid tensor dimension size`.

These confirm the blocker is not solved by changing frame count or lowering opset within supported range.

## Additional Fast-Track Experiment (2026-05-16)

### A) `group=96 -> group=1` test

- Manually changing only `Conv.group` (and also testing with expanded conv weights) removes the first Conv error,
  but import still fails at:
  - `loc("/separator.0/Slice"): error: failed to legalize operation 'onnx.Slice'`
- Conclusion: changing `group` alone is not a sufficient fix.

### B) ONNX simplification before import (`onnxsim`)

- Applied `onnxsim` with fixed input shape (`x=[1,2,1,1025]`) before `one-import-onnx`.
- Batch result over 22 exported ONNX models:
  - PASS: 6
  - FAIL: 16
- Report:
  - `logs/dnr_sfc_npu_verify/summary_import_sim.md`

Failure signatures after simplification:

1. `Circle.minimum ... got 'none'` around `Clip/min` (dominant group)
2. `Circle.add ... broadcast-compatible shapes` for some `fp512keep475` variants

This indicates simplification significantly improves importer compatibility, but additional model/export normalization is still required for full coverage.

## Recommended Fix Direction (Model/Export Side)

1. Remove dynamic `Pad`-driven depthwise conv patterns in the SFC blocks.
   - Prefer static padding in `Conv2d` where possible.
   - Avoid building pad tensors from shape ops (`Shape/Slice/Transpose/Reshape/Cast -> Pad`).

2. Avoid shape-sentinel slicing patterns that generate huge negative constants.
   - Replace dynamic slice-end logic with explicit/static reshape/slice sizes when export shape is fixed.

3. Keep tensors rank <= 4 and avoid implicit 1D emulation patterns that rely on complex dynamic reshaping around conv.
   - Aligns with `AGENT.md` constraints for NPU compatibility.

4. Validate importer early for each model class.
   - First gate: `one-import-onnx` only.
   - Only proceed to optimize/quantize after `.circle` is generated.

## Suggested Next Execution Plan

1. Pick 1 representative model from Class A and Class B.
2. Refactor the depthwise/padding block in model code to emit importer-friendly ONNX.
3. Re-export ONNX and re-test `one-import-onnx` only.
4. Once import passes, re-run full `onecc` (optimize + quantize) and propagate patch to sibling variants.
