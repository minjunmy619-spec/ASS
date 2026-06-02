# Adaptive Mel Locoformer Quant Positive Operation

Date: 2026-06-01

## Scope

Investigate the reported adaptive-mel Locoformer ONE quantization message:

```text
The minimum and maximum values are all positive.
```

Target recipe:

```text
recipes/dnr/models/adaptive-mel-locoformer-lite-sfc.rt192k.fp512keep475/config.yaml
```

## Findings

- The exact string comes from ONE `compiler/luci/pass/src/QuantizationUtils.cpp` in `compute_asym_scale_zp`.
- It is emitted when asymmetric uint8 quantization sees a strictly positive min/max range.
- In this local ONE build it is a verbose warning path, not a fatal error path.
- Re-running `onecc` with `LUCI_LOG=3` prints many `all positive` and `all negative` range warnings but still emits `model.q.circle`.
- The emitted quantized Circle artifact passes `circle-verify`.
- A real failure is reproducible if ONNX simplification is not forced for this recipe: the verifier skips simplification because `ConstantOfShape` is present, then `onnx2circle` fails during import with `error: invalid tensor dimension size`.

## Passing Flow

Command:

```bash
./.venv/bin/python tools/online/verify_npu_variants.py \
  --mode recipe \
  --recipe-name-contains adaptive-mel-locoformer-lite-sfc \
  --run-name debug_adaptive_mel_loco_quant_positive_20260601 \
  --force-onnxsim-large-shape-ops
```

Result:

- PASS: `logs/npu_verify_general/debug_adaptive_mel_loco_quant_positive_20260601/summary.md`
- ONE config uses `model.sim.onnx`, `convert_nchw_to_nhwc=True`, `replace_non_const_fc_with_batch_matmul=True`, `quantized_dtype=uint8`, `granularity=channel`, `input_type=uint8`, and `output_type=uint8`.

Additional verbose check:

```bash
PATH="/home/cmj/works/ONE/build/compiler/one-cmds:$PATH" \
LUCI_LOG=3 \
onecc -C logs/npu_verify_general/debug_adaptive_mel_loco_quant_positive_20260601/adaptive-mel-locoformer-lite-sfc.rt192k.fp512keep475/config.cfg

/home/cmj/works/ONE/build/compiler/circle-verify/circle-verify \
  logs/npu_verify_general/debug_adaptive_mel_loco_quant_positive_20260601/adaptive-mel-locoformer-lite-sfc.rt192k.fp512keep475/model.q.circle
```

Result:

- `LUCI_LOG=3` exposes the `all positive` quantization warning.
- `model.q.circle` is generated.
- `circle-verify` reports PASS.

## Reproduced Failure Flow

Command:

```bash
./.venv/bin/python tools/online/verify_npu_variants.py \
  --mode recipe \
  --recipe-name-contains adaptive-mel-locoformer-lite-sfc \
  --run-name debug_adaptive_mel_loco_quant_no_force_20260601
```

Result:

- FAIL: `logs/npu_verify_general/debug_adaptive_mel_loco_quant_no_force_20260601/summary.md`
- Failure stage: import
- ONE return code: `250`
- Log excerpt:

```text
Skipped onnxsim because the graph contains large-shape ops that can make simplification impractically slow: ConstantOfShape
Pass --force-onnxsim-large-shape-ops to run simplification anyway.
...
error: invalid tensor dimension size
onnx2circle: ... RankedTensorType ... Assertion `succeeded(...)' failed.
```

## Operational Rule

For this recipe, do not compile the raw exported ONNX directly. Use the forced simplification flow and compile the generated `model.sim.onnx`.

Treat `The minimum and maximum values are all positive.` as diagnostic noise unless the same log also contains a separate hard failure, nonzero return code, or missing `model.q.circle`.

If a user still sees a quantization-stage failure after forced simplification, collect the exact `onecc` command/config, full `one-quantize` log, calibration H5/list, and ONNX/Circle artifacts before changing the model.
