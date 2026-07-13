# ONE Mixed Precision Selection Operation

Date: 2026-07-12
Updated: 2026-07-13

## Goal

Choose a small set of quality-critical Circle layers to promote from uint8 to int16 while keeping the rest of the model uint8.

This workflow does not modify ONE compiler source. It generates stock `one-quantize --quant_config` JSON files that can be tested with existing ONE calibration and quantization tools.

## Source Basis

The selector follows the behavior in the live ONE source:

- `compiler/circle-quantizer/src/CircleQuantizer.cpp` parses `quant_config` layers by exact node/tensor `name` or `names`.
- `compiler/luci/pass/src/QuantizeWithMinMaxPass.cpp` chooses per-node dtype from `layers_info`; nodes not in the config stay at the default output dtype.
- `compiler/luci/pass/src/InsertQuantizeOpOnDTypeMismatch.cpp` has explicit mixed-dtype boundary handling for `Transpose`, `FullyConnected`, `Mul`, and `BatchMatMul`. Scattered int16 nodes around other ops can add extra Quantize boundaries, so small contiguous islands are safer.
- `compiler/circle-mpqsolver/src/bisection/BisectionSolver.cpp` uses depth-front/depth-back splits for AMPQ, so this tool also emits depth-style candidate configs.
- `compiler/circle-mpqsolver/src/core/Dumper.cpp` writes stock qconfig files with `default_quantization_dtype`, `default_granularity`, `model_path`, and `layers`.

## Tool

Use:

```bash
python /home/cmj/works/ASS/tools/online/suggest_one_mixed_precision_qconfig.py \
  --circle /path/to/model.opt.circle \
  --out-dir /home/cmj/works/ASS/logs/one_mixed_precision_suggestions/your_model
```

Outputs:

- `nodes.csv`: ranked node list with op type, depth, rough compute cost, boundary risk, optional error score, qconfig eligibility, and reasons.
- `summary.md`: human-readable top candidates and generated qconfig list.
- `summary.json`: machine-readable metadata.
- `qconfig_*.json`: stock ONE mixed-precision configs.

By default the selector hard-excludes memory/layout ops from generated qconfigs:

```text
CONCATENATION, EXPAND_DIMS, GATHER, PACK, PAD, PADV2, RESHAPE, SLICE,
SPLIT, SPLIT_V, SQUEEZE, STRIDED_SLICE, TILE, TRANSPOSE, UNPACK
```

Those ops still appear in `nodes.csv` with `eligible=0`, but they are not promoted to int16. Override this only for diagnostics:

```bash
--exclude-op ''
```

You can also hard-exclude node names:

```bash
--exclude-regex 'post_tr|pads'
```

## Recommended First Run

For an audio separation model, start with the built-in name hints and prefer final masks/heads:

```bash
python /home/cmj/works/ASS/tools/online/suggest_one_mixed_precision_qconfig.py \
  --circle logs/one/model.opt.circle \
  --out-dir logs/one_mixed_precision_suggestions/model \
  --top-k 20 \
  --island-sizes 3,5,8 \
  --depth-fractions 0.25,0.5 \
  --prefer-regex 'mask|head|decoder|sfc|attn|softmax|out'
```

Test in this order:

1. `qconfig_best_island3_int16.json`
2. `qconfig_best_island5_int16.json`
3. `qconfig_best_island8_int16.json`
4. `qconfig_depth_back_10_int16.json`, then 15/20/25 if small islands do not help.
5. `qconfig_top*_int16.json` only as a diagnostic, because it can scatter int16 nodes.

## Test A Proposal With Stock ONE

With an HDF5 calibration file:

```bash
python /home/cmj/works/ASS/tools/online/run_one_stock_quant_sweep.py \
  --input-circle logs/one/model.opt.circle \
  --calib-data data/calibration.h5 \
  --output-dir logs/one_stock_quant_sweep/model_mixed_island3 \
  --modes percentile \
  --min-percentiles 0.1,0.5,1.0 \
  --max-percentiles 99.0,99.5,99.9 \
  --quant-config logs/one_mixed_precision_suggestions/model/qconfig_best_island3_int16.json
```

With a calibration list text file:

```bash
python /home/cmj/works/ASS/tools/online/run_one_stock_quant_sweep.py \
  --input-circle logs/one/model.opt.circle \
  --calib-data data/calibration_files.txt \
  --input-data-format list \
  --output-dir logs/one_stock_quant_sweep/model_mixed_island3 \
  --modes percentile \
  --min-percentiles 0.1,0.5,1.0 \
  --max-percentiles 99.0,99.5,99.9 \
  --quant-config logs/one_mixed_precision_suggestions/model/qconfig_best_island3_int16.json
```

The sweep wrapper only calls stock `one-quantize`; the qconfig is passed through as `--quant_config`.

Do not combine `--quant-config` with `--modes ampq` in this wrapper. Stock `one-quantize --ampq` invokes `circle-mpqsolver`, and the current ONE AMPQ path does not consume an external `quant_config`. Use `percentile` or `moving_average` when testing a fixed qconfig, and run AMPQ separately when you want ONE to search for its own qconfig.

## Optional AMPQ/VISQ Inputs

If stock AMPQ already produced `FinalConfiguration.mpq.json`, feed it back into the selector:

```bash
python /home/cmj/works/ASS/tools/online/suggest_one_mixed_precision_qconfig.py \
  --circle logs/one/model.opt.circle \
  --ampq-config logs/one_stock_quant_sweep/model_ampq/FinalConfiguration.mpq.json \
  --out-dir logs/one_mixed_precision_suggestions/model_ampq_guided
```

If you have a VISQ-style JSON with per-layer error values in the form:

```json
{
  "error": [
    {"layer_a": 0.0123},
    {"layer_b": 0.0456}
  ]
}
```

then add:

```bash
--visq-json path/to/visq_errors.json
```

Those layers get a score boost, but the generated configs remain stock ONE qconfigs.

## Selection Rule

Pick the smallest qconfig that recovers separation quality:

- If pure uint8 percentile tuning is enough, use pure uint8.
- If not, try the smallest contiguous int16 island near final mask/output nodes.
- Promote depth-back regions only when late-stage island configs still fail.
- Avoid full int16 unless the measured latency is acceptable.

For this audio separation target, latency should be checked after each qconfig because int16 islands can introduce dtype conversion boundaries. The selector estimates boundary risk, but the final decision must come from actual separation quality and runtime measurements.

## SFC Small Conv2D BN NPU Result

For `logs/npu_verify_general/sfc_small_conv2d_bn_npu_quant_20260713/stream_full.opt.circle`, the selector was regenerated with depth fractions `0.10,0.15,0.20,0.25`.

Fast-screen setup:

```bash
.venv/bin/python tools/online/run_one_stock_quant_sweep.py \
  --input-circle logs/npu_verify_general/sfc_small_conv2d_bn_npu_quant_20260713/stream_full.opt.circle \
  --calib-data logs/npu_verify_general/sfc_small_conv2d_bn_npu_quant_20260713/calib.h5 \
  --calib-record-limit 16 \
  --test-data logs/npu_verify_general/sfc_small_conv2d_bn_npu_quant_20260713/calib.h5 \
  --test-record-limit 8 \
  --evaluate-result \
  --print-mse \
  --modes percentile \
  --min-percentiles 0.1 \
  --max-percentiles 99.9 \
  --quant-config <candidate-qconfig>
```

Fast-screen primary-output MSE:

| qconfig | int16 layers | mse_primary |
|---|---:|---:|
| pure uint8 percentile 0.1/99.9 | 0 | 0.000383961 |
| best_island3 | 3 | 0.000385010 |
| best_island5 | 5 | 0.000384697 |
| best_island8 | 8 | 0.000385221 |
| top20 | 20 | 0.000385513 |
| depth_back_10 | 22 | 0.000370437 |
| depth_back_15 | 27 | 0.000370476 |
| depth_back_20 | 33 | 0.000370435 |
| depth_back_25 | 37 | 0.000370594 |

`depth_back_20` was numerically lowest in the fast screen by only `2e-9` MSE versus `depth_back_10`, so `depth_back_10` was selected as the NPU-friendly choice with fewer int16 layers.

Full 64-record calibration/evaluation:

| artifact | mse_primary | mse_mean |
|---|---:|---:|
| pure uint8 full eval | 0.000381631 | 0.0000652770711111111 |
| mixed depth_back_10 full eval | 0.000370690 | 0.0000640614044444444 |

Selected artifacts:

```text
logs/npu_verify_general/sfc_small_conv2d_bn_npu_quant_20260713/stream_full.opt.mixed_depth_back10.q.circle
logs/npu_verify_general/sfc_small_conv2d_bn_npu_quant_20260713/stream_full.opt.mixed_depth_back10.qconfig.json
logs/npu_verify_general/sfc_small_conv2d_bn_npu_quant_20260713/stream_full.opt.mixed_depth_back10.summary.json
```

Tensor dtype sanity check:

```text
mixed depth_back_10: INT16 50, INT32 78, INT64 2, UINT8 233
pure uint8:          INT32 80, UINT8 249
```
