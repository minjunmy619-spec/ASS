# ONE Mixed-Precision Quantization: Source Findings and Search Workflow

This document records the stock ONE behavior used by the ASS-side tools. The
ONE compiler checkout is not modified.

## What ONE Actually Quantizes

The relevant stock path is:

1. `record-minmax` runs the Circle interpreter and records one min/max pair for
   each floating-point activation tensor. It skips constants, boolean tensors,
   and integer tensors. It supports H5, directory, random, and list input
   iterators. A list line contains one raw file per model input, separated by
   spaces.
2. `one-quantize` invokes `record-minmax`, then `circle-quantizer` with the
   selected calibration mode. The normal modes are `percentile` and
   `moving_average`.
3. `circle-quantizer` reads qconfig layer entries by Circle node name. A layer
   entry may use either `name` or `names`, and specifies a dtype and
   granularity. The default dtype applies to every layer not listed.
4. `QuantizeWithMinMaxPass` creates activation quantization parameters from the
   recorded min/max pair. Its `InsertQuantizeOp` visitor adds conversion
   operators around a layer whose configured dtype differs from the default,
   followed by `RemoveRedundantQuantizePass`.

The practical consequence is that a PyTorch module name is not the final qconfig
key. The qconfig key must be the optimized Circle node name, usually the output
tensor name after ONE import and optimization.

## ONE Quantization Math

For ordinary activations, ONE uses asymmetric uint8 and symmetric int16:

| dtype | quantized range | scale | zero point |
|---|---:|---|---|
| uint8 | `[0, 255]` | `(max(0, max) - min(0, min)) / 255`, with ONE's positive-only and negative-only cases | rounded zero location, or `0`/`255` for one-sided ranges |
| int16 | `[-32767, 32767]` | `max(abs(min), abs(max)) / 32767` with a minimum scale of `1e-8` | `0` |

ONE nudges the uint8 range to `(0 - zero_point) * scale` through
`(255 - zero_point) * scale`. The int16 scale is constructed from the symmetric
`[-32767, 32767]` range, while the runtime integer type can represent and clamp
to `[-32768, 32767]`.

Some operators do not use their observed min/max directly:

- Logistic: uint8 `(scale=1/256, zero_point=0)` and int16 `scale=1/32768`.
- Tanh: uint8 `(scale=2/256, zero_point=128)` and int16 `scale=1/32768`.
- Softmax: uint8 `scale=1/255` and int16 `scale=1/32767`.
- Floor, ceil, and related integer-output operations use an integer scale.
- Reshape, transpose, split, quantize, and similar propagation operations can
  inherit the input activation qtype.

This is why the analyzer does not reimplement these rules. It invokes stock
`circle-quantizer` once for full uint8 and once for full int16, then reads each
tensor's actual scale and zero point from those generated Circle models.

## Why the PyTorch-Hook-Only Idea Is Incomplete

Forward hooks are useful for debugging a PyTorch module, but they miss
functional operations such as `add`, `reshape`, and `softmax`, and they do not
see the nodes that ONE creates or fuses during import and optimization. A hook
name also needs a second mapping step to become a Circle node name.

The calibration analyzer therefore uses the exported ONNX graph as the common
pre-Circle representation:

```text
ONNX Runtime intermediate output
        |  name matching / fused-node aliases
Circle node name used in qconfig
        |  actual qparams from stock uint8/int16 Circle models
uint8/int16 local reconstruction error
```

This still has a deliberate approximation: ONNX Runtime values can differ from
the optimized Circle interpreter because of layout changes, fusion, and
constant folding. The stock `record-minmax` result supplies the exact Circle
calibration range, while ONNX Runtime supplies a bounded sample of activation
values for ranking.

## Tools

### 1. Local activation-error analyzer

`tools/online/analyze_one_mixed_precision_calibration.py`:

- invokes the stock `record-minmax` binary on the Circle model;
- accepts the same raw list-file format used by ONE;
- invokes stock `circle-quantizer` to produce reference full-uint8 and
  full-int16 models and reads qparams from them;
- exposes only mapped ONNX intermediate outputs, in bounded batches;
- uses deterministic reservoir sampling across all selected calibration
  records instead of retaining only the first tensor values;
- maps Circle names to ONNX node/output names, including common `/pre_tr`,
  `/post_tr`, and fused `name1;name2` aliases;
- computes clipping, MSE, and MSE reduction with the stock qparams;
- writes `nodes.csv` and a calibration-ranked qconfig.

Example:

```bash
python /home/cmj/works/ASS/tools/online/analyze_one_mixed_precision_calibration.py \
  --circle logs/one/model.opt.circle \
  --onnx logs/one/model.onnx \
  --calib-data logs/one/calibration.list.txt \
  --out-dir logs/one_mixed_precision_calibration/model \
  --max-samples 64 \
  --max-values-per-node 20000 \
  --onnx-output-batch-size 16 \
  --sampling-seed 0 \
  --top-k 24
```

Use `--recorded-circle` when a recorded model already exists. Use
`--include-op TRANSPOSE` only when there is evidence that the layout boundary
itself is worth keeping int16; memory-only nodes are excluded by default.

The local ranking is a prefilter. It is not the final quality metric.

### 2. Closed-loop stock ONE search

`tools/online/search_one_mixed_precision_qconfig.py` takes the analyzer's
`nodes.csv` and greedily evaluates candidates with the real stock toolchain:

```bash
python /home/cmj/works/ASS/tools/online/search_one_mixed_precision_qconfig.py \
  --circle logs/one/model.opt.circle \
  --calib-data logs/one/calibration.list.txt \
  --test-data logs/one/test.list.txt \
  --candidate-csv logs/one_mixed_precision_calibration/model/nodes.csv \
  --out-dir logs/one_mixed_precision_search/model \
  --max-candidates 24 \
  --max-int16 8 \
  --min-percentile 1.0 \
  --max-percentile 99.0
```

For list inputs it calls stock `one-create-quant-dataset` first, because this
ONE revision's `circle-eval-diff` does not accept list input for final MSE
evaluation. The search then runs:

```text
list -> stock one-create-quant-dataset -> H5
H5 + qconfig -> stock one-quantize
quantized Circle + H5 -> stock circle-eval-diff
```

At each round it tests every remaining candidate and adds a node only when the
selected output MSE improves by more than the configured absolute or relative
threshold. The default objective is `mse_primary`, the first separation output,
so recurrent state outputs do not dilute the quality signal. Use
`--objective mean` to reproduce all-output averaging, or
`--objective output --objective-output NAME` to target an exact output.

The candidate loader accepts both the analyzer's `true`/`false` eligibility and
the older selector's `1`/`0` format. It resolves every candidate name and index
against the current Circle graph before running a trial, preventing stale names
or missing CSV indices from producing invalid or colliding qconfigs.

The optional penalties are deliberately separate from quality:

- `--latency-weight` penalizes selected rough compute divided by the
  whole-model rough compute estimate.
- `--conversion-weight` penalizes the actual extra `QUANTIZE` and `DEQUANTIZE`
  operators reported by stock `circle-inspect`. `--boundary-weight` remains as
  a compatibility alias.

Start with both weights at zero to establish the quality curve. Then rerun with
a small positive weight and measure device latency. The rough compute estimate
and conversion count are useful proxies, not a hardware latency model.

## Recommended Selection Procedure

1. Run the analyzer with at least 32 representative records. The calibration
   data should include silence, low-level signals, and the loudest expected
   separation workload.
2. Keep the top 16-32 analyzer candidates. Do not immediately put every high
   local-error node into int16.
3. Run the closed-loop search with `--max-int16 1`, then 2, 4, 6, and 8. Plot
   final output MSE against the number of selected nodes.
4. Repeat the winning small configurations on a held-out test list.
5. Inspect the final Circle for `QUANTIZE`/`DEQUANTIZE` boundaries and measure
   latency. Prefer a contiguous int16 island when its final MSE is close to an
   isolated-node winner.

The stock AMPQ solver is a useful independent baseline, but it is not the
search backend for a fixed qconfig in this checkout. `--ampq` enters
`circle-mpqsolver`, evaluates full-model output MAE, and searches front/back
depth partitions. It does not consume an external `--quant_config`. Run it
separately and compare its qerror and latency with the fixed-qconfig search.

## Interpreting a Surprising Result

If a node has large local uint8 MSE but int16 makes final output MSE worse, the
node is probably not output-sensitive, or the dtype boundary introduced around
it dominates the benefit. If a node has small local MSE but improves final MSE,
it is likely on a high-sensitivity path, such as a recurrent state, mask head,
softmax input, or final projection. The closed-loop search is designed to find
these cases without changing ONE source.

The tools are therefore intentionally two-stage:

```text
local ONE-compatible error -> reduce expensive trials
stock final-output MSE     -> decide the actual qconfig
```

## Validation

Focused regression tests cover complete-dataset reservoir sampling, native
int16 reconstruction limits, legacy candidate CSV compatibility, stale Circle
names, primary-output objective selection, and no-gain stopping:

```bash
/home/cmj/works/ASS/.venv/bin/python -m pytest -q \
  tests/test_one_mixed_precision_tools.py
```

Lint the tool set with:

```bash
/home/cmj/works/ASS/.venv/bin/ruff check \
  tools/online/analyze_one_mixed_precision_calibration.py \
  tools/online/search_one_mixed_precision_qconfig.py \
  tools/online/run_one_stock_quant_sweep.py \
  tests/test_one_mixed_precision_tools.py
```
