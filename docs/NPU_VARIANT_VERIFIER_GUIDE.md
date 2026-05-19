# NPU Variant Verifier Guide

This guide describes the general verifier script:

- `tools/online/verify_npu_variants.py`

It runs the full PyTorch/ONNX-to-ONE flow for multiple variants and reports artifact-truth pass/fail results.

---

## 1) What It Verifies

For each variant, the verifier runs:

1. ONNX export
2. ONNX simplification (`onnxsim`)
3. Optional ONNX `Clip -> Min/Max` rewrite
4. Calibration dataset generation from ONNX runtime inputs
5. `onecc` full pipeline:
   - `one-import-onnx`
   - `one-optimize`
   - `one-quantize`
6. Artifact checks:
   - `model.circle`
   - `model.opt.circle`
   - `model.q.circle`

A variant is only `PASS` if all required artifacts exist.

---

## 2) Supported Variant Sources

The script supports two variant families:

- `recipe`: discovered from folders under `recipes/dnr/models/*/config.yaml`
- `tf`: built-in TF-MLPNet model variants (`v2` and `v1` presets)

Mode selection:

- `--mode recipe`
- `--mode tf`
- `--mode all`

---

## 3) Environment Requirements

Expected paths:

- ASS root: `/home/cmj/works/ASS`
- ONE commands: `/home/cmj/works/ONE/build/compiler/one-cmds`
- Python: `/home/cmj/works/ASS/.venv/bin/python`

The script auto-configures:

- `PATH` to include ONE commands
- `LD_LIBRARY_PATH` from ONE build library folders
- `PYTHONPATH` for ASS + TF-MLPNet
- `NUMBA_DISABLE_JIT=1`
- `NUMBA_CACHE_DIR=/home/cmj/works/ASS/logs/.numba_cache`

---

## 4) Basic Usage

From ASS root:

```bash
./.venv/bin/python tools/online/verify_npu_variants.py --mode all
```

Run only recipe variants:

```bash
./.venv/bin/python tools/online/verify_npu_variants.py --mode recipe
```

Run only TF-MLPNet variants:

```bash
./.venv/bin/python tools/online/verify_npu_variants.py --mode tf
```

---

## 5) Useful Options

Filter recipe names (example: only SFC):

```bash
./.venv/bin/python tools/online/verify_npu_variants.py \
  --mode recipe \
  --recipe-name-contains sfc
```

Custom recipe root:

```bash
./.venv/bin/python tools/online/verify_npu_variants.py \
  --mode recipe \
  --recipe-root /home/cmj/works/ASS/recipes/dnr/models
```

Dry-run discovery only:

```bash
./.venv/bin/python tools/online/verify_npu_variants.py --mode all --dry-run
```

Limit number of variants:

```bash
./.venv/bin/python tools/online/verify_npu_variants.py --mode all --limit 5
```

Temporary quantization unblock (layer granularity retry):

```bash
./.venv/bin/python tools/online/verify_npu_variants.py --mode all --quantize-layer-fallback
```

Custom output run folder:

```bash
./.venv/bin/python tools/online/verify_npu_variants.py \
  --mode all \
  --run-name full_regression_20260516
```

---

## 6) Output Layout

Results are written under:

- `logs/npu_verify_general/<run-name>/`

Top-level outputs:

- `summary.md`
- `summary.json`

Per-variant folder:

- `run.log`
- `model.onnx`
- `model.sim.onnx`
- `calib_list.txt`
- `calib.h5`
- `config.cfg`
- `model.circle`
- `model.opt.circle`
- `model.q.circle`

---

## 7) ONE Optimize Policy in Verifier

The generated ONE config enables these key passes:

- `replace_non_const_fc_with_batch_matmul=True`

  Set `replace_non_const_fc_with_batch_matmul=True` to fix "Unsupported non const input ... MatMul/tr" during quantization.

- `convert_nchw_to_nhwc=True`

This matches the previously validated workaround set for quantization compatibility.


Quantization:

- default: `granularity=channel` (preferred for accuracy)

Optional escape hatch:

- `--quantize-layer-fallback`: if quantization fails with
  `Non-channel dimension of const node must be 1`, retry once with
  `granularity=layer`. Layer-wise quantization often hurts accuracy versus channel-wise,
  so this flag is **off by default**.

Typical trigger (example): PyTorch-exported **PReLU** slopes shaped like `[C, 1, 1]` in NCHW.
ONE channel-wise const quantization (`quant_const_per_channel`) currently assumes the channel axis is the **last**
dimension for those tensors, which rejects valid `[C, 1, 1]` layouts. **Do not “fix” this by blindly reshaping slopes**
to `[1, 1, C]` in ONNX—that changes broadcast semantics relative to NCHW activations.

Proper fixes are upstream:

- extend ONE to quantize those constants along the graph’s channel axis (or handle **CirclePRelu** alpha explicitly), or
- change the source model/export (e.g. different activation) only after validating equivalence.

Until then, use `--quantize-layer-fallback` only as a temporary unblock for blocked models.

---

## 8) Troubleshooting

### A) Export failure

Symptoms:

- ONNX export command fails
- missing `model.onnx`

Checks:

- verify `.venv` dependencies
- check model/config mismatch
- inspect `<variant>/run.log` under `=== EXPORT ===`

Special handling already included:

- recipe export retries with `--n-chan 1` if common channel mismatch is detected

### B) ONNX simplification failure (`onnxsim`)

Symptoms:

- fail stage `onnxsim`

Checks:

- verify ONNX model loads correctly
- inspect simplifier section in `run.log`

Built-in fallback:

- verifier first tries `--overwrite-input-shape`
- if it fails, retries onnxsim without overwrite

### C) Quantization calibration failure

Symptoms:

- fail stage `calibration`

Checks:

- `calib_list.txt` and generated `*.npy` files
- ONNX input signatures in export artifacts

Note:

- calibration inputs are auto-generated from ONNX runtime inputs (non-initializer graph inputs)

### D) ONE compile failure (`import`/`optimize`/`quantize`)

Symptoms:

- fail stage reported from ONE logs

Checks:

- inspect `=== ONECC ===` in `run.log`
- verify shared libs in ONE build tree
- verify `model.circle`, `model.opt.circle`, `model.q.circle` existence

If ONE stops during quantization with:

`Non-channel dimension of const node must be 1`

then channel-wise quantization hit a constant layout ONE does not accept yet (often **PReLU** slopes).
Prefer fixing ONE or the model as described in **Quantization** above. Only use `--quantize-layer-fallback` if you explicitly accept the accuracy trade-off.

---

## 9) Recommended Verification Routine

1. Discovery check:
   - run `--dry-run`
2. Small smoke:
   - run `--limit 3`
3. Full run:
   - run without limit
4. Review:
   - `summary.md` for quick status
   - failed variant `run.log` for root cause

---

## 10) Notes

- Current ONNX export target is fixed to `opset=14`, `dynamo=False`.
- Quantization uses `uint8` and channel granularity by default.
- Script is designed for repeatable regression checks across variant families.
