# ONE Compilation Worklog (online_soft_band_query_sfc_2d)

This document records the full troubleshooting and execution history for compiling:

- Python model source: `/home/cmj/works/ASS/spectral_feature_compression/core/model/online_soft_band_query_sfc_2d.py`
- Main recipe/config used: `/home/cmj/works/ASS/recipes/musdb18hq/models/online-soft-band-query-sfc2d.causal96dim.12l/config.yaml`
- ONE guide followed: `/home/cmj/works/ONE/build/compiler/one-cmds/GUIDE.md`

---

## 1) Objective

Goal was to compile the target model with ONE tools (ONNX -> Circle -> Optimize -> Quantize -> Codegen).

---

## 2) Environment and Initial Constraints

- Workspace roots:
  - `/home/cmj/works/ASS`
  - `/home/cmj/works/ONE`
- ONE command tools path:
  - `/home/cmj/works/ONE/build/compiler/one-cmds`
- Prebuilt ONE venv used per guide:
  - `/home/cmj/works/ONE/build/compiler/one-cmds/venv`
- Export scripts and model code are in `ASS`.

Important discovered constraints during work:

1. `one-import-onnx` initially failed because `onnx2circle` binary was missing.
2. System did not have host `cmake` in PATH.
3. `ONE/circle-mlir` was not writable from current execution context (could not create `Makefile` symlink there).
4. Export path needed dependencies and compatibility adjustments (`onnxscript`, numba cache workaround, model op compatibility).

---

## 3) Config File Used

Config file:

- `/home/cmj/works/ASS/recipes/musdb18hq/models/online-soft-band-query-sfc2d.causal96dim.12l/config.yaml`

Key values from that config (used by exporter/builder):

- `sr: 44100`
- `n_fft: 2048`
- `hop_length: 512`
- `n_src: 4`
- `n_chan: 2`
- `online_d_model: 96`
- `online_n_layers: 12`
- `online_n_bands: 64`
- `online_causal: true`
- `online_masking: true`
- `task.model._target_`: `spectral_feature_compression.core.model.online_soft_band_query_sfc_2d.build_online_soft_band_query_sfc_system`

---

## 4) Main Output Directories

Primary output directories created/used:

- `/home/cmj/works/ASS/logs/one_compile_soft_band_query`
- `/home/cmj/works/ASS/logs/one_compile_soft_band_query_legacy`
- Temporary circle-mlir build mirror:
  - `/home/cmj/works/ASS/tmp/circle-mlir`

---

## 5) Detailed Command Sequence and Results

Below is the chronological sequence of key commands and outcomes.

### Step 0 - Read and inspect guide/model/setup

Commands (representative):

```bash
ReadFile /home/cmj/works/ONE/build/compiler/one-cmds/GUIDE.md
ReadFile /home/cmj/works/ASS/spectral_feature_compression/core/model/online_soft_band_query_sfc_2d.py
```

Result:

- Confirmed expected ONE flow:
  1. Export ONNX
  2. `one-import-onnx`
  3. `one-optimize`
  4. `one-quantize`
  5. `one-codegen`

---

### Step 1 - First ONNX export attempts

Command:

```bash
"/home/cmj/works/ASS/.venv/bin/python" \
"/home/cmj/works/ASS/tools/online/export_onnx_online_model.py" \
"/home/cmj/works/ASS/recipes/musdb18hq/models/online-soft-band-query-sfc2d.causal96dim.12l/config.yaml" \
--out "/home/cmj/works/ASS/logs/one_compile_soft_band_query/model.onnx" \
--n-chan 2 --frames 1 --opset 11 --streaming --check \
--state-meta-out "/home/cmj/works/ASS/logs/one_compile_soft_band_query/state_meta.json" \
--deploy-manifest-out "/home/cmj/works/ASS/logs/one_compile_soft_band_query/deploy_manifest.json"
```

First failures and fixes:

1. Import-time numba/librosa cache error:
   - `RuntimeError: cannot cache function '__o_fold' ...`
   - Workaround used:
     - `NUMBA_DISABLE_JIT=1`
     - `NUMBA_CACHE_DIR=/home/cmj/works/ASS/logs/.numba_cache`

2. Missing package:
   - `ModuleNotFoundError: No module named 'onnxscript'`
   - Installed:

```bash
"/home/cmj/works/ASS/.venv/bin/python" -m pip install onnxscript
```

After fixes, export succeeded and generated:

- `model.onnx`
- `model.onnx.data`
- `state_meta.json`
- `deploy_manifest.json`

---

### Step 2 - First ONE import failed due to missing `onnx2circle`

Command:

```bash
export ONE_CMDS="/home/cmj/works/ONE/build/compiler/one-cmds"
source "$ONE_CMDS/venv/bin/activate"
export PATH="$ONE_CMDS:$PATH"
one-import-onnx \
  -i "/home/cmj/works/ASS/logs/one_compile_soft_band_query/model.onnx" \
  -o "/home/cmj/works/ASS/logs/one_compile_soft_band_query/model.circle" \
  --dynamic_batch_to_single_batch
```

Error:

- `one-import-onnx: onnx2circle converter not found`

Root cause:

- `one-import-onnx` expects `onnx2circle` at:
  - `.../one-cmds/onnx2circle`, or
  - `/usr/share/circletools/bin/onnx2circle`
- Neither existed.

---

### Step 3 - Build and link `onnx2circle` (blocking issue fix)

Attempts made:

1. Build inside `ONE/circle-mlir` directly using Docker image:
   - blocked by write permission (could not create `Makefile` symlink).
2. Created writable copy:

```bash
cp -r "/home/cmj/works/ONE/circle-mlir" "/home/cmj/works/ASS/tmp/circle-mlir"
```

3. Added required `res` path for schema dependency in copied tree:

```bash
ln -s "/work/ONE/res" "/home/cmj/works/ASS/tmp/res"
```

4. Build command that succeeded:

```bash
docker run --rm \
  -v "/home/cmj/works:/work" \
  -w "/work/ASS/tmp/circle-mlir" \
  nnfw/circle-mlir-build:jammy \
  bash -lc "cmake --build build/release -j4 --target onnx2circle"
```

Produced binary:

- `/home/cmj/works/ASS/tmp/circle-mlir/build/release/circle-mlir/tools/onnx2circle/onnx2circle`

Linked into one-cmds:

```bash
ln -sfn \
"/home/cmj/works/ASS/tmp/circle-mlir/build/release/circle-mlir/tools/onnx2circle/onnx2circle" \
"/home/cmj/works/ONE/build/compiler/one-cmds/onnx2circle"
```

Result:

- Original `onnx2circle not found` blocking issue resolved.

---

### Step 4 - Post-fix import behavior and compatibility work

After `onnx2circle` existed, import progressed but failed on model/graph compatibility:

- Streaming export import failures:
  - `failed to legalize operation 'onnx.Reshape'`
  - later `Circle.minimum ... got 'none'` (from Clip lowering)
  - later dynamic reshape issue (`multiple dynamic dimensions`)

Actions taken:

1. Generated alternate ONNX via legacy exporter path (`measure_npu_model_stats.py`) to test import behavior.
2. Patched model to avoid `clamp_min` lowering to ONNX Clip.

Code change applied:

- File: `/home/cmj/works/ASS/spectral_feature_compression/core/model/online_soft_band_query_sfc_2d.py`
- Change:
  - from:
    - `coeff = coeff / coeff.sum(dim=1, keepdim=True).clamp_min(1e-6)`
  - to:
    - `coeff = coeff / (coeff.sum(dim=1, keepdim=True) + 1e-6)`

3. Exported non-streaming ONNX with new exporter and `--disable-masking`:

```bash
NUMBA_CACHE_DIR="/home/cmj/works/ASS/logs/.numba_cache" \
NUMBA_DISABLE_JIT=1 \
"/home/cmj/works/ASS/.venv/bin/python" \
"/home/cmj/works/ASS/tools/online/export_onnx_online_model.py" \
"/home/cmj/works/ASS/recipes/musdb18hq/models/online-soft-band-query-sfc2d.causal96dim.12l/config.yaml" \
--out "/home/cmj/works/ASS/logs/one_compile_soft_band_query/nonstream_model.onnx" \
--n-chan 2 --frames 1 --opset 18 --check --disable-masking \
--deploy-manifest-out "/home/cmj/works/ASS/logs/one_compile_soft_band_query/nonstream_manifest.json"
```

Then import succeeded:

```bash
one-import-onnx \
  -i "/home/cmj/works/ASS/logs/one_compile_soft_band_query/nonstream_model.onnx" \
  -o "/home/cmj/works/ASS/logs/one_compile_soft_band_query/nonstream_model.circle" \
  --dynamic_batch_to_single_batch
```

Result:

- ONNX -> Circle conversion successful (non-streaming path).

---

### Step 5 - `one-optimize` / `one-quantize` attempts and new blocker

Next command attempted (guide-aligned):

```bash
one-optimize \
  -i "/home/cmj/works/ASS/logs/one_compile_soft_band_query/nonstream_model.circle" \
  -o "/home/cmj/works/ASS/logs/one_compile_soft_band_query/nonstream_model.opt.circle" \
  --fold_cast --fold_dequantize --fuse_batchnorm_with_conv \
  --fuse_activation_function --remove_redundant_reshape --remove_unnecessary_transpose
```

Additional runtime setup performed:

- `one-cmds` expected local binaries (`circle2circle`, `circle-quantizer`, `record-minmax`, etc.) that were absent in `one-cmds` dir.
- Symlinked from existing ONE build outputs under `/home/cmj/works/ONE/build/compiler/*`.
- Added dynamic library search paths using `LD_LIBRARY_PATH` built from all `.so` dirs in `/home/cmj/works/ONE/build/compiler`.

After that, optimizer/quantizer execution reached compiler internals but failed with shape inference assertion:

- `Only support int 32`
- `CircleShapeInferenceRule.cpp: infer_reducer ... reduction_indices->dtype() == S32`

This occurred in:

- `one-optimize` (through `circle2circle`)
- `one-quantize` (through `circle-quantizer`)

Calibration dataset creation command (succeeded):

```bash
"/home/cmj/works/ASS/.venv/bin/python" -c \
"import numpy as np; np.save('/home/cmj/works/ASS/logs/one_compile_soft_band_query/calib_000.npy', np.random.randn(1,4,1,1025).astype('float32'))"

printf '%s\n' '/home/cmj/works/ASS/logs/one_compile_soft_band_query/calib_000.npy' \
> "/home/cmj/works/ASS/logs/one_compile_soft_band_query/calib_list.txt"

one-create-quant-dataset \
  -i numpy \
  -l "/home/cmj/works/ASS/logs/one_compile_soft_band_query/calib_list.txt" \
  -p "/home/cmj/works/ASS/logs/one_compile_soft_band_query/calib.h5"
```

---

## 6) Files Created / Modified

### Created artifacts

- `/home/cmj/works/ASS/logs/one_compile_soft_band_query/model.onnx`
- `/home/cmj/works/ASS/logs/one_compile_soft_band_query/model.onnx.data`
- `/home/cmj/works/ASS/logs/one_compile_soft_band_query/state_meta.json`
- `/home/cmj/works/ASS/logs/one_compile_soft_band_query/deploy_manifest.json`
- `/home/cmj/works/ASS/logs/one_compile_soft_band_query/nonstream_model.onnx`
- `/home/cmj/works/ASS/logs/one_compile_soft_band_query/nonstream_manifest.json`
- `/home/cmj/works/ASS/logs/one_compile_soft_band_query/nonstream_model.circle`
- `/home/cmj/works/ASS/logs/one_compile_soft_band_query/calib_000.npy`
- `/home/cmj/works/ASS/logs/one_compile_soft_band_query/calib_list.txt`
- `/home/cmj/works/ASS/logs/one_compile_soft_band_query/calib.h5`
- `/home/cmj/works/ASS/logs/one_compile_soft_band_query_legacy/soft_query_legacy_export.onnx`
- `/home/cmj/works/ASS/logs/one_compile_soft_band_query_legacy/npu_model_stats.json`
- `/home/cmj/works/ASS/logs/one_compile_soft_band_query_legacy/npu_model_stats.csv`

### Built and linked tooling

- Built:
  - `/home/cmj/works/ASS/tmp/circle-mlir/build/release/circle-mlir/tools/onnx2circle/onnx2circle`
- Linked into one-cmds:
  - `/home/cmj/works/ONE/build/compiler/one-cmds/onnx2circle`
  - `/home/cmj/works/ONE/build/compiler/one-cmds/circle2circle`
  - `/home/cmj/works/ONE/build/compiler/one-cmds/circle-quantizer`
  - `/home/cmj/works/ONE/build/compiler/one-cmds/record-minmax`
  - `/home/cmj/works/ONE/build/compiler/one-cmds/circle-eval-diff`
  - `/home/cmj/works/ONE/build/compiler/one-cmds/circle-mpqsolver`

### Source code modified

- `/home/cmj/works/ASS/spectral_feature_compression/core/model/online_soft_band_query_sfc_2d.py`
  - `clamp_min(1e-6)` replaced with `+ 1e-6` denominator for ONNX conversion compatibility.

---

## 7) Current Status (at time of writing)

1. `onnx2circle` missing-binary blocker: **fixed**.
2. ONNX -> Circle import:
   - streaming export path: still problematic
   - non-streaming static export: **works**
3. Optimize/quantize stage: **blocked** by reducer-index int32 requirement in current ONE compiler flow.
4. Codegen stage: **not reached** because quantized circle was not produced.

---

## 8) Suggested Next Technical Step

Address reducer index dtype compatibility before optimize/quantize:

- add ONNX post-process pass (or export-side fix) to force reduction indices/axes tensors to int32 in a ONE-compatible form;
- then rerun:
  1. `one-import-onnx`
  2. `one-optimize`
  3. `one-quantize`
  4. `one-codegen`

