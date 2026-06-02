# Operation Manual: PyTorch -> ONE -> NPU Compilation

This is a **general, reusable, root-cause-first** operation manual for compiling a PyTorch model to ONE NPU artifacts.

Use this as the standard process for any model.  
Project-specific paths/commands are provided only as **examples**.

---

## 1) Scope and Principles

### Goal

Compile a PyTorch model into ONE artifacts:

- `*.circle` (imported)
- `*.opt.circle` (optimized)
- `*.q.circle` (quantized, optional)
- backend binary (`*.bin`, optional codegen)

### Core principle

When errors happen, do not patch model/config first.  
First identify the real failure domain:

1. environment/runtime linker
2. config schema
3. dataset format/count/type/shape
4. ONNX export/lowering
5. ONE toolchain limitation/bug
6. model graph incompatibility

---

## 2) Environment Setup

Use placeholders:

- `<ONE_CMDS_DIR>`: ONE command dir (contains `onecc`, `one-import-onnx`, etc.)
- `<ONE_BUILD_LIB_ROOT>`: ONE build root containing `.so` libraries
- `<WORK_DIR>`: your project root

### 2.1 Activate ONE command environment

```bash
export ONE_CMDS="<ONE_CMDS_DIR>"
source "$ONE_CMDS/venv/bin/activate"
export PATH="$ONE_CMDS:$PATH"
```

### 2.2 Set runtime shared libraries (critical for locally built ONE)

```bash
LIBROOT="<ONE_BUILD_LIB_ROOT>"
export LD_LIBRARY_PATH="$LIBROOT/luci/import:$LIBROOT/luci/export:$LIBROOT/luci/pass:$LIBROOT/luci/service:$LIBROOT/luci/lang:$LIBROOT/luci/env:$LIBROOT/luci/profile:$LIBROOT/luci/plan:$LIBROOT/luci/log:$LIBROOT/luci/logex:$LIBROOT/luci-compute:$LIBROOT/luci-interpreter/src:$LIBROOT/dio-hdf5:$LIBROOT/loco:$LD_LIBRARY_PATH"
```

### 2.3 Validate setup

```bash
which onecc
which one-import-onnx
ldd "$ONE_CMDS/circle2circle" | rg "not found"
ldd "$ONE_CMDS/record-minmax" | rg "not found"
```

If any `not found` appears, fix `LD_LIBRARY_PATH` before continuing.

---

## 3) Config File Prepare

## 3.1 Source of config keys (authoritative)

Always read these files in your ONE build/version:

- onecc template:
  - `<ONE_CMDS_DIR>/onecc.template.cfg`
- optimize keys:
  - `<ONE_CMDS_DIR>/onelib/constant.py` (`OPTIMIZATION_OPTS`)
- quantize command behavior:
  - `<ONE_CMDS_DIR>/one-quantize`

### 3.2 Minimal generic config template

```ini
[Environment]
ONECC_ENV="ONECC"

[backend]
target=

[onecc]
one-import-onnx=True
one-optimize=True
one-quantize=True
one-codegen=False
one-profile=False
one-infer=False

[one-import-onnx]
input_path=<ABS_PATH_MODEL_ONNX>
output_path=<ABS_PATH_MODEL_CIRCLE>
dynamic_batch_to_single_batch=True

[one-optimize]
input_path=<ABS_PATH_MODEL_CIRCLE>
output_path=<ABS_PATH_MODEL_OPT_CIRCLE>

[one-quantize]
input_path=<ABS_PATH_MODEL_OPT_CIRCLE>
output_path=<ABS_PATH_MODEL_Q_CIRCLE>
input_data=<ABS_PATH_CALIB_H5>
input_data_format=h5
quantized_dtype=uint8
granularity=channel
input_type=uint8
output_type=uint8
```

### 3.3 About `include=O1`

`include=O1` may not exist in all ONE installations/config packs.  
If `onecc` throws `KeyError: 'O1'`, use explicit optimize flags instead of include groups.

---

## 4) Model ONNX Export

### 4.1 Export strategy

Prefer deterministic export:

- fixed input shape for deployment target
- clear opset choice
- explicitly inspect exported opset and operator forms

Hard requirement for this current NPU compilation flow:

- ONNX opset must be in `11~14`
- `torch.onnx.export(..., dynamo=False)` must be used

### 4.2 Generic export command (example pattern)

```bash
<PYTHON> <EXPORT_SCRIPT.py> <MODEL_OR_CONFIG_PATH> \
  --out <ABS_PATH_MODEL_ONNX> \
  --n-chan <N_CHAN> \
  --frames <T> \
  --opset <OPSET> \
  --check \
  --deploy-manifest-out <ABS_PATH_MANIFEST_JSON>
```

### 4.3 Opset/exporter guidance

- Current NPU toolchain support is constrained to ONNX opset `11~14`.
- Use `dynamo=False` in `torch.onnx.export` to avoid unsupported exporter/lowering paths.
- If using newer exporter defaults, it may emit higher opset and trigger converter fallbacks/failures.
- Always inspect:
  - final `opset_import`
  - reduction op forms (`axes` attrs vs `axes` inputs)
  - model input count and shapes

### 4.4 Common export failures

1. Missing deps (e.g. `onnxscript`)  
2. numba/librosa cache/jit side effects  
3. exporter down-conversion mismatch (especially around reducer axes semantics)

Use explicit env controls as needed:

- `NUMBA_DISABLE_JIT=1`
- `NUMBA_CACHE_DIR=<writable_cache_dir>`

---

## 5) Step-by-Step Compilation

### 5.1 Full flow

```bash
onecc -C <ABS_PATH_CONFIG_CFG>
```

### 5.2 Run partial flow by toggling `[onecc]`

- Import only: enable only `one-import-onnx`
- Import + optimize: enable `one-import-onnx`, `one-optimize`
- Quantize too: enable `one-quantize`
- Codegen: enable `one-codegen` and fill backend/target fields

### 5.3 Output expectations

- import: `model.circle`
- optimize: `model.opt.circle`
- quantize: `model.q.circle`
- codegen: backend-specific output

### 5.4 Verification rule (artifact-truth, not return-code only)

Treat stage success as valid only when both are true:

1. command return code is success
2. expected stage artifact exists on disk

Required checks:

- import success: `model.circle` exists
- optimize success: `model.opt.circle` exists
- quantize success: `model.q.circle` exists

Do not report PASS from return code alone.

---

## 6) Issues, Debug, and Workarounds (Most Important)

This section is designed for production troubleshooting.

## 6.1 Root-cause-first debug workflow

1. Reproduce with the smallest command.
2. Identify failing stage (import / optimize / quantize / codegen).
3. Validate environment (`ldd`, PATH, venv).
4. Validate model signature (`circle-inspect --tensor_shape`).
5. Validate calibration data format/count/shape.
6. Inspect ONNX graph operator forms around failing area.
7. Trace ONE source code for assert/throw site.
8. Only then pick fix location (env, config, dataset, model, toolchain).

## 6.2 Known issue: missing shared libraries

Symptoms:

- `error while loading shared libraries: lib*.so`

Root cause:

- incomplete `LD_LIBRARY_PATH` for local ONE build.

Fix:

- add missing library directories (validate via `ldd ... | rg "not found"`).

## 6.3 Known issue: `Wrong number of inputs.` in quantization calibration

Symptoms:

- `record-minmax ... Wrong number of inputs.`

Root cause:

- H5 record input count != model input count.

How to diagnose:

1. Check model input nodes:
   ```bash
   circle-inspect --tensor_shape <model.circle>
   ```
2. Check H5 structure:
   - each sample `value/<idx>/` must contain `0..N-1` datasets for N model inputs.

Fix:

- multi-input model => multi-input H5 records (space-separated files per sample when building dataset)
- or use random calibration fallback if appropriate.

## 6.4 Known issue: reducer dtype assert in optimize

Symptoms:

- `Only support int 32` in `CircleShapeInferenceRule.cpp` (`infer_reducer`)

Root cause:

- ONE shape inference expects reducer indices as S32 in that path.
- ONNX conversion/lowering may produce incompatible reducer index forms.

How to diagnose:

- inspect ONE source assert location and calling op path
- inspect ONNX reduce node signatures and converted Circle reduce nodes
- compare failing vs passing ONNX lowering patterns

Fix direction:

- short-term: adjust export/lowering pattern to avoid problematic reducer path
- long-term: patch toolchain conversion/inference consistency in ONE.

## 6.5 Known issue: unsupported non-const MatMul in quantization

Symptoms:

- `Unsupported non const input ... MatMul ...`

Root cause:

- In ONE quantization flow, `one-quantize` calls `circle-quantizer --quantize_with_minmax`.
- `circle-quantizer` runs `luci::QuantizePreCheckerPass` before `QuantizeWithMinMaxPass`.
- `QuantizePreCheckerPass` enforces const-input constraints for specific operator inputs and throws:
  - `Unsupported non const input <node_name>`
- In this model family, the failing node was typically:
  - `/compressor/MatMul/tr` (or `/pre_compressor/MatMul/tr`)
- Therefore the immediate blocker is a toolchain precondition violation at quantization precheck, not calibration-data format.

Action:

- confirm exact MatMul node and producer chain
- choose model/export rewrite only after confirming toolchain limitation.

Batch evidence:

- A full 22-model SFC verification run (after import fixes) showed:
  - import: pass
  - optimize: pass
  - quantize: fail
- common error:
  - `Unsupported non const input ... /compressor/MatMul/tr`
  - or equivalent path (for example `/pre_compressor/MatMul/tr`)

Practical implication:

- Current graph family can be made importable, but quantization may fail unless optimize passes convert the graph into quantizer-acceptable form.

Validated fix path:

- Insert optimize pass:
  - `--replace_non_const_fc_with_batch_matmul`
- Practical execution order:
  1. `one-import-onnx`
  2. `one-optimize --replace_non_const_fc_with_batch_matmul`
  3. `one-quantize ...`

Verified outcome on SFC variants:

- Before this pass:
  - import: pass
  - optimize: pass
  - quantize: fail (`Unsupported non const input ... MatMul/tr`)
- After this pass:
  - full pipeline `import -> optimize -> quantize`: `22/22` pass (artifact-verified).

## 6.6 Known issue: import fails at `onnx.Conv` legalization

Symptoms:

- `failed to legalize operation 'onnx.Conv'` at depthwise conv paths

Observed pattern:

- depthwise-like conv (`group=C`) with input from explicit/dynamic `Pad` + shape-driven index logic nearby

Notes:

- changing only `group` can move the first failure, but usually reveals the next unsupported op.
- this is typically a graph-pattern compatibility issue, not a simple opset knob issue.

## 6.7 Known issue: import fails at `onnx.Slice` legalization

Symptoms:

- `failed to legalize operation 'onnx.Slice'`

Observed pattern:

- Slice end/axes driven by runtime shape expressions (`Shape/Gather/Div/Mul/...`) rather than simple constants

Fix direction:

- favor static/foldable slice parameters for fixed deployment shape
- avoid dynamic shape arithmetic in critical hot paths when NPU graph is static

## 6.8 Practical pre-import normalization: ONNX simplification

When raw ONNX import fails but export is stable, run a controlled simplification pass before `one-import-onnx`.

Example:

```python
import onnx
from onnxsim import simplify

m = onnx.load("<model.onnx>")
ms, ok = simplify(m, overwrite_input_shapes={"x": [1, 2, 1, 1025]})
assert ok
onnx.save(ms, "<model.sim.onnx>")
```

Important:

- `overwrite_input_shapes` must match the real exported model input shape.
- Do not reuse a single shape for all variants (for example, `F=1025` vs `F=512` variants).
- Wrong simplification shape can create false importer failures (for example broadcast mismatch in `Add`).

Operational findings from a 22-model SFC batch:

- raw ONNX import: `0/22` pass
- simplified ONNX import: `6/22` pass

This means simplification is a high-value pre-import step, but not a full solution.

### 6.8.1 Adaptive-mel Locoformer `ConstantOfShape` import failure

For `adaptive-mel-locoformer-lite-sfc.rt192k.fp512keep475`, the raw exported
ONNX contains `ConstantOfShape`. If simplification is skipped, `one-import-onnx`
can fail before quantization with:

```text
error: invalid tensor dimension size
onnx2circle: ... RankedTensorType ... Assertion `succeeded(...)' failed.
```

Validated local workaround:

```bash
./.venv/bin/python tools/online/verify_npu_variants.py \
  --mode recipe \
  --recipe-name-contains adaptive-mel-locoformer-lite-sfc \
  --run-name <RUN_NAME> \
  --force-onnxsim-large-shape-ops
```

Compile the simplified `model.sim.onnx`, not the raw `model.onnx`.

### 6.8.2 Quantization warning: all-positive min/max

ONE may print this message during uint8 asymmetric quantization when verbose luci
logging is enabled:

```text
The minimum and maximum values are all positive.
```

In the current local ONE build this comes from `compute_asym_scale_zp` in
`compiler/luci/pass/src/QuantizationUtils.cpp`. It is a warning that the
calibrated range is strictly positive, not a failure by itself. Treat the compile
as failed only if the command exits nonzero or an expected artifact such as
`model.q.circle` is missing.

## 6.9 Known issue after simplification: `Clip/min` and broadcast add

Common remaining failures:

1. `Circle.minimum ... operand #1 ... got 'none'` around `Clip/min`
2. `Circle.add ... operands don't have broadcast-compatible shapes`

Interpretation:

- simplification can expose/transform Clip/Add forms into importer-unfriendly variants.
- follow root-cause-first policy: inspect exact node, inputs, and constants before model rewrite.

Practical workaround for `Clip/min` class:

- Rewrite `Clip` to explicit `Min/Max` ops before import:
  - `Clip(x, min, "") -> Max(x, min)`
  - `Clip(x, "", max) -> Min(x, max)`
  - `Clip(x, min, max) -> Min(Max(x, min), max)`

This avoids optional Clip inputs that some importer paths treat as `none`.

Batch evidence (SFC variants):

- In one verification run, this rewrite converted all `Clip/min`-class failures in the target set from FAIL to PASS at import stage.

## 6.10 How to decide what to fix

Use this rule:

1. linker/config errors -> environment/config fix.
2. strict toolchain asserts with valid graph semantics -> toolchain limitation likely.
3. graph pattern-specific failures that disappear with equivalent rewrite -> model/export workaround possible.
4. prefer toolchain patch for durable fix; keep model semantics stable.

---

## 7) Generic Command Cheat Sheet (Template)

### Setup

```bash
export ONE_CMDS="<ONE_CMDS_DIR>"
source "$ONE_CMDS/venv/bin/activate"
export PATH="$ONE_CMDS:$PATH"
export LD_LIBRARY_PATH="<REQUIRED_LIB_DIRS>:$LD_LIBRARY_PATH"
```

### Export

```bash
<PYTHON> <EXPORT_SCRIPT.py> <MODEL_OR_CONFIG> --out <MODEL.ONNX> --opset <N> --check
```

### Compile

```bash
onecc -C <CONFIG.CFG>
```

### Inspect model inputs

```bash
circle-inspect --tensor_shape <MODEL.CIRCLE>
```

---

## 8) Example / Best Practice Case Study (This Project)

This section is **example-only** and can be reused as a reference pattern.

### Example context

- Model family: online SFC streaming variant
- Toolchain: locally built ONE
- Main successful config:
  - `/home/cmj/works/ASS/logs/one_compile_soft_band_query/config_opt_o1_like.cfg`

### Example best practices applied

1. Verified linker runtime with `ldd` for both `circle2circle` and `record-minmax`.
2. Switched from single-input calibration H5 to multi-input H5 matching model input count.
3. Used root-cause tracing into ONE source before choosing model/export workaround.
4. Kept operational logs and reusable configs in a dedicated output folder.

### Example outputs

- `model.circle`
- `model.opt.circle`
- `model.q.circle`

---

## 9) References (Template + Example)

### Generic references

- `<ONE_CMDS_DIR>/onecc.template.cfg`
- `<ONE_CMDS_DIR>/onelib/constant.py`
- `<ONE_CMDS_DIR>/one-quantize`
- ONE source: reducer shape inference and record-minmax input validation

### Example project references

- `/home/cmj/works/ASS/logs/one_compile_soft_band_query/WORKLOG_ONE_COMPILATION.md`
- `/home/cmj/works/ASS/logs/one_compile_soft_band_query/config_opt_o1_like.cfg`

---

## 10) Fill-in Template (Preflight + Run Sheet)

Use this one-page template before a new model compilation.

### 10.1 Project metadata

- Project name: `______________________________`
- Model name: `______________________________`
- Owner: `______________________________`
- Date: `______________________________`

### 10.2 Required paths

- ONE command dir (`ONE_CMDS`): `______________________________`
- ONE build lib root (`LIBROOT`): `______________________________`
- Python executable: `______________________________`
- Model config / checkpoint path: `______________________________`
- ONNX export script path: `______________________________`
- Output directory: `______________________________`
- onecc config path: `______________________________`

### 10.3 Environment preflight checklist

- [ ] `onecc` resolves from `PATH`
- [ ] ONE venv activated
- [ ] `LD_LIBRARY_PATH` includes required luci/interpreter/dio paths
- [ ] `ldd $ONE_CMDS/circle2circle` has no `not found`
- [ ] `ldd $ONE_CMDS/record-minmax` has no `not found`
- [ ] export dependencies present (`onnx`, `onnxscript`, torch-compatible stack)

### 10.4 Export plan

- Export mode: `[ ] streaming` `[ ] non-streaming`
- Fixed input shape (B,C,T,F): `(___, ___, ___, ___)`
- Opset target: `_____`
- Exporter mode:
  - `[ ] new exporter`
  - `[ ] legacy (dynamo=False)`
- Output ONNX path: `______________________________`

### 10.5 Calibration plan

- Quantization mode:
  - `[ ] random calibration (no input_data in config)`
  - `[ ] explicit calibration data`
- Calibration format:
  - `[ ] h5`
  - `[ ] list/filelist`
  - `[ ] directory`
- If H5, verify:
  - model input count = `_____`
  - per-record dataset count = `_____`
  - type/shape check expected: `[ ] yes` `[ ] no (rawData)`
- Calibration artifact path: `______________________________`

### 10.6 onecc stage toggles

- `one-import-onnx`: `[ ] True` `[ ] False`
- `one-optimize`: `[ ] True` `[ ] False`
- `one-quantize`: `[ ] True` `[ ] False`
- `one-codegen`: `[ ] True` `[ ] False`

### 10.7 First-run command sheet

```bash
# 1) Setup
export ONE_CMDS="<ONE_CMDS_DIR>"
source "$ONE_CMDS/venv/bin/activate"
export PATH="$ONE_CMDS:$PATH"
export LD_LIBRARY_PATH="<REQUIRED_LIB_DIRS>:$LD_LIBRARY_PATH"

# 2) Export ONNX
<PYTHON> <EXPORT_SCRIPT.py> <MODEL_OR_CONFIG> --out <MODEL.ONNX> --opset <N> --check

# 3) Compile
onecc -C <CONFIG.CFG>
```

### 10.8 Debug triage log (fill while running)

- Failure stage: `______________________________`
- Exact error text: `______________________________`
- Repro command: `______________________________`
- Root cause category:
  - `[ ] env`
  - `[ ] config`
  - `[ ] dataset`
  - `[ ] export/lowering`
  - `[ ] toolchain bug/limitation`
  - `[ ] model graph`
- Chosen fix and reason: `______________________________`
- Re-run result: `______________________________`
