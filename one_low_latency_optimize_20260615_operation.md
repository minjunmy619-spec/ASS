# ONE Low-Latency Optimize Preset

Date: 2026-06-15

## Purpose

The source-aware MelBand Loco-CNB latency investigation showed that the default
ONE optimize path can make the post-layout Circle graph look much worse than the
imported graph if `convert_nchw_to_nhwc=True` is used without cleanup passes.
This note generalizes the fix so any deploy candidate in this repo can be
compiled and compared with the same low-latency ONE preset.

## Key compiler observation

For PyTorch/ONNX models, Conv2D is exported with logical NCHW tensors. Circle and
TFLite Conv kernels are NHWC-oriented, so the verifier still keeps:

```ini
convert_nchw_to_nhwc=True
```

However, on its own this can create many layout bridge nodes. For the current
source-aware residual-SFX Loco-CNB recipe, the optimized Circle graph changed
from:

```text
Default ONE optimize:
TRANSPOSE: 2896
```

to:

```text
Low-latency ONE optimize:
TRANSPOSE: 72
```

when cleanup/fusion passes were enabled after layout conversion.

The merged-FSMN low-latency recipe compiled to:

```text
TRANSPOSE: 48
RESHAPE: 18
STRIDED_SLICE: 8
DEPTHWISE_CONV_2D: 50
CONV_2D: 165
```

## Verifier support

Updated:

```text
tools/online/verify_npu_variants.py
```

Reusable option:

```text
--low-latency-optimize
```

This adds these ONE optimize flags after the existing baseline flags:

```ini
replace_non_const_fc_with_batch_matmul=True
convert_nchw_to_nhwc=True
remove_redundant_transpose=True
remove_unnecessary_transpose=True
remove_redundant_reshape=True
remove_unnecessary_reshape=True
remove_unnecessary_add=True
remove_unnecessary_mul=True
remove_unnecessary_div=True
fuse_mul_with_conv=True
fuse_add_with_conv=True
fuse_mul_with_fullyconnected=True
fuse_add_with_fully_connected=True
fuse_mean_with_mean=True
fuse_mul_with_div=True
transform_sqrt_div_to_rsqrt_mul=True
```

The verifier now also records Circle operator counts for every compiled variant:

```text
circle_op_counts.import
circle_op_counts.optimize
circle_op_counts.quantize
circle_top_ops
circle_opt_transpose
circle_opt_reshape
circle_opt_strided_slice
```

Additional generalized ONNX pre-import rewrite:

```text
div_static_const_to_mul
```

This safely rewrites finite floating-point static divisions:

```text
x / const  ->  x * reciprocal_const
```

It is intentionally limited to static non-zero constants. Dynamic divisions are
left in place because they still require a real reciprocal/divide computation and
usually represent a model design choice, not a converter artifact.

These are written to:

```text
summary.json
summary.md
run.log
```

## How to use on any recipe variant

Example pattern:

```bash
cd /home/cmj/works/ASS
.venv/bin/python tools/online/verify_npu_variants.py \
  --mode recipe \
  --recipe-name-contains <recipe-name-substring> \
  --run-name <run-name> \
  --force-onnxsim-large-shape-ops \
  --quantize-layer-fallback \
  --streaming \
  --low-latency-optimize
```

Compare with the baseline path by running the same command without:

```text
--low-latency-optimize
```

Then inspect:

```text
logs/npu_verify_general/<run-name>/summary.md
logs/npu_verify_general/<run-name>/summary.json
logs/npu_verify_general/<run-name>/<variant>/run.log
```

The summary table includes:

```text
Low-latency opt
Opt Transpose
Opt Reshape
Opt StridedSlice
```

The JSON contains full per-stage Circle op counts so variants can be compared by
script:

```bash
cd /home/cmj/works/ASS
.venv/bin/python - <<'PY'
import json
from pathlib import Path
summary = json.loads(Path('logs/npu_verify_general/<run-name>/summary.json').read_text())
for item in summary:
    ops = item['circle_op_counts'].get('optimize', {})
    print(item['variant'], ops.get('TRANSPOSE', 0), ops.get('RESHAPE', 0), ops.get('DIV', 0))
PY
```

The most important latency-related counts are:

```text
TRANSPOSE
RESHAPE
STRIDED_SLICE
SPLIT_V
CONCATENATION
PAD
```

Arithmetic counts are also useful:

```text
CONV_2D
DEPTHWISE_CONV_2D
BATCH_MATMUL
FULLY_CONNECTED
LOGISTIC
SOFTMAX
MEAN
RSQRT
DIV
SQRT
```

## Latency rewrite candidate reporter

Added standalone report tool:

```text
tools/online/report_latency_rewrite_candidates.py
```

Use it on any verifier artifact directory:

```bash
cd /home/cmj/works/ASS
.venv/bin/python tools/online/report_latency_rewrite_candidates.py \
  logs/npu_verify_general/<run-name>/<variant>/model.sim.onnx \
  --circle logs/npu_verify_general/<run-name>/<variant>/model.opt.circle \
  --md-out logs/npu_verify_general/<run-name>/<variant>/latency_rewrite_report.md \
  --json-out logs/npu_verify_general/<run-name>/<variant>/latency_rewrite_report.json
```

The report classifies:

```text
memory_ops_total
slow_math_ops_total
div_by_static_const
div_by_sqrt
dynamic_or_activation_div
sigmoid_outputs_consumed_by_mul
split_to_sigmoid_gate
cat_slice_state_updates
non_depthwise_grouped_conv
depthwise_grouped_conv
rank_gt4_values
activation_matmul
```

Use this report after every new model/export attempt. It identifies whether a
latency issue is likely:

- a safe ONNX rewrite;
- a PyTorch coding-style cleanup;
- a ONE optimize flag issue;
- or a real model-design ablation.

## Rewrite pattern catalogue

| Pattern | Where it appears | Safe action | Risky/model action |
|---|---|---|---|
| `x / const` | source averaging, fixed scaling | Rewrite to `x * reciprocal`; verifier also has `div_static_const_to_mul` | None; this is math-equivalent for finite non-zero constants |
| `x / sqrt(mean(x*x)+eps)` | RMSNorm | Use `--low-latency-optimize` so ONE emits `RSQRT + MUL`; keep as-is in PyTorch unless exporter improves | Remove/reduce RMSNorm blocks only as a trained ablation |
| dynamic `Div` | adaptive Mel expansion normalization | No safe generic ONNX rewrite | Pre-normalize static basis, disable dynamic renormalization, or replace with learned scale; requires quality validation |
| `Split -> Sigmoid -> Mul` | GLU/gates | None in ONE; report counts it | Replace low-value GLU gates with ReLU/ReLU6/single-branch FFN ablations |
| many `Split/Concat` | source loops, branch fusion, GLU, state handling | Some cleanup via ONE; inspect final Circle | Pack sources/channels and vectorize shared heads; merge parallel branches |
| `Concat -> Slice` state update | streaming cache | Mostly unavoidable for stateful causal conv | Reduce state tensors, state channels, context, or merge memory branches |
| non-depthwise grouped Conv | grouped conv with `1 < groups != in_channels` | Avoid in model configs | Use dense Conv or true depthwise Conv only |
| rank-4 one-frame MatMul | SFC transport, attention | Verifier rewrites `[1,1,M,K] @ [1,1,K,N]` to 2D MatMul | Reduce SFC transitions or attention branches if counts remain high |
| high final `Transpose` | layout islands after NCHW->NHWC | Use `--low-latency-optimize` cleanup passes | Reduce layout-changing MatMul/Transpose paths or port deploy graph to true NHWC |

### Current source-aware low-latency report interpretation

For the merged-FSMN source-aware low-latency recipe, the report shows:

```text
ONNX Div: 72
  div_by_sqrt: 69
  dynamic_or_activation_div: 3
  div_by_static_const: 0

Circle optimized Div: 3
Circle optimized RSQRT: 69
```

Interpretation:

- constant divisions have been removed in PyTorch or by rewrite;
- RMSNorm divisions are converted by ONE to `RSQRT + MUL`;
- the remaining `DIV: 3` are dynamic Mel-expander normalization divisions and
  should only be removed by a trained/validated model ablation.

The same report also shows:

```text
sigmoid_outputs_consumed_by_mul: 116
split_to_sigmoid_gate: 59
cat_slice_state_updates: 8
```

So next latency ablations should focus more on gates/source loops/state update
structure than on constant divisions.

## Source-aware low-latency recipe

Added low-latency residual-SFX recipe overlay:

```text
recipes/dnr/models/source-aware-melband-loco-cnb.student-npu-residual-sfx-lowlat.rt192k.fp512keep475/config.yaml
```

Added distillation overlay:

```text
recipes/dnr/models/source-aware-melband-loco-cnb.student-npu-residual-sfx-lowlat.distill.rt192k.fp512keep475/config.yaml
```

These set:

```yaml
loco_cnb_merge_dilations: true
```

and inherit the residual-SFX safety override:

```yaml
loco_cnb_mixture_consistency: false
```

The mixture-consistency override is required for all residual-SFX variants. If
the explicit 2-stem core is mixture-consistent by itself, the wrapper residual
SFX source collapses to zero/projector residual. The builder now rejects
`residual_source_enabled=True` with `mixture_consistency=True`.

The merged-FSMN option collapses the three FSMN dilation branches into one
depthwise causal Conv2D with the same max temporal context/state size. This is
model-specific, but the ONE `--low-latency-optimize` verifier flag is general.

## Recommended workflow for future variants

1. First verify correctness/export with the normal verifier path.
2. Re-run with `--low-latency-optimize`.
3. Compare `summary.md` and `summary.json` operator counts.
4. Keep the low-latency optimized artifacts only if import, optimize, and
   quantize all pass and the final memory-op counts drop.
5. If a model still has high `TRANSPOSE`/`RESHAPE`/`SPLIT_V` counts, do a
   model-level rewrite rather than relying only on ONE passes.

Useful model-level rewrites include:

- remove or merge small parallel Conv branches;
- avoid non-depthwise grouped Conv, because ONE lowers it to Split/Conv/Concat;
- reduce source loops that repeatedly call shared full-band heads;
- replace low-value GLU/Sigmoid gates with simpler activations where quality
  allows;
- reduce repeated RMSNorm blocks;
- keep attention disabled for latency unless quality requires it;
- move mixture consistency outside the NPU graph if DSP/postprocess can handle it.

### Recommended experiment ladder

Use the report counts to pick the cheapest next ablation:

1. **Safe compiler/tool rewrites**
   - `div_static_const_to_mul`
   - rank-4 one-frame MatMul flatten
   - transpose perm int64 -> int32
   - low-latency ONE cleanup/fusion flags
2. **Safe PyTorch style rewrites**
   - replace `/ float(k)` with `* (1.0 / float(k))`
   - merge equivalent branches that preserve context/output shapes
   - keep grouped conv either dense or true depthwise
3. **Low-risk trainable ablations**
   - fewer GLU gates in pooled mixers/FFNs
   - fewer RMSNorms per macro block
   - packed source-shared mask head instead of source loop
4. **Higher-risk quality ablations**
   - disable dynamic expander renormalization to remove dynamic `Div`
   - remove/reduce source competition layers
   - move mixture consistency or correction outside NPU graph

Do not apply steps 3-4 without training/distillation and per-stem quality checks.

## Validation performed

Low-latency optimizer flag on the existing residual-SFX recipe:

```bash
cd /home/cmj/works/ASS
.venv/bin/python tools/online/verify_npu_variants.py \
  --mode recipe \
  --recipe-name-contains source-aware-melband-loco-cnb.student-npu-residual-sfx.rt192k.fp512keep475 \
  --run-name source_aware_loco_latency_lowopt_20260615 \
  --force-onnxsim-large-shape-ops \
  --quantize-layer-fallback \
  --streaming \
  --low-latency-optimize
```

Result:

```text
PASS
```

Merged-FSMN low-latency recipe after generalized op-count and division-rewrite tooling:

```bash
cd /home/cmj/works/ASS
.venv/bin/python tools/online/verify_npu_variants.py \
  --mode recipe \
  --recipe-name-contains source-aware-melband-loco-cnb.student-npu-residual-sfx-lowlat.rt192k.fp512keep475 \
  --run-name source_aware_loco_mergedfsmn_lowopt_divrewrite_20260615 \
  --force-onnxsim-large-shape-ops \
  --quantize-layer-fallback \
  --streaming \
  --low-latency-optimize
```

Result:

```text
PASS
```

Final optimized Circle counts from `summary.md`:

```text
TRANSPOSE: 48
RESHAPE: 18
STRIDED_SLICE: 8
DIV: 3
RSQRT: 69
```

Standalone latency report command:

```bash
cd /home/cmj/works/ASS
.venv/bin/python tools/online/report_latency_rewrite_candidates.py \
  logs/npu_verify_general/source_aware_loco_mergedfsmn_lowopt_divrewrite_20260615/source-aware-melband-loco-cnb.student-npu-residual-sfx-lowlat.rt192k.fp512keep475/model.sim.onnx \
  --circle logs/npu_verify_general/source_aware_loco_mergedfsmn_lowopt_divrewrite_20260615/source-aware-melband-loco-cnb.student-npu-residual-sfx-lowlat.rt192k.fp512keep475/model.opt.circle
```

Synthetic validation for the generalized ONNX rewrite:

```text
rewrite_div_by_static_const_to_mul: Div -> Mul, rewritten=1
```
