# PRelu α layout experiment — findings

## Setup
- Tensor shapes: **NCHW** `input = [1, 8, 4, 5]`
- Same learned slopes `α` as length-8 vector (fixed RNG seed).

## A) NumPy: `[1,1,1,C]` is correct only with NHWC activations

- Define PRelu as `relu(x) + α * min(x, 0)` with broadcasting (matches PyTorch semantics).
- **`α` `[C,1,1]` + `x` `[1,C,H,W]`** → reference output `y_nchw`.
- **`α` `[1,1,1,C]` + `x` `[1,H,W,C]`** → `y_nhwc`; transpose back to NCHW.
- **`max(|y_nchw - transpose(y_nhwc)|)`** = `0.000000e+00` (should be ~0).

- **Wrong mix: `x` NCHW `[1,C,H,W]` with `α` `[1,1,1,C]`** → NumPy rejects broadcast (`ValueError`): arithmetic not even defined.

## B) PyTorch → ONNX export
- Output: `tiny_prelu_nchw.onnx`
- Detected PRelu slope initializer `onnx::PRelu_5` shape **`(8, 1, 1)`**.

## C) ONNX Runtime (same `x` NCHW)
- Baseline model session: **PASS** — OK
- Output shape `(1, 8, 4, 5)`, finite `True`.
- vs PyTorch same weights: max diff `0.000000e+00`.

## D) ONNX Runtime after rewriting slope → `[1,1,1,C]` (graph still NCHW)
- Rewrite: rewrote onnx::PRelu_5: (8, 1, 1) -> (1, 1, 1, 8) | onnx.checker: OK
- Session: **FAIL** — RuntimeException: [ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : Non-zero status code returned while running PRelu node. Name:'/act/PRelu' Status Message: /onnxruntime_src/onnxruntime/core/providers/cpu/math/element_wise_ops.h:540 void onnxruntime::BroadcastIterator::Init(ptrdiff_t, ptrdiff_t) axis == 1 || axis == largest was false. Attempting to broadcast an axis by a dimension other than 1. 5 by 8


## E) ONE `onecc` (optional — channel-wise quant)
- **`baseline_nchw`**: `one-import-onnx` / optimize succeed (`model.circle`, `model.opt.circle`). Quantization exits rc=`250` with `circle_quantizer: Non-channel dimension of const node must be 1` (expected for `[C,1,1]` α under channel-wise weights).
- **`slope_111c_on_nchw`**: fails **during ONNX import / shape inference** (rc=`255`), not quantization: `Incompatible broadcast matching 5 with 8` on `/act/PRelu` — width/F dimension **5** vs channel **8**. No `.circle` artifacts.

## Final conclusions

1. **`α` shaped `[1,1,1,C]` matches per-channel PRelu iff activations are `[B,H,W,C]`** (NumPy section A).
2. **Applying `[1,1,1,C]` on ONNX while inputs stay `[B,C,H,W]` breaks `PRelu`**: NumPy cannot broadcast that pair; ONNX Runtime rejects execution; ONE fails import (`Incompatible broadcast matching 5 with 8` for `[1,8,4,5]` vs `[1,1,1,8]`).
3. **`convert_nchw_to_nhwc=True` alone does not prove ONNX slope tensors were rewritten**: a layout fix requires ONE (or tooling) to transpose **`CirclePRelu` alpha** consistently with activations, not only renaming dims on paper.
4. **Runtime proof**: keeping channel-wise quantization, the robust fixes remain **ONE-side alpha handling**, **avoid `CirclePRelu` alpha layout** (subgraph / different activation), or **`granularity=layer`** as an explicit fallback.
