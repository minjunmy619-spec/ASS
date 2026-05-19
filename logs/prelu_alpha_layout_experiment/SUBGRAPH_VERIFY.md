# Subgraph `relu + alpha * min(x,0)` vs builtin `nn.PReLU` — ONE verify

- Shapes: `x` `[1,8,4,5]`, same `alpha` vector length `8` (RNG seed 1).

## PyTorch equivalence (builtin vs subgraph)
- max abs diff = **`0.000000e+00`** (expect ~0).

## ONNX ops (subgraph model)
- `tiny_prelu_subgraph.onnx` op counts: `{'Constant': 1, 'Min': 1, 'Relu': 1, 'Mul': 1, 'Add': 1}`
- Contains **`PRelu`**: **`False`**.

## ONNX Runtime
- Session: **PASS** — OK
- vs PyTorch subgraph max diff **`0.000000e+00`**.

## ONE `onecc` (same pipeline as verifier: nhwc + channel quant)
- `one-create-quant-dataset` rc=0
- `onecc` rc=`0`
- **`model.q.circle` exists**: **`True`**
- Log contains **`Non-channel dimension of const node must be 1`**: **`False`**
- Full log: `/home/cmj/works/ASS/logs/prelu_alpha_layout_experiment/onecc_run_subgraph_min_relu_mul/onecc.log`

## Conclusion
**Subgraph replacement succeeds** end-to-end with channel-wise quantization for this minimal model (no `CirclePRelu` / `[C,1,1]` alpha quant path).
