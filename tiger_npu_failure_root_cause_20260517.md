# TIGER NPU Failure Root Cause - 2026-05-17

## Scope

Investigated the failing TIGER variants with the ONE source tree at `/home/cmj/works/ONE` and the latest verification artifacts under:

- `logs/npu_verify_general/tiger_all_api_20260517/`
- `logs/npu_verify_general/tiger_transpose_rewire_retry_20260516/`

The passing control is `tiger-npu-edge-v2.rt192k`.

## Current failure: ONNX import fails at Slice legalization

Failing variants:

- `tiger-ctx-deployable.rt192k`
- `tiger-ctx-tiger-like.rt192k`
- `tiger-deployable.rt192k`
- `tiger-tiger-like.rt192k`
- `tiger-npu-edge-v1.rt192k`
- `tiger-npu-large.rt192k`

Representative error:

```text
loc("/separator/freq_path.1/Slice"): error: failed to legalize operation 'onnx.Slice'
loc("/separator/stages.0/freq_attn/Slice"): error: failed to legalize operation 'onnx.Slice'
```

## ONE source execution path for import failure

The `onecc` config invokes `one-import-onnx`, which calls the default converter `onnx2circle`:

- `/home/cmj/works/ONE/build/compiler/one-cmds/one-import-onnx`, lines 252-260

Inside `onnx2circle`, ONNX ops are converted by the MLIR pass:

- `/home/cmj/works/ONE/circle-mlir/circle-mlir/lib/pass/src/ConvertONNXToCirclePass.cpp`
- `ConvertONNXToCirclePass::runOnOperation()` registers `ConvSlice` at line 271.
- `applyFullConversion()` is called at line 283.

The failing pattern is implemented in:

- `/home/cmj/works/ONE/circle-mlir/circle-mlir/lib/pass/src/ops/SliceOp.h`

Critical conditions:

```cpp
// axes and steps must be constants
if (!(IsConstant(axes) && IsConstant(steps)))
  return mlir::failure();

// if starts/ends are not constants, dynamic lowering is attempted
return ReplaceWithDynamicStridedSlice(...);
```

In `ReplaceWithDynamicStridedSlice()`:

```cpp
assert(normalizedAxes.size() == 1 && "Dynamic slice only supports single axis");

// Dynamic input shape is allowed only the dim of axis is unknown
// Other dims should be static
int32_t axis = normalizedAxes[0];
for (size_t d = 0; d < inshape.size(); ++d)
{
  if (axis == d)
    continue;
  if (mlir::ShapedType::isDynamic(inshape[d]))
    return mlir::failure();
}
```

So dynamic `starts` or `ends` are only supported when all non-sliced dimensions are statically known.

## Failing ONNX pattern

For `tiger-ctx-deployable.rt192k`, the first failing Slice is:

```text
node: /separator/freq_path.1/Slice
data: /separator/freq_path.1/qkv_conv/norm/Add_1_output_0
data_shape after ONNX shape inference: ['unk__214', 48, 67, 'unk__217']
starts: const [0]
ends: /separator/freq_path.1/Mul_output_0, dynamic
axes: const [1]
```

The dynamic `ends` producer chain is:

```text
/separator/freq_path.1/Mul_output_0
  Mul
    Div
      Add
        Gather
          Shape(qkv_conv output)
          Constant [1]
        Constant [3]
      Constant [4]
    Constant [1]
```

This is the ONNX form emitted for splitting the QKV projection by head/channel count. Although the deployment input shapes are fixed, the exporter represents the split boundary as `Shape(qkv)[1]` arithmetic, not as literal constants.

The Slice is along axis 1, but non-axis dimensions 0 and 3 are dynamic in the converter-visible shape. This violates `ReplaceWithDynamicStridedSlice()` and makes MLIR conversion return failure.

For `tiger-npu-edge-v1.rt192k`, the same pattern appears under:

```text
/separator/stages.0/freq_attn/Slice
data_shape: ['unk__133', 24, 67, 'unk__136']
axes: const [1]
starts/ends: Shape/Gather/Div/Mul-derived
```

## Model source cause

Older TIGER attention uses dynamic channel/head splitting. In `TIGER/tiger_online.py`:

```python
qkv = self.qkv_conv(x)
head_chunks = torch.chunk(qkv, self.num_heads, dim=1)
for head_chunk in head_chunks:
    q = head_chunk[:, 0:self.head_dim, :, :]
    k = head_chunk[:, self.head_dim:2 * self.head_dim, :, :]
    v = head_chunk[:, 2 * self.head_dim:, :, :]
```

Relevant lines:

- `Unified2DAttentionOnFreqAsym.forward()`: lines 1570-1577
- `Unified2DAttentionOnFrameAsym.forward()`: lines 1430-1435

The `torch.chunk()` path exports a runtime shape arithmetic graph for chunk boundaries. That graph is legal ONNX, but it falls outside ONE's current `ONNXSliceOp` lowering constraint.

## Why edge-v2 passes import

`tiger-npu-edge-v2.rt192k` rewrites attention to a static reshape and static slices. In `TIGER/tiger_npu_edge_v2.py`:

```python
qkv = self.qkv_conv(x)
qkv = qkv.view(B, self.n_heads, 2 * self.hid_chan + self.v_hid_chan, Fb)
q = qkv[:, :, :self.hid_chan, :]
k = qkv[:, :, self.hid_chan:2*self.hid_chan, :]
v = qkv[:, :, 2*self.hid_chan:, :]
```

Observed ONNX for edge-v2:

```text
node: /separator/stages.0/freq_attn/Slice
data_shape: [1, 4, 6, 67]
starts: const [0]
ends: const [2]
axes: const [2]
steps: const [1]
```

Since `starts`, `ends`, `axes`, and `steps` are constants, ONE uses the static `ReplaceWithStridedSlice()` path and import succeeds.

## Older retry failure: quantization fails in record-minmax Transpose

The previous retry folder contains ctx graphs that were simplified enough to pass import and optimize, then failed during quantization:

```text
record-minmax: /app/ONE/compiler/luci-interpreter/src/kernels/Transpose.cpp:46:
virtual void luci_interpreter::kernels::Transpose::configure():
Assertion `perm()->shape().dim(0) == dims' failed.
```

Direct reproduction on the imported Circle model, without quantizer weight fake-quantization or `circle2circle`, still fails:

```bash
record-minmax \
  --input_model logs/npu_verify_general/tiger_transpose_rewire_retry_20260516/tiger-ctx-deployable.rt192k/model.circle \
  --output_model /tmp/opencode/tiger_ctx_model_circle_minmax.circle \
  --input_data logs/npu_verify_general/tiger_transpose_rewire_retry_20260516/tiger-ctx-deployable.rt192k/calib.h5 \
  --input_data_format h5
```

Result:

```text
rc -6
Recording 0'th data
record-minmax: .../Transpose.cpp:46: Assertion `perm()->shape().dim(0) == dims' failed.
```

The same direct `record-minmax` command on the passing edge-v2 imported Circle succeeds.

## ONE source execution path for quantization failure

`one-quantize` performs three steps:

1. `circle-quantizer --quantize_dequantize_weights`
2. `record-minmax` to execute the model and embed activation min/max
3. `circle-quantizer --quantize_with_minmax`

This flow is in:

- `/home/cmj/works/ONE/build/compiler/one-cmds/one-quantize`, lines 459-558

`record-minmax` does:

- load model: `RecordMinMax::initialize()`, `/home/cmj/works/ONE/compiler/record-minmax/src/RecordMinMax.cpp`, lines 75-100
- read calibration inputs and write them to interpreter tensors: lines 220-235
- execute interpreter: line 238

Interpreter execution path:

- `Interpreter::interpret()`, `/home/cmj/works/ONE/compiler/luci-interpreter/src/Interpreter.cpp`, line 142
- `RuntimeGraph::execute()`, `/home/cmj/works/ONE/compiler/luci-interpreter/src/core/RuntimeGraph.cpp`, lines 151-199
- each kernel runs `configure()` immediately before allocation and `execute()`:

```cpp
kernel->configure();
_tensor_alloc_plan->allocate(index);
kernel->execute();
```

The failing assert is in:

- `/home/cmj/works/ONE/compiler/luci-interpreter/src/kernels/Transpose.cpp`, lines 36-55

```cpp
int dims = input()->shape().num_dims();
const int32_t *perm_data = getTensorData<int32_t>(perm());
assert(perm()->shape().num_dims() == 1);
assert(perm()->shape().dim(0) == dims);
```

This means a Transpose kernel sees an input tensor with runtime rank `dims` different from the static permutation vector length.

## Likely mechanism for the Transpose assert

The simplified older ctx graph contains many dynamic shape-manipulation ops (`StridedSlice`, `Reshape`, `Tile`, `Transpose`). Static inspection did not show malformed Transpose perm constants: perm tensors were 1D INT32 and statically matched the associated input ranks.

However, luci-interpreter resizes tensors dynamically during `configure()`. `StridedSlice::configure()` recomputes output rank and shape at runtime from the current input shape and begin/end/stride tensors:

- `/home/cmj/works/ONE/compiler/luci-interpreter/src/kernels/StridedSlice.cpp`, lines 38-105

It constructs `output_shape_vector` from `input()->shape().num_dims()` and then calls:

```cpp
output()->resize(output_shape);
```

So a preceding dynamic StridedSlice/Reshape chain can mutate the rank observed by a later Transpose. When that later Transpose has a static perm vector generated for the compile-time rank, `perm()->shape().dim(0) != input()->shape().num_dims()` and `record-minmax` aborts.

This is not a calibration H5 shape/count issue: the failure occurs after input data is accepted and execution starts, and the direct edge-v2 control run succeeds with the same record-minmax path.

## Root cause summary

The failing older TIGER variants are structurally incompatible with the current ONE import + quantization path:

1. Their attention Q/K/V split is exported as dynamic ONNX Slice boundaries (`Shape/Gather/Div/Mul`) along channel axis.
2. ONE's `ConvSlice` dynamic lowering only supports dynamic starts/ends when all non-sliced dimensions are static.
3. The older graph presents dynamic non-axis dimensions to the converter, so raw import fails.
4. If aggressive ONNX simplification folds enough of the graph to pass import, the resulting Circle still has complex dynamic shape/rank behavior that can fail during `record-minmax` at Transpose runtime rank validation.
5. Edge-v2 passes because it changes the model graph, not because of a quantization setting: Q/K/V split is static after a fixed reshape, and Tile/ConstantOfShape-heavy state patterns are avoided.

## Recommended fix direction

Do not patch quantization first. The durable fix is to make the older TIGER graph look like edge-v2 before ONE import:

1. Replace `torch.chunk(qkv, self.num_heads, dim=1)` and dynamic per-head channel splitting with explicit fixed `view(B, H, proj_per_head, F[, T])` plus constant slices.
2. Keep deployment export at fixed `B=1`, `T=1`, `freq_bins=67`, fixed `n_heads`, fixed hidden widths.
3. Remove or rewrite `Tile` / `ConstantOfShape` state and mask construction patterns following edge-v2.
4. Re-run raw import first. Only after import is stable should the Transpose/record-minmax path be retested.
