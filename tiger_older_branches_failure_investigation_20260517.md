# Older TIGER / Ctx NPU Failure Investigation - 2026-05-17

## Scope

Investigated the failing older TIGER recipe branches from the ONE compilation pass:

- `tiger-ctx-deployable.rt192k`
- `tiger-ctx-tiger-like.rt192k`
- `tiger-deployable.rt192k`
- `tiger-tiger-like.rt192k`
- `tiger-npu-edge-v1.rt192k`
- `tiger-npu-large.rt192k`

`tiger-npu-edge-v2.rt192k` is the control case: it compiles through ONE import, optimize, and quantization.

## Finding 1: raw older branches fail in ONNX Slice legalization

The raw failing logs all stop during ONE ONNX import:

```text
loc("/separator/freq_path.1/Slice"): error: failed to legalize operation 'onnx.Slice'
loc("/separator/stages.0/freq_attn/Slice"): error: failed to legalize operation 'onnx.Slice'
```

The failing ONNX Slice nodes are Q/K/V channel splits from the older attention implementations. Example from `tiger-ctx-deployable.rt192k`:

```text
/separator/freq_path.1/Slice
data_shape ['?', 48, 67, '?']
starts/ends/axes/steps [[0], dynamic, [1]]

/separator/freq_path.1/Slice_1
data_shape ['?', 48, 67, '?']
starts/ends/axes/steps [dynamic, dynamic, [1]]
```

The pattern repeats:

| Variant | First failing family | Data shape form | Slice start/end form |
|---|---|---|---|
| `tiger-ctx-deployable` | `freq_path.1` | `['?', 48, 67, '?']` | dynamic end / dynamic starts |
| `tiger-ctx-tiger-like` | `freq_path.1` | `['?', 64, 67, '?']` | dynamic end / dynamic starts |
| `tiger-deployable` | `freq_path.1` | `['?', 48, 67, '?']` | dynamic end / dynamic starts |
| `tiger-tiger-like` | `freq_path.1` | `['?', 64, 67, '?']` | dynamic end / dynamic starts |
| `tiger-npu-edge-v1` | `stages.*.freq_attn` | `['?', 24, 67, '?']` | dynamic end / dynamic starts |
| `tiger-npu-large` | `stages.*.freq_attn` | `['?', 96, 67, '?']` | dynamic end / dynamic starts |

By contrast, the passing `tiger-npu-edge-v2` exports static Q/K/V slices:

```text
/separator/stages.0/freq_attn/Slice
data_shape [1, 4, 6, 67]
starts/ends/axes/steps [[0], [2], [2], [1]]
```

## ONE source trace for the Slice failure

The relevant converter is:

```text
/home/cmj/works/ONE/circle-mlir/circle-mlir/lib/pass/src/ops/SliceOp.h
```

Important logic:

- `ConvSlice::matchAndRewrite()` requires `axes` and `steps` to be constants.
- If `starts` and `ends` are constants, it lowers to a static `StridedSliceOp`.
- If either `starts` or `ends` is dynamic, it calls `ReplaceWithDynamicStridedSlice()`.
- `ReplaceWithDynamicStridedSlice()` has this hard condition:

```cpp
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

The older TIGER slices are along channel axis 1, but their input shape still has dynamic batch/time dimensions: `['?', C, 67, '?']`. Those dynamic non-axis dimensions violate the converter condition, so the pattern returns failure and MLIR reports `failed to legalize operation 'onnx.Slice'`.

This is why the passing edge-v2 variant works: its attention path reshapes into fixed `[1, H, proj, F]` style tensors and slices with static constants.

## Model source cause

Older attention code uses dynamic splitting patterns. For example, `TIGER/tiger_online.py`:

```python
qkv = self.qkv_conv(x)
head_chunks = torch.chunk(qkv, self.num_heads, dim=1)
...
q = head_chunk[:, 0:self.head_dim, :, :]
k = head_chunk[:, self.head_dim:2 * self.head_dim, :, :]
v = head_chunk[:, 2 * self.head_dim:, :, :]
```

That exports to ONNX with `Shape` / `Div` / `Mul`-derived split boundaries, then `Slice`.

The edge-v2 path avoids this by making the Q/K/V partition explicit after a static view. In `TIGER/tiger_npu_edge_v2.py`:

```python
qkv = self.qkv_conv(x)
qkv = qkv.view(B, self.n_heads, 2 * self.hid_chan + self.v_hid_chan, Fb)
q = qkv[:, :, :self.hid_chan, :]
k = qkv[:, :, self.hid_chan:2*self.hid_chan, :]
v = qkv[:, :, 2*self.hid_chan:, :]
```

## Finding 2: the older simplified ctx logs reach quantization, then fail in luci-interpreter Transpose

The previous retry directory:

```text
logs/npu_verify_general/tiger_transpose_rewire_retry_20260516/
```

contains ctx variants that got through import and optimize, then failed at quantization:

```text
record-minmax: .../luci-interpreter/src/kernels/Transpose.cpp:46:
Assertion `perm()->shape().dim(0) == dims' failed.
```

I reran `record-minmax` directly on `model.circle`, before `circle-quantizer` or `circle2circle`, and it fails the same way. So this is not introduced by weight fake-quantization or NCHW-to-NHWC optimization; it is already present in the imported Circle graph/runtime behavior.

Static Circle inspection:

```text
model.circle:     transpose_total=1212, static_bad_perm_len=0, perm_types={INT32: 1212}
model.opt.circle: transpose_total=7433, static_bad_perm_len=0, perm_types={INT32: 7433}
```

So the stored Circle transposes are not obviously malformed: every Transpose perm tensor is 1D INT32 and has the same static length as the static input rank.

The failing assertion is in ONE:

```text
/home/cmj/works/ONE/compiler/luci-interpreter/src/kernels/Transpose.cpp
```

```cpp
int dims = input()->shape().num_dims();
const int32_t *perm_data = getTensorData<int32_t>(perm());
...
assert(perm()->shape().num_dims() == 1);
assert(perm()->shape().dim(0) == dims);
```

Because the flatbuffer static metadata is consistent, the likely failure mechanism is runtime rank mutation before a Transpose executes. `RuntimeGraph::execute()` calls `kernel->configure()` immediately before every op execution, and kernels are allowed to resize outputs dynamically:

```cpp
kernel->configure();
_tensor_alloc_plan->allocate(index);
kernel->execute();
```

The older ctx Circle graph contains hundreds of `StridedSlice`, `Reshape`, `Tile`, and generated `Transpose` nodes. `StridedSlice::configure()` recomputes output rank/shape from runtime begin/end/stride tensors. A preceding dynamic shape operation can therefore change the rank observed by a later `Transpose`, even though the static Circle tensor rank looked valid.

## Why edge-v2 avoids both problems

`tiger-npu-edge-v2.rt192k` has:

- static Q/K/V split slices;
- no `Tile` or `ConstantOfShape`;
- far fewer Transpose nodes after import;
- full `record-minmax` success.

Observed passing artifacts:

```text
logs/npu_verify_general/tiger_edge_v2_allowlist_20260517/tiger-npu-edge-v2.rt192k/model.circle
logs/npu_verify_general/tiger_edge_v2_allowlist_20260517/tiger-npu-edge-v2.rt192k/model.opt.circle
logs/npu_verify_general/tiger_edge_v2_allowlist_20260517/tiger-npu-edge-v2.rt192k/model.q.circle
```

## Recommended fix path

Do not try to patch the ONE compiler first. The older TIGER graphs are outside the current converter comfort zone.

1. Replace `torch.chunk(qkv, ...)` and dynamic channel splitting in older `Unified2DAttentionOnFreqAsym` / `Unified2DAttentionOnFrameAsym` paths with edge-v2-style static `view` + fixed slice bounds.
2. Make export tensors fully static for the deployment cell: batch `1`, time `1`, fixed `freq_bins=67`, fixed `n_heads`, fixed per-head Q/K/V widths.
3. Remove `Tile` / `ConstantOfShape`-creating mask and state-update patterns, following the edge-v2 implementation.
4. Re-run raw ONE import. If it passes, then re-test `record-minmax`. The old Transpose assert should be re-evaluated only after the graph no longer depends on the dynamic Slice lowering path.

The practical conclusion: older TIGER/ctx branches are not blocked by a simple quantization setting. Their attention/state graph is structurally less NPU/ONE-friendly than edge-v2. The edge-v2 design is the right template for repairing them.

## Continuation: static QKV split patch - 2026-05-17

Implemented the first part of the recommended fix in:

```text
TIGER/tiger_online.py
```

Changes:

- `Unified2DAttentionOnFrameAsym` now reshapes QKV as
  `[B, n_heads, 2*hid+v_hid, T, F]` and slices Q/K/V with fixed per-head
  channel bounds.
- `Unified2DAttentionOnFreqAsym` now reshapes QKV as
  `[B, n_heads, 2*hid+v_hid, F, T]` and no longer uses
  `torch.chunk(qkv, n_heads, dim=1)`.
- T=1 frame-attention cache updates now use explicit fixed slice bounds:
  `1:window_size` and `window_size-1:window_size`.
- `ContextTimeUConvBlock` T=1 export context slicing now uses explicit
  `context_size:context_size+1` and `1:context_size+1` instead of dynamic
  negative tail slices.

Smoke validation:

```text
./.venv/bin/python -m py_compile TIGER/tiger_online.py
./.venv/bin/python -m TIGER.test_tiger_online --variant ctx_deployable --frames 3 --window sqrt_hann
```

Result:

```text
[consistency] max sequence-vs-cell diff: 0.00000262
[wrapper] output shape=(1, 12, 1025, 3)
```

ONNX inspection on `tiger-ctx-deployable.rt192k` after the patch:

```text
slice_total=472
dynamic_start_or_end=0
maxend=0
```

The original failing QKV slices are now static. Examples:

```text
/separator/freq_path.1/Slice    starts=[0] ends=[4] axes=[2] steps=[1]
/separator/freq_path.1/Slice_1  starts=[4] ends=[8] axes=[2] steps=[1]
/separator/freq_path.1/Slice_2  starts=[8] ends=[12] axes=[2] steps=[1]
/separator/frame_path.1/Slice   starts=[0] ends=[4] axes=[2] steps=[1]
/separator/frame_path.1/Slice_1 starts=[4] ends=[8] axes=[2] steps=[1]
/separator/frame_path.1/Slice_2 starts=[8] ends=[12] axes=[2] steps=[1]
```

## Continuation: verifier forced-onnxsim path

Also added an explicit diagnostic flag in:

```text
tools/online/verify_npu_variants.py
```

New flag:

```text
--force-onnxsim-large-shape-ops
```

Default behavior is unchanged: the verifier still skips `onnxsim` when the
graph contains `Tile` or `ConstantOfShape`, because simplification can be slow.
With the new flag, the verifier runs `onnxsim` anyway.

Diagnostic run:

```text
./.venv/bin/python tools/online/verify_npu_variants.py \
  --mode recipe \
  --recipe-name-contains tiger-ctx-deployable \
  --run-name tiger_old_static_qkv_force_onnxsim_compile_20260517 \
  --limit 1 \
  --force-onnxsim-large-shape-ops \
  --quantize-layer-fallback
```

Summary:

```text
| recipe | tiger-ctx-deployable.rt192k | FAIL | quantize |
```

Artifacts:

```text
logs/npu_verify_general/tiger_old_static_qkv_force_onnxsim_compile_20260517/tiger-ctx-deployable.rt192k/model.circle
logs/npu_verify_general/tiger_old_static_qkv_force_onnxsim_compile_20260517/tiger-ctx-deployable.rt192k/model.opt.circle
```

There is no `model.q.circle`; quantization fails during `record-minmax`.

Key log evidence:

```text
=== ONNXSIM ===
ConstantOfShape: 16 -> 0
Shape:           126 -> 0
Cast:             16 -> 0
Tile:            104 -> 104
Transpose:       310 -> 294

=== ONECC ===
record-minmax: .../luci-interpreter/src/kernels/Transpose.cpp:46:
Assertion `perm()->shape().dim(0) == dims' failed.
```

Manual confirmation:

```text
one-import-onnx model.force_sim.onnx -> model.force_sim.circle      PASS
one-optimize model.force_sim.circle -> model.force_sim.opt.circle   PASS
one-quantize model.force_sim.opt.circle -> model.force_sim.q.circle FAIL
```

## Updated conclusion

The static QKV/channel split repair removes the original ONNX Slice
legalization blocker. The older ctx branch can now reach the same quantization
stage as the previous simplified retry when `onnxsim` is forced.

The remaining blocker is not the original dynamic QKV slicing. It is the older
ctx graph's runtime-rank instability before a Circle `Transpose` during
`record-minmax`. The next useful repair should target the remaining old ctx
shape machinery, especially the `Tile`-heavy resize/injection path and the
`PRelu`/dynamic reshape patterns that edge-v2 already avoids.

## Continuation: finished quantized Circle compilation

Additional fixes landed after the runtime-rank diagnosis:

- `StaticFreqResize2D` now uses nearest-neighbor `F.interpolate` instead of
  `repeat_interleave`, removing the old `Tile` export pattern.
- TIGER recipe construction now applies `sanitize_for_npu_edge(...)` to the
  legacy deployable branches and `npu-large`, replacing export-hostile `PReLU`
  with NPU-safe activations through the same utility used by edge variants.
- Frame and frequency attention no longer scalar-index heads/time frames in the
  export path. Scalar indexing exported as ONNX `Gather`; ONE kept the runtime
  rank for a gathered tensor and hit:

```text
[transpose-diagnostic] input=/separator/freq_path.1/Gather_3 dims=5
perm=Circle.pseudo_const512 perm_len=3
```

- Single-frame/export attention now uses rank-4 batched `torch.matmul` instead
  of per-head rank-3 `bmm`. This prevents ONNX/Circle from lowering dynamic
  activation-attention `MatMul` into `FULLY_CONNECTED` with a non-constant
  weight input:

```text
Unsupported non const input /separator/freq_path.1/MatMul/tr
```

Validation commands:

```text
./.venv/bin/python -m py_compile TIGER/tiger_online.py TIGER/training_wrapper.py tools/online/verify_npu_variants.py

./.venv/bin/python -m TIGER.test_tiger_online \
  --variant ctx_deployable \
  --frames 3 \
  --window sqrt_hann

./.venv/bin/python tools/online/verify_npu_variants.py \
  --mode recipe \
  --recipe-name-contains tiger-ctx-deployable \
  --run-name tiger_old_batchmatmul_compile_20260517 \
  --limit 1 \
  --force-onnxsim-large-shape-ops \
  --quantize-layer-fallback

./.venv/bin/python tools/online/verify_npu_variants.py \
  --mode recipe \
  --recipe-name-contains tiger- \
  --run-name tiger_old_all_batchmatmul_compile_20260517 \
  --force-onnxsim-large-shape-ops \
  --quantize-layer-fallback
```

Smoke result:

```text
[consistency] max sequence-vs-cell diff: 0.00000250
[wrapper] output shape=(1, 12, 1025, 3)
```

Final full TIGER recipe sweep:

```text
# NPU Variant Verification Summary

- Total: 7
- PASS: 7
- FAIL: 0

| Kind | Variant | Status | Fail Stage |
|---|---|---|---|
| recipe | tiger-ctx-deployable.rt192k | PASS | - |
| recipe | tiger-ctx-tiger-like.rt192k | PASS | - |
| recipe | tiger-deployable.rt192k | PASS | - |
| recipe | tiger-npu-edge-v1.rt192k | PASS | - |
| recipe | tiger-npu-edge-v2.rt192k | PASS | - |
| recipe | tiger-npu-large.rt192k | PASS | - |
| recipe | tiger-tiger-like.rt192k | PASS | - |
```

Quantized Circle artifacts:

```text
logs/npu_verify_general/tiger_old_all_batchmatmul_compile_20260517/tiger-ctx-deployable.rt192k/model.q.circle
logs/npu_verify_general/tiger_old_all_batchmatmul_compile_20260517/tiger-ctx-tiger-like.rt192k/model.q.circle
logs/npu_verify_general/tiger_old_all_batchmatmul_compile_20260517/tiger-deployable.rt192k/model.q.circle
logs/npu_verify_general/tiger_old_all_batchmatmul_compile_20260517/tiger-npu-edge-v1.rt192k/model.q.circle
logs/npu_verify_general/tiger_old_all_batchmatmul_compile_20260517/tiger-npu-edge-v2.rt192k/model.q.circle
logs/npu_verify_general/tiger_old_all_batchmatmul_compile_20260517/tiger-npu-large.rt192k/model.q.circle
logs/npu_verify_general/tiger_old_all_batchmatmul_compile_20260517/tiger-tiger-like.rt192k/model.q.circle
```

Updated conclusion: the older TIGER recipe branches now complete ONNX export,
ONE import, optimization, calibration, and quantized Circle generation. The
root causes were a sequence of export/import mismatches: dynamic QKV slicing,
`Tile`-based resize, `PReLU`, scalar `Gather` rank handling, and rank-3
activation-attention `MatMul` lowering to const-weight `FULLY_CONNECTED`.
