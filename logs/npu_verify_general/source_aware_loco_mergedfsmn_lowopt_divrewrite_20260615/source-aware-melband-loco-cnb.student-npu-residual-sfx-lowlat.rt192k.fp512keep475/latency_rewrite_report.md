# Latency Rewrite Candidate Report

## ONNX top operators

```text
Mul: 391
Conv: 215
Add: 148
Sigmoid: 116
ReduceMean: 73
Div: 72
Sqrt: 70
Split: 66
Concat: 40
Transpose: 25
Sub: 24
Reshape: 18
Slice: 8
MatMul: 6
ReduceSum: 3
Softmax: 1
```

## ONNX memory operators

```text
Transpose: 25
Reshape: 18
Slice: 8
Split: 66
Concat: 40
```

## ONNX slow/math operators

```text
Div: 72
Sqrt: 70
ReduceMean: 73
ReduceSum: 3
Softmax: 1
Sigmoid: 116
```

## Rewrite candidate counts

```text
nodes: 1276
memory_ops_total: 157
slow_math_ops_total: 335
div_by_static_const: 0
div_by_sqrt: 69
dynamic_or_activation_div: 3
sigmoid_outputs_consumed_by_mul: 116
split_to_sigmoid_gate: 59
cat_slice_state_updates: 8
non_depthwise_grouped_conv: 0
depthwise_grouped_conv: 50
rank_gt4_values: 0
activation_matmul: 6
```

## Circle operators

```text
MUL: 268
CONV_2D: 165
ADD: 148
LOGISTIC: 116
MEAN: 73
RSQRT: 69
SPLIT_V: 66
DEPTHWISE_CONV_2D: 50
TRANSPOSE: 48
CONCATENATION: 40
PAD: 31
SUB: 24
RESHAPE: 18
STRIDED_SLICE: 8
BATCH_MATMUL: 6
DIV: 3
SUM: 3
SOFTMAX: 1
SQRT: 1
```

## Samples

### div_by_sqrt

- `/frontend/frontend.1/Div`
- `/frontend/frontend.2/norm/Div`
- `/norm/Div`
- `/time_norm/Div`
- `/band_norm/Div`
- `/ffn_norm/Div`
- `/cross_band/norm/Div`
- `/norm_1/Div`

### sigmoid_outputs_consumed_by_mul

- `/frontend/frontend.2/Sigmoid`
- `/Sigmoid`
- `/Sigmoid_1`
- `/Sigmoid_2`
- `/Sigmoid_3`
- `/cross_band/Sigmoid`
- `/Sigmoid_4`
- `/pooled_mixer/Sigmoid`

### split_to_sigmoid_gate

- `/frontend/frontend.2/Sigmoid`
- `/Sigmoid_1`
- `/Sigmoid_2`
- `/Sigmoid_3`
- `/cross_band/Sigmoid`
- `/pooled_mixer/Sigmoid`
- `/Sigmoid_5`
- `/Sigmoid_6`

### activation_matmul

- `_MatMul_npu4_mm_5`
- `_MatMul_1_npu4_mm_4`
- `_MatMul_2_npu4_mm_3`
- `_mask_head_expander_MatMul_npu4_mm_2`
- `_mask_head_expander_1_MatMul_npu4_mm_1`
- `_context_expander_MatMul_npu4_mm_0`

### cat_slice_state_updates

- `/Slice`
- `/Slice_1`
- `/Slice_2`
- `/Slice_3`
- `/Slice_4`
- `/Slice_5`
- `/Slice_6`
- `/Slice_7`

### dynamic_or_activation_div

- `/mask_head/expander/Div`
- `/mask_head/expander_1/Div`
- `/context_expander/Div`

## Recommendations

- Most `Div` fed by `Sqrt` are RMSNorm-style reciprocal square-root patterns. Compile with `--low-latency-optimize` so ONE can apply `transform_sqrt_div_to_rsqrt_mul`.
- Dynamic `Div` nodes remain real slow ops. Consider model-level ablations: pre-normalize static bases, use Softmax-normalized weights, disable dynamic renormalization, or approximate with learned scale only if quality allows.
- GLU/Sigmoid gates export as Split/Sigmoid/Mul and are not fused by ONE. Keep quality-critical gates, but try ReLU/ReLU6 or single-branch Conv blocks in low-value FFN/pooled mixers.
- Concat+Slice state updates are streaming-cache memory ops. Reduce layer count/context/state tensors, or fuse multi-branch memory blocks so fewer caches are updated per frame.
- High Transpose/Reshape counts indicate layout or MatMul transport overhead. Use `--low-latency-optimize`, inspect final Circle counts, and reduce source loops/SFC transitions when counts remain high.
- Split/Concat often comes from source loops, GLU gates, branch fusion, or streaming state. Prefer packed-channel vectorization and fewer small parallel branches.
- Keep Softmax only on the last dimension for ONE compatibility; avoid adding source/channel-axis Softmax.
