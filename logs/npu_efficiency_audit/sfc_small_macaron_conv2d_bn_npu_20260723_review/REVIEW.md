# SFC Small Macaron Conv2D BN NPU Review

Date: 2026-07-23

## Scope

Reviewed:

- `spectral_feature_compression/core/model/sfc_small_macaron_conv2d_bn_npu.py`
- the exact SFC encoder/decoder reused from
  `spectral_feature_compression/core/model/sfc_small_pyramid_dw_bn_npu.py`
- `recipes/dnr/models/sfc-small-macaron-conv2d-bn-npu.musical36.2l.onfly.rt192k/config.yaml`
- streaming tests, training integration, ONNX export, ONE import/optimization,
  real sequential calibration, UINT8 quantization, mixed precision, and Circle
  execution.

The review used a fresh deterministic export. Its ONNX, optimized Circle, and
UINT8 Circle hashes match the previous artifacts exactly.

## Findings

### High: no trained-checkpoint quality or quantization accuracy evidence

The model is initialized from the recipe because no checkpoint exists. The H5
uses real on-the-fly mixtures and correct sequential states, but activation
ranges from random weights are not representative of a trained model.

The all-UINT8 graph quantizes and passes `circle-verify`, but ONE's current
`luci-interpreter` only executes float BatchMatMul. `circle-eval-diff` aborts:

```text
luci-intp BatchMatMul(1) Unsupported type.
```

Consequently, this review proves structural compilation, not quantized
separation quality or execution on the target NPU.

### High: the requested parameter floor is not met

The recipe has 1,003,894 parameters, including 636,928 separator parameters.
This is below the requested 2.5M floor. The implementation document explains
the conflict: useful same-band Conv2D weights are evaluated at all 36 bands, so
adding roughly 1.5M such parameters would exceed the 3 GMAC/s ceiling.

This needs an explicit requirement decision; dummy or inactive parameters
would satisfy the count but not improve separation.

### Medium: fidelity is architectural, not semantic

Preserved official SFC ideas:

- exact learnable musical-band position-bias initialization;
- learnable-query encoder compression and decoder expansion;
- frequency path followed by temporal path;
- Macaron `FFN -> mixer -> FFN` residual ordering on both axes;
- same 36-band separator representation.

Changed behavior:

- official pre-RMSNorm is post-Conv BatchNorm;
- official global self-attention is a local depthwise Conv2D mixer;
- encoder/decoder RMSNorm plus SwiGLU FFNs are Conv-BN-ReLU FFNs;
- official temporal convolution context is removed from encoder/decoder and is
  causal in the separator.

The BatchNorm and causal changes are deployment requirements. ReLU in the
encoder/decoder FFNs is an additional quality tradeoff even though SiLU and Mul
already compile elsewhere in this graph.

### Medium: inherited tuning keys are silently ignored

The resolved model config contains both:

```text
freq_kernel_size=3
ffn_expansion=4
frequency_kernel_size=15
ffn_hidden=176
```

The Macaron builder accepts `freq_kernel_size` and `ffn_expansion` but does not
use them. Changing the familiar inherited `sfc_npu_freq_kernel_size` or
`sfc_npu_ffn_expansion` therefore has no effect and produces no error.

### Medium: streaming state ABI is not validated

The core does not check the number of state tensors. Five states fail with an
internal `IndexError`; seven states are accepted and the extra state is
silently discarded. The ABI should reject any count other than six with a
clear error and should validate shape, dtype, and batch where practical.

### Medium: 3 GMAC/s is a narrow MAC estimate, not measured latency

The measured estimate is:

```text
Conv MAC/frame       28,421,888
attention MAC/frame   4,723,200
total MAC/frame      33,145,088
rate                  2.85487965 GMAC/s
headroom              4.84 percent
```

It excludes Softmax, Logistic, Mul, Add, Pad, layout operations, and launch or
memory cost. It therefore cannot establish realtime latency on the target NPU.

### Low: calibration CLI naming is misleading

`prepare_one_streaming_calibration_h5.py --data-recipe` must receive the full
model recipe so `${sr}` and related interpolation keys exist. Passing the
named standalone datamodule recipe fails before the tool can override fields.

### Low: exporter fallback hides the original configuration exception

`build_model_system_from_recipe_config` catches every exception from the
aiaccel/Hydra path and silently retries with a lightweight parser. This can
turn a real recipe error into different fallback behavior and weakens export
reproducibility diagnostics.

## Verification Results

### PyTorch and training integration

```text
focused model tests             18 passed
mixed-precision tool tests       6 passed
full recipe waveform output      [1,3,1,4096]
finite training loss             yes
trainable parameter tensors      206
missing gradients                0
non-finite gradients             0
full versus frame streaming      passed
```

All 54 Conv2D layers satisfy the kernel-span limit. The only kernel forms are
`1x1`, `1x3`, depthwise `1x15`, and causal depthwise `2x1`.

### Streaming ABI

```text
input                 [1,2,1,1025]
six states            [1,128,1,36]
raw-mask output       [1,6,1,1025]
one FP16 state set    55,296 bytes
complete FP16 ABI     126,992 bytes
192 KiB headroom      69,616 bytes
```

### ONNX

```text
nodes       126
Conv         54
Add          16
Mul          16
Sigmoid       8
Relu          6
Concat        6
Reshape       6
MatMul        4
Softmax       2
Transpose     2
```

There are no Resize, Split, Slice, Pad, or ConvTranspose nodes. ONNX checker
and the edge-NPU allowlist pass.

PyTorch versus ONNX Runtime agrees for all seven outputs on a real sequential
record. Maximum absolute error is `4.18e-7`.

### Optimized NHWC Circle

```text
nodes                  130
CONV_2D                 42
DEPTHWISE_CONV_2D       12
BATCH_MATMUL             4
SOFTMAX                  2
TRANSPOSE                8
RESHAPE                  6
PAD                     10
```

All tensors have rank at most four. The separator contributes no transpose or
reshape; those operations remain at the two cross-attention boundaries.

ONNX Runtime versus float Circle agrees for all seven outputs. Maximum
absolute error is `2.98e-7`. Float Circle executes all 64 records.

Adding the available common-subexpression, fusion, removal, and
`substitute_*_to_reshape` flags produces a byte-identical optimized Circle, so
those flags do not reduce this graph further.

Using ONNX `auto_pad=SAME_UPPER` could theoretically avoid explicit Circle Pad,
but the current `one-import-onnx` fails to legalize that Conv form. Explicit
ONNX padding is required by this compiler path.

### Quantization

The all-UINT8 per-channel graph:

```text
UINT8 tensors    195
INT32 tensors     63
operators        130
Circle size      about 1.2 MB
circle-verify    passed
```

The generated top-12 mixed UINT8/INT16 proposal also verifies, but adds 23
Quantize operators:

```text
UINT8 tensors    186
INT16 tensors     32
INT32 tensors     57
INT64 tensors      6
operators        153
```

It is not a latency recommendation without checkpoint-based error data.

## Commands

```bash
.venv/bin/python tools/online/export_onnx_online_model.py \
  recipes/dnr/models/sfc-small-macaron-conv2d-bn-npu.musical36.2l.onfly.rt192k/config.yaml \
  --out logs/npu_efficiency_audit/sfc_small_macaron_conv2d_bn_npu_20260723_review/stream_rawmask.onnx \
  --seed 2026 --n-chan 1 --frames 1 --freqs 1025 --opset 14 \
  --streaming --disable-masking --check \
  --op-preset edge_npu_recommended --fail-on-disallowed-ops

one-import-onnx \
  -i logs/npu_efficiency_audit/sfc_small_macaron_conv2d_bn_npu_20260723_review/stream_rawmask.onnx \
  -o logs/npu_efficiency_audit/sfc_small_macaron_conv2d_bn_npu_20260723_review/stream_rawmask.circle \
  --keep_io_order

circle2circle \
  logs/npu_efficiency_audit/sfc_small_macaron_conv2d_bn_npu_20260723_review/stream_rawmask.circle \
  logs/npu_efficiency_audit/sfc_small_macaron_conv2d_bn_npu_20260723_review/stream_rawmask.nhwc.opt.circle \
  --convert_nchw_to_nhwc --nchw_to_nhwc_input_shape \
  --nchw_to_nhwc_output_shape --forward_transpose_op \
  --fuse_batchnorm_with_conv --fuse_batchnorm_with_dwconv \
  --fuse_activation_function --remove_duplicate_const \
  --remove_redundant_reshape --remove_redundant_transpose \
  --remove_unnecessary_add --remove_unnecessary_reshape \
  --remove_unnecessary_slice --remove_unnecessary_strided_slice \
  --remove_unnecessary_transpose --resolve_customop_batchmatmul \
  --resolve_customop_matmul

.venv/bin/python tools/online/prepare_one_streaming_calibration_h5.py \
  recipes/dnr/models/sfc-small-macaron-conv2d-bn-npu.musical36.2l.onfly.rt192k/config.yaml \
  --data-recipe recipes/dnr/models/sfc-small-macaron-conv2d-bn-npu.musical36.2l.onfly.rt192k/config.yaml \
  --source-manifest data/dcase2026_task4_dev_set/manifests/train_sources.csv \
  --out logs/npu_efficiency_audit/sfc_small_macaron_conv2d_bn_npu_20260723_review/calib_real_sequential_nhwc.h5 \
  --records 64 --mixtures 4 --duration 1.0 --seed 2026 \
  --n-fft 2048 --hop-length 512 --device cpu

one-quantize \
  -i logs/npu_efficiency_audit/sfc_small_macaron_conv2d_bn_npu_20260723_review/stream_rawmask.nhwc.opt.circle \
  -d logs/npu_efficiency_audit/sfc_small_macaron_conv2d_bn_npu_20260723_review/calib_real_sequential_nhwc.h5 \
  -f h5 \
  -o logs/npu_efficiency_audit/sfc_small_macaron_conv2d_bn_npu_20260723_review/stream_rawmask.nhwc.opt.q.circle \
  --input_model_dtype float32 --quantized_dtype uint8 \
  --granularity channel --input_type uint8 --output_type uint8 \
  --mode percentile --min_percentile 0.1 --max_percentile 99.9
```
