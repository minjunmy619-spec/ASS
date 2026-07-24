# SFC Small Macaron Conv2D cLN NPU Operation

Date: 2026-07-24

## Purpose

This is a separate normalization experiment based on the faithful Macaron
Conv2D SFC-small variant. It keeps the encoder/decoder cross-attention and the
two-block, 36-band separator, but replaces separator BatchNorm with causal
cumulative LayerNorm at the official pre-normalization sites.

Files:

- `spectral_feature_compression/core/model/sfc_small_macaron_conv2d_cln_npu.py`
- `recipes/dnr/models/sfc-small-macaron-conv2d-cln-npu.musical36.2l.onfly.rt192k/config.yaml`
- `tests/test_sfc_small_macaron_conv2d_cln_npu.py`
- `logs/npu_efficiency_audit/sfc_small_macaron_conv2d_cln_npu_20260724/`

The original BatchNorm variant is unchanged.

## Normalization Semantics

Each frequency and temporal axis path retains:

```text
cLN -> factorized SwiGLU FFN -> residual
cLN -> axis Conv2D mixer     -> residual
cLN -> factorized SwiGLU FFN -> residual
```

Two TF blocks therefore contain twelve independent cLN sites. At frame `t`,
each site computes statistics over channels, all 36 bands, and frames `0..t`.
The streaming recurrence stores bounded running first and second moments:

```text
mean_t   = mean_(t-1)   + alpha_t * (frame_mean_t   - mean_(t-1))
second_t = second_(t-1) + alpha_t * (frame_second_t - second_(t-1))
alpha_t  = 1 / (t + 1)
```

One shared alpha state is updated as:

```text
alpha_(t+1) = alpha_t / (1 + alpha_t)
```

This avoids an unbounded cumulative-sum state.

## NPU Moment Reduction

A direct `ReduceMean` or `ReduceSum` implementation compiled with four
additional transposes per cLN site after ONE's NCHW-to-NHWC conversion.

For the fixed 36-band separator, each frame moment is instead computed with:

```text
fixed 1x1 Conv2D: channel average
AvgPool2D 1x4, stride 1x4: 36 -> 9
AvgPool2D 1x9, stride 1x1: 9 -> 1
```

The pooling strides are supported (`4` and `1`), and both kernel spans are
below the project limit. This decomposition adds no transpose, reshape, slice,
split, or gather operation.

## State ABI

The state contains:

- one shared scalar alpha;
- two scalar moments for each of twelve cLN sites;
- six existing `[B,128,1,36]` temporal Conv2D caches.

This is 31 state tensors. The scalar states have negligible storage cost, but
the larger input/output count is a deployment disadvantage.

```text
one FP16 state set       55,346 bytes
complete FP16 ABI       127,092 bytes
192 KiB headroom         69,516 bytes
```

Invalid state counts are rejected explicitly.
`forward_stream` also rejects multi-frame chunks because the recurrent alpha
must advance once per frame; use `forward` for full training sequences.

## Budget

```text
parameters                         995,190
base Conv/attention MAC per frame 33,145,088
cLN moment Conv MAC per frame        110,592
estimated total MAC per frame     33,255,680
estimated rate                     2.8644 GMAC/s
```

This estimate does not include pooling, RSQRT, or elementwise latency.

## Graph Results

Raw streaming ONNX:

```text
nodes=462
Conv=78
AveragePool=48
Add=65
Mul=88
Sub=48
Sqrt=12
Transpose=2
Reshape=6
inputs=32
outputs=32
```

Optimized NHWC Circle:

```text
nodes=360
CONV_2D=66
DEPTHWISE_CONV_2D=12
AVERAGE_POOL_2D=48
BATCH_MATMUL=4
RSQRT=12
ADD=53
MUL=76
SUB=48
PAD=10
SOFTMAX=2
TRANSPOSE=8
RESHAPE=6
```

The cLN rewrite adds no transpose or reshape beyond the exact SFC
encoder/decoder attention boundary. Compared with the 130-node BN graph, it
adds 230 optimized operators and is therefore expected to have higher NPU
latency despite closer pre-normalization semantics.

## Verification

- cLN versus a direct cumulative-statistics reference: passed;
- full-sequence versus frame-streaming output: passed;
- waveform training forward/backward: finite, no missing gradients;
- focused cLN plus BN-regression tests: 12 passed;
- ONNX checker and edge-NPU operator audit: passed;
- ONE import and NHWC optimization: passed;
- optimized Circle `circle-verify`: passed;
- ONNX Runtime versus float Circle, 32 outputs: max error `1.73e-6`;
- float Circle execution on 32 sequential records: passed;
- UINT8 per-channel quantization: passed;
- quantized Circle `circle-verify`: passed.

The UINT8 graph contains 487 UINT8 and 87 INT32 tensors. Quantized numerical
evaluation remains blocked because this ONE build's `luci-interpreter`
supports only float BatchMatMul. Separation metrics also require a trained
checkpoint.

## Commands

```bash
.venv/bin/python -m pytest -q \
  tests/test_sfc_small_macaron_conv2d_cln_npu.py \
  tests/test_sfc_small_macaron_conv2d_bn_npu.py

.venv/bin/python tools/online/export_onnx_online_model.py \
  recipes/dnr/models/sfc-small-macaron-conv2d-cln-npu.musical36.2l.onfly.rt192k/config.yaml \
  --out logs/npu_efficiency_audit/sfc_small_macaron_conv2d_cln_npu_20260724/stream_rawmask.onnx \
  --seed 2026 --n-chan 1 --frames 1 --freqs 1025 --opset 14 \
  --streaming --disable-masking --check \
  --state-meta-out logs/npu_efficiency_audit/sfc_small_macaron_conv2d_cln_npu_20260724/stream_state_meta.json \
  --op-preset edge_npu_recommended --fail-on-disallowed-ops

one-import-onnx \
  -i logs/npu_efficiency_audit/sfc_small_macaron_conv2d_cln_npu_20260724/stream_rawmask.onnx \
  -o logs/npu_efficiency_audit/sfc_small_macaron_conv2d_cln_npu_20260724/stream_rawmask.circle \
  --keep_io_order

circle2circle \
  logs/npu_efficiency_audit/sfc_small_macaron_conv2d_cln_npu_20260724/stream_rawmask.circle \
  logs/npu_efficiency_audit/sfc_small_macaron_conv2d_cln_npu_20260724/stream_rawmask.nhwc.opt.circle \
  --convert_nchw_to_nhwc --nchw_to_nhwc_input_shape \
  --nchw_to_nhwc_output_shape --forward_transpose_op \
  --fuse_batchnorm_with_conv --fuse_batchnorm_with_dwconv \
  --fuse_activation_function --fuse_rsqrt --remove_duplicate_const \
  --remove_redundant_reshape --remove_redundant_transpose \
  --remove_unnecessary_add --remove_unnecessary_div \
  --remove_unnecessary_mul --remove_unnecessary_reshape \
  --remove_unnecessary_slice --remove_unnecessary_split \
  --remove_unnecessary_strided_slice --remove_unnecessary_transpose \
  --resolve_customop_batchmatmul --resolve_customop_matmul

.venv/bin/python tools/online/prepare_one_streaming_calibration_h5.py \
  recipes/dnr/models/sfc-small-macaron-conv2d-cln-npu.musical36.2l.onfly.rt192k/config.yaml \
  --data-recipe recipes/dnr/models/sfc-small-macaron-conv2d-cln-npu.musical36.2l.onfly.rt192k/config.yaml \
  --source-manifest data/dcase2026_task4_dev_set/manifests/train_sources.csv \
  --out logs/npu_efficiency_audit/sfc_small_macaron_conv2d_cln_npu_20260724/calib_real_sequential_nhwc.h5 \
  --records 32 --mixtures 2 --duration 1.0 --seed 2026 \
  --n-fft 2048 --hop-length 512 --device cpu

one-quantize \
  -i logs/npu_efficiency_audit/sfc_small_macaron_conv2d_cln_npu_20260724/stream_rawmask.nhwc.opt.circle \
  -d logs/npu_efficiency_audit/sfc_small_macaron_conv2d_cln_npu_20260724/calib_real_sequential_nhwc.h5 \
  -f h5 \
  -o logs/npu_efficiency_audit/sfc_small_macaron_conv2d_cln_npu_20260724/stream_rawmask.nhwc.opt.q.circle \
  --input_model_dtype float32 --quantized_dtype uint8 \
  --granularity channel --input_type uint8 --output_type uint8 \
  --mode percentile --min_percentile 0.1 --max_percentile 99.9
```
