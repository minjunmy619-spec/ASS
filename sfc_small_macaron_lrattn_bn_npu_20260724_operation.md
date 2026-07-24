# SFC Small Macaron Low-Rank Attention BN NPU Operation

Date: 2026-07-24

## Purpose

Replace the fixed local depthwise mixer in the BatchNorm SFC Macaron model
with a content-dependent global mixer while preserving the NPU constraints:

- exact SFC encoder and decoder with official position bias;
- frequency path followed by temporal path;
- independent `FFN -> attention -> FFN` residual branches on each axis;
- one stable `[B,C,T,F]` separator layout;
- no separator Softmax, BatchMatMul, Transpose, Reshape, Split, or Slice;
- strict one-frame streaming inference;
- less than 3 GMAC/s and 192 KiB complete FP16 ABI;
- more than 2.5M useful parameters.

Files:

- `spectral_feature_compression/core/model/sfc_small_macaron_lrattn_bn_npu.py`
- `recipes/dnr/models/sfc-small-macaron-lrattn-bn-npu.musical36.2l.r2d64g560.onfly.rt192k/config.yaml`
- `tests/test_sfc_small_macaron_lrattn_bn_npu.py`
- `logs/npu_efficiency_audit/sfc_small_macaron_lrattn_bn_npu_20260724/`

## Low-Rank Attention

For each rank `r`, pointwise Conv2D produces scalar query and key gates and a
shared 64-channel value:

```text
q_r = sigmoid(BN(Conv2D_1x1(x)))       [B,1,T,F]
k_r = sigmoid(BN(Conv2D_1x1(x)))       [B,1,T,F]
v   = BN(Conv2D_1x1(x))                [B,64,T,F]
```

The unnormalized separable attention numerator is:

```text
sum_r q_r(i) * context_r
```

This is a rank-2, content-dependent query/key interaction. It is not an exact
softmax attention matrix, but it restores dynamic global mixing that a fixed
depthwise kernel cannot represent.

### Frequency Attention

Every compressed band directly receives information from all 36 bands:

```text
k_r * v
  -> AvgPool 36 -> 9
  -> AvgPool 9 -> 1
  -> Conv2D 64 -> 560 -> 560 -> 64
  -> broadcast multiply by q_r
```

The two pooling stages obey the ONE kernel and stride constraints. Binary
broadcasting expands `[B,64,T,1]` over frequency without Tile, Resize, or
explicit Expand.

Each rank has an independent global context MLP. Its large matrices run only
at one frequency position, so they add useful nonlinear global capacity
without paying the 36-band compute multiplier.

### Temporal Attention

Each temporal rank maintains one `[B,64,1,36]` recurrent context:

```text
current_r(t) = k_r(t) * v(t)
context_r(t) = current_r(t) + 0.995 * (context_r(t-1) - current_r(t))
output_r(t) = q_r(t) * context_r(t)
```

The decay gives an effective context of about 200 frames, or 2.32 seconds at
44.1 kHz with hop 512. Every older frame has a nonzero exponentially decaying
contribution. This avoids the deployment failure mode of an unbounded
cumulative mean, whose update becomes effectively frozen after a long stream.

The training `forward` evaluates the same recurrence in a vectorized form.
Frame-by-frame `forward_stream` and full-sequence evaluation match.

## Capacity Distribution

The dense Macaron FFNs still operate at all 36 bands. Additional parameters
are therefore placed in the pooled frequency context:

```text
2 blocks * 2 ranks * (64 -> 560 -> 560 -> 64)
```

This contributes about 1.56M trainable parameters but only about 1.54M
Conv2D MAC per frame.

| Metric | Fixed Conv BN | Low-rank attention BN |
| --- | ---: | ---: |
| Parameters | 1,003,894 | 2,556,198 |
| Estimated GMAC/s | 2.855 | 2.9805 |
| State tensors | 6 | 8 |
| FP16 state bytes | 55,296 | 55,296 |
| Complete FP16 ABI | 126,992 | 126,992 |
| Optimized Circle nodes | 130 | 210 |

The byte-size state is unchanged:

```text
fixed mixer per block: pre 128 + mixer 128 + post 128
low-rank per block:    pre 128 + rank0 64 + rank1 64 + post 128
```

## Graph Results

Raw streaming ONNX:

```text
nodes=216
inputs=9 outputs=9
Add=24
AveragePool=8
Concat=4
Constant=10
Conv=82
MatMul=4
Mul=36
Relu=10
Reshape=6
Sigmoid=24
Softmax=2
Sub=4
Transpose=2
```

Optimized NHWC Circle:

```text
nodes=210
ADD=24
AVERAGE_POOL_2D=8
BATCH_MATMUL=4
CONCATENATION=4
CONV_2D=74
DEPTHWISE_CONV_2D=8
LOGISTIC=24
MUL=36
PAD=8
RESHAPE=6
SOFTMAX=2
SUB=4
TRANSPOSE=8
```

The remaining four BatchMatMul, two Softmax, eight Transpose, and six Reshape
operators belong to the exact SFC encoder and decoder. The separator adds
none of them.

## Verification

- new and fixed-BN tests: 16 passed;
- combined BN, cLN-lite, full-cLN, and low-rank tests: 25 passed;
- exact encoder and decoder position biases: passed;
- distant frequency-band gradient connectivity: passed;
- temporal first-to-last-frame gradient connectivity: passed;
- full-sequence versus streaming output: passed;
- waveform training backward: 318 trainable tensors, no missing or nonfinite
  gradients;
- ONNX checker and edge-NPU operator audit: passed;
- PyTorch versus ONNX Runtime across all nine outputs: maximum absolute error
  `2.83e-7`;
- ONE import, NHWC optimization, and both float Circle verifications: passed;
- ONNX Runtime versus optimized NHWC Circle across all nine outputs: maximum
  absolute error `2.98e-7`, maximum MAE `3.57e-8`;
- 32-record real sequential on-the-fly calibration: passed;
- UINT8 per-channel quantization and quantized Circle verification: passed;
- quantized tensors: 309 UINT8 and 91 INT32;
- optimized float Circle: approximately 9.8 MB;
- quantized Circle: approximately 2.8 MB.

The export and calibration use deterministic config-only initialization.
They prove structural compilation and quantization, not separation quality.
Float-versus-quantized separation metrics require a trained checkpoint. The
current ONE interpreter also cannot numerically execute the quantized
BatchMatMul operations retained by the exact SFC encoder and decoder.

## Semantic Limitations

The mixer is closer to official self-attention than fixed depthwise Conv2D,
but it is still an approximation:

- rank is two rather than a full attention matrix;
- query and key use bounded sigmoid features;
- there is no softmax key normalization;
- temporal context uses exponential decay instead of arbitrary pairwise access
  to every cached frame;
- BatchNorm2D remains a foldable post-convolution approximation of official
  RMSGroupNorm pre-normalization.

These choices remove separator quadratic attention and layout transport while
retaining direct global frequency access, query/key content dependence, and
long causal temporal memory.

## Commands

```bash
.venv/bin/python -m pytest -q \
  tests/test_sfc_small_macaron_lrattn_bn_npu.py \
  tests/test_sfc_small_macaron_conv2d_bn_npu.py \
  tests/test_sfc_small_macaron_conv2d_cln_lite_npu.py \
  tests/test_sfc_small_macaron_conv2d_cln_npu.py

.venv/bin/python tools/online/export_onnx_online_model.py \
  recipes/dnr/models/sfc-small-macaron-lrattn-bn-npu.musical36.2l.r2d64g560.onfly.rt192k/config.yaml \
  --out logs/npu_efficiency_audit/sfc_small_macaron_lrattn_bn_npu_20260724/stream_rawmask.onnx \
  --seed 2026 --n-chan 1 --frames 1 --freqs 1025 --opset 14 \
  --streaming --disable-masking --check \
  --state-meta-out logs/npu_efficiency_audit/sfc_small_macaron_lrattn_bn_npu_20260724/stream_state_meta.json \
  --deploy-manifest-out logs/npu_efficiency_audit/sfc_small_macaron_lrattn_bn_npu_20260724/deploy_manifest.json \
  --op-preset edge_npu_recommended --fail-on-disallowed-ops

one-import-onnx \
  -i logs/npu_efficiency_audit/sfc_small_macaron_lrattn_bn_npu_20260724/stream_rawmask.onnx \
  -o logs/npu_efficiency_audit/sfc_small_macaron_lrattn_bn_npu_20260724/stream_rawmask.circle \
  --keep_io_order

circle2circle \
  logs/npu_efficiency_audit/sfc_small_macaron_lrattn_bn_npu_20260724/stream_rawmask.circle \
  logs/npu_efficiency_audit/sfc_small_macaron_lrattn_bn_npu_20260724/stream_rawmask.nhwc.opt.circle \
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
  recipes/dnr/models/sfc-small-macaron-lrattn-bn-npu.musical36.2l.r2d64g560.onfly.rt192k/config.yaml \
  --data-recipe recipes/dnr/models/sfc-small-macaron-lrattn-bn-npu.musical36.2l.r2d64g560.onfly.rt192k/config.yaml \
  --source-manifest data/dcase2026_task4_dev_set/manifests/train_sources.csv \
  --out logs/npu_efficiency_audit/sfc_small_macaron_lrattn_bn_npu_20260724/calib_real_sequential_nhwc.h5 \
  --records 32 --mixtures 2 --duration 1.0 --seed 2026 \
  --n-fft 2048 --hop-length 512 --device cpu

one-quantize \
  -i logs/npu_efficiency_audit/sfc_small_macaron_lrattn_bn_npu_20260724/stream_rawmask.nhwc.opt.circle \
  -d logs/npu_efficiency_audit/sfc_small_macaron_lrattn_bn_npu_20260724/calib_real_sequential_nhwc.h5 \
  -f h5 \
  -o logs/npu_efficiency_audit/sfc_small_macaron_lrattn_bn_npu_20260724/stream_rawmask.nhwc.opt.q.circle \
  --input_model_dtype float32 --quantized_dtype uint8 \
  --granularity channel --input_type uint8 --output_type uint8 \
  --mode percentile --min_percentile 0.1 --max_percentile 99.9
```
