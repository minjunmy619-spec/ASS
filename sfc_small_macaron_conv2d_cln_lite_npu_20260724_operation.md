# SFC Small Macaron Conv2D cLN-Lite NPU Operation

Date: 2026-07-24

## Why Full cLN Has Many Nodes

The full-cLN model has twelve independent recurrent normalization sites. ONE
cannot fuse a normalization whose mean and variance are updated through model
state. Each logical cLN therefore expands into:

- two fixed Conv2D moment reductions;
- four AveragePool2D operations;
- running first- and second-moment updates;
- variance calculation and RSQRT;
- centering, scaling, and learned affine operations.

That changes the optimized graph from 130 BatchNorm-variant nodes to 360
full-cLN nodes.

## cLN-Lite Design

The lite variant computes cumulative statistics once at the entrance of each
frequency or temporal axis path. The three sequential sublayers still receive
pre-normalized inputs:

```text
shared statistics -> independent affine -> pre-FFN
shared statistics -> independent affine -> mixer
shared statistics -> independent affine -> post-FFN
```

The residual activation is updated between sublayers, but all three
normalizations use the axis-entry cumulative mean and variance. Their learned
affine parameters remain independent and are implemented as depthwise `1x1`
Conv2D, replacing separate affine Mul and Add nodes.

For two TF blocks this reduces recurrent statistic trackers from twelve to
four. It is an approximation of independent official pre-normalization, but
retains a normalized input at all twelve sublayer boundaries.

Files:

- `spectral_feature_compression/core/model/sfc_small_macaron_conv2d_cln_lite_npu.py`
- `recipes/dnr/models/sfc-small-macaron-conv2d-cln-lite-npu.musical36.2l.onfly.rt192k/config.yaml`
- `tests/test_sfc_small_macaron_conv2d_cln_lite_npu.py`
- `logs/npu_efficiency_audit/sfc_small_macaron_conv2d_cln_lite_npu_20260724/`

## Comparison

| Metric | BatchNorm | Full cLN | cLN-Lite |
| --- | ---: | ---: | ---: |
| Parameters | 1,003,894 | 995,190 | 995,190 |
| Estimated GMAC/s | 2.855 | 2.864 | 2.863 |
| State tensors | 6 | 31 | 15 |
| FP16 complete ABI | 126,992 B | 127,092 B | 127,028 B |
| Optimized Circle nodes | 130 | 360 | 236 |
| AveragePool2D | 0 | 48 | 16 |
| RSQRT | 0 | 12 | 4 |
| Transpose | 8 | 8 | 8 |
| Reshape | 6 | 6 | 6 |

cLN-Lite removes 124 nodes relative to full cLN while adding no layout
operations. It still has 106 more nodes than BatchNorm and is not expected to
beat foldable BatchNorm on raw latency.

## Optimized Circle

```text
nodes=236
ADD=29
AVERAGE_POOL_2D=16
BATCH_MATMUL=4
CONCATENATION=6
CONV_2D=50
DEPTHWISE_CONV_2D=24
DIV=1
LOGISTIC=8
MUL=44
PAD=10
RESHAPE=6
RSQRT=4
SOFTMAX=2
SUB=24
TRANSPOSE=8
```

## Verification

- BN, full-cLN, and cLN-lite regression tests: 16 passed;
- full-sequence versus frame-streaming output: passed;
- invalid state count and multi-frame streaming rejection: passed;
- waveform training backward: finite, no missing gradients;
- ONNX checker and NPU operator audit: passed;
- ONE import, optimization, and Circle verification: passed;
- ONNX Runtime versus float Circle: max error `3.22e-6`;
- UINT8 per-channel quantization and Circle verification: passed;
- UINT8 tensors: 335; INT32 tensors: 83.

Quantized numerical evaluation remains blocked by the current ONE
`luci-interpreter` limitation for quantized BatchMatMul. Separation quality
still requires trained checkpoints for all three normalization variants.

## Commands

The export, import, optimization, calibration, and quantization commands are
the same as the full-cLN operation document, replacing
`sfc-small-macaron-conv2d-cln-npu` with
`sfc-small-macaron-conv2d-cln-lite-npu` and using the corresponding log
directory.
