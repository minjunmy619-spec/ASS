# SFC Small Macaron Conv2D BN NPU Operation

Date: 2026-07-23

## Purpose

Implement an NPU-oriented SFC separator that preserves the official
TF-Locoformer block skeleton instead of reducing each TF block to one
frequency convolution, one temporal convolution, and one FFN.

Files:

- `spectral_feature_compression/core/model/sfc_small_macaron_conv2d_bn_npu.py`
- `recipes/dnr/models/sfc-small-macaron-conv2d-bn-npu.musical36.2l.onfly.rt192k/config.yaml`
- `tests/test_sfc_small_macaron_conv2d_bn_npu.py`
- `logs/npu_efficiency_audit/sfc_small_macaron_conv2d_bn_npu_20260723/`

## Official Skeleton

The official block processes frequency before time. Each axis path is
Macaron-style:

```text
frequency: FFN -> self-attention -> FFN
time:      FFN -> self-attention -> FFN
```

All three branches in each axis path are residual. Therefore, one official TF
block contains four independent FFNs and two axis mixers.

The NPU rewrite retains:

```text
frequency: factorized SwiGLU FFN -> frequency Conv2D mixer -> factorized SwiGLU FFN
time:      causal factorized SwiGLU FFN -> causal temporal Conv2D mixer
           -> causal factorized SwiGLU FFN
```

Two TF blocks produce eight independent SwiGLU FFNs, four frequency branches,
four temporal branches, two frequency mixers, and two temporal mixers.

## Necessary Operator Replacements

| Official operation | NPU rewrite |
| --- | --- |
| RMSGroupNorm pre-norm | BatchNorm2D attached to Conv2D |
| Linear QKV self-attention | depthwise axis Conv2D + pointwise Conv2D |
| Conv1D `C -> 2H`, kernel 8 | depthwise axis Conv2D + separate value/gate Conv2D |
| tensor split | separate value and gate projections |
| SwiGLU | `value * SiLU(gate)` |
| ConvTranspose1D `H -> C` | pointwise Conv2D `H -> C` |
| frame/frequency permutations | direct kernels on `[B,C,T,F]` |

The separate value/gate projections are algebraically equivalent to splitting
one `C -> 2H` projection and avoid exported Split or Slice operators.

The original Conv1D/ConvTranspose1D pair has an effective 15-cell receptive
field. Frequency FFNs and mixers therefore use the maximum legal `1x15`
depthwise kernel, satisfying `(15 - 1) * 1 <= 14`. Temporal branches use
causal `2x1` depthwise kernels so every next state is the current branch input
and no state slice is needed.

## Normalization

ONE provides `fuse_preactivation_batchnorm`, but its current pass recognizes a
narrow affine-plus-ReLU pattern, requires suitable positive scale, and cannot
reliably fuse one shared normalization output that fans out to both SwiGLU
projections.

The rewrite therefore places BatchNorm2D after each Conv2D/depthwise Conv2D.
This changes exact pre-norm mathematics but lets
`fuse_batchnorm_with_conv` and `fuse_batchnorm_with_dwconv` remove every
normalization node from the optimized Circle graph.

## Shape and State

The feature layout stays `[B,C,T,F]` throughout:

```text
[B,2,1,1025]
  -> exact SFC encoder
[B,128,1,36]
  -> TF Macaron block 1
[B,128,1,36]
  -> TF Macaron block 2
[B,128,1,36]
  -> exact SFC decoder
[B,6,1,1025]
```

There are no separator transposes, reshapes, resizes, pools, or frequency
down/up-sampling operations.

Each temporal path has three causal states:

1. temporal pre-FFN depthwise input;
2. temporal mixer depthwise input;
3. temporal post-FFN depthwise input.

Two blocks therefore expose six states of `[1,128,1,36]`, or NHWC
`[1,1,36,128]`.

## Budget

| Metric | Result |
| --- | ---: |
| Parameters | 1,003,894 |
| Separator parameters | 636,928 |
| Conv2D MAC/frame | 28,421,888 |
| SFC attention MAC/frame | 4,723,200 |
| Total MAC/frame | 33,145,088 |
| MAC/s at 44.1 kHz, hop 512 | 2.855 GMAC/s |
| One FP16 state set | 55,296 bytes |
| Complete FP16 streaming ABI | 126,992 bytes |

The parameter count remains below 2.5M because every ordinary separator
weight is evaluated at all 36 bands. Expanding the faithful dense branches to
2.5M useful parameters would exceed 7 GMAC/s. No inactive, pooled, or dummy
parameter branch is added.

## Tests

```bash
.venv/bin/python -m pytest \
  tests/test_sfc_small_macaron_conv2d_bn_npu.py -q
```

The tests verify:

- exact official encoder and decoder position biases;
- frequency path followed by temporal path;
- two FFNs around every axis mixer;
- four frequency and four temporal FFNs;
- unchanged SFC band count through every block;
- full-sequence and streaming equivalence;
- compute and state budgets;
- raw ONNX without Resize, ConvTranspose, Split, Slice, or Pad;
- eight exported SwiGLU gates.

## ONNX and Circle

Raw streaming ONNX:

```text
Conv=54
Add=16
Mul=16
Sigmoid=8
Relu=6
Concat=6
Reshape=6
MatMul=4
Softmax=2
Transpose=2
Resize=0 Split=0 Slice=0 Pad=0
```

Optimized NHWC Circle:

```text
ADD=16
BATCH_MATMUL=4
CONCATENATION=6
CONV_2D=42
DEPTHWISE_CONV_2D=12
LOGISTIC=8
MUL=16
PAD=10
RESHAPE=6
SOFTMAX=2
TRANSPOSE=8
total=130
```

The eight transposes and six reshapes are confined to the exact SFC
encoder/decoder cross-attention. The separator contributes none.

## Calibration and Quantization

The calibration file contains 64 sequential frame/state records from four
real one-second on-the-fly mixtures:

```text
x:       [1,1,1025,2] NHWC
state_*: [1,1,36,128] NHWC, six states
```

UINT8 per-channel quantization succeeds:

```text
UINT8 tensors: 195
INT32 tensors: 63
float Circle: approximately 3.9 MB
UINT8 Circle: approximately 1.2 MB
```

Imported, optimized, and quantized artifacts all pass `circle-verify`.
Separation metrics remain pending until a trained checkpoint exists.
