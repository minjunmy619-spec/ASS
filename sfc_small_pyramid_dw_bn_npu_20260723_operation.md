# SFC Small Pyramid Depthwise BN NPU Operation

Date: 2026-07-23

> This is retained as a frequency-pyramid ablation. It is not the strict
> same-band SFC separator. The corrected variant is documented in
> `sfc_small_sameband_dw_bn_npu_20260723_operation.md`.

## Goal

Build a trainable SFC-small variant that:

- preserves the official SFC encoder/decoder frequency-compression semantics;
- is causal at the waveform and separator levels;
- keeps feature tensors in `[B, C, T, F]` through the main pipeline;
- replaces the Locoformer separator blocks with NPU-friendly Conv2D blocks;
- has 3-4 million parameters and less than 3 GMAC/s at 44.1 kHz;
- exports through ONNX and ONE to an optimized and calibrated UINT8 Circle graph.

The implementation is separate from the previous variants:

- model: `spectral_feature_compression/core/model/sfc_small_pyramid_dw_bn_npu.py`
- recipe: `recipes/dnr/models/sfc-small-pyramid-dw-bn-npu.musical64.onfly.rt192k/config.yaml`
- tests: `tests/test_sfc_small_pyramid_dw_bn_npu.py`
- calibration: `tools/online/prepare_one_streaming_calibration_h5.py`

## SFC Fidelity

The encoder keeps the official SFC operation:

1. Project packed real/imaginary STFT bins to an inner feature dimension.
2. Generate one learnable query per musical band and attention head.
3. Project full-resolution frequency features into separate keys and values.
4. Add the official `gentle_slope` musical-band position bias to the
   query/key scores.
5. Normalize over the original frequency bins and aggregate them into 64
   learned bands.

The position bias is exact, including the official integer center and
denominator:

```text
center = (start + end) // 2
denominator = (end - start) // 2 + 1
inside band: -abs(center - frequency) / denominator
outside band: signed distance to the nearest band boundary
```

The decoder performs the inverse SFC mapping with the transposed official
position bias: 64 band tokens are attended into all 1025 output frequency
bins. Tests compare both tensors directly with
`prepare_bandit_position_bias(..., n_heads=4)` using zero tolerance.

Key and value projections remain separate. The learnable query already
contains the attention scale, so the exported graph has no KV split slices and
no runtime scale multiplication.

The separator is deliberately not an exact Locoformer implementation. It
preserves the residual frequency-mixing, causal time-mixing, and channel-FFN
roles, but realizes them with depthwise and pointwise Conv2D at a compressed
frequency rate. This is the necessary deployment approximation.

## Causal Path

The builder returns `OnlineModelWrapper`, whose STFT uses `center=False`,
forbids utterance-global input scaling, and reconstructs with causal
overlap-add. The external frontend needs `n_fft - hop_length = 1536` past
waveform samples before each new analysis frame.

Each separator block has an explicit one-frame temporal state. The deployment
recipe uses `time_kernel_size=2` and `dilation_cycle=[1]`, so:

- every output depends only on the current and previous encoded frame;
- eight serial blocks give a nine-frame separator receptive field;
- each exported next state is the current input tensor directly;
- the graph needs no `SLICE` or `STRIDED_SLICE` for state updates.

Full-sequence and frame-by-frame evaluation agree within `1e-5`; the observed
small-model maximum error was approximately `1.8e-7`.

## Shape Plan

The persistent activation layout is `[B, C, T, F]`. Time remains axis 2 and
frequency remains axis 3 from packed STFT input through mask output.

| Stage | Shape for one streaming frame | Operation |
| --- | --- | --- |
| Packed STFT | `[1, 2, 1, 1025]` | real/imaginary channels |
| SFC input projection | `[1, 32, 1, 1025]` | Conv2D |
| SFC encoder output | `[1, 64, 1, 64]` | cross-attention |
| Pyramid level 1 | `[1, 96, 1, 32]` | stride-2 frequency Conv2D |
| Pyramid level 2 | `[1, 128, 1, 16]` | stride-2 frequency Conv2D |
| Pyramid level 3 | `[1, 192, 1, 8]` | stride-2 frequency Conv2D |
| Pyramid level 4 | `[1, 256, 1, 4]` | stride-2 frequency Conv2D |
| Eight separator blocks | `[1, 256, 1, 4]` | depthwise TF Conv2D + pointwise FFN |
| Additive pyramid decoder | `[1, 64, 1, 64]` | nearest resize + Conv2D |
| SFC decoder output | `[1, 6, 1, 1025]` | inverse cross-attention |

The pyramid uses additive skips, not concatenation, on the decode path. The
eight Circle `CONCATENATION` nodes are only temporal state plus current-frame
joins. No layout transform is used inside the separator.

Attention alone needs head packing for `BATCH_MATMUL`. The optimized graph has
eight `TRANSPOSE` nodes, four around each encoder/decoder attention boundary.
Using

```text
(weight @ value.T).T = value @ weight.T
```

removes one value transpose from each attention module compared with the
earlier KV-split graph.

## Normalization

BatchNorm2D remains the deployment choice. During training it supplies
per-channel normalization; during export ONE folds its frozen affine transform
and running statistics into adjacent Conv2D or depthwise Conv2D weights.
Consequently, the optimized Circle graph contains no normalization operators.

Cumulative LayerNorm is causal, but it is not preferable for this target. It
would require runtime reductions, accumulation state, subtraction, multiply,
division or reciprocal square root, and additional reshapes. Those operations
cannot be folded into Conv2D and increase both graph and state traffic.

BatchNorm2D is not mathematically identical to RMSNorm or cumulative
LayerNorm. The retained semantic unit is the normalized residual Conv block,
while the normalization statistic is intentionally changed for NPU latency.
Training from scratch is required; weights from a differently normalized
separator should not be treated as directly equivalent.

## Budget

Default recipe measurements:

| Metric | Result |
| --- | ---: |
| Parameters | 3,279,510 |
| MAC per streaming frame | 30,135,040 |
| MAC/s at 44.1 kHz, hop 512 | 2.596 GMAC/s |
| One FP16 separator state set | 16,384 bytes |
| FP16 input + output states + frame I/O | 49,168 bytes |
| DSP ABI limit | 196,608 bytes |

The pointwise FFNs hold most useful capacity at the four-bin bottleneck, where
their frequency multiplier is 16 times smaller than at the original 64-band
rate. The encoder, decoder, and skip projections retain enough channel width
to avoid making SFC itself the narrowest representation.

## Verification

Run the model tests:

```bash
.venv/bin/python -m pytest \
  tests/test_sfc_small_conv2d_bn_npu.py \
  tests/test_sfc_small_pyramid_dw_bn_npu.py -q
```

Result:

```text
16 passed
```

Export the deterministic config-only streaming raw-mask graph:

```bash
OUT=logs/npu_efficiency_audit/sfc_small_pyramid_dw_bn_npu_20260723
RECIPE=recipes/dnr/models/sfc-small-pyramid-dw-bn-npu.musical64.onfly.rt192k/config.yaml

.venv/bin/python tools/online/export_onnx_online_model.py "$RECIPE" \
  --out "$OUT/stream_rawmask.onnx" \
  --seed 2026 \
  --n-chan 1 \
  --frames 1 \
  --freqs 1025 \
  --opset 14 \
  --streaming \
  --disable-masking \
  --check \
  --state-meta-out "$OUT/stream_rawmask_state.json"
```

The ONNX operator summary is:

```text
Conv=54 Add=32 Relu=30 Constant=14 Concat=8
Reshape=6 MatMul=4 Resize=4 Softmax=2 Transpose=2
Slice=0 Pad=0
```

Import and optimize with the local ONE build:

```bash
ONE=/home/cmj/works/ONE/build/compiler/one-cmds

"$ONE/one-import-onnx" \
  -i "$OUT/stream_rawmask.onnx" \
  -o "$OUT/stream_rawmask.circle" \
  --keep_io_order

"$ONE/circle2circle" \
  "$OUT/stream_rawmask.circle" \
  "$OUT/stream_rawmask.nhwc.opt.circle" \
  --convert_nchw_to_nhwc \
  --nchw_to_nhwc_input_shape \
  --nchw_to_nhwc_output_shape \
  --forward_transpose_op \
  --fuse_batchnorm_with_conv \
  --fuse_batchnorm_with_dwconv \
  --fuse_activation_function \
  --remove_duplicate_const \
  --remove_redundant_reshape \
  --remove_redundant_transpose \
  --remove_unnecessary_add \
  --remove_unnecessary_reshape \
  --remove_unnecessary_slice \
  --remove_unnecessary_strided_slice \
  --remove_unnecessary_transpose \
  --resolve_customop_batchmatmul \
  --resolve_customop_matmul
```

The optimized Circle graph has 138 executable nodes:

```text
ADD=32
BATCH_MATMUL=4
CONCATENATION=8
CONV_2D=38
DEPTHWISE_CONV_2D=16
PAD=20
RESHAPE=6
RESIZE_NEAREST_NEIGHBOR=4
SOFTMAX=2
TRANSPOSE=8
SLICE=0
STRIDED_SLICE=0
```

An ONNX `auto_pad=SAME_UPPER` experiment removed explicit ONNX padding, but
the current ONE ONNX importer failed to legalize that Conv form. Explicit
PyTorch padding is therefore retained. ONE lowers it to 20 Circle `PAD` nodes;
this is preferable to an unimportable graph.

## Sequential Calibration and Quantization

Generate calibration tensors from real sequential on-the-fly mixtures:

```bash
.venv/bin/python tools/online/prepare_one_streaming_calibration_h5.py \
  "$RECIPE" \
  --data-recipe "$RECIPE" \
  --source-manifest data/dcase2026_task4_dev_set/manifests/train_sources.csv \
  --out "$OUT/calib_real_sequential_nhwc.h5" \
  --records 64 \
  --mixtures 4 \
  --duration 1.0 \
  --warmup-frames 4 \
  --seed 2026
```

This synthesizes four one-second mixtures with the existing on-the-fly stem
pipeline, advances every state sequentially, and records 64 current-frame/state
snapshots in the NHWC ABI expected by the optimized graph:

```text
x:       [1, 1, 1025, 2]
state_*: [1, 1, 4, 256] for eight states
```

Quantize the exact optimized KV-split graph:

```bash
"$ONE/one-quantize" \
  -i "$OUT/stream_rawmask.nhwc.opt.circle" \
  -d "$OUT/calib_real_sequential_nhwc.h5" \
  -f h5 \
  -o "$OUT/stream_rawmask.nhwc.opt.q.circle" \
  --quantized_dtype uint8 \
  --granularity channel \
  --input_type uint8 \
  --output_type uint8 \
  --mode percentile \
  --min_percentile 0.1 \
  --max_percentile 99.9 \
  --moving_avg_batch 16 \
  --moving_avg_const 0.1
```

Quantization succeeds. The output shrinks from approximately 13.1 MB to
3.62 MB and contains 205 UINT8 and 66 INT32 tensors. Its operator topology is
unchanged, including quantized Conv2D, depthwise Conv2D, BatchMatMul, Resize,
and Softmax.

## Quality Evaluation Status

There is no trained checkpoint yet. The current `--seed 2026` model and
calibration artifacts validate deterministic structure, state evolution,
import, optimization, and quantization only. Separation metrics from random
weights would be meaningless.

ONE's `circle-eval-diff` also cannot execute this quantized attention graph in
the current checkout:

```text
luci-intp BatchMatMul(1) Unsupported type.
```

After training, repeat export and calibration with the same trained directory
or checkpoint as the positional `model_path`. Then compare float and quantized
streaming waveform outputs using the target NPU runtime, or an onert backend
that supports quantized `BATCH_MATMUL`, and report at least SI-SDR/SDR per
Speech, Music, and Effects stem plus float-to-quant output error. The
calibration tool already accepts the trained artifact without changing the
pipeline.

## Produced Artifacts

All compiler artifacts are under:

`logs/npu_efficiency_audit/sfc_small_pyramid_dw_bn_npu_20260723/`

Important files:

- `stream_rawmask.onnx`
- `stream_rawmask.circle`
- `stream_rawmask.nhwc.opt.circle`
- `calib_real_sequential_nhwc.h5`
- `calib_real_sequential_nhwc.h5.json`
- `stream_rawmask.nhwc.opt.q.circle`
- `stream_rawmask.nhwc.opt.qdq.circle`
- `float_vs_uint8_eval.txt`
