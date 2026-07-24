# SFC Small Same-Band Depthwise BN NPU Operation

Date: 2026-07-23

## Correction

The previous pyramid separator changed the 64 learned SFC bands through:

```text
64 -> 32 -> 16 -> 8 -> 4 -> 8 -> 16 -> 32 -> 64
```

That was a useful low-compute experiment, but it is not the official SFC
separator topology. The corrected variant keeps the frequency-band axis fixed:

```text
STFT 1025 bins
  -> exact SFC cross-attention encoder
64 learned musical bands
  -> eight same-band Conv2D separator blocks
64 learned musical bands
  -> exact SFC cross-attention decoder
STFT 1025 bins
```

There is no resize, frequency-stride convolution, pooling, transposed
convolution, or secondary band compression inside the separator.

## Files

- Model:
  `spectral_feature_compression/core/model/sfc_small_sameband_dw_bn_npu.py`
- Recipe:
  `recipes/dnr/models/sfc-small-sameband-dw-bn-npu.musical64.onfly.rt192k/config.yaml`
- Tests:
  `tests/test_sfc_small_sameband_dw_bn_npu.py`
- Compiler artifacts:
  `logs/npu_efficiency_audit/sfc_small_sameband_dw_bn_npu_20260723/`

The former pyramid implementation remains available as a named ablation and
is not silently overwritten.

## Separator

Every block receives and returns `[B, 80, T, 64]`:

1. Depthwise `1x3` frequency Conv2D, BatchNorm2D, ReLU, residual add.
2. Depthwise causal `2x1` temporal Conv2D, BatchNorm2D, ReLU, residual add.
3. Pointwise channel FFN `80 -> 240 -> 80`, BatchNorm2D, ReLU, residual add.

BatchNorm2D is folded into Conv2D by ONE. The optimized Circle graph has no
normalization operator. The deployment dilation is one, so each temporal block
stores one previous `[1, 80, 1, 64]` activation and exports the current
activation directly as its next state. No state slice is required.

The eight serial temporal convolutions provide a nine-frame receptive field.
They are local causal replacements for temporal Locoformer modeling. The
frequency depthwise path preserves band identity while mixing neighboring
musical bands. The pointwise FFN mixes channels independently at each of the
same 64 band positions.

## Fidelity

The corrected model preserves these official SFC properties:

- one learnable query per musical band in the encoder;
- exact official `gentle_slope` position bias;
- separate key and value projections without runtime split;
- query-folded attention scale;
- one 1025-to-64 SFC compression;
- an unchanged 64-band separator representation;
- one 64-to-1025 SFC decoder expansion using the transposed official bias.

The separator still replaces TF-Locoformer attention and SwiGLU blocks with
Conv2D for NPU deployment. It is therefore faithful to the SFC compression
topology, but not mathematically identical to Locoformer.

## Budget and Parameter Tradeoff

| Metric | Corrected same-band result |
| --- | ---: |
| Parameters | 910,022 |
| Conv2D MAC/frame | 26,292,992 |
| SFC attention MAC/frame | 8,396,800 |
| Total MAC/frame | 34,689,792 |
| MAC/s at 44.1 kHz, hop 512 | 2.988 GMAC/s |
| One FP16 state set | 81,920 bytes |
| FP16 input + output states + frame I/O | 180,240 bytes |
| DSP ABI limit | 196,608 bytes |

The original 3-4M parameter target conflicts with strict same-band dense
Conv2D under the 3 GMAC/s limit. A separator weight used at all 64 bands incurs
64 MACs per output frame. The current graph already uses 99.6% of the MAC
budget and 91.7% of the ABI budget.

Adding unused parameters, duplicating head-specific position-bias tables, or
enlarging the slow Softmax tensors would satisfy the parameter count only
nominally and would not be an effective model. This variant therefore uses
0.91M active parameters rather than manufacturing a misleading 3-4M count.
The pyramid ablation remains the 3.28M alternative when the parameter target
is more important than strict same-band fidelity.

## PyTorch and ONNX Validation

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_sfc_small_conv2d_bn_npu.py \
  tests/test_sfc_small_pyramid_dw_bn_npu.py \
  tests/test_sfc_small_sameband_dw_bn_npu.py -q
```

The same-band tests verify:

- exact encoder and decoder position biases;
- every separator block retains all 64 bands;
- full and frame-streaming outputs agree;
- compute and state budgets;
- causal `center=False` waveform wrapping;
- no Resize, ConvTranspose, Slice, or Pad in raw streaming ONNX.

Export:

```bash
OUT=logs/npu_efficiency_audit/sfc_small_sameband_dw_bn_npu_20260723
RECIPE=recipes/dnr/models/sfc-small-sameband-dw-bn-npu.musical64.onfly.rt192k/config.yaml

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
  --state-meta-out "$OUT/stream_rawmask_state.json" \
  --op-preset edge_npu_recommended \
  --fail-on-disallowed-ops
```

Raw ONNX operators:

```text
Conv=46 Add=28 Relu=26 Concat=8 Constant=6
Reshape=6 MatMul=4 Softmax=2 Transpose=2
Resize=0 ConvTranspose=0 Slice=0 Pad=0
```

## ONE Result

The same optimization flags as the pyramid experiment were applied, including
NHWC conversion, BatchNorm fusion, redundant layout removal, and
BatchMatMul/MatMul resolution.

Optimized Circle operators:

```text
ADD=28
BATCH_MATMUL=4
CONCATENATION=8
CONV_2D=30
DEPTHWISE_CONV_2D=16
PAD=12
RESHAPE=6
SOFTMAX=2
TRANSPOSE=8
RESIZE_NEAREST_NEIGHBOR=0
SLICE=0
STRIDED_SLICE=0
total=114
```

The eight remaining transposes and six reshapes belong to exact encoder and
decoder cross-attention head packing. The separator introduces none.

The imported, optimized, and UINT8 Circle files all pass `circle-verify`.

## Calibration and Quantization

Calibration uses four real one-second mixtures synthesized by the existing
on-the-fly stem dataset. It advances all eight states sequentially and records
64 frame/state snapshots:

```text
x:       [1, 1, 1025, 2] NHWC
state_*: [1, 1, 64, 80] NHWC
```

Generate it with:

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

UINT8 per-channel quantization succeeds on the exact same-band graph:

```text
float optimized Circle: approximately 3.5 MB
UINT8 Circle: approximately 1.1 MB
UINT8 tensors: 173
INT32 tensors: 54
```

No trained checkpoint exists yet, so the current calibration proves graph and
state compatibility rather than separation quality. After training, export
and calibrate the trained checkpoint using the same commands before comparing
float and quantized SI-SDR/SDR.
