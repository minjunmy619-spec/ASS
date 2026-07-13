# SFC Small Conv2D BatchNorm NPU Operation Notes

Date: 2026-07-12

## Starting Point

Checked the requested teacher/small SFC config:

`recipes/dnr/models/locoformer-small.enc-crossattn64dim.dec-crossattn64dim.musical64.learnable-query/config.yaml`

Important inherited structure:

- `sr=44100`, `n_fft=2048`, `hop_length=512`, so model STFT has `1025` bins.
- `n_src=3`, `n_chan=1`, `emb_dim=96`.
- SFC encoder: `CrossAttnEncoder`, `d_inner=64`, `n_bands=64`, `band_config=musical`, `query_type=learnable`.
- Separator: `BSLocoformer`, `n_layers=4`, RMS group norm, attention, Conv1D FFN.
- SFC decoder: `CrossAttnDecoder` with the same `d_inner=64`, `n_bands=64`, learnable query.

## ONE Source Findings Used

Relevant local ONE files checked:

- `/home/cmj/works/ONE/compiler/one-cmds/how-to-use-one-commands.txt`
  - Exposes `fuse_batchnorm_with_conv`, `fuse_batchnorm_with_dwconv`, `fuse_batchnorm_with_tconv`.
  - Exposes `remove_redundant_reshape` and `remove_redundant_transpose`.
- `/home/cmj/works/ONE/compiler/luci/pass/src/FuseBatchNormWithConvPass.cpp`
  - Documents that TensorFlow BatchNormalization is represented as `Mul` + `Add`.
  - Fuses Conv2D + Mul + Add into a new Conv2D with folded filter and bias.
- `/home/cmj/works/ONE/compiler/luci/pass/src/FuseBatchNormWithDwConvPass.cpp`
  - Same pattern exists for DepthwiseConv2D, but this model avoids depthwise conv to dodge known depthwise import fragility.
- `OPERATION_MANUAL_PYTORCH_TO_ONE_NPU.md`
  - Keeps ONNX export at opset `11~14`.
  - Uses `torch.onnx.export(..., dynamo=False)`.
  - Notes dynamic `Slice` and `ConstantOfShape` as import-risk patterns.

## Implemented Files

- `spectral_feature_compression/core/model/sfc_small_conv2d_bn_npu.py`
- `recipes/dnr/models/sfc-small-conv2d-bn-npu.musical64.onfly.rt192k/config.yaml`
- `recipes/dnr/models/sfc-small-conv2d-bn-npu.musical64.onfly.rt192k/train.sh`
- `tests/test_sfc_small_conv2d_bn_npu.py`

Top-level lazy exports were added for:

- `SFCSmallConv2DBNNPUCore`
- `SFCSmallConv2DBNNPUModel`

## Architecture

The new file is a fresh implementation, not a reuse of existing NPU variants.

Preserved SFC-small skeleton:

```text
complex STFT -> SFC encoder -> TF separator -> SFC decoder -> complex mask
```

NPU changes:

- Cross-attention encoder/decoder are replaced with Conv2D frequency transport.
- Locoformer attention blocks are replaced with Conv2D TF blocks.
- RMSNorm/RMSGroupNorm are replaced with `BatchNorm2d`.
- The deployable core uses packed-real `[B, C, T, F]` tensors.
- Complex `[B, M, F, T]` to packed `[B, 2*M, T, F]` conversion is only at the `ModelWrapper` boundary.
- Temporal modeling is causal in the separator.

Frequency transport:

- Encoder uses valid stride-2 Conv2D kernels for exact `1025 -> 512 -> 256 -> 128 -> 64`.
- Kernel schedule is `3, 2, 2, 2`, avoiding padding in the transport path.
- Decoder mirrors this with valid stride-2 TransposedConv2D kernels `2, 2, 2, 3` for `64 -> 1025`.
- This follows the NPU rule that transposed Conv2D stride should be `2`.

Separator:

- `8` Conv2D Loco-style blocks at `64` compressed bands.
- Each block has frequency mixing, causal time mixing, and a pointwise FFN.
- Default temporal kernel is `2`, dilation cycle is `[1]`.
- This keeps fp16 streaming state under the 192 KiB quota while putting most parameters into compressed-band Conv2D capacity.

Learnable query replacement:

- The original learnable-query idea is kept as low-cost learned band/frequency biases.
- This avoids attention/bmm and avoids irregular gather/scatter routing.

Default budget:

```text
parameters: 3,408,006
fp16 streaming state: 163,840 bytes = 160.00 KiB
```

## Training Recipe

The new recipe uses on-the-fly stem synthesis through:

```yaml
_base_:
  - ../../datamodules/on-the-fly-stem-tv-profiles.yaml
  - ${base_config_path}/train_base.yaml
```

The model builder is:

```yaml
_target_: spectral_feature_compression.core.model.sfc_small_conv2d_bn_npu.build_sfc_small_conv2d_bn_npu_system
```

EMA is disabled in this recipe:

```yaml
ema_weight: null
ema_update_freq: null
```

Reason: `SupTask` has a live note that BN update support for EMA is not implemented. For this BN-based model, plain supervised training is safer until EMA BN buffer handling is added.

## Validation Commands

Syntax:

```bash
.venv/bin/python -m py_compile \
  spectral_feature_compression/core/model/sfc_small_conv2d_bn_npu.py \
  tests/test_sfc_small_conv2d_bn_npu.py
```

Focused tests:

```bash
.venv/bin/python -m pytest tests/test_sfc_small_conv2d_bn_npu.py -q
```

Result:

```text
7 passed
```

Budget check:

```bash
.venv/bin/python - <<'PY'
import torch
from spectral_feature_compression.core.model.sfc_small_conv2d_bn_npu import SFCSmallConv2DBNNPUCore
m = SFCSmallConv2DBNNPUCore(n_freq=1025, n_bands=64).eval()
print(sum(p.numel() for p in m.parameters()))
print(m.state_size_bytes(dtype=torch.float16))
PY
```

Result:

```text
3408006
163840
```

The streaming API exposes eight real separator state tensors:

```text
x:       [1, 2, 1, 1025]
state_*: [1, 160, 1, 64] x 8
```

Full-size streaming ONNX export + ONE import + ONE optimize:

```bash
tmpdir=$(mktemp -d /tmp/sfc-small-conv2d-bn-npu-full.XXXXXX)
onnx_path="$tmpdir/stream_full.onnx"
circle_path="$tmpdir/stream_full.circle"
nhwc_path="$tmpdir/stream_full.nhwc.circle"

.venv/bin/python - <<PY
from pathlib import Path
import torch
from spectral_feature_compression.core.model.sfc_small_conv2d_bn_npu import SFCSmallConv2DBNNPUCore
from spectral_feature_compression.utils.onnx_streaming import StreamingStateIOWrapper, flatten_tensor_tree
m = SFCSmallConv2DBNNPUCore(n_freq=1025, n_bands=64).eval()
wrapper = StreamingStateIOWrapper(m, batch_size=1, dtype=torch.float32)
state = m.init_stream_state(batch_size=1, dtype=torch.float32)
flat_state, _ = flatten_tensor_tree(state)
x = torch.randn(1, 2, 1, 1025)
with torch.no_grad():
    torch.onnx.export(
        wrapper,
        (x, *flat_state),
        "$onnx_path",
        opset_version=11,
        input_names=["x", *[f"state_{idx}" for idx in range(len(flat_state))]],
        output_names=["y", *[f"next_state_{idx}" for idx in range(len(flat_state))]],
        do_constant_folding=True,
        dynamo=False,
    )
PY

/home/cmj/works/ONE/build/compiler/one-cmds/one-import-onnx \
  -i "$onnx_path" \
  -o "$circle_path" \
  --dynamic_batch_to_single_batch

/home/cmj/works/ONE/build/compiler/one-cmds/one-optimize \
  -i "$circle_path" \
  -o "$nhwc_path" \
  --convert_nchw_to_nhwc \
  --nchw_to_nhwc_input_shape \
  --nchw_to_nhwc_output_shape \
  --fuse_batchnorm_with_conv \
  --fuse_batchnorm_with_tconv \
  --fuse_activation_function \
  --remove_duplicate_const \
  --remove_unnecessary_add \
  --remove_unnecessary_slice \
  --remove_redundant_reshape \
  --remove_redundant_transpose
```

Result:

```text
stream_full.onnx: 13M
stream_full.circle: 13M
stream_full.nhwc.circle: 13M
```

Optimized Circle operator counts:

```text
54 CONV_2D
42 TRANSPOSE
27 ADD
20 STRIDED_SLICE
12 MUL
10 PAD
9 CONCATENATION
4 TRANSPOSE_CONV
4 RELU
3 SUB
```

## Current Limitations

- Full quantization now passes; see the 2026-07-13 update below.
- The optimized Circle graph still has compiler/layout transposes. `--convert_nchw_to_nhwc` helps but does not eliminate transposes around state and packed-mask plumbing.
- The remaining `PAD` ops are from same-width local frequency convolutions, not from the SFC frequency transport pyramid.
- Quality is not measured yet. This is a compile/latency-oriented architecture implementation and smoke validation, not a trained result.

## Next Validation Steps

1. Export a deployment mode with `masking=false` if the DSP can apply complex masks outside the NPU; this should remove much of the mask slice/mul/sub/concat tail.
2. Train the recipe against the on-the-fly TV profiles and compare against the cross-attention SFC-small teacher.
3. Replace synthetic calibration with real representative TV audio clips once manifests are available.

## 2026-07-13 Calibration And Quantization Update

Generated calibration and quantized artifacts under:

```text
logs/npu_verify_general/sfc_small_conv2d_bn_npu_quant_20260713/
```

Artifacts:

```text
stream_full.onnx            13M
stream_full.circle          13M
stream_full.opt.circle      13M
calib.h5                    21M
stream_full.opt.q.circle   3.7M
stream_full.opt.q.circle.log
calibration_manifest.json
calib_list.txt
calib_npy/
```

Calibration method:

- 64 streaming records.
- Input frame `x` comes from synthetic audio-derived STFT frames.
- State inputs are recorded by rolling `SFCSmallConv2DBNNPUCore.forward_stream()` forward, so rows after the first include nonzero state tensors.
- Calibration summary:

```text
x_min: -30.738069534301758
x_max: 31.097064971923828
state_absmax_max: 0.7862289547920227
```

Important quantization detail:

- The first attempt used `--nchw_to_nhwc_input_shape`, which converted Circle inputs to `[1,1,1025,2]` and `[1,1,64,160]`.
- That mismatched the NCHW calibration H5 and failed at `record-minmax` with `Input shape mismatch`.
- The successful run preserved external NCHW input shapes while still using `--convert_nchw_to_nhwc` internally.

Successful optimize command:

```bash
/home/cmj/works/ONE/build/compiler/one-cmds/one-optimize \
  -i logs/npu_verify_general/sfc_small_conv2d_bn_npu_quant_20260713/stream_full.circle \
  -o logs/npu_verify_general/sfc_small_conv2d_bn_npu_quant_20260713/stream_full.opt.circle \
  --convert_nchw_to_nhwc \
  --fuse_batchnorm_with_conv \
  --fuse_batchnorm_with_tconv \
  --fuse_activation_function \
  --remove_duplicate_const \
  --remove_unnecessary_add \
  --remove_unnecessary_slice \
  --remove_redundant_reshape \
  --remove_redundant_transpose
```

Successful quantize command:

```bash
/home/cmj/works/ONE/build/compiler/one-cmds/one-quantize \
  -i logs/npu_verify_general/sfc_small_conv2d_bn_npu_quant_20260713/stream_full.opt.circle \
  -d logs/npu_verify_general/sfc_small_conv2d_bn_npu_quant_20260713/calib.h5 \
  -f h5 \
  -o logs/npu_verify_general/sfc_small_conv2d_bn_npu_quant_20260713/stream_full.opt.q.circle \
  --quantized_dtype uint8 \
  --granularity channel \
  --input_type uint8 \
  --output_type uint8 \
  --mode percentile \
  --min_percentile 0.1 \
  --max_percentile 99.9 \
  --save_min_max
```

Quantization result:

```text
Recording finished. Number of recorded data: 64
stream_full.opt.q.circle exists, size 3.7M
```

Quantized Circle input dtypes:

```text
arg0 UINT8
arg1 UINT8
arg2 UINT8
arg3 UINT8
arg4 UINT8
arg5 UINT8
arg6 UINT8
arg7 UINT8
arg8 UINT8
```

Quantized Circle input shapes:

```text
arg0 [1,2,1,1025]
arg1 [1,160,1,64]
arg2 [1,160,1,64]
arg3 [1,160,1,64]
arg4 [1,160,1,64]
arg5 [1,160,1,64]
arg6 [1,160,1,64]
arg7 [1,160,1,64]
arg8 [1,160,1,64]
```

Quantized Circle operator counts:

```text
54 CONV_2D
43 TRANSPOSE
27 ADD
16 STRIDED_SLICE
12 MUL
10 PAD
9 CONCATENATION
4 TRANSPOSE_CONV
4 RELU
3 SUB
```

## 2026-07-13 Stock Quantization Sweep Update

The helper `tools/online/run_one_stock_quant_sweep.py` was patched before the sweep:

- Fixed MSE parsing so output names such as `/Concat_8` and `/Slice_7` are not mistaken for MSE values.
- Added `mse_primary`, `mse_mean`, and `mse_by_output` to `summary.json`.
- Added `--calib-record-limit` and `--test-record-limit` to generate limited H5/list files inside the output directory.
- Added `--stream-output` for live one-quantize progress when debugging.

Validation:

```bash
.venv/bin/python -m py_compile tools/online/run_one_stock_quant_sweep.py
```

Sweep command:

```bash
.venv/bin/python tools/online/run_one_stock_quant_sweep.py \
  --input-circle logs/npu_verify_general/sfc_small_conv2d_bn_npu_quant_20260713/stream_full.opt.circle \
  --calib-data logs/npu_verify_general/sfc_small_conv2d_bn_npu_quant_20260713/calib.h5 \
  --calib-record-limit 16 \
  --test-data logs/npu_verify_general/sfc_small_conv2d_bn_npu_quant_20260713/calib.h5 \
  --test-record-limit 8 \
  --evaluate-result \
  --print-mse \
  --output-dir logs/one_stock_quant_sweep/sfc_small_conv2d_bn_npu_20260713_toolopt \
  --modes percentile,moving_average \
  --min-percentiles 0.01,0.1 \
  --max-percentiles 99.9,99.99 \
  --moving-avg-batches 16 \
  --moving-avg-consts 0.05 \
  --quantized-dtype uint8 \
  --granularity channel \
  --input-type uint8 \
  --output-type uint8 \
  --circle-inspect /home/cmj/works/ONE/build/compiler/circle-inspect/circle-inspect \
  --timeout 900
```

Sweep artifacts:

```text
logs/one_stock_quant_sweep/sfc_small_conv2d_bn_npu_20260713_toolopt/
  calib_first16.h5
  test_first8.h5
  summary.json
  selection.json
  *.q.circle
  *.log
```

Ranking by `mse_primary`:

```text
1. percentile min=0.1 max=99.9   mse_primary=0.000383961  mse_mean=6.382947333333333e-05
2. percentile min=0.01 max=99.9  mse_primary=0.000384045  mse_mean=6.36930888888889e-05
3. percentile min=0.1 max=99.99  mse_primary=0.000384197  mse_mean=6.397141222222223e-05
4. percentile min=0.01 max=99.99 mse_primary=0.000384355  mse_mean=6.418553222222222e-05
5. moving_average batch=16 const=0.05 mse_primary=0.00038474 mse_mean=6.397011555555555e-05
```

Selected setting:

```text
mode=percentile
min_percentile=0.1
max_percentile=99.9
```

The full 64-record quantized artifact already uses this winning setting:

```text
logs/npu_verify_general/sfc_small_conv2d_bn_npu_quant_20260713/stream_full.opt.q.circle
```

## 2026-07-13 Mixed Precision QConfig Update

The helper `tools/online/suggest_one_mixed_precision_qconfig.py` was patched before the mixed sweep:

- Added qconfig eligibility tracking in `nodes.csv`.
- Added hard qconfig exclusion for memory/layout ops through `--exclude-op`.
- Added `--exclude-regex` for hard name-based qconfig exclusion.
- Kept excluded ops visible in the ranking report with `eligible=0` so boundary choices can be audited.

Validation:

```bash
.venv/bin/python -m py_compile tools/online/suggest_one_mixed_precision_qconfig.py tools/online/run_one_stock_quant_sweep.py
```

Candidate generation:

```bash
.venv/bin/python tools/online/suggest_one_mixed_precision_qconfig.py \
  --circle logs/npu_verify_general/sfc_small_conv2d_bn_npu_quant_20260713/stream_full.opt.circle \
  --out-dir logs/one_mixed_precision_suggestions/sfc_small_conv2d_bn_npu_20260713_depth_tune \
  --top-k 20 \
  --island-sizes 3,5,8 \
  --depth-fractions 0.10,0.15,0.20,0.25 \
  --prefer-regex 'mask|head|decoder|sfc|attn|softmax|out|freq_mix|ffn|conv'
```

Fast-screen setup used first 16 calibration records and first 8 evaluation records with fixed percentile calibration:

```text
mode=percentile
min_percentile=0.1
max_percentile=99.9
default dtype=uint8
target qconfig dtype=int16
```

Fast-screen ranking by primary separated-output MSE:

```text
pure uint8 percentile 0.1/99.9  int16_layers=0   mse_primary=0.000383961
best_island3                   int16_layers=3   mse_primary=0.000385010
best_island5                   int16_layers=5   mse_primary=0.000384697
best_island8                   int16_layers=8   mse_primary=0.000385221
top20                          int16_layers=20  mse_primary=0.000385513
depth_back_10                  int16_layers=22  mse_primary=0.000370437
depth_back_15                  int16_layers=27  mse_primary=0.000370476
depth_back_20                  int16_layers=33  mse_primary=0.000370435
depth_back_25                  int16_layers=37  mse_primary=0.000370594
```

`depth_back_20` was numerically lowest in the fast screen by only `2e-9` MSE versus `depth_back_10`.  The selected mixed qconfig is therefore `depth_back_10`, which gives the same measured quality range with fewer int16 layers and fewer likely NPU latency risks.

Full 64-record calibration/evaluation:

```text
pure uint8 full eval:
  artifact=logs/one_stock_quant_sweep/sfc_small_conv2d_bn_npu_20260713_uint8_full_eval/00_percentile_p0p1_99p9.q.circle
  mse_primary=0.000381631
  mse_mean=0.0000652770711111111

mixed depth_back_10 full eval:
  artifact=logs/one_stock_quant_sweep/sfc_small_conv2d_bn_npu_20260713_mixed_depth_back10_full/00_percentile_p0p1_99p9.q.circle
  mse_primary=0.000370690
  mse_mean=0.0000640614044444444
```

Final selected mixed artifacts were copied next to the main NPU verification artifacts:

```text
logs/npu_verify_general/sfc_small_conv2d_bn_npu_quant_20260713/stream_full.opt.mixed_depth_back10.q.circle
logs/npu_verify_general/sfc_small_conv2d_bn_npu_quant_20260713/stream_full.opt.mixed_depth_back10.qconfig.json
logs/npu_verify_general/sfc_small_conv2d_bn_npu_quant_20260713/stream_full.opt.mixed_depth_back10.summary.json
logs/npu_verify_general/sfc_small_conv2d_bn_npu_quant_20260713/stream_full.opt.mixed_depth_back10.tensor_dtype.txt
logs/npu_verify_general/sfc_small_conv2d_bn_npu_quant_20260713/stream_full.opt.mixed_depth_back10.operators.txt
```

Tensor dtype sanity check:

```text
mixed depth_back_10: INT16 50, INT32 78, INT64 2, UINT8 233
pure uint8:          INT32 80, UINT8 249
```
