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

- Cross-attention encoder/decoder are replaced with NPU-friendly SFC query transport using Conv2D projections plus per-head MatMul/Softmax.
- Locoformer attention blocks are replaced with Conv2D TF blocks.
- RMSNorm/RMSGroupNorm are replaced with `BatchNorm2d`.
- The deployable core uses packed-real `[B, C, T, F]` tensors.
- Complex `[B, M, F, T]` to packed `[B, 2*M, T, F]` conversion is only at the `ModelWrapper` boundary.
- Temporal modeling is causal in the separator.

Frequency transport:

- Encoder first embeds full STFT bins with Conv2D, then 64 learned musical-band queries cross-attend over all 1025 frequency bins.
- The encoder attention uses official `musical64` band indices and a learnable per-head band-distance position bias with shape `[4, 64, 1025]`.
- Decoder mirrors this contract: 1025 learned full-frequency queries cross-attend over the 64 compressed separator tokens with position bias `[4, 1025, 64]`.
- Projections and FFNs are implemented as Conv2D + BatchNorm2D.  Attention routing uses static per-head `MatMul` + `Softmax` instead of PyTorch `MultiheadAttention`.
- The previous fixed stride-2 pyramid (`1025 -> 512 -> 256 -> 128 -> 64`) has been removed because it did not preserve the core SFC query/routing mechanism.

Separator:

- `8` Conv2D Loco-style blocks at `64` compressed bands.
- Each block has frequency mixing, causal time mixing, and a pointwise FFN.
- Default temporal kernel is `2`, dilation cycle is `[1]`.
- This keeps fp16 streaming state under the 192 KiB quota while putting most parameters into compressed-band Conv2D capacity.

SFC query transport:

- Encoder query shape is `[4, 64, 16]` for the default `d_inner=64`, `n_sfc_heads=4`.
- Decoder query shape is `[4, 1025, 16]`.
- `band_config=musical`, `n_sfc_heads=4`, and `learnable_pos_bias=true` are explicit recipe knobs.
- This is closer to official SFC-CA than the first Conv2D pyramid version while still avoiding `LayerNorm`, `RMSNorm`, `Gemm`, `MultiheadAttention`, `Tile`, and `Expand` in the streaming ONNX smoke test.

Default budget:

```text
parameters: 3,823,782
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
8 passed
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
3823782
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

## 2026-07-15 Faithful SFC Transport Update

The first Conv2D BN NPU version used a fixed stride-2 frequency pyramid.  That version compiled, but it did not preserve the official SFC encoder/decoder mechanism.  The model file was updated to use NPU-friendly SFC query transport:

```text
full-frequency embeddings + musical band queries -> MatMul/Softmax compression to 64 tokens
64 separator tokens + full-frequency queries -> MatMul/Softmax expansion to 1025 bins
```

Validation:

```bash
.venv/bin/python -m py_compile \
  spectral_feature_compression/core/model/sfc_small_conv2d_bn_npu.py \
  tests/test_sfc_small_conv2d_bn_npu.py

.venv/bin/python -m pytest tests/test_sfc_small_conv2d_bn_npu.py -q
```

Result:

```text
8 passed
parameters: 3,823,782
fp16 streaming state: 163,840 bytes
```

Streaming ONNX smoke-test operator intent:

```text
Required: Conv, MatMul, Softmax
Forbidden: Gemm, LayerNormalization, RMSNormalization, Pad, ConstantOfShape, Expand, Tile
```

Important: the 2026-07-13 Circle/quantization artifacts below were produced from the previous fixed-pyramid model revision.  They are now stale for the current faithful SFC-query implementation and must be regenerated before deployment or quantization comparison.

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

## 2026-07-15 ONE-Guided Streaming Latency Rewrite

The July 15 faithful SFC encoder/decoder topology was rewritten for lower final
NPU latency while keeping the SFC semantic contract:

- Encoder still uses learned musical-band queries to cross-attend over full
  frequency-bin keys/values with musical position bias.
- Decoder still uses learned full-frequency queries to cross-attend over
  compressed band keys/values with the transposed musical position bias.
- The Conv2D separator and BatchNorm2D normalization remain unchanged in
  meaning.

The change is in the streaming/export implementation.  The previous streaming
attention path iterated over heads in Python, slicing the key/value channel
dimension and concatenating head outputs.  ONE does not generally erase that
kind of real head-layout plumbing:

- `RemoveRedundantTransposePass` only handles consecutive transposes with
  constant permutations.
- `SubstituteTransposeToReshapePass` only converts a transpose when the
  non-unit dimension order is unchanged.
- `RemoveRedundantReshapePass` only bypasses consecutive reshapes.
- `ConvertNCHWToNHWC` runs early and resolves custom `BatchMatMul`/`MatMul`
  before layout conversion, so the PyTorch/ONNX attention layout strongly
  affects the final Circle graph.
- `FuseBatchNormWithConvPass` folds inference BatchNorm exported as
  Conv/Mul/Add into Conv2D weights/bias, which confirms BatchNorm2D is the right
  latency replacement for RMSNorm here.

Implementation changes:

- `SFCSmallConv2DBNEncoder.forward_stream()` now uses one 4D batched
  multi-head MatMul path for the streaming frame:
  `[B,H,K,Dh] x [B,H,Dh,F] -> [B,H,K,F]`.
- `SFCSmallConv2DBNDecoder.forward_stream()` uses the symmetric batched path:
  `[B,H,F,Dh] x [B,H,Dh,K] -> [B,H,F,K]`.
- The full training forward path keeps the multi-frame implementation, so
  existing training semantics and on-the-fly synthesis remain compatible.
- `CausalConv2dBNAct.forward_stream()` returns the one-frame input directly as
  next state for the default one-frame causal context, avoiding a traced Slice
  per separator block.
- `tests/test_sfc_small_conv2d_bn_npu.py` now asserts the streaming export has
  exactly 4 `MatMul` and 2 `Softmax` ONNX ops, preventing regression back to a
  per-head graph.

Validation:

```bash
.venv/bin/python -m pytest tests/test_sfc_small_conv2d_bn_npu.py -q
```

Result:

```text
8 passed
```

Exported artifacts:

```text
logs/npu_efficiency_audit/sfc_small_conv2d_bn_npu_20260715/
  sfc_small_stream_masked.onnx
  sfc_small_stream_masked.circle
  sfc_small_stream_masked.opt.circle
  sfc_small_stream_rawmask.onnx
  sfc_small_stream_rawmask.circle
  sfc_small_stream_rawmask.opt.circle
  sfc_small_stream_perhead_reference.onnx
```

Commands:

```bash
.venv/bin/python tools/online/export_onnx_online_model.py \
  recipes/dnr/models/sfc-small-conv2d-bn-npu.musical64.onfly.rt192k/config.yaml \
  --out logs/npu_efficiency_audit/sfc_small_conv2d_bn_npu_20260715/sfc_small_stream_masked.onnx \
  --n-chan 1 --frames 1 --opset 11 --streaming --check \
  --state-meta-out logs/npu_efficiency_audit/sfc_small_conv2d_bn_npu_20260715/sfc_small_stream_masked_state.json

/home/cmj/works/ONE/build/compiler/one-cmds/one-import-onnx \
  -i logs/npu_efficiency_audit/sfc_small_conv2d_bn_npu_20260715/sfc_small_stream_masked.onnx \
  -o logs/npu_efficiency_audit/sfc_small_conv2d_bn_npu_20260715/sfc_small_stream_masked.circle \
  --dynamic_batch_to_single_batch

/home/cmj/works/ONE/build/compiler/one-cmds/one-optimize \
  -i logs/npu_efficiency_audit/sfc_small_conv2d_bn_npu_20260715/sfc_small_stream_masked.circle \
  -o logs/npu_efficiency_audit/sfc_small_conv2d_bn_npu_20260715/sfc_small_stream_masked.opt.circle \
  --convert_nchw_to_nhwc \
  --fuse_batchnorm_with_conv \
  --fuse_activation_function \
  --remove_duplicate_const \
  --remove_unnecessary_add \
  --remove_unnecessary_slice \
  --remove_unnecessary_strided_slice \
  --remove_unnecessary_reshape \
  --remove_unnecessary_transpose \
  --remove_redundant_reshape \
  --remove_redundant_transpose \
  --forward_transpose_op \
  --resolve_customop_matmul \
  --resolve_customop_batchmatmul
```

Measured ONNX operator counts:

```text
per-head reference ONNX:
  nodes=366 Constant=122 Conv=60 Add=39 Slice=28 Mul=24
  MatMul=16 Softmax=8 Transpose=8 Concat=11 Reshape=6

rewritten masked streaming ONNX:
  nodes=238 Conv=60 Constant=52 Add=33 Slice=12 Mul=18
  MatMul=4 Softmax=2 Transpose=4 Concat=9 Reshape=6

rewritten raw-mask streaming ONNX:
  nodes=179 Conv=60 Add=30 Constant=20 Concat=8 Mul=6
  MatMul=4 Softmax=2 Slice=4 Transpose=4 Reshape=6
```

The temporary per-head reference ONNX failed `one-import-onnx`:

```text
loc("/MatMul/reshape"): error: 'Circle.reshape' op requires 'output' number
of elements to match 'input' number of elements, but got 1025 and 16400
```

This reinforces the rewrite: the batched-head path is smaller and avoids an
importer failure mode.

Measured Circle operator counts:

```text
masked import Circle:
  nodes=277 TRANSPOSE=124 CONV_2D=60 ADD=31 MUL=14 PAD=12
  STRIDED_SLICE=12 CONCATENATION=9 RESHAPE=6 BATCH_MATMUL=4 SOFTMAX=2

masked optimized Circle:
  nodes=189 CONV_2D=60 TRANSPOSE=36 ADD=31 MUL=14 PAD=12
  STRIDED_SLICE=12 CONCATENATION=9 BATCH_MATMUL=4 SOFTMAX=2

raw-mask import Circle:
  nodes=250 TRANSPOSE=124 CONV_2D=60 ADD=28 PAD=12 CONCATENATION=8
  RESHAPE=6 BATCH_MATMUL=4 STRIDED_SLICE=4 MUL=2 SOFTMAX=2

raw-mask optimized Circle:
  nodes=152 CONV_2D=60 ADD=28 TRANSPOSE=26 PAD=12 CONCATENATION=8
  RESHAPE=6 BATCH_MATMUL=4 STRIDED_SLICE=4 MUL=2 SOFTMAX=2
```

Deployment guidance from this pass:

- Use the rewritten streaming path as the default SFC-small NPU export path.
- Prefer `--disable-masking` for the lowest NPU latency if the DSP/CPU side can
  apply the packed complex mask; this keeps the learned model identical but
  removes the memory-heavy mask application tail from the NPU graph.
- Keep `--convert_nchw_to_nhwc` for Circle optimization, but preserve external
  NCHW input/output shapes unless calibration H5 and runtime integration are
  regenerated for NHWC.
- Do not use `--decompose_softmax`; the SFC transport should stay as
  `BATCH_MATMUL -> SOFTMAX -> BATCH_MATMUL`.
- For this graph, do not enable the `substitute_*_to_reshape` family in the
  final latency recipe by default.  A small flag sweep showed the masked graph
  improves from 204 to 189 optimized Circle nodes when those substitutions are
  omitted and `--forward_transpose_op` is kept.  The raw-mask graph is unchanged
  at 152 nodes either way.
- The remaining `STRIDED_SLICE` ops in the raw-mask optimized graph are from
  key/value splitting and are much smaller than the original per-head split.

Optional ABI-changing flag result:

```text
with --nchw_to_nhwc_input_shape --nchw_to_nhwc_output_shape:
  masked optimized Circle:   nodes=172 memory_ops=58 TRANSPOSE=19
  raw-mask optimized Circle: nodes=134 memory_ops=38 TRANSPOSE=8
```

This is the best measured flag-level latency improvement after the batched-head
rewrite, but it changes external tensor shapes.  For example, the raw-mask input
becomes `[1, 1, 1025, 2]` and each separator state becomes `[1, 1, 64, 160]`.
Use it only if the calibration H5 writer and runtime integration are switched
to NHWC tensors.

Implemented NHWC raw-mask ABI artifacts under:

```text
logs/npu_efficiency_audit/sfc_small_conv2d_bn_npu_20260715/nhwc_abi_rawmask/
  build_nhwc_abi_rawmask.py
  stream_rawmask.onnx
  stream_rawmask.circle
  stream_rawmask.nhwc.opt.circle
  stream_rawmask.nhwc.opt.q.circle
  calib_nhwc.h5
  calib_nhwc_list.txt
  calib_npy/
  manifest.json
  stream_rawmask.nhwc.opt.tensor_shape.txt
```

Build command:

```bash
.venv/bin/python \
  logs/npu_efficiency_audit/sfc_small_conv2d_bn_npu_20260715/nhwc_abi_rawmask/build_nhwc_abi_rawmask.py \
  --records 64
```

The script exports raw-mask streaming ONNX, imports it to Circle, applies the
NHWC ABI optimize flags, generates 64 NHWC calibration records, packages
`calib_nhwc.h5` with `one-create-quant-dataset`, and quantizes with the
percentile `0.1/99.9` setting.

Sanity check:

```text
calib_nhwc.h5 records: 64
input 0: [1, 1, 1025, 2] float32
input 1-8: [1, 1, 64, 160] float32

stream_rawmask.nhwc.opt.circle:
  ADD=28 BATCH_MATMUL=4 CONCATENATION=8 CONV_2D=60 MUL=2 PAD=12
  RESHAPE=6 SOFTMAX=2 STRIDED_SLICE=4 TRANSPOSE=8

stream_rawmask.nhwc.opt.q.circle: 4.0M
```

## 2026-07-16 KV-Split Query-Scaled Variant

Implemented a new variant rather than modifying the base SFC-small NPU model:

```text
spectral_feature_compression/core/model/sfc_small_conv2d_bn_npu_kvsplit.py
recipes/dnr/models/sfc-small-conv2d-bn-npu-kvsplit.musical64.onfly.rt192k/config.yaml
```

Changes versus the base faithful SFC-small Conv2D BN NPU model:

- Replaced each shared `kv_proj: Conv2d(d, 2d)` with separate
  `key_proj: Conv2d(d, d)` and `value_proj: Conv2d(d, d)`.
- Absorbed `head_dim ** -0.5` into encoder/decoder query parameters.  For the
  adaptive encoder path, the scale is absorbed into `adaptive_pool`.
- Added `convert_sfc_small_conv2d_bn_npu_state_dict_to_kvsplit()` so a base
  checkpoint can be mapped exactly: `kv_proj.weight[:d] -> key_proj.weight`,
  `kv_proj.weight[d:] -> value_proj.weight`, and query tensors are scaled.

Validation:

```bash
.venv/bin/python -m pytest tests/test_sfc_small_conv2d_bn_npu.py -q
```

Result:

```text
11 passed
```

The equivalence test loads a converted base state dict into the KV-split model
and checks full forward masks, separated output, and streaming output.

Raw-mask streaming export and NHWC Circle artifacts:

```text
logs/npu_efficiency_audit/sfc_small_conv2d_bn_npu_kvsplit_20260716/
  stream_rawmask.onnx
  stream_rawmask.circle
  stream_rawmask.nhwc.opt.circle
  stream_rawmask.nhwc.opt.tensor_shape.txt
  stream_rawmask.circle.operators.json
  stream_rawmask.nhwc.opt.circle.operators.json
```

Measured counts:

```text
raw-mask ONNX:
  nodes=149
  Add=28 Concat=8 Constant=6 Conv=62 MatMul=4 Relu=29
  Reshape=6 Softmax=2 Transpose=4

raw-mask imported Circle:
  nodes=250
  ADD=28 BATCH_MATMUL=4 CONCATENATION=8 CONV_2D=62 PAD=12
  RESHAPE=6 SOFTMAX=2 TRANSPOSE=128

raw-mask NHWC optimized Circle:
  nodes=132
  ADD=28 BATCH_MATMUL=4 CONCATENATION=8 CONV_2D=62 PAD=12
  RESHAPE=6 SOFTMAX=2 TRANSPOSE=10
```

This removes the expected `STRIDED_SLICE=4` and `MUL=2` from the previous
raw-mask NHWC optimized Circle graph.  The graph has two more Conv2D nodes
because key/value projection is now represented by two Conv2D ops instead of one
wide Conv2D plus slices.
