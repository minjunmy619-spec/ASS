# SFC-Locoformer Conv2D 32B NPU Operation Note - 2026-06-28

## Goal

Implement the lite-SFC direction as a deployable student candidate:

- 32 adaptive mel/SFC bands.
- Conv2D as the main separator block.
- Two Conv2D Locoformer blocks by default.
- Explicit speech/music complex masks.
- Effects/SFX reconstructed as the residual source by the frequency wrapper.
- Keep ONNX/Circle graph small and avoid large streaming cache.

## Implemented Files

- `spectral_feature_compression/core/model/sfc_locoformer_conv2d_32b_npu.py`
  - `OnlineSFCLocoformerConv2D32BNPU2D`
  - `RegularConv2DLocoformerBlock2d`
  - `SourceWiseSigmoidTanhComplexHead2d`
  - `build_sfc_locoformer_conv2d_32b_npu_system`
- `spectral_feature_compression/core/model/proposed_separation_models.py`
  - Added proposal builder shim: `build_sfc_locoformer_conv2d_32b_npu_system`
- `spectral_feature_compression/__init__.py`
  - Added lazy exports.
- `recipes/dnr/models/sfc-locoformer-conv2d-32b-npu.2l.sourceaware-residual-sfx.distill.rt192k.fp512keep475/config.yaml`
  - New distillation recipe.
- `tests/test_sfc_locoformer_conv2d_32b_npu.py`
  - Focused model, wrapper, recipe, and budget tests.

## Model Structure

Default deploy recipe:

- Input: packed complex STFT, `(B, 2, T, F)`.
- Frequency preprocessing: keep 475 low bins and project high tail to 512 bins.
- SFC encoder:
  - 1x1 projection to `d_model=192`.
  - adaptive 32-band query compressor.
- Separator:
  - 2 regular Conv2D Locoformer-lite blocks.
  - time branch: gated 1x1 -> causal regular Conv2D -> 1x1.
  - band branch: gated 1x1 -> regular Conv2D over band axis -> 1x1.
  - FFN branch: gated 1x1 -> 1x1.
- SFC decoder:
  - query-conditioned soft-band expansion back to 512 bins.
- Source head:
  - cheap regular 1x1 Conv2D head.
  - sigmoid real mask and tanh imaginary mask.
  - returns true pre-transform logits for distillation.
- Wrapper:
  - predicts speech/music explicitly.
  - appends SFX as `mixture - speech - music`.

The first implementation used grouped/depthwise Conv2D inside the Locoformer
block and source head. `one-import-onnx` failed at `/band_dw/Conv`, matching the
known ONE importer weakness around grouped Conv patterns. The block was changed
to regular Conv2D, and the full-frequency source head was simplified to 1x1
Conv2D to avoid both grouped Conv and expensive full-frequency local kernels.

## Recipe Defaults

```yaml
sfc_loco_n_bands: 32
sfc_loco_d_model: 192
sfc_loco_layers: 2
sfc_loco_dilation_cycle: [1, 2]
sfc_loco_expansion: 2
sfc_loco_ffn_expansion: 4
sfc_loco_source_head_channels: 128
sfc_loco_source_refine_layers: 1
```

The recipe inherits the residual-SFX distillation stack and overrides
`task.model` to the new builder. The proposal builder accepts unused inherited
model keys only at the shim boundary so inherited Loco-CNB keys do not break
Hydra/fallback instantiation.

## Validation

### Unit Tests

```bash
.venv/bin/python -m pytest tests/test_sfc_locoformer_conv2d_32b_npu.py -q
```

Result:

```text
4 passed
```

### Lint and Compile

```bash
.venv/bin/python -m ruff check \
  spectral_feature_compression/core/model/sfc_locoformer_conv2d_32b_npu.py \
  spectral_feature_compression/core/model/proposed_separation_models.py \
  tests/test_sfc_locoformer_conv2d_32b_npu.py

PYTHONPYCACHEPREFIX=/tmp/ass_pycache .venv/bin/python -m py_compile \
  spectral_feature_compression/core/model/sfc_locoformer_conv2d_32b_npu.py \
  spectral_feature_compression/core/model/proposed_separation_models.py \
  tests/test_sfc_locoformer_conv2d_32b_npu.py
```

Result:

```text
ruff: All checks passed
py_compile: pass
```

### Budget

Measured from the recipe:

```text
params: 3,858,474
fp16 stream state: 147,456 bytes = 144.0 KiB
stream context: 6 STFT frames
core bins: 512
bands: 32
layers: 2
```

State tensors:

```text
state_0: [1, 192, 0, 512]
state_1: [1, 384, 2, 32]
state_2: [1, 384, 4, 32]
```

### ONNX Export

```bash
.venv/bin/python tools/online/export_onnx_online_model.py \
  recipes/dnr/models/sfc-locoformer-conv2d-32b-npu.2l.sourceaware-residual-sfx.distill.rt192k.fp512keep475/config.yaml \
  --out /tmp/sfc_locoformer_conv2d_32b_npu.onnx \
  --n-chan 1 \
  --frames 1 \
  --freqs 512 \
  --streaming \
  --state-meta-out /tmp/sfc_locoformer_conv2d_32b_npu_state.json \
  --deploy-manifest-out /tmp/sfc_locoformer_conv2d_32b_npu_manifest.json \
  --op-preset edge_npu_recommended
```

Result:

```text
ONNX ops: Add, Concat, Constant, Conv, Div, Gather, Identity, MatMul, Mul,
          ReduceMean, ReduceSum, Shape, Sigmoid, Slice, Softmax, Split, Sqrt,
          Sub, Tanh, Transpose
Disallowed ops: none
Initializers: 63 tensors, 15,558,040 bytes
```

ONNX node counts:

```text
nodes: 361
Conv: 29
MatMul: 3
Transpose: 11
Slice: 20
non-depthwise grouped Conv: 0
depthwise grouped Conv: 1
```

The remaining depthwise Conv is the SFC router Conv. ONE import accepts it in
this graph.

### Circle Import and Optimize

Import:

```bash
one-import-onnx \
  -i /tmp/sfc_locoformer_conv2d_32b_npu.onnx \
  -o /tmp/sfc_locoformer_conv2d_32b_npu.circle \
  --dynamic_batch_to_single_batch
```

Result: pass, artifact exists.

Optimize without layout conversion:

```bash
one-optimize \
  -i /tmp/sfc_locoformer_conv2d_32b_npu.circle \
  -o /tmp/sfc_locoformer_conv2d_32b_npu.opt_nolayout.circle
```

Result: pass, artifact exists.

Circle operator counts after import/no-layout optimize:

```text
TRANSPOSE: 72
MUL: 63
CONV_2D: 28
ADD: 25
STRIDED_SLICE: 20
LOGISTIC: 18
RSQRT: 11
MEAN: 11
CONCATENATION: 5
RESHAPE: 3
PAD: 3
FULLY_CONNECTED: 3
SUB: 2
SPLIT_V: 2
TANH: 1
SUM: 1
SOFTMAX: 1
DIV: 1
DEPTHWISE_CONV_2D: 1
```

Avoid this optimize command for now:

```bash
one-optimize \
  -i /tmp/sfc_locoformer_conv2d_32b_npu.circle \
  -o /tmp/sfc_locoformer_conv2d_32b_npu.opt.circle \
  --replace_non_const_fc_with_batch_matmul \
  --convert_nchw_to_nhwc
```

Observed failure:

```text
Internal Exception. x_rhs and y_lhs should be same
CircleBatchMatMul.cpp:140
```

Also, `--convert_nchw_to_nhwc` alone passes but increases Circle `TRANSPOSE`
operators from 72 to 426, so no-layout optimize is the better default for this
variant until layout-specific profiling says otherwise.

Quantization was not run in this pass because no real calibration H5 was
prepared. The next validation step should generate representative streaming
calibration inputs for `x` and the three state tensors, then run
`one-quantize` on `/tmp/sfc_locoformer_conv2d_32b_npu.opt_nolayout.circle`.

## Training Notes

This is the candidate to train first if the external lite-SFC result is the
best quality signal:

- It keeps the official SFC-like encoder/decoder query contract.
- It spends persistent cache only on 32 compressed bands.
- It avoids pooled capacity mixers.
- It exposes `mask` and true `mask_logits` for teacher distillation.
- The residual SFX path prevents a weak third explicit head from stealing
  capacity from speech/music.

Set `teacher_checkpoint_path` to the trained teacher checkpoint in the recipe
or with the normal aiaccel override mechanism, then launch:

```bash
bash recipes/dnr/models/sfc-locoformer-conv2d-32b-npu.2l.sourceaware-residual-sfx.distill.rt192k.fp512keep475/train.sh
```

Single-process debug launch:

```bash
.venv/bin/python aiaccel/aiaccel/torch/apps/train.py \
  recipes/dnr/models/sfc-locoformer-conv2d-32b-npu.2l.sourceaware-residual-sfx.distill.rt192k.fp512keep475/config.yaml
```

Use the same TV on-the-fly profile as the stronger recent student recipes and
track speech first, then music, then residual SFX leakage.
