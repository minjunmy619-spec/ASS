# DolphinSFCNPU

`DolphinSFCNPU` is the second, cleaner ASS adaptation of Dolphin.  It keeps the first `DolphinSFC` version untouched for comparison, while fixing the NPU-readiness issues that do not require changing the adaptive `bmm` band routing.

## Changes From DolphinSFC

- Frozen deterministic band constants via `FrozenDolphinBandSpec2d`; no `librosa` dependency and no environment-dependent fallback.
- The adaptive SFC compressor/decoder `bmm` path is intentionally kept.  It operates on 3D tensors after reshape, which is acceptable for this project direction.
- Real-valued source-gain masking now uses source/channel slicing plus concat instead of `repeat` / `repeat_interleave`, avoiding `Tile` in ONNX export.
- Decoder upsample no longer has runtime shape branches because preset band counts are chosen to make down/up band sizes exact.
- Tests now audit ONNX op sets for `edge_small`, `large_6m`, and `large_8m`.

## Validation

Run inside the ASS Docker checkout:

```bash
cd /app/ASS
./.venv/bin/python DolphinSFCNPU/test_dolphin_sfc_npu.py
```

Current ONNX op audit for all three presets:

```text
Add, Clip, Concat, Constant, Conv, ConvTranspose, Div, Gather, Identity,
MatMul, Mul, ReduceMean, ReduceSum, Reshape, Shape, Sigmoid, Slice,
Softmax, Sqrt, Transpose
```

`Tile`, `Expand`, and `ConstantOfShape` are explicitly forbidden by the test. `MatMul` is expected because the adaptive band routing is still implemented with `bmm`.

## Presets

- `edge_small`: `n_bands=32, d_model=16, num_scales=3`; smoke/export validation.
- `large_6m`: `n_bands=64, d_model=256, num_scales=3`; quality-oriented 6M-class model.
- `large_8m`: `n_bands=64, d_model=288, num_scales=3`; quality-oriented 8M-class model.

The 192 KB cache quota remains intentionally out of scope for the large presets.
