# DolphinSFCNPU

`DolphinSFCNPU` is the second, cleaner ASS adaptation of Dolphin.  It keeps the first `DolphinSFC` version untouched for comparison, while fixing the NPU-readiness issues that do not require changing the adaptive `bmm` band routing.

## Changes From DolphinSFC

- Frozen deterministic band constants via `FrozenDolphinBandSpec2d`; no `librosa` dependency and no environment-dependent fallback.
- The adaptive SFC compressor/decoder `bmm` path is intentionally kept.  It operates on 3D tensors after reshape, which is acceptable for this project direction.
- Real-valued source-gain masking now uses source/channel slicing plus concat instead of `repeat` / `repeat_interleave`, avoiding `Tile` in ONNX export.
- Decoder upsample no longer has runtime shape branches because preset band counts are chosen to make down/up band sizes exact.
- The streaming ONNX export wrapper packs the whole nested streaming-state tree
  into a single `(B, total_numel)` tensor, so the exported graph has only
  `(x, state)` inputs and `(y, next_state)` outputs. This satisfies AGENT.md
  rule 14 (small number of I/O parameters) without touching the ergonomic
  tree-shaped `forward_stream` API used in training and Python inference.
- Tests now audit ONNX op sets for `edge_small`, `large_6m`, and `large_8m`.

### Packed streaming-state I/O

The core `DolphinSFCNPUSeparator.forward_stream` still accepts and returns a
nested tuple of per-layer caches. That ergonomics is preserved for Python
callers. For ONNX export, use `DolphinSFCNPUStreamingExportWrapper`:

```python
wrapper = DolphinSFCNPUStreamingExportWrapper(model, batch_size=1, dtype=torch.float32).eval()
x = torch.randn(1, 2, 1, model.n_freq)
packed_state = wrapper.init_packed_state(batch_size=1, dtype=torch.float32)

torch.onnx.export(
    wrapper,
    (x, packed_state),
    "dolphin_sfc_npu.onnx",
    opset_version=11,
    input_names=["x", "state"],
    output_names=["y", "next_state"],
    do_constant_folding=True,
)
```

The packing layout (per-leaf shape + offset list) is frozen at wrapper
construction time, so unpack uses a static sequence of `Slice` + `Reshape`
and repack uses per-leaf `Reshape` + a single `Concat`. All of these are on
the NPU-allowed op list. No runtime shape branches, no dynamic control flow.

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

`Tile`, `Expand`, and `ConstantOfShape` are explicitly forbidden by the test.
`MatMul` is expected because the adaptive band routing is still implemented
with `bmm`. `Concat`, `Slice`, and `Reshape` are also used by the packed
streaming-state I/O wrapper.

## Presets

- `edge_small`: `n_bands=32, d_model=16, num_scales=3`; smoke/export validation.
- `large_6m`: `n_bands=64, d_model=256, num_scales=3`; quality-oriented 6M-class model.
- `large_8m`: `n_bands=64, d_model=288, num_scales=3`; quality-oriented 8M-class model.

The 192 KB cache quota remains intentionally out of scope for the large presets.
