# EdgeFusionNPU

EdgeFusionNPU is a deployment-first online separator candidate. It is not a
trained checkpoint; it is a small NPU-native architecture scaffold for the next
training run.

The design fuses:

- BandSCNet / SCNet: sparse, band-aware frequency processing and cross/narrow
  band separation.
- TF-MLPNet: time-frequency domain operation with tiny MLP/conv-style mixers.
- Moises-Light and windowed attention results: use compressed band processing
  and local/finite temporal context instead of full-sequence attention.
- TIGER NPU lessons: static export shapes, explicit caches, no scalar Gather,
  and no rank-3 activation MatMul that can lower to const-weight
  FullyConnected.
- SFC / DolphinSFCNPU: keep STFT/iSTFT outside the NPU graph and pass packed
  RI spectrogram frames through a compact core.

## Contract

Input per streaming call:

```text
x: [B, 2*n_chan, F, 1]
state: [B, num_blocks*hidden_channels, F, time_kernel - 1]
```

Output:

```text
mask: [B, n_src*n_chan, F, 1]
next_state: [B, num_blocks*hidden_channels, F, time_kernel - 1]
```

The export contract is intentionally capped at two ONNX inputs and two ONNX
outputs, with one packed 4D cache tensor. This leaves room for future host-side
metadata without exceeding the "no more than 4 inputs/outputs for states" rule.

The host runtime applies the real-valued gain masks to the complex STFT and
performs overlap-add iSTFT. Masks are non-negative and scaled by `mask_scale`
so the model can make modest gain corrections instead of being limited to pure
attenuation.

## Training vs Export

PyTorch training supports chunk or clip tensors:

```text
complex STFT x: [B, M, F, T]
output:         [B, n_src, M, F, T]
```

`EdgeFusionNPUOnlineModel` carries the packed recurrent state through all `T`
frames and can optionally return it:

```python
est, next_state = model(stft_chunk, initial_state=state, return_state=True)
```

This allows normal clip training, chunk training, and split-chunk training with
state continuity. `detach_state=True` can be used for truncated BPTT across
chunks.

ONNX export remains frame-by-frame. `EdgeFusionNPUExportWrapper` calls the
single-frame core path directly and exports only:

```text
x:          [B, 2*M, F, 1]
state:      [B, state_channels, F, context]
mask:       [B, n_src*M, F, 1]
next_state: [B, state_channels, F, context]
```

## Presets

| Preset | Hidden | Blocks | Params target | Purpose |
|---|---:|---:|---|---|
| `tiny` | 12 | 3 | very small | first compile and low-power edge baseline |
| `compact` | 16 | 5 | small | stronger candidate when using `F=513` / `n_fft=1024` |
| `balanced` | 24 | 6 | small | likely first training candidate |
| `wide` | 32 | 8 | medium-small | quality probe if memory allows |
| `compact-v2-ssmlite` | 16 | 5 | small | compact plus learnable recurrent cache update |
| `compact-v2-bandtoken` | 16 | 5 | small | compact plus Conv/ConvTranspose frequency bottleneck |
| `compact-v2-hybrid` | 16 | 5 | small | recommended first training candidate |
| `balanced-v2-hybrid` | 24 | 6 | small | stronger F=257 capacity probe |
| `big-v2-hybrid-2m` | 24 | 5 | 2.15M | larger F=257 quality candidate with low-token FFN capacity |
| `large-v2-hybrid-5m` | 24 | 7 | 5.30M | largest current F=257 quality candidate |

## DnR Recipes

Training recipes are available under `recipes/dnr/models`:

```text
edge-fusion-npu.tiny.rt192k
edge-fusion-npu.compact.fp512.rt192k
edge-fusion-npu.balanced.fp256.rt192k
edge-fusion-npu.compact-fp512.v2-ssmlite.rt192k
edge-fusion-npu.compact-fp512.v2-bandtoken.rt192k
edge-fusion-npu.compact-fp512.v2-hybrid.rt192k
edge-fusion-npu.balanced-fp256.v2-hybrid.rt192k
edge-fusion-npu.big-fp256.v2-hybrid-2m.rt192k
edge-fusion-npu.large-fp256.v2-hybrid-5m.rt192k
```

The tiny recipe keeps `n_fft=2048` and fits the 192 KiB fp16 state budget. The
compact and balanced recipes trade frequency resolution for more blocks/channels
while preserving the packed-state budget.

## Commands

Smoke:

```bash
./.venv/bin/python -m pytest EdgeFusionNPU/test_edge_fusion_npu.py -q
```

Export and compile:

```bash
./.venv/bin/python EdgeFusionNPU/export_compile.py \
  --preset tiny \
  --out-dir logs/npu_verify_general/edge_fusion_npu_tiny_packed_state_20260519 \
  --compile
```

Compact candidate:

```bash
./.venv/bin/python EdgeFusionNPU/export_compile.py \
  --preset compact \
  --n-freq 513 \
  --out-dir logs/npu_verify_general/edge_fusion_npu_compact_fp512_packed_state_20260519 \
  --compile
```

Recommended v2 candidate:

```bash
./.venv/bin/python EdgeFusionNPU/export_compile.py \
  --preset compact-v2-hybrid \
  --n-freq 513 \
  --out-dir logs/npu_verify_general/edge_fusion_npu_compact_fp512_v2_hybrid_20260519 \
  --compile
```

Larger quality candidate:

```bash
./.venv/bin/python EdgeFusionNPU/export_compile.py \
  --preset large-v2-hybrid-5m \
  --out-dir logs/npu_verify_general/edge_fusion_npu_large_fp256_v2_hybrid_5m_token_stride2_20260520 \
  --compile
```
