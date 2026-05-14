# TIGER Training Recipes

This repo includes DnR training recipes for the online TIGER family under
`recipes/dnr/models/`. They use `TIGER.training_wrapper.build_tiger_system`,
which keeps the training path on the same causal RI sequence and
`forward_sequence` cell path used by deployment export.

## Recipes

| recipe | `tiger_variant` | purpose |
|---|---|---|
| `recipes/dnr/models/tiger-npu-edge-v2.rt192k` | `npu-edge-v2` | current NPU graph-reduced V2 target |
| `recipes/dnr/models/tiger-npu-edge-v1.rt192k` | `npu-edge-v1` | prior edge preset for V1/V2 comparison |
| `recipes/dnr/models/tiger-npu-large.rt192k` | `npu-large` | larger NPU-friendly stacked separator |
| `recipes/dnr/models/tiger-ctx-deployable.rt192k` | `ctx-deployable` | ctx-state deployable baseline |
| `recipes/dnr/models/tiger-ctx-tiger-like.rt192k` | `ctx-tiger-like` | ctx-state, more paper-faithful ablation |
| `recipes/dnr/models/tiger-deployable.rt192k` | `deployable` | compact non-ctx deployable baseline |
| `recipes/dnr/models/tiger-tiger-like.rt192k` | `tiger-like` | non-ctx, more paper-faithful ablation |

All recipes inherit the DnR three-stem setup:

- sample rate: `44100`
- STFT: `n_fft=2048`, `hop_length=512`
- sources: speech, music, sfx
- causal startup packet: `256`
- RI analysis/synthesis window: `sqrt_hann`
- training wrapper chunk size: `8`

`tiger-npu-edge-v2` normalises every block with `NPURMSNormChannel`
(channel-axis RMSNorm, ~3 ONNX ops per site, no running statistics). This
replaces an earlier revision that used `BatchNorm2d` for a slightly smaller
graph; that revision trained poorly because the frame-by-frame forward pass
gave BN a tiny, highly-correlated effective batch per step and left a
train/eval distribution gap. The RMSNorm path has no such mismatch and is the
recommended default. The exported ONNX graph still sits well below the V1
node count (~800-1100 depending on compiler version); the BN graph (~600
nodes) is the only thing you give up. Training batch size still needs to be
above `1`; the recipe uses `batch_size: 4` and the training dataloader drops
the final incomplete batch.

### Training throughput and chunked unrolling

`TIGERNPUEdgeV2.forward_sequence` now unrolls the causal computation in
chunks of `chunk_size` frames instead of looping one STFT frame at a time.
With the default `chunk_size=8` carried through `build_tiger_system` and
`TIGERWaveformSeparator`, a 12-second clip (~1030 frames) runs ~130 GPU
calls instead of ~1030, which typically raises GPU utilisation from the
single-digit percent range into the 40-80% range during training.

The frame-attention sliding-window KV cache is preserved exactly across
chunk boundaries, so chunked training is numerically equivalent to the
old frame-by-frame path (the parity test in `TIGER/test_tiger_npu_edge_v2.py`
asserts `chunk_size=1` vs `chunk_size={4, 8}` match within fp32 tolerance).
The ONNX export path is unchanged: export still drives `forward_cell` with
`T=1`, so the deployment graph is bit-identical to before the chunked path
was added.

If you want to force the old behaviour for debugging, pass `chunk_size=1`
explicitly; you can also tune `chunk_size` upward (try 16 or 32) once your
GPU memory headroom allows. `chunk_size <= 0` processes the whole sequence
in a single call, which is occasionally useful for short validation clips.

## Training

Run inside the project Docker container:

```bash
cd /app/ASS
./recipes/dnr/models/tiger-npu-edge-v2.rt192k/train.sh
```

For quick single-process smoke runs, call the shared training app directly and
override the trainer limits:

```bash
cd /app/ASS
./.venv/bin/python aiaccel/aiaccel/torch/apps/train.py \
  recipes/dnr/models/tiger-npu-edge-v2.rt192k/config.yaml \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=1 \
  trainer.limit_val_batches=1 \
  datamodule.num_workers=0
```

## Deployment Checks After Training

The NPU export path is still the one-frame cell. For V2:

```bash
cd /app/ASS
./.venv/bin/python -m TIGER.test_tiger_npu_edge_v2 \
  --frames 8 \
  --onnx-out /tmp/tiger_npu_edge_v2.onnx

./.venv/bin/python tools/online/measure_npu_model_stats.py \
  --target tiger-edge-v2 \
  --out-dir /tmp/tiger_edge_v2_stats

./.venv/bin/python tools/online/export_verify_mlir.py \
  --target tiger-edge-v2 \
  --out-dir /tmp/tiger_edge_v2_export_verify \
  --forbid-op Tile,Expand,ConstantOfShape
```

The training wrapper applies the TIGER complex masks outside the exported NPU
cell, then reconstructs waveforms through the same causal RI helper. The
checkpointed parameters still belong to the TIGER core under
`model.core`.
