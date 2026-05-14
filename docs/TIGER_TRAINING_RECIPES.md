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
| `recipes/dnr/models/tf-mlpnet-edge.rt192k` | `tf-mlpnet-edge` | TF-MLPNet v3 edge preset (3.4M params, GLU + EMA global state, hidden=160, 8 blocks) |
| `recipes/dnr/models/tf-mlpnet-balance.rt192k` | `tf-mlpnet-balance` | TF-MLPNet v3 balance preset (6.4M params, hidden=208, 9 blocks, kf=7) |
| `recipes/dnr/models/tf-mlpnet-large.rt192k` | `tf-mlpnet-large` | TF-MLPNet v3 large preset (9.6M params, hidden=272, 8 blocks, feature_dim=96) |

All recipes inherit the DnR three-stem setup:

- sample rate: `44100`
- STFT: `n_fft=2048`, `hop_length=512`
- sources: speech, music, sfx
- causal startup packet: `256`
- RI analysis/synthesis window: `sqrt_hann`
- training wrapper chunk size: `8`

`tiger-npu-edge-v2` uses `BatchNorm2d` to keep the exported ONNX graph compact.
Keep the training batch size above `1`; the recipe uses `batch_size: 4` and
the training dataloader drops the final incomplete batch.

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

## TF-MLPNet v3 note

The `tf-mlpnet-*` variants use `TIGEREdgeMLPV3` (see
`TF-MLPNet/tf_mlpnet/tiger_edge_mlp_v3.py`), which swaps TIGER's `RecurrentKV`
separator for `EdgeTFMLPSeparatorV3`. v3 keeps the v2 state layout and export
wrapper exactly, and adds three quality levers on top:

- **GLU-gated mixers.** Both the frequency mixer and the causal time mixer use
  Gated Linear Units on the expanded hidden channels. Each block gets one
  extra 1x1 Conv2d per mixer; no attention / no bmm introduced.
- **Pre-LayerNorm.** Channel-only LayerNorm implemented by hand
  (mean/var/rsqrt/add/mul) so the exported ONNX graph stays within the NPU
  allowlist (no LayerNorm op).
- **EMA global state.** Instead of copying the latest frame, the per-block
  global context is a learnable-alpha EMA: `g' = sigmoid(alpha)*g + (1 - sigmoid(alpha))*update_proj(x_last)`.
  Same state footprint as v2, far richer temporal memory.

Preset sizing (numbers verified by `TF-MLPNet/tests/test_tiger_edge_mlp_smoke.py`
on a live build with `num_sources=3`, `win=2048`, `stride=512`, 8-band DnR
split summing to 1025 = enc_dim):

| preset              | hidden | L | k_f | feature_dim | total params | sep params | state (fp16) |
|---------------------|-------:|--:|----:|------------:|-------------:|-----------:|-------------:|
| `tf-mlpnet-edge`    |    160 | 8 |   5 |          48 |        3.44M |      3.13M |      105 KiB |
| `tf-mlpnet-balance` |    208 | 9 |   7 |          72 |        6.40M |      5.94M |      166 KiB |
| `tf-mlpnet-large`   |    272 | 8 |   7 |          96 |        9.62M |      9.01M |      179 KiB |

All three honour `(kernel_size-1)*dilation < 14` on every Conv2d and fit
inside the 192 KiB fp16 DSP state budget. The NPU constraint
`(time_kernel-1)*dilation < 14` is enforced at construction time in
`TIGEREdgeMLPV3.__init__`, and the frequency-kernel bound `(k_f-1) < 14`
is enforced inside `_GLUFreqMixer`.

The 8-band DnR split `(10, 28, 56, 93, 186, 186, 279, 187)` is passed via
TIGER's existing `pre_calc_bands` hook, overriding the stock
`calculate_band_widths` (which would produce ~67 single-bin bands at this
configuration and blow the state budget regardless of channel count).

v2 (`TIGEREdgeMLP`, ~0.5M params) is still importable from `tf_mlpnet` and
covered by its own smoke tests; it's retained for regressions and as the
minimal-footprint reference.

## DolphinSFCNPU DnR recipes

DolphinSFCNPU does not share TIGER's encoder/decoder, so it has its own
training wrapper and recipe family:

| recipe | `preset` | purpose |
|---|---|---|
| `recipes/dnr/models/dolphin-sfc-npu.edge-small` | `edge_small` | structural / export smoke |
| `recipes/dnr/models/dolphin-sfc-npu.large-6m` | `large_6m` | first performance target |
| `recipes/dnr/models/dolphin-sfc-npu.large-8m` | `large_8m` | larger quality-oriented variant |

These recipes use `DolphinSFCNPU.training_wrapper.build_dolphin_sfc_npu_system`
with `n_fft=4096`, `hop_length=1024` (matching the BandSCNetNPU DnR setup),
and feed the packed-real STFT contract through the existing
`OnlineModelWrapper` / `SupTask` pipeline.
