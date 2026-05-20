# BandSFCNetNPU

BandSFCNetNPU is an opt-in model family that combines the strongest local
pieces from `BandSCNetNPU` and `spectral_feature_compression`:

- SFC-style adaptive frequency transport compresses full STFT bins into a
  smaller latent band axis.
- BandSCNet-style cross-band and narrow-band stages run on those compressed
  tokens.
- Optional pooled channel mixers add parameter capacity without adding
  streaming cache.
- STFT/iSTFT stay outside the deployed graph.

## Contract

Deployable core:

```text
input:  [B, 2*n_chan, T, F]
output: [B, 2*n_src*n_chan, T, F]
state:  encoder causal-conv cache + per-stage narrow-band caches
```

The `safe` preset uses soft-band SFC transport. The `quality` presets use the
NPU-friendly cross-attention SFC transport and should be treated as quality
probes until ONNX and ONE compilation are measured on the target shape.

## Presets

| Preset | Transport | Channels | Bands | Stages | Dilation | Purpose |
|---|---|---:|---:|---:|---|---|
| `safe` | soft-band | 32 | 64 | 4 | 1,1,2,4 | first deployable baseline |
| `quality` | cross-attn | 32 | 64 | 5 | 1,1,2,4,6 | main quality candidate |
| `quality6m` | cross-attn | 40 | 64 | 4 | 1,2,4,6 | high-capacity probe |

## Smoke

```bash
cd /home/cmj/works/ASS
./.venv/bin/python -m BandSFCNetNPU.test_band_sfc_net_npu
```

## ONE Compile Notes

The current ONE quantization flow needs the same SFC-family optimize pass
documented in `OPERATION_MANUAL_PYTORCH_TO_ONE_NPU.md`:

```ini
[one-optimize]
replace_non_const_fc_with_batch_matmul=True
```

Without that pass, quantization can fail with:

```text
Unsupported non const input /MatMul/tr
```

Validated local artifacts on 2026-05-20:

| Recipe | ONNX nodes | Params | State fp16 | ONE result |
|---|---:|---:|---:|---|
| `band-sfc-net-npu.safe.rt192k.fp513` | 518 | 442,251 | 128.12 KiB | `model.circle`, `model.opt.circle`, `model.q.circle` PASS |
| `band-sfc-net-npu.quality.rt192k.fp512` | 1,074 | 2,092,715 | 186.00 KiB | `model.circle`, `model.opt.circle`, `model.q.circle` PASS |

Artifact roots:

```text
logs/npu_verify_general/band_sfc_net_npu_safe_fp513_core_v3_batchmatmul_20260520
logs/npu_verify_general/band_sfc_net_npu_quality_fp512_batchmatmul_20260520
```

## Recipes

DnR sibling recipes are under:

```text
recipes/dnr/models/band-sfc-net-npu.safe.rt192k.fp512
recipes/dnr/models/band-sfc-net-npu.safe.rt192k.fp513
recipes/dnr/models/band-sfc-net-npu.quality.rt192k.fp512
recipes/dnr/models/band-sfc-net-npu.quality6m.fp256
```

Use `safe.rt192k.fp513` for the first strict export/compile bring-up: it keeps
`n_fft=1024` and disables the frequency pre/post-projector so the ONNX graph
tests the new core directly. The fp512/fp256 recipes keep the SFC frequency
preprocessor enabled for quality experiments and should be audited as full
wrapper graphs before deployment.
