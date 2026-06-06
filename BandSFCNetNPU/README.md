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
state:  encoder causal-conv cache when non-empty + per-stage narrow-band/local caches
```

Zero-context caches are intentionally omitted from the public streaming state.
For example, the fp512 `adaptive_mel_loco_cnb_*` presets have no encoder cache
and expose 10 non-empty state tensors: local `[1, 32, 2, 48]` and FSMN
`[1, 32, 9, 48]` for each of 5 stages.  This avoids zero-sized state tensors in
stateful ONNX/ONE quantization while preserving `forward_stream` compatibility
with older 11-state exports.

The `safe` preset uses soft-band SFC transport. The `quality` presets use the
NPU-friendly cross-attention-query SFC transport and should be treated as
quality probes until ONNX and ONE compilation are measured on the target shape.

## Explicit-query variants

BandSFCNetNPU now exposes both explicit SFC query styles:

- `*_soft_band_query` / `*_soft_query`: compressor emits K-band latent tokens
  plus K-band query tokens, and the decoder uses the query side-path for
  adaptive K -> F expansion. This is the lower-risk explicit-query path.
- `*_crossattn_query`: F -> K encoder and K -> F decoder use the existing
  NPU-safe cross-attention query blocks (`Conv2d + bmm/MatMul + Softmax`). This
  is the explicit name for the previously used quality transport.

The deployable fp512 query presets are:

```text
safe_soft_band_query              # compile-smoke / low-capacity baseline
safe_crossattn_query              # compile-smoke / low-capacity baseline
balanced_soft_band_query          # recommended useful-capacity query target
balanced_crossattn_query          # recommended useful-capacity query target
quality_soft_band_query           # quality probe, larger graph
quality_crossattn_query           # quality probe, existing cross-attn path
rt_plus_soft_band_query           # residual-head quality probe
rt_plus_crossattn_query           # residual-head quality probe
causal_cnb_soft_band_query        # Proposal B CNB block, compile-smoke soft-query transport
causal_cnb_crossattn_query        # Proposal B CNB block, compile-smoke cross-attn transport
causal_cnb_balanced_soft_band_query     # useful-capacity CNB soft-query target
causal_cnb_balanced_crossattn_query     # useful-capacity CNB cross-attn target
adaptive_mel_loco_cnb_soft_band_query   # high-capacity Loco-CNB target; use mostly for distillation/fine-tune
adaptive_mel_loco_cnb_crossattn_query   # stronger transport ablation for the high-capacity Loco-CNB target
adaptive_mel_loco_cnb_stable_soft_band_query  # recommended supervised stage-1 fix
adaptive_mel_loco_cnb_stable_crossattn_query  # stable cross-attn transport ablation
adaptive_mel_loco_cnb_band56_soft_band_query  # 56-band bottleneck ablation
adaptive_mel_loco_cnb_clean_soft_band_query   # no frequency-pooled stage/IO capacity
```

`balanced_*` is the preferred starting point for training: it uses 40 latent
channels and large frequency-pooled channel mixers to reach ~4M parameters while
keeping fp16 streaming state around 160 KiB. The config exposes the key controls
for parameter count, latency, and quality: `channels`, `n_bands`, `num_stages`,
`time_kernel`, `freq_kernel`, `dilation_cycle`, `routing_normalization`,
`use_attn`, `attn_window`, `num_heads`, `head_dim`, `pooled_mixer_hidden`, and
`pooled_mixer_hidden_schedule`.

`adaptive_mel_loco_cnb_*` is the follow-up to the balanced CNB branch when
`causal_cnb_balanced_*` trains below target quality.  It is intentionally not
just another pooled-mixer variant: it adds adaptive overlapped-mel band priors,
state-free encoder capacity before compression, local TF-Locoformer-style
current/detail modeling around each CNB stage, state-free decoder capacity after
expansion, and a safely initialized residual-capable output head.

The initial `adaptive_mel_loco_cnb_soft_band_query` shape is high-capacity but
proved structurally too dominated by giant frequency-pooled mixers during early
supervised training.  For supervised stage-1, prefer
`adaptive_mel_loco_cnb_stable_soft_band_query`: it widens the real local/CNB
state path to 36 channels, reduces pooled mixer hidden sizes, disables the
unbounded residual head, and keeps fp16 state below 192 KiB.  Use
`adaptive_mel_loco_cnb_band56_soft_band_query` as the bottleneck/detail ablation
if the stable 48-band recipe still underfits Effects or high-frequency detail.
If the pooled-mixer idea itself looks too risky, use
`adaptive_mel_loco_cnb_clean_soft_band_query`: it replaces stage pooled mixers
with pointwise per-band mixers and removes frequency-pooled IO capacity.  It is
much smaller, so treat it as the clean structural diagnostic or distillation
student rather than the highest-capacity supervised target.

`quality6m_*` query variants are available for research, but at fp512 they are
above the current deployment budget (~9.1M params and ~218 KiB fp16 state), so
do not use them as strict NPU targets without resizing.

## Presets

| Preset | Transport | Channels | Bands | Stages | Dilation | Purpose |
|---|---|---:|---:|---:|---|---|
| `safe` | soft-band | 32 | 64 | 4 | 1,1,2,4 | first deployable baseline |
| `safe_soft_band_query` | soft-band-query | 32 | 64 | 4 | 1,1,2,4 | explicit-query safe ablation |
| `safe_crossattn_query` | cross-attn-query | 32 | 64 | 4 | 1,1,2,4 | cross-attn transport with safe stage shape |
| `balanced_soft_band_query` | soft-band-query | 40 | 64 | 4 | 1,1,2,4 | recommended useful-capacity soft-query target |
| `balanced_crossattn_query` | cross-attn-query | 40 | 64 | 4 | 1,1,2,4 | recommended useful-capacity cross-attn target |
| `quality` / `quality_crossattn_query` | cross-attn-query | 32 | 64 | 5 | 1,1,2,4,6 | main quality candidate |
| `quality_soft_band_query` | soft-band-query | 32 | 64 | 5 | 1,1,2,4,6 | query ablation with quality stage shape |
| `rt_plus` / `rt_plus_crossattn_query` | cross-attn-query + residual | 32 | 64 | 5 | 1,1,2,4,6 | RT+ residual candidate |
| `rt_plus_soft_band_query` | soft-band-query + residual | 32 | 64 | 5 | 1,1,2,4,6 | RT+ residual query ablation |
| `causal_cnb_soft_band_query` | soft-band-query + CNB | 24 | 48 | 5 | CNB FSMN 1,2,3 | Proposal B block, strict-state smoke target |
| `causal_cnb_crossattn_query` | cross-attn-query + CNB | 24 | 48 | 5 | CNB FSMN 1,2,3 | Proposal B cross-attn compile-smoke ablation |
| `causal_cnb_balanced_soft_band_query` | soft-band-query + CNB + pooled mixers | 32 | 48 | 5 | CNB FSMN 1,2,3 | useful-capacity CNB training target |
| `causal_cnb_balanced_crossattn_query` | cross-attn-query + CNB + pooled mixers | 32 | 48 | 5 | CNB FSMN 1,2,3 | useful-capacity CNB cross-attn target |
| `adaptive_mel_loco_cnb_soft_band_query` | adaptive-mel soft-query + Loco-CNB + IO capacity + residual | 32 | 48 | 5 | local 2 + FSMN 1,2,3 | high-capacity/distill target |
| `adaptive_mel_loco_cnb_crossattn_query` | adaptive-mel cross-attn-query + Loco-CNB + IO capacity + residual | 32 | 48 | 5 | local 2 + FSMN 1,2,3 | high-capacity transport ablation |
| `adaptive_mel_loco_cnb_stable_soft_band_query` | adaptive-mel soft-query + wider Loco-CNB + reduced pooled capacity | 36 | 48 | 5 | local 2 + FSMN 1,2,3 | recommended supervised stage-1 fix |
| `adaptive_mel_loco_cnb_stable_crossattn_query` | adaptive-mel cross-attn-query + stable Loco-CNB shape | 36 | 48 | 5 | local 2 + FSMN 1,2,3 | stable transport ablation |
| `adaptive_mel_loco_cnb_band56_soft_band_query` | adaptive-mel soft-query + 56-band bottleneck ablation | 28 | 56 | 5 | local 2 + FSMN 1,2,3 | detail/bottleneck ablation |
| `adaptive_mel_loco_cnb_clean_soft_band_query` | adaptive-mel soft-query + pointwise per-band stage mixers | 36 | 48 | 5 | local 2 + FSMN 1,2,3 | clean structural diagnostic/distill student |
| `quality6m` | cross-attn-query | 40 | 64 | 4 | 1,2,4,6 | high-capacity probe, not strict fp512 deploy |

## Smoke

```bash
cd /home/cmj/works/ASS
./.venv/bin/python -m BandSFCNetNPU.test_band_sfc_net_npu
```

## Proposal B CNB notes

The CNB presets use explicit classes from the research sketch:
`CrossBandMixer`, `CausalFSMNBandMixer`, `CompressedSelfAttentionFusion`, and
`CausalCNBBlock`.  The document sketch's `kernel_t=5` with dilations
`(1, 2, 4)` is not accepted by the current repo validator because the last
branch has span `(5 - 1) * 4 = 16 >= 14`.  The deployable presets therefore use
`cnb_dilation_schedule=(1, 2, 3)`, preserving the CNB block structure while
staying inside the current NPU kernel-span rule.

The plain `causal_cnb_*` presets are intentionally tiny compile-smoke models.
For training, prefer `causal_cnb_balanced_*`: they raise width to 32 and add
large frequency-pooled channel mixers after each CNB block.  These mixers add
millions of parameters without increasing persistent streaming cache.

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
| `band-sfc-net-npu.safe.soft-query.rt192k.fp512` | 635 | 446,476 | 128.00 KiB | `model.circle`, `model.opt.circle`, `model.q.circle` PASS |
| `band-sfc-net-npu.safe.crossattn-query.rt192k.fp512` | 721 | 457,099 | 128.00 KiB | `model.circle`, `model.opt.circle`, `model.q.circle` PASS |
| `band-sfc-net-npu.balanced.soft-query.rt192k.fp512` | 635 | 4,066,492 | 160.00 KiB | `model.circle`, `model.opt.circle`, `model.q.circle` PASS |
| `band-sfc-net-npu.balanced.crossattn-query.rt192k.fp512` | 721 | 4,083,627 | 160.00 KiB | `model.circle`, `model.opt.circle`, `model.q.circle` PASS |
| `band-sfc-net-npu.quality.soft-query.rt192k.fp512` | 1,424 | 2,082,092 | 186.00 KiB | `model.circle`, `model.opt.circle`, `model.q.circle` PASS; ONNX audit flags `And/Less/GreaterOrEqual` |
| `band-sfc-net-npu.adaptive-mel-loco-cnb.soft-query.rt192k.fp512keep475` | 701 non-Constant simplified nodes | 5,741,397 | 165.00 KiB | stateless and stateful `model.circle`, `model.opt.circle`, `model.q.circle` PASS |
| `band-sfc-net-npu.adaptive-mel-loco-cnb.crossattn-query.rt192k.fp512keep475` | 739 non-Constant simplified nodes | 5,752,548 | 165.00 KiB | stateless and stateful `model.circle`, `model.opt.circle`, `model.q.circle` PASS |
| `band-sfc-net-npu.adaptive-mel-loco-cnb.stable-soft-query.rt192k.fp512keep475` | 701 non-Constant simplified nodes | 2,852,491 | 185.62 KiB | stateless and stateful `model.circle`, `model.opt.circle`, `model.q.circle` PASS |
| `band-sfc-net-npu.adaptive-mel-loco-cnb.stable-soft-query.residual-sfx.rt192k.fp512keep475` | 687 non-Constant simplified nodes | 2,852,417 | 185.62 KiB | stateless and stateful `model.circle`, `model.opt.circle`, `model.q.circle` PASS; distill recipe uses same deploy graph |
| `band-sfc-net-npu.adaptive-mel-loco-cnb.stable-crossattn-query.rt192k.fp512keep475` | audited by verifier | 2,866,770 | 185.62 KiB | stateless `model.circle`, `model.opt.circle`, `model.q.circle` PASS |
| `band-sfc-net-npu.adaptive-mel-loco-cnb.band56-soft-query.rt192k.fp512keep475` | 701 non-Constant simplified nodes | 2,210,395 | 168.44 KiB | stateless `model.circle`, `model.opt.circle`, `model.q.circle` PASS |
| `band-sfc-net-npu.adaptive-mel-loco-cnb.clean-soft-query.rt192k.fp512keep475` | 637 non-Constant simplified nodes | 876,604 | 185.62 KiB | stateless and stateful `model.circle`, `model.opt.circle`, `model.q.circle` PASS |

The stateful adaptive-mel Loco-CNB exports use 10 state tensors and also pass
channel-wise uint8 quantization.  Artifact roots are listed below.

Artifact roots:

```text
logs/npu_verify_general/band_sfc_net_npu_safe_fp513_core_v3_batchmatmul_20260520
logs/npu_verify_general/band_sfc_net_npu_quality_fp512_batchmatmul_20260520
logs/npu_verify_general/band_sfc_safe_soft_query_20260603
logs/npu_verify_general/band_sfc_safe_crossattn_query_20260603
logs/npu_verify_general/band_sfc_balanced_soft_query_20260603
logs/npu_verify_general/band_sfc_balanced_crossattn_query_20260603
logs/npu_verify_general/band_sfc_quality_soft_query_20260603
logs/npu_verify_general/band_sfc_adaptive_mel_loco_cnb_soft_query_20260604
logs/npu_verify_general/band_sfc_adaptive_mel_loco_cnb_crossattn_query_20260604
logs/npu_verify_general/band_sfc_adaptive_mel_loco_cnb_streaming_soft_verify_20260604
logs/npu_verify_general/band_sfc_adaptive_mel_loco_cnb_streaming_cross_verify_20260604
logs/npu_verify_general/band_sfc_adaptive_mel_loco_cnb_stable_soft_query_20260605
logs/npu_verify_general/band_sfc_adaptive_mel_loco_cnb_stable_soft_query_streaming_20260605
logs/npu_verify_general/band_sfc_adaptive_mel_loco_cnb_stable_crossattn_query_20260605
logs/npu_verify_general/band_sfc_adaptive_mel_loco_cnb_band56_soft_query_20260605
logs/npu_verify_general/band_sfc_adaptive_mel_loco_cnb_clean_soft_query_20260605
logs/npu_verify_general/band_sfc_adaptive_mel_loco_cnb_clean_soft_query_streaming_20260605
```

## Recipes

DnR sibling recipes are under:

```text
recipes/dnr/models/band-sfc-net-npu.safe.rt192k.fp512
recipes/dnr/models/band-sfc-net-npu.safe.rt192k.fp513
recipes/dnr/models/band-sfc-net-npu.safe.soft-query.rt192k.fp512
recipes/dnr/models/band-sfc-net-npu.safe.crossattn-query.rt192k.fp512
recipes/dnr/models/band-sfc-net-npu.balanced.soft-query.rt192k.fp512
recipes/dnr/models/band-sfc-net-npu.balanced.crossattn-query.rt192k.fp512
recipes/dnr/models/band-sfc-net-npu.quality.rt192k.fp512
recipes/dnr/models/band-sfc-net-npu.quality.soft-query.rt192k.fp512
recipes/dnr/models/band-sfc-net-npu.quality.crossattn-query.rt192k.fp512
recipes/dnr/models/band-sfc-net-npu.rt-plus.soft-query.distill.rt192k.fp512
recipes/dnr/models/band-sfc-net-npu.rt-plus.crossattn-query.distill.rt192k.fp512
recipes/dnr/models/band-sfc-net-npu.adaptive-mel-loco-cnb.soft-query.rt192k.fp512keep475
recipes/dnr/models/band-sfc-net-npu.adaptive-mel-loco-cnb.crossattn-query.rt192k.fp512keep475
recipes/dnr/models/band-sfc-net-npu.adaptive-mel-loco-cnb.soft-query.distill.rt192k.fp512keep475
recipes/dnr/models/band-sfc-net-npu.adaptive-mel-loco-cnb.crossattn-query.distill.rt192k.fp512keep475
recipes/dnr/models/band-sfc-net-npu.adaptive-mel-loco-cnb.stable-soft-query.rt192k.fp512keep475
recipes/dnr/models/band-sfc-net-npu.adaptive-mel-loco-cnb.stable-soft-query.residual-sfx.rt192k.fp512keep475
recipes/dnr/models/band-sfc-net-npu.adaptive-mel-loco-cnb.stable-soft-query.residual-sfx.distill.rt192k.fp512keep475
recipes/dnr/models/band-sfc-net-npu.adaptive-mel-loco-cnb.stable-crossattn-query.rt192k.fp512keep475
recipes/dnr/models/band-sfc-net-npu.adaptive-mel-loco-cnb.band56-soft-query.rt192k.fp512keep475
recipes/dnr/models/band-sfc-net-npu.adaptive-mel-loco-cnb.clean-soft-query.rt192k.fp512keep475
recipes/dnr/models/band-sfc-net-npu.quality6m.fp256
```

Use `safe.rt192k.fp513` for the first strict export/compile bring-up: it keeps
`n_fft=1024` and disables the frequency pre/post-projector so the ONNX graph
tests the new core directly. The fp512/fp256 recipes keep the SFC frequency
preprocessor enabled for quality experiments and should be audited as full
wrapper graphs before deployment.
