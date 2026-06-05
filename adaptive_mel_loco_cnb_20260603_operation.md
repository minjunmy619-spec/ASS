# Adaptive Mel Loco-CNB BandSFCNetNPU Operation

Date: 2026-06-03

## Goal

Implement a stronger strict-NPU BandSFC variant after the existing
`causal_cnb_balanced_*` models trained below the desired quality.  The new
variant is intentionally not just a wider pooled-mixer preset.  It addresses the
main suspected weaknesses of the previous BandSFC variants:

1. improve features before the SFC bottleneck,
2. use overlapped adaptive/mel band priors,
3. add local TF detail modeling rather than only pooled channel capacity,
4. strengthen reconstruction after SFC expansion,
5. keep CNB/FSMN causal temporal memory,
6. keep the model suitable for teacher-student distillation from an offline
   SFC/TF-Locoformer teacher.

## Implemented model family

Added a new `stage_type="loco_cnb"` path in `BandSFCNetNPU/band_sfc_net_npu.py`.

New classes:

- `LocalTFLocoMixer`
  - RMSNorm over the frequency/band axis,
  - sigmoid-gated pointwise projections,
  - causal depthwise time `Conv2d`,
  - depthwise compressed-band `Conv2d`,
  - pointwise gated channel FFN,
  - small learnable residual scales initialized to `0.1`.
- `CausalLocoCNBBlock`
  - `LocalTFLocoMixer`,
  - `CrossBandMixer`,
  - `CausalFSMNBandMixer`,
  - `CompressedSelfAttentionFusion`,
  - optional `PooledChannelMixer`.

The activations inside the new local block use `value * sigmoid(gate)` instead
of direct SiLU so the export graph uses basic NPU-friendly ops.

## Front/back-end changes

`BandSFCNetNPU` now supports:

- `band_spec_type="adaptive_mel"`
  - uses `AdaptiveMelBandSpec2d`, with overlapped mel-style bands and denser
    low-frequency allocation.
  - after the training-pipeline review, adaptive-mel priors are now aware of
    the true hybrid fp512 frequency axis used by `fp512keep475` preprocessing.
    The first 475 core bins retain original 2048-FFT bin centers and the
    remaining projected high-frequency bins use weighted physical centers from
    the preprocessor analysis matrix.
- encoder-side capacity mixers:
  - `encoder_capacity_mixer_hidden`,
  - `encoder_capacity_mixer_layers`.
- decoder-side capacity mixers:
  - `decoder_capacity_mixer_hidden`,
  - `decoder_capacity_mixer_layers`.

The capacity mixers are frequency-pooled and state-free.  They add trainable
capacity before compression and after expansion without increasing persistent
streaming cache.

## New presets

Added to `BandSFCNetNPU/presets.py`:

- `adaptive_mel_loco_cnb_soft_query`
- `adaptive_mel_loco_cnb_soft_band_query`
- `adaptive_mel_loco_cnb_crossattn_query`

Recommended first training target:

```text
adaptive_mel_loco_cnb_soft_band_query
```

Cross-attention-query quality ablation:

```text
adaptive_mel_loco_cnb_crossattn_query
```

## Final preset shape

```text
n_bands = 48
channels = 32
num_stages = 5
transport = soft_band_query or crossattn_query
band_spec_type = adaptive_mel
stage_type = loco_cnb
cnb_kernel = 4
cnb_dilation_schedule = (1, 2, 3)
loco_expansion = 1
loco_ffn_expansion = 2
loco_time_kernel = 3
loco_band_kernel = 3
pooled_mixer_hidden = 8192 per stage
encoder_capacity_mixer_hidden = 4096, layers = 2
decoder_capacity_mixer_hidden = 4096, layers = 2
residual_head = true
```

The first design used `loco_expansion=2`, but that made fp512 state `195 KiB`.
The final design uses `loco_expansion=1`, which keeps the added local time cache
while staying below the `192 KiB` fp16 layer-cache target.

## State budget detail

For the final strict fp512 shape:

- Local time cache per stage:
  - `2 frames * 32 channels * 48 bands`
- FSMN cache per stage:
  - `9 frames * 32 channels * 48 bands`
- Total per stage:
  - `11 frames * 32 channels * 48 bands`
- Five stages:
  - `55 context frames`

Measured state:

| Preset | Params | fp16 state | Context |
|---|---:|---:|---:|
| `adaptive_mel_loco_cnb_soft_band_query` | 5,741,397 | 168,960 B / 165.00 KiB | 55 frames |
| `adaptive_mel_loco_cnb_crossattn_query` | 5,752,548 | 168,960 B / 165.00 KiB | 55 frames |

2026-06-04 streaming-export fix:

- The Loco-CNB fp512 preset uses `time_kernel=1` in the SFC encoder, so the
  encoder stream cache has shape `[1, 32, 0, 512]`.
- That zero-sized passthrough cache is not needed for runtime and caused ONE
  quantization of the stateful graph to fail with:

```text
Unsupported Op for const inputs: Initializer_next_state_0
```

- `BandSFCNetNPU.init_stream_state()` now omits zero-context encoder cache
  tensors from the public streaming state while `forward_stream()` still accepts
  the old 11-state layout for compatibility.
- The adaptive-mel Loco-CNB stateful export now exposes 10 non-empty state
  tensors: per stage, local cache `[1, 32, 2, 48]` and FSMN cache
  `[1, 32, 9, 48]`, repeated for 5 stages.
- Total persistent fp16 state is unchanged: `168,960 B / 165.00 KiB`.

## Recipes

Added supervised recipes:

```text
recipes/dnr/models/band-sfc-net-npu.adaptive-mel-loco-cnb.soft-query.rt192k.fp512keep475/config.yaml
recipes/dnr/models/band-sfc-net-npu.adaptive-mel-loco-cnb.crossattn-query.rt192k.fp512keep475/config.yaml
```

Added distillation recipes:

```text
recipes/dnr/models/band-sfc-net-npu.adaptive-mel-loco-cnb.soft-query.distill.rt192k.fp512keep475/config.yaml
recipes/dnr/models/band-sfc-net-npu.adaptive-mel-loco-cnb.crossattn-query.distill.rt192k.fp512keep475/config.yaml
```

The soft-query recipe is the main training target.  The cross-attention recipe
is a quality/expressiveness ablation after ONNX/ONE audits pass.  The distill
recipes use `TeacherStudentDistillationTask`, fail fast unless
`teacher_checkpoint_path` is set, and use the repo's SFC/TF-Locoformer-lite
teacher builder.

## Proposal builder

Added to `spectral_feature_compression/core/model/proposed_separation_models.py`:

```python
build_adaptive_mel_loco_cnb_npu_system(...)
```

This gives the new model a clean proposal-level entry point for experiments and
future distillation tasks.

## ONNX audit

Config-only one-frame export command:

```bash
cd /home/cmj/works/ASS
PYTHONPATH=. .venv/bin/python tools/online/export_onnx_online_model.py \
  recipes/dnr/models/band-sfc-net-npu.adaptive-mel-loco-cnb.soft-query.rt192k.fp512keep475/config.yaml \
  --out tmp/adaptive_mel_loco_cnb_onnx/model.onnx \
  --device cpu \
  --n-chan 1 \
  --frames 1 \
  --freqs 512 \
  --opset 14 \
  --check
```

Raw ONNX export:

```text
ONNX checker: passed
Disallowed ops: none
```

The raw legacy exporter still emits shape/padding scaffolding such as
`ConstantOfShape`, but simplification removes it.

Simplified ONNX summary:

```text
simplify_ok True
nodes=731
Add=101, Concat=2, Conv=119, Div=44, MatMul=13, Mul=161,
ReduceMean=52, ReduceSum=1, Reshape=28, Sigmoid=39, Slice=83,
Softmax=6, Sqrt=43, Sub=3, Transpose=36
Expand=0, ConstantOfShape=0, Tile=0, ScatterND=0, Unflatten=0
```

## Full ONE verification

### Stateless one-frame core graph

After the ONNX/simplification audit, the full ONE verifier was run for both new
supervised recipes:

```bash
cd /home/cmj/works/ASS
.venv/bin/python tools/online/verify_npu_variants.py \
  --mode recipe \
  --recipe-name-contains adaptive-mel-loco-cnb.soft-query.rt192k \
  --run-name band_sfc_adaptive_mel_loco_cnb_soft_query_20260604 \
  --force-onnxsim-large-shape-ops \
  --quantize-layer-fallback

.venv/bin/python tools/online/verify_npu_variants.py \
  --mode recipe \
  --recipe-name-contains adaptive-mel-loco-cnb.crossattn-query.rt192k \
  --run-name band_sfc_adaptive_mel_loco_cnb_crossattn_query_20260604 \
  --force-onnxsim-large-shape-ops \
  --quantize-layer-fallback
```

Both passed artifact-verified `import -> optimize -> quantize` with channel-wise
uint8 quantization.  The layer-wise fallback was not used.

| Recipe | Export | onnxsim | Calibration | ONE | Quant granularity | Artifact root |
|---|---:|---:|---:|---:|---|---|
| `band-sfc-net-npu.adaptive-mel-loco-cnb.soft-query.rt192k.fp512keep475` | PASS | PASS | PASS | PASS | channel | `logs/npu_verify_general/band_sfc_adaptive_mel_loco_cnb_soft_query_20260604` |
| `band-sfc-net-npu.adaptive-mel-loco-cnb.crossattn-query.rt192k.fp512keep475` | PASS | PASS | PASS | PASS | channel | `logs/npu_verify_general/band_sfc_adaptive_mel_loco_cnb_crossattn_query_20260604` |

Simplified stateless graph summaries from those runs:

| Variant | Non-Constant nodes | Main simplified ops |
|---|---:|---|
| soft-query | 701 | `Conv=119`, `MatMul=13`, `Softmax=6`, `Slice=73`, no `ConstantOfShape/Expand/Tile` |
| crossattn-query | 739 | `Conv=127`, `MatMul=15`, `Softmax=7`, `Slice=79`, no `ConstantOfShape/Expand/Tile` |

### Stateful streaming graph

Stateful streaming exports were also checked after removing the zero-sized
encoder cache from the public state contract.  `verify_npu_variants.py` now has a
`--streaming` flag that exports `forward_stream` with flattened state
inputs/outputs before running onnxsim, calibration H5 generation, and ONE
`import -> optimize -> quantize`:

```bash
cd /home/cmj/works/ASS
.venv/bin/python tools/online/verify_npu_variants.py \
  --mode recipe \
  --recipe-name-contains adaptive-mel-loco-cnb.soft-query.rt192k \
  --run-name band_sfc_adaptive_mel_loco_cnb_streaming_soft_verify_20260604 \
  --force-onnxsim-large-shape-ops \
  --quantize-layer-fallback \
  --streaming

.venv/bin/python tools/online/verify_npu_variants.py \
  --mode recipe \
  --recipe-name-contains adaptive-mel-loco-cnb.crossattn-query.rt192k \
  --run-name band_sfc_adaptive_mel_loco_cnb_streaming_cross_verify_20260604 \
  --force-onnxsim-large-shape-ops \
  --quantize-layer-fallback \
  --streaming
```

The generated stateful graphs expose `x + 10 state` inputs and
`y + 10 next_state` outputs.  They passed ONNX checker, onnxsim, calibration H5
creation, ONE import, ONE optimize, and ONE quantize.  Standard verifier
artifacts are under:

```text
logs/npu_verify_general/band_sfc_adaptive_mel_loco_cnb_streaming_soft_verify_20260604
logs/npu_verify_general/band_sfc_adaptive_mel_loco_cnb_streaming_cross_verify_20260604
```

Stateful simplified graph summaries:

| Variant | Non-Constant nodes | State tensors | ONE result |
|---|---:|---:|---|
| soft-query | 736 | 10 | `model.circle`, `model.opt.circle`, `model.q.circle` PASS |
| crossattn-query | 774 | 10 | `model.circle`, `model.opt.circle`, `model.q.circle` PASS |

## Validation

BandSFC smoke tests:

```bash
cd /home/cmj/works/ASS
PYTHONPATH=. .venv/bin/python -m BandSFCNetNPU.test_band_sfc_net_npu
```

Result:

```text
all BandSFCNetNPU smoke tests passed
```

Proposal-model and recipe-regression tests:

```bash
cd /home/cmj/works/ASS
PYTHONPATH=. .venv/bin/python -m pytest tests/test_proposed_separation_models.py -q
```

Result after training-pipeline fixes:

```text
17 passed
```

Size/state inspection:

```bash
cd /home/cmj/works/ASS
PYTHONPATH=. .venv/bin/python - <<'PY'
import torch
from BandSFCNetNPU.presets import build_band_sfc_net_npu_preset
for preset in ("adaptive_mel_loco_cnb_soft_band_query", "adaptive_mel_loco_cnb_crossattn_query"):
    model = build_band_sfc_net_npu_preset(
        preset, n_freq=512, n_fft=1022, sample_rate=44100, n_src=3, n_chan=1
    ).eval()
    params = sum(p.numel() for p in model.parameters())
    state = model.state_size_bytes(dtype=torch.float16)
    print(preset, params, state, f"{state/1024:.2f} KiB", model.stream_context_frames())
PY
```

Result:

```text
adaptive_mel_loco_cnb_soft_band_query 5741397 168960 165.00 KiB 55
adaptive_mel_loco_cnb_crossattn_query 5752548 168960 165.00 KiB 55
```

## Training-pipeline review fixes

Fixed issues found during the training-pipeline review:

- Adaptive-mel priors now use physical hybrid fp512 bin centers instead of a
  naive uniform `0..Nyquist` axis when frequency preprocessing is enabled.
- The new supervised recipe now uses the inherited
  `band_sfc_freq_preprocess_keep_bins` / `band_sfc_freq_preprocess_target_bins`
  variable names instead of misleading no-op unprefixed keys.
- The new recipe's `task.model` block uses the same two-space indentation style
  as parent recipes so the lightweight config parser and Hydra path both see the
  intended overrides.
- `build_adaptive_mel_loco_cnb_npu_system` now accepts and forwards the full
  Loco-CNB override set used by the recipe/wrapper path.
- The lightweight ONNX export config parser now parses YAML `null` values as
  `None`, which matters for distillation recipe metadata.
- Added regression tests for recipe resolution, distillation recipe declaration,
  and hybrid-frequency adaptive-mel priors.

## Distillation recommendation

For training, use this as a causal student and distill from the strongest
offline teacher available in the repo, ideally an SFC-CA / TF-Locoformer model.
Dedicated distillation recipes are now available. Recommended loss mix:

- normal waveform or spectral reconstruction loss,
- multi-resolution complex STFT loss,
- teacher mask or teacher spectrogram loss,
- optional latent/feature distillation on compressed bands if teacher/student
  feature shapes are made compatible.

The new architecture is designed to make distillation more useful than in the
older balanced CNB branch because it has trainable capacity before compression,
inside local/CNB stages, and after expansion.

## 2026-06-05 supervised stability follow-up

After the first supervised run of
`band-sfc-net-npu.adaptive-mel-loco-cnb.soft-query.rt192k.fp512keep475` peaked
around `validation/snr ~= 3.6 dB` and then dropped, the preset was reviewed for
capacity balance.  The main issue was structural: most parameters were in giant
frequency-pooled mixers and `residual_head=true` enabled an unbounded additive
residual branch during supervised stage-1.

Added stability-first follow-up recipes:

```text
recipes/dnr/models/band-sfc-net-npu.adaptive-mel-loco-cnb.stable-soft-query.rt192k.fp512keep475/config.yaml
recipes/dnr/models/band-sfc-net-npu.adaptive-mel-loco-cnb.stable-crossattn-query.rt192k.fp512keep475/config.yaml
recipes/dnr/models/band-sfc-net-npu.adaptive-mel-loco-cnb.band56-soft-query.rt192k.fp512keep475/config.yaml
```

Recommended next run:

```text
band-sfc-net-npu.adaptive-mel-loco-cnb.stable-soft-query.rt192k.fp512keep475
```

Detailed notes and commands are in:

```text
adaptive_mel_loco_cnb_stability_fix_20260605_operation.md
```
