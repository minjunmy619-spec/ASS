# Adaptive Mel Loco-CNB Stability Fix Operation

Date: 2026-06-05

## Trigger

The first supervised training run for:

```text
recipes/dnr/models/band-sfc-net-npu.adaptive-mel-loco-cnb.soft-query.rt192k.fp512keep475/config.yaml
```

peaked around `validation/snr ~= 3.6 dB` after about 5 epochs and then fell to
roughly `3.1 dB` without recovering over the next several epochs.

Inspection showed no fatal forward/export bug, but the preset had an important
structural imbalance:

- total params: `5,741,397`, fp16 state: `165.00 KiB`,
- `PooledChannelMixer` blocks dominated the trainable capacity,
- each Loco-CNB stage had `802,880 / 825,155` params in the pooled mixer
  (`97.3%` of the stage),
- actual local/CNB components were comparatively tiny,
- `residual_head=true` enabled an unbounded additive complex residual branch,
- inherited training LR reached full `1e-3` near epoch 4.

This means the run likely learned useful early masks, then the huge global
frequency-pooled current-frame mixers and residual branch started fitting train
priors more than robust time-frequency separation.

## Design fix

Added stability-first preset configs in `BandSFCNetNPU/presets.py`.

### Primary supervised stage-1 target

```text
adaptive_mel_loco_cnb_stable_soft_band_query
```

Changes from the original high-capacity shape:

| Setting | Original | Stable fix |
|---|---:|---:|
| channels | 32 | 36 |
| bands | 48 | 48 |
| pooled mixer schedule | `[8192]*5` | `[2048,4096,4096,4096,2048]` |
| encoder capacity hidden | 4096 | 2048 |
| decoder capacity hidden | 4096 | 2048 |
| residual head | true | false |
| params | 5,741,397 | 2,852,491 |
| fp16 state | 165.00 KiB | 185.62 KiB |

Rationale:

- Widen the real local/CNB recurrent representation from 32 to 36 channels.
- Reduce the giant frequency-pooled MLP dominance.
- Disable the residual branch during supervised stage-1.
- Keep fp16 state below the 192 KiB target.

### Transport ablation

```text
adaptive_mel_loco_cnb_stable_crossattn_query
```

Same stability shape, but uses cross-attention-query transport.

Measured:

```text
params = 2,866,770
fp16 state = 185.62 KiB
```

### Bottleneck/detail ablation

```text
adaptive_mel_loco_cnb_band56_soft_band_query
```

This tests whether 48 compressed bands are too coarse for DnR Effects and
high-frequency detail.

Measured:

```text
channels = 28
bands = 56
params = 2,210,395
fp16 state = 168.44 KiB
```

### Clean pointwise structural diagnostic

After further review, even the stable recipe still contains frequency-pooled
stage capacity.  Added a cleaner alternative:

```text
adaptive_mel_loco_cnb_clean_soft_band_query
```

This variant removes frequency-pooled IO capacity and replaces stage pooled
mixers with pointwise per-band channel mixers:

```text
stage_mixer_type = pointwise
pooled_mixer_hidden_schedule = [512, 1024, 1024, 1024, 512]
encoder_capacity_mixer_layers = 0
decoder_capacity_mixer_layers = 0
loco_ffn_expansion = 16
residual_head = false
```

Measured:

```text
channels = 36
bands = 48
params = 876,604
fp16 state = 185.62 KiB
```

Rationale:

- no mean-over-band global correction path,
- stage capacity is applied independently at each compressed band,
- wider local FFN keeps some per-band current-frame capacity,
- no additional streaming cache.

This is much smaller than the stable recipe, so it should be treated as a clean
structure diagnostic and/or distillation student, not necessarily the final
highest-quality supervised model.

## New recipes

Added supervised recipes:

```text
recipes/dnr/models/band-sfc-net-npu.adaptive-mel-loco-cnb.stable-soft-query.rt192k.fp512keep475/config.yaml
recipes/dnr/models/band-sfc-net-npu.adaptive-mel-loco-cnb.stable-crossattn-query.rt192k.fp512keep475/config.yaml
recipes/dnr/models/band-sfc-net-npu.adaptive-mel-loco-cnb.band56-soft-query.rt192k.fp512keep475/config.yaml
recipes/dnr/models/band-sfc-net-npu.adaptive-mel-loco-cnb.clean-soft-query.rt192k.fp512keep475/config.yaml
```

If the concern is only training instability of the original high-capacity run,
use the stable soft-query recipe.  If the concern is the pooled-mixer structure
itself, use the clean soft-query recipe first.

Training stability changes in the primary recipe:

```yaml
trainer:
  limit_val_batches: 30

datamodule:
  crop_retry: 24
  source_activity_threshold: 1.0e-4
  p_source_dropout: 0.08
  source_gain_db_min: -12.0
  source_gain_db_max: 12.0
  train_augmentations:
    p_gain: 0.45
    gain_db_min: -3.0
    gain_db_max: 3.0
    p_polarity: 0.2
    p_time_shift: 0.2
    max_time_shift_samples: 2048
    p_pitch_time: 0.1
    pitch_time_scale_min: 0.99
    pitch_time_scale_max: 1.01
    p_random_eq: 0.3
    eq_bands: 8
    eq_gain_db_min: -2.0
    eq_gain_db_max: 2.0
    p_band_dropout: 0.08
    band_dropout_width: 0.04

task:
  optimizer_config:
    optimizer_generator:
      lr: 3.e-4
      weight_decay: 1.e-3
    scheduler_generator:
      warmup_steps: 10000
      decay_start_step: 40000
      decay_stop_step: 120000
```

## Validation

### Smoke and recipe tests

```bash
cd /home/cmj/works/ASS
PYTHONPATH=. .venv/bin/python -m BandSFCNetNPU.test_band_sfc_net_npu
PYTHONPATH=. .venv/bin/python -m pytest tests/test_proposed_separation_models.py -q
```

Results:

```text
all BandSFCNetNPU smoke tests passed
18 passed
```

### ONE verification

Stateless verifier for primary stable soft-query:

```bash
cd /home/cmj/works/ASS
.venv/bin/python tools/online/verify_npu_variants.py \
  --mode recipe \
  --recipe-name-contains adaptive-mel-loco-cnb.stable-soft-query \
  --run-name band_sfc_adaptive_mel_loco_cnb_stable_soft_query_20260605 \
  --force-onnxsim-large-shape-ops \
  --quantize-layer-fallback
```

Result:

```text
PASS: model.circle, model.opt.circle, model.q.circle
```

Stateful streaming verifier for primary stable soft-query:

```bash
cd /home/cmj/works/ASS
.venv/bin/python tools/online/verify_npu_variants.py \
  --mode recipe \
  --recipe-name-contains adaptive-mel-loco-cnb.stable-soft-query \
  --run-name band_sfc_adaptive_mel_loco_cnb_stable_soft_query_streaming_20260605 \
  --force-onnxsim-large-shape-ops \
  --quantize-layer-fallback \
  --streaming
```

Result:

```text
PASS: model.circle, model.opt.circle, model.q.circle
```

Stateless verifier for stable cross-attention ablation:

```bash
cd /home/cmj/works/ASS
.venv/bin/python tools/online/verify_npu_variants.py \
  --mode recipe \
  --recipe-name-contains adaptive-mel-loco-cnb.stable-crossattn-query \
  --run-name band_sfc_adaptive_mel_loco_cnb_stable_crossattn_query_20260605 \
  --force-onnxsim-large-shape-ops \
  --quantize-layer-fallback
```

Result:

```text
PASS: model.circle, model.opt.circle, model.q.circle
```

Stateless verifier for 56-band ablation:

```bash
cd /home/cmj/works/ASS
.venv/bin/python tools/online/verify_npu_variants.py \
  --mode recipe \
  --recipe-name-contains adaptive-mel-loco-cnb.band56-soft-query \
  --run-name band_sfc_adaptive_mel_loco_cnb_band56_soft_query_20260605 \
  --force-onnxsim-large-shape-ops \
  --quantize-layer-fallback
```

Result:

```text
PASS: model.circle, model.opt.circle, model.q.circle
```

Stateless verifier for clean pointwise recipe:

```bash
cd /home/cmj/works/ASS
.venv/bin/python tools/online/verify_npu_variants.py \
  --mode recipe \
  --recipe-name-contains adaptive-mel-loco-cnb.clean-soft-query \
  --run-name band_sfc_adaptive_mel_loco_cnb_clean_soft_query_20260605 \
  --force-onnxsim-large-shape-ops \
  --quantize-layer-fallback
```

Result:

```text
PASS: model.circle, model.opt.circle, model.q.circle
```

Stateful streaming verifier for clean pointwise recipe:

```bash
cd /home/cmj/works/ASS
.venv/bin/python tools/online/verify_npu_variants.py \
  --mode recipe \
  --recipe-name-contains adaptive-mel-loco-cnb.clean-soft-query \
  --run-name band_sfc_adaptive_mel_loco_cnb_clean_soft_query_streaming_20260605 \
  --force-onnxsim-large-shape-ops \
  --quantize-layer-fallback \
  --streaming
```

Result:

```text
PASS: model.circle, model.opt.circle, model.q.circle
```

Artifact roots:

```text
logs/npu_verify_general/band_sfc_adaptive_mel_loco_cnb_stable_soft_query_20260605
logs/npu_verify_general/band_sfc_adaptive_mel_loco_cnb_stable_soft_query_streaming_20260605
logs/npu_verify_general/band_sfc_adaptive_mel_loco_cnb_stable_crossattn_query_20260605
logs/npu_verify_general/band_sfc_adaptive_mel_loco_cnb_band56_soft_query_20260605
logs/npu_verify_general/band_sfc_adaptive_mel_loco_cnb_clean_soft_query_20260605
logs/npu_verify_general/band_sfc_adaptive_mel_loco_cnb_clean_soft_query_streaming_20260605
```

## Recommendation

If you agree the pooled-mixer structure still looks bad, use this next:

```text
recipes/dnr/models/band-sfc-net-npu.adaptive-mel-loco-cnb.clean-soft-query.rt192k.fp512keep475/config.yaml
```

Use the stable recipe only if you want a higher-parameter supervised run while
still reducing the original over-capacity problem:

```text
recipes/dnr/models/band-sfc-net-npu.adaptive-mel-loco-cnb.stable-soft-query.rt192k.fp512keep475/config.yaml
```

If validation SNR still peaks early and falls:

1. compare `validation/loss` vs `validation/snr`,
2. evaluate more/full validation batches,
3. try the 56-band ablation,
4. then use distillation from a stronger offline teacher.
