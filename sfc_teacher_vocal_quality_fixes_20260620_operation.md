# SFC Teacher Song/Vocal Quality Investigation and Fixes

Date: 2026-06-20

## Scope

Teacher config originally investigated:

`recipes/dnr/models/locoformer-medium.enc-crossattn96dim.dec-crossattn96dim.musical64.learnable-query/config.yaml`

Main symptom:

- separated `speech` stem for songs/singing vocals is not smooth;
- vocal contains noisy/indistinct residual sounds;
- especially bad for songs.

User constraints:

- sample rate is consistent between training/inference: 24 kHz;
- dataset taxonomy is intentional:
  - song vocal is placed in `speech` pool;
  - instrumental/music elements are in `music` pool;
  - other sounds are in `effects` pool;
- investigation and fixes focus on the teacher itself, not student variants.

## Main diagnosis

The problem is likely not a single superficial pipeline bug. It is an interaction of teacher/data/loss/model factors:

1. `musical64` SFC band split is too generic/compressed for song-vocal fidelity at 24 kHz.
2. Full-window RMS normalization can boost sparse pseudo-separated vocal noise and vocal-gap artifacts.
3. The generic composite loss is not source/domain-aware enough for singing vocals and noisy pseudo-labels.
4. Independent dry mixing loses real song co-occurrence statistics.
5. `speech` is a broad class containing both spoken speech and singing vocals.
6. Pseudo-separated song vocal references may contain residual accompaniment/noise.
7. The medium teacher training schedule may be too short for robust song-vocal quality.

This operation implemented fixes for items 1, 2, and 3.

---

# 1. Band split issue and fix

## Problem

The original teacher uses:

```yaml
n_bands: 64
band_config: musical
```

At 24 kHz with `n_fft=2048`, `musical64` has undesirable properties for vocal quality:

- only 56 unique ranges for 64 bands;
- several duplicated low-frequency ranges;
- very wide high-frequency bands, up to about 2.35 kHz;
- insufficient resolution in important vocal/sibilance/confuser regions.

Song vocals need detail in:

- 80–300 Hz: F0/body;
- 300–1000 Hz: lower formants;
- 1–4 kHz: vocal intelligibility and presence;
- 4–8 kHz: consonants/sibilance/cymbal-confuser area.

## Fix: added `vocal64` / `speech_vocal64`

Implemented a speech/vocal-emphasis overlapped 64-band split.

Changed file:

`/home/cmj/works/ASS/spectral_feature_compression/core/model/bandit_split.py`

Added class:

```python
SpeechVocalBandsplitSpecification
```

Supported band config names:

```yaml
band_config: vocal
band_config: vocal64
band_config: speech_vocal
band_config: speech_vocal64
```

Updated encoder/decoder validation in:

`/home/cmj/works/ASS/spectral_feature_compression/core/model/enc_dec_base.py`

## Design of `vocal64`

For 24 kHz / 2048 FFT:

| Frequency region | Purpose | Bands |
|---|---|---:|
| 0–80 Hz | rumble / very low support | 3 |
| 80–300 Hz | speech/singing F0 and body | 7 |
| 300–1000 Hz | lower formants | 14 |
| 1–4 kHz | core vocal intelligibility/formants | 24 |
| 4–8 kHz | consonants/sibilance/cymbal confusers | 12 |
| 8–12 kHz | residual high band / air | 4 |
| total | | 64 |

Properties:

- 64 bands;
- 64 unique ranges;
- all FFT bins covered;
- more resolution below 8 kHz;
- less wasted capacity in duplicate low bands.

---

# 2. Source normalization issue and fix

## Problem

The on-the-fly mixer originally normalized each active source using full-window RMS:

```python
rms = stems[stem_idx].square().mean().sqrt()
stems[stem_idx] /= rms
```

This RMS is computed over the whole placed 6-second timeline.

For sparse song vocals, this is dangerous:

- vocal phrases occupy only part of the chunk;
- pseudo-separated vocal gaps can contain low-level music/noise;
- full-window RMS normalization boosts the entire stem, including gap noise;
- this can train the teacher to reproduce pseudo-label residue as `speech`.

Example failure mode:

```text
sparse vocal + low-level residual music in gaps
=> full-window RMS is low
=> normalization gain is high
=> residual music/noise becomes audible target
=> teacher speech output sounds noisy/indistinct
```

## Fix: source-specific robust normalization

Changed file:

`/home/cmj/works/ASS/spectral_feature_compression/common/datasets/on_the_fly_stem_dataset.py`

Added optional config:

```yaml
source_normalization:
```

Supported modes:

```yaml
full_rms
active_rms
percentile_rms
none
```

Supported fields:

```yaml
target_rms
mode
top_percent
activity_threshold_db
frame_ms
hop_ms
max_gain_db
min_gain_db
min_rms_db
```

Default behavior remains compatible: if only `normalize_sources: true` is used without `source_normalization`, behavior is effectively full-RMS style.

## Important protection: `min_rms_db`

Added `min_rms_db` because the key danger is not simply large gain, but boosting stems whose measured normalization RMS is too low.

Rule:

```text
if selected normalization RMS < min_rms_db and normalization gain > 1:
    skip boost for this stem
```

This prevents mostly-silent or noisy pseudo-label vocal clips from being normalized upward into strong supervised targets.

---

# 3. Explicit SNR control

## Problem

After normalization, source ratios should be controlled explicitly. Otherwise the teacher receives unrealistic speech/music/effects balances.

The existing dataset already had `stem_snr_db` support. We enabled it in the new teacher ablation config.

Sign convention from code:

```python
target_rms = anchor_rms * 10 ** (-snr_db / 20)
```

So positive SNR means the anchor is louder than the target stem.

For the current teacher, anchor is `speech`.

---

# 4. Loss stack issue and fix

## Problem

The existing `CompositeSupTask` is better than plain SNR, but it is generic. It does not explicitly address:

- song-domain weighting;
- vocal-active frame weighting;
- harmonic/formant continuity;
- perceptual vocal smoothness;
- explicit vocal-gap leakage;
- noisy pseudo-label references.

For song-vocal pseudo-labels, forcing the model to match every reference TF-bin exactly can make the teacher learn pseudo-label artifacts.

## Fix: added vocal-aware robust/truncated loss task

Created file:

`/home/cmj/works/ASS/spectral_feature_compression/core/tasks/vocal_aware_sup_task.py`

Added class:

```python
VocalAwareCompositeSupTask
```

It extends:

```python
CompositeSupTask
```

and adds teacher-only losses:

1. soft-truncated speech log-magnitude MR-STFT loss;
2. soft-truncated temporal log-magnitude gradient loss;
3. soft-truncated frequency log-magnitude gradient loss;
4. non-truncated inactive speech leakage loss.

## Why soft truncation

Implemented truncation form:

```python
tau * (1 - exp(-abs(error) / tau))
```

This is preferred over hard clipping because:

- behaves like L1 for small/medium errors;
- saturates for suspicious large errors;
- avoids hard gradient discontinuity;
- reduces overfitting to noisy pseudo-label artifacts.

Applied to speech/vocal log-magnitude losses, because pseudo-label artifacts are usually spectral:

- musical noise speckles;
- residual accompaniment harmonics;
- high-frequency roughness;
- phasey/watery artifacts.

## Why leakage loss is not truncated

Inactive speech leakage loss is intentionally non-truncated. If reference speech/vocal is inactive, we want strong pressure to keep the `speech` output clean.

This directly targets noisy/indistinct vocal gaps.

---

# 5. New teacher ablation config

Created/updated config:

`/home/cmj/works/ASS/recipes/dnr/models/locoformer-medium.enc-crossattn96dim.dec-crossattn96dim.vocal64.learnable-query/config.yaml`

This config inherits from the original medium teacher and applies the current best fixes:

1. `vocal64` speech/vocal-focused band split;
2. source-specific robust normalization;
3. max gain clamp;
4. min RMS skip-boost protection;
5. explicit speech-anchor SNR control;
6. vocal-aware robust/truncated losses;
7. stronger inactive-source penalty.

Important config contents:

```yaml
teacher_datamodule:
    synthesis:
        normalize_sources: true
        source_normalization:
            target_rms: 1.0
            mode:
                speech: percentile_rms
                music: full_rms
                effects: percentile_rms
            top_percent:
                speech: 50.0
                effects: 30.0
            frame_ms: 40.0
            hop_ms: 20.0
            max_gain_db:
                speech: 12.0
                music: 8.0
                effects: 10.0
            min_rms_db:
                speech: -45.0
                music: -55.0
                effects: -50.0
        stem_snr_db:
            enabled: true
            anchor: speech
            range:
                music: [-6, 6]
                effects: [-9, 6]
```

Loss/task additions:

```yaml
task:
    _target_: spectral_feature_compression.core.tasks.vocal_aware_sup_task.VocalAwareCompositeSupTask
    speech_source_index: 0
    speech_robust_logmag_weight: 0.05
    speech_robust_logmag_tau: 1.0
    speech_robust_logmag_resolutions:
        - [512, 128]
        - [1024, 256]
        - [2048, 512]
    vocal_active_frame_weight: 2.0
    speech_temporal_logmag_gradient_weight: 0.03
    speech_frequency_logmag_gradient_weight: 0.02
    speech_gradient_tau: 1.0
    speech_inactive_leakage_weight: 0.05
    speech_inactive_threshold_db: -45.0
    speech_inactive_softness_db: 6.0
    loss:
        zeroref_weight: 0.5
```

---

# 6. Tests added

## Band split tests

Created:

`/home/cmj/works/ASS/tests/test_speech_vocal_band_split.py`

Checks:

- `vocal64` layout for 24 kHz / 2048 FFT;
- 64 unique bands;
- all bins covered;
- alias `speech_vocal64` works;
- `CrossAttnEncoder` / `CrossAttnDecoder` accept new band config.

## Normalization tests

Created:

`/home/cmj/works/ASS/tests/test_on_the_fly_source_normalization.py`

Checks:

- percentile RMS normalization uses active/loud frames;
- max gain clamp works;
- `min_rms_db` skips boosting too-quiet stems;
- speech-anchor SNR control works.

## Vocal-aware loss tests

Created:

`/home/cmj/works/ASS/tests/test_vocal_aware_sup_task.py`

Checks:

- soft truncation saturates large errors;
- speech robust losses are near zero for identical signals;
- inactive speech leakage increases when estimated speech contains noise.

## Validation command

```bash
./.venv/bin/python -m pytest \
  tests/test_vocal_aware_sup_task.py \
  tests/test_on_the_fly_source_normalization.py \
  tests/test_speech_vocal_band_split.py -q
```

Result:

```text
9 passed
```

Warnings were only CUDA/autocast unavailable and pytest cache permission warnings.

---

# 7. Recommended next experiments

## A. Verify actual training config

Before training, confirm the actual merged config, especially:

- `n_chan` consistency;
- datamodule target;
- task target;
- source normalization fields;
- `band_config: vocal64`;
- loss weights.

Potential concern from initial config inspection:

- the medium config had `n_chan: 2`, while the on-the-fly dataset downmixes to mono. Verify actual run config and data path.

## B. Train current ablation

Use:

```text
recipes/dnr/models/locoformer-medium.enc-crossattn96dim.dec-crossattn96dim.vocal64.learnable-query/config.yaml
```

This is the main current teacher-fix ablation.

## C. Fixed song validation set

Create fixed song validation/listening examples with:

- mixture;
- speech/vocal reference;
- music reference;
- effects reference if available;
- teacher speech output.

Do not rely only on global validation loss.

## D. Reference-quality audit

Listen to the actual song `speech` references. If they already contain rough pseudo-label artifacts, teacher quality will be bounded without confidence weighting/cleaner labels.

## E. Overfit tiny song subset

Train on 10–50 paired song examples until very low train loss.

Interpretation:

- if it cannot overfit smoothly: structure/objective/reference quality is limiting;
- if it can: main issue is data distribution/training length/checkpoint selection.

## F. Longer teacher training

The medium config still likely needs more than 20 epochs for teacher quality. Consider longer training or fine-tune schedule after this ablation is verified.

---

# 8. Files changed in this operation

Implemented/modified:

- `spectral_feature_compression/core/model/bandit_split.py`
- `spectral_feature_compression/core/model/enc_dec_base.py`
- `spectral_feature_compression/common/datasets/on_the_fly_stem_dataset.py`
- `spectral_feature_compression/core/tasks/vocal_aware_sup_task.py`
- `recipes/dnr/models/locoformer-medium.enc-crossattn96dim.dec-crossattn96dim.vocal64.learnable-query/config.yaml`
- `tests/test_speech_vocal_band_split.py`
- `tests/test_on_the_fly_source_normalization.py`
- `tests/test_vocal_aware_sup_task.py`

Related analysis docs:

- `sfc_teacher_song_quality_root_cause_20260620_operation.md`
- `sfc_teacher_song_quality_deep_research_20260620.md`
- `sfc_teacher_vocal_quality_fixes_20260620_operation.md`  # this file
