# SC Input RMS Normalization

This note documents the optional bounded RMS normalization added to the source classifier (SC) frontend for USS/TSE estimated-source adaptation.

## Problem

The stage-1 SC model can perform well on oracle foreground waveforms, but its accuracy may drop when the input is a separated waveform from USS or TSE.

The SC input distribution changes from:

```text
oracle dry foreground waveform -> SC
```

to:

```text
USS/TSE estimated waveform -> SC
```

Estimated waveforms can be attenuated, over-amplified, leaky, partially masked, or near-silent. Even if the class content is correct, the absolute waveform RMS can be far from the oracle-source distribution used to train SC.

This matters because M2D and PretrainedSED branches compute log-mel-like features from waveform amplitude. A 10–30 dB loudness shift can move frontend features away from the pretrained normalization statistics and reduce class confidence.

## Implementation

The normalization is implemented in:

```text
src/models/m2dat/m2d_sc.py
```

Main helper:

```python
_bounded_rms_normalize(
    waveform,
    target_rms_db=-28.0,
    min_rms_db=-55.0,
    max_gain_db=15.0,
    min_gain_db=-12.0,
    peak_limit=0.98,
)
```

It is applied inside `M2DSingleClassifierStrong._prepare_audio()` when `input_rms_norm` is enabled. Because `M2DPretrainedSEDFusionClassifier` inherits this path, the same normalization is used for the M2D branch and the frozen PretrainedSED branches.

## Config knobs

Example config block under `sc_model.args`:

```yaml
input_rms_norm: true
input_target_rms_db: -28.0
input_min_rms_db: -55.0
input_max_gain_db: 15.0
input_min_gain_db: -12.0
input_peak_limit: 0.98
```

### `input_rms_norm`

Enables or disables the normalization.

Default in the Python class is `false`, so existing configs are unchanged unless this is explicitly enabled.

### `input_target_rms_db`

Target RMS level in dBFS for rows that are loud enough to normalize.

Example:

```yaml
input_target_rms_db: -28.0
```

If an input row has RMS `-40 dB`, the normalizer tries to apply `+12 dB` gain so the output is around `-28 dB`.

### `input_min_rms_db`

Silence/leakage protection threshold.

Rows quieter than this are left unchanged.

Example:

```yaml
input_min_rms_db: -55.0
```

This prevents tiny residual noise from being boosted into loud foreground-like audio.

### `input_max_gain_db`

Maximum allowed boost.

Example:

```yaml
input_max_gain_db: 15.0
```

Even if a row would need `+25 dB` to reach the target RMS, it will only receive up to `+15 dB`.

### `input_min_gain_db`

Maximum allowed attenuation.

Example:

```yaml
input_min_gain_db: -12.0
```

This prevents very loud rows from being attenuated too aggressively.

### `input_peak_limit`

Peak protection after RMS gain.

Example:

```yaml
input_peak_limit: 0.98
```

The final gain is capped so the waveform peak does not exceed this value. Set to `null` or a non-positive value to disable peak limiting.

## Smoke-test interpretation

A small synthetic test used two constant waveform rows:

```text
row 0 amplitude = 0.01
row 1 amplitude = 0.00001
```

For constant waveforms, RMS is approximately the amplitude:

```text
20 * log10(0.01)    = -40 dB
20 * log10(0.00001) = -100 dB
```

The test normalization used:

```text
target_rms_db = -20
min_rms_db    = -55
max_gain_db   = +20
```

Output:

```text
input RMS rows: roughly -40 dB and -100 dB
output RMS rows: [-20.0, -100.0]
```

Meaning:

| Row | Input RMS | Rule | Output RMS |
| --- | ---: | --- | ---: |
| row 0 | `-40 dB` | above `min_rms_db`, boost toward target | `-20 dB` |
| row 1 | `-100 dB` | below `min_rms_db`, leave unchanged | `-100 dB` |

This verifies the intended safety behavior:

```text
real but quiet signals are boosted;
near-silence / residual leakage is not boosted.
```

In the current SC fine-tune config, the target is more conservative:

```yaml
input_target_rms_db: -28.0
input_min_rms_db: -55.0
input_max_gain_db: 15.0
```

So a real estimated source at `-40 dB` is boosted toward `-28 dB`, while a residual at `-100 dB` remains unchanged.

## Why this is safer than naive normalization

Naive RMS normalization would do:

```python
waveform = waveform / rms * target_rms
```

For a near-silent residual at `-100 dB`, targeting `-28 dB` would require `+72 dB` gain. That can turn tiny background leakage into loud noise and cause false foreground predictions.

The bounded normalizer avoids this with three protections:

1. `input_min_rms_db`: do not boost very quiet rows.
2. `input_max_gain_db` / `input_min_gain_db`: clamp gain.
3. `input_peak_limit`: avoid excessive peaks.

## Where it applies

Because normalization is part of the SC model frontend, it applies consistently whenever that SC instance is used:

- online SC fine-tuning on USS outputs,
- offline SC fine-tuning on cached estimates,
- standalone SC evaluation,
- S5 inference through `self.sc.predict(...)`.

This is preferable to normalizing only inside one training loop, because it avoids train/inference mismatch.

## Recommended diagnostics before/after enabling

Use the online export cache to measure RMS distributions:

```text
workspace/sc_online_uss_cache_pretrainedsed_fusion/<split>/oracle_target
workspace/sc_online_uss_cache_pretrainedsed_fusion/<split>/estimate_target
workspace/sc_online_uss_cache_pretrainedsed_fusion/<split>/estimate_all
```

Compare quantiles for:

- oracle foreground sources,
- matched USS estimates,
- uncertain/bad estimates,
- unmatched estimates.

A simple metric is per-file RMS dB:

```python
rms_db = 20 * log10(sqrt(mean(waveform ** 2)))
```

If matched USS estimates are much quieter or louder than oracle foregrounds, RMS normalization is likely useful.

## Tuning guide

### If estimated sources are still too quiet

Try:

```yaml
input_max_gain_db: 18.0
input_min_rms_db: -60.0
```

### If false positives / silence mistakes increase

Make normalization stricter:

```yaml
input_min_rms_db: -50.0
input_max_gain_db: 12.0
```

### If clipping or transient distortion appears

Lower the peak limit:

```yaml
input_peak_limit: 0.9
```

or disable RMS normalization and use gain augmentation instead.

## Important follow-up

Changing input RMS can change SC logit scale and the SC energy score:

```python
energy = -torch.logsumexp(plain_logits, dim=-1)
```

Therefore, after enabling normalization, re-check both raw and gated SC metrics:

```text
raw active-source accuracy
gated active-source accuracy
silence accuracy
source F1
final S5 CAPI-SDRi
```

The energy threshold may need recalibration after normalization.
