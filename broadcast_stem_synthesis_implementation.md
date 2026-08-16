# On-the-fly broadcast stem synthesis implementation guide

## 1. Purpose and scope

This document describes the complete data implementation used to train the
causal three-stem separator with speech, music, and effects targets. It covers:

- on-the-fly source selection and mixture construction;
- source manifests and directory-based source pools;
- per-source cropping, event placement, normalization, gain, and SNR;
- label-consistent broadcast and room rendering;
- wet/dry metadata handling;
- fixed rendered multitrack datasets and real-data interleaving;
- the Lightning data module and its train/validation/test behavior;
- the frame-local leakage loss;
- fixed-split export and domain-gap audit tools;
- the `broadcast-v1` recipe and its current limitations.

The implementation is designed around one non-negotiable supervised-learning
invariant:

```text
mixture == speech_target + music_target + effects_target
```

All label-consistent processing therefore operates on the target stems before
the final mixture is constructed. A nonlinear effect is never applied only to
the mixture while leaving the targets unchanged.

This document describes the current live code, not a proposed future design.

## 2. Implementation files

| File | Responsibility |
|---|---|
| [`on_the_fly_stem_dataset.py`](spectral_feature_compression/common/datasets/on_the_fly_stem_dataset.py) | Source loading, event synthesis, normalization, gain/SNR, fixed mixtures, and deterministic interleaving |
| [`broadcast_stem_synthesis.py`](spectral_feature_compression/common/datasets/broadcast_stem_synthesis.py) | Room, shared EQ, source compression, dialogue ducking, bus compression, and shared peak limiting |
| [`on_the_fly_stem_datamodule.py`](spectral_feature_compression/common/datamodules/on_the_fly_stem_datamodule.py) | Lightning dataset construction and dataloaders for train, validation, and test |
| [`distillation_task.py`](spectral_feature_compression/core/tasks/distillation_task.py) | Frame-local silent-source leakage penalty used by the robust distillation recipe |
| [`export_fixed_stem_mixes.py`](tools/export_fixed_stem_mixes.py) | Deterministic export of fixed validation/test stems and optional mixtures |
| [`audit_separation_manifest.py`](tools/audit_separation_manifest.py) | Additivity, level, activity, SI-SDR, and inactive-frame leakage audit |
| [`broadcast-v1/config.yaml`](recipes/dnr/models/tvconv-pyramid-sourceaware-sfclite-convgru-smoothup-smoothlogit-npu.speech-music-residual-sfx.robust-distill.rt192k.fp512keep475.broadcast-v1/config.yaml) | Opt-in broadcast synthesis recipe overlay |
| [`broadcast_stem_synthesis_20260816_operation.md`](broadcast_stem_synthesis_20260816_operation.md) | Chronological implementation and validation record |

## 3. End-to-end data flow

```text
stem directories or source manifest
              |
              v
     choose active stem classes
              |
              v
 choose one or more source clips per active stem
              |
              v
 random crop + mono conversion + resampling
              |
              v
 place events into a fixed-duration stem timeline
              |
              v
 source normalization -> independent gain -> relative SNR
              |
              v
 optional BroadcastStemRenderer
   room -> shared EQ -> source compression -> ducking
         -> shared bus compression -> shared limiter
              |
              v
 references: [number_of_stems, 1, number_of_samples]
              |
              v
 mixture = references.sum(dim=0): [1, number_of_samples]
              |
              v
 Lightning batch:
 mixture [batch, 1, samples]
 references [batch, stems, 1, samples]
```

For the current 44.1 kHz, six-second recipe:

```text
number_of_samples = 44100 * 6 = 264600
wav shape          = [1, 264600]
ref shape          = [3, 1, 264600]
```

The data module adds the batch dimension.

## 4. `OnTheFlyStemDataset`

### 4.1 Dataset contract

`OnTheFlyStemDataset` returns either:

```python
wav, ref
```

or, when instantiated directly with `return_metadata=True`:

```python
wav, ref, metadata
```

The data module intentionally disallows metadata batches because the current
training tasks expect exactly `(wav, ref)`.

The two available backends are:

| Backend | Behavior |
|---|---|
| `dry_mix` | Original linear on-the-fly construction; no broadcast renderer is instantiated |
| `broadcast_mix` | Runs the same construction and then applies `BroadcastStemRenderer` to the target stems |

The `broadcast` mapping has no effect when `backend: dry_mix`.

Constructor reference:

| Argument | Role |
|---|---|
| `source_pools` / `source_manifest_csv` | Mutually exclusive source inventory inputs |
| `manifest_split` | Optional one-or-many split filter for source-manifest rows |
| `manifest_filepath_column`, `manifest_type_column`, `manifest_split_column` | Override default CSV column names |
| `source_order` | Ordered output labels; defaults to speech, music, effects |
| `sr`, `duration` | Output sample rate and fixed mixture duration |
| `dataset_length` | Virtual number of synthesized examples per epoch |
| `backend` | `dry_mix` or `broadcast_mix` |
| `file_extensions` | Accepted pool/manifest suffixes; defaults to WAV and FLAC |
| `active_stem_count`, `stem_sampling_weights` | Number and identity of active source classes |
| `clips_per_active_stem`, `source_clip_duration_range` | Event count and source crop duration |
| `short_clip_policy`, `short_clip_pad_probability` | How short events occupy the fixed timeline |
| `same_stem_placement` | Gap, overlap, offset, and overlap-limit controls |
| `stem_gain_db`, `stem_snr_db`, `relative_snr_measurement` | Level and relative-SNR controls |
| `synthesis_profiles` | Weighted per-example scene-family overrides |
| `normalize_sources`, `source_normalization` | Optional pre-gain source normalization |
| `source_activity_threshold`, `crop_retry` | Quiet random-crop rejection controls |
| `broadcast` | Renderer configuration used by `broadcast_mix` |
| `peak_norm_db`, `peak_norm_mode` | Optional shared final dataset scaling |
| `seed` | Base seed for deterministic index-based synthesis |
| `return_metadata` | Direct-dataset debug metadata switch |

### 4.2 Source input modes

Exactly one of these inputs must be configured:

1. `source_pools`: a mapping from each stem name to one or more files or
   directories. Directories are scanned recursively for configured extensions.
2. `source_manifest_csv`: one CSV path or a list of CSV paths.

Directory-pool example:

```yaml
source_pools:
  speech: [/data/speech]
  music: [/data/music_a, /data/music_b]
  effects: [/data/foley, /data/ambience]
```

Manifest input is preferred when wet/dry metadata, split filtering, or
recording-level bookkeeping is needed.

### 4.3 Source manifest schema

The default required columns are:

| Column | Required | Meaning |
|---|---:|---|
| `filepath` | yes | Audio path; relative paths are resolved against the CSV directory |
| `type` | yes | Stem class matching `source_order`, normally `speech`, `music`, or `effects` |
| `split` | only when a split filter is used | Split value selected by `manifest_split` |

All other columns are retained as source metadata. The broadcast implementation
currently consumes `is_wet`; the remaining fields are recommended for corpus
auditing and split construction.

| Optional column | Meaning |
|---|---|
| `is_wet` | Whether room/reverb is already present in the source |
| `scene_id` | Scene grouping key |
| `recording_id` | Original recording grouping key; useful for leakage-safe splits |
| `speaker_id` | Speaker grouping key |
| `content_class` | Semantic content category such as Foley, ambience, or vocal music |

Example:

```csv
filepath,type,split,is_wet,recording_id,speaker_id,content_class
audio/dx_001.wav,speech,train,false,show_001,spk_017,dialogue
audio/mx_042.wav,music,train,true,album_012,,vocal_music
audio/fx_008.wav,effects,train,false,field_008,,ambience
```

Rows with stem types outside `source_order` are ignored. Missing files, empty
paths, missing required columns, or a stem with no remaining files fail during
dataset construction.

### 4.4 Deterministic sampling

When `seed` is an integer, item `i` uses:

```text
Random(seed + i)
```

The synthesized content of an index is therefore stable across workers and
repeated access. Dataloader shuffling changes index order, not the content of a
given index. When `seed` is `None`, each access creates a newly seeded random
generator and is intentionally stochastic.

Validation and test should use fixed integer seeds or fixed exported manifests.

### 4.5 Active-stem selection

`active_stem_count` supports:

```yaml
# Weighted number of active stems
active_stem_count:
  mode: weighted
  weights: {1: 0.2, 2: 0.35, 3: 0.45}
```

```yaml
# Uniform integer range
active_stem_count:
  mode: range
  range: [1, 3]
```

```yaml
# Fixed count
active_stem_count:
  mode: fixed
  value: 3
```

A sequence such as `[1, 2, 3]` is also accepted and sampled uniformly.
The result is clamped to `[0, len(source_order)]`, so intentional all-silent
examples are possible when a profile assigns weight to count zero.

`stem_sampling_weights` controls which classes are chosen when fewer than all
stems are active. Sampling is without replacement.

### 4.6 Synthesis profiles

`synthesis_profiles` supplies weighted scene families. One profile is selected
per example and may override only:

- `active_stem_count`;
- `stem_sampling_weights`;
- `clips_per_active_stem`;
- `stem_gain_db`;
- `stem_snr_db`.

It cannot override source loading, normalization, placement policy, or broadcast
processing. The current inherited recipe contains four profiles:

| Profile | Weight | Intended distribution |
|---|---:|---|
| `football_commentary_focus` | 0.30 | Speech/effects emphasis with quieter music |
| `live_concert_vocal_music` | 0.25 | Strong music with speech/vocal content and effects |
| `karaoke_music_control` | 0.20 | Music-dominant control including one-stem mixtures |
| `general_cass` | 0.25 | General coverage, including a small all-silent probability |

These profiles describe mixture composition; they are not labels consumed by
the separator. The selected profile name is available in debug metadata.

### 4.7 Source loading and cropping

For each selected file, the loader:

1. reads audio metadata with `soundfile`;
2. chooses a random crop start when the file is longer than the requested clip;
3. reads only the required number of source-rate frames when possible;
4. averages channels to mono;
5. resamples to the configured sample rate with `torchaudio`;
6. trims any resampling excess to the requested clip size.

`source_clip_duration_range` controls the maximum source read length without
changing the final mixture duration. It accepts one global range or a per-stem
mapping:

```yaml
source_clip_duration_range:
  speech: [0.8, 3.5]
  music: [3.0, 6.0]
  effects: [0.15, 1.5]
```

This distinction is important: short source clips create events and local
silence inside the fixed six-second target; they do not produce shorter model
inputs.

`source_activity_threshold` and `crop_retry` can reject especially quiet random
crops. When enabled, the loader tries multiple crops and returns the crop with
the highest RMS if none reaches the threshold.

### 4.8 Short-clip policies and event placement

`short_clip_policy` supports:

| Policy | Behavior |
|---|---|
| `pad` | Select one clip and place it at a random position in an otherwise silent timeline |
| `loop` | Repeat one clip until the complete mixture duration is filled |
| `concatenate` | Select the configured number of clips and place them using `same_stem_placement` |
| `random_place` | Place each selected clip at an independently sampled position |
| `pad_or_concatenate` | Per stem, probabilistically choose `pad`; otherwise use concatenate/sequential placement |

For `pad_or_concatenate`, `short_clip_pad_probability` can be scalar or
per-stem. The current recipe uses:

```yaml
short_clip_pad_probability:
  speech: 0.45
  music: 0.15
  effects: 0.70
```

The placement mapping supports:

```yaml
same_stem_placement:
  mode: random_sequential  # random, random_sequential, or sequential
  initial_offset_sec_range: [0.0, 0.5]
  gap_sec_range: [0.0, 0.45]
  overlap_sec_range: [0.0, 0.30]
  allow_self_overlap: true
  max_self_overlap: 2
  placement_retry: 16
```

An occupancy vector limits self-overlap. If sequential overlap would exceed the
limit, the loader tries the first unoccupied position. All event writes are
truncated at the fixed mixture boundary.

### 4.9 Source normalization

When `normalize_sources: true`, normalization occurs after event placement and
before random gain and relative SNR. Modes may be global or per stem:

| Mode | Measurement |
|---|---|
| `none` | No normalization for that stem |
| `full_rms` | RMS over the complete fixed-duration stem, including silence |
| `active_rms` | RMS of frames above `activity_threshold_db` |
| `percentile_rms` | RMS of the loudest `top_percent` frames |

Relevant fields are:

- `target_rms`;
- `frame_ms` and `hop_ms`;
- `activity_threshold_db`;
- `top_percent`;
- `max_gain_db` and `min_gain_db`;
- `min_rms_db`.

`min_rms_db` prevents near-silent material or pseudo-label residue from being
boosted to a normal programme level. A source rejected by this guard is also
excluded from relative-SNR amplification for that example.

If `active_rms` finds no frame over its threshold, source normalization falls
back to the loudest frame. `percentile_rms` selects the loudest
`ceil(number_of_frames * top_percent / 100)` frames.

The current recipe uses percentile RMS for speech/effects, full RMS for music,
per-stem target RMS values, and bounded positive gain.

### 4.10 Independent stem gain

After normalization, `stem_gain_db` samples one independent dB gain per active
stem. It may be configured globally or overridden by the selected synthesis
profile.

```yaml
stem_gain_db:
  speech: [-8.0, 6.0]
  music: [-10.0, 5.0]
  effects: [-12.0, 8.0]
```

### 4.11 Relative SNR

`stem_snr_db` chooses an anchor stem and scales every other eligible stem
relative to its measured RMS:

```yaml
stem_snr_db:
  enabled: true
  anchor: speech       # or random_active
  anchor_min_rms_db: -50.0  # optional
  range:
    music: [6.0, 18.0]
    effects: [-6.0, 9.0]
```

Positive SNR means the anchor is louder than the adjusted stem. For sampled
SNR `s`, the target non-anchor RMS is:

```text
target_rms = anchor_rms * 10^(-s / 20)
```

`relative_snr_measurement.mode` selects `full_rms` or `active_rms`. Active RMS
is preferable when short clips occupy only part of the six-second window,
because full-window RMS otherwise confounds event duration with source level.

### 4.12 Final construction order

The exact per-item order is:

1. Create the index-specific RNG.
2. Select one synthesis profile, if configured.
3. Select active-stem count and stem classes.
4. Build a fixed-duration waveform for each active stem.
5. Normalize active sources.
6. Apply independent stem gain.
7. Apply relative-SNR scaling.
8. Run `BroadcastStemRenderer` for `broadcast_mix`.
9. Apply dataset peak normalization, if enabled.
10. Convert stems to `ref` with shape `[stems, 1, samples]`.
11. Compute `wav = ref.sum(dim=0)` exactly.

The broadcast recipe sets dataset `peak_norm_db: null` because the broadcast
renderer already owns the final shared limiter.

When dataset peak normalization is enabled, `scale_down` attenuates only when
the mixture exceeds the target peak. `normalize` scales both up and down to the
target. In either mode the same scalar is applied to all stems.

### 4.13 Debug metadata

Direct dataset use with `return_metadata=True` returns:

- `index`;
- `active_stems`;
- `source_paths` for each stem;
- `synthesis_profile`, when profiles are configured;
- `broadcast`, containing applied-stage flags and room-applied stems.

Example:

```python
from spectral_feature_compression.common.datasets.on_the_fly_stem_dataset import OnTheFlyStemDataset

dataset = OnTheFlyStemDataset(
    source_manifest_csv="/data/manifests/train_sources.csv",
    source_order=("speech", "music", "effects"),
    sr=44100,
    duration=6.0,
    dataset_length=10,
    backend="broadcast_mix",
    broadcast={"bus_peak_limit_db": -1.0},
    seed=2026,
    return_metadata=True,
)
wav, ref, metadata = dataset[0]
assert (wav == ref.sum(dim=0)).all()
```

## 5. `BroadcastStemRenderer`

### 5.1 Design rule

The renderer accepts stems shaped `[stems, channels, samples]`, clones them,
and returns transformed stems with the same shape. The mixture is not an input.
The dataset sums the transformed stems afterward.

This preserves supervised additivity even for time-varying gain processes.

### 5.2 Processing order

The renderer applies stages in this order:

1. shared room/RIR rendering;
2. shared channel EQ;
3. per-source compression;
4. speech-driven ducking of configured targets;
5. shared bus-compressor gain;
6. shared peak limiting.

Each probabilistic stage performs one Bernoulli decision per synthesized
example. Parameter ranges are sampled once per applicable stage or source.

Renderer configuration reference:

| Section | Main fields |
|---|---|
| `room` | `probability`, `rir_paths`, `shared_stems`, `wet_mix`, `max_rir_seconds`, `preserve_rms`, `unknown_wet_policy` |
| `channel_eq` | `probability`, `low_cut_hz`, `high_cut_hz`, `transition_hz` |
| `source_compression` | `probability`, `stems`, `threshold_db`, `ratio`, `frame_ms`, `attack_ms`, `release_ms` |
| `ducking` | `probability`, `speech_stem`, `target_stems`, `attenuation_db`, `activity_threshold_db`, timing fields |
| `bus_compression` | `probability`, compressor threshold/ratio/timing fields |
| top level | `bus_peak_limit_db` |

### 5.3 Wet/dry handling

Wet/dry handling is metadata-based. There is no acoustic classifier that
listens to a waveform and estimates reverberation.

The parser recognizes these values:

| Meaning | Accepted values |
|---|---|
| wet | `true`, `1`, `yes`, `y`, `wet` |
| dry | `false`, `0`, `no`, `n`, `dry` |
| unknown | missing or any unrecognized value |

The current room configuration uses:

```yaml
shared_stems: [speech, effects]
unknown_wet_policy: assume_wet
```

For a selected example, the room stage:

1. identifies non-silent active stems listed in `shared_stems`;
2. obtains every contributing source path for those stems;
3. checks every path's `is_wet` metadata;
4. applies one shared RIR only if every candidate source is considered dry;
5. skips room rendering for the whole candidate group if any source is wet,
   unknown under `assume_wet`, or lacks a source path.

This all-or-nothing decision preserves a coherent shared room. It does not put
speech and effects into different randomly selected rooms.

When data comes from `source_pools`, there is no manifest metadata. With the
safe `assume_wet` policy, room augmentation is therefore skipped. Set
`assume_dry` only when the entire source pool is known to be dry.

The current `broadcast-v1` recipe has `broadcast_rir_probability: 0.0`; wet/dry
metadata is wired through and checked by the implementation, but RIR rendering
is disabled until real RIR paths and reliable metadata are supplied.

### 5.4 Room rendering

When enabled:

- one RIR path is sampled for the example;
- the RIR is truncated to `max_rir_seconds`;
- it is converted to mono and resampled;
- zero-energy or non-finite RIRs are rejected;
- it is energy-normalized;
- FFT convolution is used and the result is truncated to the original length;
- one `wet_mix` value is shared by all candidate stems.

Each candidate becomes:

```text
rendered = dry * (1 - wet_mix) + convolve(dry, RIR) * wet_mix
```

The recipe sets `preserve_rms: false`. Independent post-room RMS restoration
would rescale stems differently and weaken their shared-room level
relationship.

### 5.5 Shared channel EQ

The EQ samples low- and high-cut frequencies, constructs smooth frequency-domain
ramps using `transition_hz`, and applies the same response to every stem. Since
the EQ is shared and linear, additivity is preserved.

This stage simulates programme/channel bandwidth variation rather than a unique
microphone response for each source.

### 5.6 Source compression

For every configured non-silent stem, the renderer:

1. computes frame power;
2. samples threshold and ratio;
3. computes downward-compressor gain above the threshold;
4. smooths the gain with attack and release coefficients;
5. applies the expanded gain envelope to that stem.

The compressor changes the supervised target itself. It is not applied only to
the mixture.

### 5.7 Dialogue ducking

The ducking detector measures frame power in `speech_stem`. Frames above
`activity_threshold_db` request a sampled negative attenuation. The gain is
attack/release smoothed and applied to every configured target stem, currently
music and effects.

Speech is not altered by this stage. If the sampled speech stem is silent, no
ducking is applied.

### 5.8 Bus compression

The renderer first computes the current mixture from the stems and derives one
compressor gain envelope from it. That same envelope is multiplied into every
stem:

```text
gain = compressor(sum(stems))
rendered_stem[i] = stem[i] * gain
```

This models shared programme dynamics while preserving:

```text
sum(rendered_stems) == sum(stems) * gain
```

### 5.9 Shared peak limiting

The final mixture peak is compared with `bus_peak_limit_db`. If it exceeds the
limit, one scalar attenuation is applied to every stem. This is a shared peak
scale, not an independent per-stem normalization.

### 5.10 Validation and failure behavior

Renderer construction rejects:

- unknown top-level broadcast fields;
- probabilities outside `[0, 1]`;
- invalid or non-finite ranges;
- non-positive frame, attack, or release times;
- compressor ratios below 1;
- positive ducking attenuation;
- unknown configured stem names;
- invalid EQ cutoffs or transition width;
- room wet mix outside `[0, 1]`;
- non-positive RIR duration;
- unsupported wet-policy values;
- missing RIR files when room probability is positive;
- positive or non-finite peak limits.

RIR loading additionally rejects a zero-energy impulse response.

## 6. Fixed mixtures and real multitrack data

### 6.1 `FixedStemMixDataset`

This dataset replays a fixed wide manifest. It is intended for:

- deterministic validation and test sets;
- real multitrack DX/MX/FX training examples;
- fixed OOD synthetic evaluation.

The manifest requires one `{stem}_filepath` column per configured stem. Empty
cells represent an inactive zero stem.

Recommended schema:

```csv
mixture_id,split,sample_rate,duration,n_samples,mixture_filepath,speech_filepath,music_filepath,effects_filepath
show_001,train,44100,6.0,264600,/data/mix.wav,/data/dx.wav,/data/mx.wav,/data/fx.wav
```

Behavior:

- relative paths are resolved against the manifest directory;
- inputs are decoded as float, converted to mono, and resampled if needed;
- short audio is zero-padded and long audio is trimmed;
- `strict_shape: true` requires manifest sample-rate and length metadata to
  agree with the configured shape;
- `use_rendered_mixture: true` loads `mixture_filepath` when it is present;
- otherwise, the mixture is reconstructed from the targets.

### 6.2 Additivity protection

For residual-Effects supervision, a delivered mixture must match its delivered
stems closely. With:

```yaml
fixed_mix_max_additivity_error_db: -40.0
```

the loader computes:

```text
10 * log10(mean((mixture - sum(stems))^2) / mean(mixture^2))
```

and rejects an example above the configured threshold. This is enforced during
sample loading in addition to the offline audit.

If the mixture is reconstructed from stems, it is additive by construction.
The runtime threshold is evaluated only when a rendered mixture path is used.

Lossy final-bus codec audio generally should not be forced into this contract:
independently delivered or encoded stems may no longer sum to the decoded
mixture. Such audio belongs in evaluation or teacher/pseudo-label adaptation
unless its delivered stems are verified additive.

### 6.3 `ProbabilisticInterleaveDataset`

The data module can blend fixed real multitracks into an on-the-fly primary
training set. Selection is deterministic per primary index:

```text
Random(seed + index).random() < probability
```

If selected, supplemental row `index % len(supplemental_dataset)` is used.
Dataset length remains the primary dataset length.

This is interleaving, not concatenation. A probability of `0.15` means an
expected 15% of primary indices resolve to supplemental samples.

## 7. `OnTheFlyStemDataModule`

### 7.1 Purpose

`OnTheFlyStemDataModule` is the Lightning integration layer. It:

- validates mutually exclusive source modes;
- constructs train, validation, and optional test datasets;
- forwards synthesis controls to the appropriate dataset;
- optionally interleaves fixed real training mixtures;
- constructs standard PyTorch dataloaders;
- preserves the `(wav, ref)` task contract.

### 7.2 Primary source selection

Training must provide exactly one of:

- `source_pools`;
- `source_manifest_csv`;
- `fixed_mix_manifest_csv`.

Validation and test each accept their corresponding `val_*` and `test_*`
forms. More than one source mode for a stage is rejected.

If no validation source is provided, validation inherits the training source
configuration. For a trustworthy quality estimate, configure an explicitly
held-out validation manifest or a fixed validation mixture manifest.

Test data is built only when a test source is explicitly present, or when a
test split is configured and can reuse the corresponding primary manifest.
Calling `setup("test")` without test data raises an error.

### 7.3 Synthesis configuration forwarding

The data module copies `synthesis` and injects defaults for:

- `duration`;
- `sr`;
- `source_order`;
- `return_metadata=False`.

`mixture_duration` is an alias that overrides the dataset duration.

These fixed-dataset controls live under `synthesis` but are removed before
constructing `OnTheFlyStemDataset`:

- `use_rendered_mixture`;
- `fixed_mix_strict_shape`;
- `fixed_mix_max_additivity_error_db`.

`val_synthesis` and `test_synthesis` are top-level shallow overrides:

```python
val_synthesis = {**synthesis, **val_synthesis}
test_synthesis = {**synthesis, **val_synthesis, **test_synthesis}
```

Consequently, overriding a nested key such as `broadcast` replaces the entire
`broadcast` mapping for that stage. Repeat all required nested broadcast fields
when using such an override.

### 7.4 Dataset construction by stage

| Stage input | Constructed dataset |
|---|---|
| Source pools or source manifest | `OnTheFlyStemDataset` |
| Fixed mixture manifest | `FixedStemMixDataset` |
| On-the-fly training plus positive supplemental probability | `ProbabilisticInterleaveDataset(primary, fixed)` |

Supplemental fixed mixtures are training-only. They cannot be added when the
primary training dataset is already a fixed mixture dataset.

### 7.5 Dataloader behavior

| Loader | Shuffle | Default drop-last | Seed source |
|---|---:|---:|---|
| Train | yes | true | `train_seed`, default `None` |
| Validation | no | false | `val_seed`, default `0` |
| Test | no | false | `test_seed`, default `0` |

`persistent_workers` defaults to true only when `num_workers > 0`. Batch size,
pinning, and drop-last behavior are independently configurable by stage.

The class reports dataset sizes during `setup()` and exposes unbatched example
shapes through `example_batch_shapes`.

### 7.6 Complete data-module example

```yaml
datamodule:
  _target_: spectral_feature_compression.common.datamodules.on_the_fly_stem_datamodule.OnTheFlyStemDataModule

  source_manifest_csv: [/data/manifests/train_sources.csv]
  val_fixed_mix_manifest_csv: /data/manifests/validation_fixed.csv
  test_fixed_mix_manifest_csv: /data/manifests/test_fixed.csv

  source_order: [speech, music, effects]
  sr: 44100
  duration: 6.0
  dataset_length: 240000
  val_dataset_length: 2000
  test_dataset_length: 2000

  batch_size: 4
  val_batch_size: 1
  test_batch_size: 1
  num_workers: 4
  train_seed: null
  val_seed: 2026
  test_seed: 2027

  supplemental_fixed_mix_manifest_csv: null
  supplemental_fixed_mix_probability: 0.0

  synthesis:
    backend: broadcast_mix
    mixture_duration: 6.0
    fixed_mix_strict_shape: true
    fixed_mix_max_additivity_error_db: -40.0

    source_clip_duration_range:
      speech: [0.8, 3.5]
      music: [3.0, 6.0]
      effects: [0.15, 1.5]

    short_clip_policy: pad_or_concatenate
    short_clip_pad_probability:
      speech: 0.45
      music: 0.15
      effects: 0.70

    same_stem_placement:
      mode: random_sequential
      initial_offset_sec_range: [0.0, 0.5]
      gap_sec_range: [0.0, 0.45]
      overlap_sec_range: [0.0, 0.30]
      allow_self_overlap: true
      max_self_overlap: 2

    normalize_sources: true
    source_normalization:
      mode:
        speech: percentile_rms
        music: full_rms
        effects: percentile_rms
      target_rms: {speech: 0.12, music: 0.10, effects: 0.10}
      top_percent: {speech: 35.0, effects: 45.0}
      activity_threshold_db: -48.0
      min_rms_db: {speech: -48.0, music: -54.0, effects: -54.0}
      max_gain_db: {speech: 12.0, music: 10.0, effects: 12.0}

    relative_snr_measurement:
      mode: active_rms
      frame_ms: 40.0
      hop_ms: 20.0
      activity_threshold_db: -48.0

    peak_norm_db: null
    broadcast:
      room:
        probability: 0.0
        rir_paths: []
        shared_stems: [speech, effects]
        wet_mix: [0.12, 0.45]
        max_rir_seconds: 1.0
        preserve_rms: false
        unknown_wet_policy: assume_wet
      channel_eq:
        probability: 0.35
        low_cut_hz: [20.0, 180.0]
        high_cut_hz: [6500.0, 20000.0]
        transition_hz: 250.0
      source_compression:
        probability: 0.30
        stems: [speech, music, effects]
        threshold_db: [-28.0, -14.0]
        ratio: [1.5, 3.0]
        frame_ms: 20.0
        attack_ms: 10.0
        release_ms: 120.0
      ducking:
        probability: 0.45
        speech_stem: speech
        target_stems: [music, effects]
        attenuation_db: [-10.0, -3.0]
        activity_threshold_db: -48.0
        frame_ms: 20.0
        attack_ms: 30.0
        release_ms: 300.0
      bus_compression:
        probability: 0.50
        threshold_db: [-24.0, -12.0]
        ratio: [1.5, 3.5]
        frame_ms: 20.0
        attack_ms: 10.0
        release_ms: 150.0
      bus_peak_limit_db: -1.0
```

The actual recipe inherits its synthesis profiles and base training settings
from the parent recipe, so inspect a strict expanded configuration before
launching training.

## 8. Frame-local leakage loss

The existing `silent_source_weight` applies only when an entire target stem is
silent for the complete crop. It cannot penalize leakage during a silent region
inside a source that is active elsewhere.

`TeacherStudentDistillationTask` adds:

```yaml
frame_silent_source_weight: 0.08
frame_silent_source_db: -80.0
frame_silent_window_ms: 80.0
frame_silent_hop_ms: 40.0
```

The implementation:

1. computes frame power for each target and estimate;
2. marks target frames at or below the configured power threshold inactive;
3. averages estimated power over inactive target frames;
4. applies configured source-loss weights when present;
5. adds the weighted penalty to the training loss;
6. logs it as `loss_frame_silent_source`.

The recipe uses `-80 dBFS` because synthetic inactive placements are exact
zeros. A higher threshold such as `-50 dBFS` would misclassify quiet but valid
ambience or dialogue tails as silence and train the model to erase them.

The loss requires `fs` when its weight is positive.

## 9. Fixed split export

`tools/export_fixed_stem_mixes.py` materializes deterministic references from
`OnTheFlyStemDataset`. Reference WAVs are always written. Mixtures are optional;
when omitted, `FixedStemMixDataset` reconstructs them exactly from references.

Example:

```bash
.venv/bin/python tools/export_fixed_stem_mixes.py \
  --source-manifest-csv /data/manifests/sources.csv \
  --source-manifest-split validation \
  --output-csv /data/manifests/validation_fixed.csv \
  --output-audio-dir /data/fixed_validation \
  --output-split validation \
  --num-examples 2000 \
  --sr 44100 \
  --duration 6.0 \
  --seed 2026 \
  --synthesis-json /data/configs/validation_synthesis.json \
  --export-mixtures \
  --audio-subtype FLOAT
```

The exported CSV contains:

- mixture ID and split;
- sample rate, duration, and sample count;
- optional mixture path;
- every rendered stem path;
- active stem names;
- JSON-encoded original source paths.

Use `FLOAT` WAV for the cleanest numerical additivity.

## 10. Manifest audit tool

`tools/audit_separation_manifest.py` supports three tiers:

| Tier | Expected data |
|---|---|
| `ood_synthetic` | Fixed labeled synthetic mixtures outside training distributions |
| `real_multitrack` | Real labeled mixture and stems |
| `real_unlabeled` | Real mixture only |

All tiers require `mixture_filepath`. Labeled tiers require every configured
`{stem}_filepath` column, although an individual cell may be empty to represent
silence. Optional `pred_{stem}_filepath` columns enable prediction metrics.

The audit reports:

- mixture RMS, peak, crest factor, and active fraction;
- the same statistics per reference stem;
- mixture-to-sum-of-stems additivity error;
- prediction SI-SDR when references and predictions are present;
- prediction energy during reference-inactive frames.

Audio is converted to mono and resampled with band-limited `torchaudio`
resampling. Length mismatches after conversion are rejected.

Example:

```bash
.venv/bin/python tools/audit_separation_manifest.py \
  /data/manifests/real_multitrack_test.csv \
  --tier real_multitrack \
  --source-order speech,music,effects \
  --sample-rate 44100 \
  --frame-ms 80 \
  --silence-db -80 \
  --output-json logs/domain_gap/real_multitrack.json
```

Run this audit before enabling supplemental real-multitrack training.

## 11. Current `broadcast-v1` recipe status

The opt-in recipe is:

```text
recipes/dnr/models/
  tvconv-pyramid-sourceaware-sfclite-convgru-smoothup-smoothlogit-npu.
  speech-music-residual-sfx.robust-distill.rt192k.fp512keep475.broadcast-v1/
  config.yaml
```

It inherits the robust-distillation base recipe and changes only the data-domain
gap controls and frame-local leakage loss.

Current important settings:

| Setting | Current value | Meaning |
|---|---:|---|
| `backend` | `broadcast_mix` | Enables the renderer |
| RIR probability | `0.0` | Room rendering remains disabled pending actual metadata/assets |
| Shared room stems | speech, effects | Music is currently not convolved by the room stage |
| Unknown wet policy | `assume_wet` | Missing metadata safely prevents RIR application |
| Channel EQ probability | `0.35` | Shared bandwidth variation |
| Source compression probability | `0.30` | Per-source dynamics |
| Ducking probability | `0.45` | Speech-controlled music/effects attenuation |
| Bus compression probability | `0.50` | Shared programme dynamics |
| Bus peak limit | `-1 dBFS` | Shared final attenuation |
| Supplemental fixed probability | `0.0` | Real multitrack blending disabled until a manifest is supplied |
| Fixed additivity maximum | `-40 dB` | Rejects incompatible delivered mixtures |
| Frame-local leakage weight | `0.08` | Penalizes output in locally inactive target frames |
| Frame silence threshold | `-80 dBFS` | Protects valid quiet programme material |

### Enabling measured RIR safely

```yaml
broadcast_rir_probability: 0.30
broadcast_rir_paths:
  - /data/rir/living_room_001.wav
  - /data/rir/studio_004.wav
```

Before enabling:

1. verify every path exists and contains a nonzero measured RIR;
2. add trustworthy `is_wet` metadata to all potentially convolved sources;
3. keep `unknown_wet_policy: assume_wet` unless every unannotated source pool is
   known to be dry;
4. audit the resulting wet/dry and activity distributions;
5. compare real-TV leakage, not only synthetic SI-SDR.

### Enabling supplemental real multitracks

```yaml
datamodule:
  supplemental_fixed_mix_manifest_csv: /data/manifests/real_multitrack_train.csv
  supplemental_fixed_mix_probability: 0.15
  synthesis:
    fixed_mix_max_additivity_error_db: -40.0
```

The nested `synthesis` shown here is illustrative. In a recipe overlay, retain
the other inherited synthesis keys according to the repository's config-merge
semantics.

## 12. Validation and tests

Relevant tests are located in:

- [`test_on_the_fly_stem_datamodule.py`](tests/test_on_the_fly_stem_datamodule.py);
- [`test_on_the_fly_source_normalization.py`](tests/test_on_the_fly_source_normalization.py);
- [`test_broadcast_stem_synthesis.py`](tests/test_broadcast_stem_synthesis.py);
- [`test_audit_separation_manifest.py`](tests/test_audit_separation_manifest.py);
- [`test_proposed_separation_models.py`](tests/test_proposed_separation_models.py).

The tests cover:

- loader-compatible manifests;
- output shapes and exact additivity;
- fixed mixture loading and shape checks;
- rejection of nonadditive rendered mixtures;
- deterministic supplemental interleaving;
- source normalization and low-energy guards;
- dialogue ducking and shared bus gain;
- shared RIR application and wet-metadata gating;
- zero-energy RIR rejection;
- invalid dynamics timing rejection;
- audit schema, resampling, additivity, and inactive leakage;
- frame-local leakage loss and quiet valid target frames.

Validation commands:

```bash
env NUMBA_CACHE_DIR=/tmp/ass_numba_cache \
    PYTEST_ADDOPTS='-p no:cacheprovider' \
    .venv/bin/python -m pytest \
    tests/test_on_the_fly_stem_datamodule.py \
    tests/test_on_the_fly_source_normalization.py \
    tests/test_broadcast_stem_synthesis.py \
    tests/test_audit_separation_manifest.py -q

env NUMBA_CACHE_DIR=/tmp/ass_numba_cache \
    PYTEST_ADDOPTS='-p no:cacheprovider' \
    .venv/bin/python -m pytest tests/test_proposed_separation_models.py -q

.venv/bin/python tools/expand_recipe_config.py \
  recipes/dnr/models/tvconv-pyramid-sourceaware-sfclite-convgru-smoothup-smoothlogit-npu.speech-music-residual-sfx.robust-distill.rt192k.fp512keep475.broadcast-v1/config.yaml \
  --strict \
  --output /tmp/ass_broadcast_v1_resolved.yaml
```

The latest implementation review passed 95 relevant tests and instantiated a
resolved-recipe loader sample with exact additivity.

## 13. What is not implemented

The current pipeline does not provide:

- automatic acoustic wet/dry classification;
- loudness-unit or true-peak broadcast metering;
- nonlinear codec augmentation inside additive supervised synthesis;
- proof that the selected probabilities match the user's actual synthesized
  training corpus or real TV programme distribution;
- automatic speaker/recording/content-disjoint split generation;
- automatic RIR selection based on scene metadata;
- a trained-checkpoint quality improvement claim.

These are data- or experiment-dependent tasks rather than missing wiring in the
current implementation.

## 14. Recommended operating procedure

1. Build source manifests with `filepath`, `type`, split keys, leakage-group
   metadata, and reliable `is_wet` annotations.
2. Instantiate `OnTheFlyStemDataset` directly with `return_metadata=True` and
   listen to examples from every synthesis profile.
3. Export fixed OOD validation/test mixtures with deterministic seeds.
4. Audit fixed synthetic and real multitrack manifests.
5. Keep RIR and supplemental real data disabled until their metadata and
   additivity checks pass.
6. Expand the final recipe strictly and verify the resolved data settings.
7. Train controlled ablations: dry baseline, activity/event changes, broadcast
   renderer, frame-local leakage loss, then real multitrack blending.
8. Select the final configuration using held-out real-TV leakage and listening
   tests in addition to synthetic separation metrics.
