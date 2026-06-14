# On-the-fly dry stem mixer datamodule operation

Date: 2026-06-13

## Goal

Add an opt-in data module for training DnR-style three-stem source separation
without prebuilding HDF5 files.  The first implementation intentionally ignores
RIR/spatialization because the available source WAVs may already be dry or wet.
It treats each input file as an already rendered source and mixes stems directly.

The dataloader output contract matches the existing separation tasks:

```python
wav.shape == [batch, 1, samples]
ref.shape == [batch, 3, 1, samples]  # [speech, music, effects]
wav == ref.sum(dim=1)
```

## Added files

- `spectral_feature_compression/common/datasets/on_the_fly_stem_dataset.py`
- `spectral_feature_compression/common/datamodules/on_the_fly_stem_datamodule.py`
- `recipes/dnr/datamodules/on-the-fly-stem-dry.yaml`
- `recipes/dnr/datamodules/on-the-fly-stem-dry-csv.yaml`
- `tests/test_on_the_fly_stem_datamodule.py`
- `tools/export_fixed_stem_mixes.py`

## Main controls

The dataset supports two source-list modes:

- folder scanning via `source_pools`
- CSV manifests via `source_manifest_csv` / `val_source_manifest_csv` /
  `test_source_manifest_csv`, using rows with at least
  `filename, split, type, filepath, sample_rate, channels`

The dataset supports config-controlled:

- fixed mixture duration
- active stem count distribution, including 0/1/2/3 active stems
- stem sampling weights
- number of clips per active stem
- same-stem concatenation/placement with configurable gaps and overlap
- independent stem gain ranges
- optional relative stem SNR scaling
- final peak normalization while preserving mixture consistency

## Example datamodule fragment

```yaml
datamodule:
  _target_: spectral_feature_compression.common.datamodules.on_the_fly_stem_datamodule.OnTheFlyStemDataModule
  sr: ${sr}
  duration: 6.0
  batch_size: 4
  val_batch_size: 1
  num_workers: 4
  dataset_length: 200000
  val_dataset_length: 2000
  source_order: [speech, music, effects]
  source_pools:
    speech:
      - /data1/speech1
      - /data1/speech2
    music:
      - /data1/music1
      - /data1/music2
    effects:
      - /data1/effects1
      - /data1/effects2
      - /data1/effects3
  synthesis:
    backend: dry_mix
    mixture_duration: ${datamodule.duration}
    active_stem_count:
      mode: weighted
      weights:
        0: 0.01
        1: 0.20
        2: 0.35
        3: 0.44
    clips_per_active_stem:
      speech: [1, 2]
      music: [1, 1]
      effects: [1, 4]
    short_clip_policy: concatenate
    same_stem_placement:
      mode: random_sequential
      initial_offset_sec_range: [0.0, 0.5]
      gap_sec_range: [0.0, 0.5]
      overlap_sec_range: [0.0, 0.25]
      allow_self_overlap: true
      max_self_overlap: 2
    stem_gain_db:
      speech: [-6, 6]
      music: [-8, 4]
      effects: [-10, 6]
    peak_norm_db: -1.0
    peak_norm_mode: scale_down
```

## Validation commands

```bash
.venv/bin/python -m pytest tests/test_on_the_fly_stem_datamodule.py -q
.venv/bin/python -m ruff check \
  spectral_feature_compression/common/datasets/on_the_fly_stem_dataset.py \
  spectral_feature_compression/common/datamodules/on_the_fly_stem_datamodule.py \
  tests/test_on_the_fly_stem_datamodule.py
```

## Review fixes on 2026-06-14

- Kept `return_metadata=True` available only for direct `OnTheFlyStemDataset`
  debugging.  `OnTheFlyStemDataModule` now fails fast if metadata batches are
  requested, because the existing training tasks expect exactly `(wav, ref)`.
- Implemented `short_clip_policy: pad` as a distinct single-clip crop/pad mode:
  one sampled clip is placed at a random valid offset inside the stem slot, with
  silence before/after the clip.  This avoids silently behaving like
  concatenation when `clips_per_active_stem > 1`.
- Added regression tests for metadata rejection and pad-policy random placement
  plus zero padding.

Validation:

```bash
.venv/bin/python -m pytest tests/test_on_the_fly_stem_datamodule.py -q
.venv/bin/python -m ruff check \
  spectral_feature_compression/common/datasets/on_the_fly_stem_dataset.py \
  spectral_feature_compression/common/datamodules/on_the_fly_stem_datamodule.py \
  tests/test_on_the_fly_stem_datamodule.py
```

## CSV manifest source lists

As an alternative to recursive folder scanning, pass one or more CSV manifests:

```yaml
datamodule:
  source_manifest_csv:
    - /data1/manifests/train_sources.csv
  val_source_manifest_csv:
    - /data1/manifests/validation_sources.csv
  test_source_manifest_csv:
    - /data1/manifests/test_sources.csv
```

Each CSV must contain at least:

```csv
filename,split,type,filepath,sample_rate,channels
speech_001.wav,train,speech,/data1/speech/speech_001.wav,44100,1
```

`type` must match the configured `source_order` names, e.g.
`[speech, music, effects]`.  `filepath` is preferred as an absolute path; if it
is relative, it is resolved relative to the CSV file location.  Rows with other
`type` values are ignored, and rows not present in the manifest are never used,
so bad/unwanted WAVs can be excluded by leaving them out of the CSV.

A single shared manifest can also be filtered by split:

```yaml
datamodule:
  source_manifest_csv:
    - /data1/manifests/all_sources.csv
  train_manifest_split: train
  val_manifest_split: validation
  test_manifest_split: test
```

Validation/test are fixed across epochs/runs when `val_seed` / `test_seed` are
set and their dataloaders stay unshuffled.  This fixes the generated mixture per
index without pre-rendering WAV files to disk.

## Fixed validation/test exporter

For fully fixed validation/test examples, export a wide fixed-mixture CSV and
rendered reference stem WAVs:

```bash
.venv/bin/python tools/export_fixed_stem_mixes.py \
  --source-manifest-csv /data1/manifests/all_sources.csv \
  --source-manifest-split validation \
  --output-csv /data1/manifests/fixed_validation.csv \
  --output-audio-dir /data1/fixed_eval_audio \
  --output-split validation \
  --num-examples 2000 \
  --sr 44100 \
  --duration 6.0 \
  --seed 2026 \
  --synthesis-json /data1/manifests/fixed_eval_synthesis.json \
  --export-mixtures
```

Reference stem WAVs are always exported because validation losses need fixed
speech/music/effects targets.  `--export-mixtures` additionally writes rendered
mixture WAVs for faster replay/listening checks.  Without `--export-mixtures`,
the datamodule reconstructs `wav` as `ref.sum(dim=1)` from the rendered refs.
Use `--audio-subtype FLOAT` (the default) to avoid PCM quantization differences.

A minimal `fixed_eval_synthesis.json` can mirror the datamodule synthesis block,
for example:

```json
{
  "backend": "dry_mix",
  "active_stem_count": {"mode": "weighted", "weights": {"1": 0.2, "2": 0.35, "3": 0.45}},
  "clips_per_active_stem": {"speech": [1, 2], "music": [1, 1], "effects": [1, 4]},
  "short_clip_policy": "concatenate",
  "stem_gain_db": {"speech": [-6, 6], "music": [-8, 4], "effects": [-10, 6]},
  "stem_snr_db": {"enabled": true, "anchor": "random_active", "range": {"speech": [-3, 9], "music": [-6, 6], "effects": [-9, 6]}},
  "peak_norm_db": -1.0,
  "peak_norm_mode": "scale_down"
}
```

The exported fixed CSV is directly usable by the datamodule:

```yaml
datamodule:
  source_manifest_csv:
    - /data1/manifests/train_sources.csv
  val_fixed_mix_manifest_csv:
    - /data1/manifests/fixed_validation.csv
  test_fixed_mix_manifest_csv:
    - /data1/manifests/fixed_test.csv
  val_synthesis:
    use_rendered_mixture: true  # default; ignored if mixture_filepath is empty
    fixed_mix_strict_shape: true  # default; fail if CSV sr/duration/n_samples mismatches this recipe
```

By default, fixed manifests are shape-strict: every row must contain
`sample_rate` and either `n_samples` or `duration` matching the datamodule `sr`
and `duration`.  This makes validation/test setup fail early instead of silently
resampling, padding, or truncating a fixed split generated with different audio
settings.  Set `fixed_mix_strict_shape: false` only for intentional migration or
debugging.

The fixed manifest has one row per mixture and contains:

```csv
mixture_id,split,sample_rate,duration,n_samples,mixture_filepath,speech_filepath,music_filepath,effects_filepath,active_stems,source_paths_json
```

## Follow-ups

- Add an opt-in `spaudsyn`/RIR backend only after source dryness/wetness and RIR
  target semantics are confirmed.
- Consider a small CLI sampler to render example mixtures for listening checks.
- Tune active-stem and gain distributions after inspecting training statistics.
