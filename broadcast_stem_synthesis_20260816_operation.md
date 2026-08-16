# Broadcast-realistic stem synthesis v1 operation record

## Goal

Reduce the synthetic-to-real domain gap for the causal speech/music/effects
separator without changing the existing `dry_mix` baseline or breaking the
residual-Effects contract:

```text
mixture == speech_target + music_target + effects_target
```

The implementation is intentionally opt-in through the `broadcast_mix`
backend and a new recipe. No quality claim is made until it is evaluated on
the actual training corpus and real TV audio.

The complete component and configuration reference is maintained in
[`broadcast_stem_synthesis_implementation.md`](broadcast_stem_synthesis_implementation.md).
This operation record remains the chronological record of changes, commands,
and data-dependent completion gates.

## Implemented changes

### Activity and event construction

- `source_clip_duration_range` accepts either one global range or one range per
  stem. The broadcast recipe uses shorter speech/effects reads so multiple
  configured events can actually fit in a six-second example.
- Relative SNR supports `active_rms`, measured only from frames above a
  configurable activity threshold.
- Sources rejected as too quiet by normalization are no longer eligible for
  SNR amplification, whether they are the anchor or a non-anchor source.
- `pad_or_concatenate` probabilities preserve realistic local silence instead
  of forcing every active stem to fill the whole training window.

### Label-consistent broadcast rendering

`BroadcastStemRenderer` operates on target stems before the mixture is summed:

1. Optional shared measured RIR for known-dry speech/effects.
2. Shared linear channel EQ.
3. Optional per-source compressor gain envelopes.
4. Dialogue side-chain ducking of music/effects.
5. One bus-compressor gain envelope computed from the mixture and applied to
   every stem.
6. One shared final peak scale applied to every stem.

Nonlinear bus processing is never applied to the mixture alone. Applying the
same bus gain envelope to every stem preserves exact target additivity.

The renderer validates timing constants, compressor ratios, EQ/room ranges,
stem names, and the peak limit before training starts. Zero-energy RIR files
are rejected instead of silently turning an active target into silence. The
recipe keeps `preserve_rms: false`: one shared normalized RIR is applied
without independently restoring each stem's pre-room RMS, preserving the
shared-room level relationship.

### Room metadata safety

Source manifests may add these optional columns:

| Column | Meaning |
|---|---|
| `filepath` | Source audio path; required |
| `type` | `speech`, `music`, or `effects`; required |
| `split` | Optional loader split |
| `is_wet` | `true` if the source already contains room/reverb |
| `scene_id` | Recommended scene grouping identifier |
| `recording_id` | Recommended source-recording leakage guard |
| `speaker_id` | Recommended speaker-disjoint split key |
| `content_class` | Recommended semantic class such as Foley, ambience, or vocal music |

The v1 recipe uses `unknown_wet_policy: assume_wet`. RIR augmentation is
therefore skipped unless all active shared-room stems have explicit dry
metadata. This avoids accidental double reverberation.

### Local leakage loss

`TeacherStudentDistillationTask` now has an opt-in frame-local silent-source
penalty. Unlike the existing clip-level penalty, it penalizes predicted energy
during silent regions inside a source that is active elsewhere in the same
six-second crop. The recipe uses `frame_silent_source_db: -80.0` because the
synthetic inactive regions are exact zeros; this avoids classifying quiet but
valid programme material around -60 dBFS as a silent target.

### Real multitrack blending

`OnTheFlyStemDataModule` can blend a fixed real multitrack dataset into the
on-the-fly training stream with:

```yaml
datamodule:
  supplemental_fixed_mix_manifest_csv: /data1/manifests/real_multitrack_train.csv
  supplemental_fixed_mix_probability: 0.15
```

The fixed manifest uses one row per mixture:

```csv
mixture_filepath,speech_filepath,music_filepath,effects_filepath
/data/mix.wav,/data/dx.wav,/data/mx.wav,/data/fx.wav
```

The rendered mixture is used as input, while the three paths are supervised
targets. Run the additivity audit before enabling this training branch. The
recipe additionally sets `fixed_mix_max_additivity_error_db: -40.0`, so a
rendered fixed mixture that differs too much from the sum of its supervised
stems fails at sample loading instead of corrupting residual-Effects labels.

Empty cells still mean an inactive stem, but labeled audit manifests must
contain every configured `{stem}_filepath` column. Audit-time sample-rate
conversion uses `torchaudio` band-limited resampling rather than linear
interpolation.

## New recipe

```text
recipes/dnr/models/tvconv-pyramid-sourceaware-sfclite-convgru-smoothup-smoothlogit-npu.speech-music-residual-sfx.robust-distill.rt192k.fp512keep475.broadcast-v1/config.yaml
```

RIR is disabled by default because the actual corpus metadata and RIR assets
were not available in this workspace. To enable it safely:

```yaml
broadcast_rir_probability: 0.30
broadcast_rir_paths:
  - /data1/room_ir/room_001.wav
  - /data1/room_ir/room_002.wav
```

Every speech/effects row that may receive RIR must also have `is_wet=false`.

## Three-tier evaluation

The audit tool accepts the same wide manifest convention for all tiers.

### Tier 1: fixed OOD synthetic

```bash
.venv/bin/python tools/audit_separation_manifest.py \
  /data1/manifests/ood_synthetic_test.csv \
  --tier ood_synthetic \
  --sample-rate 44100 \
  --output-json logs/domain_gap/ood_synthetic.json
```

Use sources, speakers, recording IDs, content classes, and RIRs absent from
training.

### Tier 2: real multitrack

```bash
.venv/bin/python tools/audit_separation_manifest.py \
  /data1/manifests/real_multitrack_test.csv \
  --tier real_multitrack \
  --sample-rate 44100 \
  --output-json logs/domain_gap/real_multitrack.json
```

If predictions are available, add `pred_speech_filepath`,
`pred_music_filepath`, and `pred_effects_filepath` columns. The report then
includes SI-SDR and reference-inactive leakage for each predicted stem.

### Tier 3: unlabeled real TV

```bash
.venv/bin/python tools/audit_separation_manifest.py \
  /data1/manifests/real_tv_unlabeled.csv \
  --tier real_unlabeled \
  --sample-rate 44100 \
  --output-json logs/domain_gap/real_tv_unlabeled.json
```

Only `mixture_filepath` is required. This tier measures input waveform/domain
statistics. Model-output listening, VAD/WER, and pseudo-label acceptance remain
data/checkpoint-dependent gates.

## Training

Set the teacher checkpoint in the recipe or command override, then run:

```bash
bash recipes/dnr/models/tvconv-pyramid-sourceaware-sfclite-convgru-smoothup-smoothlogit-npu.speech-music-residual-sfx.robust-distill.rt192k.fp512keep475.broadcast-v1/train.sh
```

Before a long run, render and listen to a deterministic fixed sample set:

```bash
.venv/bin/python tools/export_fixed_stem_mixes.py \
  --source-manifest-csv /data1/manifests/train_sources.csv \
  --synthesis-json /path/to/broadcast_synthesis.json \
  --output-csv /tmp/broadcast_preview.csv \
  --output-split preview \
  --num-examples 100 \
  --sr 44100 \
  --duration 6.0 \
  --seed 2026 \
  --export-mixtures
```

## Required ablations

Keep model weights, optimizer, seed, and training steps fixed where possible.

| Run | Renderer changes | Purpose |
|---|---|---|
| A | Original `dry_mix` | Baseline |
| B | Per-stem crops + active RMS SNR | Isolate event/level correction |
| C | B + room/channel | Isolate acoustic-domain correction |
| D | C + ducking/source/bus dynamics | Full supervised broadcast renderer |
| E | D + fixed real multitracks | Measure supervised real-data gain |
| F | E + real-TV pseudo-label adaptation | Data-dependent final stage |

Select the winning renderer using Tier 2 leakage and Tier 3 listening/VAD/WER,
not synthetic SI-SDR alone.

## Codec and pseudo-label boundary

AAC/AC-3-style final-bus coding is not additive: independently encoded stems
do not sum exactly to the encoded mixture. It is therefore not part of the
supervised residual-Effects renderer. Use codec material in one of these
separate stages:

1. Evaluation-only channel corruption.
2. Fixed real multitrack data whose delivered stems match the delivered mix.
3. Teacher/pseudo-label adaptation on real decoded programme audio.

Do not silently assign final-bus codec residual to the Effects target.

## Validation commands

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
    .venv/bin/python -m pytest tests/test_proposed_separation_models.py \
    -q

.venv/bin/python tools/expand_recipe_config.py \
  recipes/dnr/models/tvconv-pyramid-sourceaware-sfclite-convgru-smoothup-smoothlogit-npu.speech-music-residual-sfx.robust-distill.rt192k.fp512keep475.broadcast-v1/config.yaml \
  --strict \
  --output /tmp/ass_broadcast_v1_resolved.yaml
```

Review validation on 2026-08-17:

- 39 dataset, datamodule, renderer, normalization, and audit tests passed.
- 56 proposed-model/task tests passed, including the new quiet-source mask
  regression and existing ONNX audit smoke tests.
- Ruff, Python bytecode compilation, `git diff --check`, and strict recipe
  expansion passed.

## Data-dependent completion gates

The code path is complete, but these claims cannot be validated without the
actual corpus, RIR assets, real multitracks, real TV examples, and trained
checkpoint:

- the true dry/wet distribution;
- real source activity/SNR/LUFS distributions;
- speaker/recording/content overlap between splits;
- the best augmentation probabilities;
- improvement in real-TV leakage after training;
- whether an explicit three-head Effects output beats residual Effects.
