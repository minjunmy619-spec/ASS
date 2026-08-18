# CLAP and Whisper Perceptual Integration Operation

Date: 2026-08-17

> This document records the original single-prompt CLAP baseline. The
> scene-independent CLAP Audio-to-Audio and Prompt Bank extension is documented
> in `clap_a2a_prompt_bank_whisper_20260818_operation.md`; the original recipe
> remains unchanged for controlled A/B experiments.

## Scope and terminology

The research note describes **CLAP** (Contrastive Language-Audio Pretraining),
not image-oriented CLIP. This implementation therefore integrates LAION CLAP
and OpenAI Whisper. Neither network is added to the separator inference graph:
both are frozen training/evaluation teachers, so the causal NPU model and its
ONNX/Circle deployment interface remain unchanged.

The integration has two separate paths:

1. Training: differentiable CLAP semantic anti-bleed and Whisper speech-feature
   matching losses.
2. Real-wave evaluation: CLAP stem-purity scores and Whisper ASR confidence on
   already-separated WAV files, without the on-the-fly synthesis pipeline.

## Implemented files

| File | Purpose |
|---|---|
| `spectral_feature_compression/core/loss/frozen_audio_perceptual.py` | Frozen CLAP and Whisper loss adapters |
| `spectral_feature_compression/core/tasks/distillation_task.py` | Loss scheduling, weighting, source selection, and logging |
| `tools/evaluate_clap_whisper_stems.py` | No-reference scoring plus optional paired-reference diagnostics |
| `requirements-perceptual.txt` | Optional external packages |
| `recipes/dnr/models/tvconv-pyramid-sourceaware-sfclite-convgru-smoothup-smoothlogit-npu.speech-music-residual-sfx.robust-distill.rt192k.fp512keep475.broadcast-v1.clap-whisper-ft/config.yaml` | Post-convergence perceptual fine-tuning recipe |
| `recipes/dnr/models/tvconv-pyramid-sourceaware-sfclite-convgru-smoothup-smoothlogit-npu.speech-music-residual-sfx.robust-distill.rt192k.fp512keep475.broadcast-v1.clap-whisper-ft/clap_prompts.yaml` | Shared training/evaluation CLAP prompts |
| `tests/test_frozen_audio_perceptual.py` | Frozen-model, gradient, shape, and silence-mask regressions |
| `tests/test_evaluate_clap_whisper_stems.py` | Real-stem manifest and metric aggregation regression |
| `tests/test_proposed_separation_models.py` | Distillation-task integration regressions |

## Training data flow

```text
OnTheFlyStemDataModule
  -> mixture [B, C, T], reference stems [B, S, C, T]
  -> separator
  -> estimated stems [B, S, C, T]
       |-- existing waveform/spectral/distillation losses
       |-- frozen CLAP semantic loss on active Speech/Music/Effects targets
       `-- frozen Whisper feature loss on the active Speech target
  -> weighted total loss
  -> gradients update separator only
```

The current on-the-fly dataset/datamodule contract is unchanged. The perceptual
losses consume its final rendered reference stems, including the broadcast
renderer when enabled. `reference_activity_db` masks absent reference sources,
which prevents a semantic prompt from forcing content into a genuinely silent
stem.

### CLAP semantic loss

For every active estimated stem, CLAP computes an audio embedding and compares
it with cached text embeddings:

```text
L_clap = positive_weight * mean(1 - positive_similarity)
       + negative_weight * mean(relu(negative_similarity - margin))
```

The shared prompt file defines source-specific positive and negative prompts.
Speech is contrasted with Music/Effects, Music with spoken dialogue, and Effects
with dialogue and Music. Singing is not a negative for Music because it may be
valid Music content; change that choice only if the actual dataset taxonomy
assigns singing to Speech. `purity_margin = positive_similarity -
negative_similarity` is reported by the evaluation tool, where larger is
better. It is a diagnostic score, not a calibrated MOS and not a substitute for
reference-based SI-SDR on real multitracks.

Audio is resampled differentiably to CLAP's 48 kHz input rate. Training uses
the tensor input API rather than the file-list API so gradients can reach the
separator estimate. Audio longer than 10 seconds uses consecutive,
non-overlapping windows; the final short window is passed at its true length.
The current training crop is 6 seconds, so training normally uses one CLAP
window, while real programme evaluation covers the complete file.

### Whisper feature-matching loss

Only the configured `whisper_source` (Speech by default) is passed to Whisper.
Both estimate and reference are differentiably resampled to 16 kHz, converted
to log-mel features with PyTorch operations, and sent through the frozen audio
encoder. The reference branch runs under `torch.no_grad()`; the estimate branch
keeps its graph. L1 distance is averaged over the selected encoder blocks.

The research-note sample calls `whisper.encoder(mel)` directly on short mel
crops. Official Whisper checks for its fixed 30-second context and would reject
that shape. The integrated adapter explicitly executes the official encoder
convolutions and blocks and applies only the required prefix of its positional
embedding. This preserves short-crop training without padding every example to
30 seconds.

### Frozen-model and checkpoint behavior

- All CLAP/Whisper parameters have `requires_grad=False` and stay in evaluation
  mode when the parent task enters training mode.
- Frozen external networks move with the Lightning task but are intentionally
  excluded from the separator checkpoint `state_dict`. Derived CLAP prompt
  embeddings are non-persistent buffers and are rebuilt from the active model
  and prompt configuration; adapter constants such as mel filters remain
  ordinary buffers.
- CLAP/Whisper are training-side modules only. They do not change separator
  parameter count, recurrent state, causal execution, exported ONNX, or Circle
  compilation.

### Scheduling and memory

The new task arguments are:

| Key | Meaning |
|---|---|
| `clap_semantic_loss` / `_weight` | CLAP adapter and outer loss weight |
| `whisper_feature_loss` / `_weight` | Whisper adapter and outer loss weight |
| `whisper_source` | Stem used for Whisper feature matching |
| `perceptual_loss_start_step` | Optimizer step at which both losses start |
| `perceptual_loss_every_n_steps` | Compute cadence during training; validation always computes after the start step |
| `perceptual_loss_compensate_cadence` | Multiply loss on evaluated train steps by the cadence to preserve its long-run mean weight |

The recipe uses outer weights `0.10` and `0.10`, evaluates the expensive losses
every fourth optimizer step, and applies a factor of four on those steps so the
configured mean weights remain `0.10`. Raw metric logs are not scaled. It also
reduces the physical batch size to one and uses four-step gradient accumulation.
It is intended as a fine-tuning stage from a converged broadcast-v1 separator,
not as the initial training recipe.

## Installation

Install the base project first, then the optional models:

```bash
.venv/bin/python -m pip install -r requirements-perceptual.txt
```

OpenAI Whisper also requires the system `ffmpeg` executable:

```bash
ffmpeg -version
```

`laion-clap` and `openai-whisper` are deliberately not in the base requirements
because their dependencies and pretrained weights are large. For reproducible
jobs, set `clap_checkpoint_path` to a pinned local checkpoint and set
`clap_allow_download=false`. A null CLAP checkpoint with downloads enabled uses
the package's selected `model_id` download behavior.

## Fine-tuning

Launch from the repository root and provide both the student warm-start and the
existing separation-teacher checkpoint:

```bash
bash recipes/dnr/models/tvconv-pyramid-sourceaware-sfclite-convgru-smoothup-smoothlogit-npu.speech-music-residual-sfx.robust-distill.rt192k.fp512keep475.broadcast-v1.clap-whisper-ft/train.sh \
  fine_tune_checkpoint_path=/path/to/converged_separator.ckpt \
  perceptual_teacher_checkpoint_path=/path/to/separation_teacher.ckpt \
  clap_checkpoint_path=/path/to/music_audioset_epoch_15_esc_90.14.pt \
  clap_audio_model=HTSAT-base \
  clap_allow_download=false
```

Both separator checkpoint arguments are mandatory, so the recipe fails before
model construction if either is omitted. The warm-start path loads separator
weights through the existing `SupTask.pretrained_model_path` mechanism. It does
not resume optimizer state; this is a new fine-tuning run. Monitor these added
metrics:

```text
training/loss_clap_semantic
training/loss_whisper_feature
validation/loss_clap_semantic
validation/loss_whisper_feature
```

Before a long job, check GPU memory and run a short overfit/smoke job. If memory
is insufficient, first use a smaller Whisper model or increase the perceptual
cadence; do not silently remove the reference activity mask.

## Evaluation on real audio waves

The evaluation manifest describes model outputs, not source pools and not
synthesized examples. Required columns are:

```csv
recording_id,speech_filepath,music_filepath,effects_filepath
real_001,/pred/real_001_speech.wav,/pred/real_001_music.wav,/pred/real_001_effects.wav
```

All paths must refer to already-separated real programme audio. The evaluator
does only mono downmixing and resampling needed by the frozen scorers; it does
not discard any part of a source, mix stems, add RIRs, compress, or apply the
broadcast synthesis pipeline. CLAP uses complete-file 10-second windows and
Whisper uses complete-file 30-second windows. The three separated stems in each
row must have equal lengths after resampling; mismatches fail explicitly.
The CLI now defaults to the prompt banks used by the versioned A2A fine-tuning
recipe. `--clap-prompt-config` remains available for an intentional experiment
override, including reproduction of this original single-prompt baseline.

```bash
.venv/bin/python tools/evaluate_clap_whisper_stems.py \
  /data1/manifests/real_tv_separated.csv \
  --clap-checkpoint /models/clap/music_audioset_epoch_15_esc_90.14.pt \
  --clap-audio-model HTSAT-base \
  --clap-prompt-config recipes/dnr/models/tvconv-pyramid-sourceaware-sfclite-convgru-smoothup-smoothlogit-npu.speech-music-residual-sfx.robust-distill.rt192k.fp512keep475.broadcast-v1.clap-whisper-ft/clap_prompts.yaml \
  --whisper-model base \
  --whisper-language en \
  --output-json logs/real_tv_clap_whisper.json
```

Useful report fields:

- CLAP per stem: duration-weighted complete-file means plus timestamped `windows` containing
  `positive_similarity`, `negative_similarity`, and `purity_margin`.
- Whisper per requested stem: duration-weighted `avg_logprob`, duration-weighted
  `no_speech_probability`, transcript, and all decoded windows. Windows are
  decoded directly, so high-no-speech windows are not dropped by
  `transcribe()` filtering.
- Dataset summary: count, mean, standard deviation, minimum, and maximum.

By default Whisper runs on all three stems. Speech should retain intelligible
content, while Music/Effects should usually have lower speech confidence and
higher no-speech probability. Compare distributions and paired recordings,
not one universal threshold. Prompt choice, language, loudness, and programme
domain can all move the absolute scores. Missing metrics use JSON `null`; the
report never emits non-standard `NaN` values.

## Validation performed

```bash
.venv/bin/ruff check \
  spectral_feature_compression/core/loss/frozen_audio_perceptual.py \
  spectral_feature_compression/core/tasks/distillation_task.py \
  tools/evaluate_clap_whisper_stems.py \
  tests/test_frozen_audio_perceptual.py \
  tests/test_evaluate_clap_whisper_stems.py \
  tests/test_clap_whisper_recipe.py \
  tests/test_proposed_separation_models.py

.venv/bin/python -m py_compile \
  spectral_feature_compression/core/loss/frozen_audio_perceptual.py \
  spectral_feature_compression/core/tasks/distillation_task.py \
  tools/evaluate_clap_whisper_stems.py

NUMBA_CACHE_DIR=/tmp/ass-numba-cache \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_frozen_audio_perceptual.py \
  tests/test_evaluate_clap_whisper_stems.py \
  tests/test_clap_whisper_recipe.py \
  tests/test_proposed_separation_models.py
```

Results on 2026-08-17:

- Ruff and bytecode compilation passed.
- Recipe inheritance, shared prompt resolution, and both mandatory checkpoint
  interpolations passed their dedicated regression test.
- The recipe launcher passes shell syntax validation and displays CLI help correctly.
- 71 relevant tests passed, including differentiable estimate gradients, frozen
  teacher parameters, exact external-model checkpoint exclusion, silent
  reference masks, full-file window scoring, standard JSON serialization, task
  scheduling/weighting/logging, mandatory recipe checkpoints, and the existing
  separator/ONNX smoke regressions.

## Remaining data-dependent gates

The current environment does not contain `laion-clap` or `openai-whisper`, and
no pretrained checkpoints or actual separated real-TV WAV set were supplied.
Tests therefore validate the adapters with API-compatible deterministic dummy
teachers, but the following claims remain intentionally unverified:

- successful loading and peak GPU memory of the chosen real CLAP/Whisper
  checkpoints in the training environment;
- throughput impact at the configured cadence;
- CLAP/Whisper metric distributions and usable thresholds on the target
  language/programme domain;
- improvement in real-TV leakage after perceptual fine-tuning.

Complete those gates with a fixed real-TV evaluation manifest and an A/B run
against the identical broadcast-v1 checkpoint, optimizer budget, and random
seed. Keep listening tests and real multitrack reference metrics alongside the
no-reference scores to detect metric gaming.
