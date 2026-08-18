# Scene-Independent CLAP A2A, Prompt Bank, and Whisper Integration

Date: 2026-08-18

## Outcome

This extension adds the scene-independent objectives from
`audio_eval_train_with_clap_whisper.md` without changing the separator,
`OnTheFlyStemDataset`, datamodule output contract, or NPU inference graph.

The existing `ClapSemanticLoss` remains backward compatible. The original
`broadcast-v1.clap-whisper-ft` recipe still runs its single positive/negative
text-prompt objective. A new versioned recipe explicitly enables:

1. CLAP predicted-stem to matching-reference audio alignment;
2. reference-relative cross-stem anti-bleed;
3. source-taxonomy Prompt Bank classification; and
4. the existing Whisper Speech estimate/reference feature matching.

Only the separator receives gradients. CLAP and Whisper remain frozen
training/evaluation teachers and are excluded from separator checkpoints.

## Files

| File | Change |
|---|---|
| `spectral_feature_compression/core/loss/frozen_audio_perceptual.py` | Shared CLAP multi-objective loss, masks, metrics, and cached prompt banks |
| `spectral_feature_compression/core/tasks/distillation_task.py` | Logs the total CLAP loss and each raw component without adding components twice |
| `tools/evaluate_clap_whisper_stems.py` | Prompt-bank diagnostics and optional reference-stem A2A diagnostics |
| `recipes/dnr/models/tvconv-pyramid-sourceaware-sfclite-convgru-smoothup-smoothlogit-npu.speech-music-residual-sfx.robust-distill.rt192k.fp512keep475.broadcast-v1.clap-a2a-bank-whisper-ft/` | Versioned fine-tuning recipe, launcher, and universal source prompt banks |
| `tests/test_frozen_audio_perceptual.py` | A2A, anti-bleed, prompt-count normalization, silence, gradient, and validation regressions |
| `tests/test_evaluate_clap_whisper_stems.py` | Prompt-bank/reference report regressions |
| `tests/test_clap_whisper_recipe.py` | New recipe inheritance and weight regression |
| `tests/test_proposed_separation_models.py` | Component logging and total-weight regression |

## Training data flow

```text
mixture [B,C,T] -> separator -> estimate [B,S,C,T]
                                   |
reference [B,S,C,T] ---------------+
  |                                |
  |                                +-> existing waveform/spectral/distillation losses
  |                                +-> Whisper Speech feature matching
  |                                `-> one frozen CLAP instance
  |                                      +-> estimate audio embeddings (gradient kept)
  `-------------------------------------> reference embeddings (no_grad)
                                         +-> A2A match
                                         +-> relative anti-bleed
                                         `-> cached Prompt Bank classification
```

The dataset already returns final rendered Speech/Music/Effects references and
constructs the mixture by summing them. No new manifest field, scene label, or
datamodule branch is required for training.

## Loss definitions

For active source `i` and CLAP audio encoder `E`:

```text
L_match(i) = 1 - cosine(E(estimate_i), stopgrad(E(reference_i)))
```

The direct research-note penalty `relu(cosine(E(estimate_i), E(reference_j)))`
would incorrectly punish legitimate correlation between clean programme stems.
The implementation instead measures only affinity added beyond the clean
reference baseline:

```text
L_antibleed(i,j) = relu(
    cosine(E(estimate_i), E(reference_j))
  - cosine(E(reference_i), E(reference_j))
  - margin
), i != j
```

Therefore an exact clean reference has zero relative anti-bleed penalty even
when its Speech, Music, and Effects embeddings are naturally correlated.

Reference activity is computed independently for each source and each CLAP
window. Same-stem matching ignores inactive reference windows. Cross-stem
anti-bleed requires both reference sources to be active; existing waveform
silence losses remain responsible for suppressing output in an absent target.
The final short CLAP window is duration weighted rather than counted as a full
10-second window.

### Prompt Bank

All prompt embeddings are cached once per construction. They are non-persistent
buffers, so a resumed separator checkpoint cannot silently restore embeddings
from a different CLAP model or prompt file; the active configuration recomputes
them. For source bank `c`, individual prompt logits are aggregated with
`logmeanexp`:

```text
bank_logit(c) = logsumexp(prompt_logits_c) - log(number_of_prompts_c)
```

Cross entropy is then applied over the three bank logits. This keeps a bank
from receiving more prior mass merely because it has more prompt strings.
Inactive target windows are masked.

The prompt file describes source taxonomy rather than scenes. It deliberately
contains no movie/sports/documentary scene classifier. In this project version,
singing belongs to Music. If the actual source manifest puts isolated singing
vocals in Speech, update the Speech and Music banks together before training;
the same semantic concept must not be a positive label for both classes.

## Weights and scheduling

The new recipe starts conservatively:

| Objective | Weight |
|---|---:|
| CLAP A2A match | 0.05 |
| CLAP relative anti-bleed | 0.05 |
| CLAP Prompt Bank | 0.02 |
| Whisper Speech feature matching | 0.10 |
| Legacy CLAP positive/negative text loss | disabled |

The task-level CLAP multiplier is `1.0`; the component weights above are the
actual weights. This avoids a hidden second multiplication. The inherited
schedule evaluates CLAP/Whisper every fourth optimizer step during training and
multiplies their applied loss by four on those steps. This preserves the stated
long-run weights instead of unintentionally reducing them to one quarter. Raw
component logs remain unscaled, and `training/perceptual_cadence_scale` records
the applied factor. Validation computes every batch with scale `1.0`. Fine-tune
from a converged separator checkpoint; do not use these large frozen teachers
for initial training.

Logged fields include:

```text
loss_clap_semantic
loss_clap_semantic_text_positive
loss_clap_semantic_text_negative
loss_clap_semantic_audio_match
loss_clap_semantic_audio_antibleed
loss_clap_semantic_prompt_bank
loss_whisper_feature
```

Raw components are logged for diagnosis. Only `loss_clap_semantic`, multiplied
by the task-level CLAP weight, is added to the total loss.

## Training

Install the optional teachers and make their checkpoints available first:

```bash
.venv/bin/python -m pip install -r requirements-perceptual.txt
```

Run the new recipe:

```bash
recipes/dnr/models/tvconv-pyramid-sourceaware-sfclite-convgru-smoothup-smoothlogit-npu.speech-music-residual-sfx.robust-distill.rt192k.fp512keep475.broadcast-v1.clap-a2a-bank-whisper-ft/train.sh \
  fine_tune_checkpoint_path=/models/converged_student.ckpt \
  perceptual_teacher_checkpoint_path=/models/separation_teacher.ckpt \
  clap_a2a_checkpoint_path=/models/clap/music_speech_audioset_epoch_15_esc_89.98.pt
```

The recipe pins `clap_audio_model=HTSAT-base` and disables downloads because
that checkpoint uses the base audio encoder. All three checkpoint paths are
mandatory. The inherited recipe uses batch size 1 and accumulation 4 because
the separator teacher, CLAP, and Whisper coexist during fine-tuning.

## Evaluation

For unlabelled separated real audio, the existing columns remain sufficient:

```csv
recording_id,speech_filepath,music_filepath,effects_filepath
```

The report now also includes `prompt_bank_probability` and
`prompt_bank_margin` when prompt banks are configured.

For labelled synthetic or real multitrack evaluation, add all three optional
reference columns:

```csv
recording_id,speech_filepath,music_filepath,effects_filepath,reference_speech_filepath,reference_music_filepath,reference_effects_filepath
```

All three reference columns must appear together and all six waveforms must
have equal duration after resampling. The report adds:

- `reference_similarity`: matching predicted/reference CLAP cosine similarity;
- `relative_cross_stem_excess`: cross-stem similarity above the clean-reference
  baseline and configured margin.

Inactive reference windows are emitted as JSON `null`, not `NaN`.

```bash
.venv/bin/python tools/evaluate_clap_whisper_stems.py \
  /data/manifests/paired_separated_stems.csv \
  --clap-checkpoint /models/clap/music_speech_audioset_epoch_15_esc_89.98.pt \
  --clap-audio-model HTSAT-base \
  --clap-audio-antibleed-margin 0.02 \
  --whisper-model base \
  --output-json logs/clap_a2a_bank_whisper.json
```

The evaluator records the CLAP audio model and anti-bleed margin in
`clap_config`, making training/evaluation configuration drift visible.

## Validation commands

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

Validation result on 2026-08-19: Ruff, bytecode compilation, shell syntax, and
`git diff --check` passed. The complete command above passed **84 tests**,
including the separator ONNX smoke regressions already present in
`tests/test_proposed_separation_models.py`.

## Data-dependent acceptance gates

Code tests cannot establish an audio-quality improvement. Before promoting the
new recipe, compare it with the unchanged `broadcast-v1.clap-whisper-ft`
baseline using the same warm-start checkpoint, seed, optimizer steps, and real
evaluation set. Accept it only if:

1. reference separation metrics do not regress materially;
2. real-audio Speech leakage into Music/Effects decreases;
3. music containing singing stays in Music according to the chosen taxonomy;
4. listening tests do not reveal hollow Speech, damaged ambience, or suppressed
   correlated content; and
5. GPU memory and fine-tuning throughput remain operational at the configured
   four-step cadence.

No CLAP or Whisper operator enters ONNX/Circle export, so this integration does
not change NPU operator compatibility, model parameters, runtime state, or
streaming latency.

The current workspace does not have `laion-clap`, `openai-whisper`, or the real
teacher checkpoints installed. API-compatible deterministic teachers validate
tensor shapes, masking, gradients, checkpoint exclusion, and aggregation, but
real-checkpoint loading, GPU memory, throughput, and audio-quality improvement
remain data-dependent gates.
