# Two-stage robust spatial USS-conditioned TSE

This document describes the new TSE variant:

```text
src.models.deft.two_stage_tse.TwoStageRobustSpatialBridgeTSE
```

and the accompanying training/evaluation changes.

The variant is designed for the final DCASE Task 4 style pipeline where the **only external evaluation input is the mixture waveform**. All other TSE inputs must be produced internally by the system:

```text
mixture → USS → SC → TSE
```

The model therefore treats USS auxiliary outputs as **optional, noisy, mixture-derived hints**, not as oracle information.

---

## 1. Why this variant was added

Two existing TSE-style models had complementary strengths.

### `ModifiedDeFTTSEMemoryEfficientTemporal`

Strengths:

- Uses USS/TSE enrollment waveform: `enrollment [B, Q, 1, T]`.
- Supports `query_condition` / `tse_condition` via FiLM.
- Supports `temporal_conditioning` via time-varying FiLM.
- Uses memory-efficient DeFT blocks and chunked inference.
- Naturally fits the USS → SC → TSE handoff.

Weaknesses:

- Reconstructs from mixture channel `0` only.
- Does not explicitly use FOA spatial features at TSE output.
- Processes much of the backbone per query.
- Does not directly consume separate USS spatial/confidence hints.

### `DeFTTSELikeSpatialTemporal`

Strengths:

- Uses explicit FOA-inspired mixture features.
- Predicts multichannel spatial masks.
- Uses all mixture channels at reconstruction time.
- Has a shared mixture trunk, which is efficient for multiple output slots.

Weaknesses:

- Does not use enrollment waveform.
- Does not use USS bridge/query condition.
- Does not consume temporal conditioning.
- Label conditioning is late and relatively shallow.
- Uses full-axis attention rather than the memory-efficient DeFT blocks.

### Resulting design

The new variant combines the useful parts:

```text
shared mixture spatial trunk
+ per-query enrollment refinement
+ label/query/temporal/spatial FiLM
+ multichannel spatial mask output
+ robust gated optional USS hints
```

---

## 2. Main implementation files

### New model

```text
src/models/deft/two_stage_tse.py
```

Main class:

```py
TwoStageRobustSpatialBridgeTSE
```

### Training Lightning modules updated

```text
src/training/lightningmodule/online_teacher_tse.py
src/training/lightningmodule/uss_sc_tse_pipeline_finetune.py
```

These now forward extra USS-derived optional condition tensors to TSE:

```py
class_logits
silence_logits
pred_doa_vector
doa_vector
spatial_embedding
```

`used_spatial_vector` is intentionally not forwarded by default to avoid ambiguity with oracle/scheduled spatial conditioning used during USS training.

### S5 inference modules updated

```text
src/models/s5/kwo2025.py
src/models/s5/kwo2025_temporal.py
```

These now pass the same optional USS hints into TSE during final pipeline inference.

### Robust condition curriculum callback

```text
src/training/callbacks/robust_tse_condition_curriculum.py
```

Main class:

```py
RobustTSEConditionCurriculum
```

This schedules:

```py
condition_dropout
temporal_condition_dropout
spatial_condition_dropout
condition_noise_std
spatial_condition_noise_std
```

on the TSE model during training.

### Evaluation script updated

```text
src/evaluation/evaluate_stage.py
```

Updates:

- Standalone `--stage tse` now forwards the new optional TSE condition keys if present in the dataset batch.
- New `--stage s5` runs the full mixture-only pipeline through `predict_label_separate()`.
- TSE conditioning detection now recognizes `spatial_conditioner`, `temporal_conditioner`, and `confidence_gate`.

---

## 3. Model architecture

### Required inputs

The model requires only:

```py
{
    "mixture": mixture,           # [B, 4, T]
    "enrollment": enrollment,     # [B, Q, 1, T]
    "label_vector": label_vector, # [B, Q, 18] or compatible
}
```

In final S5 evaluation these are produced internally:

```text
mixture → USS foreground_waveform → enrollment
mixture → USS foreground_waveform → SC → label_vector
```

### Optional inputs

The model can also consume:

```py
query_condition / tse_condition / bridge_condition / proposal_condition
temporal_conditioning / foreground_activity_logits
class_logits
silence_logits
pred_doa_vector
doa_vector
spatial_condition
spatial_embedding
```

These are optional. The model should still run when they are absent.

### Stage 1: shared mixture spatial trunk

```text
mixture waveform
    → STFT
    → FOA/logmag/AIV/IPD feature encoder
    → memory-efficient DeFT scene blocks
    → shared scene representation [B, C, frames, freq]
```

The FOA encoder is reused from:

```py
src.models.deft.foa_spatial_features.FOASpatialFeatureEncoder
```

### Stage 2: per-query refinement

```text
shared scene representation
+ enrollment STFT features
    → fusion conv
    → per-query memory-efficient DeFT blocks
    → multichannel spatial mask head
    → waveform [B, Q, 1, T]
```

### Conditioning

The query refinement blocks receive FiLM parameters from:

```text
label_vector              → class FiLM
query/tse_condition       → gated query FiLM
temporal_conditioning     → gated temporal FiLM
pred_doa_vector/spatial   → gated spatial FiLM
```

The effective FiLM is:

```text
beta_total  = beta_label + g_query * beta_query + g_temporal * beta_temporal + g_spatial * beta_spatial
gamma_total = gamma_label + g_query * gamma_query + g_temporal * gamma_temporal + g_spatial * gamma_spatial
```

### Confidence-aware gates

The model can use a confidence gate conditioned on:

```text
enrollment RMS
USS active probability from silence_logits
USS class confidence from class_logits
label strength
presence flags for class/silence signals
```

This helps reduce reliance on under-trained or noisy USS auxiliary heads.

### Spatial output

The spatial output head predicts per-query, per-channel masks:

```text
[B, Q, mixture_channels, 3, frames, freq]
```

where the three mask components are:

```text
magnitude mask
phase real
phase imag
```

The masked multichannel complex spectra are projected with a bias-free `1x1` channel projection:

```py
self.out_conv = nn.Conv2d(mixture_channels, output_channels, kernel_size=1, bias=False)
```

### Chunked inference

Waveforms are stitched with overlap-add. Activity logits are now also stitched onto the full utterance frame timeline instead of concatenating overlapping chunk logits.

### BF16-mixed training support

`TwoStageRobustSpatialBridgeTSE` keeps numerically sensitive spectral operations in FP32 during autocast:

- STFT / ISTFT
- complex phase normalization for spatial masks
- reference-fallback complex reconstruction

This follows the same safety principle as the existing BF16-safe DeFT USS wrappers. The model is intended to work with Lightning `bf16-mixed` precision on CUDA-capable training hardware. A CPU BF16 forward smoke test is supported, but this checkout's CPU backend does not support BF16 backward, so full mixed-precision train-step validation must be done on the target GPU machine.

---

## 4. Training config

The main training config is:

```text
config/separation/modified_deft_tse_lite_6s_online_teacher_two_stage_spatial_uss_sc.yaml
```

It follows the same config-reference style as the pipeline finetune configs:

```yaml
datamodule:
  from_training_config: modified_deft_tse_lite_6s_online_teacher_uss_sc.yaml
  key: datamodule

uss_model:
  from_training_config: modified_deft_sc_primary_frozenuss_pretrainedsed_fusion.yaml
  key: lightning_module.args.uss_model

sc_model:
  from_training_config: ../label/m2d_sc_stage3_estimated_pretrainedsed_fusion.yaml
  key: lightning_module.args.model

loss:
  from_training_config: modified_deft_tse_lite_6s_temporal.yaml
  key: lightning_module.args.loss

optimizer:
  from_training_config: modified_deft_tse_lite_6s_temporal.yaml
  key: lightning_module.args.optimizer
```

Only the TSE model block is new:

```yaml
model:
  module: src.models.deft.two_stage_tse
  main: TwoStageRobustSpatialBridgeTSE
```

### Important model settings

```yaml
query_condition_dim: 256
spatial_condition_dim: 3
temporal_conditioning_enabled: true

auxiliary_gate_init: -3.0
use_confidence_gates: true

condition_dropout: 0.8
temporal_condition_dropout: 0.8
spatial_condition_dropout: 1.0
condition_noise_std: 0.05
spatial_condition_noise_std: 0.10

enable_reference_fallback: true
spatial_output_gate_init: -3.0
```

The initial dropout/noise is intentionally strong so the model first learns a robust path from:

```text
mixture + enrollment + label_vector
```

before relying on noisy USS auxiliary hints.

---

## 5. Pre-work before training

Before training this variant, prepare the same teacher checkpoints used by the online-teacher pipeline.

### Required checkpoints

```text
checkpoint/modified_deft_uss_lite_6s_unified_all_features.ckpt
checkpoint/m2d_sc_stage3_estimated_pretrainedsed_fusion.ckpt
```

These are referenced in the config as:

```yaml
uss_pretrained_ckpt: checkpoint/modified_deft_uss_lite_6s_unified_all_features.ckpt
sc_pretrained_ckpt: checkpoint/m2d_sc_stage3_estimated_pretrainedsed_fusion.ckpt
```

### Required data

The referenced datamodule expects the same generated/metadata data paths as:

```text
config/separation/modified_deft_tse_lite_6s_online_teacher_uss_sc.yaml
```

including foreground/background/interference/room IR paths and validation metadata.

### Recommended checks

Before full training, verify that:

1. The USS checkpoint emits:

   ```py
   foreground_waveform
   tse_condition
   foreground_activity_logits
   class_logits
   silence_logits
   pred_doa_vector
   spatial_embedding / doa_vector
   ```

2. The SC checkpoint works on USS foreground estimates.
3. The config resolves correctly:

   ```bash
   PYTHONPATH=dcase2026_task4_baseline .venv/bin/python - <<'PY'
   from src.utils import parse_yaml, initialize_config
   cfg = parse_yaml('dcase2026_task4_baseline/config/separation/modified_deft_tse_lite_6s_online_teacher_two_stage_spatial_uss_sc.yaml')
   model = initialize_config(cfg['lightning_module']['args']['model'])
   print(type(model).__name__)
   PY
   ```

Expected:

```text
TwoStageRobustSpatialBridgeTSE
```

---

## 6. Training command

Run from the repository root:

```text
dcase2026baseline/
```

using the project virtual environment.

Example:

```bash
PYTHONPATH=dcase2026_task4_baseline .venv/bin/python -m src.train \
  --workspace workspace \
  --config_yaml dcase2026_task4_baseline/config/separation/modified_deft_tse_lite_6s_online_teacher_two_stage_spatial_uss_sc.yaml \
  --tqdm 60
```

Optional batch-size override:

```bash
PYTHONPATH=dcase2026_task4_baseline .venv/bin/python -m src.train \
  --workspace workspace \
  --config_yaml dcase2026_task4_baseline/config/separation/modified_deft_tse_lite_6s_online_teacher_two_stage_spatial_uss_sc.yaml \
  --batchsize 1 \
  --tqdm 60
```

Checkpoints will be saved under:

```text
workspace/modified_deft_tse_lite_6s_online_teacher_two_stage_spatial_uss_sc/checkpoints/
```

---

## 7. Training curriculum

The config enables:

```py
RobustTSEConditionCurriculum
```

The schedule starts with heavy condition dropout/noise:

```yaml
condition_dropout: 0.8
temporal_condition_dropout: 0.8
spatial_condition_dropout: 1.0
condition_noise_std: 0.05
spatial_condition_noise_std: 0.10
```

and ramps toward:

```yaml
condition_dropout: 0.3
temporal_condition_dropout: 0.4
spatial_condition_dropout: 0.5
condition_noise_std: 0.02
spatial_condition_noise_std: 0.05
```

Rationale:

- early epochs: learn robust `mixture + enrollment + label` separation
- later epochs: progressively learn to exploit `tse_condition`, activity, and spatial hints
- learned gates remain trainable and are not overwritten by the callback

---

## 8. Evaluation modes

There are two different evaluation modes. They answer different questions.

## 8.1 Standalone TSE evaluation

Use this only when the dataset provides TSE inputs directly:

```py
enrollment
label_vector
```

Command shape:

```bash
PYTHONPATH=dcase2026_task4_baseline .venv/bin/python -m src.evaluation.evaluate_stage \
  --config <tse_eval_or_training_config.yaml> \
  --stage tse \
  --checkpoint <two_stage_tse_checkpoint.ckpt> \
  --split val
```

Important: `--stage tse` does **not** run USS and SC. It cannot evaluate the final mixture-only system unless the dataset already includes estimated enrollments and labels.

If the dataset does not contain `enrollment` or `dry_sources`, the evaluator raises a clear error and suggests using `--stage s5`.

## 8.2 Full final mixture-only S5 evaluation

Use this for final system behavior:

```text
mixture → USS → SC → TSE
```

`evaluate_stage.py` now supports:

```bash
--stage s5
```

Command shape:

```bash
PYTHONPATH=dcase2026_task4_baseline .venv/bin/python -m src.evaluation.evaluate_stage \
  --config <s5_eval_config_using_two_stage_tse.yaml> \
  --stage s5 \
  --split val \
  --batchsize 1
```

The S5 model must expose:

```py
predict_label_separate(mixture)
```

which is true for:

```py
Kwon2025S5
Kwon2025TemporalS5
```

### S5 eval config

A ready-to-edit S5 validation config is provided at:

```text
src/evaluation/eval_configs/kwo2025_top1_like_lite_estimated_temporal_sc_two_stage_spatial_uss_conditioned_tse.yaml
```

It sets:

```yaml
tse_ckpt: checkpoint/modified_deft_tse_lite_6s_two_stage_spatial_uss_sc.ckpt
tse_uss_conditioning_enabled: true
tse_config:
  module: src.models.deft.two_stage_tse
  main: TwoStageRobustSpatialBridgeTSE
```

After training, copy or symlink the selected checkpoint to the `tse_ckpt` path above, or edit the config to point directly to the checkpoint under `workspace/.../checkpoints/`.

This mode uses only the mixture waveform externally. USS and SC run inside the S5 wrapper and forward the optional USS hints into TSE.

---

## 9. Post-work after training

After training finishes:

1. Select the best checkpoint from:

   ```text
   workspace/modified_deft_tse_lite_6s_online_teacher_two_stage_spatial_uss_sc/checkpoints/
   ```

2. Optionally copy or symlink it into `checkpoint/`, for example:

   ```text
   checkpoint/modified_deft_tse_lite_6s_two_stage_spatial_uss_sc.ckpt
   ```

3. Update an S5 evaluation config so:

   ```yaml
   tse_ckpt: checkpoint/modified_deft_tse_lite_6s_two_stage_spatial_uss_sc.ckpt
   tse_config:
     module: src.models.deft.two_stage_tse
     main: TwoStageRobustSpatialBridgeTSE
   tse_uss_conditioning_enabled: true
   ```

4. Run full `--stage s5` validation.

5. Compare against the previous TSE using:

   - CAPI-SDRi / SDR metrics
   - source label metrics
   - silence / leakage breakdown if using `--validation_breakdown`

Example:

```bash
PYTHONPATH=dcase2026_task4_baseline .venv/bin/python -m src.evaluation.evaluate_stage \
  --config dcase2026_task4_baseline/src/evaluation/eval_configs/kwo2025_top1_like_lite_estimated_temporal_sc_two_stage_spatial_uss_conditioned_tse.yaml \
  --stage s5 \
  --split val \
  --batchsize 1 \
  --validation_breakdown \
  --result_dir workspace/eval_results
```

---

## 10. What USS bridge outputs are used

The new TSE currently uses the compressed bridge representation:

```py
tse_condition
```

as `query_condition`.

It does not directly consume raw bridge outputs:

```py
foreground_embedding
foreground_audio_embedding
prototype_logits
object_embedding
object_audio_embedding
```

This is intentional for the first robust version. `tse_condition` is the USS bridge output explicitly designed for TSE handoff and already contains projected information from semantic embedding, audio embedding, class probabilities, and DoA.

Raw bridge features can be added later as an ablation if `tse_condition` is insufficient.

---

## 11. Suggested ablation order

To understand where improvements come from, evaluate in this order:

1. Core two-stage spatial TSE only:

   ```yaml
   query_condition_dim: 0
   spatial_condition_dim: 0
   temporal_conditioning_enabled: false
   ```

2. Add `tse_condition`:

   ```yaml
   query_condition_dim: 256
   ```

3. Add temporal activity:

   ```yaml
   temporal_conditioning_enabled: true
   ```

4. Add spatial hints:

   ```yaml
   spatial_condition_dim: 3
   ```

5. Tune gate/dropout/noise curriculum.

6. Only later consider raw bridge features:

   ```py
   foreground_embedding
   foreground_audio_embedding
   prototype_logits
   ```

---

## 12. Current caveats

- The config trains TSE from scratch by default (`pretrained_model_ckpt:` is empty), because the two-stage architecture is not weight-compatible with the old single-stage TSE.
- Standalone `--stage tse` is not final system evaluation; use `--stage s5` for mixture-only final behavior.
- USS auxiliary heads may be noisy; keep condition gates/dropout/noise enabled unless ablations show they are unnecessary.
- `used_spatial_vector` is intentionally not forwarded by default to avoid oracle-leakage ambiguity.

---

## 13. Quick file index

| Role | File |
|---|---|
| Training config | `config/separation/modified_deft_tse_lite_6s_online_teacher_two_stage_spatial_uss_sc.yaml` |
| S5 eval config | `src/evaluation/eval_configs/kwo2025_top1_like_lite_estimated_temporal_sc_two_stage_spatial_uss_conditioned_tse.yaml` |
| New TSE model | `src/models/deft/two_stage_tse.py` |
| Online teacher Lightning | `src/training/lightningmodule/online_teacher_tse.py` |
| Pipeline finetune Lightning | `src/training/lightningmodule/uss_sc_tse_pipeline_finetune.py` |
| Robust condition curriculum | `src/training/callbacks/robust_tse_condition_curriculum.py` |
| S5 inference | `src/models/s5/kwo2025.py` |
| Temporal S5 inference | `src/models/s5/kwo2025_temporal.py` |
| Stage evaluator | `src/evaluation/evaluate_stage.py` |
| USS provider | `src/models/deft/unified_uss.py` |
| USS bridge producer | `src/models/deft/modified_deft_semantic_bridge.py` |
| Loss | `src/training/loss/masked_snr.py` |
