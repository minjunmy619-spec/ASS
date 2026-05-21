# Pipeline Finetune Operation

## Goal

Add one general online fine-tuning wrapper for the deployment data flow:

1. `USS(mixture) -> estimated source slots`
2. `SC(estimated source slots) -> source labels / activity`
3. `TSE(mixture, USS slots, SC labels) -> refined source slots`
4. Optional final `SC(TSE refined slots) -> updated classifier loss`

The wrapper is opt-in and does not replace the existing SC-primary USS wrapper
or the existing online-teacher TSE wrapper.

## Added Wrapper

`src/training/lightningmodule/uss_sc_tse_pipeline_finetune.py`

Main class: `USSScTSEPipelineFinetuneLightning`

Modes:

- `target_stage: tse`
  - Freezes USS and SC.
  - Trains TSE from live USS estimates and SC predictions.
  - Keeps the behavior of `OnlineTeacherTSELightning`, but under the general
    pipeline wrapper.

- `target_stage: sc_after_tse`
  - Freezes USS and TSE.
  - Uses SC once as the teacher that supplies TSE query labels.
  - Runs frozen TSE.
  - Re-matches the final TSE output waveforms back to oracle sources before
    building SC labels. This keeps SC false negatives from becoming silence
    targets and avoids blindly trusting wrong SC query labels.
  - Trains SC on the final matched TSE output waveforms with oracle-aligned
    class targets and configurable active/silence sample weights.

## Added Configs

- `config/separation/modified_deft_pipeline_finetune_tse_uss_sc.yaml`
- `config/separation/modified_deft_pipeline_finetune_sc_after_tse.yaml`

Both configs use `from_training_config` references for pretrained component
definitions:

- USS model from `modified_deft_sc_primary_frozenuss_pretrainedsed_fusion.yaml`
- SC model/loss/optimizer from `../label/m2d_sc_stage3_estimated_pretrainedsed_fusion.yaml`
- TSE model/loss/optimizer from `modified_deft_tse_lite_6s_temporal.yaml`

This keeps the architecture and loss contracts tied to the normal pretraining
configs instead of duplicating them by hand.

## Pretraining Dependencies

The pipeline wrapper assumes these checkpoints already exist or are replaced by
equivalent trained variants:

- USS checkpoint: `checkpoint/modified_deft_uss_lite_6s_unified_all_features.ckpt`
- SC checkpoint: `checkpoint/m2d_sc_stage3_estimated_pretrainedsed_fusion.ckpt`
- Initial TSE checkpoint: `checkpoint/modified_deft_tse_lite_6s_temporal.ckpt`

The SC-after-TSE stage should use the TSE checkpoint produced by the TSE
pipeline fine-tune stage. Update this value before running:

```yaml
pretrained_model_ckpt: path/to/modified_deft_pipeline_finetune_tse_uss_sc.ckpt
```

## Mode: TSE Fine-Tune

Config:

```text
config/separation/modified_deft_pipeline_finetune_tse_uss_sc.yaml
```

Data flow:

```text
batch["mixture"]
  -> frozen USS
  -> uss_out["foreground_waveform"] as TSE enrollment
  -> frozen SC predict(enrollment) for TSE query labels
  -> trainable TSE
  -> masked_snr loss against oracle sources aligned to USS slots
```

Ownership:

| Component | State | Why |
| --- | --- | --- |
| USS | frozen/eval | Provides the deployment-time source-slot distribution. |
| SC | frozen/eval | Provides the deployment-time label/query distribution. |
| TSE | trainable | Learns to refine USS slots using live SC labels. |

Required TSE input keys:

- `mixture`: mixture waveform from the batch.
- `enrollment`: `uss_out["foreground_waveform"]`, resized if needed to match
  mixture length.
- `label_vector`: SC-predicted one-hot source labels by slot, unless
  `label_source: oracle` is explicitly set.
- `query_condition`: optional USS condition tensor from `tse_condition`,
  `query_condition`, `bridge_condition`, `proposal_condition`, or composed USS
  outputs.
- `temporal_conditioning`: optional SC/USS activity conditioning.

TSE loss target keys:

- `waveform`: oracle source waveform aligned into USS slot order.
- `label_vector`: oracle label vector aligned into USS slot order.
- `active_mask`: active matched slots after optional SC-active/class gating.
- `span_sec`: optional active span targets for temporal activity loss.

Train command:

```bash
cd /home/cmj/works/ASS/dcase2026baseline/dcase2026_task4_baseline

PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=/tmp/numba_cache \
  .venv/bin/python -m src.train \
  -c config/separation/modified_deft_pipeline_finetune_tse_uss_sc.yaml \
  --workspace workspace/pipeline_finetune
```

## Mode: SC After TSE Fine-Tune

Config:

```text
config/separation/modified_deft_pipeline_finetune_sc_after_tse.yaml
```

Data flow:

```text
batch["mixture"]
  -> frozen USS
  -> SC predict(enrollment) for frozen TSE query labels
  -> frozen TSE
  -> final TSE output waveform
  -> fresh waveform-to-oracle re-match
  -> trainable SC with m2d_sc_arcface loss
```

Ownership:

| Component | State | Why |
| --- | --- | --- |
| USS | frozen/eval | Keeps source-slot proposals identical to deployment. |
| TSE | frozen/eval | Produces the refined-waveform distribution SC will see later. |
| SC | trainable | Adapts the classifier to final TSE-refined outputs. |

The final SC target is not copied from the TSE target mask. TSE supervision can
be SC-gated because bad query rows are not useful for training frozen TSE, but
the final SC update must be built from the final TSE waveform itself. Therefore
the wrapper re-runs waveform-to-oracle matching on the final TSE output before
creating:

- `class_index`
- `is_silence`
- `sample_weight`
- optional `span_sec`
- `current_epoch`
- `is_training`

This is the important safety rule:

```text
SC teacher mask for TSE supervision != final SC target mask
```

It prevents:

- SC false negatives from becoming silence targets.
- Wrong SC query labels from being treated as hard-correct final SC labels.
- Low-quality final TSE outputs from updating SC as if they were clean matches.

Train command:

```bash
cd /home/cmj/works/ASS/dcase2026baseline/dcase2026_task4_baseline

PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=/tmp/numba_cache \
  .venv/bin/python -m src.train \
  -c config/separation/modified_deft_pipeline_finetune_sc_after_tse.yaml \
  --workspace workspace/pipeline_finetune
```

## Configuration Knobs

- `target_stage`: `tse` or `sc_after_tse`.
- `match_metric`: currently expected to use `sa_sdr` for oracle matching.
- `min_match_score`: filters weak waveform matches.
- `min_estimate_energy_db`: filters silent/near-silent estimates.
- `require_sc_active_for_loss`: gates TSE supervision to rows SC believes are
  active. In `sc_after_tse`, this does not erase the final SC target because
  final SC labels are re-matched from final TSE waveforms.
- `require_sc_class_match_for_loss`: optional stricter TSE-supervision gate.
- `label_source`: normally `sc` to match deployment-time TSE query labels.
- `query_condition_enabled`: enables USS-derived TSE query/proposal features.
- `temporal_conditioning_source`: `auto`, `sc`, `uss`, `tse`, or equivalent
  supported source in the wrapper.
- `tse_refinement_passes`: `1` or `2`; pass 2 uses the pass-1 TSE output as
  enrollment and re-runs SC for second-pass labels.
- `sc_active_sample_weight`: SC loss weight for final matched active TSE rows.
- `sc_silence_sample_weight`: SC loss weight for final unmatched/silence rows.

## Diagnostics To Watch

- `teacher_matched_slots`: USS slots matched to oracle sources before TSE.
- `teacher_tse_supervised_slots`: rows retained for TSE supervision after SC
  gating.
- `teacher_sc_active_rate`: fraction of USS slots SC predicted active.
- `teacher_sc_class_match_rate`: class agreement between SC predictions and
  oracle-aligned slot labels.
- `teacher_match_score`: match quality before TSE.
- `pipeline_sc_final_matched_slots`: final TSE outputs matched to oracle
  sources.
- `pipeline_sc_final_match_score`: final TSE-to-oracle match quality.
- `pipeline_sc_top1`: SC top-1 on active final matched TSE outputs.
- `pipeline_sc_sample_weight_mean`: effective SC sample weight after active and
  silence rows are mixed.

Good SC-after-TSE training normally needs enough final matched TSE rows. If
`pipeline_sc_final_matched_slots` or `pipeline_sc_final_match_score` is low, do
not trust the SC update yet; first inspect USS/TSE outputs or tune the earlier
stages.

## Evaluation After Each Stage

After TSE pipeline fine-tuning, evaluate the full S5 pipeline with the new TSE
checkpoint before starting SC-after-TSE. Watch:

- CAPI-SDRi / SDRi summary.
- Source classification F1.
- Active-source top-1 accuracy.
- Silence accuracy and false-positive rows.
- Duplicate-class recall if using duplicate-aware eval configs.

After SC-after-TSE, run the same S5 evaluation again. The target improvement is
not only lower SC loss; it should improve final S5 source labeling without
hurting waveform quality or increasing silence false positives.

## Follow-Up Fixes

- TSE checkpoint loading now extracts exact model keys before accepting
  unrelated wrapper keys, so a pipeline checkpoint with `model.*`,
  `uss_model.*`, and `sc_model.*` can initialize the frozen TSE stage.
- `sc_after_tse` separates the mask used to decide which rows are useful for
  frozen TSE supervision from the final SC loss target. The final SC target is
  built from a fresh waveform-to-oracle match on the final TSE output.

## Validation

Focused regression command:

```bash
PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=/tmp/numba_cache \
  .venv/bin/python -m pytest \
  tests/test_task4_2026_losses.py \
  tests/test_uss_sc_joint_model_parallel.py
```

Result:

```text
47 passed, 3 warnings
```

The remaining warning is the existing pytest cache permission warning under
`/home/cmj/works/ASS/.pytest_cache`, plus upstream deprecation warnings.
