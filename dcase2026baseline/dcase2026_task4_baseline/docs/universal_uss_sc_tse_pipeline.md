# Universal USS -> SC -> TSE Pipeline Runbook

This document is the current single runbook for the universal source-separation
pipeline:

```text
dynamic FOA mixture
  -> UnifiedModifiedDeFTUSS
  -> M2DPretrainedSEDFusionClassifier
  -> OnlineTeacherTSELightning / ModifiedDeFTTSEMemoryEfficientTemporal
  -> Kwon2025TemporalS5 final evaluation
```

All commands below are intended to be copied from this file and run from the
repository root.

## 1. File Map

| Stage | Config | Model definition | Lightning module | Loss | Dataset | Evaluation |
| --- | --- | --- | --- | --- | --- | --- |
| Universal USS | `config/separation/modified_deft_uss_lite_6s_unified_all_features.yaml` | `src/models/deft/unified_uss.py` | `src/training/lightningmodule/uss_bridge.py` | `src/training/loss/uss_bridge_loss.py` | `src/datamodules/uss_dataset.py` over `src/datamodules/dataset.py` | `src/evaluation/evaluate_stage.py --stage uss` |
| SC clean bootstrap | `config/label/m2d_sc_stage1_pretrainedsed_fusion.yaml` | `src/models/m2dat/m2d_sc.py` (`M2DPretrainedSEDFusionClassifier`) | `src/training/lightningmodule/single_label_classification.py` | `src/training/loss/m2d_sc_arcface.py` | `src/datamodules/source_classifier_dataset.py` over `src/datamodules/dataset.py` | `src/evaluation/evaluate_stage.py --stage sc` |
| SC estimated-source fine-tune | `config/label/m2d_sc_stage3_estimated_pretrainedsed_fusion.yaml` | `src/models/m2dat/m2d_sc.py` (`M2DPretrainedSEDFusionClassifier`) | `src/training/lightningmodule/single_label_classification.py` | `src/training/loss/m2d_sc_arcface.py` | `src/datamodules/source_classifier_dataset.py` (`EstimatedSourceClassifierDataset`) over cached USS estimates | `src/evaluation/evaluate_stage.py --stage sc` |
| Optional on-the-fly USS -> SC joint fine-tune | `config/separation/modified_deft_uss_sc_joint_universal_pretrainedsed_fusion.yaml` | `src/models/deft/unified_uss.py` and `src/models/m2dat/m2d_sc.py` | `src/training/lightningmodule/uss_sc_joint_model_parallel.py` | `src/training/loss/uss_bridge_loss.py` and `src/training/loss/m2d_sc_arcface.py` | `src/datamodules/uss_dataset.py` over dynamic `DatasetS3` scenes | stage-evaluate USS/SC with `evaluate_stage.py` and the joint checkpoint |
| TSE bootstrap | `config/separation/modified_deft_tse_lite_6s_temporal.yaml` | `src/models/deft/modified_deft.py` (`ModifiedDeFTTSEMemoryEfficientTemporal`) | `src/training/lightningmodule/tse.py` | `src/training/loss/masked_snr.py` | `src/datamodules/tse_dataset.py` (`TSEDataset`) | `src/evaluation/evaluate_stage.py --stage tse` |
| Online-teacher TSE | `config/separation/modified_deft_tse_lite_6s_online_teacher_uss_sc.yaml`; opt-in two-pass: `config/separation/modified_deft_tse_lite_6s_online_teacher_uss_sc_2pass.yaml` | `src/models/deft/modified_deft.py` (`ModifiedDeFTTSEMemoryEfficientTemporal`) | `src/training/lightningmodule/online_teacher_tse.py` | `src/training/loss/masked_snr.py` | `src/datamodules/tse_dataset.py` (`OnlineTeacherTSEDataset`) | full S5 evaluation |
| Final S5 | two-pass current: `src/evaluation/eval_configs/kwo2025_top1_like_lite_estimated_temporal_sc_uss_conditioned_tse.yaml`; one-pass comparison: `src/evaluation/eval_configs/kwo2025_top1_like_lite_estimated_temporal_sc_uss_conditioned_tse_1pass.yaml`; two-pass-trained comparison: `src/evaluation/eval_configs/kwo2025_top1_like_lite_estimated_temporal_sc_uss_conditioned_tse_2pass_trained.yaml` | `src/models/s5/kwo2025_temporal.py`, `src/models/s5/kwo2025.py` | none | metric-only | `src/datamodules/dataset.py` (`DatasetS3`, waveform test split) | `src/evaluation/evaluate.py` |

Supporting USS feature modules:

```text
src/models/deft/modified_deft_bf16_safe.py
src/models/deft/foa_spatial_features.py
src/models/deft/uss_count_head.py
src/models/deft/spatial_heads.py
src/models/deft/modified_deft_semantic_bridge.py
```

## 2. Runtime Contracts

### USS Contract

`UnifiedModifiedDeFTUSS` consumes:

```text
mixture: [B, 4, T]
optional spatial_vector: [B, 3, 3]
```

**Special Emphasis:** `spatial_vector` may be used during early USS training as
a scheduled teacher-forcing signal, but the training schedule must end in a
predicted-only phase. Oracle DoA/source-position metadata is allowed as a loss
target; it must not remain a persistent model input for the final USS training
distribution, because final official evaluation provides only the mixture.

It always emits the baseline USS keys:

```text
foreground_waveform: [B, 3, 1, T]
interference_waveform: [B, 2, 1, T]
noise_waveform: [B, 1, 1, T]
class_logits: [B, 3, 18]
silence_logits: [B, 3]
```

The universal config also enables these opt-in outputs:

```text
count_logits
foreground_activity_logits
interference_activity_logits
noise_activity_logits
spatial_embedding
doa_vector
residual_waveform
pred_doa_vector
used_spatial_vector
object_embedding
foreground_embedding
prototype_logits
tse_condition: [B, 3, 256]
```

`tse_condition` is the live USS -> TSE bridge tensor used by the online-teacher
TSE and final S5 configs.

### SC Contract

`M2DPretrainedSEDFusionClassifier.predict()` consumes mono slot waveforms:

```text
waveform: [B * S, T]
```

It returns:

```text
label_vector: [B * S, 18]      # silence-gated one-hot class vector
raw_label_vector: [B * S, 18]  # ungated top-1 class vector
class_indices: [B * S]
probabilities: [B * S]
energy: [B * S]
silence: [B * S]
```

Final S5 reshapes this back to `[B, S, ...]`.

### Online-Teacher TSE Contract

`OnlineTeacherTSELightning` receives dynamic synthesized batches from
`OnlineTeacherTSEDataset`. The training dataset is not an exported cache; it is
`DatasetS3(mode=generate)` wrapped by `OnlineTeacherTSEDataset`, so each epoch
sees dynamically synthesized mixtures from `spatial_audio_synthesizer`. The
validation dataset is `DatasetS3(mode=metadata)` over
`data/dev_set/metadata/valid.json`, also wrapped by `OnlineTeacherTSEDataset`.

```text
mixture: [B, 4, T]
waveform: [B, 3, 1, T]        # oracle clean targets
label_vector: [B, 3, 18]      # oracle labels
active_mask: [B, 3]
span_sec: optional [B, 3, 2]
```

Inside the Lightning module:

```text
frozen USS(mixture) -> foreground_waveform + tse_condition
frozen SC(foreground_waveform) -> label_vector
TSE input -> mixture, enrollment=USS foreground_waveform,
             label_vector=SC labels, query_condition=USS tse_condition
loss target -> oracle sources aligned into USS estimate-slot order
```

Only the TSE parameters are optimized in this stage. USS and SC are loaded from
checkpoints, set to eval/frozen mode every step, and used only as online
teachers. This is not joint USS/SC/TSE training.

The exported estimated-audio cache from section 6 is for adapting SC to the USS
output distribution. The final online-teacher TSE does not read
`workspace/sc_finetune_universal*/estimate_target`; it recreates the USS
estimated enrollments online from each dynamic mixture.

### Final S5 Contract

`Kwon2025TemporalS5.predict_label_separate()` performs the live final pipeline:

```text
mixture
  -> USS foreground slots + activity + tse_condition
  -> SC labels
  -> TSE pass 1 with query_condition and temporal_conditioning
  -> SC labels
  -> optional TSE pass 2 with pass-1 TSE output as enrollment
  -> optional SC labels after pass 2
  -> final labels, probabilities, waveforms
```

The final S5 wrapper has an explicit comparison flag:

```yaml
tse_refinement_passes: 1  # stop after pass 1
tse_refinement_passes: 2  # current two-pass refinement behavior
```

Use `tse_refinement_passes: 1` when evaluating the standard one-pass
online-teacher TSE checkpoint against its trained distribution. Use
`tse_refinement_passes: 2` only for the current two-pass S5 behavior or for a
TSE checkpoint trained with the opt-in unrolled two-pass recipe.

The final S5 config uses:

```text
USS checkpoint: checkpoint/modified_deft_uss_lite_6s_unified_all_features.ckpt
SC checkpoint: checkpoint/m2d_sc_stage3_estimated_pretrainedsed_fusion.ckpt
TSE checkpoint: checkpoint/modified_deft_tse_lite_6s_online_teacher_unified_uss_sc.ckpt
```

## 3. One-Time Setup

```bash
cd /home/cmj/works/ASS/dcase2026baseline/dcase2026_task4_baseline
source .venv/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

mkdir -p checkpoint workspace/universal workspace/universal/stage_eval workspace/universal/final_eval
```

Install or verify the external model assets:

```bash
cd /home/cmj/works/ASS/dcase2026baseline/dcase2026_task4_baseline
source .venv/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

mkdir -p external checkpoint/pretrainedsed

test -d src/modules/spatial_audio_synthesizer || {
  echo "Missing src/modules/spatial_audio_synthesizer. Install SpAudSyn before training generated scenes."
  exit 1
}

test -f checkpoint/m2d_as_vit_base-80x1001p16x16p32k-240413_AS-FT_enconly/weights_ep69it3124-0.47998.pth || {
  wget -P checkpoint https://github.com/nttcslab/m2d/releases/download/v0.3.0/m2d_as_vit_base-80x1001p16x16p32k-240413_AS-FT_enconly.zip
  unzip -n checkpoint/m2d_as_vit_base-80x1001p16x16p32k-240413_AS-FT_enconly.zip -d checkpoint
}

test -d external/PretrainedSED || git clone https://github.com/fschmid56/PretrainedSED.git external/PretrainedSED

for name in BEATs_strong_1.pt ATST-F_strong_1.pt fpasst_strong_1.pt; do
  test -f "checkpoint/pretrainedsed/${name}" || \
    wget -nc -P checkpoint/pretrainedsed "https://github.com/fschmid56/PretrainedSED/releases/download/v0.0.1/${name}"
done
```

Quick code/config smoke before training:

```bash
cd /home/cmj/works/ASS/dcase2026baseline/dcase2026_task4_baseline
source .venv/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python -m py_compile \
  src/models/deft/unified_uss.py \
  src/models/m2dat/m2d_sc.py \
  src/training/lightningmodule/uss_sc_joint_model_parallel.py \
  src/training/lightningmodule/online_teacher_tse.py \
  src/evaluation/evaluate_stage.py \
  src/evaluation/evaluate.py \
  src/models/s5/kwo2025.py \
  src/models/s5/kwo2025_temporal.py

NUMBA_CACHE_DIR=/tmp/numba_cache python -m pytest -q \
  tests/test_unified_uss.py \
  tests/test_task4_2026_losses.py \
  tests/test_eval_conditioning.py \
  tests/test_uss_sc_joint_model_parallel.py \
  tests/test_spatial_conditioning_curriculum.py \
  -o cache_dir=/tmp/pytest_cache_dcase2026
```

## 4. Train USS

Train the universal USS:

The universal config includes `SpatialConditioningCurriculum`, which anneals
semantic-bridge spatial conditioning from oracle-mixed warmup to predicted-only
conditioning:

```text
epochs 0-24:    predicted_spatial_prob=0.25, spatial_mix_fallback_prob=0.05
epochs 25-173:  linear anneal toward predicted_spatial_prob=1.0,
                spatial_mix_fallback_prob=0.0
epochs 174-249: predicted-only spatial conditioning
```

**Special Emphasis:** keep `foreground_doa`/source position available in the
target dict for DoA supervision, but do not rely on it as a lasting model input.
The final predicted-only USS phase is what makes the training input distribution
match final S5 and official evaluation.

If you change `max_epochs`, also change `warmup_epochs` and `anneal_epochs` so
the run still has a non-trivial predicted-only tail before selecting the final
USS checkpoint.

```bash
cd /home/cmj/works/ASS/dcase2026baseline/dcase2026_task4_baseline
source .venv/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python -m src.train \
  -c config/separation/modified_deft_uss_lite_6s_unified_all_features.yaml \
  -w workspace/universal \
  --tqdm 60
```

Promote the checkpoint to the path used by TSE and final S5 configs:

```bash
cd /home/cmj/works/ASS/dcase2026baseline/dcase2026_task4_baseline
mkdir -p checkpoint
cp workspace/universal/modified_deft_uss_lite_6s_unified_all_features/checkpoints/last.ckpt \
  checkpoint/modified_deft_uss_lite_6s_unified_all_features.ckpt
```

Stage-evaluate USS and write predicted source waveforms:

```bash
cd /home/cmj/works/ASS/dcase2026baseline/dcase2026_task4_baseline
source .venv/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python -m src.evaluation.evaluate_stage \
  -c config/separation/modified_deft_uss_lite_6s_unified_all_features.yaml \
  --stage uss \
  --checkpoint checkpoint/modified_deft_uss_lite_6s_unified_all_features.ckpt \
  --split val \
  --batchsize 1 \
  --num_workers 0 \
  --result_dir workspace/universal/stage_eval \
  --waveform_output_dir workspace/universal/stage_waveforms \
  --compare_assignment \
  --validation_breakdown
```

## 5. Train SC

Train the PretrainedSED-fusion source classifier:

```bash
cd /home/cmj/works/ASS/dcase2026baseline/dcase2026_task4_baseline
source .venv/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python -m src.train \
  -c config/label/m2d_sc_stage1_pretrainedsed_fusion.yaml \
  -w workspace/universal \
  --tqdm 60
```

Promote the SC checkpoint:

```bash
cd /home/cmj/works/ASS/dcase2026baseline/dcase2026_task4_baseline
mkdir -p checkpoint
cp workspace/universal/m2d_sc_stage1_pretrainedsed_fusion/checkpoints/last.ckpt \
  checkpoint/m2d_sc_stage1_pretrainedsed_fusion.ckpt
```

Stage-evaluate SC in the same gated mode used by final S5:

```bash
cd /home/cmj/works/ASS/dcase2026baseline/dcase2026_task4_baseline
source .venv/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python -m src.evaluation.evaluate_stage \
  -c config/label/m2d_sc_stage1_pretrainedsed_fusion.yaml \
  --stage sc \
  --checkpoint checkpoint/m2d_sc_stage1_pretrainedsed_fusion.ckpt \
  --split val \
  --batchsize 8 \
  --num_workers 0 \
  --sc_prediction_mode gated \
  --result_dir workspace/universal/stage_eval
```

Optional raw top-1 SC diagnostic:

```bash
python -m src.evaluation.evaluate_stage \
  -c config/label/m2d_sc_stage1_pretrainedsed_fusion.yaml \
  --stage sc \
  --checkpoint checkpoint/m2d_sc_stage1_pretrainedsed_fusion.ckpt \
  --split val \
  --batchsize 8 \
  --num_workers 0 \
  --sc_prediction_mode raw \
  --result_dir workspace/universal/stage_eval
```

## 6. Export USS Estimated Audio And Fine-Tune SC

The universal pipeline should adapt the SC model to the waveform distribution
created by the trained USS. This step exports cached USS foreground estimates
and then fine-tunes the PretrainedSED-fusion SC on those estimated waveforms.
This cache is an SC training artifact, not the final TSE training dataset.

Create the train cache from the generated train split. This mode uses USS
foreground estimates as audio and assigns oracle labels by PIT matching against
the clean reference sources:

```bash
cd /home/cmj/works/ASS/dcase2026baseline/dcase2026_task4_baseline
source .venv/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python -m src.evaluation.export_sc_finetune_cache \
  -c config/separation/modified_deft_uss_lite_6s_unified_all_features.yaml \
  --checkpoint checkpoint/modified_deft_uss_lite_6s_unified_all_features.ckpt \
  --mode uss_oracle \
  --split train \
  --output_root workspace/sc_finetune_universal \
  --manifest_path workspace/sc_finetune_universal/match_manifest.csv \
  --min_match_score -10.0 \
  --min_match_margin -1000000000.0 \
  --min_energy_db -60.0 \
  --clean_match_score 0.0 \
  --clean_match_margin 2.0 \
  --batchsize 1 \
  --num_workers 0 \
  --overwrite
```

Create the validation cache from the metadata validation split:

```bash
cd /home/cmj/works/ASS/dcase2026baseline/dcase2026_task4_baseline
source .venv/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python -m src.evaluation.export_sc_finetune_cache \
  -c config/separation/modified_deft_uss_lite_6s_unified_all_features.yaml \
  --checkpoint checkpoint/modified_deft_uss_lite_6s_unified_all_features.ckpt \
  --mode uss_oracle \
  --split val \
  --output_root workspace/sc_finetune_universal_valid \
  --manifest_path workspace/sc_finetune_universal_valid/match_manifest.csv \
  --min_match_score -10.0 \
  --min_match_margin -1000000000.0 \
  --min_energy_db -60.0 \
  --clean_match_score 0.0 \
  --clean_match_margin 2.0 \
  --batchsize 1 \
  --num_workers 0 \
  --overwrite
```

`uss_oracle` now filters PIT matches before writing `estimate_target/*.wav`.
By default it writes only clean matches. Bad matches and uncertain matches stay
as silence slots in `EstimatedSourceClassifierDataset`, and their quality rows
are preserved in `match_manifest.csv`. Add `--save_uncertain` only for an
explicit noisy-label experiment.

The cache layout consumed by `EstimatedSourceClassifierDataset` is:

```text
workspace/sc_finetune_universal/soundscape/
workspace/sc_finetune_universal/oracle_target/
workspace/sc_finetune_universal/estimate_target/
workspace/sc_finetune_universal_valid/soundscape/
workspace/sc_finetune_universal_valid/oracle_target/
workspace/sc_finetune_universal_valid/estimate_target/
```

Fine-tune SC on the cached Universal USS estimates:

```bash
cd /home/cmj/works/ASS/dcase2026baseline/dcase2026_task4_baseline
source .venv/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python -m src.train \
  -c config/label/m2d_sc_stage3_estimated_pretrainedsed_fusion.yaml \
  -w workspace/universal \
  --tqdm 60
```

Promote the estimated-source SC checkpoint:

```bash
cd /home/cmj/works/ASS/dcase2026baseline/dcase2026_task4_baseline
mkdir -p checkpoint
cp workspace/universal/m2d_sc_stage3_estimated_pretrainedsed_fusion/checkpoints/last.ckpt \
  checkpoint/m2d_sc_stage3_estimated_pretrainedsed_fusion.ckpt
```

Stage-evaluate the estimated-source adapted SC:

```bash
cd /home/cmj/works/ASS/dcase2026baseline/dcase2026_task4_baseline
source .venv/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python -m src.evaluation.evaluate_stage \
  -c config/label/m2d_sc_stage3_estimated_pretrainedsed_fusion.yaml \
  --stage sc \
  --checkpoint checkpoint/m2d_sc_stage3_estimated_pretrainedsed_fusion.ckpt \
  --split val \
  --batchsize 8 \
  --num_workers 0 \
  --sc_prediction_mode gated \
  --result_dir workspace/universal/stage_eval
```

For pseudo-label adaptation instead of oracle-label adaptation, replace
`--mode uss_oracle` with `--mode uss_pseudo`. Use that only when the USS class
head is already reliable enough to provide labels.

### Optional On-The-Fly USS -> SC Joint Fine-Tune

Use this only when you want SC fine-tuning to see live Universal USS estimates
during training instead of reading `workspace/sc_finetune_universal` wav
caches. This path is opt-in and does not replace the cached stage3 SC recipe
above.

The joint config runs this loop every step:

```text
dynamic mixture
  -> Universal USS
  -> PIT match USS estimates to oracle dry sources
  -> quality gate by SDR, margin, and estimated-source energy
  -> PretrainedSED-fusion SC loss on clean matched estimates
  -> update USS every step and SC every sc_update_every steps
```

The default universal joint config has:

```yaml
freeze_sc: false
sc_update_every: 4
use_uncertain_matches: false
min_match_score: -10.0
clean_match_score: 0.0
clean_match_margin: 2.0
min_energy_db: -60.0
```

So SC is actually fine-tuned online. Set `freeze_sc: true` only for the older
frozen-SC-teacher behavior where the SC loss updates USS but not SC.

Required checkpoints before this optional path:

```text
checkpoint/modified_deft_uss_lite_6s_unified_all_features.ckpt
checkpoint/m2d_sc_stage1_pretrainedsed_fusion.ckpt
```

Run the joint fine-tune with two visible GPUs:

```bash
cd /home/cmj/works/ASS/dcase2026baseline/dcase2026_task4_baseline
source .venv/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

CUDA_VISIBLE_DEVICES=0,1 python -m src.train \
  -c config/separation/modified_deft_uss_sc_joint_universal_pretrainedsed_fusion.yaml \
  -w workspace/universal \
  --tqdm 60
```

Promote the joint checkpoint to a separate opt-in path:

```bash
cd /home/cmj/works/ASS/dcase2026baseline/dcase2026_task4_baseline
mkdir -p checkpoint
cp workspace/universal/modified_deft_uss_sc_joint_universal_pretrainedsed_fusion/checkpoints/last.ckpt \
  checkpoint/modified_deft_uss_sc_joint_universal_pretrainedsed_fusion.ckpt
```

That checkpoint contains both `uss_model.*` and `sc_model.*` weights. The stage
and S5 loaders can extract the correct prefix when this same checkpoint is
passed as a USS checkpoint or an SC checkpoint.

Stage-evaluate the USS part of the joint checkpoint:

```bash
cd /home/cmj/works/ASS/dcase2026baseline/dcase2026_task4_baseline
source .venv/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python -m src.evaluation.evaluate_stage \
  -c src/evaluation/eval_configs/kwo2025_top1_like_lite_estimated_temporal_sc_uss_conditioned_tse.yaml \
  --stage uss \
  --checkpoint checkpoint/modified_deft_uss_sc_joint_universal_pretrainedsed_fusion.ckpt \
  --batchsize 1 \
  --num_workers 0 \
  --result_dir workspace/universal/stage_eval_joint_uss_sc \
  --waveform_output_dir workspace/universal/stage_waveforms_joint_uss_sc \
  --compare_assignment \
  --validation_breakdown
```

Stage-evaluate the SC part of the joint checkpoint:

```bash
cd /home/cmj/works/ASS/dcase2026baseline/dcase2026_task4_baseline
source .venv/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python -m src.evaluation.evaluate_stage \
  -c src/evaluation/eval_configs/kwo2025_top1_like_lite_estimated_temporal_sc_uss_conditioned_tse.yaml \
  --stage sc \
  --checkpoint checkpoint/modified_deft_uss_sc_joint_universal_pretrainedsed_fusion.ckpt \
  --batchsize 8 \
  --num_workers 0 \
  --sc_prediction_mode gated \
  --result_dir workspace/universal/stage_eval_joint_uss_sc
```

To make the downstream online-teacher TSE or final S5 use this optional joint
checkpoint, set both USS and SC checkpoint fields to:

```text
checkpoint/modified_deft_uss_sc_joint_universal_pretrainedsed_fusion.ckpt
```

Keep it as a separate comparison branch unless you intentionally want to replace
the cached `checkpoint/modified_deft_uss_lite_6s_unified_all_features.ckpt` and
`checkpoint/m2d_sc_stage3_estimated_pretrainedsed_fusion.ckpt` promotion paths.

## 7. Train TSE Bootstrap

The online-teacher TSE config initializes from the standard temporal TSE
checkpoint. Train that bootstrap first if it does not already exist.

```bash
cd /home/cmj/works/ASS/dcase2026baseline/dcase2026_task4_baseline
source .venv/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python -m src.train \
  -c config/separation/modified_deft_tse_lite_6s_temporal.yaml \
  -w workspace/universal \
  --tqdm 60
```

Promote the bootstrap checkpoint:

```bash
cd /home/cmj/works/ASS/dcase2026baseline/dcase2026_task4_baseline
mkdir -p checkpoint
cp workspace/universal/modified_deft_tse_lite_6s_temporal/checkpoints/last.ckpt \
  checkpoint/modified_deft_tse_lite_6s_temporal.ckpt
```

Stage-evaluate the bootstrap TSE with oracle enrollments:

```bash
cd /home/cmj/works/ASS/dcase2026baseline/dcase2026_task4_baseline
source .venv/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python -m src.evaluation.evaluate_stage \
  -c config/separation/modified_deft_tse_lite_6s_temporal.yaml \
  --stage tse \
  --checkpoint checkpoint/modified_deft_tse_lite_6s_temporal.ckpt \
  --split val \
  --batchsize 1 \
  --num_workers 0 \
  --result_dir workspace/universal/stage_eval \
  --waveform_output_dir workspace/universal/stage_waveforms \
  --compare_assignment \
  --validation_breakdown
```

## 8. Train Online-Teacher TSE

This is the main TSE for the universal pipeline. It freezes the trained USS and
SC checkpoints, runs them online on dynamically synthesized mixtures, and trains
only the TSE model. The train split uses `OnlineTeacherTSEDataset` with
`DatasetS3(mode=generate)`, so the mixture is synthesized on the fly; the
validation split uses `OnlineTeacherTSEDataset` with
`DatasetS3(mode=metadata)`.

The runtime data flow during each training step is:

```text
dynamic mixture
  -> frozen UnifiedModifiedDeFTUSS
  -> USS foreground_waveform used as TSE enrollment
  -> USS tse_condition used as TSE query_condition
  -> frozen M2DPretrainedSEDFusionClassifier predicts TSE label_vector
  -> trainable ModifiedDeFTTSEMemoryEfficientTemporal
  -> masked_snr loss against oracle dry sources aligned to USS estimate slots
```

Do not point this config at `workspace/sc_finetune_universal` caches. Those
exported wavs are consumed by `m2d_sc_stage3_estimated_pretrainedsed_fusion.yaml`
to produce the SC checkpoint used here.

Required checkpoints before this step:

```text
checkpoint/modified_deft_uss_lite_6s_unified_all_features.ckpt
checkpoint/m2d_sc_stage3_estimated_pretrainedsed_fusion.ckpt
checkpoint/modified_deft_tse_lite_6s_temporal.ckpt
```

Run the training:

```bash
cd /home/cmj/works/ASS/dcase2026baseline/dcase2026_task4_baseline
source .venv/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python -m src.train \
  -c config/separation/modified_deft_tse_lite_6s_online_teacher_uss_sc.yaml \
  -w workspace/universal \
  --tqdm 60
```

This standard recipe is one-pass training:

```yaml
tse_refinement_passes: 1
```

It trains TSE with USS estimated enrollments, which matches final S5 only when
the final S5 comparison config also uses `tse_refinement_passes: 1`.

Promote the online-teacher TSE checkpoint to the path used by final S5:

```bash
cd /home/cmj/works/ASS/dcase2026baseline/dcase2026_task4_baseline
mkdir -p checkpoint
cp workspace/universal/modified_deft_tse_lite_6s_online_teacher_uss_sc/checkpoints/last.ckpt \
  checkpoint/modified_deft_tse_lite_6s_online_teacher_unified_uss_sc.ckpt
```

### Optional Two-Pass Online-Teacher TSE

Use this only when you want the TSE to train on the same second-pass enrollment
distribution used by two-pass final S5. The second pass is unrolled inside
`OnlineTeacherTSELightning`: pass 1 uses frozen USS estimates as enrollment,
then pass 2 uses the detached pass-1 TSE output as enrollment and re-runs frozen
SC for the pass-2 query label. Oracle dry sources remain loss targets only.

```bash
cd /home/cmj/works/ASS/dcase2026baseline/dcase2026_task4_baseline
source .venv/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python -m src.train \
  -c config/separation/modified_deft_tse_lite_6s_online_teacher_uss_sc_2pass.yaml \
  -w workspace/universal \
  --tqdm 60
```

Promote the two-pass-trained checkpoint to its separate comparison path:

```bash
cd /home/cmj/works/ASS/dcase2026baseline/dcase2026_task4_baseline
mkdir -p checkpoint
cp workspace/universal/modified_deft_tse_lite_6s_online_teacher_uss_sc_2pass/checkpoints/last.ckpt \
  checkpoint/modified_deft_tse_lite_6s_online_teacher_unified_uss_sc_2pass.ckpt
```

Do not use `evaluate_stage.py --stage tse` with the final S5 config for this
online-teacher TSE. That isolated stage does not run live USS, so it cannot
create the required `query_condition`. Use full S5 evaluation below.

## 9. Final S5 Inference And Evaluation

Audit required runtime assets:

```bash
cd /home/cmj/works/ASS/dcase2026baseline/dcase2026_task4_baseline

for path in \
  checkpoint/modified_deft_uss_lite_6s_unified_all_features.ckpt \
  checkpoint/m2d_sc_stage3_estimated_pretrainedsed_fusion.ckpt \
  checkpoint/modified_deft_tse_lite_6s_online_teacher_unified_uss_sc.ckpt \
  external/PretrainedSED \
  checkpoint/pretrainedsed
do
  test -e "$path" && echo "OK      $path" || echo "MISSING $path"
done
```

Run full final S5 evaluation with the current two-pass S5 behavior and write
predicted source waveforms:

```bash
cd /home/cmj/works/ASS/dcase2026baseline/dcase2026_task4_baseline
source .venv/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python -m src.evaluation.evaluate \
  -c src/evaluation/eval_configs/kwo2025_top1_like_lite_estimated_temporal_sc_uss_conditioned_tse.yaml \
  --batchsize 1 \
  --result_dir workspace/universal/final_eval \
  --waveform_output_dir workspace/universal/final_waveforms \
  --compare_assignment \
  --validation_breakdown
```

Compare one-pass final S5 against the standard one-pass online-teacher TSE
checkpoint:

```bash
cd /home/cmj/works/ASS/dcase2026baseline/dcase2026_task4_baseline
source .venv/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python -m src.evaluation.evaluate \
  -c src/evaluation/eval_configs/kwo2025_top1_like_lite_estimated_temporal_sc_uss_conditioned_tse_1pass.yaml \
  --batchsize 1 \
  --result_dir workspace/universal/final_eval_1pass \
  --waveform_output_dir workspace/universal/final_waveforms_1pass \
  --compare_assignment \
  --validation_breakdown
```

Compare a two-pass-trained TSE checkpoint with two-pass final S5:

```bash
cd /home/cmj/works/ASS/dcase2026baseline/dcase2026_task4_baseline
source .venv/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python -m src.evaluation.evaluate \
  -c src/evaluation/eval_configs/kwo2025_top1_like_lite_estimated_temporal_sc_uss_conditioned_tse_2pass_trained.yaml \
  --batchsize 1 \
  --result_dir workspace/universal/final_eval_2pass_trained \
  --waveform_output_dir workspace/universal/final_waveforms_2pass_trained \
  --compare_assignment \
  --validation_breakdown
```

Final S5 inference uses the same online handoff as online-teacher training:

```text
test mixture
  -> Universal USS estimates source slots, activity, and tse_condition
  -> estimated-source-adapted SC predicts labels for those USS slots
  -> online-teacher TSE checkpoint refines the USS estimates
  -> temporal S5 either stops after pass 1 or repeats SC/TSE refinement
  -> final labels/waveforms
```

No exported estimated-audio cache is read during final S5 inference.

Read the final summary:

```bash
cat workspace/universal/final_eval/kwo2025_top1_like_lite_estimated_temporal_sc_uss_conditioned_tse_summary.json
```

The final waveform outputs are written under:

```text
workspace/universal/final_waveforms/kwo2025_top1_like_lite_estimated_temporal_sc_uss_conditioned_tse/
```

## 10. Final S5 Stage Diagnostics

Use these commands when the full S5 score is bad and you need to isolate the
failure.

Evaluate only the final USS from the S5 config:

```bash
cd /home/cmj/works/ASS/dcase2026baseline/dcase2026_task4_baseline
source .venv/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python -m src.evaluation.evaluate_stage \
  -c src/evaluation/eval_configs/kwo2025_top1_like_lite_estimated_temporal_sc_uss_conditioned_tse.yaml \
  --stage uss \
  --batchsize 1 \
  --num_workers 0 \
  --result_dir workspace/universal/stage_eval \
  --waveform_output_dir workspace/universal/stage_waveforms \
  --compare_assignment \
  --validation_breakdown
```

Evaluate only the final SC from the S5 config:

```bash
cd /home/cmj/works/ASS/dcase2026baseline/dcase2026_task4_baseline
source .venv/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python -m src.evaluation.evaluate_stage \
  -c src/evaluation/eval_configs/kwo2025_top1_like_lite_estimated_temporal_sc_uss_conditioned_tse.yaml \
  --stage sc \
  --batchsize 8 \
  --num_workers 0 \
  --sc_prediction_mode gated \
  --result_dir workspace/universal/stage_eval
```

For final TSE, use the full S5 command in section 9. The final TSE depends on
live USS `tse_condition`; isolated TSE stage evaluation cannot reproduce that
handoff unless a bridge-aware dataset provides saved query conditions.

## 11. Common Checks

Check that the training outputs exist:

```bash
find workspace/universal -maxdepth 5 -path '*/checkpoints/*.ckpt' | sort
```

Check that promoted checkpoints exist:

```bash
ls -lh \
  checkpoint/modified_deft_uss_lite_6s_unified_all_features.ckpt \
  checkpoint/m2d_sc_stage1_pretrainedsed_fusion.ckpt \
  checkpoint/m2d_sc_stage3_estimated_pretrainedsed_fusion.ckpt \
  checkpoint/modified_deft_tse_lite_6s_temporal.ckpt \
  checkpoint/modified_deft_tse_lite_6s_online_teacher_unified_uss_sc.ckpt \
  checkpoint/modified_deft_tse_lite_6s_online_teacher_unified_uss_sc_2pass.ckpt
```

Check the optional on-the-fly joint USS/SC checkpoint when using that branch:

```bash
ls -lh checkpoint/modified_deft_uss_sc_joint_universal_pretrainedsed_fusion.ckpt
```

Run the compatibility tests after editing the pipeline:

```bash
cd /home/cmj/works/ASS/dcase2026baseline/dcase2026_task4_baseline
source .venv/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

NUMBA_CACHE_DIR=/tmp/numba_cache python -m pytest -q tests/test_unified_uss.py tests/test_task4_2026_losses.py tests/test_eval_conditioning.py tests/test_uss_sc_joint_model_parallel.py tests/test_spatial_conditioning_curriculum.py
```

## 12. Current Promotion Order

Use this order when starting from scratch:

```text
1. Train universal USS.
2. Promote universal USS checkpoint.
3. Train PretrainedSED-fusion SC on clean sources.
4. Promote clean SC checkpoint.
5. Export Universal USS estimated-source train/validation caches.
6. Fine-tune SC on cached Universal USS estimates.
7. Promote estimated-source SC checkpoint.
7a. Optional: train and promote the on-the-fly Universal USS -> SC joint checkpoint as a separate comparison branch.
8. Train temporal TSE bootstrap.
9. Promote temporal TSE bootstrap checkpoint.
10. Train one-pass online-teacher TSE with frozen universal USS and estimated-source SC.
11. Promote online-teacher TSE checkpoint.
12. Run one-pass final S5 comparison with `tse_refinement_passes: 1`.
13. Run current two-pass final S5 comparison with `tse_refinement_passes: 2`.
14. Optional: train and promote the unrolled two-pass online-teacher TSE.
15. Optional: run the two-pass-trained final S5 comparison.
16. Use stage diagnostics only when the final score needs debugging.
```

This order matters because the online-teacher TSE learns against the distribution
created by the frozen USS and SC teachers that final S5 will also use.
The optional joint USS/SC checkpoint is not promoted over the default cached
stage3 SC branch unless you explicitly choose that comparison and point
downstream USS/SC checkpoint fields at the joint checkpoint.
For two-pass final S5, use the opt-in unrolled two-pass TSE recipe before
treating two-pass results as the aligned training/evaluation distribution.
