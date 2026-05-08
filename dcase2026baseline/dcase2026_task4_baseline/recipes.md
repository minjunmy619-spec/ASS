# Recipes

This file collects experimental recipes added on top of the DCASE 2026 Task 4
baseline. The official-style baseline remains available in
`config/label/m2dat_*.yaml` and `config/separation/resunetk_capisdr.yaml`.

## Quick Recommendation

For new DeFT experiments, start with the memory-efficient path:

```bash
cd ./dcase2026_task4_baseline

# stage 0: memory-efficient universal source separation
python -m src.train -c config/separation/modified_deft_uss_lite_6s.yaml -w workspace/separation

# stage 1: single-label classifier, ArcFace stage
python -m src.train -c config/label/m2d_sc_stage1_strong.yaml -w workspace/label

# stage 2: single-label classifier, energy/silence stage
python -m src.train -c config/label/m2d_sc_stage2_strong.yaml -w workspace/label

# stage 3: memory-efficient target sound extractor
python -m src.train -c config/separation/modified_deft_tse_lite_6s.yaml -w workspace/separation

# optional stage 4: fine-tune SC on cached USS/TSE estimated sources
python -m src.train -c config/label/m2d_sc_stage3_estimated_strong.yaml -w workspace/label
```

For a stronger classification branch, replace the `_strong` SC stages with the
BEATs-fusion configs:

```bash
python -m src.train -c config/label/m2d_sc_stage1_beats_fusion.yaml -w workspace/label
python -m src.train -c config/label/m2d_sc_stage2_beats_fusion.yaml -w workspace/label
python -m src.train -c config/label/m2d_sc_stage3_estimated_beats_fusion.yaml -w workspace/label
```

If you want a more frame-oriented auxiliary encoder, use the analogous fPaSST
configs:

```bash
python -m src.train -c config/label/m2d_sc_stage1_fpasst_fusion.yaml -w workspace/label
python -m src.train -c config/label/m2d_sc_stage2_fpasst_fusion.yaml -w workspace/label
python -m src.train -c config/label/m2d_sc_stage3_estimated_fpasst_fusion.yaml -w workspace/label
```

For the official PretrainedSED multi-branch variant, start with the clean
stage-1 sibling:

```bash
python -m src.train -c config/label/m2d_sc_stage1_pretrainedsed_fusion.yaml -w workspace/label
```

This variant uses frozen BEATs, ATST-F, and fPaSST branches from the
PretrainedSED v0.0.1 release, projects every branch to the M2D embedding size,
and defaults to `fusion_strategy: weighted_avg`. It is currently a clean-source
stage-1 recipe; add stage-2 and stage-3 siblings before treating it as a final
S5 SC branch.

For pseudo-label or noisy estimated-source caches, keep the earlier stages the
same and switch only the optional stage-3 SC fine-tuning config to a robust
sibling:

```bash
python -m src.train -c config/label/m2d_sc_stage3_estimated_strong_robust.yaml -w workspace/label
python -m src.train -c config/label/m2d_sc_stage3_estimated_temporal_strong_robust.yaml -w workspace/label
python -m src.train -c config/label/m2d_sc_stage3_estimated_beats_fusion_robust.yaml -w workspace/label
python -m src.train -c config/label/m2d_sc_stage3_estimated_fpasst_fusion_robust.yaml -w workspace/label
```

The robust siblings use label smoothing plus soft loss truncation: after a
1-epoch warmup, the highest-loss active estimated-source labels in each batch are
downweighted rather than trusted fully. This is intended for label noise from
swapped, impure, or pseudo-labeled separator outputs.

Then place or symlink the trained checkpoints where the evaluation config
expects them:

```text
checkpoint/modified_deft_uss_lite_6s.ckpt
checkpoint/m2d_sc_stage3_estimated_strong.ckpt
checkpoint/m2d_sc_stage3_estimated_beats_fusion.ckpt
checkpoint/m2d_sc_stage3_estimated_fpasst_fusion.ckpt
checkpoint/modified_deft_tse_lite_6s.ckpt
```

Run 10s evaluation with:

```bash
python -m src.evaluation.evaluate \
  -c src/evaluation/eval_configs/kwo2025_top1_like_lite_estimated_sc.yaml \
  --result_dir workspace/evaluation
```

To isolate one stage while keeping the same evaluation dataset and checkpoint
paths from an S5 eval config, use `evaluate_stage.py`:

```bash
# USS only: mixture -> USS foreground slots, scored against oracle sources.
python -m src.evaluation.evaluate_stage \
  -c src/evaluation/eval_configs/kwo2025_top1_like_lite_estimated_sc.yaml \
  --stage uss \
  --result_dir workspace/evaluation_stage \
  --waveform_output_dir workspace/evaluation_stage_waveforms

# SC only: oracle source waveforms -> source labels.
python -m src.evaluation.evaluate_stage \
  -c src/evaluation/eval_configs/kwo2025_top1_like_lite_estimated_sc.yaml \
  --stage sc \
  --result_dir workspace/evaluation_stage

# TSE only: oracle enrollment and oracle labels -> extracted targets.
python -m src.evaluation.evaluate_stage \
  -c src/evaluation/eval_configs/kwo2025_top1_like_lite_estimated_sc.yaml \
  --stage tse \
  --result_dir workspace/evaluation_stage \
  --waveform_output_dir workspace/evaluation_stage_waveforms
```

USS and TSE report CAPI-SDRi plus label metrics. SC reports mixture/source
label metrics plus source-level top-1, active-source top-1, and silence
accuracy. Each run writes both per-soundscape results and a summary JSON when
`--result_dir` is provided.

Use `src/evaluation/eval_configs/kwo2025_top1_like_lite_beats_fusion_sc.yaml`
for the BEATs-fusion classifier checkpoint, or
`src/evaluation/eval_configs/kwo2025_top1_like_lite_fpasst_fusion_sc.yaml`
for the fPaSST-fusion checkpoint.

To test duplicate same-class recall without changing the default evaluation
path, run the sibling config that matches the classifier branch:

```bash
python -m src.evaluation.evaluate \
  -c src/evaluation/eval_configs/kwo2025_top1_like_lite_estimated_sc_duplicate_recall.yaml \
  --result_dir workspace/evaluation
```

Other duplicate-recall siblings are:

```text
src/evaluation/eval_configs/kwo2025_top1_like_lite_strong_sc_duplicate_recall.yaml
src/evaluation/eval_configs/kwo2025_top1_like_lite_beats_fusion_sc_duplicate_recall.yaml
src/evaluation/eval_configs/kwo2025_top1_like_lite_fpasst_fusion_sc_duplicate_recall.yaml
```

This mode recovers a silenced source slot only when its raw classifier top class
matches an already active class in the same mixture and it passes probability
and waveform-energy gates.

This path is the safest default because both DeFT-heavy stages use local
attention instead of global full-clip attention.

## Model Picker

| Goal | Config | Model class | Use when |
| --- | --- | --- | --- |
| Official-style AT + SS baseline | `config/label/m2dat_*.yaml`, `config/separation/resunetk_capisdr.yaml` | `M2dAt`, `ResUNet30` | You want the stable baseline path. |
| Strong spatial USS, full global DeFT | `config/separation/modified_deft_uss.yaml` | `ModifiedDeFTUSSSpatial` | You want maximum context and have enough GPU memory. |
| Shorter USS training, chunked eval | `config/separation/modified_deft_uss_5s.yaml` | `ChunkedModifiedDeFTUSSSpatial` | You want the same global DeFT block but need shorter 5s training clips. |
| Memory-efficient spatial USS | `config/separation/modified_deft_uss_lite_6s.yaml` | `ModifiedDeFTUSSMemoryEfficient` | Recommended new USS path; local time attention and grouped frequency attention. |
| KAIST-like oracle TSE, full global DeFT | `config/separation/modified_deft_tse.yaml` | `ModifiedDeFTTSE` | You want the original oracle-enrollment TSE structure and can afford memory. |
| Memory-efficient oracle TSE | `config/separation/modified_deft_tse_lite_6s.yaml` | `ModifiedDeFTTSEMemoryEfficient` | Recommended new TSE path; fixes the TSE global-attention memory issue. |
| USS-conditioned temporal TSE | `config/separation/modified_deft_tse_lite_*_temporal_estimated_enrollment_uss_conditioned.yaml` | `ModifiedDeFTTSEMemoryEfficientTemporal` | You want TSE to receive USS proposal/bridge context, not only waveform enrollment plus class label. |
| Strong M2D source classifier | `config/label/m2d_sc_stage*_strong.yaml` | `M2DSingleClassifierStrong` | Recommended SC path; attentive statistics pooling and multi-crop prediction. |
| Estimated-source SC fine-tuning | `config/label/m2d_sc_stage3_estimated_strong.yaml` | `EstimatedSourceClassifierDataset` + `M2DSingleClassifierStrong` | Optional final SC adaptation on cached USS/TSE outputs. |
| Robust estimated-source SC fine-tuning | `config/label/m2d_sc_stage3_estimated_strong_robust.yaml` | `EstimatedSourceClassifierDataset` + robust `m2d_sc_arcface` | Opt-in soft truncation for noisy estimated-source labels. |
| Temporal strong M2D source classifier | `config/label/m2d_sc_stage*_temporal_strong.yaml` | `M2DSingleClassifierTemporalStrong` | Use when you want span-supervised activity outputs for temporal S5. |
| Robust temporal estimated-source SC | `config/label/m2d_sc_stage3_estimated_temporal_strong_robust.yaml` | `EstimatedSourceClassifierDataset` + robust/activity `m2d_sc_arcface` | Temporal strong branch with the same noisy-label handling. |
| M2D + BEATs fusion SC | `config/label/m2d_sc_stage*_beats_fusion.yaml` | `M2DPretrainedFusionClassifier` | Best first pretrained upgrade for classification; keep M2D and add frozen BEATs semantics. |
| Estimated-source BEATs fusion SC | `config/label/m2d_sc_stage3_estimated_beats_fusion.yaml` | `EstimatedSourceClassifierDataset` + `M2DPretrainedFusionClassifier` | Strongest current SC recipe; adapt the fused classifier to USS/TSE artifacts. |
| Robust estimated-source BEATs fusion SC | `config/label/m2d_sc_stage3_estimated_beats_fusion_robust.yaml` | `EstimatedSourceClassifierDataset` + robust `m2d_sc_arcface` | Same BEATs fusion branch with noisy-label soft truncation. |
| M2D + fPaSST fusion SC | `config/label/m2d_sc_stage*_fpasst_fusion.yaml` | `M2DPretrainedFusionClassifier` | Strong frame-oriented pretrained upgrade; useful when partial event timing cues matter. |
| Estimated-source fPaSST fusion SC | `config/label/m2d_sc_stage3_estimated_fpasst_fusion.yaml` | `EstimatedSourceClassifierDataset` + `M2DPretrainedFusionClassifier` | Good ablation against BEATs fusion on distorted USS/TSE outputs. |
| Robust estimated-source fPaSST fusion SC | `config/label/m2d_sc_stage3_estimated_fpasst_fusion_robust.yaml` | `EstimatedSourceClassifierDataset` + robust `m2d_sc_arcface` | Same fPaSST fusion branch with noisy-label soft truncation. |
| Multi-branch PretrainedSED fusion SC | `config/label/m2d_sc_stage1_pretrainedsed_fusion.yaml` | `M2DPretrainedSEDFusionClassifier` | Clean-source stage-1 experiment that fuses M2D with official BEATs, ATST-F, and fPaSST release branches. |
| USS count gate | `config/separation/modified_deft_uss_lite_6s_count_head*.yaml` | `ModifiedDeFTUSSMemoryEfficientCountHead` | Predict foreground count and gate excess slots in S5. |
| USS FOA spatial features | `config/separation/modified_deft_uss_lite_6s_foa_spatial_features*.yaml` | `ModifiedDeFTUSSMemoryEfficientSpatialFeatures*` | Add FOA log-magnitude, AIV, and IPD features before the DeFT separator. |
| USS residual slots | `config/separation/modified_deft_uss_lite_6s_residual.yaml` | `ModifiedDeFTUSSMemoryEfficientResidual` | Keep extra non-foreground residual/trash slots separate from foreground predictions. |
| USS spatial-head CAPI candidate | `config/separation/modified_deft_uss_lite_6s_spatial_capi_strong.yaml` | `ModifiedDeFTUSSMemoryEfficientSpatialFeaturesCountSpatialHead` | Current high-feature USS candidate with FOA features, count head, DoA/spatial heads, and CAPI-oriented losses. |
| AudioSet-Strong source-pool curriculum | See `docs/audioset_strong_augmentation.md` | `DatasetS3(mode=generate)` with `spatial_sound_scene_sources` | Mix official and AudioSet-Strong dry-source pools with explicit ratios instead of folder merges. |
| Stage diagnostics and calibration | `src/evaluation/evaluate_stage.py`, `src/evaluation/calibrate_sc_energy_thresholds.py` | `StageEvaluator`, calibration CLI | Isolate USS/SC/TSE quality and calibrate per-class SC silence thresholds before final S5 comparisons. |
| One-stage label-query separator | `config/separation/deft_tse_like.yaml` | `DeFTTSELikeSpatial` | You want a simpler label-query separator instead of `USS -> SC -> TSE`. |

## What Changed

### Spatial Masking

`ModifiedDeFTUSSSpatial` and `DeFTTSELikeSpatial` replace the old 2-output
complex mask:

```text
[complex_mask_real, complex_mask_imag]
```

with a 3-component phase-aware mask for every target and every mixture channel:

```text
[magnitude_mask, phase_mask_real, phase_mask_imag]
```

The reconstruction path is:

1. Predict one magnitude mask and one phase-rotation vector per input channel.
2. Apply `sigmoid` to magnitude masks.
3. Normalize the phase real/imag pair with `magphase()`.
4. Reconstruct complex estimates for all mixture channels.
5. Use a learned `4 -> 1` projection for mono output.
6. Run ISTFT and return the same output keys as before.

This follows the useful part of the ResUNet final stage where `K = 3` means
magnitude plus phase components, while preserving spatial information until the
last projection.

### Memory-Efficient DeFT Blocks

The original DeFT block applies global attention over the full STFT time axis
and full frequency axis. For 10s audio at 32 kHz with `window_size=1024` and
`hop_size=320`, the model sees about:

```text
T ~= 1001 frames
F = 513 bins
```

That makes memory dominated by attention activations, not parameters.

`MemoryEfficientDeFTBlock` keeps the same `[B, C, T, F]` interface but changes
the attention pattern:

- time attention is local within `time_window_size` STFT frames
- frequency attention is local within `freq_group_size` bins
- odd-numbered blocks use half-window shifts so neighboring windows/groups can
  exchange information across depth

Default lite settings:

```yaml
time_window_size: 128
freq_group_size: 64
shift_windows: true
```

Increasing either window can improve context but raises memory quadratically
along that axis.

### Strong Source Classifier

The original `M2DSingleClassifier` is intentionally compact: it averages all M2D
time tokens, projects once, and applies an ArcFace head. That is a reasonable
baseline, but it is probably too simple for a top-ranking system because source
estimates can be partial, distorted, or contain residual interference.

`M2DSingleClassifierStrong` keeps the M2D backbone and output contract, but
changes the classifier head:

- attentive statistics pooling over M2D time tokens
- concatenated mean, max, attention mean, and attention std features
- MLP projection before ArcFace
- dropout and weight decay for better regularization
- optional multi-crop prediction for long source estimates

The output keys are unchanged:

```text
embedding, logits, plain_logits, energy
```

so the existing `SingleLabelClassificationLightning`, `m2d_sc_arcface` loss,
and S5 inference code can use it directly.

The M2D-SC fine-tuning selector now follows the same convention as the
multi-output M2D tagger: `2_blocks` really unfreezes the last two transformer
blocks plus the output head. Earlier classifier runs from before this fix may
have trained mostly the output head instead of the intended backbone blocks.

### Pretrained Fusion Source Classifier

`M2DPretrainedFusionClassifier` is the one-auxiliary-branch upgrade when the
classification branch becomes the bottleneck:

```text
[M2D attentive embedding, frozen pretrained audio embedding] -> fusion MLP -> ArcFace / CE head
```

It intentionally does not replace M2D. M2D stays as the task-specific branch
that has already been used by the baseline, while the auxiliary branch adds a
broader semantic prior from a pretrained audio model. The output keys stay
unchanged:

```text
embedding, logits, plain_logits, energy
```

so the same lightning module, `m2d_sc_arcface` loss, and S5 inference code can
use it directly.

Recommended one-branch model order:

| Priority | Aux model | Why |
| ---: | --- | --- |
| 1 | BEATs | Best first upgrade. It is strong for AudioSet-like event semantics and tends to be useful for noisy separated sources. |
| 2 | fPaSST | Strong next option when frame-level event timing cues matter more than broad semantic priors. |
| 3 | ATST / ATST-F | Attractive when temporal event cues matter. For official PretrainedSED ATST-F, use the multi-branch class below instead of the older single-branch wrapper. |
| 4 | AST | Not provided as a distinct official PretrainedSED branch. In `M2DPretrainedSEDFusionClassifier`, `AST` is accepted only as a compatibility alias to `fpasst`. |

Current backend support:

- `aux_backend: official_beats`: local Microsoft BEATs code plus a local `.pt`
  checkpoint. This is the default in the BEATs-fusion configs.
- `aux_backend: official_fpasst`: local PretrainedSED code plus a local
  `fPaSST` checkpoint. This is the default in the fPaSST-fusion configs.
- `aux_backend: transformers`: Hugging Face `AutoModel` or
  `AutoModelForAudioClassification`, useful for AST and any ATST/BEATs wrapper
  published with `trust_remote_code`.
- `aux_backend: torchscript`: local TorchScript encoder that returns
  `embedding`, `features`, or `logits`.
- `aux_backend: identity`: tiny smoke-test-only fallback; do not use for real
  experiments.

`M2DPretrainedSEDFusionClassifier` is the official PretrainedSED multi-branch
path. It supports:

```yaml
pretrainedsed_repo_root: external/PretrainedSED
pretrainedsed_checkpoint_dir: checkpoint/pretrainedsed
pretrainedsed_models: [BEATs, ATST-F, fpasst]
pretrainedsed_checkpoint_variant: strong_1
freeze_pretrainedsed: true
fusion_strategy: weighted_avg
```

The class imports the official PretrainedSED wrappers, loads v0.0.1 release
assets such as `BEATs_strong_1.pt`, `ATST-F_strong_1.pt`, and
`fpasst_strong_1.pt`, removes prediction heads for feature-only mode, and
preserves the same SC `forward()` and `predict()` contracts used by S5. Fusion
strategies are `weighted_avg`, `concat_head`, and `late_fusion`.

The BEATs config defaults use frozen AudioSet fine-tuned outputs as the
auxiliary semantic vector:

```yaml
aux_model: beats
aux_backend: official_beats
aux_weight: checkpoint/beats/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
beats_source_dir: external/BEATs
aux_embedding_dim: 527
beats_use_finetuned_logits: true
freeze_aux: true
fusion_mode: concat_mlp
```

If you prefer hidden BEATs representations instead of AudioSet logits, switch to
a pretrained BEATs checkpoint and set:

```yaml
aux_weight: checkpoint/beats/BEATs_iter3_plus_AS2M.pt
aux_embedding_dim: 768
beats_use_finetuned_logits: false
```

For a distinct AST checkpoint outside PretrainedSED, the older
`M2DPretrainedFusionClassifier` can still use Hugging Face:

```yaml
aux_model: ast
aux_backend: transformers
aux_weight: MIT/ast-finetuned-audioset-10-10-0.4593
aux_embedding_dim: 768
aux_input_mode: feature_extractor
aux_use_logits: false
aux_sample_rate: 16000
```

If you want AST AudioSet logits as the semantic vector, set
`aux_use_logits: true` and `aux_embedding_dim: 527`.

For ATST-F, the safest path is to wrap your ATST-F checkpoint as a TorchScript
encoder or expose it through a local Python package, then use:

```yaml
aux_model: atst
aux_backend: torchscript
aux_weight: checkpoint/atst/atst_f_encoder.ts
aux_embedding_dim: <your_encoder_dim>
```

For fPaSST through the public PretrainedSED implementation, use this shape:

```yaml
aux_model: fpasst
aux_backend: official_fpasst
aux_weight: checkpoint/fpasst/fpasst_im.pt
fpasst_source_dir: external/PretrainedSED
aux_embedding_dim: 768
aux_sample_rate: 16000
fpasst_seq_len: 250
fpasst_embed_dim: 768
freeze_aux: true
```

This backend loads the frame-level fPaSST wrapper, runs its mel frontend on the
source waveform, returns the frame sequence, and then mean-pools it into the
fusion head. Compared with BEATs, it is a little more biased toward temporal
event structure and a little less toward broad AudioSet semantics.

Before submitting a system, re-check the current Task 4 external-resource rules.
This repo now makes the pretrained branch configurable, but challenge
eligibility still depends on the official task policy and resource list.

Reference implementations/docs:

- BEATs official code and checkpoints:
  <https://github.com/microsoft/unilm/tree/master/beats>
- PretrainedSED fPaSST wrapper and checkpoints:
  <https://github.com/fschmid56/PretrainedSED>
- PaSST official pretrained models:
  <https://github.com/kkoutini/PaSST>
- Hugging Face AST model docs:
  <https://huggingface.co/docs/transformers/model_doc/audio-spectrogram-transformer>
- ATST-SED / ATST-F reference implementation:
  <https://github.com/Audio-WestlakeU/ATST-SED>

### Chunked Evaluation

The short/lite configs train on 5s or 6s synthesized clips, but evaluation
datasets contain fixed 10s waveforms. The chunked model variants keep
evaluation at 10s by splitting long inputs internally and overlap-adding the
output waveform back to the original length.

Current chunk defaults:

| Config | Train duration | Eval chunk | Eval hop |
| --- | ---: | ---: | ---: |
| `modified_deft_uss_5s.yaml` | 5.0s | 6.0s | 5.0s |
| `modified_deft_uss_lite_6s.yaml` | 6.0s | 10.0s | 8.0s |
| `modified_deft_tse_lite_6s.yaml` | 6.0s | 10.0s | 8.0s |

With the lite models, 10s eval chunks are usually more realistic because the
attention block itself is memory-bounded.

## Training Guides

### Recommended Memory-Efficient Multi-Stage System

Train these stages:

```bash
cd ./dcase2026_task4_baseline

python -m src.train -c config/separation/modified_deft_uss_lite_6s.yaml -w workspace/separation
python -m src.train -c config/label/m2d_sc_stage1_strong.yaml -w workspace/label
python -m src.train -c config/label/m2d_sc_stage2_strong.yaml -w workspace/label
python -m src.train -c config/separation/modified_deft_tse_lite_6s.yaml -w workspace/separation
```

The two lite DeFT stages synthesize 6s mixtures on the fly using SpAudSyn:

```yaml
spatial_sound_scene:
  duration: 6.0
  max_event_dur: 6.0
```

They use `bf16-mixed` by default. If your GPU does not support bf16 well, change
the trainer precision to `16-mixed` or `32-true` in the YAML. `32-true` is
safer numerically but much heavier.

After the bootstrap S5 system can export estimated slots, fine-tune TSE on the
same enrollment distribution it will see at inference:

```bash
python -m src.evaluation.export_sc_finetune_cache \
  -c src/evaluation/eval_configs/kwo2025_top1_like_lite_strong_sc.yaml \
  --output_root workspace/sc_finetune \
  --mode pseudo_s5 \
  --batchsize 1

python -m src.train \
  -c config/separation/modified_deft_tse_lite_6s_estimated_enrollment.yaml \
  -w workspace/separation
```

This fine-tune uses `workspace/sc_finetune/estimate_target` as TSE enrollment,
`workspace/sc_finetune/oracle_target` as the target waveform, and random 6s
crops so the training length remains aligned with the lite TSE recipe. Active
oracle targets without an estimated enrollment are masked out by default, so the
model is not trained to hallucinate missing USS slots from class labels alone.
The config loads model weights from `checkpoint/modified_deft_tse_lite_6s.ckpt`
through `pretrained_model_ckpt`; do not use `train.py -r` for this fine-tune
unless you intentionally want to resume the full trainer state.
For final 10s alignment before the official-style S5 evaluation, copy the best
6s estimated-enrollment checkpoint to
`checkpoint/modified_deft_tse_lite_6s_estimated_enrollment.ckpt`, then run:

```bash
python -m src.train \
  -c config/separation/modified_deft_tse_lite_10s_estimated_enrollment.yaml \
  -w workspace/separation
```

This final alignment config keeps the full cached 10s waveforms
(`crop_seconds: null`), uses `batch_size: 1`, a lower learning rate, and a short
10-epoch schedule. The validation dataloader remains the existing oracle
validation sanity check; use full S5 evaluation as the real promotion gate for
the aligned TSE checkpoint.
For the temporal TSE variant, use
`config/separation/modified_deft_tse_lite_6s_temporal_estimated_enrollment.yaml`
after placing the temporal bootstrap weights at
`checkpoint/modified_deft_tse_lite_6s_temporal.ckpt`, then use
`config/separation/modified_deft_tse_lite_10s_temporal_estimated_enrollment.yaml`
after placing the temporal 6s estimated-enrollment checkpoint at
`checkpoint/modified_deft_tse_lite_6s_temporal_estimated_enrollment.ckpt`.

If the USS branch emits proposal features, or if you want S5 to synthesize a
compact proposal vector from USS class/count/spatial/activity outputs, use the
USS-conditioned temporal TSE siblings:

```bash
python -m src.train \
  -c config/separation/modified_deft_tse_lite_6s_temporal_estimated_enrollment_uss_conditioned.yaml \
  -w workspace/separation

python -m src.train \
  -c config/separation/modified_deft_tse_lite_10s_temporal_estimated_enrollment_uss_conditioned.yaml \
  -w workspace/separation
```

These configs add `query_condition_dim: 256` to
`ModifiedDeFTTSEMemoryEfficientTemporal`. The TSE model accepts
`query_condition`, `tse_condition`, `bridge_condition`, or
`proposal_condition`; missing features are allowed so old cache exports still
run. During full S5 inference, enable `tse_uss_conditioning_enabled: true` so
USS proposal cues are forwarded into stage-2 and stage-3 TSE:

```bash
python -m src.evaluation.evaluate \
  -c src/evaluation/eval_configs/kwo2025_top1_like_lite_estimated_temporal_sc_uss_conditioned_tse.yaml \
  --result_dir workspace/evaluation
```

When `export_sc_finetune_cache.py --mode pseudo_s5` is run with a conditioned
S5 config, it writes the usual `estimate_target/` waveforms and also saves the
matching per-soundscape proposal tensors under
`workspace/sc_finetune/uss_bridge_features/`. The conditioned TSE configs read
those files through `bridge_feature_dir`, keeping the training cache aligned
with the live S5 handoff.

### BEATs + M2D Fusion Classifier

Use this when the strong M2D classifier is stable and you want a higher-rank
classification branch:

```bash
python -m src.train -c config/label/m2d_sc_stage1_beats_fusion.yaml -w workspace/label
python -m src.train -c config/label/m2d_sc_stage2_beats_fusion.yaml -w workspace/label
```

Then create the estimated-source cache with your current S5 system and run the
adaptation stage:

```bash
python -m src.evaluation.export_sc_finetune_cache \
  -c src/evaluation/eval_configs/kwo2025_top1_like_lite_strong_sc.yaml \
  --output_dir workspace/sc_finetune \
  --mode pseudo_s5

python -m src.train -c config/label/m2d_sc_stage3_estimated_beats_fusion.yaml -w workspace/label
```

Evaluate the fused SC checkpoint with:

```bash
python -m src.evaluation.evaluate \
  -c src/evaluation/eval_configs/kwo2025_top1_like_lite_beats_fusion_sc.yaml \
  --result_dir workspace/evaluation
```

Expected local assets:

```text
external/BEATs/BEATs.py
checkpoint/beats/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
checkpoint/m2d_as_vit_base-80x1001p16x16p32k-240413_AS-FT_enconly/weights_ep69it3124-0.47998.pth
```

`external/` and `*.pt` are git-ignored so local pretrained code/checkpoints do
not get committed accidentally.

Training advice:

- Keep `freeze_aux: true` first. Let the M2D tail and fusion head adapt before
  trying to fine-tune the auxiliary model.
- Start with `batch_size: 24` or lower. BEATs adds memory and CPU/GPU work even
  when frozen.
- Use `fusion_mode: concat_mlp` for the first ablation. Try `gated_mlp` if the
  auxiliary model helps some classes but hurts others.
- Compare against `M2DSingleClassifierStrong` on the same cached estimated
  sources before deciding whether fusion is worth the extra inference cost.

### fPaSST + M2D Fusion Classifier

Use this when you want a stronger auxiliary branch with more explicit framewise
event structure:

```bash
python -m src.train -c config/label/m2d_sc_stage1_fpasst_fusion.yaml -w workspace/label
python -m src.train -c config/label/m2d_sc_stage2_fpasst_fusion.yaml -w workspace/label
```

Then create the estimated-source cache and adapt on separator outputs:

```bash
python -m src.evaluation.export_sc_finetune_cache \
  -c src/evaluation/eval_configs/kwo2025_top1_like_lite_strong_sc.yaml \
  --output_dir workspace/sc_finetune \
  --mode pseudo_s5

python -m src.train -c config/label/m2d_sc_stage3_estimated_fpasst_fusion.yaml -w workspace/label
```

Evaluate with:

```bash
python -m src.evaluation.evaluate \
  -c src/evaluation/eval_configs/kwo2025_top1_like_lite_fpasst_fusion_sc.yaml \
  --result_dir workspace/evaluation
```

Expected local assets:

```text
external/PretrainedSED/models/frame_passt/fpasst_wrapper.py
checkpoint/fpasst/fpasst_im.pt
checkpoint/m2d_as_vit_base-80x1001p16x16p32k-240413_AS-FT_enconly/weights_ep69it3124-0.47998.pth
```

Practical note:

- `fpasst_im.pt` is the public SSL checkpoint listed by the PretrainedSED
  project. It is a good default starting point for frozen fusion.
- Keep `freeze_aux: true` first. If you later try unfreezing fPaSST, reduce the
  learning rate sharply and expect a bigger memory/time bill than the frozen
  setup.
- Compare BEATs and fPaSST on the same stage-3 estimated-source cache. In this
  repo, that is the cleanest apples-to-apples ablation.

### Full-Context Spatial USS

Use this only if memory allows:

```bash
python -m src.train -c config/separation/modified_deft_uss.yaml -w workspace/separation
```

This uses `ModifiedDeFTUSSSpatial` with full global DeFT attention. It can be
very expensive for 10s audio because each block attends over the full time and
frequency axes.

### Short 5s USS With Chunked 10s Evaluation

Use this when you want the original global DeFT block but cannot train on 10s:

```bash
python -m src.train -c config/separation/modified_deft_uss_5s.yaml -w workspace/separation
```

This reduces training memory by synthesizing 5s mixtures, while
`ChunkedModifiedDeFTUSSSpatial` can process 10s evaluation audio in 6s chunks.

The tradeoff is that the model is trained with shorter temporal context, which
can affect SA-SDR for long events or events crossing chunk boundaries.

### Original KAIST-Like TSE

The older TSE config remains available:

```bash
python -m src.train -c config/separation/modified_deft_tse.yaml -w workspace/separation
```

It uses `ModifiedDeFTTSE`, raw enrollment waveform injection, and class
conditioning, but its final mask is still the older channel-0 complex-mask
synthesis. It also uses global DeFT attention, so it is memory-heavy.

### Original Simple M2D Source Classifier

The old two-stage classifier configs remain available:

```bash
python -m src.train -c config/label/m2d_sc_stage1.yaml -w workspace/label
python -m src.train -c config/label/m2d_sc_stage2.yaml -w workspace/label
```

Use them for ablation against `M2DSingleClassifierStrong`. For top-rank
experiments, prefer the `_strong` configs.

### Fine-Tuning SC On Estimated Sources

Clean dry-source classifier training is useful, but the classifier is used on
imperfect USS/TSE estimates during S5 inference. A top-rank system should adapt
SC to those artifacts.

The new stage-3 SC config is:

```bash
python -m src.train -c config/label/m2d_sc_stage3_estimated_strong.yaml -w workspace/label
```

For the pretrained fusion classifier, use:

```bash
python -m src.train -c config/label/m2d_sc_stage3_estimated_beats_fusion.yaml -w workspace/label
```

It expects a cached waveform dataset:

```text
workspace/sc_finetune/soundscape
workspace/sc_finetune/oracle_target
workspace/sc_finetune/estimate_target
```

`soundscape` and `oracle_target` follow the normal synthesized waveform dataset
layout. `estimate_target` contains separated sources from your trained USS/TSE
pipeline. Estimated files must use this naming convention:

```text
<soundscape>_<slot>_<label>.wav
```

Example:

```text
soundscape_000001_0_Speech.wav
soundscape_000001_1_Doorbell.wav
```

The label in the filename is used as the SC training target.

There are two ways to build the estimate cache:

- Supervised adaptation: run your separator/TSE with oracle labels and save
  estimates using oracle class names in the filenames. This is the preferred
  validation/dev adaptation path.
- Pseudo-label adaptation: run the full S5 pipeline and save estimates using
  predicted class names. This is easier, but label noise can hurt the SC if the
  first-pass classifier is weak.

Use the export utility to create the cache:

```bash
# supervised cache: oracle labels + TSE-estimated waveforms
python -m src.evaluation.export_sc_finetune_cache \
  -c src/evaluation/eval_configs/kwo2025_top1_like_lite_strong_sc.yaml \
  --output_root workspace/sc_finetune \
  --mode oracle_tse \
  --batchsize 1

# pseudo-label cache: full S5 predictions and predicted labels
python -m src.evaluation.export_sc_finetune_cache \
  -c src/evaluation/eval_configs/kwo2025_top1_like_lite_strong_sc.yaml \
  --output_root workspace/sc_finetune_pseudo \
  --mode pseudo_s5 \
  --batchsize 1
```

The supervised mode writes:

```text
workspace/sc_finetune/soundscape/<soundscape>.wav
workspace/sc_finetune/oracle_target/<soundscape>_<slot>_<oracle_label>.wav
workspace/sc_finetune/estimate_target/<soundscape>_<slot>_<oracle_label>.wav
```

The pseudo-label mode writes the same folder structure, but
`estimate_target` filenames use the model-predicted labels.

After training, use the matching evaluation config:

```bash
python -m src.evaluation.evaluate \
  -c src/evaluation/eval_configs/kwo2025_top1_like_lite_estimated_sc.yaml \
  --result_dir workspace/evaluation
```

That config expects:

```text
checkpoint/modified_deft_uss_lite_6s.ckpt
checkpoint/m2d_sc_stage3_estimated_strong.ckpt
checkpoint/modified_deft_tse_lite_6s.ckpt
```

For an ablation without estimated-source SC adaptation, use
`kwo2025_top1_like_lite_strong_sc.yaml`, which expects
`checkpoint/m2d_sc_stage2_strong.ckpt`.

### One-Stage Label-Query Separator

For a simpler separator that skips the explicit `USS -> SC -> TSE` chain:

```bash
python -m src.train -c config/separation/deft_tse_like.yaml -w workspace/separation
```

This uses `DeFTTSELikeSpatial`, which already has the all-channel spatial final
stage. It is trained through `LabelQueriedSeparationLightning` and
`class_aware_pit`.

## Task4 2026 Loss Recipe

DCASE 2026 Task4 differs from the previous setting in two important ways:

- a mixture may contain multiple simultaneous target sources with the same
  semantic class
- a mixture may contain no target sound class at all

The losses therefore should not assume a unique class label per mixture, and
they should not let all-silence targets become zero-gradient corner cases.

### Class-Aware PIT for Label-Queried Sources

`src/training/loss/class_aware_pit.py` now provides safe reusable helpers:

- `class_aware_pit_loss(...)`: only permits permutations among slots with the
  same query label
- `permutation_invariant_loss(...)`: ordinary PIT for unlabeled slots
- `inactive_source_energy_loss(...)`: suppresses output energy for inactive or
  unmatched prediction slots
- `unmatched_prediction_mask(...)`: identifies prediction slots not selected by
  the best PIT assignment

Use class-aware PIT for label-queried target extraction, where duplicated query
labels create slot ambiguity:

```text
query labels: [Speech, Speech, Dog]
allowed:      Speech slots may swap with Speech slots
blocked:      Speech slots may not swap with Dog slots
```

This follows the 2026 baseline paper: it keeps the benefit of label-conditioned
source extraction while removing the random same-class alignment problem.

### TSE Loss

`src/training/loss/masked_snr.py` now keeps the old `masked_snr_loss(...)`
helper for compatibility, but `get_loss_func(...)` uses the stronger Task4 2026
objective:

```text
loss =
  class_aware_pit_sdr_loss(active target queries)
  + lambda_inactive * energy(unmatched or inactive output slots)
```

This means:

- duplicated same-class target queries can be swapped during training
- different-class target queries cannot be swapped
- padded silence queries are pushed toward zero output
- all-zero target mixtures still produce a differentiable suppression loss

Recommended starting value:

```yaml
loss:
  module: src.training.loss.masked_snr
  main: get_loss_func
  args:
    lambda_inactive: 0.05
```

### USS Loss

USS is not label-queried, so foreground slots are unordered. The updated
`src/training/loss/uss_loss.py` uses full PIT for foreground matching, with a
class cost inside the matching objective:

```text
foreground_pair_cost =
  sa_sdr_cost(foreground_waveform)
  + lambda_class_match * cross_entropy(class_logits)
```

After the best foreground assignment is selected:

- matched active slots get waveform and class losses
- unmatched foreground slots get inactive-energy suppression
- `silence_logits` are supervised against the PIT-derived active/inactive slot
  state rather than the original dataset slot order
- inactive class logits are regularized toward a uniform distribution

Interference is intentionally not class-aware. The interference branch uses
ordinary PIT across interference slots plus inactive-energy suppression for
padded slots:

```text
interference_loss =
  pit_sdr_loss(active interference slots)
  + lambda_inactive_interference * energy(unmatched interference slots)
```

Noise/background remains a single auxiliary branch. It uses SI-SNR when active
and energy suppression when inactive.

The residual consistency term is available but disabled by default:

```yaml
loss:
  module: src.training.loss.uss_loss
  main: get_loss_func
  args:
    lambda_non_foreground: 0.01
    lambda_class_match: 1.0
    lambda_inactive_foreground: 0.05
    lambda_inactive_interference: 0.01
    lambda_inactive_noise: 0.01
    lambda_residual: 0.0
```

Keep `lambda_residual` small or zero because the training targets are dry/direct
components while the input mixture includes spatial rendering and background
terms. Treat this as an ablation, not a default.

### Zero-Target Training

The separation configs now sample zero-target mixtures:

```yaml
nevent_range: [0, 3]
```

This applies to:

- `config/separation/resunetk_capisdr.yaml`
- `config/separation/deft_tse_like.yaml`
- `config/separation/modified_deft_uss.yaml`
- `config/separation/modified_deft_uss_5s.yaml`
- `config/separation/modified_deft_uss_lite_6s.yaml`
- `config/separation/modified_deft_tse.yaml`
- `config/separation/modified_deft_tse_lite_6s.yaml`

The S5 inference wrapper also guards the no-target case: if the source
classifier returns all-zero labels for a sample after USS, `Kwon2025S5` returns
silent target outputs for that sample instead of running TSE refinement and
risking hallucinated target labels.

### Smoke Tests

Run the focused CPU checks with:

```bash
./.venv/bin/python tests/test_task4_2026_losses.py
```

The test covers:

- duplicate same-class target swap gives zero/low loss
- distinct labels cannot swap
- all-silence TSE targets produce nonzero output gradients
- `Kwon2025S5` returns all-silence outputs without invoking TSE when SC predicts
  no target labels

## Evaluation Guides

### Recommended Lite Multi-Stage Evaluation

Use this after training the lite USS, M2D-SC stage 2, and lite TSE checkpoints:

```bash
python -m src.evaluation.evaluate \
  -c src/evaluation/eval_configs/kwo2025_top1_like_lite_estimated_sc.yaml \
  --result_dir workspace/evaluation
```

This evaluation config reads fixed 10s files from:

```text
data/dev_set/synthesized/test/soundscape
data/dev_set/synthesized/test/oracle_target
```

It does not shorten evaluation audio. The lite USS and lite TSE models handle
long inputs internally, and the strong SC averages predictions over 5s crops
with 2.5s hop when classifying longer source estimates.

If you want the same lite USS/TSE system with the strong clean-source
classifier, use:

```bash
python -m src.evaluation.evaluate \
  -c src/evaluation/eval_configs/kwo2025_top1_like_lite_strong_sc.yaml \
  --result_dir workspace/evaluation
```

If you want the same lite USS/TSE system with the older simple classifier, use:

```bash
python -m src.evaluation.evaluate \
  -c src/evaluation/eval_configs/kwo2025_top1_like_lite_uss.yaml \
  --result_dir workspace/evaluation
```

### Chunked-USS Evaluation With Original TSE

This config uses chunked USS but keeps the original full-attention TSE:

```bash
python -m src.evaluation.evaluate \
  -c src/evaluation/eval_configs/kwo2025_top1_like_chunked_uss.yaml \
  --result_dir workspace/evaluation
```

Use it only when you trained `modified_deft_uss_5s.yaml` and still want the
older `ModifiedDeFTTSE` stage.

### Original KAIST-Like Evaluation

```bash
python -m src.evaluation.evaluate \
  -c src/evaluation/eval_configs/kwo2025_top1_like.yaml \
  --result_dir workspace/evaluation
```

This uses the original full-attention DeFT USS/TSE settings from that eval
config.

## Checkpoint Notes

Checkpoint compatibility is intentionally strict:

- `ModifiedDeFTUSS` checkpoints do not directly load into
  `ModifiedDeFTUSSSpatial`, because the audio head shape changed and `out_conv`
  was added.
- `ModifiedDeFTUSSSpatial` checkpoints do not directly load into
  `ModifiedDeFTUSSMemoryEfficient`, because the DeFT block modules changed.
- `ModifiedDeFTTSE` checkpoints do not directly load into
  `ModifiedDeFTTSEMemoryEfficient` for the same reason.
- `DeFTTSELike` checkpoints do not directly load into `DeFTTSELikeSpatial`.
- `M2DSingleClassifier` checkpoints do not directly load into
  `M2DSingleClassifierStrong`, because the pooling/projection head changed.
- `M2DSingleClassifierStrong` checkpoints do not directly load into
  `M2DPretrainedFusionClassifier`, because the fusion head and auxiliary
  encoder branch are new. You can partially transfer the M2D backbone and strong
  M2D projection layers, but train a fresh fusion/ArcFace head.

When switching between these families, train a fresh checkpoint or write an
explicit partial-loading script that only transfers compatible layers.

## Dataset And External Data Notes

The DeFT separation configs use SpAudSyn on the fly. Make sure the baseline data
layout exists before training:

```text
data/dev_set/sound_event/train
data/dev_set/interference/train
data/dev_set/noise/train
data/dev_set/room_ir/train
```

If EARS and SemanticHearing have been added with `add_data.sh`, they become part
of the training pools under those folders. The evaluation configs do not use
SpAudSyn; they read prepared 10s waveform files from the synthesized test set.

For estimated-source SC fine-tuning, `DatasetS3` reads `estimate_target_dir` and
`EstimatedSourceClassifierDataset` trains on `est_dry_sources` / `est_label`.
The label is parsed from the estimate filename, so keep filenames aligned with
the intended target labels.

## Source-Classifier Energy Threshold Calibration

`M2D-SC` predicts a raw class for every source slot, then suppresses the slot as
silence when its class energy is above the configured threshold. The default
threshold is intentionally only a starting point:

```yaml
energy_thresholds:
  default: -3.5
```

To calibrate per-class thresholds on validation source slots, run:

```bash
python -m src.evaluation.calibrate_sc_energy_thresholds \
  -c config/label/m2d_sc_stage2_strong.yaml \
  --checkpoint checkpoint/m2d_sc_stage2_strong.ckpt \
  --device cuda \
  --batch-size 32 \
  --num-workers 8 \
  --beta 0.5 \
  --max-fpr 0.05 \
  --default-threshold -3.5 \
  --output-dir workspace/calibration/m2d_sc_stage2_strong
```

For estimated-source adapted classifiers, point the script at the matching
stage-3 label config and checkpoint:

```bash
python -m src.evaluation.calibrate_sc_energy_thresholds \
  -c config/label/m2d_sc_stage3_estimated_strong.yaml \
  --checkpoint checkpoint/m2d_sc_stage3_estimated_strong.ckpt \
  --device cuda \
  --batch-size 32 \
  --num-workers 8 \
  --beta 0.5 \
  --max-fpr 0.05 \
  --default-threshold -3.5 \
  --output-dir workspace/calibration/m2d_sc_stage3_estimated_strong
```

The script writes:

```text
workspace/calibration/.../energy_thresholds.yaml
workspace/calibration/.../energy_thresholds.json
workspace/calibration/.../energy_threshold_stats.csv
```

Paste the `energy_thresholds` block into the SC model args in the evaluation
config you are running. For example:

```yaml
energy_thresholds:
  0: -5.8
  1: -5.2
  2: -6.1
  default: -3.5
```

The calibration objective treats validation slots whose raw predicted class
matches the true active class as positives; silence slots and wrong-class slots
are negatives for that predicted class. `--beta 0.5` favors precision, which is
usually safer for reducing false positives from silent slots. Increase beta if
the calibrated thresholds suppress too many true sources.

Before running, make sure the model's base pretrained weight and the fine-tuned
SC checkpoint paths referenced by the config exist locally. The script
instantiates the SC model from the config, so missing `weight_file` artifacts
will fail before calibration starts.

## Practical Tuning

Recommended first try on a memory-constrained GPU:

```yaml
batch_size: 2
precision: bf16-mixed
time_window_size: 128
freq_group_size: 64
```

For `M2DSingleClassifierStrong`, start with:

```yaml
batch_size: 32
precision: bf16-mixed
pooling_hidden_dim: 512
projection_hidden_dim: 1024
dropout: 0.2
eval_crop_seconds: 5.0
eval_crop_hop_seconds: 2.5
```

If memory is still too high:

- reduce `batch_size` to `1`
- reduce `n_deft_blocks` from `6` to `4`
- reduce `time_window_size` from `128` to `96` or `64`
- reduce `freq_group_size` from `64` to `32`
- keep `hop_size: 320` unless you intentionally want a time-resolution tradeoff

If quality is too low and memory allows:

- increase `time_window_size` to `192` or `256`
- increase `freq_group_size` to `96` or `128`
- keep `shift_windows: true`
- compare 6s training against 5s training on the same validation split

## Quick CPU Sanity Checks

These checks do not measure quality. They only verify config instantiation,
shape compatibility, and finite losses on CPU.

The M2D-SC construction check requires the pretrained M2D checkpoint referenced
by the YAML. If that checkpoint is not present, skip that one check or run it
after placing the M2D weights under `checkpoint/`.

```bash
cd ./dcase2026_task4_baseline

# Check the memory-efficient USS model can be constructed from YAML.
PYTHONPATH=. .venv/bin/python - <<'PY'
import yaml
from src.utils import initialize_config

with open("config/separation/modified_deft_uss_lite_6s.yaml") as f:
    cfg = yaml.safe_load(f)
model = initialize_config(cfg["lightning_module"]["args"]["model"])
print(type(model).__name__, type(model.blocks[0]).__name__, model.time_window_size, model.freq_group_size)
PY

# Check the memory-efficient TSE model can be constructed from YAML.
PYTHONPATH=. .venv/bin/python - <<'PY'
import yaml
from src.utils import initialize_config

with open("config/separation/modified_deft_tse_lite_6s.yaml") as f:
    cfg = yaml.safe_load(f)
model = initialize_config(cfg["lightning_module"]["args"]["model"])
print(type(model).__name__, type(model.blocks[0]).__name__, model.time_window_size, model.freq_group_size)
PY

# Check the strong source classifier can be constructed from YAML.
PYTHONPATH=. .venv/bin/python - <<'PY'
import yaml
from src.utils import initialize_config

with open("config/label/m2d_sc_stage2_strong.yaml") as f:
    cfg = yaml.safe_load(f)
model = initialize_config(cfg["lightning_module"]["args"]["model"])
print(type(model).__name__, type(model.pool).__name__, model.embedding[-1].normalized_shape)
PY

# Check the estimated-source SC dataset can be configured.
PYTHONPATH=. .venv/bin/python - <<'PY'
import yaml

with open("config/label/m2d_sc_stage3_estimated_strong.yaml") as f:
    cfg = yaml.safe_load(f)
dargs = cfg["datamodule"]["args"]["train_dataloader"]["dataset"]["args"]
print(dargs["source_prefix"], dargs["base_dataset"]["args"]["config"]["estimate_target_dir"])
PY

# Check the cache exporter CLI imports.
PYTHONPATH=. .venv/bin/python -m src.evaluation.export_sc_finetune_cache --help

# Check the lite USS training dataset produces 6s audio.
PYTHONPATH=. .venv/bin/python - <<'PY'
import yaml
from src.utils import initialize_config

with open("config/separation/modified_deft_uss_lite_6s.yaml") as f:
    cfg = yaml.safe_load(f)
dset = initialize_config(cfg["datamodule"]["args"]["train_dataloader"]["dataset"])
item = dset[0]
print(item["mixture"].shape, item["mixture"].shape[-1] / 32000)
PY

# Check the lite TSE training dataset produces 6s mixture and enrollment audio.
PYTHONPATH=. .venv/bin/python - <<'PY'
import yaml
from src.utils import initialize_config

with open("config/separation/modified_deft_tse_lite_6s.yaml") as f:
    cfg = yaml.safe_load(f)
dset = initialize_config(cfg["datamodule"]["args"]["train_dataloader"]["dataset"])
item = dset[0]
print(item["mixture"].shape, item["enrollment"].shape, item["mixture"].shape[-1] / 32000)
PY
```

## Remaining Work

- Temporal-aware SC phase 1 is now implemented as an opt-in path:
  `config/label/m2d_sc_stage1_temporal_strong.yaml`,
  `config/label/m2d_sc_stage2_temporal_strong.yaml`, and
  `config/label/m2d_sc_stage3_estimated_temporal_strong.yaml`.
- These configs use `M2DSingleClassifierTemporalStrong`, which keeps the normal
  SC `label_vector` / `probabilities` / `energy` prediction contract and adds a
  frame-level `activity_logits` head trained from `span_sec`.
- `DatasetS3`, `SourceClassifierDataset`, `EstimatedSourceClassifierDataset`,
  `TSEDataset`, and `USSDataset` now carry span tensors when available. Silence
  or unknown timing uses `[-1, -1]`, so older losses and models can ignore the
  new fields.
- The activity term is controlled by `lambda_activity` in
  `src/training/loss/m2d_sc_arcface.py`; existing non-temporal configs leave it
  at the default `0.0`.
- Recommended temporal rollout:
  1. train `m2d_sc_stage1_temporal_strong.yaml` on clean generated sources;
  2. continue with `m2d_sc_stage2_temporal_strong.yaml` for energy/silence
     calibration;
  3. fine-tune with `m2d_sc_stage3_estimated_temporal_strong.yaml` after
     exporting estimated-source caches.
- Later phases should consume the same span contract in temporal USS/TSE heads
  and then add S5 stage-to-stage temporal handoff. The current temporal SC phase
  deliberately leaves previous USS/TSE behavior unchanged.
- Temporal USS/TSE phase 2 is also available as an opt-in path:
  `config/separation/modified_deft_uss_lite_6s_temporal.yaml` and
  `config/separation/modified_deft_tse_lite_6s_temporal.yaml`.
- Non-memory-efficient temporal variants are available too:
  `config/separation/modified_deft_uss_temporal.yaml`,
  `config/separation/modified_deft_uss_5s_temporal.yaml`,
  `config/separation/modified_deft_tse_temporal.yaml`, and
  `config/separation/deft_tse_like_temporal.yaml`.
- These use `ModifiedDeFTUSSSpatialTemporal`,
  `ChunkedModifiedDeFTUSSSpatialTemporal`, `ModifiedDeFTTSETemporal`, and
  `DeFTTSELikeSpatialTemporal`, respectively. The original non-temporal configs
  still use the old classes.
- `ModifiedDeFTUSSMemoryEfficientTemporal` adds foreground/interference/noise
  activity heads while preserving the previous waveform, class, and silence
  outputs. Its loss uses `lambda_activity_foreground`,
  `lambda_activity_interference`, and `lambda_activity_noise`; the older USS
  recipes leave these at their default zero values.
- `ModifiedDeFTTSEMemoryEfficientTemporal` adds per-query activity logits to
  the memory-efficient TSE path. Its activity loss is controlled by
  `lambda_activity` in `src/training/loss/masked_snr.py`, again defaulting to
  zero for previous TSE configs.
- The recommended temporal order is SC first, then temporal USS, then temporal
  TSE, then temporal S5 evaluation with
  `src/evaluation/eval_configs/kwo2025_top1_like_lite_estimated_temporal_sc.yaml`
  or the duplicate-recall sibling
  `src/evaluation/eval_configs/kwo2025_top1_like_lite_estimated_temporal_sc_duplicate_recall.yaml`.
- `Kwon2025TemporalS5` passes predicted USS/SC activity cues between stages,
  zeros inactive frames after USS/TSE, suppresses slots with no temporal support,
  prevents duplicate recall from reviving unsupported slots, and forwards the
  current activity trace to temporal TSE as `temporal_conditioning`.
- `ModifiedDeFTTSETemporal` and `ModifiedDeFTTSEMemoryEfficientTemporal` consume
  `temporal_conditioning` with an optional time-FiLM conditioner, so the TSE
  blocks can change their computation from the predicted active span rather than
  using activity only as a post-hoc waveform gate.
- The TSE models also accept `query_condition`, `tse_condition`,
  `bridge_condition`, or `proposal_condition`. When `query_condition_dim > 0`,
  this proposal vector is projected into per-query FiLM offsets alongside the
  class label condition. The S5 wrappers forward USS context only when
  `tse_uss_conditioning_enabled: true`, so existing evaluation configs keep the
  old behavior.
- TSE-only evaluation with `evaluate_stage.py --stage tse` forwards
  `query_condition`, `tse_condition`, `bridge_condition`,
  `proposal_condition`, and `temporal_conditioning` when the stage dataset
  provides them. For live USS-derived proposal handoff, use full S5 evaluation,
  because oracle TSE stage evaluation does not know the USS proposal-to-oracle
  slot assignment.
- Estimated-enrollment TSE configs train on USS/S5-generated enrollment
  waveforms from `workspace/sc_finetune/estimate_target`. The
  `*_uss_conditioned.yaml` siblings additionally train TSE to use USS proposal
  context; this is the preferred alignment path after USS gains new
  spatial/temporal/activity heads.
- Per-class energy thresholds for `M2D-SC` should be calibrated from validation
  data instead of relying on a placeholder default.
- `m2d_sc_arcface.get_loss_func` has an opt-in duplicate-recall term for reducing
  missed same-class duplicates. `SourceClassifierDataset` marks active sources
  whose label appears more than once in the same mixture; set
  `lambda_duplicate_recall > 0` and a stricter `duplicate_m_in` such as `-8.0`
  in the stage-2 source-classifier loss to push duplicated active sources away
  from the silence threshold.
- `m2d_sc_arcface.get_loss_func` also has an opt-in robust estimated-source mode:
  set `robust_loss_mode: truncation` to downweight the high-loss active source
  labels in each batch after `truncation_warmup_epochs`. The duplicate-recall
  term shares the same per-source truncation weights so a suspicious pseudo-label
  is not amplified as a duplicate-recall target.
- Full validation quality still needs real GPU runs; the CPU checks only verify
  that the code paths are wired correctly.

## Review And Fast-Test Notes

The current implementation was reviewed with CPU-focused smoke tests. These are
not quality measurements, but they catch shape, config, loss, and wiring
regressions before launching expensive GPU runs.

Fixes made during review:

- `src/train.py`: `--batchsize` no longer assumes every config has a
  `val_dataloader`.
- `src/utils.py`: missing `transformers` no longer breaks basic training imports.
- `.gitignore`: local `data/`, `.venv/`, `external_data/`, and the local
  SpAudSyn symlink are ignored.
- `src/datamodules/dataset.py`: waveform-mode source file matching now uses the
  exact soundscape filename pattern instead of a broad `startswith`, avoiding
  prefix collisions such as `soundscape_000001` matching `soundscape_0000010`.
- `src/evaluation/evaluate.py`: the optional label-conditioned prediction path
  now calls `self.model.separate(...)`.
- `src/models/m2dat/m2d_sc.py`: `2_blocks` now truly unfreezes the last two M2D
  transformer blocks plus the classifier head.
- `src/models/m2dat/m2d_sc.py`: added `M2DPretrainedFusionClassifier` with
  frozen auxiliary BEATs/fPaSST/AST/ATST-capable encoder backends and
  `concat_mlp`/`gated_mlp` fusion.
- `config/label/m2d_sc_stage*_beats_fusion.yaml`: added the recommended BEATs
  fusion SC training path for clean-source, energy/silence, and
  estimated-source stages.
- `config/label/m2d_sc_stage*_fpasst_fusion.yaml`: added the analogous fPaSST
  fusion SC training path.
- `config/**/*.yaml`: all training configs now include a `val_dataloader`,
  `is_validation: true`, and a `ModelCheckpoint` callback monitoring
  `epoch_val/loss`.
- Validation dataloaders for generate-mode USS/TSE/SC configs use
  `data/dev_set/metadata/valid.json`. Estimated-source SC stage-3 configs train
  on the cached estimated-source dataset and validate on clean validation
  sources so checkpointing works without requiring a second estimated-source
  cache.
- `src/evaluation/eval_configs/kwo2025_top1_like_lite_beats_fusion_sc.yaml`:
  added an S5 evaluation config using the fused source classifier.
- `src/evaluation/eval_configs/kwo2025_top1_like_lite_fpasst_fusion_sc.yaml`:
  added the corresponding fPaSST-fusion S5 evaluation config.
- `.gitignore`: local `external/` pretrained-code folders and `*.pt`
  checkpoints are ignored.

Fast checks run:

```bash
# Python syntax for source files, excluding the local SpAudSyn symlink
PYTHONPATH=. .venv/bin/python -m py_compile $(find src -name '*.py' -not -path 'src/modules/spatial_audio_synthesizer/*' | sort)

# YAML parse for all train/eval configs
PYTHONPATH=. .venv/bin/python - <<'PY'
import yaml
from pathlib import Path
for path in sorted(Path("config").rglob("*.yaml")) + sorted(Path("src/evaluation/eval_configs").rglob("*.yaml")):
    with open(path) as f:
        yaml.safe_load(f)
    print("ok", path)
PY
```

Additional targeted checks passed:

- `ModifiedDeFTUSSMemoryEfficient` forward plus USS loss on tiny tensors.
- `ModifiedDeFTTSEMemoryEfficient` forward plus Task4 2026 CA-PIT TSE loss on
  tiny tensors.
- `DeFTTSELikeSpatial` forward plus class-aware PIT loss on tiny tensors.
- `M2DSingleClassifierStrong` forward, predict, loss, and `2_blocks`
  trainability using a stub M2D backbone because the real M2D checkpoint was not
  present on this host.
- `M2DPretrainedFusionClassifier` forward, predict, loss, and fusion-shape
  checks using a stub M2D backbone and the `identity` aux backend because the
  host has no BEATs/M2D checkpoints installed.
- `M2DPretrainedFusionClassifier` `official_fpasst` path using a temporary fake
  PretrainedSED package and local `.pt` checkpoint to exercise the import,
  checkpoint remapping, mel-forward, sequence pooling, and predict flow.
- `EstimatedSourceClassifierDataset` with a temporary cache, including exact
  filename matching and silence padding.
- `export_sc_finetune_cache.py --help`.
- `export_sc_finetune_cache.py` with a temporary waveform dataset and fake S5
  model, producing the expected `soundscape/`, `oracle_target/`, and
  `estimate_target/` files.
- `tests/test_task4_2026_losses.py`, covering same-class CA-PIT swaps,
  distinct-label no-swap behavior, all-silence suppression gradients, and the
  all-silence `Kwon2025S5` inference guard.
- YAML validation/checkpoint audit over every training config:
  each config has a validation dataloader, `is_validation: true`, and a
  checkpoint callback with `monitor: epoch_val/loss`, `mode: min`,
  `save_top_k: 1`, and `save_last: true`.
- Validation dataset construction was checked for every training config with
  `initialize_config(...)`.
- Checkpoint/TQDM callback construction was checked for every training config
  with `initialize_config(...)`.
