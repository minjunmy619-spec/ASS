# Audio Separation Research Gap Audit

Date: 2026-05-29

This note summarizes what is still missing or only partially implemented from
the research notes and status investigations:

- `audio_separtion_model_research.md`
- `audio_source_separation_edge_sota_research_20260528.md`
- `auido_separtion_model_more.md`
- `tiger_older_branches_failure_investigation_20260517.md`

It compares those recommendations against the current ASS repo implementation.

## Current Implemented Base

The current repo now has the main deployable skeleton recommended by the
research notes:

- `BandSFCNet-RT+` as an opt-in preset through `band_sfc_preset: rt_plus`.
- Proposal builders for:
  - `SFC-Locoformer-Lite+`
  - `BandSFCNet-RT+`
  - `Hierarchical-SFC-FFI-Lite`
  - `EdgeFusion-SFC-Distilled`
- Teacher-student training plumbing through `TeacherStudentDistillationTask`.
- DnR and MUSDB sibling recipes for the RT+ student path.
- DnR distillation recipes for BandSFC RT+ and EdgeFusion-SFC.
- Streaming ONNX export validation for RT+ with no `Tile`, `Expand`, or
  `ConstantOfShape` in the exported graph.
- A reusable TIGER/ONE-derived ONNX risk audit in
  `tools/online/audit_onnx_model.py`, also wired into
  `tools/online/export_verify_mlir.py`.
- Composite spectral distillation losses for complex RI, log magnitude,
  multi-resolution STFT, and transient preservation through
  `CompositeSeparationSpectralLoss` and `TeacherStudentDistillationTask`.
- A pinned three-stem benchmark contract and result/listening templates in
  `docs/AUDIO_SEPARATION_BENCHMARK_CONTRACT.md` and `docs/templates/`.
- Proposal smoke tests for the RT+ preset, the SFC-Locoformer teacher builder,
  middle/edge proposal builders, and the distillation-task fail-fast behavior.
- A completed TIGER older-branch compile investigation showing that all seven
  TIGER recipe branches can now reach quantized Circle after export-safe graph
  rewrites.

This is a useful foundation, but it is not yet the full research plan.

## Missing Or Partial Items

### 1. No Trained Teacher Or Student Evidence Yet

Status: Missing

The research notes repeatedly recommend training a strong SFC/TF-Locoformer
teacher first, then distilling into edge students.

Current gap:

- No trained or downloaded `SFC-Locoformer-Lite+` checkpoint is wired as the
  actual teacher.
- `BandSFCNet-RT+` has runnable recipes but no trained metric table.
- `EdgeFusion-SFC-Distilled` has a distillation recipe but no trained student.
- No MUSDB/DnR SDR, SNR, SI-SDR, cSDR, or uSDR results are reported for the new
  proposal variants.

Required next work:

- Train or import a small SFC-Locoformer teacher.
- Train BandSFC RT+ with and without distillation.
- Train EdgeFusion-SFC only after the BandSFC student is useful.
- Add a results table for quality, state, compute, and export status.

### 2. Full Loss Stack Is Not Implemented

Status: Partial

Implemented:

- Supervised SNR-style loss through existing task losses.
- Teacher output L1 loss.
- Mixture consistency loss.
- Low-frequency weighted loss.
- Silent-source penalty.
- Complex RI STFT loss.
- Log-magnitude consistency loss.
- Multi-resolution complex STFT loss.
- Transient waveform-difference loss.

Missing:

- Latent/intermediate SFC-band distillation.
- Teacher mask or logit distillation.
- Source-activity-aware loss weighting.

Required next work:

- Add optional latent hooks in teacher and student models.
- Add teacher mask/logit outputs where the teacher exposes them cleanly.
- Add source-activity-aware weighting beyond silent-source penalties.

### 3. True Sparse U-Net Mel-SFC Is Not Implemented

Status: Missing

The notes propose `Sparse U-Net Mel-SFC` as a music-first fallback with:

- Sparse low/mid/high-band branches.
- Asymmetric encoder/decoder depth.
- U-Net skip paths.
- Lightweight TF-Locoformer or local/global bottleneck blocks.
- Shared complex mask head.

Current repo state:

- `Hierarchical-SFC-FFI-Lite` is exposed as a related middle-tier direction.
- It is not the same as the proposed sparse asymmetric U-Net.

Required next work:

- Add `SparseBandUNetEncoder`.
- Add `SparseBandUNetDecoder`.
- Add sparse band routing for low/mid/high frequency regions.
- Add a bottleneck option using Lite-Locoformer or CNB blocks.
- Add MUSDB-first configs and smoke tests.

### 4. Prompted Asymmetric SFC Is Not Implemented

Status: Missing

The notes propose a future unified model with prompt-conditioned outputs.

Missing:

- Prompt embeddings.
- Prompt-conditioned shared decoder.
- Prompt dropout training.
- Task/source prompt taxonomy.
- Category-aware PIT for same-class sources.
- Unified speech/music/SFX output contract.

Reason not implemented yet:

This needs a real product/task/data contract. Adding placeholder prompt code
without dataset and evaluation semantics would be misleading.

Required next work:

- Define prompt IDs and source taxonomy.
- Define fixed-output vs prompted-output training stages.
- Add datamodule support for prompt batches.
- Add category-aware PIT only where multiple same-class outputs exist.

### 5. SepReformer-Style Early Source Split Is Missing

Status: Missing

The notes recommend testing early source disentanglement at compressed SFC token
level.

Missing:

- Source split module after SFC compression.
- Shared source refiner.
- Weight-shared reconstruction decoder.
- Cross-source reconstruction logic.

Current repo state:

- Current implementations mostly use shared latent processing and late output
  heads.
- RT+ adds a residual correction head, but this is not early source splitting.

Required next work:

- Add a source-token split option to the SFC teacher or middle-tier model.
- Evaluate it first in non-strict or middle-tier mode before making an NPU
  student.

### 6. Mamba2 Or Residual Refinement Branch Is Missing

Status: Missing

The notes do not recommend replacing SFC-CA with Mamba by default. They do
recommend testing Mamba/Mamba2 selectively in:

- A long temporal branch after SFC compression.
- A second-stage residual correction/refinement branch.
- TS-BSMamba2-style correction.

Current repo state:

- SFC-Mamba encoder/decoder code exists.
- The targeted Mamba2 residual/refinement branch is not implemented.

Required next work:

- Add a second-stage residual refinement module behind an explicit config flag.
- Keep it outside the strict NPU path until exportability is proven.
- Compare against the existing RT+ residual head.

### 7. Adaptive Mel / Overlapped Perceptual Band Mapping Is Incomplete

Status: Partial

Implemented:

- SFC cross-attention compression.
- `band_config: musical` in existing SFC-style recipes.
- Frequency preprocessing such as `fp512keep475`.

Missing:

- Explicit 80-band overlapped mel-style front-end.
- Direct ablation between:
  - fixed bands,
  - mel-overlap bands,
  - SFC-CA,
  - SFC-Mamba.
- Low-frequency overlap controls for bass/music preservation.

Required next work:

- Add a mel-overlap band spec or config mode.
- Add sibling configs that differ only by band mapping.
- Report quality and deployment metrics for each.

### 8. Full Deployment Validation Is Not Complete

Status: Partial

Done:

- RT+ streaming ONNX export.
- ONNX checker.
- Forbidden-op audit for `Tile`, `Expand`, and `ConstantOfShape`.
- RT+ fp512 state check around `186 KiB` fp16 layer-cache.

Missing:

- Full ONE import/optimize/quantize/`circle-verify` for RT+.
- GMAC/s measurement for RT+.
- ONNX node count table across new proposal variants.
- MLIR op count table.
- Runtime latency measurement.
- Listening notes for bass, drums/transients, vocals/speech leakage, and
  artifacts.

Required next work:

- Run `tools/online/measure_npu_model_stats.py`.
- Run `tools/online/export_verify_mlir.py` without `--skip-emit-mlir`.
- Add RT+ to the verification matrix.

### 9. Quantization-Aware Training Is Missing

Status: Missing

The notes recommend QAT with distillation because source separation is sensitive
to input/output activation quantization.

Missing:

- QAT recipe.
- Activation sensitivity policy.
- Distillation-aware QAT objective.
- Option to keep fragile input/output edges at higher precision.
- Quantized quality comparison against FP32/FP16.

Required next work:

- First train a strong FP32/FP16 model.
- Add QAT only after quality is stable.
- Track SDR/SNR loss from quantization separately from architecture quality.

### 10. Training Recipe Details Are Not Fully Implemented

Status: Implemented for recipe plumbing; pending trained evidence

Existing recipes cover the base datamodule, source dropout, optimizer, EMA, and
basic training schedule.

Implemented:

- `SourceSeparationAugmenter` with opt-in gain perturbation, polarity inversion,
  stereo/channel swap, time shift, mild pitch/time perturbation, random EQ, and
  mild band dropout.
- Datamodule/dataset flags for training-only augmentation, source remixing,
  source RMS normalization, configurable source gain range, active-crop retry,
  source activity threshold, and minimum active-source crops where references
  are available.
- Staged DnR configs for:
  - augmented SFC-Locoformer teacher training,
  - supervised RT+ student warm start,
  - chunk-causal RT+ distillation,
  - strict short-chunk RT+ distillation fine-tuning.
- `docs/AUDIO_SEPARATION_TRAINING_RECIPE.md` with stage order and launch
  commands.

Required next work:

- Run the staged recipes and fill the benchmark manifest with local metrics.
- Tune augmentation probabilities if listening tests show transient smearing or
  speech/music leakage.

### 11. Ablation Matrix Is Missing

Status: Missing

The notes request architecture-level ablations, not random hyperparameter
sweeps.

Missing ablations:

- Fixed bands vs mel-overlap vs SFC-CA vs SFC-Mamba.
- CNB blocks vs Lite-Locoformer vs sparse U-Net bottleneck.
- Offline teacher vs chunk-causal student vs strict frame-streaming student.
- BandSFC safe vs quality vs RT+.
- BandSFC RT+ vs Hierarchical-SFC-FFI-Lite vs EdgeFusion-SFC.
- With and without teacher distillation.
- With and without low-frequency loss.
- With and without residual correction head.

Required next work:

- Create sibling config groups for each ablation.
- Add a results manifest schema.
- Include quality, state, GMAC/s, ONNX ops, MLIR ops, and ONE verification.

### 12. DolphinSFCNPU Distillation Is Missing

Status: Implemented for recipe plumbing; pending trained evidence

The second research note lists `DolphinSFCNPU slim 6m/8m distilled` as a
possible medium-high quality deployable candidate.

Current repo state:

- DolphinSFCNPU variants exist.
- Dedicated distillation recipes now exist for:
  - `dolphin-sfc-npu.slim-6m.distill.rt192k.fp512keep475`
  - `dolphin-sfc-npu.slim-8m.distill.rt192k.fp512keep475`
- These recipes use `TeacherStudentDistillationTask`, the same composite loss
  stack as the RT+ distillation path, and the same augmentation controls as the
  staged student recipes.
- `slim_8m` was tightened to about 6.55M params so the larger Dolphin probe
  remains inside the repo's edge budget checks.
- `docs/DOLPHIN_SFC_NPU_DISTILLATION.md` documents launch commands and teacher
  checkpoint compatibility.
- The benchmark manifest template now includes DolphinSFCNPU distilled rows.

Required next work:

- Train `slim_6m` first with a real teacher checkpoint.
- Run the same quality/deployment metrics as BandSFC RT+.
- Only run `slim_8m` if `slim_6m` shows useful quality and stays within the
  state/GMAC budget.

### 13. External Quality Baselines Are Not Integrated

Status: Partial

The notes call out several baselines or ceilings:

- Existing SFC-Locoformer small/medium.
- BS-RoFormer or Mel-Band RoFormer as music separation ceiling.
- TF-Locoformer-M as speech quality-efficiency reference.
- BandSCNet and BandSFC safe/quality.
- EdgeFusion scaffold variants.

Current gap:

- A benchmark/result manifest template now exists and separates `paper` rows
  from `local` rows.
- The manifest is not yet populated with local measured checkpoints.
- External baselines are represented as template rows, not validated through the
  local evaluation scripts.

Required next work:

- Fill the manifest from actual local evaluation runs.
- Use the same evaluation scripts for local variants where possible.

### 14. TIGER / ONE Failure Lessons Are Not Yet A Reusable Checklist

Status: Partial

The TIGER investigation is important because it produced concrete ONE compiler
failure patterns and fixes, not just TIGER-specific notes.

Resolved in TIGER:

- Dynamic Q/K/V channel splits from `torch.chunk` were replaced with static
  `view` plus fixed slice bounds.
- `Tile`-exporting resize paths were removed.
- Export-hostile `PReLU` was sanitized for NPU recipes.
- Scalar indexing patterns that lowered to problematic `Gather` rank handling
  were avoided.
- Rank-3 activation-attention `MatMul` paths that lowered to unsupported
  non-constant `FULLY_CONNECTED` were replaced with rank-4 batched matmul.
- All older TIGER recipe branches now complete ONNX export, ONE import,
  optimization, calibration, and quantized Circle generation.

Current gap:

- These lessons are now encoded in `tools/online/audit_onnx_model.py` and
  exposed through `tools/online/export_verify_mlir.py`.
- RT+ ONNX export still shows strict-edge risk flags beyond the older
  `Tile`/`Expand`/`ConstantOfShape` check. On the current
  `band_sfc_net_rt_plus_stream.onnx` artifact, the new audit reports dynamic
  `Slice` bounds and rank<=3 activation `MatMul` patterns.
- The audit has not yet been run across every proposal candidate.
- TIGER compile success does not make TIGER the primary quality candidate; it
  mainly provides proven graph hygiene patterns for the strict NPU path.

Required next work:

- Run that audit for RT+, Hierarchical-SFC-FFI-Lite, DolphinSFCNPU, and
  EdgeFusion-SFC before spending long training time.
- Rewrite or prove-safe the RT+ dynamic `Slice` and rank<=3 activation `MatMul`
  paths before treating RT+ as fully strict-edge clean.
- Keep the edge-v2/static-shape pattern as the template for future strict NPU
  attention or cross-band modules.

### 15. Dataset, Stem, And Metric Contract Is Still Not Pinned

Status: Partial

The research notes cover multiple task families: MUSDB-style music stems, DnR
speech/music/effects stems, speech separation, universal separation, and future
prompted tasks. The project target is currently a three-stem TV/edge model
(`speech`, `music`, `effects`), but several recipes and proposals still mix
three-stem CASS and four-stem MUSDB assumptions.

Current gap:

- The primary three-stem benchmark contract is now pinned in
  `docs/AUDIO_SEPARATION_BENCHMARK_CONTRACT.md`.
- Result and listening-note CSV templates now exist under `docs/templates/`.
- The teacher path is discussed for both MUSDB and DnR, but the exact checkpoint
  selection and conversion from four-stem music training to three-stem TV/CASS
  deployment is not defined.
- Prompted/unified future work depends on source taxonomy, but that taxonomy is
  not pinned even for the fixed three-stem product path.

Required next work:

- Fill the result manifest after local training/evaluation runs.
- Select or train the actual three-stem teacher checkpoint.
- Keep MUSDB four-stem experiments as a quality/generalization track, not as a
  replacement for the DnR-style three-stem target.

## Recommended Execution Order

1. Train or import the `SFC-Locoformer-Lite+` three-stem teacher.
2. Fix or justify the RT+ strict-edge audit risks: dynamic `Slice` and rank<=3
   activation `MatMul`.
3. Add latent SFC-band distillation hooks if the teacher/student interfaces can
   expose matching compressed features cleanly.
4. Train `BandSFCNet-RT+` with and without distillation.
5. Run full stats and ONE verification for RT+.
6. Fill the benchmark and listening templates from local evaluation runs.
7. Add the ablation matrix around band mapping, residual head, and distillation.
8. Add true `Sparse U-Net Mel-SFC` if RT+ quality still leaves a large gap.
9. Add DolphinSFCNPU distillation as a second deployable candidate.
10. Only then invest in Prompted Asymmetric SFC unless unified prompted
   separation becomes a near-term product requirement.

## Bottom Line

The repo now implements the main skeleton recommended by the research notes,
especially the `BandSFCNet-RT+` edge-student path and the teacher-distillation
plumbing. The remaining work is mostly in four buckets:

1. Real training evidence and benchmark tables.
2. Completing the proposed loss, augmentation, ablation, and deployment
   validation workflow.
3. Implementing the larger missing architecture families: true Sparse U-Net
   Mel-SFC, Prompted Asymmetric SFC, early source-split SepReformer-style
   refinement, and targeted Mamba2 residual refinement.
4. Turning the TIGER/ONE failure lessons and the three-stem product target into
   reusable regression checks and result manifests.
