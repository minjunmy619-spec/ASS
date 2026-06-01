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
- `BandSFCNet-RT+` 2-mask residual-SFX ablation, where the core predicts Speech
  and Music and the wrapper reconstructs Effects as `mixture - speech - music`.
- Proposal builders for:
  - `SFC-Locoformer-Lite+`
  - `Adaptive-Mel-SFC-Locoformer-Lite`
  - `BandSFCNet-RT+`
  - `Hierarchical-SFC-FFI-Lite`
  - `Sparse U-Net Mel-SFC`
  - `Prompted Asymmetric SFC`
  - `SFC-SepReformer-MultiStem`
  - `SFC-Residual-Refinement`
  - `Adaptive-Mel-SFC`
  - `EdgeFusion-SFC-Distilled`
- Teacher-student training plumbing through `TeacherStudentDistillationTask`.
- DnR and MUSDB sibling recipes for the RT+ student path.
- Opt-in DnR wrapper ablations for PCEN, DC-bypass, and 2-mask residual-SFX.
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
  adaptive-mel Locoformer-lite student, prompted asymmetric SFC, middle/edge
  proposal builders, and the distillation-task fail-fast behavior.
- NPU-friendly capacity mixers on the current DnR proposal recipes so the main
  candidate probes now sit inside the requested `2-7M` parameter range without
  increasing streaming cache size.
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
- `Adaptive-Mel-SFC-Locoformer-Lite` now has a strict online recipe and ONNX/ONE
  validation, but no trained metric table yet.
- The 2-mask residual-SFX BandSFC RT+ ablation has ONNX/ONE validation but no
  trained per-stem evidence yet; SFX must be checked carefully because it is the
  residual error bucket.
- `EdgeFusion-SFC-Distilled` has a distillation recipe but no trained student.
- No MUSDB/DnR SDR, SNR, SI-SDR, cSDR, or uSDR results are reported for the new
  proposal variants.

Required next work:

- Train or import a small SFC-Locoformer teacher.
- Train BandSFC RT+ with and without distillation.
- Train the 2-mask residual-SFX ablation only as a controlled comparison against
  the normal 3-mask RT+ recipe.
- Train EdgeFusion-SFC only after the BandSFC student is useful.
- Add a results table for quality, state, compute, and export status.

### 2. Full Loss Stack Is Not Implemented

Status: Implemented for loss plumbing; pending trained/tuned weights

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
- Teacher spectral mask distillation.
- Teacher spectral logit distillation.
- Latent/intermediate feature distillation through aux outputs or forward hooks.
- Source-activity-aware waveform loss weighting.

Missing:

- Trained evidence for the full loss stack.
- Chosen latent hook pairs for each final teacher/student architecture.
- Tuned loss weights per dataset and model family.

Required next work:

- Run ablations for mask/logit/source-activity terms.
- Enable non-zero latent distillation only after matching teacher/student feature
  hooks are selected and shape-compatible.

### 3. True Sparse U-Net Mel-SFC

Status: Implemented; DnR ONNX/ONE validated; pending training and MUSDB export

The notes propose `Sparse U-Net Mel-SFC` as a music-first fallback with:

- Sparse low/mid/high-band branches.
- Asymmetric encoder/decoder depth.
- U-Net skip paths.
- Lightweight TF-Locoformer or local/global bottleneck blocks.
- Shared complex mask head.

Implemented:

- Added `SparseBandUNetEncoder` and `SparseBandUNetDecoder` in
  `spectral_feature_compression/core/model/sparse_unet_mel_sfc_2d.py`.
- Added branch-local sparse low/mid/high overlapped mel routing through
  `RegionalMelBandSpec2d`.
- Added asymmetric encoder/downsample/bottleneck/decoder branch processing with
  U-Net skip paths.
- Used lightweight causal ConvNeXt-style 2D blocks as the bottleneck option, so
  tensors remain 4D and the operator set stays close to the online SFC variants.
- Added a shared packed-complex mask head and online waveform builder through
  `build_sparse_unet_mel_sfc_music_system`.
- Added MUSDB-first and DnR sibling recipes:
  - `recipes/musdb18hq/models/sparse-unet-mel-sfc.music.rt192k.fp512keep475/config.yaml`
  - `recipes/dnr/models/sparse-unet-mel-sfc.rt192k.fp512keep475/config.yaml`
- Added a smoke test covering waveform builder output shape, packed-core
  streaming shape, and a tiny fp16 state budget.

Current caveats:

- This is still a quality/ablation fallback, not a validated replacement for
  the strict RT+ NPU student.
- No trained MUSDB or DnR metrics exist yet.
- DnR packed-core export now passes ONNX export, ONE import, ONE optimize,
  ONE quantize, and `circle-verify`; the capacity-updated DnR core has about
  `2.12M` parameters, `140 KiB` fp16 state, and a `576`-node simplified ONNX
  graph after the no-`Max` expander-denominator rewrite.
- No trained MUSDB or DnR metrics, MUSDB export result, or GMAC/s table exists
  yet for this variant.

Required next work:

- Train the MUSDB recipe first and compare against existing soft-band SFC
  recipes.
- Run the DnR sibling only if the music-first probe is useful.
- Run the MUSDB sibling export if the music-first probe becomes useful.

### 4. Prompted Asymmetric SFC

Status: Implemented for fixed-output online DnR core; DnR ONNX/ONE validated; pending true prompt-batch training support and metrics

The notes propose a future unified model with prompt-conditioned outputs.

Implemented:

- Added `spectral_feature_compression/core/model/prompted_asymmetric_sfc_2d.py`.
- Added fixed static prompt embeddings for the configured output stems, with the
  default DnR labels `speech`, `music`, and `effects`.
- Added `PromptConditioner2d`, `PromptedTokenSplitter2d`,
  `PromptedSharedRefiner2d`, `PromptedCrossSourceMixer2d`, and
  `PromptedSharedDecoder2d`.
- Added `OnlinePromptedAsymmetricSFC2D`, which uses SFC query compression,
  a deeper shared causal encoder body, and a shallower prompt-conditioned shared
  decoder/head.
- Kept tensors 4D by packing fixed prompted outputs into channels and by using
  static Python loops over the configured prompts.
- Kept the exported ONNX/NPU core input contract as one packed complex STFT input
  `[B, 2*M, T, F]`; static prompts are model parameters, not extra ONNX inputs.
- Added optional external prompt embeddings for PyTorch experiments, but the
  deployable recipe uses static prompts for compiler stability.
- Added recipe:
  `recipes/dnr/models/prompted-asymmetric-sfc.rt192k.fp512keep475/config.yaml`.
- Added prompt metadata to ONNX deploy manifests and streaming run manifests.
- Added smoke tests for waveform builder output, prompt manifest, external prompt
  embedding path, full-vs-streaming equality, and fp16 state budget.

Current caveats:

- This is a fixed-output prompted student for the three-stem DnR product path;
  it is not yet a dynamic prompt-input ONNX contract.
- Prompt dropout training is not implemented yet.
- Task/source prompt taxonomy beyond the fixed DnR labels is not pinned.
- Category-aware PIT for multiple same-class outputs is not implemented.
- No trained DnR or multi-task metrics exist yet.
- The default DnR packed core has about `2.46M` parameters and `96 KiB` fp16
  layer-cache state at `512` preprocessed bins after capacity mixing.
- DnR packed-core export passes ONNX export, ONNX simplification, calibration
  dataset generation, ONE import, ONE optimize, ONE quantize, and direct
  `circle-verify`; the simplified ONNX graph has `456` nodes.

Required next work:

- Train the fixed-output DnR prompted recipe against BandSFC RT+ and SFC-SepReformer.
- Define prompt IDs and source taxonomy for non-DnR tasks before adding dynamic
  prompt inputs to ONNX/ONE export.
- Add datamodule/task support for prompt dropout and prompt subset batches.
- Add category-aware PIT only where multiple same-class outputs exist.


### 5. SepReformer-Style Early Source Split

Status: Implemented; DnR ONNX/ONE validated; pending training and MUSDB export

The notes recommend testing early source disentanglement at compressed SFC token
level.

Implemented:

- Added `spectral_feature_compression/core/model/source_split_sfc_2d.py`.
- Added `SourceTokenSplitter2d` immediately after SFC soft-band query
  compression.
- Added `SharedSourceRefiner2d`, which applies the same causal 2D refiner blocks
  to each source token stream.
- Added `SourceSharedReconstructionDecoder2d`, which reuses one SFC query
  expander and one mask head across all sources.
- Added `CrossSourceReconstructionMixer2d`, which mixes each source token with
  other-source mean context and mixture-token context before reconstruction.
- Kept runtime tensors 4D by packing the fixed source axis into channels
  instead of using the conceptual `[B, N, D, T, K]` tensor from the research
  sketch.
- Exposed the proposal through `build_sfc_sepreformer_multistem_system`.
- Added sibling recipes:
  - `recipes/dnr/models/sfc-sepreformer-multistem.rt192k.fp512keep475/config.yaml`
  - `recipes/musdb18hq/models/sfc-sepreformer-multistem.rt192k.fp512keep475/config.yaml`
- Added a smoke test covering waveform builder output shape, source-split token
  shape, packed-core streaming shape, and a tiny fp16 state budget.

Current caveats:

- This is a source-disentanglement ablation/middle-tier candidate, not a proven
  strict NPU student.
- Static per-source loops export and compile for the DnR packed-core recipe; the
  capacity-updated DnR core has about `2.45M` parameters, the simplified ONNX
  graph has `435` nodes, and the quantized Circle passes `circle-verify`.
- No trained DnR or MUSDB results exist yet.

Required next work:

- Train the DnR recipe first because the source split should help the three-stem
  universal task most.
- Compare against `online-soft-band-query-sfc2d` with the same preprocessing and
  loss stack.
- Run the MUSDB sibling export if the DnR ablation is worth continuing.

### 6. Mamba2 Or Residual Refinement Branch

Status: Implemented; DnR ONNX/ONE validated; pending training and MUSDB export

The notes do not recommend replacing SFC-CA with Mamba by default. They do
recommend testing Mamba/Mamba2 selectively in:

- A long temporal branch after SFC compression.
- A second-stage residual correction/refinement branch.
- TS-BSMamba2-style correction.

Implemented:

- Added `spectral_feature_compression/core/model/residual_refinement_sfc_2d.py`.
- Added `Mamba2LiteTemporalBranch2d`, a causal dilated latent-band branch after
  SFC compression. It targets the long-context ablation role of Mamba2 without
  importing unsupported Mamba2 kernels into the strict path.
- Added `ResidualCorrectionHead2d`, a second-stage full-band correction head that
  consumes the mixture, first estimate, and refined SFC token context.
- Added `OnlineResidualRefinementSFC2D` and wrapper/builder plumbing through
  `build_sfc_residual_refinement_system`.
- The branch predicts a packed complex residual added to the primary masked
  estimate, making it directly comparable with the existing BandSFC RT+
  residual head.
- Added recipes:
  - `recipes/dnr/models/sfc-residual-refinement.rt192k.fp512keep475/config.yaml`
  - `recipes/musdb18hq/models/sfc-residual-refinement.rt192k.fp512keep475/config.yaml`
- Added a smoke test covering waveform builder output shape, packed-core
  streaming shape, and trainable correction/long-branch scales.

Current caveats:

- This is not a real Mamba2 kernel implementation. Actual Mamba2 should remain a
  teacher/middle-tier ablation until export and runtime support are known.
- The default uses causal dilated 2D blocks so tensors stay 4D and the state is
  budgetable; at `512` preprocessed bins the default fp16 layer cache is about
  `144 KiB` and the capacity-updated DnR core has about `2.45M` parameters.
- No trained metric comparison against BandSFC RT+ exists yet.
- DnR packed-core export now passes ONNX export, ONE import, ONE optimize,
  ONE quantize, and `circle-verify`; the simplified ONNX graph has `282` nodes.

Required next work:

- Train the DnR residual-refinement recipe and compare against BandSFC RT+ and
  online soft-band query SFC with the same data/loss stack.
- Run the MUSDB sibling export if the DnR ablation is worth continuing.
- Only add true Mamba2 after a non-strict teacher/middle-tier experiment proves
  quality benefit worth the export risk.

### 7. Adaptive Mel / Overlapped Perceptual Band Mapping

Status: Implemented; DnR ONNX/ONE validated for fixed80, mel-overlap80, and the adaptive-mel Locoformer-lite student; pending trained ablation results

Implemented:

- Existing SFC cross-attention compression remains available.
- Existing `band_config: musical` recipes remain unchanged.
- Frequency preprocessing such as `fp512keep475` remains available.
- Added `spectral_feature_compression/core/model/adaptive_mel_sfc_2d.py`.
- Added `AdaptiveMelBandSpec2d`, an explicit overlapped mel basis with default
  `80` bands.
- Added low-frequency controls for bass/music preservation:
  - `low_freq_hz`
  - `low_freq_band_fraction`
  - `overlap_factor`
  - `low_freq_overlap_factor`
- Added `fixed` / `linear` / `uniform` band modes to `SoftBandSpec2d` so fixed
  linear bands can be compared directly against mel-overlap bands.
- Added `OnlineAdaptiveMelSFC2D` and proposal builder
  `build_adaptive_mel_sfc_ablation_system`.
- Added `spectral_feature_compression/core/model/adaptive_mel_locoformer_lite_sfc_2d.py`
  with `OnlineAdaptiveMelLocoformerLiteSFC2D`, which keeps adaptive mel SFC
  routing and uses alternating causal time, band, and pointwise gated
  Locoformer-lite blocks.
- Exposed the deployable Proposal-A student through
  `build_adaptive_mel_locoformer_lite_system`.
- Added sibling band-mapping ablation recipes:
  - `recipes/dnr/models/bandmap-ablation.fixed80.rt192k.fp512keep475/config.yaml`
  - `recipes/musdb18hq/models/bandmap-ablation.fixed80.rt192k.fp512keep475/config.yaml`
  - `recipes/dnr/models/bandmap-ablation.mel-overlap80.rt192k.fp512keep475/config.yaml`
  - `recipes/musdb18hq/models/bandmap-ablation.mel-overlap80.rt192k.fp512keep475/config.yaml`
  - `recipes/dnr/models/bandmap-ablation.sfc-ca80.teacher/config.yaml`
  - `recipes/musdb18hq/models/bandmap-ablation.sfc-ca80.teacher/config.yaml`
  - `recipes/musdb18hq/models/bandmap-ablation.sfc-mamba64.teacher/config.yaml`
- Added strict DnR student recipe:
  - `recipes/dnr/models/adaptive-mel-locoformer-lite-sfc.rt192k.fp512keep475/config.yaml`
- Added a smoke test checking the explicit mel basis, low-frequency overlap
  control, waveform builder output, packed-core streaming shape, and deploy-size
  fp16 state budget.

Current caveats:

- SFC-Mamba remains a non-strict teacher/offline ablation because it depends on
  `mamba_ssm` and is not an NPU-safe runtime path.
- No trained direct comparison exists yet for fixed80 vs mel-overlap80 vs SFC-CA
  vs SFC-Mamba.
- DnR fixed80 and mel-overlap80 packed-core exports both pass ONNX export,
  ONE import, ONE optimize, ONE quantize, and `circle-verify`; fixed80 remains
  the simple `208`-node control, while the capacity-updated mel-overlap80 core
  has about `2.46M` parameters and `268` simplified ONNX nodes.
- DnR adaptive-mel Locoformer-lite packed-core export passes ONNX export,
  ONNX simplification, calibration generation, ONE import, ONE optimize, ONE
  quantize, and direct `circle-verify`; the capacity-updated core has about
  `2.50M` parameters, `316` simplified ONNX nodes, and the default fp16 layer
  cache is about `120 KiB`.

Required next work:

- Train the adaptive-mel Locoformer-lite, fixed80, and mel-overlap80 DnR recipes
  under the same loss stack.
- Compare against the SFC-CA teacher and the existing SFC-Mamba teacher branch
  where dependencies are available.
- Fill the result manifest with quality, state, node-count, and export metrics.

### 8. Full Deployment Validation Is Not Complete

Status: Partial

Done:

- RT+ streaming ONNX export.
- ONNX checker.
- Forbidden-op audit for `Tile`, `Expand`, and `ConstantOfShape`.
- RT+ fp512 state check around `186 KiB` fp16 layer-cache.
- Full ONE import/optimize/quantize/`circle-verify` for RT+.
- DnR ONNX/ONE validation for new proposal candidates:
  - `band-sfc-net-npu.rt-plus.2mask-residual-sfx.rt192k.fp512`: `444`
    simplified ONNX nodes, `core_n_src=2`, wrapper output `n_src=3`.
  - `sparse-unet-mel-sfc.rt192k.fp512keep475`: `576` simplified ONNX nodes.
  - `sfc-sepreformer-multistem.rt192k.fp512keep475`: `435` simplified ONNX nodes.
  - `sfc-residual-refinement.rt192k.fp512keep475`: `282` simplified ONNX nodes.
  - `bandmap-ablation.fixed80.rt192k.fp512keep475`: `208` simplified ONNX nodes.
  - `bandmap-ablation.mel-overlap80.rt192k.fp512keep475`: `268` simplified ONNX nodes.
  - `adaptive-mel-locoformer-lite-sfc.rt192k.fp512keep475`: `316` simplified ONNX nodes.
  - `prompted-asymmetric-sfc.rt192k.fp512keep475`: `456` simplified ONNX nodes.
- DnR ONNX/ONE validation for remaining deployable candidates:
  - `dolphin-sfc-npu.large-6m.fp512keep475`: `314` simplified ONNX nodes.
  - `dolphin-sfc-npu.slim-6m.distill.rt192k.fp512keep475`: `314` simplified ONNX nodes.
  - `edge-fusion-sfc-distilled.rt192k`: `168` simplified ONNX nodes.
  - `online-hierarchical-soft-band-parallel-ffi-sfc2d.rt192k.speech-lowfreq-narrow.causal20dim.0-1-1l.128-96-48b`:
    `252` simplified ONNX nodes.

Missing:

- GMAC/s measurement for RT+.
- MLIR op count table.
- Runtime latency measurement.
- Listening notes for bass, drums/transients, vocals/speech leakage, and
  artifacts.

Required next work:

- Run `tools/online/measure_npu_model_stats.py`.
- Run `tools/online/export_verify_mlir.py` without `--skip-emit-mlir`.
- Extend the same validation to MUSDB sibling configs if the DnR probes justify
  continuing them.

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
  - supervised RT+ PCEN and DC-bypass wrapper ablations,
  - supervised RT+ 2-mask residual-SFX ablation,
  - chunk-causal RT+ distillation,
  - strict short-chunk RT+ distillation fine-tuning.
- `docs/AUDIO_SEPARATION_TRAINING_RECIPE.md` with stage order and launch
  commands.

Required next work:

- Run the staged recipes and fill the benchmark manifest with local metrics.
- Tune augmentation probabilities if listening tests show transient smearing or
  speech/music leakage.

### 11. Ablation Matrix Is Missing

Status: Partial

The notes request architecture-level ablations, not random hyperparameter
sweeps.

Implemented ablation rows now include:

- BandSFC RT+ 2-mask residual-SFX, with Speech/Music predicted explicitly and
  Effects reconstructed wrapper-side as the residual.
- Wrapper-side PCEN and DC-bypass for BandSFC RT+.
- Fixed80 vs mel-overlap80 band mapping.
- Sparse U-Net Mel-SFC, source-split SFC, and residual-refinement SFC proposal
  probes.

Still missing or pending-trained-evidence ablations:

- Fixed bands vs mel-overlap vs SFC-CA vs SFC-Mamba.
- CNB blocks vs Lite-Locoformer vs sparse U-Net bottleneck.
- Offline teacher vs chunk-causal student vs strict frame-streaming student.
- BandSFC safe vs quality vs RT+.
- BandSFC RT+ vs Hierarchical-SFC-FFI-Lite vs EdgeFusion-SFC.
- With and without teacher distillation.
- With and without low-frequency loss.
- With and without residual correction head.
- With and without 2-mask residual-SFX, measured per stem.

Required next work:

- Create sibling config groups for each ablation.
- Train/evaluate the sibling config groups already added.
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
- The audit has now been run across the current capacity-updated DnR proposal
  candidates.  The remaining recurring conservative flag is rank<=3 activation
  `MatMul`; sparse U-Net also required forced ONNX simplification after replacing
  the expander denominator `Max` pattern.
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
- Residual-source ablations are still scored as fixed DnR three-stem outputs;
  result rows should keep `n_src=3` and record `core_n_src=2` in notes.
- The teacher path is discussed for both MUSDB and DnR, but the exact checkpoint
  selection and conversion from four-stem music training to three-stem TV/CASS
  deployment is not defined.
- Prompted/unified future work beyond the fixed DnR `speech/music/effects`
  labels still depends on a broader source taxonomy.

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
7. Train the ablation matrix around band mapping, residual head, residual-SFX,
   and distillation.
8. Add true `Sparse U-Net Mel-SFC` if RT+ quality still leaves a large gap.
9. Add DolphinSFCNPU distillation as a second deployable candidate.
10. Train the fixed-output Prompted Asymmetric SFC DnR recipe only if unified
    prompted separation becomes a near-term product requirement.

## Bottom Line

The repo now implements the main skeleton recommended by the research notes,
especially the `BandSFCNet-RT+` edge-student path and the teacher-distillation
plumbing. The remaining work is mostly in four buckets:

1. Real training evidence and benchmark tables.
2. Completing the proposed loss, augmentation, ablation, and deployment
   validation workflow with trained metrics.
3. Filling empirical evidence for the implemented architecture probes; dynamic
   prompt-batch training and true Mamba2 kernels remain future work.
4. Turning the TIGER/ONE failure lessons and the three-stem product target into
   reusable regression checks and result manifests.
