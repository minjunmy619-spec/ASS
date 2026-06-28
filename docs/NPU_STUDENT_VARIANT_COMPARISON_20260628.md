# NPU Student Variant Comparison - 2026-06-28

## Scope

This note compares the repo's deployable student candidates for the TV
speech/music/effects separation target.  Teacher/offline recipes are excluded
unless they also expose an online/NPU student core.

The numbers below use:

- live parameter/state measurements from the current checkout;
- current raw streaming ONNX exports where available, with `frames=1`,
  `n_chan=1`, and `freqs=512` unless the recipe naturally uses another core
  frequency count;
- prior ONE verifier documentation for families that need a family-specific
  export path or already have a verified simplified graph.

`raw nodes` and `verified nodes` are intentionally separated.  Raw ONNX is the
node pressure seen before simplification.  Verified nodes are from focused
ONNX/ONE verifier runs and are better evidence for deployability.

## Metrics Snapshot

| Family / representative recipe | Core | Params | fp16 state | Raw ONNX nodes | Verified nodes / ONE status | Main structure |
|---|---|---:|---:|---:|---|---|
| `tvconv-pyramid-npu...` | `TVConvPyramidNPUSeparator2D` | 2.06M | 180 KiB | 273 | raw export clean | Conv-only frequency pyramid, basic residual-SFX 2-mask head |
| `tvconv-pyramid-convgru-npu...` | `TVConvPyramidNPUSeparator2D` | 1.78M | 130 KiB | 268 | raw export clean | TVConv pyramid with ConvGRU bottleneck/cache |
| `tvconv-pyramid-convlstm-npu...` | `TVConvPyramidNPUSeparator2D` | 1.83M | 140 KiB | 269 | raw export clean | TVConv pyramid with ConvLSTM bottleneck/cache |
| `tvconv-pyramid-sfclite-query-npu...` | `TVConvPyramidNPUSeparator2D` | 2.06M core / 2.59M total | 180 KiB | 273 | raw export clean | TVConv plus learnable-query frequency preprocessing |
| `tvconv-pyramid-sourceaware-sfclite-convgru-npu...` | `TVConvPyramidNPUSeparator2D` | 3.72M core / 4.24M total | 130 KiB | 276 | raw export clean, prior ONE pass exists | Source-aware TVConv, learnable-query SFC-lite preprocessor, ConvGRU cache |
| `band-sfc-net-npu.rt-plus.2mask-residual-sfx...` | `BandSFCNetNPU` | 2.09M | 186 KiB | 1062 | base RT+ verified at 444 nodes; residual-SFX should be re-run | BandSFC RT+ with 2-mask speech/music prediction and residual SFX |
| `band-sfc-net-npu.adaptive-mel-loco-cnb.stable-soft-query.residual-sfx...` | `BandSFCNetNPU` | 2.85M | 185.6 KiB | 1514 | current raw export clean but node-heavy | Adaptive-mel/Loco-CNB BandSFC residual-SFX variant |
| `band-sfc-net-npu.balanced.soft-query...` | `BandSFCNetNPU` | 4.07M | 160 KiB | not remeasured | safe/balanced query variants passed prior verifier | Useful-capacity BandSFC soft-query tier |
| `band-sfc-net-npu.quality/rt-plus...` | `BandSFCNetNPU` | about 2.09M | 186 KiB | not all remeasured | prior ONE verifier pass for RT+ family | Quality/RT+ BandSFC cross-attention or soft-query tier |
| `dolphin-sfc-npu.slim-6m.distill...` | `DolphinSFCNPUSeparator` | 5.17M | 162 KiB | family exporter path | 314 nodes, ONE import/opt/quant/circle-verify pass | Dolphin-style multi-scale separator with SFC frequency compression |
| `edge-fusion-sfc-distilled.rt192k` | `EdgeFusionNPU2DPackedCoreAdapter` | 5.30M | 168.7 KiB | family exporter path | 168 nodes, ONE import/opt/quant/circle-verify pass | Low-node EdgeFusion stack with heavy 1x1 token-capacity bottleneck |
| `band-scnet-npu.rt192k-param6m` | `BandSCNetNPU` | 6.34M | 178 KiB | 648 | MLIR ops 3595, prior compile path measured | Conventional BandSCNet-style edge baseline |
| `band-scnet-npu.rt192k-param2m` | `BandSCNetNPU` | 2.31M | 190.9 KiB | 806 | MLIR skipped in prior table | Lower-capacity BandSCNet baseline, state very close to budget |
| `source-aware-melband-loco-cnb.tvfix-strong-nopool...` | `OnlineSourceAwareMelBandLocoCNBStudentSFC2D` | 3.37M | 173.25 KiB | 1718 | raw export clean but node-heavy | Source-aware melband Loco-CNB student, no pooled mixers |
| `source-aware-melband-loco-cnb.tvfix-capacity-pcen-normlite...` | `OnlineSourceAwareMelBandLocoCNBStudentSFC2D` | 6.47M | 174.25 KiB | 1265 | raw export clean but still large | Larger Loco-CNB with PCEN/norm-lite path |
| `source-aware-melband-roformer.student-npu...` | `OnlineSourceAwareMelBandStudentSFC2D` | 0.18M | 164 KiB | 993 | raw export clean | Very small source-aware melband student |
| `source-aware-melband-roformer.student-npu-strong...` | `OnlineSourceAwareMelBandStrongStudentSFC2D` | 1.38M | 186 KiB | 1834 | raw export clean but node-heavy | Stronger source-aware melband decoder/correction stack |
| `source-aware-melband-roformer.student-npu-strong-nodelite...` | `OnlineSourceAwareMelBandStrongStudentSFC2D` | 6.12M | 186 KiB | 1823 | raw export clean but not deploy-friendly | Capacity-up node-lite attempt; high initializer/cache pressure |
| `adaptive-mel-locoformer-lite-sfc...` | `OnlineAdaptiveMelLocoformerLiteSFC2D` | 2.50M | 120 KiB | 641 | prior simplified verifier pass around 256 nodes | Adaptive mel SFC router plus causal Locoformer-lite blocks |
| `source-aware-residual-sfc...` | `OnlineSourceAwareResidualSFC2D` | 2.87M | 170.5 KiB | 889 | prior verifier pass exists | Source-aware residual SFC with shared and per-source paths |
| `sfc-residual-refinement...` | `OnlineResidualRefinementSFC2D` | 2.45M | 144 KiB | 453 | 222 verified nodes, ONE pass | First-pass SFC plus residual correction head |
| `sparse-unet-mel-sfc...` | `SparseUNetMelSFC2D` | 2.12M | 140 KiB | 952 | 555 verified nodes, ONE pass | Low/mid/high sparse mel U-Net SFC fallback |
| `prompted-asymmetric-sfc...` | `OnlinePromptedAsymmetricSFC2D` | 2.46M | 96 KiB | 861 | prior verifier pass exists | Static source prompts, asymmetric encoder/decoder |
| `bandmap-ablation.mel-overlap80...` | `OnlineAdaptiveMelSFC2D` | 2.46M | 90 KiB | 459 | 208 verified nodes, ONE pass | Mel-overlap band-map ablation, little source-specific modeling |
| `online-soft-band-query-sfc2d.rt192k...` | `OnlineSoftBandQuerySFC2D` | 0.031M | 120 KiB | 352 | legacy raw graph includes `Tile`/`Expand`/`ConstantOfShape` | Tiny legacy soft-query SFC baseline |
| `online-soft-band-query-dilated-sfc2d.param1p5m...` | `OnlineSoftBandQueryDilatedSFC2D` | 1.51M | 128 KiB | 664 | raw export clean | Scaled legacy soft-query/dilated SFC baseline |
| `online-hierarchical-soft-band-parallel-ffi-sfc2d.rt192k...` | `OnlineHierarchicalSoftBandParallelFFISFC2D` | 0.023M | 176.3 KiB | not remeasured | 252 verified nodes, ONE pass | Tiny hierarchical FFI/SFC baseline |
| `tiger-npu-edge-v2...` | `TIGERNPUEdgeV2` | 4.26M | family-specific | generic exporter path failed here | prior TIGER compile path passes | TIGER-style streaming cell and static reshape/slice discipline |
| `tf-mlpnet-edge...` | `TIGEREdgeMLP` | 0.71M | family-specific | generic exporter path failed here | useful as export discipline only | TF-MLP style separator under TIGER wrapper |

## Family-Level Assessment

### TVConv Pyramid

Strong points:

- Best current raw node profile among newly implemented variants: 268-276 nodes.
- Uses NPU-friendly Conv/ConvTranspose/elementwise structure with no MatMul.
- ConvGRU source-aware SFC-lite variant has the best state margin: 130 KiB.
- The source-aware SFC-lite ConvGRU recipe lands in the desired useful capacity
  range at 4.24M total parameters.

Weak points:

- Quality risk is real.  A mostly convolutional separator may not learn enough
  long-range source structure for hard speech/music/SFX interference.
- The SFC-lite learnable-query preprocessor currently contributes wrapper-side
  parameters.  For final deployment, that path must either be exported as part
  of the preprocessing graph or frozen into an external preprocessing stage.
- Base/GRU/LSTM variants predict only speech/music explicitly and use residual
  SFX.  This is good for node count, but it can turn SFX into an error bucket.

Verdict: best strict-NPU candidate.  Train it, but do not assume it is the
quality winner until it beats Dolphin/BandSFC on real validation.

### DolphinSFCNPU

Strong points:

- Very good deployable envelope: 5.17M parameters, 162 KiB state, 314 verified
  simplified ONNX nodes, and prior ONE import/opt/quant/circle-verify pass.
- Better quality hypothesis than pure conv-only models because it keeps
  multi-scale separation and SFC-style frequency compression.
- State budget is healthier than RT+ and Loco-CNB while still using meaningful
  capacity.

Weak points:

- Needs real TV-profile training and teacher distillation evidence.
- Uses a family-specific exporter path; keep the existing forced-simplification
  verifier in the loop before committing to long training.
- `slim_8m` is now budget-tightened, but should not be trained before `slim_6m`
  proves useful.

Verdict: best high-quality deployable candidate to train first or in parallel
with the TVConv source-aware GRU.

### BandSFCNetNPU

Strong points:

- Best aligned with the repo's SFC research direction and previous evidence:
  adaptive SFC compression, band-aware separation, and compact state.
- RT+ is around 2.09M parameters with 186 KiB fp16 state.
- Balanced query tier gives a 4.07M useful-capacity option at 160 KiB state.

Weak points:

- Current residual-SFX raw export has 1062 nodes.  The stable adaptive-mel
  residual-SFX raw export reaches 1514 nodes.
- 186 KiB state leaves little integration margin under the 192 KiB target.
- Existing safe variants are too small, while older quality6m variants exceed
  state/parameter budget.

Verdict: keep as quality anchor and distillation baseline, but it is not the
lowest-risk product graph unless the residual-SFX export is simplified and
re-verified through ONE.

### BandSCNetNPU

Strong points:

- `rt192k_param6m` is a strong conventional baseline: 6.34M parameters,
  178 KiB state, 648 ONNX nodes, and about 2.15 GMAC/s in the prior review.
- This is the cleanest way to test whether a larger band-split student beats
  newer custom structures under the same TV data and loss.

Weak points:

- Less source-aware and less adaptive than SFC-style students.
- Full graph I/O with high native frequency count still needs careful memory
  accounting, even when persistent state fits.
- `rt192k_param2m` is too close to the state limit and has more nodes than the
  larger param6m preset.

Verdict: train as a baseline race, not as the only product candidate.

### Loco-CNB Source-Aware Students

Strong points:

- Explicitly source-aware, causal, and cache-bounded.
- PCEN/norm-lite variant reduces raw nodes from 1718 to 1265 while increasing
  capacity to 6.47M.
- No pooled capacity mixer in the corrected variants.

Weak points:

- You already observed poor separation quality without distillation.
- Raw node count is still too high for a comfortable TV NPU path.
- Most temporal modeling happens after aggressive melband compression, so
  high-frequency detail and source competition can be weak.
- More parameters did not automatically fix the quality issue; the failure is
  likely structural plus training/objective, not only capacity.

Verdict: do not train first.  Use it only as a diagnostic after the stronger
students establish a reference.

### Source-Aware MelBand RoFormer Students

Strong points:

- Good teacher-alignment story and explicit source-aware output structure.
- Base model is extremely small.

Weak points:

- Base model is under-capacity at only 0.18M parameters.
- Strong and nodelite variants are node-heavy: about 1830 raw ONNX nodes.
- Nodelite reaches 6.12M parameters but does not solve the graph-size problem.

Verdict: useful for distillation/logit-domain experiments, not a first training
priority for TV NPU deployment.

### EdgeFusion

Strong points:

- Lowest verified graph size in the current docs: 168 simplified ONNX nodes.
- 5.30M parameters and 168.7 KiB state are inside the target envelope.
- Very attractive if the hardware rejects higher-node candidates.

Weak points:

- Much of the capacity is spent in repeated 1x1 token-capacity layers.
- The native 257-bin core may sacrifice detail unless the frequency
  preprocessing/postprocessing path is carefully tuned.
- Quality is unproven compared with DolphinSFC/BandSFC/TVConv.

Verdict: train after DolphinSFC and source-aware TVConv unless the NPU node
budget becomes the dominant blocker.

### Older SFC Baselines And Ablations

Strong points:

- Residual refinement, bandmap, prompted-asymmetric, SepReformer-multistem, and
  sparse U-Net variants are useful ablations with prior verifier coverage.
- Some have excellent state budgets: bandmap 90 KiB, prompted 96 KiB, residual
  refinement 144 KiB.

Weak points:

- They are mostly structure ablations, not the strongest quality hypotheses.
- Legacy soft-query tiny variants are too small for high-quality TV separation.
- Some legacy raw exports still show shape-materialization ops such as
  `Tile`, `Expand`, and `ConstantOfShape`.

Verdict: keep for regression, ablation, and sanity checks.  Do not spend the
first long TV training run here.

### TIGER / TF-MLPNet

Strong points:

- Useful history for static reshape, streaming cache, and compiler-safe
  sequence processing.
- TIGER edge-v2 has a known passing family-specific compile path.

Weak points:

- Not the current quality frontier for this repo.
- Generic online export path is not as smooth as the SFC/TVConv/Dolphin paths.
- TF-MLPNet edge is much too small for the target quality level.

Verdict: engineering reference only.

## Priority Training List

### Priority 1: DolphinSFCNPU slim-6m distillation

Recipe:

```text
recipes/dnr/models/dolphin-sfc-npu.slim-6m.distill-mixsoftmax.rt192k.fp512keep475/config.yaml
```

Fallback equivalent if the mix-softmax repair is not desired:

```text
recipes/dnr/models/dolphin-sfc-npu.slim-6m.distill.rt192k.fp512keep475/config.yaml
```

Why first:

- Best quality/deployability balance in the repo today: 5.17M params,
  162 KiB state, 314 verified nodes, ONE pass.
- More likely than pure TVConv to learn difficult source interactions.
- Low enough node count that success would be realistic for TV NPU.

Training goal: TV on-the-fly profile, robust supervised loss, teacher waveform
and mask/logit distillation, and speech-weighted validation.

### Priority 2: TVConv source-aware SFC-lite ConvGRU

Recipe:

```text
recipes/dnr/models/tvconv-pyramid-sourceaware-sfclite-convgru-npu.speech-music-residual-sfx.robust-distill.rt192k.fp512keep475/config.yaml
```

Why second, or parallel with Priority 1:

- Best strict-NPU graph among serious students: 276 raw nodes, 130 KiB state,
  4.24M total params.
- Source-aware head plus ConvGRU gives it more modeling power than the basic
  TVConv variants without exploding graph size.
- If this trains well, it is the cleanest deployment candidate.

Precondition before product export: decide whether the learnable-query
preprocessor is exported, frozen, or moved outside the NPU graph.

### Priority 3: BandSFC RT+ 2-mask residual-SFX

Recipe:

```text
recipes/dnr/models/band-sfc-net-npu.rt-plus.2mask-residual-sfx.rt192k.fp512/config.yaml
```

Why:

- Best SFC-aligned local anchor and important reference for whether the
  data/loss stack is working.
- Good parameter count at 2.09M, but it may be under-capacity for the target.
- Raw residual-SFX graph needs another ONE verification pass before product
  training becomes expensive.

Train this as a reference and distillation baseline, not necessarily as the
final low-node deployment graph.

### Priority 4: BandSCNetNPU rt192k_param6m

Recipe:

```text
recipes/dnr/models/band-scnet-npu.rt192k-param6m/config.yaml
```

Why:

- Strong conventional baseline: 6.34M params, 178 KiB state, 648 ONNX nodes,
  2.15 GMAC/s.
- Gives a clean answer to whether the custom SFC/TVConv students are actually
  better than a larger band-split baseline under the same TV data.

### Priority 5: EdgeFusion-SFC distilled

Recipe:

```text
recipes/dnr/models/edge-fusion-sfc-distilled.rt192k/config.yaml
```

Why:

- Lowest verified node count, 168 nodes.
- Keep it as the strictest graph-size fallback after the better quality
  hypotheses are measured.

### Priority 6: Loco-CNB PCEN norm-lite capacity

Recipe:

```text
recipes/dnr/models/source-aware-melband-loco-cnb.tvfix-capacity-pcen-normlite.sup.rt192k.fp512keep475/config.yaml
```

Why not higher:

- You already saw weak quality from the Loco-CNB TV family.
- PCEN/norm-lite helps graph size, but the raw graph is still 1265 nodes.
- This should be a diagnostic retrain only after the top candidates set the
  quality bar.

## Variants To Deprioritize For Now

- RoFormer strong / nodelite students: too many nodes for the deployment target.
- TVConv base, GRU, LSTM without source-aware SFC-lite: useful ablations, but
  weaker than the source-aware ConvGRU candidate.
- BandSFC safe variants: compile smoke tests, too small for quality.
- BandSFC quality6m variants: over budget in prior notes.
- Legacy tiny online SFC variants: good regression tests, not enough capacity.
- TIGER/TF-MLPNet: keep for compiler-pattern reference rather than first-line
  quality training.

## Recommended First Experiment Matrix

Run these with the same TV on-the-fly profile, same validation set, and same
metrics manifest:

| Run | Recipe | Purpose |
|---|---|---|
| A | `dolphin-sfc-npu.slim-6m.distill-mixsoftmax.rt192k.fp512keep475` | quality-first deployable student |
| B | `tvconv-pyramid-sourceaware-sfclite-convgru-npu.speech-music-residual-sfx.robust-distill.rt192k.fp512keep475` | low-node product graph candidate |
| C | `band-sfc-net-npu.rt-plus.2mask-residual-sfx.rt192k.fp512` | SFC quality anchor and residual-SFX reference |
| D | `band-scnet-npu.rt192k-param6m` | conventional strong baseline |

Minimum reporting fields:

- speech/music/SFX SDR or SI-SDR, plus speech leakage score;
- subjective speech artifacts on football/commentary and live-concert clips;
- residual/mixture-consistency error;
- params, fp16 state, raw ONNX nodes, simplified nodes, ONE import/opt/quant,
  and `circle-verify`;
- runtime memory including all graph inputs/outputs, not only persistent cache.

