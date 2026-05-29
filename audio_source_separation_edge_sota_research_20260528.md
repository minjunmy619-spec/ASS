# Edge-Deployable SOTA Audio Source Separation Research - 2026-05-28

This note is a deeper refresh of `audio_separtion_model_research.md`.  I kept
that earlier file intact and treated it as prior context.  The conclusion is
mostly consistent with the earlier note, but the evidence is tightened around
the current repo state and the 2026 SFC paper/source tables.

## Executive Conclusion

The strongest path is not a single model copied from one paper.  The best
quality/compute frontier is:

1. Use an **SFC-CA / TF-Locoformer quality teacher** as the main non-causal or
   chunked inference target.
2. Build a **BandSFCNet-RT+ edge student** by distilling that teacher into the
   current `BandSFCNetNPU` / `online-crossattn-query-sfc2d` style deployment
   family.
3. Keep **EdgeFusion-SFC Distilled** as the strictest NPU fallback where
   compile reliability matters more than absolute SDR.

Trying to make the current pure `BandSCNetNPU`, `TF-MLPNet`, or tiny
`EdgeFusionNPU` variants reach RoFormer/SFC-level quality only by tuning is
unlikely to work.  The quality gap is architectural: the current edge branches
mostly lack enough global frequency modeling, adaptive encoder/decoder
capacity, and teacher-guided training.

## What The Literature Says

### 1. 30+ year review: what still matters

The 2025 review "30+ Years of Source Separation Research" frames the modern
field around four recurring facts:

- Supervised DNN separation became dominant because it learns source priors
  directly from data.
- Modern SOTA systems usually use complex-spectrum estimation or waveform
  estimation rather than only magnitude masking.
- Dual-path or multi-path modeling is central: one path models local/time
  structure, another models spectral or global context.
- The field is moving toward realistic data, unknown/time-varying sources,
  real-world metrics, and lightweight low-latency edge deployment.

The practical takeaway for this repo: a winning model should still be
time-frequency based, complex-valued or complex-mask based, and multi-path.  It
should not collapse into a tiny causal Conv2d stack unless deployment is the
only objective.

Source:

- https://arxiv.org/html/2501.11837v1

### 2. Offline music separation frontier: RoFormer, SCNet, SFC, Moises-Light

**BS-RoFormer** remains the main offline music-separation reference.  It uses a
band-split module, hierarchical inner-band and inter-band Transformers, and
RoPE.  The paper reports 9.80 dB average SDR on MUSDB18-HQ without extra data
and won SDX23 when trained with additional data.

**Mel-Band RoFormer** improves the band design by replacing heuristic,
non-overlapping bands with mel-scale overlapped bands.  The useful idea for us
is not necessarily the exact model size; it is the frequency prior: low
frequency and perceptual overlap matter.

**SCNet** shows that sparse frequency compression is a serious efficiency
direction.  It uses low/mid/high subbands, stronger modeling where information
density is higher, and reports 9.0 dB average SDR on MUSDB18-HQ with lower
inference cost than HT Demucs.

**Moises-Light** is especially important for edge work.  It shows that a
carefully designed band-split U-Net plus RoPE bottleneck and better training
can approach or beat larger models.  Its table reports:

| Model | Avg SDR | Params |
|---|---:|---:|
| BS-RoFormer | 9.80 | 72M x 4 single-stem models in the table |
| SCNet-L | 9.69 | 42M |
| SCNet-S | 9.00 | 10M |
| Moises-Light proposed | 9.96 | 5M x 4 single-stem models |
| Moises-Light-S | 8.65 | 2M x 4 single-stem models |

Important Moises-Light design lessons:

- Replace dual-path RNN with RoPE Transformer blocks.
- Use group-conv band splitting to avoid huge per-band MLP cost.
- Apply split/merge ideas throughout the encoder/decoder, not only at the
  outermost layers.
- Use asymmetric encoder/decoder depth: heavier encoder, lighter decoder.
- Train with polarity inversion, pitch shift, temporal shift, channel flip, and
  multi-resolution complex spectrogram loss.
- Bass still benefits from higher frequency resolution and Transformer
  sequence modeling, so aggressive compression must be handled carefully.

Sources:

- https://arxiv.org/abs/2309.02612
- https://arxiv.org/abs/2310.01809
- https://arxiv.org/abs/2401.13276
- https://arxiv.org/html/2510.06785

### 3. SFC is the strongest code-adjacent evidence

The 2026 SFC paper is the most directly useful source for this repo because
this checkout already contains its implementation and many online adaptations.
Its core claim is precise:

- Band-split compression works, but it is not input-adaptive and uses distinct
  parameters per subband.
- SFC replaces the per-band encoder/decoder with one sequence-modeling
  compressor/decoder.
- SFC has two variants: cross-attention (`SFC-CA`) and Mamba (`SFC-Mamba`).
- Experiments show SFC outperforms the classic band-split module across
  separator sizes and compression ratios.

The source tables in the arXiv package show:

| ID | Model | Encoder/Decoder | Params | MUSDB avg cSDR | MUSDB avg uSDR |
|---|---|---|---:|---:|---:|
| A1 | BS-Locoformer small | Band-split 64 | 34.7M | 8.72 | 8.26 |
| A2 | SFC-Locoformer small | SFC-Mamba 64 | 5.1M | 8.86 | 8.63 |
| A3 | SFC-Locoformer small | SFC-CA 64 | 5.8M | 9.27 | 8.95 |
| B1 | BS-Locoformer medium | Band-split 64 | 55.5M | 9.42 | 8.79 |
| B2 | SFC-Locoformer medium | SFC-Mamba 64 | 15.2M | 9.56 | 9.13 |
| B3 | SFC-Locoformer medium | SFC-CA 64 | 16.0M | 9.95 | 9.38 |

On DnR:

| ID | Model | Encoder/Decoder | Params | DnR avg SNR | DnR avg SI-SDR |
|---|---|---|---:|---:|---:|
| C1 | BS-Locoformer small | Band-split 64 | 34.7M | 11.40 | 10.96 |
| C3 | SFC-Locoformer small | SFC-CA 64 | 5.8M | 11.78 | 11.38 |
| D1 | BS-Locoformer medium | Band-split 64 | 55.5M | 11.93 | 11.57 |
| D3 | SFC-Locoformer medium | SFC-CA 64 | 16.0M | 12.21 | 11.83 |

The training/inference cost table matters too:

| ID | Size | Enc/Dec | Params | FLOPs G/s | RTF |
|---|---|---|---:|---:|---:|
| J1 | Small | BS | 34.7M | 36.49 | 0.0017 |
| J2 | Small | SFC-Mamba | 5.1M | 40.14 | 0.0025 |
| J3 | Small | SFC-CA | 5.8M | 41.04 | 0.0018 |
| J4 | Medium | BS | 55.5M | 100.06 | 0.0031 |
| J6 | Medium | SFC-CA | 16.0M | 110.37 | 0.0035 |

Interpretation:

- `SFC-CA` is the highest-value encoder/decoder idea in this repo.
- The 5.8M small SFC-CA teacher is a much better first quality target than any
  tiny edge-only variant.
- The compute is still too high for the strictest TV-class NPU target, so it
  should be the teacher or high-quality chunked mode, not necessarily the final
  deployed cell.
- `SFC-Mamba` is useful but slower and lower quality than `SFC-CA` in the
  reported setting, so Mamba should be used selectively for temporal memory,
  not as the default replacement for attention.

Sources:

- https://arxiv.org/abs/2602.08671
- Local verified tables from `/tmp/sfc_src/tables/main_results_musdb.tex`
- Local verified tables from `/tmp/sfc_src/tables/main_results_dnr.tex`
- Local verified tables from `/tmp/sfc_src/tables/train_inf_costs.tex`

### 4. Speech separation SOTA lessons

**TF-GridNet** remains a key architecture lesson: each block combines
intra-frame spectral modeling, sub-band temporal modeling, and full-band
self-attention.  This is almost the shape we want, but the full model is not an
edge model.

**TF-Locoformer** removes RNNs from TF dual-path models and uses convolutional
FFNs for local modeling so attention can focus on global patterns.  This is the
best separator block to pair with SFC for the quality teacher.

**SepReformer** is important because it performs source separation earlier in
the network and reconstructs with a weight-shared decoder.  The useful idea for
this repo is early source disentanglement plus shared reconstruction, which can
reduce the burden on the final mask head.

**FLASepformer** is a useful efficiency direction: focused linear attention can
match SOTA with lower memory and faster inference.  It is more relevant for the
teacher or middle-tier student than for the strict NPU path, because custom
linear attention must still be lowered safely.

Sources:

- https://arxiv.org/abs/2209.03952
- https://arxiv.org/abs/2408.03440
- https://arxiv.org/abs/2406.05983
- https://arxiv.org/abs/2508.19528

### 5. Mamba line: promising, but not a blanket answer

Mamba and Mamba2 papers are attractive because they offer long-context sequence
modeling with recurrent-style state.  The music separation papers show real
promise:

- TS-BSMamba2 uses a two-stage band-split Mamba2 network with residual mapping.
- Mamba2 Meets Silence reports strong vocal-separation results and argues that
  Mamba2 helps with sparse vocal regions and varying input lengths.
- U-Mamba-Net shows a lightweight Mamba/U-Net speech separation direction.

However, the 2026 SFC paper is a direct warning: in that code path, SFC-Mamba
was lower quality and slower than SFC-CA despite lower nominal FLOPs.  For this
repo, Mamba2 should be tested in two targeted places:

- long temporal branch after SFC compression, where sequence length is small;
- second-stage residual/refinement branch, where it can correct mask artifacts.

It should not replace all cross-attention compression by default.

Sources:

- https://arxiv.org/abs/2409.06245
- https://arxiv.org/abs/2508.14556
- https://arxiv.org/abs/2412.18217

### 6. Foundation/prompted separation is not the edge target

AudioSep and SAM Audio are important for general-source and promptable
separation, but their design point is different:

- AudioSep separates open-domain sounds using natural language queries.
- SAM Audio unifies text, visual, and temporal-span prompts with a diffusion
  transformer and reports strong broad benchmarks.

These are useful teacher/benchmark references for future universal source
separation, but they are too heavy and too prompt-conditioned for the current
moderate-parameter realtime edge objective.

Sources:

- https://arxiv.org/abs/2308.05037
- https://arxiv.org/abs/2512.18099

## Current Repo Assessment

### `spectral_feature_compression/`

Strengths:

- Contains the official SFC encoder/decoder concepts:
  `CrossAttnEncoder`, `CrossAttnDecoder`, `MambaEncoder`, `MambaDecoder`.
- Contains `BSLocoformer`, which is the right quality separator backbone.
- Contains online 2D NPU variants:
  `online_crossattn_query_sfc_2d.py`, `online_soft_band_query_sfc_2d.py`,
  hierarchical soft-band, FFI, parallel FFI, dilated, and GRU-like variants.
- Tooling exists for state reporting, ONNX export, op audit, MLIR export, and
  GMAC measurement.

Weaknesses:

- The full online default configs with `d_model=96` and `n_layers=12` exceed a
  192 KiB state budget.
- Existing legacy log
  `logs/one_compile_soft_band_query_legacy/npu_model_stats.json` shows a
  769k-param online soft-band-query model at 8.83 GMAC/s and 960 KiB fp16
  state, with forbidden ops (`Tile`, `Expand`, `ConstantOfShape`).  That
  legacy graph is not the deployment target.
- The quality teacher exists as code/recipes, but no local pretrained weights
  are present except the download script.

Best use:

- Quality teacher: `locoformer-small/medium + SFC-CA`.
- Edge student components: `online-crossattn-query-sfc2d` and
  `online-soft-band-query-sfc2d` under `rt192k`/`rt128k` recipes.

### `BandSCNetNPU/`

Strengths:

- Closest local implementation to the published Band-SCNet realtime idea.
- Uses sparse down/up bands, cross-band and narrow-band blocks, bounded
  causal attention, and real deployment checks.
- Recent docs report candidate budgets at 44.1 kHz, hop 512:

| Preset | Params | State fp16 | GMAC/s | ONNX nodes |
|---|---:|---:|---:|---:|
| edge_small | 10,411 | 108.75 KiB | 0.2055 | 460 |
| rt192k | 62,115 | 190.88 KiB | 1.2588 | 716 |
| rt192k_plus | 72,915 | 178.00 KiB | 1.6178 | 588 |
| rt192k_param2m | 2,311,059 | 190.88 KiB | 1.4493 | 806 |
| rt192k_param6m | 6,340,019 | 178.00 KiB | 2.1513 | 648 |

Weaknesses:

- The published Band-SCNet number is 7.79 dB SDR, 2.59M params, and 92 ms
  latency.  Good realtime, but far from SFC/RoFormer offline quality.
- The local branch is deployability-heavy; it still needs real trained
  checkpoints and objective SDR tables.
- Static low/mid/high sparse compression is weaker than input-adaptive SFC.

Best use:

- Good hard-realtime baseline and student backbone.
- Do not use as the final quality frontier without SFC-style adaptive
  compression and teacher distillation.

Source:

- https://www.isca-archive.org/interspeech_2025/yang25d_interspeech.html

### `BandSFCNetNPU/`

This is currently the best local edge candidate.

Strengths:

- Combines SFC-style frequency transport with BandSCNet cross-band/narrow-band
  stages.
- Presets already separate deployment-first and quality variants:
  `safe`, `quality`, `quality6m`.
- README reports successful ONE path for:

| Recipe | ONNX nodes | Params | State fp16 | ONE result |
|---|---:|---:|---:|---|
| safe rt192k fp513 | 518 | 442,251 | 128.12 KiB | import/opt/quant PASS |
| quality rt192k fp512 | 1,074 | 2,092,715 | 186.00 KiB | import/opt/quant PASS |

Weaknesses:

- Not yet a published-quality architecture; it is a promising integration.
- The quality preset still needs trained SDR and artifact listening tests.
- Cross-attention transport must keep avoiding bad MatMul/FC lowering in ONE.

Best use:

- Main edge student architecture.
- The first serious experiment should be `quality.rt192k.fp512`, trained with
  teacher distillation from SFC-Locoformer.

### `DolphinSFCNPU/`

Strengths:

- Good U-Net style multi-scale architecture.
- Strong deployment cleanup: one temporal cache per block, stateless frequency
  compression, packed state, and presets in 3.6M to 7.7M range under 192 KiB.
- Good source-prior idea folded into a pointwise gate.

Weaknesses:

- Audio-only Dolphin adaptation loses the original visual/target conditioning
  advantage.
- It likely needs stronger global frequency modeling to match SFC-CA or
  RoFormer-class separation.
- The current simplifications are likely good for deployability but may damage
  high-frequency and transient quality unless trained with a strong teacher.

Best use:

- Middle-tier multi-scale student or ablation against BandSFCNet.
- Could contribute its slim multi-scale U-shape to a future `SFC-UNet-RT`.

### `EdgeFusionNPU/`

Strengths:

- Cleanest deployment contract: `(x, state) -> (mask, next_state)`.
- Strong NPU discipline: no STFT/iSTFT in graph, packed state, static shapes,
  simple ops, low-frequency bias, optional band bottleneck and token capacity.
- Good place to ship the first strict product-like model.

Weaknesses:

- It is explicitly not a trained checkpoint.
- Tiny/compact variants are likely too small for SOTA-quality separation.
- The architecture is a scaffold.  It needs a teacher and a real training plan.

Best use:

- Final constrained NPU student after a stronger BandSFCNet or SFC teacher is
  trained.

### `TIGER/` and `TF-MLPNet/`

Strengths:

- Useful streaming-cell pattern: one-frame deployable cell plus long-sequence
  training wrapper.
- Good lessons around cache I/O, windowing, avoiding dynamic Slice/Gather, and
  keeping training/deployment consistent.
- TF-MLPNet is useful as a tiny edge lesson: Conv2d/MLP-like mixers, causal
  time, static 4D activations.

Weaknesses:

- Not the best quality direction for music/source separation in this repo.
- TIGER attention and subband logic can become graph-heavy or brittle.
- TF-MLPNet is speech-first and likely underpowered for high-fidelity music or
  DnR unless used as a distilled low-tier student.

Best use:

- Reuse streaming-cell discipline and cache contract.
- Do not use as the main SOTA candidate.

## Proposed Model Structures

### Proposal A: `SFC-Locoformer-Lite+` quality teacher

Goal:

- Best chance of SOTA-like quality with moderate parameters.
- Not the strictest edge NPU model; this is the teacher and quality ceiling.

Keep:

- SFC-CA encoder/decoder from `spectral_feature_compression/core/model/crossattn_enc_dec.py`.
- TF-Locoformer separator from `spectral_feature_compression/core/model/bslocoformer.py`.
- Moises-Light training recipe: stronger augmentation and multi-resolution
  complex spectrogram loss.
- Optional SepRe-style early source split at the compressed SFC token level.

Discard or constrain:

- Full BS-RoFormer size and per-stem huge models.
- Full-sequence time attention for deployment.  Use chunked inference and
  window overlap; optionally use local/linear time attention.
- Mamba as default SFC compressor.  Keep it only as an ablation or residual
  temporal branch.

High-level architecture:

```text
complex STFT [B, M, F, T]
  -> SFC-CA encoder: F -> K, K=64 or 80, learnable query
  -> optional SepRe source-token split: [B, D, K, T] -> [B, N, D/N, K, T]
  -> 4 to 6 TF-Locoformer blocks
       frequency path: global attention over K bands
       time path: local/windowed attention or convolutional FFN
       FFN: SwiGLU Conv1d / depthwise temporal conv
  -> SFC-CA decoder: K -> F with encoder side query
  -> complex mask or complex residual
  -> mixture consistency projection
```

Implementation entry point:

```python
from spectral_feature_compression import CrossAttnEncoder, CrossAttnDecoder
from spectral_feature_compression import BSLocoformer


def build_sfc_locoformer_lite_plus(
    *,
    n_src: int,
    n_chan: int,
    sample_rate: int = 44100,
    n_fft: int = 2048,
    n_bands: int = 64,
    d_inner: int = 64,
    d_model: int = 96,
    n_layers: int = 4,
):
    encoder = CrossAttnEncoder(
        d_inner=d_inner,
        d_model=d_model,
        n_chan=n_chan,
        sample_rate=sample_rate,
        n_fft=n_fft,
        n_bands=n_bands,
        band_config="musical",
        query_type="learnable",
        n_heads=4,
    )
    decoder = CrossAttnDecoder(
        d_inner=d_inner,
        d_model=d_model,
        n_src=n_src,
        n_chan=n_chan,
        sample_rate=sample_rate,
        n_fft=n_fft,
        n_bands=n_bands,
        band_config="musical",
        query_type="learnable",
        n_heads=4,
    )
    return BSLocoformer(
        encoder=encoder,
        decoder=decoder,
        n_src=n_src,
        n_chan=n_chan,
        n_layers=n_layers,
        emb_dim=d_model,
        attention_dim=d_model,
        n_heads=4,
        ffn_type="swiglu_conv1d",
        ffn_hidden_dim=2 * d_model,
        conv1d_kernel=8,
        masking=True,
    )
```

Target budget:

- Small teacher: 5M to 8M params, roughly SFC-CA small class.
- Quality teacher: 12M to 18M params, roughly SFC-CA medium class.
- Compute: not strict NPU; expected tens of GFLOPs/s for high-quality chunked
  inference.

Core value:

- Gives the repo a real quality ceiling.  All edge models should be judged by
  how much of this teacher they preserve.

Main risks:

- Too expensive for strict realtime NPU.
- Needs careful long-window overlap-add to avoid boundary artifacts.

### Proposal B: `BandSFCNet-RT+` main edge student

Goal:

- Best chance of strong realtime edge deployment from existing code.
- Preserve adaptive SFC compression while staying under a roughly 2M to 6M
  parameter / 3 GMAC/s / 192 KiB state target where possible.

Keep:

- `BandSFCNetNPU` cross-attn transport from `quality`.
- BandSCNet cross-band/narrow-band blocks.
- Dilation schedule `(1, 1, 2, 4, 6)` but validate NPU kernel span.
- Bounded causal attention only after F -> K compression.

Add:

- Teacher distillation from Proposal A.
- Complex residual head in addition to gain mask head.
- Low-frequency preservation loss for bass/music.
- Multi-resolution STFT loss and mixture consistency.
- Optional second-stage residual correction head inspired by TS-BSMamba2.

Discard or constrain:

- Full output at F=2049 if the I/O budget is counted with state; prefer
  `fp512keep475` or `fp513` recipes for first deployment.
- Repeated tensor expansion that creates `Tile`/`Expand`.
- Any shape path that triggers ONE `FULLY_CONNECTED` lowering for activation
  MatMul.

High-level architecture:

```text
packed complex STFT [B, 2M, T, F]
  -> optional fixed frequency preprocessor: F -> F'
  -> SFC transport: F' -> K=64
       safe: soft-band pooling
       quality: NPU-safe cross-attention
  -> L BandSFC stages:
       CrossBandBlock: stateless frequency mixing over K
       DilatedNarrowBandBlock: causal temporal conv plus optional bounded attn
       PooledChannelMixer: parameter capacity without persistent state
  -> SFC decoder: K -> F'
  -> complex mask/residual head
  -> host applies mask/residual and iSTFT
```

Implementation entry point:

```python
from BandSFCNetNPU.presets import build_band_sfc_net_npu_preset


def build_band_sfc_net_rt_plus(
    *,
    n_freq: int,
    n_src: int = 3,
    n_chan: int = 1,
    quality: bool = True,
):
    preset = "quality" if quality else "safe"
    model = build_band_sfc_net_npu_preset(
        preset,
        n_freq=n_freq,
        n_src=n_src,
        n_chan=n_chan,
        masking=True,
    )
    return model
```

Distillation training sketch:

```python
def separation_student_loss(
    student_est,
    teacher_est,
    target,
    mixture,
    student_latent=None,
    teacher_latent=None,
):
    loss = 0.0
    loss = loss + complex_l1(student_est, target)
    loss = loss + 0.5 * multi_resolution_stft_loss(student_est, target)
    loss = loss + 0.2 * si_sdr_loss(student_est, target)
    loss = loss + 0.3 * complex_l1(student_est, teacher_est.detach())
    loss = loss + 0.1 * mixture_consistency_loss(student_est, mixture)
    loss = loss + 0.2 * low_frequency_weighted_l1(student_est, target, max_hz=300)
    if student_latent is not None and teacher_latent is not None:
        loss = loss + 0.05 * latent_l2(student_latent, teacher_latent.detach())
    return loss
```

Target budget:

- First deployable quality run: 2.0M to 2.5M params, state <= 192 KiB.
- Quality probe: 5M to 6.5M params, state <= 192 KiB, GMAC/s <= 3 if possible.
- Existing validated anchor: `band-sfc-net-npu.quality.rt192k.fp512` at about
  2.09M params and 186 KiB fp16 state with ONE import/opt/quant pass.

Core value:

- This is the most realistic "strong and deployable" architecture already in
  the tree.

Main risks:

- It may still lag the offline SFC teacher by 1 dB or more.
- Needs real training evidence; current validation is mostly graph/budget.

### Proposal C: `Hierarchical-SFC-FFI-Lite` middle-tier model

Goal:

- Bridge Proposal A and B: more multi-scale quality than BandSFCNet, less cost
  than full SFC-Locoformer.

Keep:

- Existing `online_hierarchical_soft_band_ffi_sfc_2d.py`.
- Existing `online_hierarchical_soft_band_parallel_ffi_sfc_2d.py`.
- Multi-scale K schedule such as `128 -> 96 -> 48`.
- Frequency path stateless, time path causal.

Add:

- A small global frequency attention or focused-linear attention branch only at
  the bottleneck K=48 level.
- SepRe-style early source split after the first compression stage.
- Teacher distillation from Proposal A.

Discard or constrain:

- Large `d_model=96, n_layers=12` defaults for strict NPU.
- Wide parallel branches unless state budget is explicitly measured.

High-level architecture:

```text
packed complex STFT
  -> soft/crossattn SFC front compression F -> 128
  -> stage 0: frequency interleave + causal time branch
  -> downsample 128 -> 96
  -> stage 1: FFI block
  -> downsample 96 -> 48
  -> bottleneck:
       parallel temporal branches, e.g. dilation 1 and 6
       optional tiny focused linear attention over time
  -> upsample and skip merge
  -> decoder to full F
```

Implementation entry point:

```python
# Candidate file to add later:
# spectral_feature_compression/core/model/online_hierarchical_sfc_ffi_lite.py

class HierarchicalSFCFFILite(nn.Module):
    def __init__(self, n_freq, n_src, n_chan, bands=(128, 96, 48), d_model=20):
        super().__init__()
        self.encoder = FrontSFCCompressor(n_freq=n_freq, n_bands=bands[0], d_model=d_model)
        self.pre = FFIStage(d_model=d_model, n_bands=bands[0], time_dilations=(1,))
        self.mid = FFIStage(d_model=d_model, n_bands=bands[1], time_dilations=(1,))
        self.bottleneck = ParallelTemporalFFIBlock(
            d_model=d_model,
            n_bands=bands[2],
            branch_dilations=(1, 6),
        )
        self.decoder = FrontSFCDecoder(n_freq=n_freq, n_bands=bands[0], n_src=n_src)

    def forward_stream(self, x, state):
        # Keep the same explicit state tuple style as current online SFC.
        ...
```

Target budget:

- `d_model=20` class around the prior rt192k parallel-FFI cache target.
- 3M to 6M params if a bottleneck capacity branch is added.
- GMAC/s likely between BandSFCNet and full SFC teacher.

Core value:

- Best place to test whether multi-scale U-Net quality beats single-scale
  BandSFCNet under similar state budgets.

Main risks:

- More complex exporter and cache accounting.
- If not distilled, may not learn enough global separation behavior.

### Proposal D: `SFC-SepReformer-MultiStem`

Goal:

- Improve source disentanglement by splitting sources earlier instead of asking
  the final head to perform all disentanglement.
- Most useful for DnR / universal three-stem tasks and multi-speaker style
  separation.

Keep:

- SFC-CA encoder.
- SepReformer early split and weight-shared decoder concept.
- TF-Locoformer or local/global Transformer blocks.

Discard or constrain:

- Time-domain encoder if the target is existing STFT/NPU pipeline.
- Huge full-sequence transformer if edge deployment is required.

High-level architecture:

```text
complex STFT
  -> SFC-CA F -> K
  -> shared analysis blocks
  -> source split module: [B, D, K, T] -> [B, N, D_s, K, T]
  -> source-wise small local/global blocks with shared weights
  -> cross-source reconstruction decoder
  -> SFC decoder K -> F
```

Implementation sketch:

```python
class SourceSplitModule(nn.Module):
    def __init__(self, d_model: int, n_src: int):
        super().__init__()
        self.n_src = n_src
        self.proj = nn.Conv2d(d_model, n_src * d_model, kernel_size=1)

    def forward(self, z):
        b, d, t, k = z.shape
        z = self.proj(z)
        return z.view(b, self.n_src, d, t, k)


class SharedSourceRefiner(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, z):
        b, n, d, t, k = z.shape
        z = z.reshape(b * n, d, t, k)
        z = self.block(z)
        return z.reshape(b, n, d, t, k)
```

Target budget:

- Teacher/middle tier first: 6M to 12M params.
- NPU student only after proving source split helps.

Core value:

- Directly attacks the "late mask head bottleneck" that likely hurts small
  edge models.

Main risks:

- Source permutation and source ordering must be handled carefully.
- For fixed music stems, the benefit may be less than for speech/DnR.

### Proposal E: `EdgeFusion-SFC-Distilled`

Goal:

- Fastest path to a strict NPU-shippable model.

Keep:

- `EdgeFusionNPU` packed state and static two-input/two-output contract.
- `ssm_lite`, band bottleneck, token capacity options.
- Low-frequency bias.

Add:

- Distillation from Proposal B or A.
- Multi-resolution loss and mixture consistency.
- A small SFC-like soft-band front-end, if it can be kept as constants plus
  Conv/MatMul without forbidden ops.

Discard or constrain:

- Any architecture feature that increases persistent state more than quality.
- Any graph op not accepted by ONE without a known rewrite pass.

High-level architecture:

```text
single STFT frame [B, 2M, F, 1] + packed state
  -> low-frequency bias
  -> input pointwise projection
  -> N repeated EdgeFusion blocks
       causal time memory
       freq depthwise mixing
       pointwise gate
       optional ssm_lite state update
  -> band/token capacity bottleneck
  -> real gain mask
```

Implementation entry point:

```python
from EdgeFusionNPU.edge_fusion_npu import build_edge_fusion_npu_preset


def build_edgefusion_sfc_distilled():
    return build_edge_fusion_npu_preset("large-v2-hybrid-5m")
```

Target budget:

- 2M to 5M params.
- Persistent state under 192 KiB.
- GMAC/s under 3.

Core value:

- Production fallback.  It may not win quality, but it should be easiest to
  ship and debug.

Main risks:

- Without distillation, it will probably sound weak.
- If the token capacity bottleneck over-compresses frequency too early, bass
  and transient separation will suffer.

## Training Plan

Architecture alone is not enough.  The papers show that training recipe
matters as much as block choice.

### Stage 1: build the quality teacher

Train `SFC-Locoformer-Lite+` on MUSDB18-HQ and DnR separately.

Required losses:

- complex L1 / complex MAE on RI spectrogram;
- multi-resolution complex STFT loss;
- time-domain SI-SDR or SNR term;
- mixture consistency;
- low-frequency weighted loss for bass/music;
- silent-source penalty so the model does not hallucinate in inactive stems.

Required augmentation:

- random stem mixing;
- random gain;
- polarity inversion;
- channel flip;
- temporal shift;
- pitch shift for music, carefully limited for speech/DnR;
- random EQ or mild band dropout for robustness.

### Stage 2: distill into `BandSFCNet-RT+`

Use teacher outputs and intermediate compressed features:

- waveform/STFT target loss to ground truth;
- teacher output loss;
- latent loss on SFC compressed bands;
- stem activity/silence consistency;
- low-frequency teacher-student emphasis.

Use the current validated configs:

- `recipes/dnr/models/band-sfc-net-npu.quality.rt192k.fp512/config.yaml`
- `recipes/dnr/models/band-sfc-net-npu.safe.rt192k.fp513/config.yaml`
- matching MUSDB recipe should be added if absent.

### Stage 3: distill into strict `EdgeFusion`

Only after the BandSFC student is useful, distill further:

- teacher = BandSFCNet-RT+ or SFC-Locoformer;
- student = EdgeFusion large-v2-hybrid-5m or balanced-v2-hybrid;
- deploy with the same chunk/frame cadence used in export.

### Stage 4: validation matrix

Every model should report:

- params;
- GMAC/s at sample rate/hop;
- state fp16/fp32 KiB;
- ONNX node count and forbidden op count;
- ONE import/opt/quant/circle-verify result;
- MUSDB cSDR/uSDR or DnR SNR/SI-SDR;
- streaming vs offline numerical diff;
- listening notes for bass, drums/transients, vocals/speech leakage, artifacts.

## Concrete Implementation Order

1. Add a MUSDB `BandSFCNetNPU` recipe matching the DnR `quality.rt192k.fp512`
   pattern.
2. Add a training wrapper option for teacher-output distillation.
3. Train the SFC-Locoformer small teacher or download/check a pretrained SFC
   checkpoint if available.
4. Train `BandSFCNetNPU quality` student with and without distillation.
5. Run `tools/online/measure_npu_model_stats.py` and
   `tools/online/export_verify_mlir.py`.
6. Only then spend time on `EdgeFusionNPU` strict deployment.

## Expected Ranking

| Rank | Model | Quality expectation | Deployability expectation |
|---:|---|---|---|
| 1 | SFC-Locoformer-Lite+ | Best | Chunked GPU/CPU/NPU only if relaxed |
| 2 | Hierarchical-SFC-FFI-Lite | High | Medium risk |
| 3 | BandSFCNet-RT+ | Best strict-edge candidate | High if using validated presets |
| 4 | DolphinSFCNPU slim 6m/8m distilled | Medium-high | High |
| 5 | EdgeFusion-SFC-Distilled | Medium | Highest |
| 6 | pure BandSCNetNPU / TF-MLPNet without distillation | Medium-low | High |

## Final Recommendation

Invest first in **BandSFCNet-RT+ distilled from SFC-Locoformer-Lite+**.

Reason:

- SFC-CA is the best evidence-backed way to preserve quality with fewer
  parameters than band-split.
- BandSFCNetNPU is already the closest local deployable integration of SFC and
  BandSCNet.
- Distillation is the missing piece between "compileable edge model" and
  "separation quality that can compete."

The model to implement next is not a brand-new isolated family.  It should be a
quality-oriented extension of `BandSFCNetNPU` plus a teacher-student training
recipe, because that reuses the strongest local graph work while importing the
strongest SOTA quality signal from SFC/RoFormer/Moises-Light.
