# EdgeFusionNPU Investigation and NPU Bring-up - 2026-05-19

## Goal

Find a small, online, NPU-friendly audio separation direction by first surveying
current local and public model families, then distilling their strongest ideas
into a compileable edge deployment candidate.

This document focuses on the NPU core. STFT/iSTFT, overlap-add, loudness
management, and final waveform reconstruction stay outside the NPU graph.

## Sources Checked

Local repo families:

- `TIGER/`
- `BandSCNetNPU/`
- `Dolphin/`
- `DolphinSFC/`
- `DolphinSFCNPU/`
- `spectral_feature_compression/`
- `TF-MLPNet/`
- `recipes/dnr/models/`
- `recipes/musdb18hq/models/`
- `tools/online/*`

Public papers / model families:

- TF-MLPNet: Tiny Real-Time Neural Speech Separation
  - https://arxiv.org/abs/2508.03047
- Band-SCNet: A Causal, Lightweight Model for High-Performance Real-Time Music Source Separation
  - https://www.isca-archive.org/interspeech_2025/yang25d_interspeech.pdf
- TF-GridNet: Integrating Full- and Sub-Band Modeling for Speech Separation
  - https://arxiv.org/abs/2211.12433
- Band-Split RoPE Transformer / BS-RoFormer
  - https://arxiv.org/abs/2309.02612
- Mel-Band RoFormer
  - https://arxiv.org/abs/2310.01809
- Band-Split RNN
  - https://arxiv.org/abs/2209.15174
- Generalized Bandsplit Neural Network for Cinematic Audio Source Separation
  - https://arxiv.org/abs/2309.02539

## Broad Model Analysis

| Family | Strong Points | Weak Points for Edge Online NPU |
|---|---|---|
| TIGER | Strong online DnR-style structure; explicit streaming states; recent local fixes make TIGER recipes compile to ONE quantized Circle. | Quality-oriented variants can be graph-heavy. Attention export must be carefully shaped to avoid scalar `Gather`, dynamic Slice, and bad `FULLY_CONNECTED` lowering. |
| BandSCNet / BandSCNetNPU | Best current public direction for real-time music separation: sparse compression, cross-band and narrow-band blocks, strong quality/latency/parameter tradeoff. | Original paper includes attention/fusion choices that need careful NPU rewriting. Local NPU branch still needs systematic train/eval evidence. |
| Dolphin | Audio-visual and target-speaker-style strengths; strong conditioning idea when visual/enrollment side information exists. | Visual branch is not appropriate for this target edge audio-only NPU core; more system complexity and not universal for music/DnR. |
| DolphinSFC / DolphinSFCNPU | Useful audio-only adaptation direction; SFC compression reduces frequency cost. | Compression/decoder quality must be validated after NPU simplification; large variants can exceed memory or operator budgets. |
| spectral_feature_compression | Rich local family of online SFC, soft-band, cross-attn-query, hierarchical, and parallel FFI variants; good tooling and recipes. | Some high-fidelity variants rely on MatMul/attention/band constants and are more fragile than pure Conv2d for first-pass NPU deployment. |
| TF-MLPNet | Excellent tiny-edge lesson: TF-domain, alternating channel/frequency MLPs, causal conv over time, designed for low-power accelerators and QAT. | Speech-first; FC/MLP layout can become reshape-heavy unless rewritten as pointwise/frequency Conv2d. |
| TF-GridNet | Very strong separation quality via full-band, sub-band temporal, and cross-frame attention modules. | Too expensive and attention-heavy for the smallest edge target unless distilled into local conv/grid mixers. |
| BSRNN / generalized bandsplit | Excellent inductive bias: split spectrogram into perceptual bands and model band/time sequences. | RNN sequence ops and variable band partitions are less friendly to static NPU export. |
| BS-RoFormer / Mel-RoFormer | Strong non-real-time quality; mel/overlapped band mapping is a useful prior. | Transformer/RoPE attention is expensive, hard to stream causally, and fragile for this ONE quantization path. |

## Design Decision

The first deployable candidate should not copy the biggest SOTA model. It should
compress the common high-value ideas into a small graph that the NPU compiler can
reliably lower:

1. Use STFT-domain packed RI input, like BandSCNet, SFC, TF-GridNet, and TF-MLPNet.
2. Keep the NPU graph frame-streaming with explicit state tensors, like TIGER and
   the local online SFC wrappers.
3. Use static low-frequency emphasis as a cheap mel/bandsplit prior.
4. Use cross-band depthwise frequency Conv2d and narrow-band causal time Conv2d,
   approximating BandSCNet/TF-GridNet full/sub-band modeling without attention.
5. Use pointwise Conv2d + sigmoid gating as a CSA/GLU-lite fusion block.
6. Output real-valued gain masks only; host DSP applies masks and performs iSTFT.
7. Avoid known ONE/NPU failure modes:
   - no scalar head/time indexing exported as `Gather`
   - no `Tile`, `Expand`, or `ConstantOfShape`
   - no PReLU
   - no rank-3 activation-vs-activation `MatMul`
   - no STFT/iSTFT in graph
   - no dynamic batch or dynamic frequency dimension

## Implemented Folder

```text
EdgeFusionNPU/
├── README.md
├── INVESTIGATION_20260519.md
├── __init__.py
├── edge_fusion_npu.py
├── export_compile.py
└── test_edge_fusion_npu.py
```

## Implemented Alternatives

| Preset | Hidden | Blocks | Intended Use |
|---|---:|---:|---|
| `tiny` | 12 | 3 | Budget-safe first edge target and compile control. |
| `compact` | 16 | 5 | More capacity under the 192 KiB fp16 state budget when trained/exported at `n_fft=1024`, `F=513`. |
| `balanced` | 24 | 6 | First serious training candidate if state memory can be reduced by frequency preprocessing or cache compression. |
| `wide` | 32 | 8 | Quality probe, likely needs a larger device or compressed-state variant. |

The current compiled artifact is `tiny`. The larger alternatives are present as
architecture presets but should not be treated as deployment-ready until their
state budget is measured and reduced.

## NPU Contract

Input:

```text
x:       [1, 2*n_chan, F, 1]
state:   [1, num_blocks*hidden_channels, F, time_kernel - 1]
```

Output:

```text
mask:         [1, n_src*n_chan, F, 1]
next_state:   [1, num_blocks*hidden_channels, F, time_kernel - 1]
```

The ONNX export uses two inputs (`x`, `state`) and two outputs (`mask`,
`next_state`). All cache/state data is packed into one 4D tensor to keep every
preset below the requested 4-input/output state limit.

Compiled tiny preset:

```text
n_chan=1
n_src=3
n_freq=1025
hidden_channels=12
num_blocks=3
time_kernel=3
```

## Validation and Experiments

DnR training recipes added:

```text
recipes/dnr/models/edge-fusion-npu.tiny.rt192k/config.yaml
recipes/dnr/models/edge-fusion-npu.compact.fp512.rt192k/config.yaml
recipes/dnr/models/edge-fusion-npu.balanced.fp256.rt192k/config.yaml
```

Each recipe has a sibling `train.sh` matching the existing DnR recipe launch
pattern.

Smoke test:

```text
./.venv/bin/python -m pytest EdgeFusionNPU/test_edge_fusion_npu.py -q
```

Result:

```text
1 passed
```

Historical first compile attempt before packed-state export:

```text
./.venv/bin/python EdgeFusionNPU/export_compile.py \
  --preset tiny \
  --out-dir logs/npu_verify_general/edge_fusion_npu_tiny_20260519 \
  --compile
```

Result: compile passed, but state was too large for the normal 192 KiB edge
budget:

```text
params=4147
state_fp16_kib=256.25
compiled=true
```

Budget repair:

- Changed `tiny` from 16 channels / 4 blocks to 12 channels / 3 blocks.

Budgeted compile after shrinking `tiny`, before packing state:

```text
./.venv/bin/python EdgeFusionNPU/export_compile.py \
  --preset tiny \
  --out-dir logs/npu_verify_general/edge_fusion_npu_tiny_budget_20260519 \
  --compile
```

Result:

```text
compiled=true
params=1923
param_fp16_kib=3.76
state_fp16_kib=144.14
state_fp32_kib=288.28
```

Final packed-state compiles:

```text
./.venv/bin/python EdgeFusionNPU/export_compile.py \
  --preset tiny \
  --out-dir logs/npu_verify_general/edge_fusion_npu_tiny_packed_state_20260519 \
  --compile

./.venv/bin/python EdgeFusionNPU/export_compile.py \
  --preset compact \
  --n-freq 513 \
  --out-dir logs/npu_verify_general/edge_fusion_npu_compact_fp512_packed_state_20260519 \
  --compile

./.venv/bin/python EdgeFusionNPU/export_compile.py \
  --preset balanced \
  --n-freq 257 \
  --out-dir logs/npu_verify_general/edge_fusion_npu_balanced_fp256_packed_state_20260519 \
  --compile
```

Final packed-state results:

| Variant | ONNX Inputs | ONNX Outputs | Params | State fp16 KiB | ONE q.circle |
|---|---:|---:|---:|---:|---|
| `tiny` F=1025 | 2 | 2 | 1,923 | 144.14 | PASS |
| `compact` F=513 | 2 | 2 | 5,091 | 160.31 | PASS |
| `balanced` F=257 | 2 | 2 | 12,699 | 144.56 | PASS |

Each final graph uses:

```text
inputs:  x, state
outputs: mask, next_state
```

ONNX operator audit:

```text
./.venv/bin/python tools/online/audit_onnx_model.py \
  logs/npu_verify_general/edge_fusion_npu_tiny_packed_state_20260519/model.onnx \
  --op-preset edge_npu_recommended
```

Result:

```text
Ops (8): Add, Concat, Constant, Conv, Mul, Relu, Sigmoid, Slice
Disallowed ops: none
Initializers (fp16 estimate): 5.76 KiB
```

The streaming state budget is recorded by `export_compile.py` in each manifest
because this custom wrapper uses a custom packed-state export path:

```text
tiny:     state_fp16_kib=144.14
compact:  state_fp16_kib=160.31
balanced: state_fp16_kib=144.56
```

Compiled artifacts:

```text
logs/npu_verify_general/edge_fusion_npu_tiny_packed_state_20260519/model.q.circle
logs/npu_verify_general/edge_fusion_npu_compact_fp512_packed_state_20260519/model.q.circle
logs/npu_verify_general/edge_fusion_npu_balanced_fp256_packed_state_20260519/model.q.circle
```

## Current Conclusion

`EdgeFusionNPU tiny` is a successful NPU bring-up candidate, not a quality claim.
It proves the fused design can be expressed as a very small, static,
online-friendly, quantized ONE graph.

The model is likely too small to be SOTA in quality as-is. Its purpose is to
establish a clean deployment substrate. The next serious quality candidate should
keep this compile-clean backbone but add one of the following, in order:

1. Frequency preprocessed `F=512/513` variant to free state budget for more
   channels/blocks.
2. A band-token bottleneck branch using only Conv2d/Resize/Slice-safe ops.
3. A QAT training run comparing `tiny`, `compact-fp512`, and TIGER
   `npu-edge-v2`.
4. Optional teacher distillation from BandSCNet or Mel/BS-RoFormer outputs.

## Recommended Next Architecture

The most promising next deployable model is:

```text
EdgeFusionNPU-compact-fp512
```

with:

- `n_fft=1024`, giving 513 bins,
- 16 hidden channels,
- 5 blocks,
- the same Conv2d-only cross-band/narrow-band/gated mixer,
- optional teacher distillation from BandSCNet/Mel-RoFormer/BS-RoFormer,
- no attention in the first deployable graph.

This should retain the strongest ideas from current SOTA families while staying
inside the NPU compiler's proven operator envelope.

## Extended Latest Alternatives Survey

The second sweep added more recent 2025-2026 evidence and changed the design
pressure slightly: the model should still avoid full attention in the deployable
graph, but it should not stay as a pure local Conv2d stack if we want a serious
quality candidate.

| Model / Paper | New Evidence | What to Reuse | What to Avoid on Edge NPU |
|---|---|---|---|
| Moises-Light, 2025, https://arxiv.org/abs/2510.06785 | A lightweight band-split U-Net can approach much larger models when the band path is designed carefully and trained with enough data. | Add a tiny band-token bottleneck so the model has a compressed global-ish frequency path. | Do not copy a full U-Net decoder with large skip tensors into the NPU streaming frame graph. |
| Windowed Sink Attention for vocal separation, 2025, https://arxiv.org/abs/2510.25745 | Mel-Band-RoFormer attention was found to be highly local; windowed sink attention recovered much of quality with a very large FLOP reduction. | Treat local temporal context plus a small persistent memory as the deployable substitute for full temporal attention. | Avoid full temporal attention, RoPE, and long dynamic sequences in the first NPU graph. |
| TF-MLPNet, 2025, https://arxiv.org/abs/2508.03047 | Tiny TF-domain separation can run on very low-power accelerators with QAT and small causal chunks. | Keep STFT-domain operation, channel/frequency pointwise mixing, causal time convs, and QAT-first training. | Avoid FC-heavy reshape layouts that may become fragile in ONNX/ONE; prefer Conv2d equivalents. |
| Band-SCNet, Interspeech 2025, https://www.isca-archive.org/interspeech_2025/yang25d_interspeech.html | Causal music separation with sparse compression, cross-band/narrow-band blocks, 2.59M params, and 92 ms latency is a strong real-time target. | Keep cross-band + narrow-band separation and compressed fusion as the main quality prior. | Do not directly carry CSA/attention unless it is proven to compile and fit state budget. |
| SCNet, 2024, https://arxiv.org/abs/2401.13276 | Sparse compression improves music separation while reducing compute by focusing model capacity where spectral information is denser. | Use frequency compression as a learned bottleneck, especially for low/mid bands. | Avoid fixed complicated band maps inside the graph if they require constants, gathers, or shape ops. |
| BS-RoFormer, 2023, https://arxiv.org/abs/2309.02612 | Band-split plus hierarchical inner/inter-band modeling is still a high-quality teacher direction. | Use it as a distillation teacher and as architectural evidence for band-level modeling. | Transformer/RoPE blocks are not a good first deployable graph for this ONE path. |
| Mel-Band RoFormer, 2023, https://arxiv.org/abs/2310.01809 | Mel/overlapped bands beat heuristic non-overlapped band splitting for several stems. | Keep static low-frequency emphasis and mel-like compression in training/data preprocessing. | Avoid overlapped learned gather-style band construction in the NPU graph until a static-lowered version is tested. |
| SPMamba, 2024, https://arxiv.org/abs/2404.02063 | State-space modeling can replace heavier recurrent/attention modules in TF separation and capture longer dependencies with linear complexity. | Borrow the state-space idea as a learnable recurrent cache update. | Raw bidirectional Mamba is not online and not ONE-friendly as-is. |
| SepMamba, 2024, https://arxiv.org/abs/2410.20997 | Mamba U-Net variants can beat similarly sized Transformer models with lower compute and memory; causal variants are reported. | Prefer recurrent finite-state memory over quadratic attention when extending temporal range. | Avoid selective-scan custom kernels or bidirectional context in deployment. |
| Omni-directional Mamba, 2026, https://arxiv.org/abs/2601.16603 | Recent work points out that single-axis Mamba misses 2D spectrogram dependencies; multi-directional scanning improves quality. | Keep the idea that both time and frequency context matter; combine recurrent state with band bottleneck. | Multi-directional scan schedules are too complex for the first edge NPU artifact. |

## Deep Fusion V2 Decision

The second-pass deployable direction is:

```text
EdgeFusionNPU-compact-v2-hybrid
```

It keeps the original two-input/two-output packed-state contract and adds two
NPU-compiled mechanisms:

- `ssm_lite` cache update: each block stores a learned recurrent state instead
  of only shifting raw frames through the cache. This is the small deployable
  version of the Mamba/SSM lesson.
- `BandTokenBottleneck`: a Conv2d -> pointwise Conv2d -> ConvTranspose2d
  frequency bottleneck. This is the NPU-friendly version of the
  Moises-Light/Band-SCNet/SCNet band-compression lesson.

The avoided pieces are equally important:

- no STFT/iSTFT in graph,
- no full temporal attention,
- no RoPE,
- no dynamic band gathers,
- no scalar-index `Gather`,
- no rank-3 activation `MatMul`,
- no extra cache tensors beyond the single packed `state`.

## V2 Experiments

New DnR training recipes:

```text
recipes/dnr/models/edge-fusion-npu.compact-fp512.v2-ssmlite.rt192k/config.yaml
recipes/dnr/models/edge-fusion-npu.compact-fp512.v2-bandtoken.rt192k/config.yaml
recipes/dnr/models/edge-fusion-npu.compact-fp512.v2-hybrid.rt192k/config.yaml
recipes/dnr/models/edge-fusion-npu.balanced-fp256.v2-hybrid.rt192k/config.yaml
```

Validation commands:

```text
./.venv/bin/python -m py_compile \
  EdgeFusionNPU/export_compile.py \
  EdgeFusionNPU/edge_fusion_npu.py \
  EdgeFusionNPU/training_wrapper.py

./.venv/bin/python -m pytest EdgeFusionNPU/test_edge_fusion_npu.py -q
```

Result:

```text
2 passed
```

The pytest run still reports the pre-existing `.pytest_cache` permission
warning, but all EdgeFusion tests pass.

V2 ONE compiles:

```text
./.venv/bin/python EdgeFusionNPU/export_compile.py \
  --preset compact-v2-ssmlite \
  --n-freq 513 \
  --out-dir logs/npu_verify_general/edge_fusion_npu_compact_fp512_v2_ssmlite_20260519 \
  --compile

./.venv/bin/python EdgeFusionNPU/export_compile.py \
  --preset compact-v2-bandtoken \
  --n-freq 513 \
  --out-dir logs/npu_verify_general/edge_fusion_npu_compact_fp512_v2_bandtoken_20260519 \
  --compile

./.venv/bin/python EdgeFusionNPU/export_compile.py \
  --preset compact-v2-hybrid \
  --n-freq 513 \
  --out-dir logs/npu_verify_general/edge_fusion_npu_compact_fp512_v2_hybrid_20260519 \
  --compile

./.venv/bin/python EdgeFusionNPU/export_compile.py \
  --preset balanced-v2-hybrid \
  --n-freq 257 \
  --out-dir logs/npu_verify_general/edge_fusion_npu_balanced_fp256_v2_hybrid_20260519 \
  --compile
```

Measured V2 results:

| Variant | ONNX Inputs | ONNX Outputs | Params | State fp16 KiB | Key Ops | ONE q.circle |
|---|---:|---:|---:|---:|---|---|
| `compact-v2-ssmlite` F=513 | 2 | 2 | 5,251 | 160.31 | Conv, Add, Mul, Sigmoid, Slice | PASS |
| `compact-v2-bandtoken` F=513 | 2 | 2 | 6,211 | 160.31 | Conv, ConvTranspose, Add, Mul, Slice | PASS |
| `compact-v2-hybrid` F=513 | 2 | 2 | 6,371 | 160.31 | Conv, ConvTranspose, Add, Mul, Sigmoid, Slice | PASS |
| `balanced-v2-hybrid` F=257 | 2 | 2 | 15,483 | 144.56 | Conv, ConvTranspose, Add, Mul, Sigmoid, Slice | PASS |

ONNX op audits with `edge_npu_recommended`:

```text
compact-v2-ssmlite:  Disallowed ops: none
compact-v2-bandtoken: Disallowed ops: none
compact-v2-hybrid:   Disallowed ops: none
balanced-v2-hybrid:  Disallowed ops: none
```

Compiled V2 artifacts:

```text
logs/npu_verify_general/edge_fusion_npu_compact_fp512_v2_ssmlite_20260519/model.q.circle
logs/npu_verify_general/edge_fusion_npu_compact_fp512_v2_bandtoken_20260519/model.q.circle
logs/npu_verify_general/edge_fusion_npu_compact_fp512_v2_hybrid_20260519/model.q.circle
logs/npu_verify_general/edge_fusion_npu_balanced_fp256_v2_hybrid_20260519/model.q.circle
```

## Updated Recommendation

Train in this order:

1. `edge-fusion-npu.compact-fp512.v2-hybrid.rt192k`
2. `edge-fusion-npu.compact-fp512.v2-ssmlite.rt192k`
3. `edge-fusion-npu.compact-fp512.v2-bandtoken.rt192k`
4. `edge-fusion-npu.balanced-fp256.v2-hybrid.rt192k`

The compact hybrid is the best first serious candidate because it carries both
new inductive biases while keeping F=513 resolution, 6.4K parameters, one packed
state tensor, and a verified quantized ONE artifact.

The balanced hybrid is the best capacity probe but should be treated as an
F=257 quality-risk experiment. It has more channels/blocks and still smaller
state than compact, but the lower frequency resolution may hurt music and dense
effects unless frequency preprocessing/distillation compensates for it.

For quality, the next necessary step is training with teacher distillation from
at least one strong offline teacher such as BandSCNet, BS-RoFormer, Mel-RoFormer,
or Moises-Light. The NPU graph is now ready enough that model quality, not
compiler compatibility, should be the next bottleneck.

## Review Pass and Fixes

Review date: 2026-05-19.

Findings:

1. SSM-lite cache update was not bounded.
   - Issue: `new_cell = decay * prev + input_gate * x` allowed the two positive
     coefficients to sum above one. With repeated frames, the cache could drift
     upward and silently change the operating range seen by quantization.
   - Fix: changed it to a convex update:

     ```text
     new_cell = decay * prev + (1 - decay) * x
     ```

     This keeps the new state between the previous state and the incoming
     hidden frame for each element.

2. Export CLI clobbered preset-native frequency sizes.
   - Issue: `export_compile.py` defaulted `--n-freq` to 1025 and always passed it
     as an override. As a result, `--preset compact` exported F=1025 unless the
     caller remembered to pass `--n-freq 513`, contradicting the preset and
     recipe.
   - Fix: made `--n-freq` optional. When omitted, the preset's own `n_freq` is
     used. Review checks now show:

     ```text
     --preset compact            -> n_freq=513
     --preset balanced-v2-hybrid -> n_freq=257
     ```

3. Mask head was too restrictive for separation quality.
   - Issue: the real mask was hard-capped at 1.0, so the model could only
     attenuate the mixture. That is too weak for DnR/source-separation cases
     where a target source may need modest gain correction.
   - Fix: added `mask_scale` with a conservative default of `1.25`. This keeps a
     bounded, quantization-friendly non-negative mask while allowing limited
     amplification.

4. Runtime state validation was incomplete.
   - Issue: the forward path validated state channels but not state frequency or
     context width, so bad host state tensors could fail later with less helpful
     shape errors.
   - Fix: added explicit checks for state frequency bins and context length.

5. Tests did not cover the risky behavior.
   - Issue: the previous tests only checked basic one-frame shape/range behavior.
   - Fix: added tests for V2 variants, convex SSM-lite cache behavior,
     preset-native export frequency expectations, and bad-state rejection.

6. DnR launch scripts assumed PBS and `nvidia-smi`.
   - Issue: the generated `train.sh` scripts used `sort -u $PBS_NODEFILE` and
     `nvidia-smi` unconditionally. They could fail on a local smoke machine
     before training started.
   - Fix: all EdgeFusionNPU recipe launchers now default to one process/node when
     PBS or `nvidia-smi` is unavailable.

Post-fix validation:

```text
./.venv/bin/python -m py_compile \
  EdgeFusionNPU/edge_fusion_npu.py \
  EdgeFusionNPU/export_compile.py \
  EdgeFusionNPU/training_wrapper.py

./.venv/bin/python -m pytest EdgeFusionNPU/test_edge_fusion_npu.py -q

bash -n recipes/dnr/models/edge-fusion-npu*/train.sh
```

Result:

```text
5 passed
```

The pytest cache permission warning remains environment-only and does not affect
the model tests.

Post-fix recommended compile:

```text
./.venv/bin/python EdgeFusionNPU/export_compile.py \
  --preset compact-v2-hybrid \
  --out-dir logs/npu_verify_general/edge_fusion_npu_compact_fp512_v2_hybrid_reviewfix_20260519 \
  --compile
```

Result:

| Variant | n_freq | ONNX Inputs | ONNX Outputs | Params | State fp16 KiB | ONE q.circle |
|---|---:|---:|---:|---:|---:|---|
| `compact-v2-hybrid` review-fix | 513 | 2 | 2 | 6,291 | 160.31 | PASS |

ONNX op audit:

```text
Ops: Add, Concat, Constant, Conv, ConvTranspose, Identity, Mul, Relu, Sigmoid, Slice, Sub
Disallowed ops: none
```

Compiled artifact:

```text
logs/npu_verify_general/edge_fusion_npu_compact_fp512_v2_hybrid_reviewfix_20260519/model.q.circle
```

## Chunk Training / Frame Export Fix

Update date: 2026-05-20.

The training path now supports chunk or clip processing while preserving the
single-frame ONNX export contract.

PyTorch training paths:

```text
EdgeFusionNPU core:
  input  [B, 2*n_chan, F, T]
  state  [B, state_channels, F, context]
  output [B, n_src*n_chan, F, T], next_state

EdgeFusionNPUOnlineModel:
  input  complex STFT [B, M, F, T]
  output complex estimates [B, n_src, M, F, T]
```

`EdgeFusionNPUOnlineModel.forward(...)` now accepts:

```text
initial_state: optional packed state for split-chunk training
return_state:  return final packed state with the estimates
detach_state:  detach incoming state for truncated BPTT
```

This means a full clip and split chunks with carried state produce the same
result, while training can choose whether gradients cross chunk boundaries.

ONNX export path:

```text
EdgeFusionNPUExportWrapper.forward(x, state)
```

calls the single-frame `_forward_frame(...)` path directly. Exported ONNX remains:

```text
inputs:  x, state
outputs: mask, next_state
```

with `x` and `mask` fixed to one streaming frame.

Validation:

```text
./.venv/bin/python -m py_compile \
  EdgeFusionNPU/edge_fusion_npu.py \
  EdgeFusionNPU/training_wrapper.py \
  EdgeFusionNPU/export_compile.py

./.venv/bin/python -m pytest EdgeFusionNPU/test_edge_fusion_npu.py -q
```

Result:

```text
9 passed
```

Post-fix frame export and ONE compile:

```text
./.venv/bin/python EdgeFusionNPU/export_compile.py \
  --preset compact-v2-hybrid \
  --out-dir logs/npu_verify_general/edge_fusion_npu_compact_fp512_v2_hybrid_chunktrain_frameexport_20260520 \
  --compile
```

Result:

| Variant | n_freq | ONNX Inputs | ONNX Outputs | Params | State fp16 KiB | ONE q.circle |
|---|---:|---:|---:|---:|---:|---|
| `compact-v2-hybrid` chunk-train/frame-export | 513 | 2 | 2 | 6,291 | 160.31 | PASS |

ONNX op audit:

```text
Disallowed ops: none
```

Compiled artifact:

```text
logs/npu_verify_general/edge_fusion_npu_compact_fp512_v2_hybrid_chunktrain_frameexport_20260520/model.q.circle
```

## Bigger 2-6M Parameter Variants

Update date: 2026-05-20.

The earlier EdgeFusionNPU variants were intentionally tiny and proved the export
contract, but they were likely under-parameterized for quality. The bigger
variants scale model capacity through stateless per-frame 1x1 Conv FFN expansion
inside each streaming block:

```text
hidden -> capacity_channels -> hidden
```

This adds parameters without widening the recurrent cache. The streaming state
therefore remains one packed tensor and still uses only:

```text
inputs:  x, state
outputs: mask, next_state
```

New presets:

| Preset | Hidden | Blocks | Capacity Channels | n_freq | Params | Param fp16 | State fp16 KiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| `big-v2-hybrid-2m` | 24 | 5 | 8,192 | 257 | 2,020,483 | 3.85 MiB | 120.47 |
| `large-v2-hybrid-5m` | 24 | 7 | 16,384 | 257 | 5,637,235 | 10.75 MiB | 168.66 |

New DnR recipes:

```text
recipes/dnr/models/edge-fusion-npu.big-fp256.v2-hybrid-2m.rt192k/config.yaml
recipes/dnr/models/edge-fusion-npu.large-fp256.v2-hybrid-5m.rt192k/config.yaml
```

The recipes inherit the F=257 balanced recipe. The `big` recipe uses batch size
2; the `large` recipe uses batch size 1 because its per-frame FFN activations
are substantially larger.

Validation:

```text
./.venv/bin/python -m py_compile \
  EdgeFusionNPU/edge_fusion_npu.py \
  EdgeFusionNPU/training_wrapper.py \
  EdgeFusionNPU/export_compile.py

./.venv/bin/python -m pytest EdgeFusionNPU/test_edge_fusion_npu.py -q
```

Result:

```text
10 passed
```

ONE compiles:

```text
./.venv/bin/python EdgeFusionNPU/export_compile.py \
  --preset big-v2-hybrid-2m \
  --out-dir logs/npu_verify_general/edge_fusion_npu_big_fp256_v2_hybrid_2m_20260520 \
  --compile

./.venv/bin/python EdgeFusionNPU/export_compile.py \
  --preset large-v2-hybrid-5m \
  --out-dir logs/npu_verify_general/edge_fusion_npu_large_fp256_v2_hybrid_5m_20260520 \
  --compile
```

Results:

| Variant | ONNX Inputs | ONNX Outputs | Params | State fp16 KiB | ONE q.circle |
|---|---:|---:|---:|---:|---|
| `big-v2-hybrid-2m` | 2 | 2 | 2,020,483 | 120.47 | PASS |
| `large-v2-hybrid-5m` | 2 | 2 | 5,637,235 | 168.66 | PASS |

ONNX op audits:

```text
big-v2-hybrid-2m:   Disallowed ops: none
large-v2-hybrid-5m: Disallowed ops: none
```

Compiled artifacts:

```text
logs/npu_verify_general/edge_fusion_npu_big_fp256_v2_hybrid_2m_20260520/model.q.circle
logs/npu_verify_general/edge_fusion_npu_large_fp256_v2_hybrid_5m_20260520/model.q.circle
```

Recommendation:

- Train `big-v2-hybrid-2m` first as the safer quality/cost point.
- Train `large-v2-hybrid-5m` if the target NPU can afford the larger 1x1 Conv
  weight bandwidth and activation footprint.
- Keep `compact-v2-hybrid` as the fallback for low-power devices.

## Big Variant Status Correction

Update date: 2026-05-20.

The first big-variant implementation reached the parameter targets by applying a
huge 1x1 Conv FFN at the full frequency resolution. That compiled, but it was
not the right edge-NPU shape: parameter count looked good while the implied
per-frame compute and activation footprint were too large.

A second attempt moved the large FFN onto a highly compressed frequency token,
but used grouped `ConvTranspose2d` for the upsampler. ONE rejected it:

```text
loc("/token_capacity/up/up.0/ConvTranspose"): error: failed to legalize operation 'onnx.ConvTranspose'
```

The corrected design now uses:

- frequency-token bottleneck from F=257 down to one token,
- large 1x1 Conv FFN only at that one-token resolution,
- dense stride-2 `ConvTranspose2d` stages for upsampling,
- `band_stride=2` so transposed conv stride follows the AGENTS rule.

Corrected presets:

| Preset | Params | Param fp16 | State fp16 KiB | Rough GMAC/s at 44.1k hop512 | ONE q.circle |
|---|---:|---:|---:|---:|---|
| `compact-v2-hybrid` stride2 | 5,779 | 12.17 KiB | 160.31 | 0.224 | PASS |
| `big-v2-hybrid-2m` token stride2 | 2,148,515 | 4.10 MiB | 120.47 | 0.478 | PASS |
| `large-v2-hybrid-5m` token stride2 | 5,304,419 | 10.12 MiB | 168.66 | 0.832 | PASS |

The GMAC/s values are rough Conv/ConvTranspose hook estimates, not hardware
profiling, but they are useful for screening against the project guideline of
less than 3 GMAC/s.

The big DnR recipes were also corrected to train with the full STFT frontend:

```text
n_fft=2048
hop_length=512
freq_preprocess_enabled=true
freq_preprocess_keep_bins=192
freq_preprocess_target_bins=257
freq_preprocess_mode=triangular
```

This avoids the earlier quality-risky recipe path that inherited `n_fft=512`
just to make the core see F=257. The NPU core still exports and compiles at
F=257, while training and host inference preserve the higher-resolution STFT
frontend and apply fixed frequency preprocessing.

Corrected compiled artifacts:

```text
logs/npu_verify_general/edge_fusion_npu_compact_fp512_v2_hybrid_stride2_20260520/model.q.circle
logs/npu_verify_general/edge_fusion_npu_big_fp256_v2_hybrid_2m_token_stride2_20260520/model.q.circle
logs/npu_verify_general/edge_fusion_npu_large_fp256_v2_hybrid_5m_token_stride2_20260520/model.q.circle
```
