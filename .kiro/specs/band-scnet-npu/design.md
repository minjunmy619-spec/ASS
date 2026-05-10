# Design: Band-SCNet-NPU — NPU-Native Audio Source Separation

This document specifies the technical design for the Band-SCNet-NPU model. It integrates the strongest ideas from Band-SCNet (Interspeech 2025), SCNet (ICASSP 2024), DTTNet / RT-STT, TIGER (ICLR 2025), and the project's existing Online SFC family, while being strictly compliant with the NPU constraints in `AGENT.md`.

The target task is real-time, causal, 3-stem separation (Speech / Music / Effects) at 44.1 kHz for on-device TV NPU deployment under a 192 KiB streaming-state budget.

---

## 1. High-Level Architecture

```
 packed-complex STFT   [B, 2M, T, F]   (F = 2049; real/imag interleaved on channel axis)
        │                               (STFT is computed by the host/DSP, NOT inside the ONNX graph)
        ▼
  X_spec  [B, 2M, T, F]
        │
 ┌──────────────┐
 │  Sparse      │   asymmetric depth per band
 │  Downsample  │   low  (≤17.5% of F):  3× ConvMod, stride-chain 1
 │  Encoder     │   mid  (39.2% of F):   2× ConvMod, stride-chain 2×2 (=4)
 │  (SD)        │   high (43.3% of F):   1× ConvMod, stride-chain 2×2×2×2 (=16)
 └──────────────┘
        │
        ▼
  Z_enc  [B, C, T, F']          (F' = F/1 + F/4 + F/16 per band, concatenated on F axis)
        │
 ┌─────────────────────────────┐
 │  Separation Network         │   L groups of alternating blocks
 │  ─────────────────────      │
 │  × L {                       │
 │      CrossBandBlock         │  frequency-axis Conv2d(1, Kf)
 │      NarrowBandBlock        │  time-axis causal Conv2d(Kt, 1) + optional bounded attention
 │  }                           │
 └─────────────────────────────┘
        │
        ▼
  Z_sep  [B, C, T, F']
        │
 ┌──────────────┐
 │  Sparse      │   inverse pyramid of the encoder
 │  Upsample    │   high band: 4× ConvTranspose2d stride=2 (×16)
 │  Decoder     │   mid band:  2× ConvTranspose2d stride=2 (×4)
 │  (SU)        │   low band:  0× (identity)
 └──────────────┘
        │
        ▼
  Z_dec  [B, C, T, F]
        │
 ┌──────────────────────┐
 │  Source Mask Head    │   3× (source-gain mask, one per stem)
 │  Conv2d → ReLU → Conv2d → (sigmoid + tanh gate)
 └──────────────────────┘
        │
        ▼
 masks  [B, 3, 2M, T, F]  (3 stems, real/imag packed — reshape happens OUTSIDE the ONNX graph)
        │
        │  apply_packed_complex_mask + iSTFT   (host/DSP runtime, NOT inside the exported ONNX graph)
        ▼
 stems_wave  [B, 3, M, samples]
```

Total streaming state: one causal cache per Conv2d that has `(kernel_t - 1) > 0`, one per bounded attention block. All caches kept fp16 in fewer than 192 KiB.

> **STFT / iSTFT are NOT part of the exported ONNX model.** The model consumes a pre-computed packed-complex STFT tensor and emits a packed-complex mask tensor. STFT preprocessing and iSTFT post-processing live in the host/DSP streaming runtime (see `spectral_feature_compression/core/model/online_model_wrapper.py::CausalISTFTOLA` and `tools/online/run_streaming_inference.py::StreamingISTFTWriter`). This matches the existing convention in `tools/online/export_onnx_online_model.py` and `TF-MLPNet/tf_mlpnet/export_onnx.py`.

---

## 2. Design Choices and Paper Traceability

| Block | Inspired by | Why we adopted it | How we adapted it for NPU |
|-------|-------------|-------------------|---------------------------|
| Sparse low/mid/high band split with asymmetric depth | Band-SCNet, SCNet | Lets low frequencies (where most source energy sits) get more capacity without blowing up FLOPs on high frequencies | Use strided `Conv2d(kernel=(1,3), stride=(1,2))` chains instead of `GConv1D stride=4/16`, obeying NPU stride=2 rule |
| Interleaved Cross-band / Narrow-band blocks | Band-SCNet, SpatialNet | Decouples frequency-correlation modeling from temporal modeling. Best known tradeoff in Band-SCNet (7.79 dB on MUSDB18-HQ) | Cross-band uses Conv2d(1,Kf); Narrow-band uses causal Conv2d(Kt,1). No `GConv1D`. |
| Causal temporal Conv2d (not LSTM) | Online SFC, RT-STT | Streaming-friendly, single-pass, no iterative unroll, NPU-native | Implement `CausalConv2d(Kt,1)` with explicit state cache (already exists in `online_sfc_2d.py`) |
| Sparse downsample / upsample (stride chain) | Band-SCNet (SD/SU) | Reduces F dim early, saves FLOPs, aligns skip connections | Replace `deconv stride=4/16` with 2/4 chained `ConvTranspose2d stride=2` |
| Bounded KV causal attention (optional) | TIGER MSA, Band-SCNet MHSA | Longer-range temporal modeling without unbounded state | Attention over a fixed KV window W (e.g. 16 frames), implemented as `bmm + softmax` with a ring-buffer KV cache. Bound: `state_W = (num_heads * head_dim * W)` |
| Source-gain masking (multi-output) | Online SFC, DolphinSFCNPU | Single forward pass yields all 3 stems, matches downstream iSTFT | `Conv2d → ReLU → Conv2d` heads, sigmoid-bounded gain masks on packed real/imag |
| Gated linear unit (a ⊗ σ(b)) | Band-SCNet GLU, Dolphin | Adds nonlinearity without SiLU in the deployment graph | Sigmoid + elementwise multiply (both NPU-supported) |
| PReLU as default activation | Band-SCNet, SCNet, Dolphin | Per `TF-MLPNet/context.md`, PReLU is on the NPU supported-op list. Keep PReLU in the deployed graph; only fall back to ReLU if int8 quantization later proves unstable for PReLU. | Use `nn.PReLU` directly; no sanitization pass removes it. |
| SiLU materialised as `sigmoid + mul` | NPU compliance | SiLU is not on the NPU op list; expressing it as two basic ops keeps the graph portable. | If a SiLU-style gate is needed, compose `sigmoid(x) * x` explicitly. |
| STFT / iSTFT outside the exported graph | Project convention (existing export tools) | Complex STFT is computed by the host/DSP as preprocessing; the model consumes packed-complex `[B, 2M, T, F]`. iSTFT is host-side post-processing. | Keeps ONNX graph small, avoids torchaudio STFT op (not in NPU set); matches `export_onnx_online_model.py` & `TF-MLPNet/export_onnx.py`. |
| No constants in forward pass | NPU rule 8 | Band/basis priors are ONNX initializers (baked as registered buffers) | `register_buffer("routing_bias", ...)` pattern from `online_soft_band_sfc_2d.py` |

---

## 3. Configuration Profiles

Two presets will ship:

| Preset | `C (base channels)` | `L (num stages)` | `K_t` (time kernel) | `K_f` (freq kernel) | Attention | Params | State (fp16) |
|--------|--------------------|------------------|---------------------|---------------------|-----------|--------|--------------|
| `edge_small` (smoke-test / NPU validation) | 16 | 2 | 5 | 3 | off | ~0.5 M | ~40 KiB |
| `rt192k` (deployment target) | 48 | 4 | 5 | 3 | 1 bounded attn per stage, W=16, heads=4 | ~2.5 M | ~170 KiB |

The edge preset is intended to validate the full ONNX→MLIR→.so pipeline quickly; the main preset targets the Band-SCNet parameter class (2.59 M). A third, quality-biased preset (`rt192k_plus` with `C=64, L=3`) may be derived later.

---

## 4. Module Specifications

All tensors follow `[B, C, T, F]` layout. No axis is folded into the batch dimension.

### 4.1 Input Front-End

- The model's input is a **pre-computed** packed-complex STFT tensor `[B, 2M, T, F]` where channels are `[real_0, imag_0, real_1, imag_1, ...]`. The host-side streaming runtime computes the STFT and applies `pack_complex_stft_as_2d` (already implemented in `online_sfc_2d.py`) before feeding the tensor into the exported ONNX model.
- **No STFT is part of the exported ONNX graph.** STFT and iSTFT live entirely in the streaming runtime, matching the existing convention in this repository. The deployment pipeline is:

  `waveform → (host) STFT → (host) pack_complex_stft_as_2d → [ONNX model] → (host) unpack + apply mask → (host) iSTFT → waveform per stem`

  This keeps the ONNX graph small, avoids the torchaudio STFT op (not supported by the NPU), and lets the DSP use its own highly tuned STFT implementation.

### 4.2 Sparse Downsample Encoder (SD)

Implemented as three parallel branches routed by a pre-computed band-mask:

```python
class SparseDownsampleEncoder(nn.Module):
    # Inputs:
    #   x: [B, 2M, T, F]        F = 2049
    # Outputs:
    #   z_low:  [B, C, T, F_l]  F_l = floor(F * 0.175)
    #   z_mid:  [B, C, T, F_m]  F_m = floor(F * 0.392 / 4)
    #   z_high: [B, C, T, F_h]  F_h = floor(F * 0.433 / 16)
```

Each branch:

```
 Slice F axis  →  Conv2d(1×1) lift to C  →  N× ConvBlock  →  [B, C, T, F_band]
```

Where `N = {3, 2, 1}` for `{low, mid, high}` and each ConvBlock is:

```python
ConvBlock = Sequential(
    RMSNorm2d(C),
    CausalConv2d(C, C, kernel=(Kt, 1), groups=C),  # depthwise causal time conv
    nn.PReLU(num_parameters=C),                     # NPU-supported per TF-MLPNet/context.md
    Conv2d(C, C, kernel=(1, 3), padding=(0, 1)),   # pointwise-ish frequency mix
    GatedAct(),                                     # sigmoid-gated, described below
)
```

Stride chains for inter-branch size reduction:

- Low: no stride → keeps `F_l`.
- Mid: `Conv2d(stride=(1,2))` × 2 interleaved with ConvBlocks → total F-stride 4.
- High: `Conv2d(stride=(1,2))` × 4 interleaved with ConvBlocks → total F-stride 16.

Why: every stride is 2, satisfying NPU rule 6.

### 4.3 CrossBandBlock

```python
class CrossBandBlock(nn.Module):
    # Captures cross-frequency correlations at a single time step.
    # Input:  [B, C, T, F']
    # Output: [B, C, T, F']
    def forward(self, x):
        y = self.norm(x)
        y = self.freq_conv(y)                # Conv2d(C, 2C, kernel=(1, Kf), padding=(0, Kf//2))
        a, b = y.chunk(2, dim=1)
        y = a * torch.sigmoid(b)             # GLU
        y = self.pointwise(y)                # Conv2d(C, C, 1x1)
        return x + y                          # residual
```

- **No temporal convolution here** → no streaming state needed for this block.
- `freq_conv` kernel `(1, Kf)`: the only operator touching the frequency axis. Obeys `(Kf-1)*d ≤ 14`.
- Residual add is in-place safe (NPU supports).

### 4.4 NarrowBandBlock

```python
class NarrowBandBlock(nn.Module):
    # Captures per-frequency temporal dynamics.
    # Input:  [B, C, T, F']
    # Output: [B, C, T, F']
    def __init__(self, C, Kt=5, use_attn=False, attn_window=16, num_heads=4, head_dim=8):
        self.norm       = RMSNorm2d(C)
        self.causal_dw  = CausalConv2d(C, 2C, kernel=(Kt, 1), groups=C)   # depthwise time conv, doubled for GLU
        self.pointwise  = nn.Conv2d(C, C, kernel_size=1)
        if use_attn:
            self.attn = BoundedCausalAttn(C, window=attn_window, num_heads=num_heads, head_dim=head_dim)
```

Forward:
```
 y = norm(x)
 a, b = causal_dw(y).chunk(2, dim=1)
 y = a * sigmoid(b)                     # GLU
 if use_attn:
     y = y + attn(y)                    # optional, see 4.5
 y = pointwise(y)
 return x + y
```

Streaming state per NarrowBandBlock:
- `state_dw`: shape `[B, C, Kt-1, F']` fp32 (for training) / fp16 (for deployment)
- `state_attn` (only if `use_attn=True`): ring-buffer KV cache, shape `[B, num_heads, attn_window, head_dim * 2]` per block

### 4.5 BoundedCausalAttn (optional)

Design constraints:
- Causal (only past and current frames contribute).
- Bounded window `W` → bounded state size.
- Per-frequency independent (no attention across F axis) → enables straightforward 3D `bmm`.

Formulation:
```
# Input x: [B, C, T, F']
# Reshape to [B * F', C, T] → operate on T axis only
# Project to Q, K, V:  [B*F', num_heads, head_dim, T]
# KV cache keeps only last W frames
# score = bmm(Q_t, K_cache^T) / sqrt(head_dim)        # shape [B*F', num_heads, 1, W+1]
# attn  = softmax(score, dim=-1)
# out   = bmm(attn, V_cache)                          # shape [B*F', num_heads, 1, head_dim]
```

Implementation notes:
- Reshape fuses `B` and `F'` into the "batch-like" leading dim **only when using `bmm`** (bmm is `[B_eff, N, M] × [B_eff, M, P]`), which the project already does in `SoftBandQueryCompressor2d`. This is permitted because bmm's leading dim is not interpreted as a real batch by the NPU op lowering; the stream-state caches remain `[B, H, W, D]` 4D tensors.
- All four dims after reshape are ≤ 4.
- Softmax is along the last dim of a 3D tensor; supported.

KV state layout per attention block: `[B, num_heads, W, 2 * head_dim]` (K and V concatenated along last dim to halve graph nodes).

### 4.6 Sparse Upsample Decoder (SU)

Mirror of Section 4.2:

- Low branch: `Conv2d(C, C, 1x1)` only.
- Mid branch: `ConvTranspose2d(C, C, kernel=(1, 2), stride=(1, 2))` ×2 + ConvBlocks.
- High branch: `ConvTranspose2d(C, C, kernel=(1, 2), stride=(1, 2))` ×4 + ConvBlocks.

`kernel=2, stride=2` means no overlap, no checkerboard artifact when combined with a 1×1 refinement afterwards. `stride=2` is the only value allowed by rule 6 for `ConvTranspose2d`.

Skip connections: encoder output of each branch is concatenated on the channel dim with the corresponding decoder stage output, then collapsed back to C via a `Conv2d(2C, C, 1x1)`. This is the "Fusion Network" in Band-SCNet's terminology.

### 4.7 Source Mask Head

```python
class SourceMaskHead(nn.Module):
    def __init__(self, C, in_2m, num_sources=3):
        self.proj  = nn.Conv2d(C, num_sources * in_2m, kernel_size=1)
        self.gate  = nn.Conv2d(C, num_sources * in_2m, kernel_size=1)
    def forward(self, x):
        a = self.proj(x)
        b = torch.sigmoid(self.gate(x))
        mask = a * b                                                 # [B, S*2M, T, F]
        return mask
```

Downstream, `mask` is reshaped to `[B, S, 2M, T, F]` **outside the ONNX graph** (in the streaming-runtime wrapper), so the exported graph itself stays ≤ 4D.

### 4.8 GatedAct

Standalone helper used throughout:

```python
class GatedAct(nn.Module):
    # Takes concatenated [a, b] along channel dim, returns a * sigmoid(b).
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * torch.sigmoid(b)
```

This is the project's SiLU replacement: `SiLU(x) == x * sigmoid(x)` is materialised explicitly with `sigmoid + mul` rather than a fused op, matching existing TIGER-NPU-Edge sanitization.

---

## 5. Streaming State Layout

The streaming state is represented as a `Dict[str, torch.Tensor]` at the PyTorch level, and as an explicit ordered tuple of state tensors at the ONNX level. Layout per preset:

### `edge_small` (L=2, Kt=5, no attention)

| State tensor | Shape | Bytes (fp16) |
|--------------|-------|--------------|
| sd_low_dw_state × 3 | `[1, 16, 4, F_l]` | 3 × 16 × 4 × 359 × 2 = ~44 KiB |
| sd_mid_dw_state × 2 | `[1, 16, 4, F_m]` | 2 × 16 × 4 × 201 × 2 = ~25 KiB |
| sd_high_dw_state × 1 | `[1, 16, 4, F_h]` | 1 × 16 × 4 × 55 × 2 = ~7 KiB |
| nb_dw_state × L=2 | `[1, 16, 4, F']` | 2 × 16 × 4 × 615 × 2 = ~76 KiB |

Total: ~152 KiB → fits 192 KiB with headroom.
_(Numbers will be tightened once the exact F_l / F_m / F_h split is measured; the design budgets ≤ 170 KiB to leave a 22 KiB safety margin for iSTFT scratch.)_

### `rt192k` (L=4, Kt=5, 1 attention/stage, W=16, heads=4, head_dim=8)

| State tensor | Shape | Bytes (fp16) |
|--------------|-------|--------------|
| sd_*_dw_state × 6 | same as edge but C=48 | ~96 KiB |
| nb_dw_state × L=4 | C=48 | ~80 KiB |
| nb_attn_kv_state × L=4 | `[1, 4, 16, 16]` (heads, W, 2*D) | 4 × 4 × 16 × 16 × 2 = ~8 KiB |

Total: ~184 KiB. A `state_size_bytes()` method must verify this at init time and raise if the budget is exceeded.

---

## 6. Forward / Streaming API

Mirrors the existing project convention so that `tools/online/export_verify_mlir.py` and friends work out of the box.

```python
class BandSCNetNPU(nn.Module):
    def forward(self, x_spec: torch.Tensor) -> torch.Tensor:
        """
        Training-mode full-sequence forward.
        x_spec: [B, 2M, T, F]
        returns: mask [B, S*2M, T, F]
        """

    def init_stream_state(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> dict[str, torch.Tensor]:
        """Initializes all per-layer state tensors, zero-filled."""

    def forward_stream(
        self,
        x_frame: torch.Tensor,                      # [B, 2M, 1, F] — single frame
        state: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Returns (mask_frame [B, S*2M, 1, F], new_state)."""

    def state_size_bytes(self, dtype: torch.dtype = torch.float16) -> int:
        """Compile-time total state size used for budget verification."""
```

For ONNX export, an adapter converts the dict into a flat tuple (same pattern as `export_onnx_online_model.py`). The state-name ordering is deterministic and documented in the model docstring.

---

## 7. Integration Points

### 7.1 Project layout

```
BandSCNetNPU/
├── __init__.py
├── README.md
├── band_scnet_npu.py                 # main model
├── blocks.py                         # CrossBandBlock, NarrowBandBlock, BoundedCausalAttn, GatedAct
├── sparse_io.py                      # SparseDownsampleEncoder, SparseUpsampleDecoder, band split utils
├── presets.py                        # edge_small / rt192k factory functions
├── streaming.py                      # flat-state adapter for ONNX export
└── test_band_scnet_npu.py            # smoke tests (streaming consistency, ONNX export, op audit)
```

Existing primitives to reuse (import, do not duplicate):

| From | Import |
|------|--------|
| `spectral_feature_compression.core.model.online_sfc_2d` | `CausalConv2d`, `RMSNorm2d`, `pack_complex_stft_as_2d`, `apply_packed_complex_mask`, `_runtime_assert` |
| `spectral_feature_compression.core.model.online_soft_band_sfc_2d` | `SoftBandSpec2d` (optional, if we decide to borrow the band-bias pattern) |
| `TIGER.npu_edge_utils` | `sanitize_for_npu_edge`, `verify_npu_edge_constraints` |

### 7.2 Tool compatibility

Each tool the spec must integrate with and what Band-SCNet-NPU must expose so the tool works untouched:

| Tool | Required hook |
|------|---------------|
| `tools/online/export_onnx_online_model.py` | Register the model under a new `--target band-scnet-npu` flag with an optional `--preset {edge_small,rt192k}` |
| `tools/online/audit_onnx_model.py` | No model-side change; audit uses the existing `edge_npu_recommended` op allowlist |
| `tools/online/export_verify_mlir.py` | Same as above; passes the exported ONNX through onnx-mlir --EmitMLIR and red-flag scans the textual MLIR |
| `tools/online/measure_npu_model_stats.py` | Register Band-SCNet-NPU presets alongside the existing ready-suite |
| `tools/online/report_streaming_state_size.py` | The `state_size_bytes()` method makes this trivial |

### 7.3 Training recipe

A new recipe `recipes/band_scnet_npu/` shall contain:

- `config/model/band_scnet_npu_rt192k.yaml` — hydra config pointing at `BandSCNetNPU.presets.rt192k()`.
- `config/task/dnr_3stem_causal.yaml` — DnR task config for S/M/E 3-stem separation, reusing the existing causal-training task wiring.
- `train.sh`, `eval_streaming.sh` — thin launch scripts matching the Online SFC pattern.

---

## 8. NPU Constraint Verification Plan

Each module must pass the following automated checks before the full pipeline runs. These are built into the test file (`test_band_scnet_npu.py`) and also surfaced by `tools/online/audit_onnx_model.py`.

| Rule | Check | Pass criterion |
|------|-------|----------------|
| 1, 2 | ONNX op audit | only ops in `edge_npu_recommended` allowlist |
| 3, 4 | static shape inspection | every tensor `ndim ≤ 4`; dim-0 is always batch |
| 5 | walk Conv2d / ConvTranspose2d layers | `(k-1) * d ≤ 14` on each spatial axis |
| 6 | walk ConvTranspose2d layers | stride ∈ {2} only |
| 7 | walk all modules | no `AdaptiveAvgPool2D` |
| 8 | ONNX graph walk | no constant tensor is a direct input to a Conv/MatMul in forward |
| 9 | ONNX op audit | no `ScatterND`, `Unflatten`, `Expand`, `Tile`, `ConstantOfShape` |
| 10 | node-count audit | ≤ 2 000 nodes (NFR-5) |
| 11 | same as 10 | — |
| 12 | streaming-consistency test | `forward(x)` vs framewise `forward_stream()` max abs diff ≤ 1e-5 |
| 13 | `state_size_bytes(torch.float16)` | ≤ 196 608 (= 192 KiB) |
| 14 | MLIR red-flag scan | no `onnx.If`, `onnx.Loop`, `onnx.Scan`, `scf.if`, `scf.for`, `scf.while`, `cf.cond_br` |

The test file should fail loudly (assert + descriptive error) if any check fails, so CI surfaces the exact broken rule.

---

## 9. Open Design Questions (to resolve during implementation)

1. **Exact band split**: Band-SCNet uses 17.5 / 39.2 / 43.3 %. With F=2049 this gives ~358 / 803 / 887 bins. Do we round to a multiple of 16 so the high-band F/16 stride chain stays integer? Proposed: round `(low, mid, high)` to `(352, 800, 896)` and refine with a `ceil` guard.
2. **Attention on/off for `edge_small`**: keeping it off reduces the state budget to ~152 KiB and halves the node count, which is ideal for the MLIR smoke test. Proposal: default off for `edge_small`, default on for `rt192k`.
3. **Frequency-axis preprocessing reuse**: the Online SFC family has `FrequencyPreprocessedOnlineModel` (optional mel / fp-keep pre-filter). The design treats Band-SCNet-NPU as raw-STFT only in v1 and adds pre-filter as a v2 follow-up to keep the initial surface area small.
4. **Band-prior bias in the encoder**: SCNet derives its sparse encoder from subband energy. For simplicity we start with _no_ learned routing bias; if quality lags the baseline, revisit by adding a `register_buffer` bias computed from the triangular band basis in `FrozenDolphinBandSpec2d`.

These are not blockers; each has a documented fallback.

---

## 10. Why this design, not a direct Band-SCNet port

A direct port would still require: Conv1D → Conv2D, stride-4/16 deconv chaining, MHSA state management, and per-frequency Linear → Conv2d(1×1). That is essentially a rewrite. The design above captures the architectural _ideas_ (sparse SD/SU pyramid, interleaved cross-band / narrow-band, lightweight attention) using building blocks already proven NPU-compatible in this repository. It also fits naturally alongside the existing Online SFC family and reuses its streaming / export / MLIR tooling verbatim. (PReLU is kept as the default activation since the NPU supports it; SiLU is the one activation that had to go, and it is replaced by an explicit `sigmoid + mul` pair. STFT / iSTFT are handled by the host/DSP runtime and never appear inside the exported ONNX graph.)
