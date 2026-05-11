# BandSCNetNPU — NPU-native 3-stem audio source separator

Causal, streaming, NPU-compatible separator for 3-stem audio (Speech /
Music / Effects) at 44.1 kHz on a TV-class NPU under a 192 KiB streaming
state budget.

This module is the implementation of the spec at
`.kiro/specs/band-scnet-npu/`. See `design.md` for the full architectural
rationale.

## What's inside

```
BandSCNetNPU/
├── __init__.py
├── README.md                  — this file
├── band_scnet_npu.py          — full model (BandSCNetNPU, export wrapper)
├── blocks.py                  — CrossBandBlock, NarrowBandBlock, BoundedCausalAttn, GatedAct
├── sparse_io.py               — SparseDownsampleEncoder, SparseUpsampleDecoder, band split
├── streaming.py               — flat-state adapter helpers for torch.onnx.export
├── presets.py                 — edge_small / rt192k factories
└── test_band_scnet_npu.py     — streaming-consistency, constraint, and ONNX smoke tests
```

## Architecture snapshot

```
 packed-complex STFT  [B, 2M, T, F]             (STFT/iSTFT live OUTSIDE the ONNX graph)
        │
        ▼
   zero-pad F up to a clean split (n_freq_padded)
        │
 ┌──────────────┐
 │ Sparse       │  three branches, asymmetric depth
 │ Downsample   │  low  (≈17.5%): lift → [stride2] → [stride2] → ConvBlock
 │ Encoder (SD) │  mid  (≈39.2%): lift → [stride2] → [stride2] → ConvBlock
 │              │  high (≈43.3%): lift → [stride2]*4 → ConvBlock
 └──────────────┘
        │
        ▼
   concat on F'        [B, C_p, T, F_l/4 + F_m/4 + F_h/16]
        │
        ▼
   pre_sep_proj 1x1  [B, C_s, T, F']
        │
        │ × L stages:
        │   CrossBandBlock   (frequency mixing, stateless)
        │   NarrowBandBlock  (causal time mixing + bounded causal attention)
        ▼
   post_sep_proj 1x1  [B, C_p, T, F']
        │
 ┌──────────────┐
 │ Sparse       │  mirror of SD; stride=2 ConvTranspose2d chains; skip-fused 1x1
 │ Upsample     │
 │ Decoder (SU) │
 └──────────────┘
        │
        ▼
   crop back to n_freq
        │
        ▼
   Source Mask Head  (Conv2d → PReLU → Conv2d → sigmoid per stem)
        │
        ▼
   mask  [B, n_src*n_chan, T, F]  (applied to packed-complex input → [B, 2*n_src*n_chan, T, F])
```

### Op inventory (per NPU allowlist in `AGENT.md` + `TF-MLPNet/context.md`)

Used: `Conv2d`, `ConvTranspose2d` (stride=2), `bmm`, `softmax`, `sigmoid`,
`PReLU`, elementwise `add`/`mul`/`sub`, `reshape`/`transpose`/`concat`/`slice`.

Not used (forbidden by allowlist): `Expand`, `Tile`, `ConstantOfShape`,
`ScatterND`, `If`, `Loop`, `Scan`, `Conv1d`, `AdaptiveAvgPool2d`, `SiLU`,
`LSTM`/`GRU`.

## Presets

`build_band_scnet_npu_preset(name, n_freq=...)` returns a `BandSCNetNPU`.
Measured inside the project Docker container at `n_freq=2049` (standard STFT
with `n_fft=4096`) using `tools/online/measure_npu_model_stats.py` and
`tools/online/export_verify_mlir.py`:

| Preset        | channels (sep / pyr) | stages | Kt | attention                 | params | state fp16 | GMAC/s | ONNX nodes | MLIR ops |
|---------------|----------------------|--------|----|---------------------------|-------:|-----------:|-------:|-----------:|---------:|
| `edge_small`  | 16 / 8               | 2      | 5  | off                       | 10,411 | 108.75 KiB | 0.2055 | 460        | 2,798    |
| `rt192k`      | 40 / 8               | 3      | 3  | W=16, heads=4, head_dim=8 | 62,115 | 190.88 KiB | 1.2588 | 716        | 3,798    |
| `rt192k_plus` | 56 / 8               | 2      | 3  | W=16, heads=4, head_dim=8 | 72,915 | 178.00 KiB | 1.6178 | 588        | 3,257    |

The state-only number is not the full deployment I/O budget. At `n_freq=2049`
and fp16, the streaming-cell signature also carries the current input frame
(`x`, 8,196 bytes) and masked output frame (`y`, 24,588 bytes). That means:

| Preset        | state only | state + x + y | input state + output state + x + y |
|---------------|-----------:|--------------:|-----------------------------------:|
| `edge_small`  | 108.75 KiB | 140.77 KiB    | 249.52 KiB                         |
| `rt192k`      | 190.88 KiB | 222.89 KiB    | 413.77 KiB                         |
| `rt192k_plus` | 178.00 KiB | 210.02 KiB    | 388.02 KiB                         |

So `rt192k` and `rt192k_plus` are valid ONNX/MLIR bring-up candidates, but they
are not final hardware-ready presets until the exact DSP/NPU memory accounting
is confirmed or the state/signature is reduced. Parameter counts are deliberately
small because the 192 KiB state/cache quota dominates: per NarrowBandBlock state
is `1 * C_sep * (Kt-1) * F' * 2 bytes`, and with F'≈F/16≈128-348 even a modest
`C_sep * L` grows quickly. If the quota is relaxed, widen `channels` or raise
`num_stages` / `time_kernel` accordingly (see `presets.py` for the formulas).

## Streaming / NPU contract

- Input: packed-complex STFT tensor `[B, 2*n_chan, T, F]`
- Output: masked packed-complex tensor `[B, 2*n_src*n_chan, T, F]`
- STFT / iSTFT are NOT part of the exported ONNX graph. The host/DSP
  streaming runtime is responsible for framing the waveform into STFT
  frames, packing real/imag into the channel axis via
  `pack_complex_stft_as_2d`, and un-packing + iSTFT the output.

### Streaming API

```python
from BandSCNetNPU import build_band_scnet_npu_preset

model = build_band_scnet_npu_preset("rt192k", n_freq=2049).eval()

# one-shot (training / offline eval)
y = model(packed_complex_stft)   # [B, 2, T, 2049] -> [B, 6, T, 2049]

# single-frame streaming
state = model.init_stream_state(batch_size=1, dtype=torch.float32)
for t in range(T):
    frame = packed_complex_stft[:, :, t:t+1, :]   # [1, 2, 1, 2049]
    y_t, state = model.forward_stream(frame, state)

print(model.state_size_bytes(dtype=torch.float16))   # DSP budget check
```

### ONNX export (streaming cell)

```python
import torch
from BandSCNetNPU import build_band_scnet_npu_preset, BandSCNetNPUStreamingExportWrapper
from spectral_feature_compression.utils.onnx_streaming import flatten_tensor_tree

model = build_band_scnet_npu_preset("edge_small", n_freq=2049).eval()
wrapper = BandSCNetNPUStreamingExportWrapper(model, batch_size=1, dtype=torch.float32).eval()
x = torch.randn(1, 2, 1, model.n_freq)
flat_state, _ = flatten_tensor_tree(tuple(model.init_stream_state(batch_size=1, dtype=torch.float32)))

torch.onnx.export(
    wrapper,
    (x, *flat_state),
    "band_scnet_npu_edge_small.onnx",
    opset_version=11,
    input_names=["x", *[f"state_{i}" for i in range(len(flat_state))]],
    output_names=["y", *[f"next_state_{i}" for i in range(len(flat_state))]],
    do_constant_folding=True,
    dynamo=False,
)
```

For end-to-end MLIR verification (inside the project's docker container,
per `AGENT.md`):

```bash
cd /app/ASS
./.venv/bin/python tools/online/export_verify_mlir.py \
  --target band-scnet-npu \
  --band-scnet-npu-preset rt192k \
  --freqs 2049 \
  --n-chan 1 \
  --label BandSCNetNPU_rt192k \
  --out-dir /tmp/export_verify_band_scnet_rt192k \
  --allow-op PRelu \
  --fail-on-disallowed-ops
```

`PRelu` is used intentionally by the model and is supported by the target-side
model design. The generic `edge_npu_recommended` audit preset does not include
it yet, so use `--allow-op PRelu` for strict audit runs unless the shared
allowlist has been updated.

For repeatable parameter/MAC/node summaries:

```bash
cd /app/ASS
./.venv/bin/python tools/online/measure_npu_model_stats.py \
  --target band-scnet-npu \
  --band-scnet-npu-preset rt192k_plus \
  --freqs 2049 \
  --n-chan 1 \
  --out-dir /tmp/npu_model_stats_band_scnet_rt192k_plus
```

## Run the local test suite

Most tests are pure-PyTorch; the final ONNX smoke also needs the ONNX exporter
dependencies available in the active environment. From the project root:

```bash
./.venv/bin/python -m BandSCNetNPU.test_band_scnet_npu
```

Coverage:

- block-level shape and streaming-consistency checks (CrossBand / NarrowBand with and without attn)
- sparse-pyramid round-trip shape test
- band-split soundness across `n_freq ∈ {128, 257, 513, 1025, 2049}`
- full-model streaming-vs-full-sequence parity for both presets
- NPU kernel-size + transposed-stride constraint walk
- streaming state byte budget (≤ 192 KiB fp16) for both presets
- ONNX export smoke (opset 11) + `onnx.checker` + forbidden-op check

If the ONNX smoke fails with `ModuleNotFoundError: No module named 'onnxscript'`
under a new PyTorch exporter default, use the deployment tooling above for the
authoritative export/MLIR check or pass `dynamo=False` in direct
`torch.onnx.export` calls.

Expected:

```
[pass] test_cross_band_block_shape
[pass] test_narrow_band_block_streaming_consistency_no_attn
[pass] test_narrow_band_block_streaming_consistency_with_attn
[pass] test_sparse_pyramid_round_trip_shape
[pass] test_band_split_sums_to_n_freq
[pass] test_edge_small_forward_shape
[pass] test_edge_small_streaming_matches_full
[pass] test_rt192k_streaming_matches_full
[pass] test_npu_conv_constraints_both_presets
[pass] test_state_budget_edge_small
[pass] test_state_budget_rt192k
[pass] test_streaming_onnx_export_edge_small

all 12 tests passed
```

## Paper traceability

| Component                                | Source paper / module               | NPU adaptation                                                                |
|------------------------------------------|-------------------------------------|-------------------------------------------------------------------------------|
| Sparse band split (low/mid/high) + SD/SU | Band-SCNet (Interspeech 2025), SCNet (ICASSP 2024) | stride-chain 2×2×2×2 instead of GConv1D stride=4/16                         |
| CrossBandBlock / NarrowBandBlock         | Band-SCNet, SpatialNet              | Conv2d(1, Kf) + causal Conv2d(Kt, 1), no GConv1D                              |
| Causal temporal Conv2d (no LSTM)         | RT-STT (2024), Online SFC           | native — already NPU-compatible                                               |
| Bounded causal KV attention              | TIGER MSA (ICLR 2025), Band-SCNet MHSA | fixed window W, frequency-pooled KV, `bmm + softmax` only                   |
| GLU gate (`a * sigmoid(b)`)              | Band-SCNet, Dolphin                 | SiLU is replaced by explicit `sigmoid + mul`                                  |
| PReLU activation                         | Band-SCNet, SCNet, Dolphin          | PReLU is NPU-supported per `TF-MLPNet/context.md`                              |
| Source-gain masking                      | Online SFC, DolphinSFCNPU           | real-valued gain per stem, applied to packed complex via channel-wise mul     |
| STFT / iSTFT outside the graph           | Project convention                  | STFT, unpack, and iSTFT performed by the host/DSP, not the ONNX model         |

## Known limitations

- The current presets trade parameter count for DSP state budget. If the
  192 KiB quota is relaxed, widen `channels` / increase `num_stages`.
  Design-doc targets of 0.5 M / 2.5 M params would require ~2× the
  present state budget.
- The frequency axis is zero-padded to the next split-compatible width
  (typically +3 bins out of 2049) and cropped back before masking.
- `tools/online/measure_npu_model_stats.py` and
  `tools/online/export_verify_mlir.py` support `--target band-scnet-npu`.
  `tools/online/export_onnx_online_model.py` is still not the preferred
  exporter for this model; use the deployment tools shown above.
- The current training recipe is
  `recipes/dnr/models/band-scnet-npu.rt192k/config.yaml`.
