# Requirements: Band-SCNet-NPU — NPU-Native Audio Source Separation Model

## Overview

Design and implement a new audio source separation model architecture that integrates the strongest ideas from Band-SCNet, SCNet, DTTNet/RT-STT, TIGER, and the existing Online SFC family, while being fully compatible with the project's strict NPU deployment constraints from day one.

The model targets 3-stem cinematic audio source separation (Speech/Music/Effects) at 44.1 kHz for real-time streaming inference on edge devices (TV) with NPU.

---

## Functional Requirements

### FR-1: Model Architecture — Sparse Compression Encoder/Decoder
The model shall use a sparse compression encoder that splits the frequency axis into low, medium, and high bands with asymmetric processing depth (more convolution modules for low-frequency, fewer for high-frequency), inspired by Band-SCNet/SCNet. The decoder shall perform the inverse upsampling. All downsampling/upsampling shall use only stride=2 Conv2d/ConvTranspose2d (chained as needed to achieve effective stride 4 or 16).

### FR-2: Model Architecture — Interleaved Cross-band and Narrow-band Separator
The separation network shall use L groups of alternating Cross-band and Narrow-band blocks (inspired by Band-SCNet/SpatialNet):
- **Cross-band block**: Captures frequency correlations using Conv2d with kernel (1, K_f) along frequency axis, plus a pointwise frequency-mixing layer implemented as bmm or Conv2d(1,1).
- **Narrow-band block**: Captures temporal information using causal Conv2d with kernel (K_t, 1) along time axis, with gating (sigmoid * linear). Optionally includes a lightweight causal attention (bmm + softmax) with bounded KV window for longer-range temporal context.

### FR-3: Model Architecture — Multi-Output (3 Stems)
The model shall output all 3 stems (Speech, Music, Effects) simultaneously from a single forward pass. The output shall be source-gain masks applied to the input spectrogram (real-valued masking on packed real/imag channels).

### FR-4: Model Architecture — Causal/Streaming Design
The model shall be strictly temporally causal. Output at frame t shall depend only on frames ≤ t. The model shall provide:
- `forward(x)` — full-sequence forward for training
- `forward_stream(x_frame, state)` — single-frame streaming forward for deployment
- `init_stream_state(batch_size, device, dtype)` — initialize streaming state

### FR-5: NPU Operator Compliance
The model shall use ONLY the following operators in its forward pass:
- Conv2d, ConvTranspose2d (stride=2 only for transposed)
- bmm (via reshape to 3D)
- softmax, sigmoid
- ReLU (no PReLU, no SiLU in deployed graph — SiLU can be used in training-only paths or approximated as sigmoid*x)
- reshape, transpose (minimize usage)
- elementwise: add, mul, sub, div
- padding (zero-pad for causal convolutions)

### FR-6: Tensor Dimension Constraints
All tensors in the forward pass shall have at most 4 dimensions. For 4D tensors, dimension 0 shall always be the batch dimension (set to 1 for NPU export). No folding of other dimensions into the batch dimension.

### FR-7: Convolution Kernel Constraints
All Conv2d and ConvTranspose2d layers shall satisfy: `(kernel_size - 1) * dilation ≤ 14` on each spatial axis.

### FR-8: Streaming State Budget
The total streaming state (all causal caches + any attention KV buffers) shall fit within 192 KiB when stored in fp16. The model shall provide a `state_size_bytes(dtype)` method to report this.

### FR-9: ONNX Export
The model shall be exportable to ONNX (opset 11) in streaming mode with explicit state tensor inputs/outputs. The exported graph shall contain no dynamic control flow (no If, Loop, Scan ops). The number of state tensors shall be minimized (target: ≤ 30 total inputs/outputs).

### FR-10: MLIR Compilation
The exported ONNX model shall pass onnx-mlir --EmitMLIR without errors and the emitted MLIR shall contain no red-flag patterns (onnx.If, onnx.Loop, onnx.Scan, scf.if/for/while, cf.cond_br).

### FR-11: No Constant Tensors in Forward Pass
No learnable or fixed constant tensors shall be used directly in the forward computation graph. Band/basis priors shall either be embedded as ONNX initializers (model weights) or externalized as explicit graph inputs.

### FR-12: Integration with Existing Infrastructure
The model shall integrate with:
- The existing ONNX export tool (`tools/online/export_onnx_online_model.py`)
- The MLIR verification pipeline (`tools/online/export_verify_mlir.py`)
- The NPU model stats tool (`tools/online/measure_npu_model_stats.py`)
- The streaming state budget checker (`tools/online/report_streaming_state_size.py`)
- The training infrastructure (aiaccel + hydra configs)

### FR-13: Training Recipe
The model shall include at least one training recipe for the DnR dataset (Speech/Music/Effects, 3-stem) with configuration compatible with the existing recipe infrastructure.

### FR-14: Model Size Target
The model shall have ≤ 3M parameters in its deployment configuration. A smaller "edge" preset (≤ 1M params) shall also be provided for smoke testing and NPU graph validation.

---

## Non-Functional Requirements

### NFR-1: Separation Quality Target
The model should target ≥ 7.5 dB SDR on MUSDB18-HQ (all stems average) when trained, which would be competitive with Band-SCNet's 7.79 dB. For DnR (Speech/Music/Effects), the target is competitive with existing baselines.

### NFR-2: Latency
The model shall support a maximum algorithmic latency of 92 ms (matching Band-SCNet's STFT window of 4096 at 44.1 kHz). The hop size shall be ~23 ms (1024 samples at 44.1 kHz).

### NFR-3: Real-Time Factor
The model should achieve RTF < 0.5 on a single CPU core (i7-class), confirming real-time feasibility before NPU deployment.

### NFR-4: Minimize Memory Operations
The design should minimize Slice, Transpose, and Cat operations in the ONNX graph, as these are slow on the target NPU. Prefer reshape over transpose where possible.

### NFR-5: Minimize Graph Node Count
The design should prefer fewer, larger operations (e.g., one Conv2d with more channels) over many small operations (e.g., many grouped convolutions with few channels each). Target: ≤ 2000 ONNX nodes for the streaming cell.

### NFR-6: Code Organization
The model shall be implemented in a new directory `BandSCNetNPU/` at the project root, following the same pattern as `DolphinSFCNPU/`. It shall include:
- Model implementation
- Test script (streaming consistency + ONNX export smoke)
- README with architecture description and usage examples

### NFR-7: Reproducibility
All hyperparameters shall be documented. The model shall be deterministic given a fixed random seed.

---

## Architecture Design Rationale (Traceability)

| Design Choice | Source Model | NPU Adaptation |
|---------------|-------------|----------------|
| Sparse frequency compression (asymmetric low/mid/high) | Band-SCNet, SCNet | Stride=2 chaining instead of stride=4/16 |
| Interleaved Cross-band + Narrow-band blocks | Band-SCNet, SpatialNet | Conv2d(1,K) + Conv2d(K,1) instead of GConv1D |
| Causal temporal Conv2d (no LSTM/RNN) | Online SFC, RT-STT | Native — already NPU-compatible |
| Single-path (no iterative unrolling) | RT-STT | Avoids ONNX node explosion |
| Lightweight causal attention (bounded window) | TIGER, Band-SCNet MHSA | bmm + softmax with fixed KV window size |
| Multi-output masking | Online SFC, DolphinSFCNPU | Source-gain masks via channel slicing |
| Gated mixing (sigmoid * linear) | Band-SCNet GLU, TIGER | sigmoid + mul (both NPU-supported) |
| Frequency-mixing via bmm | Online SFC soft-band | 3D bmm after reshape (NPU-supported) |
| ReLU activation (no PReLU/SiLU) | NPU constraint | Direct replacement; SiLU approximated as sigmoid*x where needed |

---

## Acceptance Criteria

1. Model builds and runs forward pass without errors
2. `forward_stream()` output matches `forward()` output (numerical consistency ≤ 1e-5)
3. Streaming state fits within 192 KiB fp16
4. All Conv2d layers satisfy (k-1)*d ≤ 14
5. ONNX export succeeds with opset 11
6. ONNX checker passes
7. ONNX op audit passes against `edge_npu_recommended` preset
8. onnx-mlir --EmitMLIR succeeds without red flags
9. No forbidden ops (Tile, Expand, ConstantOfShape, If, Loop, Scan)
10. Parameter count ≤ 3M (main preset), ≤ 1M (edge preset)
11. Training recipe exists and runs without errors
12. README documents architecture, usage, and design rationale
