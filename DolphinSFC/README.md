# DolphinSFC

`DolphinSFC` is an ASS-traceable audio-only adaptation of **Dolphin: Efficient Audio-Visual Speech Separation with Discrete Lip Semantics and Multi-Scale Global-Local Attention**.

## What Transfers From Dolphin

- Dolphin's deployable insight is the separator, not the video-only DP-LipCoder dependency: a single forward pass with multi-scale global/local modeling can replace heavier iterative separators.
- The original global attention block uses coarse self-attention; this version uses a longer causal depthwise Conv2d branch as an NPU-safe global-context approximation.
- The original local attention block uses DCT heat diffusion; this version uses learned causal depthwise Conv2d smoothing, which keeps the heat/smoothing bias but avoids runtime DCT/FFT constants.
- The original visual semantic tokens are replaced by `DolphinSourcePriorCoder2d`, an audio-derived semantic/source prior over compressed SFC band tokens.

## ASS/NPU Contract

- Input: packed complex STFT tensor `(B, 2 * n_chan, T, F)`.
- Output: packed complex separated tensor `(B, 2 * n_src * n_chan, T, F)`.
- Default masking: real-valued source gain masks are repeated onto real/imag channels, avoiding the temporary 5D interleave used by some complex-mask helpers.
- Streaming path: `init_stream_state(...)` and `forward_stream(x[:, :, t:t+1, :], state)`.
- Runtime tensors stay at four dimensions or below.
- The separator uses Conv2d, ConvTranspose2d, elementwise ops, reshape/concat, softmax, and the existing SFC compressor/decoder `bmm` path.
- Temporal kernels are causal and respect `(kernel_size - 1) * dilation < 14`.

## Why This Is Not A Literal Dolphin Port

Dolphin is an audio-visual speech separation model. Generic ASS/SFC targets speech, music, and effects, and the edge deployment path has no lip video. Keeping DP-LipCoder literally would make the model unusable for this task. The portable idea is the compact global-local, multi-scale separator. The visual semantics are reinterpreted as a trainable audio source-prior branch.

## Smoke Test

Run inside the project Docker as requested by `prj_context.md`:

```bash
cd /app/ASS
./.venv/bin/python DolphinSFC/test_dolphin_sfc.py
```

Expected checks:

- full sequence forward matches frame-by-frame streaming forward;
- ONNX export/checker smoke test runs if `onnx` is installed;
- parameter count and fp16 streaming-state size are printed.

## Example Construction

```python
from DolphinSFC import DolphinSFCSeparator

model = DolphinSFCSeparator(
    n_freq=1025,
    n_fft=2048,
    sample_rate=44100,
    n_bands=64,
    n_src=3,
    d_model=288,
    num_scales=3,
)
```

Or use a named preset:

```python
from DolphinSFC import build_dolphin_sfc_preset

model = build_dolphin_sfc_preset(
    "large_8m",
    n_freq=1025,
    n_fft=2048,
    sample_rate=44100,
)
```

## Presets

- `large_8m`: `n_bands=64, d_model=288, num_scales=3`; about 8.0M parameters for 1025-bin STFT. This is the recommended first training recipe for quality.
- `large_6m`: `n_bands=64, d_model=256, num_scales=3`; about 6.3M parameters for 1025-bin STFT.
- `edge_small`: `n_bands=32, d_model=16, num_scales=3`; about 30k parameters and about 116 KB fp16 streaming state for structure/export smoke tests.

The large presets intentionally exceed the 192 KB streaming-cache target in their current form. They are useful for proving the Dolphin/SFC backbone quality first; after that, cache reduction should be handled as a separate deployment pass.
