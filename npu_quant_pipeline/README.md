# NPU Quantization Toolkit

A self-contained, PyTorch-only toolkit for preparing **pretrained models** for
post-training quantization (PTQ), with optional ONNX export and audio evaluation.
No AIMET, vendor SDK, custom C++ extension, or training framework is required.

The goal is to reduce measured quality loss, not promise lossless INT8 inference.
The pipeline compares candidates on held-out inputs and keeps only candidates
that do not worsen its selection metric. **Always validate the compiled NPU
model**, because hardware arithmetic and compiler quantization can differ from
this simulator.

## Features

| Function | Scope |
| --- | --- |
| BN folding | Eval-mode Conv1d/2d/3d followed directly by matching BN |
| Cross-layer equalization (CLE) | Same-kind Conv/Linear pairs, direct or through module ReLU |
| Post-norm smoothing | Standard affine LayerNorm/RMSNorm output into a single Linear |
| Calibrated simulation | Conv/Linear input/output activations and per-output-channel weights |
| AdaRound | Differentiable local reconstruction, hard-rounding evaluation, nearest-round fallback |
| Bias correction | Full-model feature capture and element-weighted channel residuals |
| Quality gates | Held-out output NMSE or a user-provided task score |
| Layer sensitivity | Rank single-layer floating-point bypasses for mixed-precision decisions |
| Audio metrics | SNR, SI-SDR, RMS gain, DC error; optional BSS SDR/SIR/SAR |
| Export | Optimized float ONNX plus a separate, vendor-neutral encoding report |

Graph rewrites conservatively exclude branches, reused/aliased parameters,
unsupported nonlinearities, grouped CLE, and hook-bearing paths. They check
floating-point equivalence before quantization. No pre-RMSNorm channel scaling,
arbitrary activation clamps, or BN gamma clipping is performed.

## Install

Python 3.10+ and PyTorch 2.2+ are required. Install the appropriate CPU/CUDA build
of PyTorch for your machine first, then:

```sh
pip install -e .
# Optional float ONNX export:
pip install -e '.[onnx]'
# Optional mir_eval SDR/SIR/SAR:
pip install -e '.[audio]'
```

The core runtime imports only PyTorch and Python's standard library. `nn.RMSNorm`
smoothing requires a PyTorch version that provides that module. The tested
environment is Python 3.12 / PyTorch 2.14 CPU. CUDA tests skip when unavailable.

## Quick Start

Load your pretrained checkpoint using your model's own loading code first.
Inputs must use exactly the same preprocessing, amplitude convention, shapes,
and streaming state as deployment.

```python
import torch
from npu_quant import NPUQuantizer, QuantConfig, layer_sensitivity

model = pretrained_model.eval()  # Your already-loaded PyTorch model

# Each item is a tuple of POSITIONAL MODEL ARGUMENTS, not (audio, labels).
# The toolkit caches at most max_batches from each iterable on CPU, and moves
# inputs to the model's device. Use separate calibration and validation splits.
calibration = ((mixture,) for mixture, targets in calibration_loader)
validation = [(mixture,) for mixture, targets in validation_loader]

tool = NPUQuantizer(model, QuantConfig(weight_bits=8, activation_bits=8))
result = tool.optimize(
    calibration,
    validation,
    max_batches=16,
    fold_bn=True,
    equalize=True,
    smooth=False,             # Opt in for standard norm -> Linear paths
    adaround_iterations=100,  # 0 disables reconstruction (the default)
    bias_correction=True,     # Pipeline correction requires AdaRound enabled
)

print(result.report['baseline'])
print(result.report['selected'])
print([(s['stage'], s['accepted']) for s in result.report['stages']])

# This result.model uses ordinary float operators. result.simulation adds
# fake-quantization for evaluation; it is NOT an integer NPU model.
example_args = (validation[0][0].to(next(model.parameters()).device),)
result.export_onnx(
    example_args, 'optimized.onnx',
    input_names=['mixture'], output_names=['stems'],
)
result.save_encodings('encodings.json')

ranking = layer_sensitivity(model, result.simulation, validation[:4])
print(ranking[:5])
```

For multiple model inputs, yield `(mixture, state)`; for a model accepting one
dictionary argument, yield `({'audio': mixture, 'state': state},)`. Keyword-only
forward arguments require a small positional adapter module. Avoid passing an
unbounded stream to `layer_sensitivity`, which caches all supplied inputs.

The pipeline copies models and runs them in eval mode. It never changes your
original weights. Calibration input caches are bounded by batch count, **not by
bytes**; use small batches/chunks for long audio. Reconstruction additionally
caches one layer's activations at a time and can be expensive on large models.
Use deterministic, stateless inference: custom forwards that mutate buffers or
external state are not supported. Multi-device/sharded models are unsupported.

## Task-Level Gates

Default output NMSE measures agreement with the pretrained model, not separation
against ground-truth stems. Supply a higher-is-better score for task-level
selection. The callback receives the candidate simulation; it should not mutate
the model. Use an independent final test set after tuning on validation data.

```python
from npu_quant import waveform_metrics

# labeled_validation contains non-silent, aligned reference stems on the same
# device as predictions, with shape [batch, stems, ..., time].
def score_fn(candidate):
    values = []
    for mixture, targets in labeled_validation:
        prediction = candidate(mixture)
        values.append(waveform_metrics(targets, prediction)['snr_db'].flatten())
    return torch.cat(values).mean().item()

result = tool.optimize(calibration_inputs, validation_inputs, score_fn=score_fn)
```

SI-SDR is scale-invariant: an output can be much too quiet and still have a good
SI-SDR. Inspect `gain_db` and gain-sensitive SNR as well. Silent reference signals
raise explicitly; evaluate silence leakage separately rather than silently
dropping it. `bss_metrics(reference, estimate)` provides optional `mir_eval`
SDR/SIR/SAR for `[stems, time]`, with fixed stem ordering. It does not silently
downmix stereo or permute stems. BSS evaluation can be slow for long clips.

## Precision and Diagnostics

```python
config = QuantConfig(
    weight_bits=8,
    activation_bits=8,
    symmetric_activations=False,
    exclude=('decoder.output',),          # Exact name and its descendants
    overrides={'encoder.conv': (8, 16)},  # (weight bits, activation bits)
)
```

Use names from `model.named_modules()`. `''` denotes the root. Unknown names
raise instead of silently ignoring a typo. Exclusions and precision overrides
configure **simulation and rounding**, not NPU compiler precision. They do not
disable graph equalization of those modules. Transfer the chosen precision
policy to your vendor toolchain separately. A layer bypass does not undo weight
rounding already baked into `result.model` by AdaRound.

`quantization_report(result.simulation)` records each quantized module's weight,
input, and output scales, zero points, channel axis, and integer bounds. Large
scales alone do not establish the cause of failure. Compare actual output error
and layer sensitivity; a boundary zero point is normal for one-sided data.

The simulator uses calibrated min/max activation ranges, per-tensor activations,
and per-output-channel weights. Supported precision is 2-16 bits. Symmetric
weights use a **narrow signed range** (`[-127, 127]` for 8 bits); asymmetric
activations use unsigned `[0, 255]`. Rounding is PyTorch round-to-nearest-even.
Constant ranges include zero and remain representable. Calibration spans all
provided batches and freezes before evaluation; it is never reset on inference.

## ONNX and NPU

1. Establish a pretrained float quality baseline on representative audio.
2. Run the toolkit and inspect accepted/rejected stages and sensitivity.
3. Export `result.model` to float ONNX. The helper runs ONNX checker by default.
4. Quantize and compile with your NPU vendor's toolchain, using representative
   calibration data and supported precision overrides.
5. Compare ONNX runtime and compiled NPU outputs to the original model. Measure
   task metrics, output gain, silence leakage, and listen for pumping/artifacts.

`encodings.json` is **not a vendor config** and its module names are **not ONNX
tensor names**. Compiler fusion/renaming requires an explicit backend adapter.
AdaRound weights are stored as dequantized float values on fixed grids. To retain
the learned rounding, the backend must preserve the saved scales and conventions;
recomputing scales or requantizing differently can undo the benefit. Until your
backend supports this, start with BN folding/CLE and leave AdaRound disabled.

Export rejects the fake-quantized simulation. It does not generate QDQ ONNX or an
integer executable. The default exporter is PyTorch's legacy ONNX exporter
(`dynamo=False`) to require only `onnx`, not `onnxscript`. Its control-flow and
operator limitations still apply. Check exported runtime behavior, not just
structural validity; checker does not prove numerical equivalence.

The simulator does not model integer bias/accumulation, vendor requantization,
residual-add quantizers, LUT approximations, or RMSNorm internal square/reduce/
sqrt/reciprocal precision. Normalizations remain floating-point. It can therefore
underestimate your real NPU's RMSNorm error. Prefer backend-supported higher
precision for sensitive reductions when measured necessary; FP16 squares can
also overflow. This toolkit cannot make an unsupported operator hardware-native.

## Unsupported Graphs

FX tracing failures raise clearly. Custom RMSNorm classes may trace into primitive
operations and are not automatically smoothed. SiLU/GELU are not crossed by CLE.
Grouped/depthwise convolution weights can be simulated and rounded, but are not
equalized. ConvTranspose, functional Conv/Linear calls, and custom subclasses are
not quantized automatically. Root standard Conv/Linear modules are supported.
Unexecuted supported layers fail calibration; cover required branches explicitly.

To use simulation/reconstruction without FX rewrites:

```python
result = tool.optimize(
    calibration_inputs, validation_inputs,
    fold_bn=False, equalize=False, smooth=False,
)
```

Low-level APIs (`optimize_graph`, `calibrate_simulation`, `adaround`,
`bias_correct`) are available independently. Low-level reconstruction is not
end-to-end quality-gated; the pipeline adds that gate. When simulating an
AdaRound result independently, pass its fixed encodings:

```python
from npu_quant import adaround, calibrate_simulation

rounded, rows = adaround(model, calibration_inputs, iterations=100)
sim = calibrate_simulation(
    rounded, calibration_inputs,
    weight_encodings={row['layer']: row for row in rows},
)
```

These low-level calls replay calibration data, so use a list or recreate a
generator. Direct `calibrate_simulation` expects inputs already on the model's
device. Save/load simulation state into a freshly constructed matching wrapper
structure and config; plain module `state_dict` does not store Python config.

## Tests and Example

```sh
python -m unittest discover -s tests -v
python examples/demo.py
python examples/demo.py --export optimized.onnx
```

The demo is a synthetic smoke test, not a trained separator or an SDR benchmark.
See [design notes](docs/design.md) for the mathematical corrections to the saved
Gemini work and algorithm references.
