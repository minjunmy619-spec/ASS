# Design and Algorithm Notes

## Why Not Reuse the Saved Code?

`previous_works.md` is preserved as historical material, not executable or
validated implementation. Several claims there are unsafe:

1. **Pre-RMSNorm channel smoothing is not equivalent.** For channel scales `s`,
   dividing `x` by `s` and multiplying norm gamma by `s` preserves the numerator,
   but changes `sqrt(mean(x**2 / s**2) + eps)`. This denominator depends on the
   input and cannot generally be compensated by constant gamma. Scaling both
   residual branches does not fix this problem.
2. **Global scaling is not exactly invariant with unchanged epsilon.** For a
   positive scalar `k`, RMSNorm of `k*x` has effective epsilon `eps/k**2`.
   Changing epsilon to `k**2*eps` preserves the ideal expression with unchanged
   gamma, but does not automatically improve relative quantization resolution.
   This toolkit deliberately does not rewrite norm input scales or epsilon.
3. **Only some activations commute with positive scales.** ReLU does; SiLU and
   GELU generally do not. Branches and shared consumers require compensation at
   every affected edge. Registration order is not execution order.
4. **Assigning quantized weights through `.data` breaks reconstruction gradients.**
   AdaRound here uses `torch.func.functional_call` to connect output loss to the
   rounding parameters, and tests verify nonzero reconstruction gradients.
5. **Internal-layer bias correction needs feature inputs, not raw waveforms.**
   We hook full-model runs and weight each observation by its number of elements,
   including variable-length and uneven calibration batches.
6. **Fake-quantized or rounded float weights are not an NPU deployment artifact.**
   Operator coverage, graph fusion, accumulator precision, encoding conventions,
   and compiler support must be established separately.

## Implemented Math

BN folding uses PyTorch's eval-mode fusion implementation and checks supported
channel semantics, running statistics, aliases, and graph users. The graph stage
requires eval mode and verifies outputs. Equivalent float expressions may still
differ slightly through rounding.

For a safe two-layer chain with weight ranges `a` on first-layer output channels
and `b` on second-layer input channels, CLE uses `s = sqrt(b/a)`:

```text
W1' = s * W1; bias1' = s * bias1; W2' = W2 / s
```

Scales are positive and bounded, zero-range channels are left unchanged, and the
only supported intermediate nonlinearity is a standard module ReLU. No residual
path is crossed. Changes to hook-bearing paths are excluded.

For safe post-norm smoothing, `y = norm(x)` and `z = Linear(y)`. With positive
channel scales `s`:

```text
gamma' = gamma / s; beta' = beta / s; W_linear' = W_linear * s
```

This keeps the norm input and denominator unchanged. We use a bounded
activation/weight-range heuristic with exponent 0.5 and validate float
equivalence. It can improve the following Linear's input quantization, **not the
norm's internal energy calculation**.

AdaRound learns a stretched sigmoid rounding offset from local output MSE, with
a warmup and a gradually hardened regularizer. Float layer targets come from the
current graph-optimized reference, not from incompatible pre-CLE intermediate
features. Candidate inputs include already-rounded preceding layers. Hard learned
rounding and nearest rounding are compared on all cached local samples, retaining
the better one. The high-level pipeline then rechecks full-model simulated
quality on held-out data and may reject the whole stage.

Bias correction adds `mean(reference_output - candidate_output)` per channel.
This minimizes that local squared-error offset for cached samples. It does not
guarantee zero final waveform DC or absence of audible clicks; downstream
nonlinearities can turn local improvements into global regressions. The pipeline
therefore gates this stage too.

## Reference Material

The implementation is independent code based on the algorithms, not a vendored
copy of AIMET. No AIMET binaries or source are required at runtime.

- [Qualcomm AIMET repository](https://github.com/quic/aimet)
- [AIMET batch norm folding](https://github.com/quic/aimet/blob/develop/TrainingExtensions/torch/src/python/aimet_torch/batch_norm_fold.py): reviewed graph/folding constraints and encoding implications.
- [AIMET AdaRound optimizer](https://github.com/quic/aimet/blob/develop/TrainingExtensions/torch/src/python/aimet_torch/_base/adaround/adaround_optimizer.py): reviewed local activation sampling, differentiable reconstruction, and hard/soft rounding checks.
- Nagel et al., [Data-Free Quantization Through Weight Equalization and Bias Correction](https://arxiv.org/abs/1906.04721).
- Nagel et al., [Up or Down? Adaptive Rounding for Post-Training Quantization](https://arxiv.org/abs/2004.10568).
- Xiao et al., [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438).

AIMET is substantially broader: hardware-aware QuantSim, richer graph support,
training-based workflows, and backend integration are intentionally not replicated
here. The initial tool emphasizes small dependencies, explicit limitations, and
testable safety rather than claiming feature parity.

## Next Backend Step

Before selecting a backend adapter, obtain the NPU model/toolchain version,
supported weight/activation precisions, per-channel support, accumulator and bias
formats, rounding/saturation conventions, and an example encoding table. Then map
module encodings to the exported graph's tensor names and compare paired
intermediate outputs. Backend-specific calibration or mixed precision should be
driven by those measurements, not a universal scale threshold.
