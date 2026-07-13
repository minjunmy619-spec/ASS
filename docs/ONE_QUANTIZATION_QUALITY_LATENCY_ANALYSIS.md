# ONE Quantization Quality/Latency Analysis

Date: 2026-07-12

Scope:

- ONE source checkout: `/home/cmj/works/ONE` at `de7f4736dc`
- ASS checkout: `/home/cmj/works/ASS` at `d89432b`
- Goal: reduce audio separation quality loss from uint8 quantization while avoiding the large latency increase seen with full int16.

Note: the ONE worktree appears heavily locally modified. This document is based on the live source files, not on an upstream release tag.

## Short Recommendation

Do not jump directly from full uint8 to full int16. Since we should not modify the ONE compiler source, use the existing ONE quantization controls first:

1. **Stay uint8 and sweep existing calibration ranges.**
   This has the best latency profile because it only changes quantization parameters. Use `tools/online/run_one_stock_quant_sweep.py` to sweep stock `percentile` and `moving_average` modes.

2. **Use channel-wise uint8 weights plus calibrated per-layer uint8 activations.**
   The ASS verifier already emits `granularity=channel`, which is important because ONE activation quantization remains per-layer while weights and bias can be per-channel.

3. **Use selective int16 only for a few quality-critical islands through stock AMPQ.**
   ONE supports mixed uint8/int16 through `quant_config` and `one-quantize --ampq`. Keep int16 islands contiguous and small, otherwise Quantize boundaries can erase the latency benefit.

True per-tensor histogram/MSE calibration inside `record-minmax` would require ONE source changes because stock `record-minmax` does not expose activation histograms. For the current constraint, treat MSE as an external selection metric over stock-generated candidates.

## Current ONE Quantization Flow

The high-level `one-quantize` wrapper is a three-stage post-training quantization flow:

1. `circle-quantizer --quantize_dequantize_weights`
2. `record-minmax`
3. `circle-quantizer --quantize_with_minmax`

Source evidence:

- `compiler/one-cmds/one-quantize:455-566` builds and runs the weight fake-quant, min/max recording, and final quantize commands.
- `compiler/one-cmds/one-quantize:89-152` exposes `quantized_dtype`, `granularity`, `input_type`, `output_type`, calibration percentiles, moving average mode, and `save_min_max`.
- `compiler/one-cmds/one-quantize:328-341` defaults to `mode=percentile`, `min_percentile=1.0`, `max_percentile=99.0`, and `bisection_type=auto`.
- `compiler/one-cmds/onecc.template.cfg:97-115` confirms onecc config keys for quantization.

Important implication:

- If `input_data` is absent, `record-minmax` can run random calibration. The driver warns this does not represent inference workload (`compiler/record-minmax/driver/Driver.cpp:58-61`). For source separation this is usually unacceptable.

## How Uint8 Loses Quality

ONE activation quantization is per-layer, not per-channel:

- `QuantizeActivation` only accepts one min/max per activation tensor and asserts `min.size() == 1` and `max.size() == 1` (`compiler/luci/pass/src/QuantizeActivation.cpp:59-86`).
- `QuantizeWithMinMaxPass` explicitly says activation is quantized from recorded min/max, while weights and bias are handled later (`compiler/luci/pass/src/QuantizeWithMinMaxPass.cpp:515-549`).
- The command help says weight granularity can be `layer` or `channel`, but activation is per-layer (`compiler/one-cmds/one-quantize:93-97` and `compiler/circle-mpqsolver/src/CircleMPQSolver.cpp:115-119`).

For uint8, ONE uses asymmetric scale/zero-point:

- `compute_asym_scale_zp` maps min/max to `[0, 255]`, forces zero into range, and clips very small scale to `1e-5` (`compiler/luci/pass/src/QuantizationUtils.cpp:150-208`).
- Positive-only and negative-only ranges are treated specially and can print warnings; the prior ASS note confirms these warnings are diagnostic, not fatal, when artifacts are produced and `circle-verify` passes (`docs/adaptive_mel_loco_quant_positive_operation.md:19-26`).

For int16, ONE uses symmetric scale:

- `compute_sym_scale` maps to signed int range and uses a much smaller minimum scale for int16 (`1e-8`) than uint8 (`compiler/luci/pass/src/QuantizationUtils.cpp:116-148`).

This matches your observation: full int16 keeps quality because activation resolution is much higher, but latency increases because the whole graph moves to int16 kernels and larger memory traffic.

## Weight/Bias Granularity

Channel-wise weight quantization is already implemented and should remain the default for the separator:

- `circle-quantizer --quantize_with_minmax` supports output dtypes `uint8` and `int16` and granularity `layer` and `channel` (`compiler/luci/pass/src/CircleQuantizer.cpp:478-485`).
- Layer-wise quantization only supports uint8 (`compiler/luci/pass/src/CircleQuantizer.cpp:541-543`).
- Channel-wise weight quantization computes per-channel min/max and scale, then stores `quantized_dimension` (`compiler/luci/pass/src/QuantizeWeights.cpp:294-329`).
- Bias quantization uses layer-wise input scale plus per-channel weight scale when granularity is channel-wise (`compiler/luci/pass/src/QuantizeBias.cpp:190-238`).

ASS already follows this path:

- `tools/online/verify_npu_variants.py:869-877` emits `quantized_dtype=uint8`, configurable `granularity`, `input_type=uint8`, and `output_type=uint8`.
- `docs/NPU_VARIANT_VERIFIER_GUIDE.md:210-219` says channel granularity is preferred for accuracy and layer fallback is an escape hatch.

## Mixed Precision Support

ONE has two mixed-precision mechanisms.

### Manual `quant_config`

`circle-quantizer` can read per-layer dtype/granularity from JSON:

- Config parser reads `name`, `names`, `dtype`, `granularity`, and `alternate` mappings (`compiler/circle-quantizer/src/CircleQuantizer.cpp:67-143`).
- Tests show a default uint8 model with selected int16 layers (`compiler/one-cmds/tests/one-quantize_022.qconf.json:1-20`).

Example shape:

```json
{
  "default_quantization_dtype": "uint8",
  "default_granularity": "channel",
  "layers": [
    {
      "names": [
        "final_mask_projection",
        "decoder_output_projection"
      ],
      "dtype": "int16",
      "granularity": "channel"
    }
  ]
}
```

In the quantizer pass, selected int16 nodes get Quantize ops at their boundaries:

- `QuantizeWithMinMaxPass` looks up layer-specific dtype and granularity, otherwise uses the default dtype (`compiler/luci/pass/src/QuantizeWithMinMaxPass.cpp:566-588`).
- If a node dtype differs from default, `InsertQuantizeOp` wraps its inputs/outputs (`compiler/luci/pass/src/QuantizeWithMinMaxPass.cpp:590-624`).
- Constants are directly quantized to the selected op dtype rather than degraded through uint8 first (`compiler/luci/pass/src/QuantizeWithMinMaxPass.cpp:149-162`).

Latency caution:

- Every isolated int16 node can create extra Quantize boundaries. Keep int16 regions contiguous.
- `RemoveQDQForMixedPrecisionOpPass` currently recognizes only `FULLY_CONNECTED` and `BATCH_MATMUL` (`compiler/luci/pass/src/RemoveQDQForMixedPrecisionOpPass.cpp:68-76`), so Conv2D-heavy separators should not expect all mixed boundaries to disappear.

### AMPQ Solver

`one-quantize --ampq` invokes `circle-mpqsolver`:

- The wrapper records min/max first, then calls `circle-mpqsolver` (`compiler/one-cmds/one-quantize:690-877`).
- Bisection compares full int16 and full uint8 fake-quant output MAE, then searches for a mixed model under a target error ratio (`compiler/circle-mpqsolver/src/bisection/BisectionSolver.cpp:123-152`).
- It writes `FinalConfiguration.mpq.json` when `--save_intermediate` is used (`compiler/circle-mpqsolver/src/core/Dumper.cpp:90-95` and `compiler/circle-mpqsolver/src/core/DumpingHooks.cpp:59-66`).
- The scoring metric is output MAE against fp32 (`compiler/circle-mpqsolver/src/core/ErrorMetric.cpp:28-75`).

Limitations:

- Bisection is depth-split based, not latency-aware (`compiler/circle-mpqsolver/README.md:7-28`).
- The auto direction uses VISQ layer errors if available (`compiler/circle-mpqsolver/src/bisection/VISQErrorApproximator.cpp:45-62`).
- The MAE proxy may not track SI-SDR, SDR, stem leakage, or perceptual artifact severity.
- `circle-mpqsolver` has fixed patterns for layernorm and decomposed softmax only (`compiler/circle-mpqsolver/src/pattern/PatternResolver.cpp:192-206` and `320-349`).

## Optimization Passes That Matter

Optimization does not solve quantization precision directly, but it can reduce latency and avoid extra quantized operators:

- The O1 group contains folding, conv/add/mul fusion, activation fusion, redundant reshape/transpose removal, and quantization compatibility passes (`compiler/one-cmds/onelib/constant.py:21-93`).
- `CircleOptimizer` runs canonicalization, shape/type inference, then selected passes under restart strategy (`compiler/luci/pass/src/CircleOptimizer.cpp:266-428`).
- `FuseAddWithConvPass` folds channel bias additions into Conv2D bias (`compiler/luci/pass/src/FuseAddWithConvPass.cpp:49-138`).
- `FuseMulWithConvPass` folds scalar/channel multipliers into Conv2D weights/bias (`compiler/luci/pass/src/FuseMulWithConvPass.cpp:65-184`).
- `RemoveRedundantQuantizePass` removes identical or subsequent Quantize ops (`compiler/luci/pass/src/RemoveRedundantQuantizePass.cpp:21-83`).

ASS verifier already enables low-latency optimize flags including redundant transpose/reshape removal, unnecessary arithmetic removal, and conv add/mul fusion (`tools/online/verify_npu_variants.py:27-36`).

## Audio-Specific Calibration Plan

This is the first thing to try because it should not increase inference latency.

Use a representative calibration set, not random data:

- Use post-wrapper NPU inputs: spectrogram features after the same normalization used in deployment.
- Include quiet speech, music-only, effects-only, dense mixture, transients, and low-SNR examples.
- For stateful models, include realistic non-zero streaming states after warm-up. A calibration set with only zero caches can underrepresent state-dependent activations.
- Keep calibration chunk shape exactly equal to deployment shape.

Run a calibration sweep while keeping the graph uint8/channel:

```ini
[one-quantize]
input_path=<model.opt.circle>
output_path=<model.q.p001_999.circle>
input_data=<calib.h5 or calib_files.txt>
input_data_format=<h5-or-list>
quantized_dtype=uint8
granularity=channel
input_type=uint8
output_type=uint8
mode=percentile
min_percentile=0.1
max_percentile=99.9
```

Sweep:

- `0.0 / 100.0`: no percentile clipping, useful to detect clipping harm.
- `0.1 / 99.9`
- `0.5 / 99.5`
- `1.0 / 99.0`: current default.
- `2.0 / 98.0`: stronger clipping, useful if outliers are wasting uint8 range.
- `mode=moving_average`, `moving_avg_batch=16`, `moving_avg_const=0.05` and `0.1`.

Measure:

- Artifact truth: `model.circle`, `model.opt.circle`, `model.q.circle`.
- `circle-verify` on each quantized model.
- ONE `evaluate_result` MAE/MSE as a quick compiler-side proxy.
- Real separation metrics on the same validation clips: SI-SDR/SDR per stem, speech/music/effects leakage, and listening notes for transient/ambience damage.
- Latency and operator count, because qparam changes should not change graph topology.

Expected result:

- If one percentile setting recovers most of the lost quality with unchanged latency, keep uint8.
- If all uint8 settings fail similarly, the quality loss is probably sensitivity of specific layers, not just bad calibration ranges.

## Selective Int16 Plan

Start from default uint8/channel and promote only high-impact, low-cost regions.

First candidates for `BandSFCNetNPU` and related SFC students:

1. Final mask/output projection and residual-SFX head.
2. SFC query/expansion path around K-to-F reconstruction.
3. Cross-attention logits/softmax path if the model uses `*_crossattn_query`.
4. Late decoder stages nearest output, especially if errors sound like mask quantization noise.
5. Early compressor only if AMPQ/VISQ says front-end errors dominate.

Avoid:

- Promoting every Conv2D in a stage independently.
- Scattering isolated int16 nodes throughout the graph.
- Full int16 input/output unless measured latency is acceptable.

AMPQ commands to generate a starting config:

```bash
one-quantize \
  --input_path <model.opt.circle> \
  --output_path <model.ampq.q.circle> \
  --input_data <calib.h5 or calib_files.txt> \
  --input_data_format <h5-or-list> \
  --quantized_dtype uint8 \
  --granularity channel \
  --input_type uint8 \
  --output_type uint8 \
  --mode percentile \
  --min_percentile 0.5 \
  --max_percentile 99.5 \
  --ampq \
  --ampq_algorithm bisection \
  --ampq_qerror_ratio 0.03 \
  --bisection_type auto \
  --save_intermediate
```

Then inspect:

- `FinalConfiguration.mpq.json`
- `errors.mpq.txt`
- intermediate `Configuration_*.mpq.json`

Prune or rewrite the final config manually if AMPQ selects too many int16 layers. Then rerun ordinary quantization with:

```bash
one-quantize \
  --input_path <model.opt.circle> \
  --output_path <model.manual_mpq.q.circle> \
  --input_data <calib.h5 or calib_files.txt> \
  --input_data_format <h5-or-list> \
  --quantized_dtype uint8 \
  --granularity channel \
  --input_type uint8 \
  --output_type uint8 \
  --quant_config <edited_qconf.json>
```

Suggested AMPQ sweep:

- `ampq_qerror_ratio=0.01`
- `0.03`
- `0.05`
- Add wider values only if these still behave like pure uint8.

Try both:

- `--bisection_type i16_back`: likely better for mask/output damage.
- `--bisection_type i16_front`: useful if SFC compression quantization is the dominant damage.

## Stock-Only Constraint

Do not patch ONE for this workflow. True per-tensor histogram/MSE calibration would need activation distribution data that stock `record-minmax` does not emit. The practical stock-only replacement is:

- sweep `percentile` and `moving_average` calibration,
- evaluate model output MSE or ASS separation metrics outside ONE,
- use stock AMPQ for selective int16 candidates,
- keep the candidate with the best quality/latency tradeoff.

The helper for this is `tools/online/run_one_stock_quant_sweep.py`.

## Recommended Experiment Order

1. Baseline current uint8/channel result.
   Record quality metrics, latency, operator counts, and logs.

2. Calibration sweep.
   Try percentile and moving-average settings with the same model and calibration data.

3. Calibration data upgrade.
   Replace random or narrow calibration with real streaming warm-state calibration.

4. AMPQ bisection sweep.
   Use `--save_intermediate`; inspect and prune `FinalConfiguration.mpq.json`.

5. Manual qconfig small islands.
   Start with final output head and SFC expansion/softmax path.

6. If still bad, reduce the candidate set to the best stock percentile/AMPQ variants and run full ASS evaluation.

## Practical Pass/Fail Gate

Accept a candidate only if all are true:

- `model.q.circle` exists and `circle-verify` passes.
- Latency remains within the uint8 budget or only slightly above it.
- SI-SDR/SDR drop versus fp32 is within the project threshold.
- No stem has an obvious failure mode: speech musical noise, music pumping, effects loss, or transient smearing.
- The same calibration and validation clips are used across candidates.

## Bottom Line

The source code points to a clear strategy:

- Full uint8 is fast but fragile because activation quantization is per-layer and depends heavily on calibration min/max.
- Full int16 is accurate but slow because every activation and weight path is wider.
- The best near-term path is **uint8/channel with better calibration**, then **small, contiguous int16 islands selected by AMPQ or qconfig**.
- Under the no-ONE-source-change constraint, the right tool is a stock quantization sweep plus external MSE/ASS evaluation, not a compiler patch.
