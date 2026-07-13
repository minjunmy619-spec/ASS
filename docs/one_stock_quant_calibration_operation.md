# ONE Stock Quantization Calibration Operation

## Goal

Improve uint8 audio separation quality without modifying the ONE compiler source.

Stock ONE does not expose activation histograms from `record-minmax`; it records min/max metadata and supports `percentile` and `moving_average` calibration modes. Therefore this workflow treats MSE calibration as an external model-selection loop:

1. generate several stock ONE quantized candidates,
2. evaluate output MSE or ASS separation metrics,
3. keep the fastest candidate whose quality drop is acceptable,
4. optionally use stock AMPQ for small selective int16 regions.

This keeps runtime latency comparable to normal ONE uint8 except for any AMPQ-selected int16 islands.

## Tool

Use:

```bash
/home/cmj/works/ASS/tools/online/run_one_stock_quant_sweep.py
```

It only calls existing ONE tools:

- `one-quantize`
- `record-minmax`, indirectly through `one-quantize`
- `circle-quantizer`, indirectly through `one-quantize`
- `circle-mpqsolver`, indirectly through `one-quantize --ampq`
- `circle-inspect`, optionally for operator counts

## Basic Percentile And AMPQ Sweep

With an HDF5 calibration file:

```bash
python /home/cmj/works/ASS/tools/online/run_one_stock_quant_sweep.py \
  --input-circle /path/to/model.opt.circle \
  --calib-data /path/to/calibration.h5 \
  --output-dir /home/cmj/works/ASS/logs/one_stock_quant_sweep/your_model \
  --modes percentile,ampq \
  --min-percentiles 0.01,0.05,0.1,0.5,1.0 \
  --max-percentiles 99.0,99.5,99.9,99.95,99.99 \
  --ampq-qerror-ratios 0.01,0.03,0.05
```

With a text file list of raw calibration records:

```bash
python /home/cmj/works/ASS/tools/online/run_one_stock_quant_sweep.py \
  --input-circle /path/to/model.opt.circle \
  --calib-data /path/to/calibration_files.txt \
  --modes percentile,ampq \
  --min-percentiles 0.01,0.05,0.1,0.5,1.0 \
  --max-percentiles 99.0,99.5,99.9,99.95,99.99
```

The script uses `--input-data-format auto` by default. It maps `.h5` / `.hdf5` to `h5`, `.txt` / `.lst` / `.list` / `.filelist` to `list`, and directories to `directory`. You can override this with `--input-data-format list`.

## Include Stock Output MSE Evaluation

If you already have test data in ONE input-data format:

```bash
python /home/cmj/works/ASS/tools/online/run_one_stock_quant_sweep.py \
  --input-circle /path/to/model.opt.circle \
  --calib-data /path/to/calibration.h5 \
  --test-data /path/to/test.h5 \
  --evaluate-result \
  --print-mse \
  --modes percentile,moving_average,ampq
```

Stock `one-quantize` has one `--input_data_format` flag, so when `--evaluate-result` is used, keep calibration data and test data in the same ONE input-data format.

The script writes:

- one quantized `.q.circle` per candidate,
- one log per candidate,
- `summary.json` with command, return code, elapsed time, parsed MSE when printed, per-output MSE values, and optional Circle operator counts.

The parsed MSE fields are:

- `mse_primary`: first reported output MSE, usually the separated-output tensor.
- `mse_mean`: mean MSE across all reported outputs, including state outputs.
- `mse_by_output`: raw output-name to MSE mapping from `circle-eval-diff`.

For streaming models with state outputs, rank first by `mse_primary`, then inspect `mse_mean` to make sure state quantization is not obviously worse.

## Faster Sweep Subsets

Full streaming calibration/evaluation H5 files can be slow.  The wrapper can create limited H5/list files inside `--output-dir` before launching `one-quantize`:

```bash
python /home/cmj/works/ASS/tools/online/run_one_stock_quant_sweep.py \
  --input-circle /path/to/model.opt.circle \
  --calib-data /path/to/calibration.h5 \
  --calib-record-limit 16 \
  --test-data /path/to/test.h5 \
  --test-record-limit 8 \
  --evaluate-result \
  --print-mse \
  --modes percentile,moving_average
```

This is intended for quick calibration-knob ranking.  After selecting a setting, rerun that setting with the full representative calibration H5.

Use `--stream-output` when actively debugging a candidate; otherwise the wrapper keeps candidate output in per-run logs and prints compact result lines.

## Recommended Selection Order

1. First compare pure uint8 percentile candidates.
2. Try `moving_average` if percentile is unstable across calibration clips.
3. Use `ampq` only when pure uint8 quality is still too low.
4. Rank by ASS quality metric first, then latency/operator mix.

For the audio separator, start with channel-wise weights:

```bash
--quantized-dtype uint8 --granularity channel --input-type uint8 --output-type uint8
```

## Important Limitation

This is not true per-tensor histogram/MSE calibration inside `record-minmax`. Stock ONE does not provide the activation distribution data needed for that. The stock-only alternative is to sweep the exposed calibration knobs and select by external MSE or separation quality.

If later you decide ONE source changes are allowed, the right compiler-side target would be a histogram or sampled-MSE observer inside `record-minmax`, but that is intentionally outside this workflow.
