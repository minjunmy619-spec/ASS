# ONE Mixed Precision Suggestions

This report uses stock ONE mixed-precision behavior. It does not modify ONE source.

Generated qconfigs exclude memory/layout ops by default to avoid int16 islands made of reshape, slice, pad, concat, or transpose boundaries.

## Top Candidates

| rank | score | op | name | reasons |
|---:|---:|---|---|---|
| 1 | 93.2 | `CONV_2D` | `/output/Conv` | quality-sensitive CONV_2D; graph output path; name hint: out; late-stage node |
| 2 | 90.5 | `SOFTMAX` | `/Softmax_1` | quality-sensitive SOFTMAX; name hint: softmax; late-stage node |
| 3 | 83.6 | `BATCH_MATMUL` | `/MatMul_2` | quality-sensitive BATCH_MATMUL; late-stage node |
| 4 | 83.6 | `BATCH_MATMUL` | `/MatMul_3` | quality-sensitive BATCH_MATMUL; late-stage node |
| 5 | 80.5 | `SOFTMAX` | `/Softmax` | quality-sensitive SOFTMAX; name hint: softmax |
| 6 | 73.6 | `BATCH_MATMUL` | `/MatMul` | quality-sensitive BATCH_MATMUL |
| 7 | 73.6 | `BATCH_MATMUL` | `/MatMul_1` | quality-sensitive BATCH_MATMUL |
| 8 | 67.1 | `CONV_2D` | `/value_proj_1/Conv` | quality-sensitive CONV_2D; name hint: value; late-stage node |
| 9 | 67.1 | `CONV_2D` | `/key_proj_1/Conv` | quality-sensitive CONV_2D; name hint: key; late-stage node |
| 10 | 65.1 | `CONV_2D` | `/output/conv_6/Conv` | quality-sensitive CONV_2D; name hint: out; late-stage node |
| 11 | 64.6 | `CONV_2D` | `/value/conv_3/Conv` | quality-sensitive CONV_2D; name hint: value; late-stage node |
| 12 | 55.3 | `CONV_2D` | `/input/conv_1/Conv` | quality-sensitive CONV_2D; late-stage node |

## Generated QConfigs

- `qconfig_top12_int16.json`: top-ranked individual nodes (12 layers)
- `qconfig_best_island4_int16.json`: local graph island around best node (1 layers)
- `qconfig_best_island8_int16.json`: local graph island around best node (1 layers)
- `qconfig_best_island12_int16.json`: local graph island around best node (1 layers)
- `qconfig_depth_front_25_int16.json`: AMPQ-style depth front split at 0.25 (22 layers)
- `qconfig_depth_back_25_int16.json`: AMPQ-style depth back split at 0.25 (23 layers)
- `qconfig_depth_front_50_int16.json`: AMPQ-style depth front split at 0.5 (50 layers)
- `qconfig_depth_back_50_int16.json`: AMPQ-style depth back split at 0.5 (51 layers)
- `qconfig_depth_front_75_int16.json`: AMPQ-style depth front split at 0.75 (77 layers)
- `qconfig_depth_back_75_int16.json`: AMPQ-style depth back split at 0.75 (79 layers)

## How To Test One Proposal

```bash
one-quantize \
  --input_path <model.opt.circle> \
  --output_path <model.mixed.q.circle> \
  --input_data <calib.h5-or-list.txt> \
  --input_data_format <h5-or-list> \
  --quantized_dtype uint8 \
  --granularity channel \
  --input_type uint8 \
  --output_type uint8 \
  --quant_config <one-of-the-json-files>
```

Prefer small contiguous islands first. Isolated int16 nodes can add Quantize boundaries.
