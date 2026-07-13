# ONE Mixed Precision Suggestions

This report uses stock ONE mixed-precision behavior. It does not modify ONE source.

Generated qconfigs exclude memory/layout ops by default to avoid int16 islands made of reshape, slice, pad, concat, or transpose boundaries.

## Top Candidates

| rank | score | op | name | reasons |
|---:|---:|---|---|---|
| 1 | 95.0 | `TRANSPOSE_CONV` | `/decoder/up/up.0/bn/BatchNormalization;/decoder/up/up.0/act/Relu/TransposeConv` | quality-sensitive TRANSPOSE_CONV; name hint: decoder; late-stage node; matched prefer-regex |
| 2 | 94.8 | `TRANSPOSE_CONV` | `/decoder/up/up.1/bn/BatchNormalization;/decoder/up/up.1/act/Relu/TransposeConv` | quality-sensitive TRANSPOSE_CONV; name hint: decoder; late-stage node; matched prefer-regex |
| 3 | 94.6 | `TRANSPOSE_CONV` | `/decoder/up/up.2/bn/BatchNormalization;/decoder/up/up.2/act/Relu/TransposeConv` | quality-sensitive TRANSPOSE_CONV; name hint: decoder; late-stage node; matched prefer-regex |
| 4 | 94.3 | `TRANSPOSE_CONV` | `/decoder/up/up.3/bn/BatchNormalization;/decoder/up/up.3/act/Relu/TransposeConv` | quality-sensitive TRANSPOSE_CONV; name hint: decoder; late-stage node; matched prefer-regex |
| 5 | 90.4 | `CONV_2D` | `/decoder/output/Conv` | quality-sensitive CONV_2D; name hint: out,decoder; late-stage node; matched prefer-regex |
| 6 | 74.6 | `CONV_2D` | `/time_proj/conv_7/Conv` | quality-sensitive CONV_2D; late-stage node; matched prefer-regex |
| 7 | 74.6 | `CONV_2D` | `/ffn/ffn.1/conv_7/Conv` | quality-sensitive CONV_2D; late-stage node; matched prefer-regex |
| 8 | 74.2 | `CONV_2D` | `/conv_7/Conv;/act_7/Relu` | quality-sensitive CONV_2D; late-stage node; matched prefer-regex |
| 9 | 72.8 | `CONV_2D` | `/ffn/ffn.0/conv_7/Conv;/ffn/ffn.0/act_7/Relu` | quality-sensitive CONV_2D; late-stage node; matched prefer-regex |
| 10 | 64.6 | `CONV_2D` | `/freq_mix/freq_mix.1/conv/Conv` | quality-sensitive CONV_2D; matched prefer-regex |
| 11 | 64.6 | `CONV_2D` | `/time_proj/conv/Conv` | quality-sensitive CONV_2D; matched prefer-regex |
| 12 | 64.6 | `CONV_2D` | `/ffn/ffn.1/conv/Conv` | quality-sensitive CONV_2D; matched prefer-regex |
| 13 | 64.6 | `CONV_2D` | `/freq_mix/freq_mix.1/conv_1/Conv` | quality-sensitive CONV_2D; matched prefer-regex |
| 14 | 64.6 | `CONV_2D` | `/time_proj/conv_1/Conv` | quality-sensitive CONV_2D; matched prefer-regex |
| 15 | 64.6 | `CONV_2D` | `/ffn/ffn.1/conv_1/Conv` | quality-sensitive CONV_2D; matched prefer-regex |
| 16 | 64.6 | `CONV_2D` | `/freq_mix/freq_mix.1/conv_2/Conv` | quality-sensitive CONV_2D; matched prefer-regex |
| 17 | 64.6 | `CONV_2D` | `/time_proj/conv_2/Conv` | quality-sensitive CONV_2D; matched prefer-regex |
| 18 | 64.6 | `CONV_2D` | `/ffn/ffn.1/conv_2/Conv` | quality-sensitive CONV_2D; matched prefer-regex |
| 19 | 64.6 | `CONV_2D` | `/freq_mix/freq_mix.1/conv_3/Conv` | quality-sensitive CONV_2D; matched prefer-regex |
| 20 | 64.6 | `CONV_2D` | `/time_proj/conv_3/Conv` | quality-sensitive CONV_2D; matched prefer-regex |

## Generated QConfigs

- `qconfig_top20_int16.json`: top-ranked individual nodes (20 layers)
- `qconfig_best_island3_int16.json`: local graph island around best node (3 layers)
- `qconfig_best_island5_int16.json`: local graph island around best node (5 layers)
- `qconfig_best_island8_int16.json`: local graph island around best node (8 layers)
- `qconfig_depth_front_25_int16.json`: AMPQ-style depth front split at 0.25 (22 layers)
- `qconfig_depth_back_25_int16.json`: AMPQ-style depth back split at 0.25 (37 layers)
- `qconfig_depth_front_50_int16.json`: AMPQ-style depth front split at 0.5 (44 layers)
- `qconfig_depth_back_50_int16.json`: AMPQ-style depth back split at 0.5 (60 layers)

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
