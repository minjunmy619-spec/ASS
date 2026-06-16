# NPU Variant Verification Summary

- Total: 1
- PASS: 1
- FAIL: 0

| Kind | Variant | Status | Fail Stage | Low-latency opt | Opt Transpose | Opt Reshape | Opt StridedSlice |
|---|---|---|---|---:|---:|---:|---:|
| recipe | source-aware-melband-loco-cnb.student-npu-residual-sfx-lowlat.rt192k.fp512keep475 | PASS | - | True | 48 | 18 | 8 |

## Circle optimized top operators

### recipe:source-aware-melband-loco-cnb.student-npu-residual-sfx-lowlat.rt192k.fp512keep475

```text
MUL: 268
CONV_2D: 165
ADD: 148
LOGISTIC: 116
MEAN: 73
RSQRT: 69
SPLIT_V: 66
DEPTHWISE_CONV_2D: 50
TRANSPOSE: 48
CONCATENATION: 40
PAD: 31
SUB: 24
```

