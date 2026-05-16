# Add-Failure Retest With Correct Simplify Shape

- Total: 4
- PASS: 2
- FAIL: 2

| Model | Status | Detail |
|---|---|---|
| online-soft-band-query-sfc2d.rt128k.fp512keep475.causal16dim.6l.64b | PASS | shape=[1, 2, 1, 512] |
| online-soft-band-query-sfc2d.rt192k.fp512keep475.causal24dim.6l.64b | PASS | shape=[1, 2, 1, 512] |
| online-soft-band-sfc2d.rt128k.fp512keep475.causal16dim.6l.64b | FAIL | loc("/expander/Clip/min"): error: 'Circle.minimum' op operand #1 must be tensor of 32-bit float or 32/64-bit signless integer values, but got 'none' |
| online-soft-band-sfc2d.rt192k.fp512keep475.causal24dim.6l.64b | FAIL | loc("/expander/Clip/min"): error: 'Circle.minimum' op operand #1 must be tensor of 32-bit float or 32/64-bit signless integer values, but got 'none' |
