# Simplified ONNX Import Summary

- Total: 22
- PASS: 6
- FAIL: 16

| Model | Status | Reason |
|---|---|---|
| online-hard-band-sfc2d.causal96dim.12l.musical64 | PASS | - |
| online-hard-band-sfc2d.mel.causal96dim.12l | PASS | - |
| online-hierarchical-soft-band-ffi-sfc2d.speech-lowfreq-narrow.causal96dim.1-2-2l | FAIL | loc("/mid_expander/Clip/min"): error: 'Circle.minimum' op operand #1 must be tensor of 32-bit float or 32/64-bit signless integer values, but got 'none' |
| online-hierarchical-soft-band-parallel-ffi-sfc2d.rt128k.speech-lowfreq-narrow.causal14dim.0-1-1l.128-96-48b | FAIL | loc("/mid_expander/Clip/min"): error: 'Circle.minimum' op operand #1 must be tensor of 32-bit float or 32/64-bit signless integer values, but got 'none' |
| online-hierarchical-soft-band-parallel-ffi-sfc2d.rt192k.speech-lowfreq-narrow.causal20dim.0-1-1l.128-96-48b | FAIL | loc("/mid_expander/Clip/min"): error: 'Circle.minimum' op operand #1 must be tensor of 32-bit float or 32/64-bit signless integer values, but got 'none' |
| online-hierarchical-soft-band-sfc2d.causal96dim.1-2-2l.musical128 | FAIL | loc("/mid_expander/Clip/min"): error: 'Circle.minimum' op operand #1 must be tensor of 32-bit float or 32/64-bit signless integer values, but got 'none' |
| online-hierarchical-soft-band-sfc2d.mel.causal96dim.1-2-2l | FAIL | loc("/mid_expander/Clip/min"): error: 'Circle.minimum' op operand #1 must be tensor of 32-bit float or 32/64-bit signless integer values, but got 'none' |
| online-sfc2d.causal96dim.12l.musical64 | FAIL | loc("/decoder/Clip/min"): error: 'Circle.minimum' op operand #1 must be tensor of 32-bit float or 32/64-bit signless integer values, but got 'none' |
| online-soft-band-dilated-sfc2d.causal96dim.12l.musical64 | FAIL | loc("/expander/Clip/min"): error: 'Circle.minimum' op operand #1 must be tensor of 32-bit float or 32/64-bit signless integer values, but got 'none' |
| online-soft-band-dilated-sfc2d.mel.causal96dim.12l | FAIL | loc("/expander/Clip/min"): error: 'Circle.minimum' op operand #1 must be tensor of 32-bit float or 32/64-bit signless integer values, but got 'none' |
| online-soft-band-gru-sfc2d.causal96dim.12l.musical64 | FAIL | loc("/expander/Clip/min"): error: 'Circle.minimum' op operand #1 must be tensor of 32-bit float or 32/64-bit signless integer values, but got 'none' |
| online-soft-band-gru-sfc2d.mel.causal96dim.12l | FAIL | loc("/expander/Clip/min"): error: 'Circle.minimum' op operand #1 must be tensor of 32-bit float or 32/64-bit signless integer values, but got 'none' |
| online-soft-band-query-sfc2d.causal96dim.12l.musical64 | PASS | - |
| online-soft-band-query-sfc2d.mel.causal96dim.12l | PASS | - |
| online-soft-band-query-sfc2d.rt128k.causal16dim.6l.64b | PASS | - |
| online-soft-band-query-sfc2d.rt128k.fp512keep475.causal16dim.6l.64b | FAIL | loc("/compressor/Add"): error: 'Circle.add' op operands don't have broadcast-compatible shapes |
| online-soft-band-query-sfc2d.rt192k.causal24dim.6l.64b | PASS | - |
| online-soft-band-query-sfc2d.rt192k.fp512keep475.causal24dim.6l.64b | FAIL | loc("/compressor/Add"): error: 'Circle.add' op operands don't have broadcast-compatible shapes |
| online-soft-band-sfc2d.causal96dim.12l.musical64 | FAIL | loc("/expander/Clip/min"): error: 'Circle.minimum' op operand #1 must be tensor of 32-bit float or 32/64-bit signless integer values, but got 'none' |
| online-soft-band-sfc2d.mel.causal96dim.12l | FAIL | loc("/expander/Clip/min"): error: 'Circle.minimum' op operand #1 must be tensor of 32-bit float or 32/64-bit signless integer values, but got 'none' |
| online-soft-band-sfc2d.rt128k.fp512keep475.causal16dim.6l.64b | FAIL | loc("/compressor/Add"): error: 'Circle.add' op operands don't have broadcast-compatible shapes |
| online-soft-band-sfc2d.rt192k.fp512keep475.causal24dim.6l.64b | FAIL | loc("/compressor/Add"): error: 'Circle.add' op operands don't have broadcast-compatible shapes |
