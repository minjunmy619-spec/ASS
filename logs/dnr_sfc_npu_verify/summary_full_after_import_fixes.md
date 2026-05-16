# Full Pipeline Summary After Import Fixes

- Total: 22
- PASS: 0
- FAIL: 22

| Model | Status | Fail Stage | Source ONNX | Reason |
|---|---|---|---|---|
| online-hard-band-sfc2d.causal96dim.12l.musical64 | FAIL | quantize | model.sim.onnx | circle_quantizer:   what():  Unsupported non const input /compressor/MatMul/tr |
| online-hard-band-sfc2d.mel.causal96dim.12l | FAIL | quantize | model.sim.onnx | circle_quantizer:   what():  Unsupported non const input /compressor/MatMul/tr |
| online-hierarchical-soft-band-ffi-sfc2d.speech-lowfreq-narrow.causal96dim.1-2-2l | FAIL | quantize | model.sim.clipfix.onnx | circle_quantizer:   what():  Unsupported non const input /pre_compressor/MatMul/tr |
| online-hierarchical-soft-band-parallel-ffi-sfc2d.rt128k.speech-lowfreq-narrow.causal14dim.0-1-1l.128-96-48b | FAIL | quantize | model.sim.clipfix.onnx | circle_quantizer:   what():  Unsupported non const input /pre_compressor/MatMul/tr |
| online-hierarchical-soft-band-parallel-ffi-sfc2d.rt192k.speech-lowfreq-narrow.causal20dim.0-1-1l.128-96-48b | FAIL | quantize | model.sim.clipfix.onnx | circle_quantizer:   what():  Unsupported non const input /pre_compressor/MatMul/tr |
| online-hierarchical-soft-band-sfc2d.causal96dim.1-2-2l.musical128 | FAIL | quantize | model.sim.clipfix.onnx | circle_quantizer:   what():  Unsupported non const input /pre_compressor/MatMul/tr |
| online-hierarchical-soft-band-sfc2d.mel.causal96dim.1-2-2l | FAIL | quantize | model.sim.clipfix.onnx | circle_quantizer:   what():  Unsupported non const input /pre_compressor/MatMul/tr |
| online-sfc2d.causal96dim.12l.musical64 | FAIL | quantize | model.sim.clipfix.onnx | circle_quantizer:   what():  Unsupported non const input /compressor/MatMul/tr |
| online-soft-band-dilated-sfc2d.causal96dim.12l.musical64 | FAIL | quantize | model.sim.clipfix.onnx | circle_quantizer:   what():  Unsupported non const input /compressor/MatMul/tr |
| online-soft-band-dilated-sfc2d.mel.causal96dim.12l | FAIL | quantize | model.sim.clipfix.onnx | circle_quantizer:   what():  Unsupported non const input /compressor/MatMul/tr |
| online-soft-band-gru-sfc2d.causal96dim.12l.musical64 | FAIL | quantize | model.sim.clipfix.onnx | circle_quantizer:   what():  Unsupported non const input /compressor/MatMul/tr |
| online-soft-band-gru-sfc2d.mel.causal96dim.12l | FAIL | quantize | model.sim.clipfix.onnx | circle_quantizer:   what():  Unsupported non const input /compressor/MatMul/tr |
| online-soft-band-query-sfc2d.causal96dim.12l.musical64 | FAIL | quantize | model.sim.onnx | circle_quantizer:   what():  Unsupported non const input /compressor/MatMul/tr |
| online-soft-band-query-sfc2d.mel.causal96dim.12l | FAIL | quantize | model.sim.onnx | circle_quantizer:   what():  Unsupported non const input /compressor/MatMul/tr |
| online-soft-band-query-sfc2d.rt128k.causal16dim.6l.64b | FAIL | quantize | model.sim.onnx | circle_quantizer:   what():  Unsupported non const input /compressor/MatMul/tr |
| online-soft-band-query-sfc2d.rt128k.fp512keep475.causal16dim.6l.64b | FAIL | quantize | model.sim.correctshape.onnx | circle_quantizer:   what():  Unsupported non const input /compressor/MatMul/tr |
| online-soft-band-query-sfc2d.rt192k.causal24dim.6l.64b | FAIL | quantize | model.sim.onnx | circle_quantizer:   what():  Unsupported non const input /compressor/MatMul/tr |
| online-soft-band-query-sfc2d.rt192k.fp512keep475.causal24dim.6l.64b | FAIL | quantize | model.sim.correctshape.onnx | circle_quantizer:   what():  Unsupported non const input /compressor/MatMul/tr |
| online-soft-band-sfc2d.causal96dim.12l.musical64 | FAIL | quantize | model.sim.clipfix.onnx | circle_quantizer:   what():  Unsupported non const input /compressor/MatMul/tr |
| online-soft-band-sfc2d.mel.causal96dim.12l | FAIL | quantize | model.sim.clipfix.onnx | circle_quantizer:   what():  Unsupported non const input /compressor/MatMul/tr |
| online-soft-band-sfc2d.rt128k.fp512keep475.causal16dim.6l.64b | FAIL | quantize | model.sim.correctshape.clipfix.onnx | circle_quantizer:   what():  Unsupported non const input /compressor/MatMul/tr |
| online-soft-band-sfc2d.rt192k.fp512keep475.causal24dim.6l.64b | FAIL | quantize | model.sim.correctshape.clipfix.onnx | circle_quantizer:   what():  Unsupported non const input /compressor/MatMul/tr |
