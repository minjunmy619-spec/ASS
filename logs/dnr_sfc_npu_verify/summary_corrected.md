# DNR SFC NPU Verification Summary (Corrected)

- Total: 22
- PASS: 0
- FAIL: 22

| Model | Status | Fail Stage | Reason | model.circle | model.opt.circle | model.q.circle |
|---|---|---|---|---|---|---|
| online-hard-band-sfc2d.causal96dim.12l.musical64 | FAIL | import | failed to legalize onnx.Conv | False | False | False |
| online-hard-band-sfc2d.mel.causal96dim.12l | FAIL | import | failed to legalize onnx.Conv | False | False | False |
| online-hierarchical-soft-band-ffi-sfc2d.speech-lowfreq-narrow.causal96dim.1-2-2l | FAIL | import | failed to legalize onnx.Conv | False | False | False |
| online-hierarchical-soft-band-parallel-ffi-sfc2d.rt128k.speech-lowfreq-narrow.causal14dim.0-1-1l.128-96-48b | FAIL | import | failed to legalize onnx.Conv | False | False | False |
| online-hierarchical-soft-band-parallel-ffi-sfc2d.rt192k.speech-lowfreq-narrow.causal20dim.0-1-1l.128-96-48b | FAIL | import | failed to legalize onnx.Conv | False | False | False |
| online-hierarchical-soft-band-sfc2d.causal96dim.1-2-2l.musical128 | FAIL | import | failed to legalize onnx.Conv | False | False | False |
| online-hierarchical-soft-band-sfc2d.mel.causal96dim.1-2-2l | FAIL | import | failed to legalize onnx.Conv | False | False | False |
| online-sfc2d.causal96dim.12l.musical64 | FAIL | import | failed to legalize onnx.Conv | False | False | False |
| online-soft-band-dilated-sfc2d.causal96dim.12l.musical64 | FAIL | import | failed to legalize onnx.Conv | False | False | False |
| online-soft-band-dilated-sfc2d.mel.causal96dim.12l | FAIL | import | failed to legalize onnx.Conv | False | False | False |
| online-soft-band-gru-sfc2d.causal96dim.12l.musical64 | FAIL | import | failed to legalize onnx.Conv | False | False | False |
| online-soft-band-gru-sfc2d.mel.causal96dim.12l | FAIL | import | failed to legalize onnx.Conv | False | False | False |
| online-soft-band-query-sfc2d.causal96dim.12l.musical64 | FAIL | import | failed to legalize onnx.Conv | False | False | False |
| online-soft-band-query-sfc2d.mel.causal96dim.12l | FAIL | import | failed to legalize onnx.Conv | False | False | False |
| online-soft-band-query-sfc2d.rt128k.causal16dim.6l.64b | FAIL | import | invalid tensor dimension size in onnx2circle | False | False | False |
| online-soft-band-query-sfc2d.rt128k.fp512keep475.causal16dim.6l.64b | FAIL | import | invalid tensor dimension size in onnx2circle | False | False | False |
| online-soft-band-query-sfc2d.rt192k.causal24dim.6l.64b | FAIL | import | failed to legalize onnx.Conv | False | False | False |
| online-soft-band-query-sfc2d.rt192k.fp512keep475.causal24dim.6l.64b | FAIL | import | failed to legalize onnx.Conv | False | False | False |
| online-soft-band-sfc2d.causal96dim.12l.musical64 | FAIL | import | failed to legalize onnx.Conv | False | False | False |
| online-soft-band-sfc2d.mel.causal96dim.12l | FAIL | import | failed to legalize onnx.Conv | False | False | False |
| online-soft-band-sfc2d.rt128k.fp512keep475.causal16dim.6l.64b | FAIL | import | invalid tensor dimension size in onnx2circle | False | False | False |
| online-soft-band-sfc2d.rt192k.fp512keep475.causal24dim.6l.64b | FAIL | import | failed to legalize onnx.Conv | False | False | False |
