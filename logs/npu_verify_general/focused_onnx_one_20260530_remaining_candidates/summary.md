# NPU Variant Verification Summary

- Total: 4
- PASS: 3
- FAIL: 1

| Kind | Variant | Status | Fail Stage |
|---|---|---|---|
| recipe | dolphin-sfc-npu.large-6m.fp512keep475 | PASS | - |
| recipe | dolphin-sfc-npu.slim-6m.distill.rt192k.fp512keep475 | PASS | - |
| recipe | edge-fusion-sfc-distilled.rt192k | FAIL | export |
| recipe | online-hierarchical-soft-band-parallel-ffi-sfc2d.rt192k.speech-lowfreq-narrow.causal20dim.0-1-1l.128-96-48b | PASS | - |
