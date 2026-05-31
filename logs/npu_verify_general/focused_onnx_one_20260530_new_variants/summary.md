# NPU Variant Verification Summary

- Total: 5
- PASS: 2
- FAIL: 3

| Kind | Variant | Status | Fail Stage |
|---|---|---|---|
| recipe | sparse-unet-mel-sfc.rt192k.fp512keep475 | FAIL | export |
| recipe | sfc-sepreformer-multistem.rt192k.fp512keep475 | FAIL | export |
| recipe | sfc-residual-refinement.rt192k.fp512keep475 | PASS | - |
| recipe | bandmap-ablation.fixed80.rt192k.fp512keep475 | PASS | - |
| recipe | bandmap-ablation.mel-overlap80.rt192k.fp512keep475 | FAIL | export |
