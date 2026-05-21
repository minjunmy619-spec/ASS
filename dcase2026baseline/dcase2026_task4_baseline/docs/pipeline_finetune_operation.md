# Pipeline Finetune Operation

## Goal

Add one general online fine-tuning wrapper for the deployment data flow:

1. `USS(mixture) -> estimated source slots`
2. `SC(estimated source slots) -> source labels / activity`
3. `TSE(mixture, USS slots, SC labels) -> refined source slots`
4. Optional final `SC(TSE refined slots) -> updated classifier loss`

The wrapper is opt-in and does not replace the existing SC-primary USS wrapper
or the existing online-teacher TSE wrapper.

## Added Wrapper

`src/training/lightningmodule/uss_sc_tse_pipeline_finetune.py`

Main class: `USSScTSEPipelineFinetuneLightning`

Modes:

- `target_stage: tse`
  - Freezes USS and SC.
  - Trains TSE from live USS estimates and SC predictions.
  - Keeps the behavior of `OnlineTeacherTSELightning`, but under the general
    pipeline wrapper.

- `target_stage: sc_after_tse`
  - Freezes USS and TSE.
  - Uses SC once as the teacher that supplies TSE query labels.
  - Runs frozen TSE.
  - Trains SC on the final TSE output waveforms with oracle-aligned class
    targets and configurable active/silence sample weights.

## Added Configs

- `config/separation/modified_deft_pipeline_finetune_tse_uss_sc.yaml`
- `config/separation/modified_deft_pipeline_finetune_sc_after_tse.yaml`

Both configs use `from_training_config` references for pretrained component
definitions:

- USS model from `modified_deft_sc_primary_frozenuss_pretrainedsed_fusion.yaml`
- SC model/loss/optimizer from `../label/m2d_sc_stage3_estimated_pretrainedsed_fusion.yaml`
- TSE model/loss/optimizer from `modified_deft_tse_lite_6s_temporal.yaml`

This keeps the architecture and loss contracts tied to the normal pretraining
configs instead of duplicating them by hand.

## Validation

Focused regression command:

```bash
PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=/tmp/numba_cache \
  .venv/bin/python -m pytest \
  tests/test_task4_2026_losses.py \
  tests/test_uss_sc_joint_model_parallel.py
```

Result:

```text
45 passed, 3 warnings
```

The remaining warning is the existing pytest cache permission warning under
`/home/cmj/works/ASS/.pytest_cache`, plus upstream deprecation warnings.
