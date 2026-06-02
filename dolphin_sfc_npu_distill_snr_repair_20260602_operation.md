# DolphinSFCNPU Distillation SNR Repair Operation

Date: 2026-06-02

## Trigger

The recipe below was reported to plateau around validation SNR `6.9`, while the
target expectation was above `8.0`.

```text
recipes/dnr/models/dolphin-sfc-npu.slim-6m.distill.rt192k.fp512keep475/config.yaml
```

## Findings

- The resolved config keeps the intended `css_validation: true` and EMA fields,
  so the reported validation SNR is from the chunked/online student path.
- No training checkpoint, TensorBoard event file, or `merged_config.yaml` for
  the reported run was present under the recipe directory in this workspace.
- The distillation task previously logged only student validation SNR.  It did
  not log the teacher checkpoint SNR, so a weak or STFT-mismatched teacher could
  not be separated from a weak student.
- `BandSFCNetNPU` initializes its mask head to a conservative mixture split.
  `DolphinSFCNPU` previously left the output head random; with the independent
  sigmoid gain head, an untrained 3-stem model emitted roughly `1.3x` to `1.4x`
  mixture energy summed across stems.
- Dolphin's independent sigmoid gain masks make mixture consistency a loss
  penalty only.  A source-axis softmax gain head can make reduced-frequency
  estimates sum to the projected mixture by construction while keeping tensors
  4D and using NPU-supported `Softmax`.

## Changes

- Added deterministic DolphinSFCNPU output-head initialization:
  - zero output projection weights;
  - sigmoid mode bias initialized to `logit(1 / n_src)`;
  - softmax mode bias initialized to zero, giving equal source gains.
- Added `mask_activation: sigmoid|softmax` to `DolphinSFCNPUSeparator`,
  `build_dolphin_sfc_npu_preset`, and
  `DolphinSFCNPU.training_wrapper.build_dolphin_sfc_npu_system`.
- Added validation diagnostics in
  `spectral_feature_compression.core.tasks.distillation_task.TeacherStudentDistillationTask`:
  - `validation/teacher_snr`;
  - `validation/student_teacher_snr`.
- Added a separate repair recipe:

```text
recipes/dnr/models/dolphin-sfc-npu.slim-6m.distill-mixsoftmax.rt192k.fp512keep475/config.yaml
```

- Fixed `tools/online/export_onnx_online_model.py` metadata resolution for
  recipe overlays whose model fields mirror top-level values, for example
  `freq_preprocess_enabled: ${freq_preprocess_enabled}`.
- Added `mask_postprocess` metadata to the deploy manifest.  This is required
  for `--disable-masking` exports because a softmax-trained model emits raw
  logits and the external postprocess must apply `softmax`, not the old sigmoid
  gain rule.

## Suggested Next Training Command

Use the same teacher checkpoint as the previous run, but first verify
`validation/teacher_snr` is above the student target.

```bash
PYTHONPATH=$PWD/aiaccel:$PWD ./.venv/bin/python -m aiaccel.torch.apps.train \
  recipes/dnr/models/dolphin-sfc-npu.slim-6m.distill-mixsoftmax.rt192k.fp512keep475/config.yaml \
  teacher_checkpoint_path=/path/to/sfc_locoformer_teacher.ckpt
```

If the teacher checkpoint was trained with a different STFT contract, add the
matching overrides:

```bash
teacher_n_fft=2048 teacher_hop_length=512
```

## Validation Commands

```bash
PYTHONPATH=$PWD/aiaccel:$PWD ./.venv/bin/python -m pytest \
  DolphinSFCNPU/test_dolphin_sfc_npu.py
```

```bash
PYTHONPATH=$PWD/aiaccel:$PWD ./.venv/bin/python - <<'PY'
from pathlib import Path
from omegaconf import OmegaConf as oc
from aiaccel.config import load_config, resolve_inherit

cfg_path = "recipes/dnr/models/dolphin-sfc-npu.slim-6m.distill-mixsoftmax.rt192k.fp512keep475/config.yaml"
config = load_config(cfg_path, {
    "config_path": cfg_path,
    "working_directory": str(Path(cfg_path).parent.resolve()),
    "base_config_path": str(Path("aiaccel/aiaccel/torch/apps/config").resolve()),
})
print(oc.to_yaml(resolve_inherit(config), resolve=False))
PY
```

Focused NPU verification:

```bash
PYTHONPATH=$PWD/aiaccel:$PWD ./.venv/bin/python tools/online/verify_npu_variants.py \
  --mode recipe \
  --recipe-name-contains distill-mixsoftmax \
  --run-name dolphin_mixsoftmax_force_onnxsim_20260602 \
  --quantize-layer-fallback \
  --force-onnxsim-large-shape-ops
```

Result:

```text
[PASS] recipe:dolphin-sfc-npu.slim-6m.distill-mixsoftmax.rt192k.fp512keep475
```

Artifacts:

```text
logs/npu_verify_general/dolphin_mixsoftmax_force_onnxsim_20260602/dolphin-sfc-npu.slim-6m.distill-mixsoftmax.rt192k.fp512keep475/model.circle
logs/npu_verify_general/dolphin_mixsoftmax_force_onnxsim_20260602/dolphin-sfc-npu.slim-6m.distill-mixsoftmax.rt192k.fp512keep475/model.opt.circle
logs/npu_verify_general/dolphin_mixsoftmax_force_onnxsim_20260602/dolphin-sfc-npu.slim-6m.distill-mixsoftmax.rt192k.fp512keep475/model.q.circle
```

Note: without `--force-onnxsim-large-shape-ops`, ONE import failed with
`invalid tensor dimension size`; forced simplification removed
`ConstantOfShape` and related shape ops before import.

The refreshed deploy manifest includes:

```json
{
  "trained_masking_enabled": true,
  "inside_graph": false,
  "activation": "softmax",
  "external_postprocess_required": true
}
```
