# SFC-small standalone recipe configs

## Scope

The following NPU-oriented SFC-small recipes are now complete standalone
training configurations:

- `sfc-small-sameband-dw-bn-npu.musical64.onfly.rt192k`
- `sfc-small-pyramid-dw-bn-npu.musical64.onfly.rt192k`
- `sfc-small-macaron-conv2d-bn-npu.musical36.2l.onfly.rt192k`
- `sfc-small-macaron-conv2d-cln-npu.musical36.2l.onfly.rt192k`
- `sfc-small-macaron-conv2d-cln-lite-npu.musical36.2l.onfly.rt192k`
- `sfc-small-macaron-lrattn-bn-npu.musical36.2l.r2d64g560.onfly.rt192k`

Each `config.yaml` directly defines the trainer, on-the-fly stem datamodule,
task, loss, optimizer, scheduler, and model. The recipes contain neither
`_base_` nor `_inherit_`.

The top-level `sfc_npu_*` interpolation aliases were removed. Model constructor
values are literal and local to `task.model`, making the effective architecture
visible without following another file. Deprecated null constructor entries
were also omitted.

## Validation

Parse, resolve, and instantiate every standalone recipe:

```bash
PYTHONPATH=.:aiaccel .venv/bin/python -m pytest -q \
  tests/test_sfc_small_standalone_configs.py
```

Run the model-specific architecture, streaming, export, and budget checks:

```bash
PYTHONPATH=.:aiaccel .venv/bin/python -m pytest -q \
  tests/test_sfc_small_sameband_dw_bn_npu.py \
  tests/test_sfc_small_pyramid_dw_bn_npu.py \
  tests/test_sfc_small_macaron_conv2d_bn_npu.py \
  tests/test_sfc_small_macaron_conv2d_cln_npu.py \
  tests/test_sfc_small_macaron_conv2d_cln_lite_npu.py \
  tests/test_sfc_small_macaron_lrattn_bn_npu.py
```

Training continues to use the normal entry point:

```bash
PYTHONPATH=.:aiaccel .venv/bin/python -m aiaccel.torch.apps.train \
  recipes/dnr/models/<variant>/config.yaml
```

## Results

- Standalone recipe checks: `12 passed`
- Model-specific architecture/streaming/export checks: `36 passed`
- Real on-the-fly synthesis smoke test:
  - mixture: `(1, 1, 44100)`
  - sources: `(1, 3, 1, 44100)`
  - all samples finite

The direct `.venv/bin/pytest` launcher has a stale shebang in this checkout, so
the recorded commands use `.venv/bin/python -m pytest`.
