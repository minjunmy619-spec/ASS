# BandSFCNetNPU Query Variants Operation

Date: 2026-06-03

## Goal

Add explicit SFC query variants to `BandSFCNetNPU`, matching the Dolphin query
variant work:

- soft-band-query: K-band latent tokens + K-band query side-path;
- cross-attention query: F<->K query attention transport.

Then review the code carefully and run an NPU compilation pass.

## Review Findings And Fixes

A focused code review was performed before NPU compilation. Key findings and
fixes:

1. `crossattn_query` is an explicit alias for the existing BandSFC quality
   cross-attention transport. This is now documented in `BandSFCNetNPU/README.md`.
2. Query preset factories originally accepted only `**kwargs`, making direct
   positional calls like `safe_soft_query(65)` fail. All query factories now
   mirror the base preset signatures.
3. The user-facing terminology requested `soft-band-query`, while the first
   implementation used `soft_query`. The core now accepts `soft_band_query` as
   an alias, and presets expose both `*_soft_band_query` and `*_soft_query`.
4. `query_type` is now threaded through:
   - `BandSFCNetNPU/presets.py`;
   - `BandSFCNetNPU/training_wrapper.py`;
   - recipe overlays.
5. `quality6m_*` query variants are retained as research probes only. At fp512
   they exceed the strict deployment budget (~9.1M params, ~218 KiB fp16 state).
6. The initial `safe_*` query models were too small (~0.45M params), so added
   `balanced_*` useful-capacity variants around 4.1M params. Capacity is placed
   in latent channels and frequency-pooled channel mixers, which increases
   modeling power without increasing temporal streaming cache too much.
7. Added config-file controls for critical parameter/latency/performance knobs:
   `n_bands`, `channels`, `num_stages`, `time_kernel`, `freq_kernel`,
   `dilation_cycle`, `transport`, `routing_normalization`, `use_attn`,
   `attn_window`, `num_heads`, `head_dim`, `pooled_mixer_hidden`,
   `pooled_mixer_hidden_schedule`, `residual_head`, and `query_type`.
8. Added tests for direct factory API, streaming parity, query-type threading,
   capacity override threading, useful-capacity range, and deployable fp512
   budget checks.

## Code Changes

- `BandSFCNetNPU/band_sfc_net_npu.py`
  - Added `transport="soft_band_query"` / `"soft_query"` path using
    `SoftBandQueryCompressor2d` and `SoftBandQueryExpander2d`.
  - Added `transport="crossattn_query"` as explicit alias for the existing
    cross-attention query encoder/decoder path.
- `BandSFCNetNPU/presets.py`
  - Added named presets:
    - `safe_soft_band_query`, `safe_crossattn_query`
    - `balanced_soft_band_query`, `balanced_crossattn_query`
    - `quality_soft_band_query`, `quality_crossattn_query`
    - `rt_plus_soft_band_query`, `rt_plus_crossattn_query`
    - `quality6m_soft_band_query`, `quality6m_crossattn_query` (research only)
  - Added `_soft_query` aliases for the soft-band-query presets.
- `BandSFCNetNPU/training_wrapper.py`
  - Added `query_type` argument.
- `BandSFCNetNPU/test_band_sfc_net_npu.py`
  - Added query variant tests.
- `BandSFCNetNPU/README.md`
  - Documented query variants, deployable recipes, budgets, and NPU pass results.

## Recipe Overlays Added

```text
recipes/dnr/models/band-sfc-net-npu.safe.soft-query.rt192k.fp512/config.yaml
recipes/dnr/models/band-sfc-net-npu.safe.crossattn-query.rt192k.fp512/config.yaml
recipes/dnr/models/band-sfc-net-npu.balanced.soft-query.rt192k.fp512/config.yaml
recipes/dnr/models/band-sfc-net-npu.balanced.crossattn-query.rt192k.fp512/config.yaml
recipes/dnr/models/band-sfc-net-npu.quality.soft-query.rt192k.fp512/config.yaml
recipes/dnr/models/band-sfc-net-npu.quality.crossattn-query.rt192k.fp512/config.yaml
recipes/dnr/models/band-sfc-net-npu.rt-plus.soft-query.distill.rt192k.fp512/config.yaml
recipes/dnr/models/band-sfc-net-npu.rt-plus.crossattn-query.distill.rt192k.fp512/config.yaml
```

## Validation Performed

Static checks:

```bash
PYTHONPATH=/home/cmj/works/ASS/aiaccel:/home/cmj/works/ASS \
PYTHONPYCACHEPREFIX=/tmp/opencode/pycache \
/home/cmj/works/ASS/.venv/bin/python -m py_compile \
  BandSFCNetNPU/band_sfc_net_npu.py \
  BandSFCNetNPU/presets.py \
  BandSFCNetNPU/training_wrapper.py \
  BandSFCNetNPU/test_band_sfc_net_npu.py
# PASS
```

```bash
/home/cmj/works/ASS/.venv/bin/python -m ruff check \
  BandSFCNetNPU/band_sfc_net_npu.py \
  BandSFCNetNPU/presets.py \
  BandSFCNetNPU/training_wrapper.py \
  BandSFCNetNPU/test_band_sfc_net_npu.py
# PASS
```

Unit/smoke tests:

```bash
PYTHONPATH=/home/cmj/works/ASS/aiaccel:/home/cmj/works/ASS \
/home/cmj/works/ASS/.venv/bin/python -m pytest BandSFCNetNPU/test_band_sfc_net_npu.py
# 13 passed
```

Recipe resolution and fp512 budget check:

```text
safe_soft_band_query       params=446,476   fp16_state=131,072 bytes  # compile-smoke only
safe_crossattn_query       params=457,099   fp16_state=131,072 bytes  # compile-smoke only
balanced_soft_band_query   params=4,066,492 fp16_state=163,840 bytes  # recommended useful-capacity
balanced_crossattn_query   params=4,083,627 fp16_state=163,840 bytes  # recommended useful-capacity
quality_soft_band_query    params=2,082,092 fp16_state=190,464 bytes
quality_crossattn_query    params=2,092,715 fp16_state=190,464 bytes
rt_plus_soft_band_query    params=2,082,290 fp16_state=190,464 bytes
rt_plus_crossattn_query    params=2,092,913 fp16_state=190,464 bytes
quality6m_soft_band_query  params=9,084,796 fp16_state=223,232 bytes  # over budget
quality6m_crossattn_query  params=9,101,931 fp16_state=223,232 bytes  # over budget
```

## NPU Compilation Pass

The safe query variants were compiled through the full verifier flow:

- ONNX export from config-only recipe;
- onnxsim;
- calibration dataset generation;
- ONE import;
- ONE optimize;
- ONE channel-wise quantize.

### Safe soft-band-query

Command:

```bash
PYTHONPATH=/home/cmj/works/ASS/aiaccel:/home/cmj/works/ASS \
/home/cmj/works/ASS/.venv/bin/python tools/online/verify_npu_variants.py \
  --mode recipe \
  --recipe-name-contains band-sfc-net-npu.safe.soft-query.rt192k.fp512 \
  --run-name band_sfc_safe_soft_query_20260603 \
  --quantize-layer-fallback \
  --force-onnxsim-large-shape-ops
```

Result:

```text
[PASS] recipe:band-sfc-net-npu.safe.soft-query.rt192k.fp512
ONNX nodes: 635
Artifacts:
logs/npu_verify_general/band_sfc_safe_soft_query_20260603/band-sfc-net-npu.safe.soft-query.rt192k.fp512/model.circle
logs/npu_verify_general/band_sfc_safe_soft_query_20260603/band-sfc-net-npu.safe.soft-query.rt192k.fp512/model.opt.circle
logs/npu_verify_general/band_sfc_safe_soft_query_20260603/band-sfc-net-npu.safe.soft-query.rt192k.fp512/model.q.circle
```

### Safe cross-attention-query

Command:

```bash
PYTHONPATH=/home/cmj/works/ASS/aiaccel:/home/cmj/works/ASS \
/home/cmj/works/ASS/.venv/bin/python tools/online/verify_npu_variants.py \
  --mode recipe \
  --recipe-name-contains band-sfc-net-npu.safe.crossattn-query.rt192k.fp512 \
  --run-name band_sfc_safe_crossattn_query_20260603 \
  --quantize-layer-fallback \
  --force-onnxsim-large-shape-ops
```

Result:

```text
[PASS] recipe:band-sfc-net-npu.safe.crossattn-query.rt192k.fp512
ONNX nodes: 721
Artifacts:
logs/npu_verify_general/band_sfc_safe_crossattn_query_20260603/band-sfc-net-npu.safe.crossattn-query.rt192k.fp512/model.circle
logs/npu_verify_general/band_sfc_safe_crossattn_query_20260603/band-sfc-net-npu.safe.crossattn-query.rt192k.fp512/model.opt.circle
logs/npu_verify_general/band_sfc_safe_crossattn_query_20260603/band-sfc-net-npu.safe.crossattn-query.rt192k.fp512/model.q.circle
```

### Balanced soft-band-query

Command:

```bash
PYTHONPATH=/home/cmj/works/ASS/aiaccel:/home/cmj/works/ASS \
/home/cmj/works/ASS/.venv/bin/python tools/online/verify_npu_variants.py \
  --mode recipe \
  --recipe-name-contains band-sfc-net-npu.balanced.soft-query.rt192k.fp512 \
  --run-name band_sfc_balanced_soft_query_20260603 \
  --quantize-layer-fallback \
  --force-onnxsim-large-shape-ops
```

Result:

```text
[PASS] recipe:band-sfc-net-npu.balanced.soft-query.rt192k.fp512
ONNX nodes: 635
Params/state: 4,066,492 params, 160 KiB fp16 state
Artifacts:
logs/npu_verify_general/band_sfc_balanced_soft_query_20260603/band-sfc-net-npu.balanced.soft-query.rt192k.fp512/model.circle
logs/npu_verify_general/band_sfc_balanced_soft_query_20260603/band-sfc-net-npu.balanced.soft-query.rt192k.fp512/model.opt.circle
logs/npu_verify_general/band_sfc_balanced_soft_query_20260603/band-sfc-net-npu.balanced.soft-query.rt192k.fp512/model.q.circle
```

### Balanced cross-attention-query

Command:

```bash
PYTHONPATH=/home/cmj/works/ASS/aiaccel:/home/cmj/works/ASS \
/home/cmj/works/ASS/.venv/bin/python tools/online/verify_npu_variants.py \
  --mode recipe \
  --recipe-name-contains band-sfc-net-npu.balanced.crossattn-query.rt192k.fp512 \
  --run-name band_sfc_balanced_crossattn_query_20260603 \
  --quantize-layer-fallback \
  --force-onnxsim-large-shape-ops
```

Result:

```text
[PASS] recipe:band-sfc-net-npu.balanced.crossattn-query.rt192k.fp512
ONNX nodes: 721
Params/state: 4,083,627 params, 160 KiB fp16 state
Artifacts:
logs/npu_verify_general/band_sfc_balanced_crossattn_query_20260603/band-sfc-net-npu.balanced.crossattn-query.rt192k.fp512/model.circle
logs/npu_verify_general/band_sfc_balanced_crossattn_query_20260603/band-sfc-net-npu.balanced.crossattn-query.rt192k.fp512/model.opt.circle
logs/npu_verify_general/band_sfc_balanced_crossattn_query_20260603/band-sfc-net-npu.balanced.crossattn-query.rt192k.fp512/model.q.circle
```

### Quality soft-band-query

Command:

```bash
PYTHONPATH=/home/cmj/works/ASS/aiaccel:/home/cmj/works/ASS \
/home/cmj/works/ASS/.venv/bin/python tools/online/verify_npu_variants.py \
  --mode recipe \
  --recipe-name-contains band-sfc-net-npu.quality.soft-query.rt192k.fp512 \
  --run-name band_sfc_quality_soft_query_20260603 \
  --quantize-layer-fallback \
  --force-onnxsim-large-shape-ops
```

Result:

```text
[PASS] recipe:band-sfc-net-npu.quality.soft-query.rt192k.fp512
ONNX nodes: 1,424
Artifacts:
logs/npu_verify_general/band_sfc_quality_soft_query_20260603/band-sfc-net-npu.quality.soft-query.rt192k.fp512/model.circle
logs/npu_verify_general/band_sfc_quality_soft_query_20260603/band-sfc-net-npu.quality.soft-query.rt192k.fp512/model.opt.circle
logs/npu_verify_general/band_sfc_quality_soft_query_20260603/band-sfc-net-npu.quality.soft-query.rt192k.fp512/model.q.circle
```

Note: the ONNX audit for this full-forward `T=1` export still reports
`And/Less/GreaterOrEqual` from the bounded attention path, but ONE import,
optimization, and quantization all completed. Treat this as a compile-positive
quality ablation, not yet the cleanest strict-edge graph.

## Follow-up Recommendation

Use the two balanced query variants as the main useful-capacity NPU training
baselines. The safe query variants remain compile-smoke baselines only. The
quality soft-query graph has a positive ONE compilation result, but it is larger
and has comparison ops in the ONNX audit; prefer it only if it gives clear
quality gains over `balanced_*`.
