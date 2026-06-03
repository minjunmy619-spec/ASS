# DolphinSFCNPU Query Variants Operation

Date: 2026-06-03

## Goal

Add Dolphin SFC variants that test whether an explicit SFC-style query side-path
improves the current DolphinSFCNPU slim deployment candidate without breaking
online/causal NPU constraints.

## Added Variants

### Soft-band-query Dolphin

Preset suffix: `_soft_query`

Example preset:

```text
slim_6m_soft_query
```

Architecture delta:

```text
in_proj
  -> StatelessSoftBandQueryCompressor2d
       -> latent K-band tokens
       -> query K-band tokens
  -> existing Dolphin multi-scale encoder/decoder on latent tokens
  -> DolphinSoftBandQueryDecoder2d(latent, query_tokens)
  -> source gain mask head
```

This is the safer query ablation because the side-path remains on the compressed
K-band axis. It adds no temporal cache and keeps the same packed streaming state
layout as the original Dolphin slim variants.

### Cross-attention-query Dolphin

Preset suffix: `_crossattn_query`

Example preset:

```text
slim_6m_crossattn_query
```

Architecture delta:

```text
in_proj
  -> StatelessCrossAttentionQueryCompressor2d
       -> F-to-K cross-attention latent tokens
       -> full-resolution side embedding
  -> existing Dolphin multi-scale encoder/decoder on latent tokens
  -> DolphinCrossAttentionQueryDecoder2d(latent, side_embedding)
       -> F queries attend to K latent tokens
  -> source gain mask head
```

This is closer to query/cross-attention SFC but introduces more `MatMul`,
`Softmax`, reshape, and transpose traffic. It should be treated as a quality
ablation first, then verified through ONNX/ONE.

## Files Changed

- `DolphinSFCNPU/dolphin_sfc.py`
  - Added stateless soft-band-query compressor/decoder.
  - Added stateless cross-attention-query compressor/decoder.
  - Added `query_variant` and `query_type` options to `DolphinSFCNPUSeparator`.
  - Added `compressor_freq_kernel` and `ffn_expansion` model knobs.
  - Added `build_dolphin_sfc_npu_from_config(...)` for recipe-side overrides.
  - Added named preset suffixes `_soft_query`, `_soft_band_query`, and `_crossattn_query`.
- `DolphinSFCNPU/training_wrapper.py`
  - Plumbed `query_variant`, `query_type`, `n_bands`, `d_model`, `num_scales`,
    `widths`, `blocks_per_scale`, `time_kernels`, `freq_kernels`,
    `compressor_freq_kernel`, and `ffn_expansion` into recipe construction.
- `DolphinSFCNPU/test_dolphin_sfc_npu.py`
  - Added streaming-equivalence coverage for edge query variants.
- `DolphinSFCNPU/README.md`
  - Documented the new variant families and recipes.
- Added DnR distillation overlays:
  - `recipes/dnr/models/dolphin-sfc-npu.slim-6m.soft-query.distill-mixsoftmax.rt192k.fp512keep475/config.yaml`
  - `recipes/dnr/models/dolphin-sfc-npu.slim-6m.crossattn-query.distill-mixsoftmax.rt192k.fp512keep475/config.yaml`

## Example Training Commands

```bash
PYTHONPATH=$PWD/aiaccel:$PWD ./.venv/bin/python -m aiaccel.torch.apps.train \
  recipes/dnr/models/dolphin-sfc-npu.slim-6m.soft-query.distill-mixsoftmax.rt192k.fp512keep475/config.yaml \
  teacher_checkpoint_path=/path/to/sfc_locoformer_teacher.ckpt
```

```bash
PYTHONPATH=$PWD/aiaccel:$PWD ./.venv/bin/python -m aiaccel.torch.apps.train \
  recipes/dnr/models/dolphin-sfc-npu.slim-6m.crossattn-query.distill-mixsoftmax.rt192k.fp512keep475/config.yaml \
  teacher_checkpoint_path=/path/to/sfc_locoformer_teacher.ckpt
```

## Validation Commands

Focused tests:

```bash
PYTHONPATH=$PWD/aiaccel:$PWD ./.venv/bin/python -m pytest \
  DolphinSFCNPU/test_dolphin_sfc_npu.py::test_query_variant_presets_forward_stream_match \
  DolphinSFCNPU/test_dolphin_sfc_npu.py::test_query_variant_builder_flag_matches_named_preset
```

Full DolphinSFCNPU tests:

```bash
PYTHONPATH=$PWD/aiaccel:$PWD ./.venv/bin/python -m pytest DolphinSFCNPU/test_dolphin_sfc_npu.py
```

Validation performed in this workspace:

```text
PYTHONPATH=/home/cmj/works/ASS/aiaccel:/home/cmj/works/ASS \
PYTHONPYCACHEPREFIX=/tmp/opencode/pycache \
/home/cmj/works/ASS/.venv/bin/python -m py_compile \
  DolphinSFCNPU/dolphin_sfc.py \
  DolphinSFCNPU/training_wrapper.py \
  DolphinSFCNPU/test_dolphin_sfc_npu.py
# PASS
```

```text
PYTHONPATH=/home/cmj/works/ASS/aiaccel:/home/cmj/works/ASS \
/home/cmj/works/ASS/.venv/bin/python -m pytest \
  DolphinSFCNPU/test_dolphin_sfc_npu.py::test_query_variant_presets_forward_stream_match \
  DolphinSFCNPU/test_dolphin_sfc_npu.py::test_query_variant_builder_flag_matches_named_preset
# 2 passed
```

```text
PYTHONPATH=/home/cmj/works/ASS/aiaccel:/home/cmj/works/ASS \
/home/cmj/works/ASS/.venv/bin/python -m pytest DolphinSFCNPU/test_dolphin_sfc_npu.py
# 15 passed
```

```text
/home/cmj/works/ASS/.venv/bin/python -m ruff check \
  DolphinSFCNPU/dolphin_sfc.py \
  DolphinSFCNPU/training_wrapper.py \
  DolphinSFCNPU/test_dolphin_sfc_npu.py \
  DolphinSFCNPU/__init__.py
# PASS
```

Recipe overlay resolution was checked for both new configs.  Both resolve to
`mask_activation: softmax` inherited from the mixsoftmax repair recipe and the
expected query settings:

```text
slim_6m_soft_query      query_variant=soft_band_query  query_type=adaptive
slim_6m_crossattn_query query_variant=crossattn_query   query_type=adaptive
```

ONNX smoke export was checked for `edge_small_soft_query` and
`edge_small_crossattn_query` through `DolphinSFCNPUStreamingExportWrapper`:

```text
inputs=2 outputs=2 forbidden=[] has_softmax=True has_matmul=True
```

Representative `F=475` capacity/state check:

```text
slim_6m                    params=5,170,488 fp16_state=165,888
slim_6m_soft_query         params=5,253,049 fp16_state=165,888
slim_6m_soft_band_query    params=5,253,049 fp16_state=165,888
slim_6m_crossattn_query    params=5,312,904 fp16_state=165,888
```

The query recipes now expose the capacity knobs directly:

```text
dolphin_n_bands
dolphin_d_model
dolphin_num_scales
dolphin_widths
dolphin_blocks_per_scale
dolphin_time_kernels
dolphin_freq_kernels
dolphin_compressor_freq_kernel
dolphin_ffn_expansion
```

NPU verifier results after adding config controls:

```text
[PASS] dolphin-sfc-npu.slim-6m.soft-query.distill-mixsoftmax.rt192k.fp512keep475
ONNX nodes: 782
Artifacts:
logs/npu_verify_general/dolphin_sfc_slim6m_soft_query_controls_20260603/dolphin-sfc-npu.slim-6m.soft-query.distill-mixsoftmax.rt192k.fp512keep475/model.circle
logs/npu_verify_general/dolphin_sfc_slim6m_soft_query_controls_20260603/dolphin-sfc-npu.slim-6m.soft-query.distill-mixsoftmax.rt192k.fp512keep475/model.opt.circle
logs/npu_verify_general/dolphin_sfc_slim6m_soft_query_controls_20260603/dolphin-sfc-npu.slim-6m.soft-query.distill-mixsoftmax.rt192k.fp512keep475/model.q.circle
```

```text
[PASS] dolphin-sfc-npu.slim-6m.crossattn-query.distill-mixsoftmax.rt192k.fp512keep475
ONNX nodes: 761
Artifacts:
logs/npu_verify_general/dolphin_sfc_slim6m_crossattn_query_controls_20260603/dolphin-sfc-npu.slim-6m.crossattn-query.distill-mixsoftmax.rt192k.fp512keep475/model.circle
logs/npu_verify_general/dolphin_sfc_slim6m_crossattn_query_controls_20260603/dolphin-sfc-npu.slim-6m.crossattn-query.distill-mixsoftmax.rt192k.fp512keep475/model.opt.circle
logs/npu_verify_general/dolphin_sfc_slim6m_crossattn_query_controls_20260603/dolphin-sfc-npu.slim-6m.crossattn-query.distill-mixsoftmax.rt192k.fp512keep475/model.q.circle
```

Recommended NPU verifier follow-up for future capacity edits:

```bash
PYTHONPATH=$PWD/aiaccel:$PWD ./.venv/bin/python tools/online/verify_npu_variants.py \
  --mode recipe \
  --recipe-name-contains dolphin-sfc-npu.slim-6m.soft-query \
  --run-name dolphin_soft_query_20260603 \
  --quantize-layer-fallback \
  --force-onnxsim-large-shape-ops
```

```bash
PYTHONPATH=$PWD/aiaccel:$PWD ./.venv/bin/python tools/online/verify_npu_variants.py \
  --mode recipe \
  --recipe-name-contains dolphin-sfc-npu.slim-6m.crossattn-query \
  --run-name dolphin_crossattn_query_20260603 \
  --quantize-layer-fallback \
  --force-onnxsim-large-shape-ops
```
