# Dolphin Source-Aware NPU Variant - 2026-06-28

## Goal

Implement a quality-recovery DolphinSFCNPU student after the default
`slim_6m` recipe showed insufficient separation quality.

The new variant keeps the Dolphin deployable strengths:

- compressed shared SFC/Dolphin trunk;
- one streaming cache per trunk block;
- 4D tensors and NPU-friendly Conv/MatMul/Softmax-style operators;
- state below the 192 KiB fp16 target.

It adds the missing quality-oriented structure:

- per-source compressed-token refinement for explicit speech/music heads;
- optional complex-mask output for the explicit sources;
- residual SFX reconstruction by default;
- optional gated residual SFX preset for leakage-suppression ablation;
- true mask/logit aux output for distillation.

## Implemented Files

- `DolphinSFCNPU/dolphin_sfc.py`
  - Added `DolphinSourceTokenRefinementBlock2d`.
  - Added `DolphinSourceTokenRefiner2d`.
  - Added `SourceAwareDolphinSFCNPUSeparator`.
  - Added presets:
    - `source_aware_6m`
    - `source_aware_6m_gated_sfx`
    - `source_aware_6m_crossattn`
- `DolphinSFCNPU/training_wrapper.py`
  - Threads source-aware builder options.
  - Supports `return_aux=True` for Dolphin wrappers.
- `DolphinSFCNPU/__init__.py`
  - Exports `SourceAwareDolphinSFCNPUSeparator`.
- `DolphinSFCNPU/test_dolphin_sfc_npu.py`
  - Added source-aware forward, streaming, aux, and budget tests.
- `recipes/dnr/models/dolphin-sfc-npu.sourceaware-6m.complex-residual.distill.rt192k.fp512keep475/config.yaml`
  - Training recipe for the first source-aware Dolphin run.

## Architecture

The default recipe uses:

```text
preset: source_aware_6m
n_bands: 56
query_variant: soft_band_query
source_head_type: complex_residual
sfx_residual_mode: residual
source_refine_layers: 2
source_refine_freq_kernel: 5
source_refine_expansion: 2
```

Signal path:

```text
packed complex STFT
  -> 1x1 input projection
  -> stateless SFC compressor F=512 -> K=56
  -> shared 3-scale Dolphin encoder/decoder trunk
  -> per-source compressed-token refiners for speech/music
  -> per-source complex mask heads
  -> explicit speech/music estimates
  -> SFX = mixture - speech - music
```

The default residual SFX mode preserves exact mixture consistency.  The
`source_aware_6m_gated_sfx` preset adds a learned gate on the residual SFX
bucket for leakage suppression experiments, but that no longer guarantees
exact mixture consistency.

## Measured Profile

For:

```text
recipes/dnr/models/dolphin-sfc-npu.sourceaware-6m.complex-residual.distill.rt192k.fp512keep475/config.yaml
```

Live config-only measurement:

```text
core: SourceAwareDolphinSFCNPUSeparator
params: 5,689,922
fp16 state: 189.0 KiB
n_freq: 512
n_bands: 56
query_variant: soft_band_query
head: complex_residual
sfx: residual
```

ONNX export/audit:

```text
raw ONNX nodes: 953
state tensors: 8
initializer size: 21.91 MiB
disallowed ops: none
op types: Add, Concat, Constant, Conv, ConvTranspose, Div, Gather, Identity,
          MatMul, Mul, ReduceMean, ReduceSum, Reshape, Shape, Sigmoid, Slice,
          Softmax, Sqrt, Sub, Tanh, Transpose
```

This is larger than default Dolphin's verified simplified graph, but it adds
the source-aware/refinement capacity that default `slim_6m` lacked.

## Distillation Notes

The new source-aware core returns:

```text
aux["mask"]
aux["mask_domain"] = "packed_complex_mask"
aux["mask_logits"]
aux["mask_logits_domain"] = "source_aware_dolphin_complex_mask_logits"
aux["mask_logits_transform"] = "sigmoid_tanh_complex_mask"
aux["explicit_source_count"] = 2
aux["residual_source_index"] = 2
```

The recipe sets:

```yaml
task:
  request_model_aux: true
  require_model_aux: true
  mask_aux_alignment: shared_prefix
```

This makes the speech/music explicit masks/logits available for teacher
distillation instead of falling back only to waveform-derived pseudo-masks.

## Validation Commands

```bash
.venv/bin/python -m pytest DolphinSFCNPU/test_dolphin_sfc_npu.py -q
.venv/bin/python -m pytest DolphinSFCNPU/test_dolphin_sfc_npu.py -q -k 'source_aware'
.venv/bin/python -m ruff check DolphinSFCNPU/dolphin_sfc.py DolphinSFCNPU/training_wrapper.py DolphinSFCNPU/test_dolphin_sfc_npu.py
PYTHONPYCACHEPREFIX=/tmp/ass_pycache .venv/bin/python -m py_compile DolphinSFCNPU/dolphin_sfc.py DolphinSFCNPU/training_wrapper.py

.venv/bin/python tools/online/export_onnx_online_model.py \
  recipes/dnr/models/dolphin-sfc-npu.sourceaware-6m.complex-residual.distill.rt192k.fp512keep475/config.yaml \
  --out /tmp/dolphin_sourceaware_6m.onnx \
  --n-chan 1 \
  --frames 1 \
  --freqs 512 \
  --streaming \
  --state-meta-out /tmp/dolphin_sourceaware_6m_state.json \
  --deploy-manifest-out /tmp/dolphin_sourceaware_6m_manifest.json \
  --op-preset edge_npu_recommended

.venv/bin/python - <<'PY'
import onnx
m = onnx.load('/tmp/dolphin_sourceaware_6m.onnx')
onnx.checker.check_model(m)
print('onnx_check_ok', len(m.graph.node))
PY
```

Observed validation:

```text
DolphinSFCNPU/test_dolphin_sfc_npu.py: 19 passed
source-aware test subset: 4 passed
ruff: all checks passed
py_compile: passed with PYTHONPYCACHEPREFIX=/tmp/ass_pycache
ONNX checker: passed, 953 nodes
```

