# BandSFCNetNPU Causal CNB Variant Operation

Date: 2026-06-03

## Goal

Implement the `Proposal B — Causal BandSFC-CNB` sketch from
`auido_separtion_model_more.md` as an opt-in BandSFCNetNPU variant without
changing existing `safe`, `balanced`, `quality`, or `rt_plus` behavior.

## Implemented classes

Added explicit Proposal-B-style blocks in `BandSFCNetNPU/band_sfc_net_npu.py`:

- `CrossBandMixer`
  - grouped latent-band frequency mixer
- `CausalFSMNBandMixer`
  - FSMN-style causal narrow-band memory mixer
  - stores one shared max-context cache for all dilation branches to reduce
    persistent streaming state
- `CompressedSelfAttentionFusion`
  - stateless compressed-band self-attention applied independently per frame
- `CausalCNBBlock`
  - exact named block flow:
    `CrossBandMixer -> CausalFSMNBandMixer -> CompressedSelfAttentionFusion`

## NPU constraint adjustment

The proposal document sketch uses:

```python
CausalFSMNBandMixer(d_model, kernel_t=5, dilation_schedule=(1, 2, 4))
```

The current repo validator rejects the last branch because:

```text
(kernel_size - 1) * dilation = (5 - 1) * 4 = 16 >= 14
```

Therefore the deployable presets use the nearest safe schedule:

```yaml
band_sfc_cnb_kernel: 5
band_sfc_cnb_dilation_schedule: [1, 2, 3]
```

This preserves the CNB block structure while respecting the current NPU
kernel-span rule.

## New presets

Added preset configs in `BandSFCNetNPU/presets.py`:

- `causal_cnb_soft_query`
- `causal_cnb_soft_band_query`
- `causal_cnb_crossattn_query`
- `causal_cnb_balanced_soft_query`
- `causal_cnb_balanced_soft_band_query`
- `causal_cnb_balanced_crossattn_query`

Current compile-smoke shape:

```text
n_bands=48
channels=24
num_stages=5
time_kernel=1          # SFC transport routing temporal kernel
freq_kernel=3
stage_type=causal_cnb
cnb_kernel=5
cnb_dilation_schedule=(1, 2, 3)
```

The lower channel/band count is intentional for the first strict-state smoke
target because each CNB stage carries a 12-frame causal FSMN cache.

Useful-capacity balanced shape:

```text
n_bands=48
channels=32
num_stages=5
time_kernel=1
freq_kernel=3
stage_type=causal_cnb
cnb_kernel=5
cnb_dilation_schedule=(1, 2, 3)
pooled_mixer_hidden=8192 per CNB stage
```

The balanced shape uses state-free pooled channel mixers after each CNB block to
reach the useful 2-7M parameter range without increasing persistent streaming
cache.

## New recipes

Added DnR recipe overlays:

- `recipes/dnr/models/band-sfc-net-npu.causal-cnb.soft-query.rt192k.fp512/config.yaml`
- `recipes/dnr/models/band-sfc-net-npu.causal-cnb.crossattn-query.rt192k.fp512/config.yaml`
- `recipes/dnr/models/band-sfc-net-npu.causal-cnb.balanced.soft-query.rt192k.fp512/config.yaml`
- `recipes/dnr/models/band-sfc-net-npu.causal-cnb.balanced.crossattn-query.rt192k.fp512/config.yaml`

The balanced soft-query recipe is the recommended CNB training target.  The
cross-attention-query recipe is an expressiveness/quality ablation.

## Code-review fixes

A later review fixed these concrete issues:

- Added positive-value validation for `CrossBandMixer.freq_kernel`.
- Added positive-value validation for `CausalFSMNBandMixer.kernel_t`.
- Fixed zero-context FSMN streaming state handling so a hypothetical
  `kernel_t=1` configuration returns an empty cache instead of retaining the
  current frame.
- Fixed `CausalCNBBlock.forward(..., state=...)` so it no longer silently
  returns a stale state.  Stateful single-frame calls now dispatch to
  `forward_stream`.

## Measured preset size/state

Current strict fp512 CNB smoke targets:

| Preset | Params | fp16 state | Context |
|---|---:|---:|---:|
| `causal_cnb_soft_band_query` | 37,260 | 138,240 B | 60 frames |
| `causal_cnb_crossattn_query` | 43,307 | 138,240 B | 60 frames |
| `causal_cnb_balanced_soft_band_query` | 4,070,812 | 184,320 B | 60 frames |
| `causal_cnb_balanced_crossattn_query` | 4,081,963 | 184,320 B | 60 frames |

The first two are exact-structure / compile-smoke CNB variants.  The balanced
variants are the useful-capacity training targets.  The 60-frame context comes
from five CNB stages, each with a 12-frame FSMN cache.

## ONNX audit

Legacy ONNX export initially emits `ConstantOfShape` from padding/shape
scaffolding, but `onnxsim` removes it.  Simplified graphs contain no `Expand`,
`ConstantOfShape`, `Tile`, `ScatterND`, or `Unflatten`.

Simplified op summaries:

```text
causal_cnb_soft_band_query:
  nodes=378
  Add=52, Concat=2, Conv=61, Div=20, MatMul=13, Mul=70,
  ReduceMean=19, ReduceSum=1, Reshape=28, Sigmoid=15, Slice=33,
  Softmax=6, Sqrt=19, Sub=3, Transpose=36

causal_cnb_crossattn_query:
  nodes=416
  Add=54, Concat=1, Conv=69, Div=21, MatMul=15, Mul=79,
  ReduceMean=21, Reshape=34, Sigmoid=15, Slice=39, Softmax=7,
  Sqrt=21, Sub=3, Transpose=37

causal_cnb_balanced_soft_band_query:
  nodes=448
  Add=62, Concat=2, Conv=71, Div=25, MatMul=13, Mul=85,
  ReduceMean=29, ReduceSum=1, Reshape=28, Sigmoid=20, Slice=43,
  Softmax=6, Sqrt=24, Sub=3, Transpose=36

causal_cnb_balanced_crossattn_query:
  nodes=486
  Add=64, Concat=1, Conv=79, Div=26, MatMul=15, Mul=94,
  ReduceMean=31, Reshape=34, Sigmoid=20, Slice=49, Softmax=7,
  Sqrt=26, Sub=3, Transpose=37
```

## Validation commands

Recommended local smoke command:

```bash
cd /home/cmj/works/ASS
PYTHONPATH=. .venv/bin/python -m BandSFCNetNPU.test_band_sfc_net_npu
```

Recommended size/state inspection:

```bash
cd /home/cmj/works/ASS
PYTHONPATH=. .venv/bin/python - <<'PY'
import torch
from BandSFCNetNPU.presets import build_band_sfc_net_npu_preset
for preset in ("causal_cnb_soft_band_query", "causal_cnb_crossattn_query"):
    model = build_band_sfc_net_npu_preset(preset, n_freq=512, n_src=3, n_chan=1).eval()
    params = sum(p.numel() for p in model.parameters())
    state = model.state_size_bytes(dtype=torch.float16)
    print(preset, params, state)
PY
```
