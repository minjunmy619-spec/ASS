# Adaptive Mel Locoformer-Lite IO-Balanced Variant Operation

Date: 2026-06-03

## Goal

Create a new variant of
`recipes/dnr/models/adaptive-mel-locoformer-lite-sfc.rt192k.fp512keep475/config.yaml`
with a more reasonable parameter distribution.  The original recipe placed
almost all trainable parameters in separator-side frequency-pooled capacity
mixers, while the compressor/expander and input/output sides were very small.

## Code changes

Updated `spectral_feature_compression/core/model/adaptive_mel_locoformer_lite_sfc_2d.py`:

- Added optional encoder-side capacity mixers:
  - `encoder_capacity_mixer_hidden`
  - `encoder_capacity_mixer_layers`
- Added optional decoder-side capacity mixers:
  - `decoder_capacity_mixer_hidden`
  - `decoder_capacity_mixer_layers`
- Applied encoder capacity after `in_proj` and before the adaptive mel SFC
  compressor.
- Applied decoder capacity after the SFC expander and before `out_proj`.
- Streaming path mirrors full forward exactly; these mixers are stateless and
  add no persistent stream cache.

Threaded the same controls through
`spectral_feature_compression/core/model/proposed_separation_models.py`.

## New recipe

Added:

```text
recipes/dnr/models/adaptive-mel-locoformer-lite-sfc.io-balanced.rt192k.fp512keep475/config.yaml
```

Configuration summary:

```yaml
adaptive_mel_loco_d_model: 40
adaptive_mel_loco_capacity_hidden: 2048
adaptive_mel_loco_capacity_layers: 2
adaptive_mel_loco_encoder_capacity_hidden: 4096
adaptive_mel_loco_encoder_capacity_layers: 2
adaptive_mel_loco_decoder_capacity_hidden: 4096
adaptive_mel_loco_decoder_capacity_layers: 2
```

## Parameter/state comparison

Measured with fp512 core settings:

| Variant | Params | fp16 state |
|---|---:|---:|
| baseline | 2,497,376 | 122,880 B |
| IO-balanced | 2,635,138 | 153,600 B |

Baseline parameter distribution:

```text
encoder_capacity      0
compressor            5,970
separator             78,080
separator_capacity    2,408,708
expander              4,292
decoder_capacity      0
out_proj              198
```

IO-balanced parameter distribution:

```text
encoder_capacity      999,586
compressor            8,402
separator             120,640
separator_capacity    499,874
expander              6,644
decoder_capacity      999,586
out_proj              246
```

The new distribution is much less concentrated in the middle capacity branch
while keeping total params in the useful 2-7M range and stream state under the
192 KiB budget.

## Validation

Targeted tests:

```bash
cd /home/cmj/works/ASS
PYTHONPATH=. .venv/bin/python -m pytest tests/test_proposed_separation_models.py -q
```

Result: `14 passed`.

ONNX export/audit from a single-frame fp512 core input:

- Raw legacy export contains `ConstantOfShape` from exporter padding scaffolding.
- `onnxsim` removes `ConstantOfShape`.
- Simplified graph has no `Expand`, `ConstantOfShape`, `Tile`, `ScatterND`, or
  `Unflatten`.

Simplified IO-balanced ONNX op summary:

```text
nodes=373
Add=48, Concat=2, Conv=55, Div=23, MatMul=3, Mul=93,
ReduceMean=28, ReduceSum=1, Reshape=8, Sigmoid=31, Slice=44,
Softmax=1, Sqrt=22, Sub=3, Transpose=11
```

## Notes

The encoder/decoder capacity mixers are frequency-pooled, so they add parameter
capacity without increasing streaming state.  They improve parameter placement
but still represent global per-frame channel capacity rather than detailed
frequency-local decoder modeling.  If this variant trains well but still lacks
high-frequency/SFX detail, the next follow-up should be a lightweight local
encoder/decoder refinement block with carefully bounded 2D kernels.
