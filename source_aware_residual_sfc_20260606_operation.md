# Source-Aware Residual SFC NPU Design

Date: 2026-06-06

## Trigger

The previous adaptive-mel Loco-CNB BandSFC branch is deployable, but the low
supervised validation SNR around `3.5 dB` suggests it is not a strong enough
separator.  The residual-SFX ablation is useful for diagnosis, but it does not
materially increase modeling power.  This pass therefore designs a new strict
online/NPU student architecture instead of only selecting another existing
variant.

## Design target

DnR 3-stem separation (`Speech`, `Music`, `Effects/SFX`) needs source priors,
not just a compact generic Conv2d stack:

- speech: formants, harmonic pitch, fricatives, syllabic temporal envelopes;
- music: dense harmonic structure, sustained notes, wide-band content;
- effects: transients, impacts, noise-like and ambience-like events.

The edge/NPU constraints remain:

- causal / online streaming;
- runtime tensors `<= 4D`;
- avoid 1D ops and unsupported control flow;
- use only NPU-stable primitives such as `Conv2d`, `bmm`/`MatMul`, `Softmax`,
  `Sigmoid`, `Reshape`, `Transpose`, `Slice`, `Concat`, elementwise ops;
- fp16 persistent stream state `< 192 KiB`;
- parameters `< 7M`;
- target compute `< 3 GMAC/s` at `44100 / 512` frames/s.

## New architecture

Implemented:

```text
spectral_feature_compression/core/model/source_aware_residual_sfc_2d.py
```

Main classes/builders:

- `OnlineSourceAwareResidualSFC2D`
- `OnlineSourceAwareResidualSFCModel`
- `LowRankResidualCorrectionHead2d`
- `build_source_aware_residual_sfc_system`

The new deployment recipe is:

```text
recipes/dnr/models/source-aware-residual-sfc.rt192k.fp512keep475/config.yaml
```

A teacher-guided recipe is also prepared:

```text
recipes/dnr/models/source-aware-residual-sfc.distill.rt192k.fp512keep475/config.yaml
```

### Data flow

```text
packed complex STFT [B, 2*M, T, F]
  -> 1x1 projection
  -> adaptive SFC query compressor: F=512 -> K=56 bands
  -> shared dilated band-mix analysis blocks
  -> pooled channel capacity mixers on [B, C, T, 1]
  -> long temporal compressed-token branch
  -> early source split: shared tokens -> Speech/Music/SFX token streams
  -> shared per-source causal token refiner
  -> cross-source shared reconstruction decoder -> primary complex masks
  -> low-rank full-band residual correction from mixture + masks + context
  -> corrected complex masks
  -> complex estimates [B, 2*N*M, T, F]
```

### Why this is different from the old BandSFC branch

The old stable adaptive-mel Loco-CNB branch spends most of its trainable capacity
inside frequency-pooled channel mixers.  That is compute-safe, but weakly tied
to actual source disentanglement.  The new design spends capacity and state on
separation-relevant structure:

1. **Early source split**: source streams are separated before decoding, not only
   at the final mask head.
2. **Cross-source decoder**: each source sees mixture-token and other-source
   context before reconstruction, which is important for competition between
   speech/music/effects.
3. **Compressed long-context branch**: temporal refinement happens on `K=56`
   SFC tokens, not the full frequency axis.
4. **Low-rank full-band mask correction**: local full-frequency artifacts and
   transient leakage are corrected with only `12` correction channels, avoiding a
   large recurrent cache while preserving host-side mask application semantics.
5. **No source-axis 5D tensor**: the source axis is packed into channels, so all
   runtime tensors remain 4D.

## Default deploy configuration

```yaml
source_aware_residual_d_model: 28
source_aware_residual_n_bands: 56
source_aware_residual_shared_layers: 2
source_aware_residual_source_layers: 2
source_aware_residual_long_branch_layers: 1
source_aware_residual_correction_layers: 1
source_aware_residual_correction_channels: 12
source_aware_residual_shared_capacity_hidden: 8192
source_aware_residual_shared_capacity_layers: 4
source_aware_residual_kernel_size: [3, 3]
source_aware_residual_routing_kernel_size: [1, 3]
source_aware_residual_dilation_cycle: [1, 2, 4]
source_aware_residual_long_branch_dilation_cycle: [1, 2, 4]
freq_preprocess_keep_bins: 475
freq_preprocess_target_bins: 512
```

The routing kernel is `[1, 3]`, so the adaptive SFC compressor does not add a
full-frequency temporal cache.  The persistent state is spent on compressed-band
shared/source/long branches plus the low-rank full-band correction branch.

## Profiling result

Command:

```bash
cd /home/cmj/works/ASS
.venv/bin/python - <<'PY'
import torch
from spectral_feature_compression.core.model.source_aware_residual_sfc_2d import OnlineSourceAwareResidualSFC2D

MAC_KEYS=("conv","mm","bmm","matmul","addmm")
model = OnlineSourceAwareResidualSFC2D(
    n_freq=512,
    n_bands=56,
    sample_rate=44100,
    n_src=3,
    n_chan=1,
    d_model=28,
    n_shared_layers=2,
    n_source_layers=2,
    long_branch_layers=1,
    correction_layers=1,
    correction_channels=12,
    shared_capacity_hidden=8192,
    shared_capacity_layers=4,
).eval()
x = torch.randn(1, 2, 1, model.n_freq)
with torch.no_grad():
    model(x)
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) as prof:
    with torch.no_grad():
        y = model(x)
flops = 0
for evt in prof.key_averages():
    f = int(evt.flops or 0)
    if f and any(k in evt.key.lower() for k in MAC_KEYS):
        flops += f
mac = flops / 2.0
print(tuple(y.shape))
print(sum(p.numel() for p in model.parameters()))
print(model.state_size_bytes(dtype=torch.float16) / 1024.0)
print(len(model.init_stream_state(batch_size=1, dtype=torch.float32)))
print(int(mac))
print(mac * (44100 / 512) / 1e9)
PY
```

Measured locally:

| Model/core | Params | fp16 state | State tensors | MAC/frame | GMAC/s |
|---|---:|---:|---:|---:|---:|
| `source_aware_residual_sfc` | 2,867,434 | 170.50 KiB | 10 | 14,905,856 | 1.284 |
| `source_split` proposal | 2,449,230 | 112.00 KiB | 8 | 15,321,088 | 1.320 |
| `residual_refine` proposal | 2,451,392 | 144.00 KiB | 5 | 11,744,256 | 1.012 |
| old stable residual-SFX core | 2,852,417 | 185.62 KiB | 5 | 15,627,456 | 1.346 |

Interpretation:

- The new model stays below all hard budgets.
- It has more source-aware structure than `source_split` alone and adds a
  residual correction path absent from `source_split`.
- It has explicit source streams absent from `residual_refine` alone.
- It uses less state and slightly less compute than the old stable residual-SFX
  BandSFC core while outputting all 3 stems explicitly.

## Tests

Focused test:

```bash
cd /home/cmj/works/ASS
PYTHONPATH=.:aiaccel .venv/bin/python -m pytest \
  tests/test_proposed_separation_models.py::test_source_aware_residual_sfc_builder_forward_streaming_and_recipe -q
```

Result:

```text
1 passed
```

Full proposal test file:

```bash
cd /home/cmj/works/ASS
PYTHONPATH=.:aiaccel .venv/bin/python -m pytest tests/test_proposed_separation_models.py -q
```

Result:

```text
19 passed
```

The new tests check:

- waveform wrapper forward shape;
- packed 2D core forward shape;
- frame-by-frame streaming equality with full forward;
- 10-state streaming cache contract for the deploy preset;
- deploy parameter range `2M..7M`;
- fp16 state `< 192 KiB`;
- recipe config resolution and instantiation.

## ONE/NPU verification

Stateless verification command:

```bash
cd /home/cmj/works/ASS
.venv/bin/python tools/online/verify_npu_variants.py \
  --mode recipe \
  --recipe-name-contains source-aware-residual-sfc.rt192k.fp512keep475 \
  --run-name source_aware_residual_sfc_20260606 \
  --force-onnxsim-large-shape-ops \
  --quantize-layer-fallback
```

Result:

```text
[PASS] recipe:source-aware-residual-sfc.rt192k.fp512keep475
```

Stateful streaming verification command:

```bash
cd /home/cmj/works/ASS
.venv/bin/python tools/online/verify_npu_variants.py \
  --mode recipe \
  --recipe-name-contains source-aware-residual-sfc.rt192k.fp512keep475 \
  --run-name source_aware_residual_sfc_streaming_20260606 \
  --force-onnxsim-large-shape-ops \
  --quantize-layer-fallback \
  --streaming
```

Result:

```text
[PASS] recipe:source-aware-residual-sfc.rt192k.fp512keep475
```

ONNX graph summary from generated artifacts:

| Export | Runtime inputs | Runtime outputs | ONNX nodes | Top structural ops |
|---|---:|---:|---:|---|
| stateless | 1 | 1 | 1257 | `Conv=79`, `MatMul=6`, `Softmax=1` |
| streaming | 11 | 11 | 1033 | `Conv=79`, `MatMul=6`, `Softmax=1` |

The streaming runtime inputs/outputs are `x` plus 10 state tensors and `y` plus
10 next-state tensors.  The generated ONNX/Circle verifier artifact folders were
removed after recording these results to avoid leaving large untracked files in
`logs/`; rerun the commands above if artifacts are needed.

## Suggested training order

### 1. Supervised warm start

```bash
cd /home/cmj/works/ASS
PYTHONPATH=.:aiaccel .venv/bin/python -m aiaccel.torch.apps.train \
  recipes/dnr/models/source-aware-residual-sfc.rt192k.fp512keep475/config.yaml
```

Monitor:

- `validation/snr` overall and per stem if available;
- direct short validation vs CSS validation;
- SFX transient quality and silent-source false positives;
- whether correction scale grows too aggressively.

### 2. Distillation from SFC-Locoformer-lite-plus teacher

Set `teacher_checkpoint_path` before launch:

```bash
cd /home/cmj/works/ASS
PYTHONPATH=.:aiaccel .venv/bin/python -m aiaccel.torch.apps.train \
  recipes/dnr/models/source-aware-residual-sfc.distill.rt192k.fp512keep475/config.yaml \
  teacher_checkpoint_path=/path/to/teacher.ckpt \
  task.pretrained_model_path=/path/to/source_aware_residual_supervised_student.ckpt
```

The distillation recipe keeps the same student but increases teacher loss weight
slightly (`0.35`) and transient loss weight (`0.08`) because the residual mask
correction head is intended to learn artifact/transient repair.

## Important caveat

This pass validates architecture, streaming equivalence, profiling budgets,
recipe instantiation, ONNX export, Circle import/optimization, and quantization.
It does **not** prove separation quality yet.  Also, the streaming export uses
10 state tensors: the byte budget fits, but if the final runtime requires fewer
ONNX IO edges, add a packed-state wrapper similar to the DolphinSFCNPU export
wrapper.

The next required evidence is a real DnR training run and comparison against:

1. old stable adaptive-mel Loco-CNB residual-SFX;
2. `source_split` proposal;
3. `residual_refine` proposal;
4. `sparse_unet` proposal if compute budget can tolerate it.
