# Source-Aware MelBand RoFormer Teacher

Date: 2026-06-11

## Trigger

The earlier strict-NPU `source-aware-residual-sfc` student still used a large
pooled capacity bank to fit the deployment budget.  That is not the right first
step for a serious separator.  This pass implements a performance-first teacher
architecture before any NPU simplification.

## Principle

The model is intentionally **not** NPU-first.  The architecture puts parameters
where separation quality should benefit most:

- adaptive overlapped mel-band tokenization;
- local time-frequency convolution;
- time-axis rotary attention;
- band-axis rotary attention;
- explicit source tokens for Speech/Music/SFX;
- source-axis attention for source competition;
- mixture/source fusion inside every decoder block;
- complex mask plus complex residual reconstruction;
- mixture-consistency projection;
- supervised training with SNR plus multi-resolution spectral losses.

Only after this model proves useful should it be distilled/simplified into an
online/NPU student.

## Implementation

New model file:

```text
spectral_feature_compression/core/model/source_aware_melband_roformer.py
```

Main classes:

- `SourceAwareMelBandRoformer2D`
- `SourceAwareMelBandRoformerModel`
- `RotarySelfAttention`
- `MelBandRoformerBlock2d`
- `SourceAwareDecoderBlock2d`
- `SourceAxisAttention2d`
- `MixtureSourceFusion2d`

Registered builder:

```text
spectral_feature_compression.core.model.proposed_separation_models.build_source_aware_melband_roformer_teacher_system
```

New supervised training task:

```text
spectral_feature_compression/core/tasks/composite_sup_task.py
```

Main class:

- `CompositeSupTask`

New recipe:

```text
recipes/dnr/models/source-aware-melband-roformer.teacher/config.yaml
```

## Architecture

```text
complex STFT [B, M, F, T]
  -> packed RI [B, 2M, T, F]
  -> RI + magnitude + log-magnitude feature frontend
  -> adaptive overlapped mel SFC compressor, F=1025 -> K=128
  -> encoder blocks:
       local TF depthwise conv
       time-axis rotary self-attention
       band-axis rotary self-attention
       channel FFN
  -> initialize explicit source tokens from mixture tokens + source embeddings
  -> decoder blocks:
       per-source MelBand RoFormer block
       source-axis attention over Speech/Music/SFX at every time/band position
       mixture/source/other-source fusion
  -> SFC expansion K=128 -> F=1025
  -> source-shared reconstruction head:
       complex mask
       complex residual
  -> apply mask + residual
  -> mixture-consistency projection
  -> complex estimates [B, N, M, F, T]
```

## Default teacher recipe size

```yaml
melband_roformer_n_bands: 128
melband_roformer_d_model: 192
melband_roformer_n_heads: 8
melband_roformer_source_attention_heads: 1
melband_roformer_encoder_layers: 8
melband_roformer_decoder_layers: 4
melband_roformer_ffn_mult: 4
melband_roformer_conv_kernel_size: [5, 5]
melband_roformer_routing_kernel_size: [3, 5]
```

Full task/model instantiation result:

```text
CompositeSupTask
ModelWrapper
SourceAwareMelBandRoformer2D
n_freq = 1025
n_bands = 128
d_model = 192
params = 10,958,019
composite spectral loss enabled = True
```

This parameter count is deliberately above the strict-NPU students and below the
very large SFC/Locoformer medium class.  It is meant to be a trainable quality
teacher/reference, not the final deployment model.

## Training objective

The recipe uses `CompositeSupTask`, combining the existing supervised SNR loss
with additional quality losses:

```yaml
mixture_consistency_weight: 0.1
low_frequency_weight: 0.2
low_frequency_hz: 300.0
complex_ri_weight: 0.5
log_magnitude_weight: 0.2
multi_resolution_stft_weight: 0.3
multi_resolution_stft_resolutions:
  - [512, 128]
  - [1024, 256]
  - [2048, 512]
transient_weight: 0.08
```

This is much closer to a serious separation training setup than the previous
student-only supervised recipe.

## Follow-up Config Refresh

On 2026-06-25, the teacher recipe was made standalone instead of inheriting from
`online-soft-band-query-sfc2d.causal96dim.12l.musical64/config.yaml`.

Key changes:

- Inlined the inherited top-level training scalars, trainer defaults, supervised
  loss, optimizer, and scheduler settings directly in
  `recipes/dnr/models/source-aware-melband-roformer.teacher/config.yaml`.
- Replaced the inherited HDF5 datamodule with the same TV-domain on-the-fly
  stem synthesis profile used by the TVConv student recipe: football/commentary,
  live-concert vocal/music, karaoke music-control, and general CASS profiles.
- Kept the teacher `batch_size: 1` while matching the student's synthesis
  distribution; the RoFormer teacher is much heavier than the NPU student.
- Removed stale inherited online-student model keys from the teacher model
  mapping so the config describes only the RoFormer teacher.

## Validation

Focused teacher test:

```bash
cd /home/cmj/works/ASS
PYTHONPATH=.:aiaccel .venv/bin/python -m pytest \
  tests/test_proposed_separation_models.py::test_source_aware_melband_roformer_teacher_forward_and_recipe -q
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
20 passed
```

Ruff:

```bash
cd /home/cmj/works/ASS
.venv/bin/python -m ruff check \
  spectral_feature_compression/core/model/source_aware_melband_roformer.py \
  spectral_feature_compression/core/tasks/composite_sup_task.py \
  spectral_feature_compression/core/model/proposed_separation_models.py \
  spectral_feature_compression/__init__.py \
  tests/test_proposed_separation_models.py
```

Result:

```text
All checks passed
```

Full task instantiation:

```bash
cd /home/cmj/works/ASS
PYTHONPATH=.:aiaccel .venv/bin/python - <<'PY'
from pathlib import Path
from hydra.utils import instantiate
from aiaccel.config import load_config, resolve_inherit
p = Path('recipes/dnr/models/source-aware-melband-roformer.teacher/config.yaml')
config = load_config(p, {
    'config_path': str(p),
    'working_directory': str(p.parent.resolve()),
    'base_config_path': str(Path('aiaccel/aiaccel/torch/apps/config').resolve()),
})
config = resolve_inherit(config)
task = instantiate(config.task).eval()
core = task.model.model.core
print(type(task).__name__)
print(type(task.model).__name__)
print(type(core).__name__)
print(sum(p.numel() for p in core.parameters()))
print(task.composite_loss.enabled)
PY
```

Result:

```text
CompositeSupTask
ModelWrapper
SourceAwareMelBandRoformer2D
10958019
True
```

Follow-up full task instantiation after standalone config refresh:

```bash
cd /home/cmj/works/ASS
PYTHONPATH=.:aiaccel .venv/bin/python - <<'PY'
from pathlib import Path
from hydra.utils import instantiate
from aiaccel.config import load_config, resolve_inherit
p = Path('recipes/dnr/models/source-aware-melband-roformer.teacher/config.yaml')
config = resolve_inherit(load_config(p, {
    'config_path': str(p),
    'working_directory': str(p.parent.resolve()),
    'base_config_path': str(Path('aiaccel/aiaccel/torch/apps/config').resolve()),
}))
task = instantiate(config.task).eval()
print(type(task).__name__)
print(type(task.model.model.core).__name__)
print(config.datamodule._target_)
print([profile.name for profile in config.datamodule.synthesis.synthesis_profiles])
PY
```

Result:

```text
CompositeSupTask
SourceAwareMelBandRoformer2D
spectral_feature_compression.common.datamodules.on_the_fly_stem_datamodule.OnTheFlyStemDataModule
['football_commentary_focus', 'live_concert_vocal_music', 'karaoke_music_control', 'general_cass']
```

Follow-up focused recipe test:

```bash
cd /home/cmj/works/ASS
.venv/bin/python -m pytest \
  tests/test_proposed_separation_models.py::test_source_aware_melband_roformer_teacher_forward_and_recipe -q
```

Result:

```text
1 passed
```

Follow-up focused suite:

```bash
cd /home/cmj/works/ASS
.venv/bin/python -m pytest \
  tests/test_on_the_fly_source_normalization.py \
  tests/test_proposed_separation_models.py -q
```

Result:

```text
49 passed
```

## Training command

```bash
cd /home/cmj/works/ASS
PYTHONPATH=.:aiaccel .venv/bin/python -m aiaccel.torch.apps.train \
  recipes/dnr/models/source-aware-melband-roformer.teacher/config.yaml
```

## Next step

Train this teacher and compare validation SNR/SI-SDR against existing teachers
and proposal branches.  If it is strong, then design the NPU student as a
distilled approximation of this architecture, not as a hand-sized model with a
large pooled capacity patch.
