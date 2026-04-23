# Recipes

This file collects experimental recipes added on top of the DCASE 2026 Task 4
baseline. The official-style baseline remains available in
`config/label/m2dat_*.yaml` and `config/separation/resunetk_capisdr.yaml`.

## Quick Recommendation

For new DeFT experiments, start with the memory-efficient path:

```bash
cd ./dcase2026_task4_baseline

# stage 0: memory-efficient universal source separation
python -m src.train -c config/separation/modified_deft_uss_lite_6s.yaml -w workspace/separation

# stage 1: single-label classifier, ArcFace stage
python -m src.train -c config/label/m2d_sc_stage1_strong.yaml -w workspace/label

# stage 2: single-label classifier, energy/silence stage
python -m src.train -c config/label/m2d_sc_stage2_strong.yaml -w workspace/label

# stage 3: memory-efficient target sound extractor
python -m src.train -c config/separation/modified_deft_tse_lite_6s.yaml -w workspace/separation

# optional stage 4: fine-tune SC on cached USS/TSE estimated sources
python -m src.train -c config/label/m2d_sc_stage3_estimated_strong.yaml -w workspace/label
```

Then place or symlink the trained checkpoints where the evaluation config
expects them:

```text
checkpoint/modified_deft_uss_lite_6s.ckpt
checkpoint/m2d_sc_stage3_estimated_strong.ckpt
checkpoint/modified_deft_tse_lite_6s.ckpt
```

Run 10s evaluation with:

```bash
python -m src.evaluation.evaluate \
  -c src/evaluation/eval_configs/kwo2025_top1_like_lite_estimated_sc.yaml \
  --result_dir workspace/evaluation
```

This path is the safest default because both DeFT-heavy stages use local
attention instead of global full-clip attention.

## Model Picker

| Goal | Config | Model class | Use when |
| --- | --- | --- | --- |
| Official-style AT + SS baseline | `config/label/m2dat_*.yaml`, `config/separation/resunetk_capisdr.yaml` | `M2dAt`, `ResUNet30` | You want the stable baseline path. |
| Strong spatial USS, full global DeFT | `config/separation/modified_deft_uss.yaml` | `ModifiedDeFTUSSSpatial` | You want maximum context and have enough GPU memory. |
| Shorter USS training, chunked eval | `config/separation/modified_deft_uss_5s.yaml` | `ChunkedModifiedDeFTUSSSpatial` | You want the same global DeFT block but need shorter 5s training clips. |
| Memory-efficient spatial USS | `config/separation/modified_deft_uss_lite_6s.yaml` | `ModifiedDeFTUSSMemoryEfficient` | Recommended new USS path; local time attention and grouped frequency attention. |
| KAIST-like oracle TSE, full global DeFT | `config/separation/modified_deft_tse.yaml` | `ModifiedDeFTTSE` | You want the original oracle-enrollment TSE structure and can afford memory. |
| Memory-efficient oracle TSE | `config/separation/modified_deft_tse_lite_6s.yaml` | `ModifiedDeFTTSEMemoryEfficient` | Recommended new TSE path; fixes the TSE global-attention memory issue. |
| Strong M2D source classifier | `config/label/m2d_sc_stage*_strong.yaml` | `M2DSingleClassifierStrong` | Recommended SC path; attentive statistics pooling and multi-crop prediction. |
| Estimated-source SC fine-tuning | `config/label/m2d_sc_stage3_estimated_strong.yaml` | `EstimatedSourceClassifierDataset` + `M2DSingleClassifierStrong` | Optional final SC adaptation on cached USS/TSE outputs. |
| One-stage label-query separator | `config/separation/deft_tse_like.yaml` | `DeFTTSELikeSpatial` | You want a simpler label-query separator instead of `USS -> SC -> TSE`. |

## What Changed

### Spatial Masking

`ModifiedDeFTUSSSpatial` and `DeFTTSELikeSpatial` replace the old 2-output
complex mask:

```text
[complex_mask_real, complex_mask_imag]
```

with a 3-component phase-aware mask for every target and every mixture channel:

```text
[magnitude_mask, phase_mask_real, phase_mask_imag]
```

The reconstruction path is:

1. Predict one magnitude mask and one phase-rotation vector per input channel.
2. Apply `sigmoid` to magnitude masks.
3. Normalize the phase real/imag pair with `magphase()`.
4. Reconstruct complex estimates for all mixture channels.
5. Use a learned `4 -> 1` projection for mono output.
6. Run ISTFT and return the same output keys as before.

This follows the useful part of the ResUNet final stage where `K = 3` means
magnitude plus phase components, while preserving spatial information until the
last projection.

### Memory-Efficient DeFT Blocks

The original DeFT block applies global attention over the full STFT time axis
and full frequency axis. For 10s audio at 32 kHz with `window_size=1024` and
`hop_size=320`, the model sees about:

```text
T ~= 1001 frames
F = 513 bins
```

That makes memory dominated by attention activations, not parameters.

`MemoryEfficientDeFTBlock` keeps the same `[B, C, T, F]` interface but changes
the attention pattern:

- time attention is local within `time_window_size` STFT frames
- frequency attention is local within `freq_group_size` bins
- odd-numbered blocks use half-window shifts so neighboring windows/groups can
  exchange information across depth

Default lite settings:

```yaml
time_window_size: 128
freq_group_size: 64
shift_windows: true
```

Increasing either window can improve context but raises memory quadratically
along that axis.

### Strong Source Classifier

The original `M2DSingleClassifier` is intentionally compact: it averages all M2D
time tokens, projects once, and applies an ArcFace head. That is a reasonable
baseline, but it is probably too simple for a top-ranking system because source
estimates can be partial, distorted, or contain residual interference.

`M2DSingleClassifierStrong` keeps the M2D backbone and output contract, but
changes the classifier head:

- attentive statistics pooling over M2D time tokens
- concatenated mean, max, attention mean, and attention std features
- MLP projection before ArcFace
- dropout and weight decay for better regularization
- optional multi-crop prediction for long source estimates

The output keys are unchanged:

```text
embedding, logits, plain_logits, energy
```

so the existing `SingleLabelClassificationLightning`, `m2d_sc_arcface` loss,
and S5 inference code can use it directly.

The M2D-SC fine-tuning selector now follows the same convention as the
multi-output M2D tagger: `2_blocks` really unfreezes the last two transformer
blocks plus the output head. Earlier classifier runs from before this fix may
have trained mostly the output head instead of the intended backbone blocks.

### Chunked Evaluation

The short/lite configs train on 5s or 6s synthesized clips, but evaluation
datasets contain fixed 10s waveforms. The chunked model variants keep
evaluation at 10s by splitting long inputs internally and overlap-adding the
output waveform back to the original length.

Current chunk defaults:

| Config | Train duration | Eval chunk | Eval hop |
| --- | ---: | ---: | ---: |
| `modified_deft_uss_5s.yaml` | 5.0s | 6.0s | 5.0s |
| `modified_deft_uss_lite_6s.yaml` | 6.0s | 10.0s | 8.0s |
| `modified_deft_tse_lite_6s.yaml` | 6.0s | 10.0s | 8.0s |

With the lite models, 10s eval chunks are usually more realistic because the
attention block itself is memory-bounded.

## Training Guides

### Recommended Memory-Efficient Multi-Stage System

Train these stages:

```bash
cd ./dcase2026_task4_baseline

python -m src.train -c config/separation/modified_deft_uss_lite_6s.yaml -w workspace/separation
python -m src.train -c config/label/m2d_sc_stage1_strong.yaml -w workspace/label
python -m src.train -c config/label/m2d_sc_stage2_strong.yaml -w workspace/label
python -m src.train -c config/separation/modified_deft_tse_lite_6s.yaml -w workspace/separation
```

The two lite DeFT stages synthesize 6s mixtures on the fly using SpAudSyn:

```yaml
spatial_sound_scene:
  duration: 6.0
  max_event_dur: 6.0
```

They use `bf16-mixed` by default. If your GPU does not support bf16 well, change
the trainer precision to `16-mixed` or `32-true` in the YAML. `32-true` is
safer numerically but much heavier.

### Full-Context Spatial USS

Use this only if memory allows:

```bash
python -m src.train -c config/separation/modified_deft_uss.yaml -w workspace/separation
```

This uses `ModifiedDeFTUSSSpatial` with full global DeFT attention. It can be
very expensive for 10s audio because each block attends over the full time and
frequency axes.

### Short 5s USS With Chunked 10s Evaluation

Use this when you want the original global DeFT block but cannot train on 10s:

```bash
python -m src.train -c config/separation/modified_deft_uss_5s.yaml -w workspace/separation
```

This reduces training memory by synthesizing 5s mixtures, while
`ChunkedModifiedDeFTUSSSpatial` can process 10s evaluation audio in 6s chunks.

The tradeoff is that the model is trained with shorter temporal context, which
can affect SA-SDR for long events or events crossing chunk boundaries.

### Original KAIST-Like TSE

The older TSE config remains available:

```bash
python -m src.train -c config/separation/modified_deft_tse.yaml -w workspace/separation
```

It uses `ModifiedDeFTTSE`, raw enrollment waveform injection, and class
conditioning, but its final mask is still the older channel-0 complex-mask
synthesis. It also uses global DeFT attention, so it is memory-heavy.

### Original Simple M2D Source Classifier

The old two-stage classifier configs remain available:

```bash
python -m src.train -c config/label/m2d_sc_stage1.yaml -w workspace/label
python -m src.train -c config/label/m2d_sc_stage2.yaml -w workspace/label
```

Use them for ablation against `M2DSingleClassifierStrong`. For top-rank
experiments, prefer the `_strong` configs.

### Fine-Tuning SC On Estimated Sources

Clean dry-source classifier training is useful, but the classifier is used on
imperfect USS/TSE estimates during S5 inference. A top-rank system should adapt
SC to those artifacts.

The new stage-3 SC config is:

```bash
python -m src.train -c config/label/m2d_sc_stage3_estimated_strong.yaml -w workspace/label
```

It expects a cached waveform dataset:

```text
workspace/sc_finetune/soundscape
workspace/sc_finetune/oracle_target
workspace/sc_finetune/estimate_target
```

`soundscape` and `oracle_target` follow the normal synthesized waveform dataset
layout. `estimate_target` contains separated sources from your trained USS/TSE
pipeline. Estimated files must use this naming convention:

```text
<soundscape>_<slot>_<label>.wav
```

Example:

```text
soundscape_000001_0_Speech.wav
soundscape_000001_1_Doorbell.wav
```

The label in the filename is used as the SC training target.

There are two ways to build the estimate cache:

- Supervised adaptation: run your separator/TSE with oracle labels and save
  estimates using oracle class names in the filenames. This is the preferred
  validation/dev adaptation path.
- Pseudo-label adaptation: run the full S5 pipeline and save estimates using
  predicted class names. This is easier, but label noise can hurt the SC if the
  first-pass classifier is weak.

Use the export utility to create the cache:

```bash
# supervised cache: oracle labels + TSE-estimated waveforms
python -m src.evaluation.export_sc_finetune_cache \
  -c src/evaluation/eval_configs/kwo2025_top1_like_lite_strong_sc.yaml \
  --output_root workspace/sc_finetune \
  --mode oracle_tse \
  --batchsize 1

# pseudo-label cache: full S5 predictions and predicted labels
python -m src.evaluation.export_sc_finetune_cache \
  -c src/evaluation/eval_configs/kwo2025_top1_like_lite_strong_sc.yaml \
  --output_root workspace/sc_finetune_pseudo \
  --mode pseudo_s5 \
  --batchsize 1
```

The supervised mode writes:

```text
workspace/sc_finetune/soundscape/<soundscape>.wav
workspace/sc_finetune/oracle_target/<soundscape>_<slot>_<oracle_label>.wav
workspace/sc_finetune/estimate_target/<soundscape>_<slot>_<oracle_label>.wav
```

The pseudo-label mode writes the same folder structure, but
`estimate_target` filenames use the model-predicted labels.

After training, use the matching evaluation config:

```bash
python -m src.evaluation.evaluate \
  -c src/evaluation/eval_configs/kwo2025_top1_like_lite_estimated_sc.yaml \
  --result_dir workspace/evaluation
```

That config expects:

```text
checkpoint/modified_deft_uss_lite_6s.ckpt
checkpoint/m2d_sc_stage3_estimated_strong.ckpt
checkpoint/modified_deft_tse_lite_6s.ckpt
```

For an ablation without estimated-source SC adaptation, use
`kwo2025_top1_like_lite_strong_sc.yaml`, which expects
`checkpoint/m2d_sc_stage2_strong.ckpt`.

### One-Stage Label-Query Separator

For a simpler separator that skips the explicit `USS -> SC -> TSE` chain:

```bash
python -m src.train -c config/separation/deft_tse_like.yaml -w workspace/separation
```

This uses `DeFTTSELikeSpatial`, which already has the all-channel spatial final
stage. It is trained through `LabelQueriedSeparationLightning` and
`class_aware_pit`.

## Evaluation Guides

### Recommended Lite Multi-Stage Evaluation

Use this after training the lite USS, M2D-SC stage 2, and lite TSE checkpoints:

```bash
python -m src.evaluation.evaluate \
  -c src/evaluation/eval_configs/kwo2025_top1_like_lite_estimated_sc.yaml \
  --result_dir workspace/evaluation
```

This evaluation config reads fixed 10s files from:

```text
data/dev_set/synthesized/test/soundscape
data/dev_set/synthesized/test/oracle_target
```

It does not shorten evaluation audio. The lite USS and lite TSE models handle
long inputs internally, and the strong SC averages predictions over 5s crops
with 2.5s hop when classifying longer source estimates.

If you want the same lite USS/TSE system with the strong clean-source
classifier, use:

```bash
python -m src.evaluation.evaluate \
  -c src/evaluation/eval_configs/kwo2025_top1_like_lite_strong_sc.yaml \
  --result_dir workspace/evaluation
```

If you want the same lite USS/TSE system with the older simple classifier, use:

```bash
python -m src.evaluation.evaluate \
  -c src/evaluation/eval_configs/kwo2025_top1_like_lite_uss.yaml \
  --result_dir workspace/evaluation
```

### Chunked-USS Evaluation With Original TSE

This config uses chunked USS but keeps the original full-attention TSE:

```bash
python -m src.evaluation.evaluate \
  -c src/evaluation/eval_configs/kwo2025_top1_like_chunked_uss.yaml \
  --result_dir workspace/evaluation
```

Use it only when you trained `modified_deft_uss_5s.yaml` and still want the
older `ModifiedDeFTTSE` stage.

### Original KAIST-Like Evaluation

```bash
python -m src.evaluation.evaluate \
  -c src/evaluation/eval_configs/kwo2025_top1_like.yaml \
  --result_dir workspace/evaluation
```

This uses the original full-attention DeFT USS/TSE settings from that eval
config.

## Checkpoint Notes

Checkpoint compatibility is intentionally strict:

- `ModifiedDeFTUSS` checkpoints do not directly load into
  `ModifiedDeFTUSSSpatial`, because the audio head shape changed and `out_conv`
  was added.
- `ModifiedDeFTUSSSpatial` checkpoints do not directly load into
  `ModifiedDeFTUSSMemoryEfficient`, because the DeFT block modules changed.
- `ModifiedDeFTTSE` checkpoints do not directly load into
  `ModifiedDeFTTSEMemoryEfficient` for the same reason.
- `DeFTTSELike` checkpoints do not directly load into `DeFTTSELikeSpatial`.
- `M2DSingleClassifier` checkpoints do not directly load into
  `M2DSingleClassifierStrong`, because the pooling/projection head changed.

When switching between these families, train a fresh checkpoint or write an
explicit partial-loading script that only transfers compatible layers.

## Dataset And External Data Notes

The DeFT separation configs use SpAudSyn on the fly. Make sure the baseline data
layout exists before training:

```text
data/dev_set/sound_event/train
data/dev_set/interference/train
data/dev_set/noise/train
data/dev_set/room_ir/train
```

If EARS and SemanticHearing have been added with `add_data.sh`, they become part
of the training pools under those folders. The evaluation configs do not use
SpAudSyn; they read prepared 10s waveform files from the synthesized test set.

For estimated-source SC fine-tuning, `DatasetS3` reads `estimate_target_dir` and
`EstimatedSourceClassifierDataset` trains on `est_dry_sources` / `est_label`.
The label is parsed from the estimate filename, so keep filenames aligned with
the intended target labels.

## Practical Tuning

Recommended first try on a memory-constrained GPU:

```yaml
batch_size: 2
precision: bf16-mixed
time_window_size: 128
freq_group_size: 64
```

For `M2DSingleClassifierStrong`, start with:

```yaml
batch_size: 32
precision: bf16-mixed
pooling_hidden_dim: 512
projection_hidden_dim: 1024
dropout: 0.2
eval_crop_seconds: 5.0
eval_crop_hop_seconds: 2.5
```

If memory is still too high:

- reduce `batch_size` to `1`
- reduce `n_deft_blocks` from `6` to `4`
- reduce `time_window_size` from `128` to `96` or `64`
- reduce `freq_group_size` from `64` to `32`
- keep `hop_size: 320` unless you intentionally want a time-resolution tradeoff

If quality is too low and memory allows:

- increase `time_window_size` to `192` or `256`
- increase `freq_group_size` to `96` or `128`
- keep `shift_windows: true`
- compare 6s training against 5s training on the same validation split

## Quick CPU Sanity Checks

These checks do not measure quality. They only verify config instantiation,
shape compatibility, and finite losses on CPU.

The M2D-SC construction check requires the pretrained M2D checkpoint referenced
by the YAML. If that checkpoint is not present, skip that one check or run it
after placing the M2D weights under `checkpoint/`.

```bash
cd ./dcase2026_task4_baseline

# Check the memory-efficient USS model can be constructed from YAML.
PYTHONPATH=. .venv/bin/python - <<'PY'
import yaml
from src.utils import initialize_config

with open("config/separation/modified_deft_uss_lite_6s.yaml") as f:
    cfg = yaml.safe_load(f)
model = initialize_config(cfg["lightning_module"]["args"]["model"])
print(type(model).__name__, type(model.blocks[0]).__name__, model.time_window_size, model.freq_group_size)
PY

# Check the memory-efficient TSE model can be constructed from YAML.
PYTHONPATH=. .venv/bin/python - <<'PY'
import yaml
from src.utils import initialize_config

with open("config/separation/modified_deft_tse_lite_6s.yaml") as f:
    cfg = yaml.safe_load(f)
model = initialize_config(cfg["lightning_module"]["args"]["model"])
print(type(model).__name__, type(model.blocks[0]).__name__, model.time_window_size, model.freq_group_size)
PY

# Check the strong source classifier can be constructed from YAML.
PYTHONPATH=. .venv/bin/python - <<'PY'
import yaml
from src.utils import initialize_config

with open("config/label/m2d_sc_stage2_strong.yaml") as f:
    cfg = yaml.safe_load(f)
model = initialize_config(cfg["lightning_module"]["args"]["model"])
print(type(model).__name__, type(model.pool).__name__, model.embedding[-1].normalized_shape)
PY

# Check the estimated-source SC dataset can be configured.
PYTHONPATH=. .venv/bin/python - <<'PY'
import yaml

with open("config/label/m2d_sc_stage3_estimated_strong.yaml") as f:
    cfg = yaml.safe_load(f)
dargs = cfg["datamodule"]["args"]["train_dataloader"]["dataset"]["args"]
print(dargs["source_prefix"], dargs["base_dataset"]["args"]["config"]["estimate_target_dir"])
PY

# Check the cache exporter CLI imports.
PYTHONPATH=. .venv/bin/python -m src.evaluation.export_sc_finetune_cache --help

# Check the lite USS training dataset produces 6s audio.
PYTHONPATH=. .venv/bin/python - <<'PY'
import yaml
from src.utils import initialize_config

with open("config/separation/modified_deft_uss_lite_6s.yaml") as f:
    cfg = yaml.safe_load(f)
dset = initialize_config(cfg["datamodule"]["args"]["train_dataloader"]["dataset"])
item = dset[0]
print(item["mixture"].shape, item["mixture"].shape[-1] / 32000)
PY

# Check the lite TSE training dataset produces 6s mixture and enrollment audio.
PYTHONPATH=. .venv/bin/python - <<'PY'
import yaml
from src.utils import initialize_config

with open("config/separation/modified_deft_tse_lite_6s.yaml") as f:
    cfg = yaml.safe_load(f)
dset = initialize_config(cfg["datamodule"]["args"]["train_dataloader"]["dataset"])
item = dset[0]
print(item["mixture"].shape, item["enrollment"].shape, item["mixture"].shape[-1] / 32000)
PY
```

## Remaining Work

- The current TSE training wrapper uses oracle foreground enrollments from the
  dataset. The 2025 KAIST-style system trained TSE with USS-generated
  enrollments, so a future fidelity upgrade is to cache or generate USS
  enrollments from a trained USS checkpoint.
- Per-class energy thresholds for `M2D-SC` should be calibrated from validation
  data instead of relying on a placeholder default.
- Full validation quality still needs real GPU runs; the CPU checks only verify
  that the code paths are wired correctly.

## Review And Fast-Test Notes

The current implementation was reviewed with CPU-focused smoke tests. These are
not quality measurements, but they catch shape, config, loss, and wiring
regressions before launching expensive GPU runs.

Fixes made during review:

- `src/train.py`: `--batchsize` no longer assumes every config has a
  `val_dataloader`.
- `src/utils.py`: missing `transformers` no longer breaks basic training imports.
- `.gitignore`: local `data/`, `.venv/`, `external_data/`, and the local
  SpAudSyn symlink are ignored.
- `src/datamodules/dataset.py`: waveform-mode source file matching now uses the
  exact soundscape filename pattern instead of a broad `startswith`, avoiding
  prefix collisions such as `soundscape_000001` matching `soundscape_0000010`.
- `src/evaluation/evaluate.py`: the optional label-conditioned prediction path
  now calls `self.model.separate(...)`.
- `src/models/m2dat/m2d_sc.py`: `2_blocks` now truly unfreezes the last two M2D
  transformer blocks plus the classifier head.

Fast checks run:

```bash
# Python syntax for source files, excluding the local SpAudSyn symlink
PYTHONPATH=. .venv/bin/python -m py_compile $(find src -name '*.py' -not -path 'src/modules/spatial_audio_synthesizer/*' | sort)

# YAML parse for all train/eval configs
PYTHONPATH=. .venv/bin/python - <<'PY'
import yaml
from pathlib import Path
for path in sorted(Path("config").rglob("*.yaml")) + sorted(Path("src/evaluation/eval_configs").rglob("*.yaml")):
    with open(path) as f:
        yaml.safe_load(f)
    print("ok", path)
PY
```

Additional targeted checks passed:

- `ModifiedDeFTUSSMemoryEfficient` forward plus USS loss on tiny tensors.
- `ModifiedDeFTTSEMemoryEfficient` forward plus masked-SNR loss on tiny tensors.
- `DeFTTSELikeSpatial` forward plus class-aware PIT loss on tiny tensors.
- `M2DSingleClassifierStrong` forward, predict, loss, and `2_blocks`
  trainability using a stub M2D backbone because the real M2D checkpoint was not
  present on this host.
- `EstimatedSourceClassifierDataset` with a temporary cache, including exact
  filename matching and silence padding.
- `export_sc_finetune_cache.py --help`.
- `export_sc_finetune_cache.py` with a temporary waveform dataset and fake S5
  model, producing the expected `soundscape/`, `oracle_target/`, and
  `estimate_target/` files.
