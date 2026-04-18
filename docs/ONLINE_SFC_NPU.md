## Online / realtime SFC (2D-only) for edge NPU

This repo now includes a refactored "online" variant of the SFC separator designed for **edge NPUs** with constraints:
- **only 2D tensors** in the deployable graph
- deployable kernels are built from `Conv2d`, `torch.bmm` / ONNX `MatMul`, and basic tensor ops
- tensors are **no more than 4D**
- **ONNX export uses `batch_size=1`**
- each deployed conv must satisfy **`(kernel_size - 1) * dilation < 14`**

The deployable core is `OnlineSFC2D`:
- **input**: `x` packed real/imag spectrogram features: `(1, 2*M, T, F)`
- **output**: `y` packed real/imag estimated complex spectrogram: `(1, 2*N*M, T, F)`

Where:
- \(M\) = number of audio channels (`n_chan`)
- \(N\) = number of sources (`n_src`)
- \(T\) = number of STFT frames per inference call
- \(F\) = number of frequency bins (`n_fft//2 + 1`)

The usual STFT / iSTFT is intentionally kept **outside** the NPU graph.

---

## 1) Training

### Data preparation (MUSDB18HQ example)

Follow the existing recipe:

```bash
dataset_name=musdb18hq
./recipes/${dataset_name}/scripts/data.sh
```

This produces the HDF5 files referenced by the training config.

### Train the online model

```bash
./recipes/musdb18hq/models/online-sfc2d.causal96dim.12l/train.sh
```

For the closest NPU-friendly approximation to the paper's cross-attention
encoder/decoder path, use:

```bash
./recipes/musdb18hq/models/online-crossattn-query-sfc2d.causal96dim.12l/train.sh
```

For smaller deployment-oriented variants of the same family, the repository now
also includes:

- `recipes/musdb18hq/models/online-crossattn-query-sfc2d.rt192k.causal24dim.6l.64b/config.yaml`
- `recipes/musdb18hq/models/online-crossattn-query-sfc2d.rt192k.fp512keep475.causal24dim.6l.64b/config.yaml`
- `recipes/musdb18hq/models/online-crossattn-query-sfc2d.rt128k.causal16dim.6l.64b/config.yaml`
- `recipes/musdb18hq/models/online-crossattn-query-sfc2d.rt128k.fp512keep475.causal16dim.6l.64b/config.yaml`

`soft-band-query` vs `crossattn-query`:

- `soft-band-query` is still the safer deployment-first family.
- `crossattn-query` is the closer architecture-level rewrite of the paper's adaptive encoder/decoder path.
- `soft-band-query` keeps the explicit query side-path, but its encoder/decoder are still pooling/basis style operators rather than true cross-attention.
- `crossattn-query` replaces those with NPU-friendly cross-attention blocks implemented using `Conv2d + MatMul/bmm + Softmax`.
- In practice: use `soft-band-query` first when compiler/runtime robustness matters most, and use `crossattn-query` when the experiment goal is fidelity to the paper under the same online/NPU constraints.

Artifacts will be written in that model directory (checkpoints, merged config, logs), consistent with the other recipes.

---

## 2) Wav inference (PyTorch)

After training, you can run separation using the existing CLI (it loads `merged_config.yaml` and averages checkpoints):

```bash
python -m spectral_feature_compression.core.separate one \
  /path/to/model_directory_or_ckpt \
  /path/to/mix.wav \
  /path/to/out/est.wav \
  --device cuda \
  --css_segment_size 12 \
  --css_shift_size 6
```

Or for batch evaluation (MUSDB test set layout):

```bash
./recipes/musdb18hq/scripts/separate.sh /path/to/model_directory_or_ckpt 12 6
```

---

## 3) Export ONNX for edge NPU

### What gets exported

For NPU deployment we export **only** the core 2D model (no STFT/iSTFT).

There are now two export modes:
- **stateless**: `x -> y`
- **streaming/stateful**: `x, state_0, ..., state_N -> y, next_state_0, ..., next_state_N`

The stateful export is the preferred deployment path when you need strict
online equivalence and explicit cache handoff between inference calls.

### Export command

Pick fixed shapes for your device (many NPUs require static shapes). Example for:
- stereo input \(M=2\)
- 4 sources \(N=4\)
- `T=64` frames per call
- `F=1025` bins (for `n_fft=2048`)

```bash
./.venv/bin/python tools/online/export_onnx_online_model.py \
  /path/to/model_directory_or_ckpt \
  --out /path/to/online_sfc2d.onnx \
  --n-chan 2 \
  --frames 64 \
  --streaming \
  --state-meta-out /path/to/online_sfc2d.state.json \
  --externalize-band-constants \
  --disable-masking \
  --opset 11 \
  --check \
  --fail-on-disallowed-ops
```

Notes:
- `--streaming` exports the strict `forward_stream(...)` path with flattened state tensors.
- `--state-meta-out` writes the flattened state names and shapes for your runtime.
- `--freqs` is now optional; if omitted, the exporter uses `core.n_freq`. This is especially useful when recipe-side frequency preprocessing reduces `1025 -> 512`.
- selected band/basis priors are embedded in ONNX by default, just like other model weights / initializers.
- `--externalize-band-constants` is a fallback path that exposes selected band/basis tensors as graph inputs when your converter is sensitive to embedded static priors.
- `--disable-masking` keeps packed complex multiply outside the graph if your NPU compiler struggles with the masking subgraph.
- `--keep-initializers-as-inputs` is available as a fallback when a converter is unusually sensitive to embedded ONNX constants / buffers. It exposes weights as graph inputs too, so use it only when needed.
- `--constants-out` writes the externalized band/basis tensors to a `.npz` package for device-side persistent loading.
- `--deploy-manifest-out` writes a JSON deployment manifest that records whether band/basis priors were embedded or externalized, plus input/output names and memory estimates.
- the exporter now audits the ONNX ops after export and can fail if they are outside the selected allowlist preset.

### Export examples by feature

Stateless export:

```bash
./.venv/bin/python tools/online/export_onnx_online_model.py \
  /path/to/model_directory_or_ckpt \
  --out /tmp/online_soft_band_stateless.onnx \
  --n-chan 2 \
  --frames 64 \
  --opset 11 \
  --check
```

Export directly from a recipe `config.yaml` before training:

```bash
./.venv/bin/python tools/online/export_onnx_online_model.py \
  recipes/musdb18hq/models/online-soft-band-sfc2d.rt192k.causal24dim.6l.64b/config.yaml \
  --out /tmp/online_soft_band_from_config.onnx \
  --n-chan 2 \
  --frames 64 \
  --streaming \
  --state-meta-out /tmp/online_soft_band_from_config.state.json \
  --deploy-manifest-out /tmp/online_soft_band_from_config.deploy_manifest.json \
  --opset 11 \
  --check
```

This path instantiates the model directly from the recipe config and is useful
for NPU graph bring-up before any checkpoint exists. The manifest records this
as `source_mode: config_only_recipe`.

Streaming export with explicit state I/O:

```bash
./.venv/bin/python tools/online/export_onnx_online_model.py \
  /path/to/model_directory_or_ckpt \
  --out /tmp/online_soft_band_streaming.onnx \
  --n-chan 2 \
  --frames 64 \
  --streaming \
  --state-meta-out /tmp/online_soft_band_streaming.state.json \
  --opset 11 \
  --check
```

Streaming export with selected band/basis constants moved to graph inputs:

```bash
./.venv/bin/python tools/online/export_onnx_online_model.py \
  /path/to/model_directory_or_ckpt \
  --out /tmp/online_soft_band_streaming_extconst.onnx \
  --n-chan 2 \
  --frames 64 \
  --streaming \
  --state-meta-out /tmp/online_soft_band_streaming_extconst.state.json \
  --externalize-band-constants \
  --opset 11 \
  --check
```

Complete deployment package export:

```bash
./.venv/bin/python tools/online/export_onnx_online_model.py \
  /path/to/model_directory_or_ckpt \
  --out /tmp/online_soft_band_streaming.onnx \
  --n-chan 2 \
  --frames 64 \
  --streaming \
  --state-meta-out /tmp/online_soft_band_streaming.state.json \
  --externalize-band-constants \
  --constants-out /tmp/online_soft_band_streaming.constants.npz \
  --deploy-manifest-out /tmp/online_soft_band_streaming.deploy_manifest.json \
  --opset 11 \
  --check
```

Streaming export with masking kept outside the graph:

```bash
./.venv/bin/python tools/online/export_onnx_online_model.py \
  /path/to/model_directory_or_ckpt \
  --out /tmp/online_soft_band_nomask.onnx \
  --n-chan 2 \
  --frames 64 \
  --streaming \
  --state-meta-out /tmp/online_soft_band_nomask.state.json \
  --disable-masking \
  --opset 11 \
  --check
```

Streaming wav-directory inference through the real `forward_stream` path:

```bash
./.venv/bin/python tools/online/run_streaming_inference.py \
  /path/to/model_directory_or_ckpt \
  /path/to/test_wavs \
  /tmp/streaming_eval \
  --pattern mixture.wav \
  --chunk-frames 8 \
  --output-group test \
  --manifest-out /tmp/streaming_eval/run_manifest.json
```

This script:

- loads any current online model from checkpoint, trained directory, or recipe config
- runs true stateful chunked inference instead of the offline whole-waveform path
- writes ordered stem wavs per sample, e.g. `test/<song>/00_bass.wav`
- preserves a directory layout that can be fed directly into the existing SDR scripts

Example `MUSDB` metric run:

```bash
./recipes/musdb18hq/scripts/evaluate_sdr.py \
  --est_dir /tmp/streaming_eval \
  --ref_dir /path/to/musdb/test \
  --metric sisdr
```

Example `DnR` streaming inference:

```bash
./.venv/bin/python tools/online/run_streaming_inference.py \
  /path/to/dnr_model_directory_or_ckpt \
  /path/to/dnr_test_wavs \
  /tmp/dnr_streaming_eval \
  --pattern mix.wav \
  --chunk-frames 8 \
  --output-group test \
  --stem-name speech \
  --stem-name music \
  --stem-name sfx
```

If you want a one-command batch run that also calls the existing metric
scripts, use the generic wrapper:

```bash
./tools/online/run_streaming_eval.sh musdb /path/to/model_directory_or_ckpt 8 cuda
./tools/online/run_streaming_eval.sh dnr /path/to/model_directory_or_ckpt 8 cuda
```

The dataset-specific wrappers still exist and delegate to the same entrypoint:

```bash
./recipes/musdb18hq/scripts/streaming_eval.sh /path/to/model_directory_or_ckpt 8 cuda
./recipes/dnr/scripts/streaming_eval.sh /path/to/model_directory_or_ckpt 8 cuda
```

Last-resort export when the converter is sensitive to embedded initializers:

```bash
./.venv/bin/python tools/online/export_onnx_online_model.py \
  /path/to/model_directory_or_ckpt \
  --out /tmp/online_soft_band_keep_init_inputs.onnx \
  --n-chan 2 \
  --frames 64 \
  --freqs 1025 \
  --streaming \
  --state-meta-out /tmp/online_soft_band_keep_init_inputs.state.json \
  --keep-initializers-as-inputs \
  --opset 11 \
  --check
```

### Packing / unpacking for NPU runtime

Your runtime should:
- compute complex STFT externally: `(1, M, F, T)`
- pack real/imag into channels: `(1, 2*M, T, F)`:
  - `x[:, 0::2] = real`, `x[:, 1::2] = imag`
- run ONNX:
  - stateless export: input `x`, output `y`
  - streaming export: input `x + state tensors`, output `y + next_state tensors`
- unpack to complex and iSTFT externally.

The reference packing/unpacking functions are in:
- `spectral_feature_compression/core/model/online_sfc_2d.py`

---

## 4) Notes on "realtime"

`OnlineSFC2D` uses **causal Conv2d along time**, so it is compatible with streaming:
- output at a frame depends only on current/past frames (within the conv receptive field)
- you can run it chunk-by-chunk and keep a small left-context buffer

For strict NPUs you will typically export a fixed `T`.

Recommended deployment order:
- first choice: use the new stateful ONNX export and pass cache tensors between calls
- fallback only when the runtime cannot handle multi-input/multi-output state: use stateless export with an external overlap/ring-buffer policy

### Strict realtime notes

The online models now use:
- frame-local normalization over the frequency axis only
- `center=False` STFT/iSTFT in the online waveform wrapper
- no global input scaling in the strict realtime wrapper

The code now also validates the NPU span rule during model construction:

- time axis: `(kernel_t - 1) * dilation_t < 14`
- frequency axis: `(kernel_f - 1) * dilation_f < 14`

If a recipe violates this constraint, model construction now fails early with a
`ValueError` instead of silently creating an undeployable configuration.

For exact streaming equivalence, use the core-level `forward_stream(...)` API with
layer caches. The helper methods are available on:
- `OnlineSFC2D`
- `OnlineSoftBandSFC2D`
- `OnlineHardBandSFC2D`
- `OnlineHierarchicalSoftBandSFC2D`
- `OnlineHierarchicalSoftBandFFISFC2D`
- `OnlineHierarchicalSoftBandParallelFFISFC2D`
- `OnlineSoftBandDilatedSFC2D`
- `OnlineSoftBandGRUSFC2D`

The training-facing online builders now use `OnlineModelWrapper`, which rejects
global scaling because it leaks future information.

### ONNX op audit

The repo now includes:

```bash
./.venv/bin/python tools/online/audit_onnx_model.py /path/to/model.onnx
```

This reports:
- the unique ONNX ops present in the graph
- per-op counts
- initializer count / bytes
- any ops outside the selected allowlist preset
- optional deployment-memory totals if you also pass `--state-meta`

That check is deployment-oriented. `onnx.checker` still only tells you whether
the model is valid ONNX, not whether your target NPU compiler will accept it.

Example with a stateful export and a strict 192 KiB fp16 budget check:

```bash
./.venv/bin/python tools/online/audit_onnx_model.py \
  /tmp/online_soft_band_streaming_extconst.onnx \
  --state-meta /tmp/online_soft_band_streaming_extconst.state.json \
  --budget-kib 192 \
  --budget-dtype fp16 \
  --fail-on-disallowed-ops \
  --fail-on-budget
```

Example when you want to allow one extra op during bring-up:

```bash
./.venv/bin/python tools/online/audit_onnx_model.py \
  /tmp/online_soft_band_streaming_extconst.onnx \
  --state-meta /tmp/online_soft_band_streaming_extconst.state.json \
  --allow-op ScatterND
```

### Family overview

The current online / NPU-oriented families differ mainly in the separator and
state mechanism:

| family | separator | streaming state | ONNX op11 | NPU deployment risk | recommended use |
|---|---|---|---|---|---|
| `plain` | causal gated depthwise `Conv2d` stack | per-layer causal conv cache | yes | low | simplest baseline |
| `soft-band` | same conv separator on top of adaptive soft band routing | separator cache + compressor cache | yes | low | primary adaptive baseline |
| `soft-band-query` | adaptive soft band routing plus an explicit query side-path from compressor to decoder | separator cache + compressor cache | yes | medium | most SFC-like deployment-friendly baseline |
| `hard-band` | same conv separator on top of fixed band routing | mainly separator cache | yes | low | strongest control / ablation baseline |
| `hierarchical-soft-band` | front-end SFC compression followed by multi-scale interleaved temporal modeling | per-stage compressor cache + temporal block caches | yes | medium | multi-scale frequency/time candidate |
| `hierarchical-soft-band-ffi` | hierarchical soft-band backbone with explicit frequency-path -> frame-path interleaving | per-stage compressor cache + per-block time caches | yes | medium | TIGER-inspired online variant |
| `hierarchical-soft-band-parallel-ffi` | hierarchical soft-band backbone with parallel multi-receptive-field time branches | per-stage compressor cache + nested per-branch time caches | yes | medium | multi-branch temporal candidate under cache limits |
| `soft-band-dilated` | causal dilated time-mixing conv + explicit band-mix conv | per-layer cache sized by each layer's dilation | yes | medium | higher-context conv candidate |
| `soft-band-gru` | custom ConvGRU-style recurrent stack built from `Conv2d`, `sigmoid`, `tanh`, and elementwise ops | per-layer recurrent hidden state | yes | medium | recurrent long-context candidate |

Practical guidance:

- If you want the lowest deployment risk, start with `plain`, `soft-band`, or `hard-band`.
- If you want the closest approximation to the paper's encoder/decoder contract without leaving the NPU-friendly op set, prioritize `soft-band-query`.
- If you want to test whether multi-scale frequency compression plus per-scale temporal modeling beats a single-scale separator, prioritize `hierarchical-soft-band`.
- If you want the highest likely upside without leaving conv-style operators, prioritize `soft-band-dilated`.
- If you specifically want to test whether explicit recurrent memory beats cache-heavy conv context under the same budget, prioritize `soft-band-gru`.

### Cache budget check

You can inspect cache sizes for deployment with:

```bash
./.venv/bin/python tools/online/report_streaming_state_size.py --variant plain
./.venv/bin/python tools/online/report_streaming_state_size.py --variant soft
./.venv/bin/python tools/online/report_streaming_state_size.py --variant hard
```

The checker now reports four different deployment-relevant numbers:
- strict streaming `layer_cache`
- older `input_history`
- selected externalized `band/basis constants`
- `all model params + buffers`

This is intentional. The earlier 192 KiB workflow mainly tracked streaming
state. Your current deployment assumption is stricter: exported parameter payload
may also consume the same DSP budget, so the tool now prints both:
- `state + band/basis constants`
- `state + all model params + buffers`

Practical examples:

Check a default variant under the stricter fp16 budget:

```bash
./.venv/bin/python tools/online/report_streaming_state_size.py \
  --variant soft \
  --budget-kib 192 \
  --budget-dtype fp16
```

Check the query-side-path family under the same budget:

```bash
./.venv/bin/python tools/online/report_streaming_state_size.py \
  --variant soft_query \
  --d-model 24 \
  --n-layers 6 \
  --n-bands 64 \
  --budget-kib 192 \
  --budget-dtype fp16
```

Check the same family with frequency preprocessing enabled:

```bash
./.venv/bin/python tools/online/report_streaming_state_size.py \
  --variant soft_query \
  --d-model 24 \
  --n-layers 6 \
  --n-bands 64 \
  --n-freq 1025 \
  --freq-preprocess-enabled \
  --freq-preprocess-keep-bins 475 \
  --freq-preprocess-target-bins 512 \
  --budget-kib 192 \
  --budget-dtype fp16
```

Recommended first A/B for frequency preprocessing on MUSDB:

- baseline: `recipes/musdb18hq/models/online-soft-band-query-sfc2d.rt192k.causal24dim.6l.64b/config.yaml`
- stronger compression: `recipes/musdb18hq/models/online-soft-band-query-sfc2d.rt192k.fp512keep384.causal24dim.6l.64b/config.yaml`
- middle point: `recipes/musdb18hq/models/online-soft-band-query-sfc2d.rt192k.fp512keep475.causal24dim.6l.64b/config.yaml`
- more conservative compression: `recipes/musdb18hq/models/online-soft-band-query-sfc2d.rt192k.fp640keep475.causal24dim.6l.64b/config.yaml`

Recommended first A/B for DnR:

- `recipes/dnr/models/online-soft-band-query-sfc2d.rt192k.causal24dim.6l.64b/config.yaml`
- `recipes/dnr/models/online-soft-band-query-sfc2d.rt192k.fp512keep475.causal24dim.6l.64b/config.yaml`

Suggested execution order:

1. MUSDB baseline
   `recipes/musdb18hq/models/online-soft-band-query-sfc2d.rt192k.causal24dim.6l.64b/config.yaml`
2. MUSDB middle-point preprocessing
   `recipes/musdb18hq/models/online-soft-band-query-sfc2d.rt192k.fp512keep475.causal24dim.6l.64b/config.yaml`
3. MUSDB conservative preprocessing
   `recipes/musdb18hq/models/online-soft-band-query-sfc2d.rt192k.fp640keep475.causal24dim.6l.64b/config.yaml`
4. MUSDB aggressive preprocessing
   `recipes/musdb18hq/models/online-soft-band-query-sfc2d.rt192k.fp512keep384.causal24dim.6l.64b/config.yaml`
5. MUSDB non-query control
   `recipes/musdb18hq/models/online-soft-band-sfc2d.rt192k.fp512keep475.causal24dim.6l.64b/config.yaml`
6. DnR baseline
   `recipes/dnr/models/online-soft-band-query-sfc2d.rt192k.causal24dim.6l.64b/config.yaml`
7. DnR middle-point preprocessing
   `recipes/dnr/models/online-soft-band-query-sfc2d.rt192k.fp512keep475.causal24dim.6l.64b/config.yaml`

Interpretation:

- Step 1 -> 2 tells you whether the default `1025 -> 512` frontend is worth keeping.
- Step 2 -> 3 tells you whether the quality loss is mainly from squeezing too hard.
- Step 2 -> 4 tells you whether stronger compression buys enough deployment headroom to justify itself.
- Step 2 -> 5 separates the gain from frequency preprocessing and the gain from the query side-path.
- Step 6 -> 7 checks whether the same frontend choice transfers to DnR.

Fail fast in CI when a candidate exceeds the chosen budget:

```bash
./.venv/bin/python tools/online/report_streaming_state_size.py \
  --variant soft_dilated \
  --d-model 24 \
  --n-layers 6 \
  --n-bands 64 \
  --dilation-cycle 1 1 1 1 1 2 \
  --budget-kib 192 \
  --budget-dtype fp16 \
  --fail-on-budget
```

Inspect an FFI family candidate:

```bash
./.venv/bin/python tools/online/report_streaming_state_size.py \
  --variant hierarchical_soft_parallel_ffi \
  --d-model 20 \
  --pre-layers 0 \
  --mid-layers 1 \
  --bottleneck-layers 1 \
  --pre-bands 128 \
  --mid-bands 96 \
  --bottleneck-bands 48 \
  --time-branch-kernel-sizes 3 3 \
  --time-branch-dilations 1 6 \
  --budget-kib 192 \
  --budget-dtype fp16
```

For the default MUSDB online configs (`n_chan=2`, `n_fft=2048`, `n_bands=64`,
`d_model=96`, `n_layers=12`), all three variants exceed a 192 KiB cache budget
even in fp16, and the stricter `state + parameter payload` interpretation is
larger again. This means strict realtime deployment under that budget requires a
smaller architecture and/or more aggressive state or parameter compression
outside the current PyTorch reference implementation.

### Budget-constrained reference recipes

For quick experiments under realistic DSP state limits, the repository now
includes two pre-sized recipe sets for MUSDB18-HQ:

- `rt192k`: intended to keep the strict `forward_stream(...)` layer cache within
  a 192 KiB fp16 budget
- `rt128k`: a more conservative set with extra headroom

Recommended search ranges before you go lower-level or start quantizing caches:

- plain / soft-band:
  - `d_model`: `16, 24, 32`
  - `n_layers`: `4, 6, 8`
  - `n_bands`: `48, 64`
- hard-band:
  - `d_model`: `32, 48, 64`
  - `n_layers`: `4, 6, 8`
  - `n_bands`: `48, 64`

Validated `rt192k` recipes:

- `recipes/musdb18hq/models/online-sfc2d.rt192k.causal24dim.6l.64b/config.yaml`
- `recipes/musdb18hq/models/online-soft-band-sfc2d.rt192k.causal24dim.6l.64b/config.yaml`
- `recipes/musdb18hq/models/online-soft-band-sfc2d.rt192k.mel.causal24dim.6l.64b/config.yaml`
- `recipes/musdb18hq/models/online-soft-band-query-sfc2d.rt192k.causal24dim.6l.64b/config.yaml`
- `recipes/musdb18hq/models/online-soft-band-gru-sfc2d.rt192k.causal40dim.6l.64b/config.yaml`
- `recipes/musdb18hq/models/online-soft-band-gru-sfc2d.rt192k.mel.causal40dim.6l.64b/config.yaml`
- `recipes/musdb18hq/models/online-soft-band-dilated-sfc2d.rt192k.causal24dim.6l.64b/config.yaml`
- `recipes/musdb18hq/models/online-soft-band-dilated-sfc2d.rt192k.mel.causal24dim.6l.64b/config.yaml`
- `recipes/musdb18hq/models/online-soft-band-dilated-sfc2d.rt192k.maxdil.causal16dim.6l.64b/config.yaml`
- `recipes/musdb18hq/models/online-soft-band-dilated-sfc2d.rt192k.maxdil.mel.causal16dim.6l.64b/config.yaml`
- `recipes/musdb18hq/models/online-hierarchical-soft-band-sfc2d.rt192k.causal20dim.1-1-3l.128-96-48b/config.yaml`
- `recipes/musdb18hq/models/online-hierarchical-soft-band-sfc2d.rt192k.mel.causal20dim.1-1-3l.128-96-48b/config.yaml`
- `recipes/musdb18hq/models/online-hard-band-sfc2d.rt192k.causal48dim.8l.64b/config.yaml`
- `recipes/musdb18hq/models/online-hard-band-sfc2d.rt192k.mel.causal48dim.8l.64b/config.yaml`

Validated `rt128k` recipes:

- `recipes/musdb18hq/models/online-sfc2d.rt128k.causal16dim.6l.64b/config.yaml`
- `recipes/musdb18hq/models/online-soft-band-sfc2d.rt128k.causal16dim.6l.64b/config.yaml`
- `recipes/musdb18hq/models/online-soft-band-sfc2d.rt128k.mel.causal16dim.6l.64b/config.yaml`
- `recipes/musdb18hq/models/online-soft-band-query-sfc2d.rt128k.causal16dim.6l.64b/config.yaml`
- `recipes/musdb18hq/models/online-soft-band-gru-sfc2d.rt128k.causal24dim.10l.64b/config.yaml`
- `recipes/musdb18hq/models/online-soft-band-gru-sfc2d.rt128k.mel.causal24dim.10l.64b/config.yaml`
- `recipes/musdb18hq/models/online-soft-band-dilated-sfc2d.rt128k.causal16dim.6l.64b/config.yaml`
- `recipes/musdb18hq/models/online-soft-band-dilated-sfc2d.rt128k.mel.causal16dim.6l.64b/config.yaml`
- `recipes/musdb18hq/models/online-hierarchical-soft-band-sfc2d.rt128k.causal12dim.1-1-3l.128-96-48b/config.yaml`
- `recipes/musdb18hq/models/online-hierarchical-soft-band-sfc2d.rt128k.mel.causal12dim.1-1-3l.128-96-48b/config.yaml`
- `recipes/musdb18hq/models/online-hard-band-sfc2d.rt128k.causal32dim.8l.64b/config.yaml`
- `recipes/musdb18hq/models/online-hard-band-sfc2d.rt128k.mel.causal32dim.8l.64b/config.yaml`

At the time of validation, the strict fp16 layer-cache sizes were:

- `rt192k plain`: `168.09 KiB`
- `rt192k soft-band`: `168.09 KiB`
- `rt192k soft-band gru`: `190.16 KiB`
- `rt192k hierarchical-soft-band`: `185.08 KiB`
- `rt192k soft-band dilated`: `180.09 KiB`
- `rt192k soft-band dilated maxdil`: `184.06 KiB`
- `rt192k hard-band`: `192.00 KiB`
- `rt128k plain`: `112.06 KiB`
- `rt128k soft-band`: `112.06 KiB`
- `rt128k soft-band gru`: `126.09 KiB`
- `rt128k hierarchical-soft-band`: `111.05 KiB`
- `rt128k soft-band dilated`: `120.06 KiB`
- `rt128k hard-band`: `128.00 KiB`

### GRU separator candidates under budget

For the custom ConvGRU-style separator, the first budget-safe candidates are:

- `rt192k`
  - `d_model=40`, `n_layers=6`
  - strict fp16 layer-cache: `190.16 KiB`
  - recurrent state carries history, so explicit conv context remains small
- `rt128k`
  - `d_model=24`, `n_layers=10`
  - strict fp16 layer-cache: `126.09 KiB`
  - this is a useful conservative entry point if you want room for runtime SRAM overhead

Compared with the dilated separator family, the GRU family shifts the history
mechanism from explicit left-context caches to compact hidden states. That does
not guarantee better device latency, but it is the most direct way in this repo
to test whether recurrent memory is a better use of the same budget than wider
or more dilated conv stacks.

### Hierarchical separator candidates under budget

For the hierarchical soft-band family, the first budget-safe candidates are:

- `rt192k`
  - `d_model=20`
  - `layers=1/1/3`
  - `bands=128/96/48`
  - strict fp16 layer-cache: `185.08 KiB`
  - context: `24` frames
- `rt128k`
  - `d_model=12`
  - `layers=1/1/3`
  - `bands=128/96/48`
  - strict fp16 layer-cache: `111.05 KiB`
  - context: `24` frames

This family is useful when you want to test a different structural hypothesis
than the single-scale separators: first use an SFC-style inductive bias to map
the original STFT bins to `128` latent bands, then continue compressing and
doing temporal modeling at multiple latent frequency scales instead of only at
one final band resolution.

### How to interpret the memory numbers

There are now two useful budgeting interpretations in this repo:

- `layer_cache` only
- `layer_cache + exported parameter payload`

`layer_cache` is still the right first check for strict causality and streaming
correctness:

- use `forward_stream(...)`
- keep only the per-layer causal caches
- assume the cache is stored in `fp16`

This is the number that matches the non-streaming reference forward and is still
the most relevant measure when model weights live outside the DSP SRAM budget.

The stricter deployment interpretation is the one you requested for Edge bring-up:

- start from the strict `layer_cache`
- add selected externalized `band/basis constants`
- if your runtime shares the same DSP SRAM pool with the exported weights or buffers, also add the exported parameter payload

That is why the tools now print both:

- `state + band/basis constants`
- `state + all model params + buffers`

Do **not** size your DSP memory budget around the older raw-input-history /
recompute idea. That path was intentionally disabled because it did not preserve
exact equivalence to the reference model.

Even when the strict `layer_cache` fits, real deployment still needs room for:

- STFT / iSTFT ring buffers outside the NPU graph
- DMA staging / tensor ping-pong buffers
- framework or runtime bookkeeping
- output overlap-add buffers
- optional quantization scratch space

This is why the `rt128k` set can still be attractive even when your published
DSP limit is `192 KiB`: it leaves room for non-model SRAM consumers, and under
the stricter interpretation it may be the difference between barely fitting and
not fitting at all.

### Dilated separator candidates under 192 KiB

For the new soft-band dilated separator there are now two useful `192 KiB`
class choices:

- balanced
  - `d_model=24`, `n_layers=6`, `dilation_cycle=[1,1,1,1,1,2]`
  - strict fp16 layer-cache: `180.09 KiB`
  - context: `16` frames
- maxdil
  - `d_model=16`, `n_layers=6`, `dilation_cycle=[1,2,4,6,1,1]`
  - strict fp16 layer-cache: `184.06 KiB`
  - context: `32` frames

The current cache implementation already supports this without redesign:

- each separator layer keeps its own causal state tensor sized to that layer's
  actual dilation
- `forward_stream(...)` and `init_stream_state(...)` already use per-layer state
  tuples

So increasing dilation does increase cache size, but it does not require a new
state or cache mechanism.

### Recommended A/B for the dilated separator

For the new soft-band dilated family, the highest-value first comparison is not
against the plain soft-band baseline. It is:

- balanced vs maxdil under the same `192 KiB` deployment class

Run these two pairs first:

- `musical`
  - `recipes/musdb18hq/models/online-soft-band-dilated-sfc2d.rt192k.causal24dim.6l.64b/config.yaml`
  - `recipes/musdb18hq/models/online-soft-band-dilated-sfc2d.rt192k.maxdil.causal16dim.6l.64b/config.yaml`
- `mel`
  - `recipes/musdb18hq/models/online-soft-band-dilated-sfc2d.rt192k.mel.causal24dim.6l.64b/config.yaml`
  - `recipes/musdb18hq/models/online-soft-band-dilated-sfc2d.rt192k.maxdil.mel.causal16dim.6l.64b/config.yaml`

What this comparison tells you:

- whether wider channels (`24dim`) are more important than a much longer
  temporal context (`32` frames)
- whether the answer changes between `musical` and `mel` priors
- whether the stronger separator should be tuned by width first or by dilation
  first

After this A/B:

- if `balanced` wins clearly, keep width and only increase dilation more
  cautiously
- if `maxdil` wins clearly, prioritize temporal context in the next separator
  iterations
- if they are close, prefer `balanced` for easier optimization unless device
  latency strongly favors `maxdil`

### Recommended experiment order

If the goal is to pick a deployable model quickly without running the full
matrix blindly, this order gives the best signal per training run.

#### For a 192 KiB class target

1. `soft musical` vs `soft mel`
   - Compare:
     - `recipes/musdb18hq/models/online-soft-band-sfc2d.rt192k.causal24dim.6l.64b/config.yaml`
     - `recipes/musdb18hq/models/online-soft-band-sfc2d.rt192k.mel.causal24dim.6l.64b/config.yaml`
   - Goal: choose the better prior while keeping architecture fixed.
2. `soft best-prior` vs `hard same-budget family`
   - Compare the better soft-band winner against:
     - `recipes/musdb18hq/models/online-hard-band-sfc2d.rt192k.causal48dim.8l.64b/config.yaml`
     - or the `mel` hard-band variant if `mel` won in step 1.
   - Goal: measure whether input-adaptive routing is worth the extra complexity.
3. `soft best-prior` vs `plain`
   - Compare the best soft-band winner against:
     - `recipes/musdb18hq/models/online-sfc2d.rt192k.causal24dim.6l.64b/config.yaml`
   - Goal: check whether explicit band priors still help after adding strict
     streaming and cache constraints.

#### For a tighter 128 KiB class target

1. `soft musical` vs `soft mel`
   - Compare:
     - `recipes/musdb18hq/models/online-soft-band-sfc2d.rt128k.causal16dim.6l.64b/config.yaml`
     - `recipes/musdb18hq/models/online-soft-band-sfc2d.rt128k.mel.causal16dim.6l.64b/config.yaml`
2. `soft best-prior` vs `hard`
   - Compare the best soft-band winner against:
     - `recipes/musdb18hq/models/online-hard-band-sfc2d.rt128k.causal32dim.8l.64b/config.yaml`
     - or the `mel` hard-band variant if `mel` won.
3. `soft best-prior` vs `plain`
   - Compare the best soft-band winner against:
     - `recipes/musdb18hq/models/online-sfc2d.rt128k.causal16dim.6l.64b/config.yaml`

If training resources are limited, the soft-band prior comparison should usually
come first. It answers the highest-value question with the cleanest
parameter-matched setup.

### Practical decision rules

After you have a few finished runs, these heuristics are a reasonable way to
choose a deployment candidate:

- choose `soft-band` over `hard-band` only if the separation gain is consistent
  and worth the runtime / implementation overhead on your target stack
- prefer `rt128k` over `rt192k` when the quality gap is small, because SRAM head
  room is usually more valuable than a marginal metric gain during bring-up
- if `musical` and `mel` are close, prefer the one that gives stabler training
  and simpler downstream tuning on your dataset
- if two candidates are close in quality, prefer the one with the smaller
  context length and smaller cache footprint

### Result tracking template

Keep one row per finished run. A simple TSV / CSV / spreadsheet with the fields
below is enough to make the comparison reproducible.

Recommended columns:

- `recipe`
- `family`
- `prior`
- `budget_class`
- `d_model`
- `n_layers`
- `n_bands`
- `context_frames`
- `layer_cache_fp16_kib`
- `train_wallclock_per_epoch`
- `best_val_loss`
- `best_checkpoint`
- `separation_metric_main`
- `separation_metric_notes`
- `onnx_export_ok`
- `onnx_runtime_latency_ms`
- `streaming_equivalence_ok`
- `notes`

Example template:

```text
recipe,family,prior,budget_class,d_model,n_layers,n_bands,context_frames,layer_cache_fp16_kib,best_val_loss,best_checkpoint,separation_metric_main,onnx_export_ok,onnx_runtime_latency_ms,streaming_equivalence_ok,notes
online-soft-band-sfc2d.rt192k.causal24dim.6l.64b,soft,musical,rt192k,24,6,64,14,168.09,0.0000,/path/to/ckpt,0.0000,yes,0.0,yes,baseline soft-band candidate
```

A prefilled CSV starter file is available at:

- `docs/templates/online_budget_results.csv`

At minimum, do not skip these five fields:

- `recipe`
- `layer_cache_fp16_kib`
- `best_checkpoint`
- `separation_metric_main`
- `onnx_runtime_latency_ms`

Those are the fields that most often decide whether a model is actually
deployable.

### Suggested workflow

1. Use `tools/online/run_budget_sweep.sh --dry-run` to confirm the comparison
   set you want to launch.
2. Train the selected recipes.
3. Record the best checkpoint and validation numbers for each run.
4. Export the best candidates with `tools/online/export_onnx_online_model.py`.
5. Measure device-side latency with the final static input shape.
6. Only then decide between `rt128k` and `rt192k`, because offline metrics alone
   are not enough for an NPU deployment choice.
Frequency-preprocessed export example (`1025 -> 512`, keep the first `475` bins exactly and project the rest with a triangular high-band basis):

```bash
./.venv/bin/python tools/online/export_onnx_online_model.py \
  recipes/musdb18hq/models/online-soft-band-query-sfc2d.rt192k.fp512keep475.causal24dim.6l.64b/config.yaml \
  --out /tmp/online_soft_band_query_fp512.onnx \
  --n-chan 2 \
  --frames 64 \
  --streaming \
  --state-meta-out /tmp/online_soft_band_query_fp512.state.json \
  --deploy-manifest-out /tmp/online_soft_band_query_fp512.deploy_manifest.json \
  --opset 11 \
  --check
```

The resulting deploy manifest records:
- `frequency_preprocessing.enabled`
- `frequency_preprocessing.keep_bins`
- `frequency_preprocessing.target_bins`
- `frequency_preprocessing.mode`
- `frequency_preprocessing.full_n_freq`

so the device-side frontend knows it must run `1025 -> 512` before the online core and `512 -> 1025` afterwards.
