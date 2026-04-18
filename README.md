<!--
Copyright (c) 2026 National Institute of Advanced Industrial Science and Technology (AIST), Japan

SPDX-License-Identifier: MIT
-->

# Spectral Feature Compression for Source Separation

[![IEEE DOI](https://img.shields.io/badge/IEEE/DOI-10.1109/TASLPRO.2026.3663929-blue.svg)](https://doi.org/10.1109/TASLPRO.2026.3663929)
[![arXiv](https://img.shields.io/badge/arXiv-2602.08671-b31b1b.svg)](https://arxiv.org/abs/2602.08671)

This repository includes source code for the following paper:

```
@article{saijo2026input,
  title={Input-Adaptive Spectral Feature Compression by Sequence Modeling for Source Separation},
  author={Saijo, Kohei and Bando, Yoshiaki},
  journal={IEEE Transactions on Audio, Speech and Language Processing},
  year={2026},
  publisher={IEEE}
}
```

## 1. Installation

```sh
# install this repo
git clone https://github.com/b-sigpro/spectral-feature-compression.git
cd spectral-feature-compression
pip install -e .
pip install -r requirements.txt

# install Mamba, see https://github.com/state-spaces/mamba for more details
pip install --no-build-isolation mamba-ssm[causal-conv1d]

# install aiaccel
git clone https://github.com/aistairc/aiaccel.git
cd aiaccel
git checkout 117d8d5d335540b6d331993ffc02d4b64f5e02a1  # the commit where we tested our code
pip install -e .
cd ../
```

## 2. How to use the Spectral Feature Compression module

The repository supports three modules:
- `CrossAttnEncoder, CrossAttnDecoder`: Spectral Feature Compression by Cross-Attention (SFC-CA; Section III-B in the paper)
- `MambaEncoder, MambaDecoder`: Spectral Feature Compression by Mamba (SFC-CA; Section III-C in the paper)
- `BanditEncoder, BanditDecoder`: Band-split module from the [BandIt paper](https://github.com/kwatcharasupat/bandit)

Once the installation is done, one can use the these modules easily by importing them. The code below is an example to use SFC-CA:
```python
import torch
from spectral_feature_compression import CrossAttnEncoder, CrossAttnDecoder

sample_rate = 44100
n_fft = 2048
n_batch, n_chan, n_frames, n_freqs = 4, 2, 100, n_fft//2+1
n_src = 4

encoder = CrossAttnEncoder(d_inner=64, d_model=128, n_chan=n_chan, sample_rate=sample_rate, n_fft=n_fft, n_bands=64).to("cuda")
decoder = CrossAttnDecoder(d_inner=64, d_model=128, n_src=n_src, n_chan=n_chan, sample_rate=sample_rate, n_fft=n_fft, n_bands=64).to("cuda")


# the modules assume a complex input of (n_batch, n_chan, n_frames, n_freqs) or float input of (n_batch, 2*n_chan, n_frames, n_freqs)
input = torch.randn((n_batch, n_chan, n_frames, n_freqs), dtype=torch.complex64, device="cuda")
enc_output, dec_query = encoder(input) # enc_output: (n_batch, d_model, n_frames, n_bands)
dec_output, _ = decoder(enc_output, query=dec_query) # dec_output: (n_batch, 2*n_chan*n_src, n_frames, n_freqs)
```
Note that the SFC encoder returns two tensors; the first one is the compressed output and the second one is the non-compressed tensor. The second tensor will be used as the query in the decoder when using the `adaptive` query, while it's just ignored when using the `learnable` query (please refer to the paper for more details).

The repository supports the TF-Locoformer separator, which can be used as
```python
from spectral_feature_compression import BSLocoformer

# assuming the encoder and decoder are initialized, as shown above
# the default parameters are for the small-sized model
separator = BSLocoformer(encoder=encoder, decoder=decoder, n_src=4, n_chan=2).to("cuda")

# the modules assume a complex input of (n_batch, n_chan, n_freqs, n_frames)
input = torch.randn((n_batch, n_chan, n_freqs, n_frames), dtype=torch.complex64, device="cuda")
output = separator(input) # (n_batch, n_src, n_chan, n_freqs, n_frames)
```

## 3. Pre-trained models

We provide some pre-trained weights of the TF-Locoformer model trained on the MUSDB18HQ or DnR dataset at a [Hugging Face repository](https://huggingface.co/kohei0209/spectral_feature_compression).
```sh
python model_weights/download_pretrained_weights.py --dst_dir ./model_weights
```
By default, it makes the `model_weights` directory and download models under it.

Once the models are downloaded, one can use them to separate sources by `separate_sample.py`. An example to run it is shown below:
```sh
python separate_sample.py model_weights/musdb18hq/locoformer-small.enc-crossattn64dim.dec-crossattn64dim.musical64.learnable-query /path/to/audio-file /path/to/output-directory
```


## 4. Training a separation model
Assume you are now at `./spectral-feature-compression`.

### Data preparation

We provide a shell script, `data.sh`, to easily prepare the data. `data.sh` does the following processes:

1. Download the MUSDB18-HQ or the DnR dataset and uncompress it
2. Only on MUSDB18-HQ: Split training and validation set following the common split
3. Apply unsupervised source activity detection (introduced in the BSRNN paper) to the training data and save the segmented audio files as an HDF5 file

It can be run as:
```sh
dataset_name=musdb18hq # or dnr
./recipes/${dataset_name}/scripts/data.sh
```


### Training

Training can be run by running `train.sh` at each directory:
```sh
./recipes/musdb18hq/models/locoformer-small.enc-crossattn64dim.dec-crossattn64dim.musical64.learnable-query/train.sh
```

The directory strucure after training is as follows. The lightning's checkpoints are saved under `checkpoints`. The training progress can be watched with Tensorboard.
```sh
recipes/musdb18hq/models/locoformer-small.enc-crossattn64dim.dec-crossattn64dim.musical64.learnable-query
├── checkpoints
│   ├── epoch=xxxx.ckpt
│   ├── epoch=xxxx.ckpt
│   ├── epoch=xxxx.ckpt
│   ├── epoch=xxxx.ckpt
│   ├── epoch=xxxx.ckpt
│   └── last.ckpt
├── config.yaml
├── events.out.tfevents.xxx.xxx.xxx.x
├── hparams.yaml
├── log.txt
├── merged_config.yaml
└── train.sh
```


### Evaluation

Once you finish the training, running `separate.sh` runs inference and scoring:
```sh
./recipes/musdb18hq/scripts/separate.sh /path/to/model_directory
```
Here, `/path/to/model_directory` can be either a path to directory including the `checkpoints` directory or a direct path to `.ckpt` file.
For instance, when evaluating `recipes/musdb18hq/models/locoformer-small.enc-crossattn64dim.dec-crossattn64dim.musical64.learnable-query`, you can give `recipes/musdb18hq/models/locoformer-small.enc-crossattn64dim.dec-crossattn64dim.musical64.learnable-query` or `recipes/musdb18hq/models/locoformer-small.enc-crossattn64dim.dec-crossattn64dim.musical64.learnable-query/checkpoints/xxx.ckpt`.
In the former case, all the checkpoints under that directory except for `last.ckpt` are averaged, and the averaged parameters are used for evaluation.

The segment and shift size in inference are by default set to 12 and 6 seconds, respectively.
One can change these configurations by giving them as the second and third arguments when running `separate.sh`.



## 5. Online / realtime model for edge NPU

This repo also includes an **online/realtime, 2D-only** refactor intended for **edge NPU** deployment (ONNX export with `batch_size=1`, tensors \(\le\) 4D, `Conv2d` / `torch.bmm` plus basic tensor ops).

See `docs/ONLINE_SFC_NPU.md`.

All current online kernels are kept within the deploy-time NPU convolution span
constraint:

- `(kernel_size - 1) * dilation < 14`

### Online model variants

For quick A/B comparisons, the repository now includes several online 2D-only families under `recipes/musdb18hq/models/`:

- `online-sfc2d.causal96dim.12l`: compressed-band online separator with learned spectral compression and decoding
- `online-soft-band-sfc2d.causal96dim.12l`: soft band-routing variant with input-adaptive routing weights and a static band prior
- `online-soft-band-query-sfc2d.causal96dim.12l`: soft band-routing variant with an explicit query side-path from compressor to decoder
- `online-crossattn-query-sfc2d.causal96dim.12l`: NPU-friendly cross-attention encoder/decoder variant that keeps the paper's encoder-side embedding -> decoder query contract
- `online-hierarchical-soft-band-sfc2d.causal96dim.1-2-2l`: SFC front-end compression to 128 bands followed by hierarchical interleaved frequency compression and temporal modeling
- `online-soft-band-gru-sfc2d.causal96dim.12l`: soft band-routing variant with a custom ConvGRU-style separator built only from Conv2d and elementwise ops
- `online-soft-band-dilated-sfc2d.causal96dim.12l`: stronger soft-band variant with a dilated time-mixing separator and explicit band-axis mixing
- `online-hard-band-sfc2d.causal96dim.12l`: hard/static band baseline with the same separator stack but no input-adaptive routing

Quick family view:

| family | separator | streaming state | deployment risk | best first use |
|---|---|---|---|---|
| `plain` | causal gated depthwise `Conv2d` stack | per-layer causal conv cache | low | simplest online baseline |
| `soft-band` | same conv separator with adaptive soft band routing | separator cache + compressor cache | low | main adaptive baseline |
| `soft-band-query` | adaptive soft band routing with an explicit query side-path into the decoder | separator cache + compressor cache | medium | closest deployment-friendly approximation to the paper's encoder/decoder contract |
| `crossattn-query` | NPU-friendly encoder/decoder cross-attention with conv separator in the latent K domain | separator cache + encoder cache | medium | closest architecture-level approximation to the paper's adaptive encoder/decoder path |
| `hard-band` | same conv separator with fixed band routing | mainly separator cache | low | strongest control baseline |
| `hierarchical-soft-band` | front-end SFC compression followed by multi-scale interleaved temporal modeling | per-stage compressor cache + temporal block caches | medium | multi-scale frequency/time candidate |
| `hierarchical-soft-band-ffi` | hierarchical soft-band backbone with explicit frequency-path -> frame-path interleaving | per-stage compressor cache + per-block time caches | medium | TIGER-inspired online variant |
| `hierarchical-soft-band-parallel-ffi` | hierarchical soft-band backbone with explicit frequency-path plus parallel multi-receptive-field time branches | per-stage compressor cache + per-branch time caches | medium | multi-branch temporal candidate under cache limits |
| `soft-band-dilated` | dilated causal conv separator with explicit band-mix conv | per-layer cache scaled by dilation | medium | larger-context conv candidate |
| `soft-band-gru` | custom ConvGRU-style recurrent separator | per-layer recurrent hidden state | medium | recurrent long-context candidate |

Use `soft-band-dilated` when you want the highest likely upside while staying in
conv-style operators. Use `soft-band-gru` when you specifically want to test
whether explicit recurrent memory beats cache-heavier conv context under the
same budget. The longer deployment-focused comparison is documented in
`docs/ONLINE_SFC_NPU.md`.

Several of the band-aware families have `musical` and `mel` prior variants:

- `recipes/musdb18hq/models/online-soft-band-sfc2d.causal96dim.12l/config.yaml`
- `recipes/musdb18hq/models/online-hierarchical-soft-band-sfc2d.causal96dim.1-2-2l/config.yaml`
- `recipes/musdb18hq/models/online-soft-band-gru-sfc2d.causal96dim.12l/config.yaml`
- `recipes/musdb18hq/models/online-soft-band-dilated-sfc2d.causal96dim.12l/config.yaml`
- `recipes/musdb18hq/models/online-soft-band-sfc2d.mel.causal96dim.12l/config.yaml`
- `recipes/musdb18hq/models/online-hard-band-sfc2d.causal96dim.12l/config.yaml`
- `recipes/musdb18hq/models/online-hard-band-sfc2d.mel.causal96dim.12l/config.yaml`

This gives a simple comparison ladder:

- plain online compressed model vs soft band prior
- soft band prior vs hard/static band prior
- `musical` prior vs `mel` prior

Quick commands for the new `soft-band-query` family:

```sh
./recipes/musdb18hq/models/online-soft-band-query-sfc2d.causal96dim.12l/train.sh
./.venv/bin/python tools/online/report_streaming_state_size.py --variant soft_query --d-model 24 --n-layers 6 --n-bands 64 --budget-kib 192 --budget-dtype fp16
./.venv/bin/python tools/online/export_onnx_online_model.py \
  recipes/musdb18hq/models/online-soft-band-query-sfc2d.rt192k.causal24dim.6l.64b/config.yaml \
  --out /tmp/soft_query_rt192k.onnx \
  --n-chan 2 \
  --frames 64 \
  --streaming \
  --state-meta-out /tmp/soft_query_rt192k.state.json \
  --deploy-manifest-out /tmp/soft_query_rt192k.deploy_manifest.json \
  --check
```

Quick commands for the new `crossattn-query` family:

```sh
./recipes/musdb18hq/models/online-crossattn-query-sfc2d.causal96dim.12l/train.sh
./.venv/bin/python tools/online/report_streaming_state_size.py --variant crossattn_query --d-model 24 --n-layers 6 --n-bands 64 --budget-kib 192 --budget-dtype fp16
./.venv/bin/python tools/online/export_onnx_online_model.py \
  recipes/musdb18hq/models/online-crossattn-query-sfc2d.rt192k.causal24dim.6l.64b/config.yaml \
  --out /tmp/crossattn_query_rt192k.onnx \
  --n-chan 2 \
  --frames 64 \
  --streaming \
  --state-meta-out /tmp/crossattn_query_rt192k.state.json \
  --deploy-manifest-out /tmp/crossattn_query_rt192k.deploy_manifest.json \
  --check
```

New `crossattn-query` deployment-oriented recipes now included:

- `recipes/musdb18hq/models/online-crossattn-query-sfc2d.rt192k.causal24dim.6l.64b/config.yaml`
- `recipes/musdb18hq/models/online-crossattn-query-sfc2d.rt192k.fp512keep475.causal24dim.6l.64b/config.yaml`
- `recipes/musdb18hq/models/online-crossattn-query-sfc2d.rt128k.causal16dim.6l.64b/config.yaml`
- `recipes/musdb18hq/models/online-crossattn-query-sfc2d.rt128k.fp512keep475.causal16dim.6l.64b/config.yaml`

`soft-band-query` vs `crossattn-query`:

- `soft-band-query` is the better deployment-first baseline. It is simpler, lighter, and the safer family when NPU compile/runtime risk matters most.
- `crossattn-query` is more faithful to the paper. Its encoder really does adaptive `F -> K` cross-attention and its decoder really does adaptive `K -> F` cross-attention using the encoder side embedding.
- `soft-band-query` preserves the paper's spirit with band-aware pooling plus an explicit query side-path, but it does not preserve the original encoder/decoder operator pattern.
- `crossattn-query` preserves that operator pattern much better, but pays for it with more initializers and a heavier ONNX graph.
- If the goal is “closest to the paper under the NPU rules”, choose `crossattn-query`.
- If the goal is “highest confidence edge deployment candidate”, start from `soft-band-query`.

Quick commands for the new frequency-preprocessed online variants (`1025 -> 512`, keep first `475` bins, triangular high-band projection):

```sh
./recipes/musdb18hq/models/online-soft-band-query-sfc2d.rt192k.fp512keep475.causal24dim.6l.64b/train.sh
./.venv/bin/python tools/online/report_streaming_state_size.py \
  --variant soft_query \
  --d-model 24 \
  --n-layers 6 \
  --n-bands 64 \
  --n-freq 1025 \
  --freq-preprocess-enabled \
  --freq-preprocess-keep-bins 475 \
  --freq-preprocess-target-bins 512
./.venv/bin/python tools/online/export_onnx_online_model.py \
  recipes/musdb18hq/models/online-soft-band-query-sfc2d.rt192k.fp512keep475.causal24dim.6l.64b/config.yaml \
  --out /tmp/soft_query_rt192k_fp512.onnx \
  --n-chan 2 \
  --frames 64 \
  --streaming \
  --state-meta-out /tmp/soft_query_rt192k_fp512.state.json \
  --deploy-manifest-out /tmp/soft_query_rt192k_fp512.deploy_manifest.json \
  --check
```

Streaming inference from a wav directory, with ordered stem wavs written in a
metrics-friendly layout:

```sh
./.venv/bin/python tools/online/run_streaming_inference.py \
  /path/to/model_directory_or_ckpt \
  /path/to/test_wavs \
  /tmp/streaming_eval \
  --pattern mixture.wav \
  --chunk-frames 8 \
  --output-group test
```

For `MUSDB`, this writes directories like
`/tmp/streaming_eval/test/<song>/00_bass.wav`, `01_drums.wav`,
`02_vocals.wav`, `03_other.wav`, which can be listened to directly and are also
compatible with the existing SDR evaluation scripts:

```sh
./recipes/musdb18hq/scripts/evaluate_sdr.py \
  --est_dir /tmp/streaming_eval \
  --ref_dir /path/to/musdb/test \
  --metric sisdr
```

For `DnR`, override the stem names so the written order matches
`speech / music / sfx`:

```sh
./.venv/bin/python tools/online/run_streaming_inference.py \
  /path/to/model_directory_or_ckpt \
  /path/to/dnr_test_wavs \
  /tmp/dnr_streaming_eval \
  --pattern mix.wav \
  --chunk-frames 8 \
  --output-group test \
  --stem-name speech \
  --stem-name music \
  --stem-name sfx
```

There is also a unified batch shell wrapper that runs streaming inference and
then calls the existing SDR scripts directly:

```sh
./tools/online/run_streaming_eval.sh musdb /path/to/model_directory_or_ckpt 8 cuda
./tools/online/run_streaming_eval.sh dnr /path/to/model_directory_or_ckpt 8 cuda
```

The dataset-specific wrappers are still available and now delegate to the same
generic entrypoint:

```sh
./recipes/musdb18hq/scripts/streaming_eval.sh /path/to/model_directory_or_ckpt 8 cuda
./recipes/dnr/scripts/streaming_eval.sh /path/to/model_directory_or_ckpt 8 cuda
```

### DnR online variants

The same online families can also be trained on `DnR` (`speech / music / sfx`,
`n_src=3`, `n_chan=1`). The repository now includes these `DnR` recipes:

- `recipes/dnr/models/online-sfc2d.causal96dim.12l.musical64/config.yaml`
- `recipes/dnr/models/online-soft-band-sfc2d.causal96dim.12l.musical64/config.yaml`
- `recipes/dnr/models/online-soft-band-sfc2d.mel.causal96dim.12l/config.yaml`
- `recipes/dnr/models/online-soft-band-sfc2d.rt192k.fp512keep475.causal24dim.6l.64b/config.yaml`
- `recipes/dnr/models/online-soft-band-sfc2d.rt128k.fp512keep475.causal16dim.6l.64b/config.yaml`
- `recipes/dnr/models/online-soft-band-query-sfc2d.causal96dim.12l.musical64/config.yaml`
- `recipes/dnr/models/online-soft-band-query-sfc2d.mel.causal96dim.12l/config.yaml`
- `recipes/dnr/models/online-soft-band-query-sfc2d.rt192k.causal24dim.6l.64b/config.yaml`
- `recipes/dnr/models/online-soft-band-query-sfc2d.rt128k.causal16dim.6l.64b/config.yaml`
- `recipes/dnr/models/online-soft-band-query-sfc2d.rt192k.fp512keep475.causal24dim.6l.64b/config.yaml`
- `recipes/dnr/models/online-soft-band-query-sfc2d.rt128k.fp512keep475.causal16dim.6l.64b/config.yaml`
- `recipes/dnr/models/online-soft-band-dilated-sfc2d.causal96dim.12l.musical64/config.yaml`
- `recipes/dnr/models/online-soft-band-dilated-sfc2d.mel.causal96dim.12l/config.yaml`
- `recipes/dnr/models/online-soft-band-gru-sfc2d.causal96dim.12l.musical64/config.yaml`
- `recipes/dnr/models/online-soft-band-gru-sfc2d.mel.causal96dim.12l/config.yaml`
- `recipes/dnr/models/online-hierarchical-soft-band-sfc2d.causal96dim.1-2-2l.musical128/config.yaml`
- `recipes/dnr/models/online-hierarchical-soft-band-sfc2d.mel.causal96dim.1-2-2l/config.yaml`
- `recipes/dnr/models/online-hierarchical-soft-band-ffi-sfc2d.speech-lowfreq-narrow.causal96dim.1-2-2l/config.yaml`
- `recipes/dnr/models/online-hierarchical-soft-band-parallel-ffi-sfc2d.rt192k.speech-lowfreq-narrow.causal20dim.0-1-1l.128-96-48b/config.yaml`
- `recipes/dnr/models/online-hierarchical-soft-band-parallel-ffi-sfc2d.rt128k.speech-lowfreq-narrow.causal14dim.0-1-1l.128-96-48b/config.yaml`
- `recipes/dnr/models/online-hard-band-sfc2d.causal96dim.12l.musical64/config.yaml`
- `recipes/dnr/models/online-hard-band-sfc2d.mel.causal96dim.12l/config.yaml`

For `DnR`, the main band split priors are `musical`, `mel`, and the new
`speech_lowfreq_narrow`.
`musical` is already used by the existing `DnR` offline recipes and the
pre-existing `DnR` online baseline, so it is a valid starting point rather than
an obvious mismatch. `mel` is the most natural comparison if you want to test a
prior that may be more speech-friendly. `speech_lowfreq_narrow` is a
TIGER-inspired low-frequency-dense prior that allocates narrower bands below
roughly 1-2 kHz and wider bands in higher frequencies.

If you want a quick first comparison on `DnR`, the highest-value order is:

- `soft-band musical` vs `soft-band mel`
- `soft-band musical` vs `hard-band musical`
- `soft-band musical` vs `hierarchical musical`
- `hierarchical musical` vs `hierarchical-soft-band-ffi speech_lowfreq_narrow`

For the focused hierarchical DnR comparison, use:

- `recipes/dnr/models/online-hierarchical-soft-band-sfc2d.causal96dim.1-2-2l.musical128/config.yaml`
- `recipes/dnr/models/online-hierarchical-soft-band-ffi-sfc2d.speech-lowfreq-narrow.causal96dim.1-2-2l/config.yaml`
- `recipes/dnr/models/online-hierarchical-soft-band-parallel-ffi-sfc2d.rt192k.speech-lowfreq-narrow.causal20dim.0-1-1l.128-96-48b/config.yaml`

There is also a dedicated launcher for this 3-way comparison:

```bash
./tools/online/run_dnr_hierarchical_tiger_ab.sh --dry-run
./tools/online/run_dnr_hierarchical_tiger_ab.sh
```

For the parallel multi-receptive-field variant, the repository now includes:

- `rt192k`: `recipes/dnr/models/online-hierarchical-soft-band-parallel-ffi-sfc2d.rt192k.speech-lowfreq-narrow.causal20dim.0-1-1l.128-96-48b/config.yaml`
- `rt128k`: `recipes/dnr/models/online-hierarchical-soft-band-parallel-ffi-sfc2d.rt128k.speech-lowfreq-narrow.causal14dim.0-1-1l.128-96-48b/config.yaml`

You can select it from the DnR sweep with:

```bash
./tools/online/run_dnr_online_sweep.sh hierarchical-soft-parallel-ffi --dry-run
./tools/online/run_dnr_online_sweep.sh hierarchical-soft-parallel-ffi
```

You can run a `DnR`-only online sweep with:

```sh
./tools/online/run_dnr_online_sweep.sh
./tools/online/run_dnr_online_sweep.sh soft --dry-run
./tools/online/run_dnr_online_sweep.sh soft-query --dry-run
./tools/online/run_dnr_online_sweep.sh hierarchical-soft
```

Quick `DnR soft-band-query` commands:

```sh
./recipes/dnr/models/online-soft-band-query-sfc2d.causal96dim.12l.musical64/train.sh
./tools/online/run_dnr_online_sweep.sh soft-query --dry-run
./.venv/bin/python tools/online/export_onnx_online_model.py \
  recipes/dnr/models/online-soft-band-query-sfc2d.rt192k.causal24dim.6l.64b/config.yaml \
  --out /tmp/dnr_soft_query_rt192k.onnx \
  --n-chan 1 \
  --frames 64 \
  --streaming \
  --state-meta-out /tmp/dnr_soft_query_rt192k.state.json \
  --deploy-manifest-out /tmp/dnr_soft_query_rt192k.deploy_manifest.json
```

Quick `DnR soft-band-query` commands with frequency preprocessing:

```sh
./recipes/dnr/models/online-soft-band-query-sfc2d.rt192k.fp512keep475.causal24dim.6l.64b/train.sh
./tools/online/run_dnr_online_sweep.sh soft-query --dry-run
./.venv/bin/python tools/online/export_onnx_online_model.py \
  recipes/dnr/models/online-soft-band-query-sfc2d.rt192k.fp512keep475.causal24dim.6l.64b/config.yaml \
  --out /tmp/dnr_soft_query_rt192k_fp512.onnx \
  --n-chan 1 \
  --frames 64 \
  --streaming \
  --state-meta-out /tmp/dnr_soft_query_rt192k_fp512.state.json \
  --deploy-manifest-out /tmp/dnr_soft_query_rt192k_fp512.deploy_manifest.json
```

For strict realtime cache-constrained deployment, the repository also includes
pre-sized MUSDB recipes that keep the fp16 layer cache within typical DSP
budgets:

- `rt192k`
  - `recipes/musdb18hq/models/online-sfc2d.rt192k.causal24dim.6l.64b/config.yaml`
  - `recipes/musdb18hq/models/online-soft-band-sfc2d.rt192k.causal24dim.6l.64b/config.yaml`
  - `recipes/musdb18hq/models/online-soft-band-sfc2d.rt192k.fp512keep475.causal24dim.6l.64b/config.yaml`
  - `recipes/musdb18hq/models/online-soft-band-sfc2d.rt192k.mel.causal24dim.6l.64b/config.yaml`
  - `recipes/musdb18hq/models/online-soft-band-query-sfc2d.rt192k.causal24dim.6l.64b/config.yaml`
  - `recipes/musdb18hq/models/online-soft-band-query-sfc2d.rt192k.fp512keep475.causal24dim.6l.64b/config.yaml`
  - `recipes/musdb18hq/models/online-soft-band-query-sfc2d.rt192k.fp512keep384.causal24dim.6l.64b/config.yaml`
  - `recipes/musdb18hq/models/online-soft-band-query-sfc2d.rt192k.fp640keep475.causal24dim.6l.64b/config.yaml`
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
- `rt128k`
  - `recipes/musdb18hq/models/online-sfc2d.rt128k.causal16dim.6l.64b/config.yaml`
  - `recipes/musdb18hq/models/online-soft-band-sfc2d.rt128k.causal16dim.6l.64b/config.yaml`
  - `recipes/musdb18hq/models/online-soft-band-sfc2d.rt128k.fp512keep475.causal16dim.6l.64b/config.yaml`
  - `recipes/musdb18hq/models/online-soft-band-sfc2d.rt128k.mel.causal16dim.6l.64b/config.yaml`
  - `recipes/musdb18hq/models/online-soft-band-query-sfc2d.rt128k.causal16dim.6l.64b/config.yaml`
  - `recipes/musdb18hq/models/online-soft-band-query-sfc2d.rt128k.fp512keep475.causal16dim.6l.64b/config.yaml`
  - `recipes/musdb18hq/models/online-soft-band-gru-sfc2d.rt128k.causal24dim.10l.64b/config.yaml`
  - `recipes/musdb18hq/models/online-soft-band-gru-sfc2d.rt128k.mel.causal24dim.10l.64b/config.yaml`
  - `recipes/musdb18hq/models/online-soft-band-dilated-sfc2d.rt128k.causal16dim.6l.64b/config.yaml`
  - `recipes/musdb18hq/models/online-soft-band-dilated-sfc2d.rt128k.mel.causal16dim.6l.64b/config.yaml`
  - `recipes/musdb18hq/models/online-hierarchical-soft-band-sfc2d.rt128k.causal12dim.1-1-3l.128-96-48b/config.yaml`
  - `recipes/musdb18hq/models/online-hierarchical-soft-band-sfc2d.rt128k.mel.causal12dim.1-1-3l.128-96-48b/config.yaml`
  - `recipes/musdb18hq/models/online-hard-band-sfc2d.rt128k.causal32dim.8l.64b/config.yaml`
  - `recipes/musdb18hq/models/online-hard-band-sfc2d.rt128k.mel.causal32dim.8l.64b/config.yaml`

The `rt192k` set is a good first pass for a 192 KiB cache budget. The `rt128k`
set leaves more margin and is a safer starting point when runtime bookkeeping,
DMA staging, or framework overhead also consume SRAM. See
`docs/ONLINE_SFC_NPU.md` for the validation notes and recommended search ranges.

For a focused frequency-preprocessing A/B on MUSDB, the most useful three-way
comparison is:

- baseline: `online-soft-band-query-sfc2d.rt192k.causal24dim.6l.64b`
- middle point: `online-soft-band-query-sfc2d.rt192k.fp512keep475.causal24dim.6l.64b`
- stronger compression: `online-soft-band-query-sfc2d.rt192k.fp512keep384.causal24dim.6l.64b`
- more conservative compression: `online-soft-band-query-sfc2d.rt192k.fp640keep475.causal24dim.6l.64b`

Recommended execution order if you want the highest signal with the fewest runs:

1. `online-soft-band-query-sfc2d.rt192k.causal24dim.6l.64b`
   Purpose: establish the no-preprocessing baseline.
2. `online-soft-band-query-sfc2d.rt192k.fp512keep475.causal24dim.6l.64b`
   Purpose: test the default `1025 -> 512` frontend with a moderate high-band squeeze.
3. `online-soft-band-query-sfc2d.rt192k.fp640keep475.causal24dim.6l.64b`
   Purpose: check whether backing off to `640` bins recovers high-frequency quality.
4. `online-soft-band-query-sfc2d.rt192k.fp512keep384.causal24dim.6l.64b`
   Purpose: test the more aggressive setting only after you know the middle point is viable.
5. `online-soft-band-sfc2d.rt192k.fp512keep475.causal24dim.6l.64b`
   Purpose: separate the effect of frequency preprocessing from the effect of the query side-path itself.

If `rt192k` is already too slow for broad sweeps, mirror the same order with:

- `online-soft-band-query-sfc2d.rt128k.causal16dim.6l.64b`
- `online-soft-band-query-sfc2d.rt128k.fp512keep475.causal16dim.6l.64b`

Recommended execution order for DnR:

1. `online-soft-band-query-sfc2d.rt192k.causal24dim.6l.64b`
2. `online-soft-band-query-sfc2d.rt192k.fp512keep475.causal24dim.6l.64b`
3. `online-soft-band-sfc2d.rt192k.fp512keep475.causal24dim.6l.64b`

This keeps the first DnR pass focused on one question:
- does the `1025 -> 512` frontend help deployment enough without unacceptable quality loss?

You can inspect the stricter memory totals directly:

```sh
./.venv/bin/python tools/online/report_streaming_state_size.py --variant soft --budget-kib 192 --budget-dtype fp16
./.venv/bin/python tools/online/report_streaming_state_size.py --variant soft_dilated --d-model 24 --n-layers 6 --dilation-cycle 1 1 1 1 1 2 --fail-on-budget
./.venv/bin/python tools/online/report_streaming_state_size.py --variant hierarchical_soft_parallel_ffi --d-model 20 --pre-layers 0 --mid-layers 1 --bottleneck-layers 1 --time-branch-kernel-sizes 3 3 --time-branch-dilations 1 6
```

The report now shows:

- strict streaming `layer_cache`
- `input_history`
- selected externalized `band/basis constants`
- `all model params + buffers`
- `state + band/basis constants`
- `state + all model params + buffers`

For the custom ConvGRU separator, the first budget-safe candidates are:

- `rt192k`: `d_model=40`, `n_layers=6`, strict fp16 layer-cache `190.16 KiB`
- `rt128k`: `d_model=24`, `n_layers=10`, strict fp16 layer-cache `126.09 KiB`

For the new hierarchical soft-band family, the first budget-safe candidates are:

- `rt192k`: `d_model=20`, `layers=1/1/3`, `bands=128/96/48`, strict fp16 layer-cache `185.08 KiB`
- `rt128k`: `d_model=12`, `layers=1/1/3`, `bands=128/96/48`, strict fp16 layer-cache `111.05 KiB`

For the new dilated soft-band separator, the first budget-safe candidates are:

- `rt192k`: `d_model=24`, `n_layers=6`, `dilation_cycle=[1,1,1,1,1,2]`
- `rt128k`: `d_model=16`, `n_layers=6`, `dilation_cycle=[1,1,1,1,1,2]`

These stay within both:

- the strict fp16 layer-cache budget
- the NPU span rule `(kernel_size - 1) * dilation < 14`

If your priority is to push dilation as far as possible while still staying
under the `192 KiB` strict layer-cache budget, use the `maxdil` recipes:

- `recipes/musdb18hq/models/online-soft-band-dilated-sfc2d.rt192k.maxdil.causal16dim.6l.64b/config.yaml`
- `recipes/musdb18hq/models/online-soft-band-dilated-sfc2d.rt192k.maxdil.mel.causal16dim.6l.64b/config.yaml`

They use:

- `d_model=16`
- `n_layers=6`
- `dilation_cycle=[1,2,4,6,1,1]`

and reach:

- `32` frames of strict streaming context
- `184.06 KiB` fp16 layer-cache

The most useful first A/B for this new separator family is:

- `balanced` vs `maxdil` with `musical`
  - `recipes/musdb18hq/models/online-soft-band-dilated-sfc2d.rt192k.causal24dim.6l.64b/config.yaml`
  - `recipes/musdb18hq/models/online-soft-band-dilated-sfc2d.rt192k.maxdil.causal16dim.6l.64b/config.yaml`
- `balanced` vs `maxdil` with `mel`
  - `recipes/musdb18hq/models/online-soft-band-dilated-sfc2d.rt192k.mel.causal24dim.6l.64b/config.yaml`
  - `recipes/musdb18hq/models/online-soft-band-dilated-sfc2d.rt192k.maxdil.mel.causal16dim.6l.64b/config.yaml`

This isolates a high-value deployment question:

- is extra channel capacity more valuable than a longer temporal context when
  both candidates still satisfy the same NPU and `192 KiB` constraints?

### Batch budget sweeps

You can launch the `rt192k` / `rt128k` comparison matrix with:

```sh
./tools/online/run_budget_sweep.sh
```

The script accepts optional filters:

```sh
./tools/online/run_budget_sweep.sh rt192k
./tools/online/run_budget_sweep.sh rt128k soft
./tools/online/run_budget_sweep.sh rt192k soft-query
./tools/online/run_budget_sweep.sh rt192k soft-dilated
./tools/online/run_budget_sweep.sh rt192k soft-gru
./tools/online/run_budget_sweep.sh rt192k hierarchical-soft
./tools/online/run_budget_sweep.sh rt192k hard
```

The first argument is the budget set: `all`, `rt192k`, or `rt128k`. The second
argument is the family: `all`, `plain`, `soft`, `soft-query`, `soft-dilated`, `soft-gru`, `hierarchical-soft`, or `hard`.

Useful options for longer sweeps:

```sh
./tools/online/run_budget_sweep.sh rt192k --dry-run
./tools/online/run_budget_sweep.sh rt128k soft --log-root logs/rt128k_soft
./tools/online/run_budget_sweep.sh rt192k hard --stop-on-error
```

By default the script keeps going if one recipe fails, and writes one log file
per launched run under `logs/online_budget_sweep/`.

### Summarizing budget runs

After some runs finish, you can auto-generate a summary CSV from the recipe
directories:

```sh
./.venv/bin/python tools/online/summarize_budget_runs.py
```

By default this reads the starter template in
`docs/templates/online_budget_results.csv`, rescans the `rt192k` / `rt128k`
recipe directories, fills in the static architecture fields, and auto-detects:

- `best_checkpoint`
- `best_val_loss` when Lightning checkpoint metadata is available
- `onnx_export_ok` if an `.onnx` file exists under the recipe directory

The generated file is written to:

```text
docs/templates/online_budget_results.summary.csv
```

Useful options:

```sh
./.venv/bin/python tools/online/summarize_budget_runs.py --include-nonbudget
./.venv/bin/python tools/online/summarize_budget_runs.py --output /tmp/budget_summary.csv
```

### Exporting ONNX

The online wrappers expose a 2D-only core that can be exported to ONNX. The helper script below loads a trained online model, extracts the inner `core`, and exports it with a fixed `(B, C, T, F)` input shape.

For deployment, prefer the streaming export path so the layer cache/state is part of the ONNX graph interface:

```sh
./.venv/bin/python tools/online/export_onnx_online_model.py \
  recipes/musdb18hq/models/online-soft-band-sfc2d.causal96dim.12l \
  --out /tmp/online_soft_band_sfc2d.onnx \
  --n-chan 2 \
  --frames 64 \
  --streaming \
  --state-meta-out /tmp/online_soft_band_sfc2d.state.json \
  --externalize-band-constants \
  --disable-masking \
  --opset 11 \
  --check \
  --fail-on-disallowed-ops
```

`--streaming` exports `x, state_0, ..., state_N -> y, next_state_0, ..., next_state_N`.
By default, selected band/basis priors stay embedded in ONNX initializers just like other model weights.
`--externalize-band-constants` is the fallback that moves them to explicit graph inputs.
`--disable-masking` keeps packed complex multiply outside the graph, which is often easier on stricter NPU compilers.
`--keep-initializers-as-inputs` is available as a last-resort fallback when a converter is especially sensitive to embedded constants / buffers.

If you want a device-facing package instead of only an `.onnx` file:

```sh
./.venv/bin/python tools/online/export_onnx_online_model.py \
  recipes/musdb18hq/models/online-soft-band-sfc2d.causal96dim.12l \
  --out /tmp/online_soft_band_sfc2d.onnx \
  --n-chan 2 \
  --frames 64 \
  --streaming \
  --state-meta-out /tmp/online_soft_band_sfc2d.state.json \
  --externalize-band-constants \
  --constants-out /tmp/online_soft_band_sfc2d.constants.npz \
  --deploy-manifest-out /tmp/online_soft_band_sfc2d.deploy_manifest.json
```

You can also export directly from a recipe `config.yaml` before training:

```sh
./.venv/bin/python tools/online/export_onnx_online_model.py \
  recipes/musdb18hq/models/online-soft-band-sfc2d.rt192k.causal24dim.6l.64b/config.yaml \
  --out /tmp/online_soft_band_from_config.onnx \
  --n-chan 2 \
  --frames 64 \
  --streaming \
  --state-meta-out /tmp/online_soft_band_from_config.state.json \
  --deploy-manifest-out /tmp/online_soft_band_from_config.deploy_manifest.json
```

You can also audit an existing ONNX file without re-exporting:

```sh
./.venv/bin/python tools/online/audit_onnx_model.py /tmp/online_soft_band_sfc2d.onnx
```

To check the stricter 192 KiB deployment interpretation against a stateful
export:

```sh
./.venv/bin/python tools/online/audit_onnx_model.py \
  /tmp/online_soft_band_sfc2d.onnx \
  --state-meta /tmp/online_soft_band_sfc2d.state.json \
  --budget-kib 192 \
  --budget-dtype fp16 \
  --fail-on-budget
```

To export a different online variant, replace the model directory with one of the other online recipes listed above.

## Copyright and license

Released under `MIT` license, as found in the [LICENSE.md](LICENSE.md) file.

All files, except as noted below:

```
Copyright (c) 2026 National Institute of Advanced Industrial Science and Technology (AIST), Japan

SPDX-License-Identifier: MIT
```

The following file:

- `spectral_feature_compression/core/model/bandit_split.py`

was adapted from https://github.com/kwatcharasupat/bandit (license included in [LICENSES/Apache-2.0.md](LICENSES/Apache-2.0.md))

```
Copyright (c) 2026 National Institute of Advanced Industrial Science and Technology (AIST), Japan
Copyright (c) 2023 Karn Watcharasupat

SPDX-License-Identifier: MIT
SPDX-License-Identifier: Apache-2.0
```

The following file:

- `spectral_feature_compression/core/model/bslocoformer.py`

was adapted from https://github.com/merlresearch/tf-locoformer (license included in [LICENSES/Apache-2.0.md](LICENSES/Apache-2.0.md))

```
Copyright (c) 2026 National Institute of Advanced Industrial Science and Technology (AIST), Japan
Copyright (c) 2024 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: MIT
SPDX-License-Identifier: Apache-2.0
```

The following file:

- `spectral_feature_compression/core/model/average_model_params.py`

was adapted from https://github.com/espnet/espnet (license included in [LICENSES/Apache-2.0.md](LICENSES/Apache-2.0.md))

```
Copyright (c) 2026 National Institute of Advanced Industrial Science and Technology (AIST), Japan
Copyright (c) 2017 ESPnet Developers

SPDX-License-Identifier: MIT
SPDX-License-Identifier: Apache-2.0
```

The following files:

- `spectral_feature_compression/core/loss/snr.py`

were adapted from https://github.com/kohei0209/self-remixing (license included in [LICENSES/MIT.md](LICENSES/MIT.md))

```
Copyright (c) 2026 National Institute of Advanced Industrial Science and Technology (AIST), Japan
Copyright (c) 2024 Kohei Saijo

SPDX-License-Identifier: MIT
SPDX-License-Identifier: MIT
```
