# SFC teacher song/vocal quality root-cause investigation

Date: 2026-06-20

## Scope

Teacher recipe:

`recipes/dnr/models/locoformer-medium.enc-crossattn96dim.dec-crossattn96dim.musical64.learnable-query/config.yaml`

Symptom: teacher `speech` output for songs/singing vocals is rough, not smooth, and contains noisy/indistinct artifacts.

User constraints/assumptions:

- Dataset stem taxonomy is intentional: song vocal is in `speech`; instrumental music has basically no vocal; other material is `effects`.
- Training/inference sample rate is consistently 24 kHz, so sample-rate mismatch is not considered the root cause.
- Investigation focuses only on the teacher, not student variants.

## Files inspected

- `recipes/dnr/models/locoformer-medium.enc-crossattn96dim.dec-crossattn96dim.musical64.learnable-query/config.yaml`
- `recipes/dnr/models/locoformer-small.enc-crossattn64dim.dec-crossattn64dim.musical64.learnable-query/config.yaml`
- `spectral_feature_compression/common/datasets/on_the_fly_stem_dataset.py`
- `spectral_feature_compression/common/datamodules/on_the_fly_stem_datamodule.py`
- `spectral_feature_compression/core/tasks/sup_task.py`
- `spectral_feature_compression/core/tasks/composite_sup_task.py`
- `spectral_feature_compression/core/loss/snr.py`
- `spectral_feature_compression/core/loss/composite_separation.py`
- `spectral_feature_compression/core/model/model_wrapper.py`
- `spectral_feature_compression/core/model/bslocoformer.py`
- `spectral_feature_compression/core/model/crossattn_enc_dec.py`
- `spectral_feature_compression/core/model/enc_dec_base.py`
- `spectral_feature_compression/core/model/bandit_split.py`

Commands/examples run during inspection:

```bash
find spectral_feature_compression recipes/dnr -maxdepth 4 -type f | head -200
rg -n "locoformer-medium|enc-crossattn96dim|loss_supervised|validation/snr|loss_composite|ThresSNRLoss" logs recipes/dnr -g '!**/*.ckpt' -g '!**/*.pt' -g '!**/*.pth' | head -200
```

## Key teacher configuration facts

- 24 kHz audio, `n_fft=2048`, `hop_length=512`.
- 6-second training chunks.
- On-the-fly dry additive mixer with independent source pools.
- 3 stems: `speech`, `music`, `effects`.
- Model: BS-Locoformer + cross-attention SFC encoder/decoder.
- `n_layers=6`, `emb_dim=128`, `attention_dim=128`, `d_inner=96`, `n_bands=64`, `band_config=musical`, `query_type=learnable`.
- Loss stack: inherited `ThresSNRLossWithInactiveSource` plus `CompositeSupTask` auxiliary losses:
  - mixture consistency 0.1
  - low-frequency magnitude 0.2 below 300 Hz
  - complex RI 0.5
  - log magnitude 0.2
  - multi-resolution STFT 0.3
  - transient 0.08
- Training max epochs reduced to 20 in the medium config, while the official small base recipe used 150 epochs.

## Root-cause ranking

### P0 — Teacher training budget is probably far too short for this model/data/task

The medium teacher overrides the inherited official schedule to only:

```yaml
trainer:
  max_epochs: 20
```

The inherited small official recipe uses 150 epochs with a warmup/decay scheduler. The medium recipe also sets `scheduler_generator: null` and `lr=1e-4`, while the inherited recipe uses `lr=1e-3` with warmup and long decay.

For a broad 3-stem TV/song separator trained from independent pools, 20 epochs is likely undertrained. Singing-vocal quality usually improves late because the model first learns easy energy separation and only later learns stable fine spectral masks for harmonics, sibilants, tails, and accompaniment rejection.

Expected symptom of undertraining: speech output has recognizable vocal but rough texture, musical-noise-like residue, unstable consonants, and indistinct residual accompaniment. This matches the report.

### P1 — On-the-fly dry random mixing does not match real song/program mixtures

The datamodule constructs examples by independently sampling source pools and summing:

```python
ref = stems[:, None, :].contiguous()
wav = ref.sum(dim=0).contiguous()
```

This is not equivalent to real TV/song mixtures. For songs, vocal and accompaniment are normally co-occurring stems from the same song, with matched key, tempo, arrangement, reverb, compression, limiting, stereo/mastering, and correlated leakage/artifacts. Randomly combining vocal from song A with instrumental from song B creates diversity but weakens the real separation cues and hard cases.

This can produce a teacher that performs acceptably on synthetic validation but is unstable on real songs.

### P2 — `speech` is a broad heterogeneous class: spoken speech + singing vocals

The single `speech` output must cover:

- clean/news speech;
- sports announcers;
- movie dialogue;
- singing vocals;
- possibly pseudo-separated vocal stems with artifacts.

Singing vocals are acoustically closer to music than spoken speech: harmonic, sustained, pitch-synchronous, reverberant, often compressed, and strongly overlapping with instruments. If song vocals are not domain-balanced or fine-tuned, the model can optimize easier spoken-speech cases and leave singing vocals less smooth.

### P3 — 24 kHz + `n_fft=2048` + `musical64` SFC is a compressed teacher, not a high-fidelity song-vocal teacher

At 24 kHz the teacher only models up to 12 kHz. This can be acceptable for TV edge deployment, but it is not ideal for a teacher whose output should be perceptually clean on songs. Vocal breath/sibilance and many music-confuser cues live around and above 8–12 kHz.

The SFC encoder compresses 1025 FFT bins to 64 musical bands. The musical filterbank is perceptual/overlapped, but high-frequency bands are broad compared with detailed vocal/music separation needs. Broad bands can smear cymbals, sibilance, guitar/string noise, and vocal reverb tails into the same compressed tokens, causing unstable masks/noisy vocal output.

Known SOTA music separation directions such as Band-Split RNN, Mel-Band RoFormer, and SCNet also use subband compression, but they preserve stronger frequency detail and/or use more music-specific hierarchical modeling than this moderate SFC teacher recipe.

### P4 — Pseudo-separated song vocal references can cap the teacher quality

The dataset organization is semantically correct, but if song vocals were obtained by a separator rather than true multitrack stems, the `speech` reference may already contain:

- musical noise;
- residual instruments;
- phase/watery artifacts;
- over-suppressed harmonics;
- rough sibilance;
- truncated vocal/reverb tails.

The teacher cannot reliably become cleaner than its supervised target unless the training objective/data curation explicitly counters those artifacts. It may learn to reproduce target artifacts.

### P5 — Objective is richer than plain SNR, but still not song-vocal/perceptual enough

The current teacher does use `CompositeSupTask`, so it is not only waveform SNR. However, the base objective is still inherited `ThresSNRLossWithInactiveSource`, and the auxiliary spectral losses are generic.

Potential issues:

- no explicit vocal/music perceptual loss;
- no source-aware class weighting for singing-vocal frames;
- inactive leakage penalty remains inherited `zeroref_weight: 0.1`, weak for speech leakage/noise calibration;
- low-frequency auxiliary emphasis below 300 Hz does not directly target vocal smoothness/sibilance/music leakage;
- generic complex RI/log-mag losses may still reward reproducing pseudo-label artifacts.

### P6 — Active-stem distribution makes full 3-stem mixtures dominate

```yaml
active_stem_count:
  weights:
    1: 0.01
    2: 0.14
    3: 0.85
```

Most examples contain all stems. This helps mixture robustness, but it gives few clean single-stem or two-stem calibration cases. For teacher quality, music-only/effects-only/speech-only examples are useful to teach clean inactivity and reduce noisy speech-head leakage.

### P7 — Same-stem placement and normalization may create unnatural vocal examples

For speech, the recipe can concatenate 1–2 clips with random sequential gaps/overlap. `normalize_sources: true` normalizes the full 6-second source timeline RMS before gain. For sparse clips, this can amplify active portions depending on silence/gap content. The resulting vocal density/SNR distribution may not match songs.

This is not a code bug, but it is a data-distribution mismatch.

### P8 — Validation/checkpoint selection is not song-specific enough

The medium on-the-fly validation uses synthetic examples from the same mixer unless overridden. Overall validation loss can improve while song-vocal perceptual quality remains poor. Teacher checkpoint selection needs fixed domain-specific song validation and listening examples.

## Highest-value confirmation experiments

1. **Train longer / resume longer**: run 80–150 epochs or equivalent steps with scheduler; compare fixed song examples every 5–10 epochs.
2. **Overfit a tiny paired song subset**: if the model cannot produce smooth vocal when overfitting true/pseudo paired stems, structure/objective/reference quality is limiting; if it can, the main issue is data distribution/training budget.
3. **Audit song vocal references**: listen to the `speech` references for failing songs before training. If references are rough, teacher output will be rough.
4. **Fixed paired song validation**: create a manifest with song vocal/instrumental/effects from the same program/song and track per-domain SDR plus listening samples.
5. **Paired song fine-tune**: fine-tune the teacher on paired/co-occurring song mixtures instead of only random pool mixes.
6. **Ablate stronger inactive/source calibration**: increase `zeroref_weight` to 0.5–1.0 and include more 1-stem/2-stem examples.
7. **Band config ablation for teacher only**: test more bands or mel/SCNet-like banding for high-frequency detail if compute is acceptable for teacher training.

## Practical conclusion

The most likely root cause is not sample rate and not stem label organization. It is the combination of:

1. undertrained medium teacher schedule;
2. synthetic independent dry mixing that does not match co-occurring songs;
3. broad `speech` class mixing dialogue and singing vocals;
4. compressed 24 kHz musical64 SFC representation limiting fine vocal/music detail;
5. possible pseudo-separated vocal target artifacts;
6. validation that does not explicitly select for song-vocal quality.
