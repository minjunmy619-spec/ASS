# Deep research: why the SFC/Locoformer teacher can fail on song vocals

Date: 2026-06-20

Target teacher config:

`recipes/dnr/models/locoformer-medium.enc-crossattn96dim.dec-crossattn96dim.musical64.learnable-query/config.yaml`

User constraints:

- Same 24 kHz sample rate is used for training/inference.
- Stem taxonomy is intentional: singing vocal belongs to `speech`, instrumental belongs to `music`, other content belongs to `effects`.
- Focus is only teacher quality, not student/distillation variants.

## 1. High-priority code/config finding: exact config has a suspicious channel mismatch

The medium config sets inside the teacher model:

```yaml
n_chan: &n_chan 2
```

But `OnTheFlyStemDataset._load_audio()` downmixes every source file to mono:

```python
audio = torch.from_numpy(audio_np.T.copy()).float().mean(dim=0, keepdim=True)
...
audio = audio.squeeze(0)
```

and later returns:

```python
ref = stems[:, None, :].contiguous()  # [N, 1, T]
wav = ref.sum(dim=0).contiguous()     # [1, T]
```

Therefore the on-the-fly training batch is mono. A `BSLocoformer` instantiated with `n_chan=2` expects 4 real/imag input channels in the SFC input convolution, while mono STFT produces only 2 real/imag channels.

If this exact config really trained successfully, then one of these must be true:

1. the actual `merged_config.yaml` differs and `n_chan=1` was used;
2. the dataloader was not this on-the-fly mono source-pool path;
3. local code was changed after training;
4. the reported model path is not exactly this config.

This needs immediate verification from the actual run's `merged_config.yaml` and model summary. If the training somehow used mismatched channel assumptions or a different config than intended, teacher quality analysis can be misleading.

## 2. Training schedule is probably inadequate for a broad song/TV teacher

The medium recipe overrides the inherited official small recipe from 150 epochs to 20 epochs:

```yaml
trainer:
  max_epochs: 20
```

It also removes the inherited scheduler:

```yaml
scheduler_generator: null
lr: 1.0e-4
```

For broad 3-stem TV/music separation, the hard part is not learning coarse energy routing. The hard part is stable fine-grained separation of partially overlapping harmonic structures: vocal fundamentals/formants/sibilants vs guitars/synths/cymbals/reverb/noise. That usually improves late in training.

Likely undertraining symptom:

- vocal exists but sounds grainy/watery;
- consonants and breath are unstable;
- accompaniment residue appears as indistinct noise;
- output is not temporally smooth;
- inactive regions of speech output contain low-level texture.

This matches the user symptom strongly.

## 3. On-the-fly normalization can amplify pseudo-vocal artifacts and silence-region noise

The dataset constructs a full 6-second source timeline, then normalizes the active stem RMS:

```python
rms = stems[stem_idx].square().mean().sqrt()
stems[stem_idx] /= rms.clamp_min(1.0e-8)
```

This RMS is computed over the entire placed 6-second stem, including gaps/silence. For singing vocals, activity can be sparse: phrases, breaths, pauses, reverb tails. If a pseudo-separated vocal has residual musical noise in nominally silent gaps, full-window RMS normalization can raise that residual noise relative to the mixture.

This can create references where the `speech` target includes boosted separator noise in vocal gaps. The teacher is then trained to reproduce that texture as speech.

This is especially dangerous for song vocals because pseudo-separated vocal stems often contain low-level accompaniment residue between phrases.

## 4. Synthetic independent dry mixing breaks co-occurrence statistics

The on-the-fly mixer samples stems independently from source pools and dry-adds them. For songs, real mixtures are not independent:

- vocal and instrumental are from the same composition;
- same key/tempo/harmony;
- same room/reverb/master bus;
- same compression/limiting/EQ;
- same stereo image and production style;
- same stem-separation artifacts if pseudo-stems are used.

Independent dry mixing produces a different problem distribution. It can even make some examples easier than real songs because unrelated instrumental does not lock harmonically/rhythmically to the vocal. Conversely, it can create impossible/unrealistic combinations that encourage the network to rely on brittle spectral heuristics.

For a teacher, this is bad: a teacher should model the real distribution better than the student, not only synthetic mixtures.

## 5. The `speech` class is acoustically multi-modal

The teacher has one `speech` output head for both spoken speech and singing vocals.

Spoken speech:

- mostly non-sustained phonemes;
- formant-driven;
- relatively sparse harmonic structure;
- different prosody;
- often center/dialogue-like.

Singing vocal:

- sustained notes;
- strong harmonic stack;
- vibrato;
- pitch locked to musical scale;
- heavy reverb/delay/compression;
- timbrally close to instruments.

A single unconditioned output head can learn a compromise. If spoken speech is more common or easier, the model may optimize spoken speech first and leave singing vocal as a weaker subdomain. This gives rough song vocals even when news/dialogue separation sounds acceptable.

## 6. SFC `musical64` compression at 24 kHz is a likely structural quality bottleneck

Using the project code with the current virtual environment, the band layout for `musical64`, `n_fft=2048`, `sr=24000` is:

```text
musical n=64, unique ranges=56
min/max band width: 2 / 201 bins
min/max width Hz: 23.4 / 2355.5 Hz
first 12 bands:
(0,3), (1,3), (1,3), (1,3), (1,3), (1,3), (1,4), (1,4), (2,4), (2,4), (2,5), (3,5)
last 8 bands:
(425,530), (474,591), (530,660), (591,736), (660,822), (737,917), (823,1024), (918,1025)
```

Consequences:

- Several low-frequency bands duplicate almost the same bin range, wasting part of the 64-token budget.
- High-frequency bands become very wide, above 1–2 kHz per band near the top.
- Song-vocal perceptual artifacts often live exactly in high-frequency detail: sibilance, breath, cymbal bleed, guitar string noise, reverb tails.
- The decoder must reconstruct 1025 frequency bins from 64 compressed tokens plus learned queries. That can create unstable high-band masks.

Comparison from project band code at 24 kHz:

```text
mel64:     unique=64, max width ~1090 Hz
ERB64:     unique=64, max width ~1406 Hz
musical64: unique=56, max width ~2355 Hz
bark64:    unique=64, max width ~5590 Hz
```

`musical64` is not obviously the best teacher choice for song-vocal fidelity at 24 kHz.

## 7. 24 kHz teacher bandwidth is a teacher-quality tradeoff

Even with consistent training/inference sample rate, 24 kHz means Nyquist is 12 kHz. This is aligned with edge deployment, but a teacher usually should be stronger than the student.

For song vocals, upper-band information helps separate:

- breath and air;
- fricatives/sibilants;
- cymbal vs vocal consonants;
- vocal reverb tails;
- mastered music brightness.

A 24 kHz teacher can never model above 12 kHz. If the separated result is judged against full-band song expectations, the vocal will naturally sound less open and potentially rough after resampling/playback.

## 8. Complex-mask objective/architecture can produce phasey or watery artifacts

The model predicts a complex mask and multiplies it by the mixture STFT:

```python
batch = batch0.unsqueeze(1) * batch
```

This is efficient, but it has known issues for music:

- if target phase differs from mixture phase in overlapping regions, mask estimation is ill-conditioned;
- small phase errors cause time-domain roughness;
- high-frequency complex masks are especially unstable;
- artifacts appear as musical noise or watery/phasiness.

Complex RI and MR-STFT losses help, but do not fully solve this if the reference itself has pseudo-separation artifacts or if the band compression lacks detail.

## 9. Loss stack is generic, not source/domain-aware

The current teacher uses composite losses, which is good, but they are generic waveform/spectral losses. Missing for this failure mode:

- no song-domain weighting;
- no vocal-frame weighting;
- no harmonic continuity loss;
- no perceptual vocal quality loss;
- no explicit leakage/silence calibration beyond inherited inactive-source handling;
- no confidence weighting for pseudo-label quality.

The inherited inactive penalty appears weak for a teacher:

```yaml
zeroref_weight: 0.1
```

This can allow low-level speech-head noise/leakage, which is audible in song vocals and in non-vocal gaps.

## 10. Active-stem mix distribution may reduce clean calibration

The current distribution is dominated by full mixtures:

```yaml
active_stem_count:
  weights:
    1: 0.01
    2: 0.14
    3: 0.85
```

Full mixtures are important, but only 1% single-stem examples is low for calibrating output-head silence and clean reconstruction. More music-only/effects-only/speech-only examples can reduce speech-output noise and leakage.

## 11. CSS/inference can add secondary roughness on long songs

`ModelWrapper.css()` processes long audio segment by segment. Scaling is applied per segment inside `forward()`, then overlaps are averaged. For long songs, segment-wise normalization and overlap averaging can cause slight pumping or boundary texture changes.

This is probably secondary, but should be tested by comparing:

- one short 6-second excerpt processed directly;
- the same excerpt processed inside a long CSS song run;
- boundary vs non-boundary regions.

## 12. SOTA context and what it implies

Relevant source-separation trends:

- **Band-Split RNN**: strong music separation by keeping meaningful subband structure and modeling band/sequence interactions.
- **Mel-Band RoFormer**: strong music-source separation by using overlapping mel bands and rotary/hierarchical sequence modeling.
- **SCNet**: sparse compression with source-separation-specific frequency treatment and efficient strong music results.
- **Demucs/Hybrid Transformer Demucs family**: strong song separation benefits from hybrid waveform + spectrogram modeling and large context.
- **BS-RoFormer/MelBand RoFormer variants**: commonly used for high-quality vocal/instrumental separation, often at 44.1 kHz with strong capacity.

Implication: the current SFC teacher is conceptually related to subband compression, but it is configured as a moderate compressed TV separator, not as a maximum-quality music-vocal teacher.

## 13. Diagnostic experiments that can isolate root cause

### A. Verify exact run config

Check actual `merged_config.yaml` from the training run:

- `task.model.n_chan`
- `task.model.model.n_chan`
- datamodule target
- sample rate
- loss target
- max epochs
- scheduler

The read config has a suspicious `n_chan=2` vs mono dataloader issue.

### B. Overfit true/pseudo paired song stems

Use 10–50 song examples. Train until training loss is extremely low.

Interpretation:

- Cannot overfit smoothly: structure/objective/reference quality is limiting.
- Can overfit smoothly: main issue is data distribution/training length/checkpoint selection.

### C. Reference-quality audit

For failing songs, listen to reference stems:

- speech/vocal reference;
- music reference;
- effects reference;
- mixture;
- teacher output.

If target vocal has rough artifacts, teacher will inherit them.

### D. Synthetic-vs-paired A/B

Train/fine-tune two teachers:

1. current independent random dry mixer;
2. fixed paired song mixes from same song/program.

Evaluate on fixed song validation. This directly tests the co-occurrence mismatch hypothesis.

### E. Band-map ablation

Teacher-only ablations:

- `musical64` vs `mel64` vs `erb64`;
- `musical128` if affordable;
- 44.1/48 kHz stronger teacher if teacher quality matters more than deployment alignment.

### F. Training-budget ablation

Same data/model:

- 20 epochs;
- 50 epochs;
- 100 epochs;
- 150 epochs.

Listen to fixed song examples at each checkpoint. If quality improves monotonically, do not redesign before training sufficiently.

### G. Inactive/leakage ablation

Try:

```yaml
zeroref_weight: 0.5
```

and/or:

```yaml
zeroref_weight: 1.0
```

Also increase single/two-stem examples, e.g. less than 85% full 3-stem.

## 14. Most likely combined explanation

The failure is probably not one single bug. It is an interaction:

1. possible exact-run config/channel inconsistency to verify;
2. short 20-epoch teacher training;
3. random independent dry mixing instead of paired real songs;
4. broad `speech` class covering both dialogue and singing;
5. pseudo-separated vocal targets possibly containing artifacts;
6. 24 kHz + `musical64` SFC losing high-frequency/detail needed for smooth vocals;
7. weak inactive/leakage calibration and non-song-specific validation.

## 15. Recommended next action order

1. Verify actual `merged_config.yaml` and channel count.
2. Build fixed song validation/listening set.
3. Audit vocal reference quality.
4. Overfit a tiny song subset.
5. Resume/train longer with proper scheduler.
6. Fine-tune on paired/co-occurring song stems.
7. Only then change architecture/band mapping if needed.

## 16. Implementation update: added speech/vocal-emphasis band split

Added a new `SpeechVocalBandsplitSpecification` in:

- `spectral_feature_compression/core/model/bandit_split.py`

Accepted band config names:

- `vocal`
- `vocal64`
- `speech_vocal`
- `speech_vocal64`

Also updated cross-attention SFC encoder/decoder validation in:

- `spectral_feature_compression/core/model/enc_dec_base.py`

Added a teacher ablation config:

- `recipes/dnr/models/locoformer-medium.enc-crossattn96dim.dec-crossattn96dim.vocal64.learnable-query/config.yaml`

Added tests:

- `tests/test_speech_vocal_band_split.py`

Validation command:

```bash
./.venv/bin/python -m pytest tests/test_speech_vocal_band_split.py -q
```

Result:

```text
3 passed
```

## 17. Implementation update: vocal-aware robust/truncated teacher loss

Added a teacher-only task:

- `spectral_feature_compression/core/tasks/vocal_aware_sup_task.py`

Class:

- `VocalAwareCompositeSupTask`

It extends `CompositeSupTask` and adds optional speech/vocal-aware losses:

1. soft-truncated speech log-magnitude MR-STFT loss;
2. soft-truncated temporal log-magnitude gradient loss;
3. soft-truncated frequency log-magnitude gradient loss;
4. non-truncated inactive speech leakage loss.

Rationale:

- soft truncation prevents noisy pseudo-separated vocal labels from dominating;
- temporal/frequency logmag gradients encourage smoother harmonic/formant continuity;
- inactive leakage loss keeps vocal gaps clean and is intentionally not truncated.

The `vocal64` teacher config now uses this task and sets:

```yaml
speech_robust_logmag_weight: 0.05
speech_robust_logmag_tau: 1.0
vocal_active_frame_weight: 2.0
speech_temporal_logmag_gradient_weight: 0.03
speech_frequency_logmag_gradient_weight: 0.02
speech_gradient_tau: 1.0
speech_inactive_leakage_weight: 0.05
speech_inactive_threshold_db: -45.0
speech_inactive_softness_db: 6.0
loss:
  zeroref_weight: 0.5
```

Added tests:

- `tests/test_vocal_aware_sup_task.py`

Validation command:

```bash
./.venv/bin/python -m pytest \
  tests/test_vocal_aware_sup_task.py \
  tests/test_on_the_fly_source_normalization.py \
  tests/test_speech_vocal_band_split.py -q
```

Result:

```text
9 passed
```
