# Model-First Redesign of the SFC Teacher

## Summary

Keep the current on-the-fly synthesis and composite loss unchanged during the first comparison. Redesign the teacher to address three structural weaknesses that directly match the reported symptoms:

- sources are not separated until the final decoder;
- reconstruction depends almost entirely on a shared compressed mask representation;
- independently predicted complex masks can jitter and need not sum to the mixture.

The new teacher remains mono/24 kHz, retains SFC-CA and TF-Locoformer, and targets 20–35M parameters.

## Architecture

### Tensor flow

```text
Complex mixture STFT [B, 1, F, T]
    ↓
SFC-CA encoder with 80 tokens
    ↓
Shared mixture representation [B, 144, T, 80]
    ↓
4 shared TF-Locoformer blocks
    ↓
Learned source split for speech/music/effects
    ↓
Source representations [B, 3, 144, T, 80]
    ↓
4 source-aware TF-Locoformer blocks
    ↓
Adaptive full-band SFC decoder per source
    ↓
Complex mask + complex residual + confidence
    ↓
Mixture-consistent complex estimates
    ↓
ISTFT
```

### Shared mixture encoder

Use the existing SFC-CA encoder with:

```yaml
n_bands: 80
d_inner: 112
d_model: 144
n_heads: 4
band_config: musical
query_type: learnable
learnable_pos_bias: true
mask_outside_bands: false
```

The increase from 64 to 80 tokens gives the teacher more frequency capacity without turning it into an uncompressed full-band transformer.

Pass the encoder’s original per-frequency embeddings forward as a full-band skip for the decoder. They must not be discarded after compression.

### Shared separator

Apply four TF-Locoformer blocks to the mixture tokens:

```yaml
n_shared_layers: 4
emb_dim: 144
n_heads: 8
attention_dim: 144
ffn_hidden_dim: [224, 224]
conv1d_kernel: 8
conv1d_shift: 1
num_groups: 8
dropout: 0.1
tf_order: ft
```

These layers model common harmonic, temporal, and programme context before committing features to an output source.

### Early source split

After the fourth shared block, create three streams:

```text
H_source[s] = source_seed(H_mixture) + source_embedding[s]
```

Use one learned embedding for each fixed semantic source:

```text
speech
music
effects
```

The embeddings break output symmetry and allow the speech stream to learn a singing/dialogue prior before reconstruction.

### Source-aware refinement

Use four distinct source-refinement layers. Each layer performs:

1. A weight-shared TF-Locoformer block over the flattened `[batch × source]` dimension.
2. Source-axis attention across speech/music/effects at every time/band position.
3. Mixture–source fusion using:
   - the current source;
   - the shared mixture representation;
   - the mean of the other two sources.
4. Residual update with learnable layer scale initialized to `0.1`.

Weight sharing prevents parameter count from tripling while allowing source embeddings and source-axis attention to produce different behaviour.

This directly addresses vocal/music ambiguity: speech and music features compete and exchange information before the final mask, instead of being emitted as unrelated channels from one shared tensor.

### Adaptive source decoder

Replace the current decoder that emits all sources from the same latent tensor.

For each source:

1. Flatten source into the batch dimension.
2. Repeat the encoder’s full-frequency embeddings for each source.
3. Use a shared `CrossAttnDecoder` configured with `n_src=1`.
4. Use adaptive decoder queries derived from the mixture’s original per-bin embeddings.
5. Reshape the result back to `[B, source, feature, T, F]`.

The adaptive full-band query supplies fine harmonic, consonant, breath, and sibilance detail that may be weakened in the 80-token bottleneck.

## Reconstruction Head

For every source and time-frequency bin, emit five values:

```text
mask_real
mask_imag
residual_real
residual_imag
confidence
```

### Stabilized complex mask

Bound mask components smoothly:

```text
mask = 2 × tanh(raw_mask / 2)
```

This still permits mask magnitude above one, which is necessary for destructive phase interference, while preventing extreme mask spikes that can create isolated musical noise.

### Residual reconstruction

Construct the initial source estimate as:

```text
Y_raw[s] = complex_multiply(X, mask[s])
           + residual_scale × residual[s]
```

Use one learnable scalar `residual_scale`, initialized to `0.05` through a positive parameterization.

The residual branch reconstructs details that a mixture mask cannot represent cleanly, especially when vocal and accompaniment energy cancel in the mixture phase.

### Confidence-weighted mixture consistency

Enforce exact complex-STFT mixture consistency:

```text
correction = X - sum(Y_raw)
weights = softmax(confidence, source_dimension)
Y[s] = Y_raw[s] + weights[s] × correction
```

This guarantees:

```text
sum(Y[s]) == X
```

The confidence weighting is preferable to dividing the correction equally: ambiguous bins are assigned to the source the model considers most plausible.

## Implementation Route

Create a new `SourceAwareSFCLocoformerTeacher` rather than modifying `BSLocoformer` in place.

Reuse:

- `CrossAttnEncoder`;
- `CrossAttnDecoder`;
- `TFLocoformerBlock`;
- source-axis attention and mixture/source fusion concepts already present in the source-aware Mel-Band implementation.

Add a builder named:

```text
build_source_aware_sfc_locoformer_teacher_system
```

Add a dedicated teacher recipe containing the complete datamodule and `CompositeSupTask` configuration. Do not inherit a recipe containing the HDF5 datamodule.

The builder must expose only architecture settings needed by the recipe; avoid adding deployment/NPU options to this offline teacher.

## Loss and Data Controls

For the architecture comparison:

- retain the actual on-the-fly mixer unchanged;
- retain the current composite-loss weights unchanged;
- retain `2048/512`, six-second training segments, optimizer, EMA, schedule, and number of updates;
- use the same training manifest and random seed.

Do not combine synthesis repair with the model comparison. Otherwise an improvement cannot be attributed to the redesigned teacher.

The hard mixture-consistency projection makes the existing soft consistency loss redundant. Set its weight to zero only in the full redesigned model; leave all other composite components unchanged.

## Controlled Experiments

### Run A: current teacher

Train or evaluate the existing 16M SFC teacher as the baseline.

### Run B: capacity control

Build a 20–35M version of the existing late-split architecture using the same `d_model=144`, eight total Locoformer layers, and 80 SFC tokens, but retain the existing shared all-source decoder.

Purpose: determine how much improvement comes from capacity alone.

### Run C: source-aware teacher

Use the full early-split, adaptive-decoder, residual-reconstruction, confidence-projection design.

All three runs must use identical:

- data;
- loss;
- batch construction;
- optimizer;
- number of updates;
- initialization seed where compatible;
- validation examples.

Run B and C for 20,000 optimizer steps before committing to full training.

### Interpretation

- B and C both beat A equally: insufficient capacity was the main cause.
- C clearly beats B: late source separation/reconstruction was the main cause.
- C improves SI-SDR but not listening quality: add mask-continuity regularization in a separate run.
- Neither B nor C improves: the dominant cause is targets or synthesis, and architecture work should stop.

## Evaluation

Create a fixed evaluation set, without changing training synthesis:

- at least 20 failing songs;
- clean vocal/music/effects references;
- six-second excerpts for direct-forward evaluation;
- complete songs for CSS evaluation;
- additional movie, sports, and news controls.

Measure:

- speech/vocal SI-SDR and SNR;
- complex and log-magnitude error;
- speech leakage during vocal pauses;
- `||mixture - sum(estimates)||`;
- first- and second-order temporal error of vocal log magnitude;
- blinded listening preference.

Listen specifically for:

- watery or phasey vocal texture;
- isolated tonal noise;
- broken sustained notes;
- damaged consonants and sibilance;
- accompaniment residue;
- pumping or CSS boundary changes.

## Tests

- Output shape matches `[B, 3, 1, F, T]`.
- Source embeddings produce distinguishable source streams.
- The shared source decoder does not multiply parameters by three.
- The model contains 20–35M trainable parameters.
- Complex-mask components remain bounded in `[-2, 2]`.
- Residual scale starts at `0.05` and receives finite gradients.
- Confidence weights sum to one across sources.
- Separated complex estimates sum to the mixture within numerical tolerance.
- Forward/backward works in bf16 mixed precision.
- Six-second and CSS inference return finite waveforms with exact input length.
- Existing SFC and Locoformer tests remain unchanged and passing.

## Acceptance Criteria

Advance Run C to full 150-epoch training only if, after the short comparison:

- it beats capacity-matched Run B by at least 0.5 dB median song-vocal SI-SDR, or wins at least 70% of blinded song comparisons;
- temporal vocal log-magnitude error improves by at least 10%;
- inactive-vocal leakage does not worsen;
- no non-song domain loses more than 0.25 dB SI-SDR;
- mixture reconstruction error is at numerical precision;
- parameter count remains within 20–35M.

If Run C succeeds, train it fully and select the checkpoint using aggregate validation plus song-vocal SI-SDR. If it fails, return to target/synthesis investigation rather than continuing to scale the model.

## Assumptions

- Teacher remains mono and 24 kHz.
- SFC-CA and TF-Locoformer remain the architectural foundation.
- Moderate 20–35M teacher size is acceptable.
- Student, distillation, NPU export, and realtime constraints are out of scope.
- Data synthesis is held fixed during the model-first diagnosis.
