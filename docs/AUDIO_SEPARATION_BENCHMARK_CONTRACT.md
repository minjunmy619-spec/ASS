# Audio Separation Benchmark Contract

Date: 2026-05-29

This contract pins the primary evaluation target for the current edge-NPU work.
It prevents three-stem TV/CASS results, four-stem MUSDB experiments, and
paper-reported external baselines from being mixed in one ambiguous table.

## Primary Deployment Target

- Task: online causal three-stem separation for TV/edge deployment.
- Output stems, in order: `speech`, `music`, `effects`.
- Primary data family: DnR-style cinematic audio separation.
- Input/output sample rate: `44100` Hz unless a recipe explicitly says otherwise.
- Default STFT contract: host-side STFT/iSTFT, `n_fft=2048`, `hop_length=512`.
- Exported NPU core input: packed real/imag complex STFT tensor.
- Streaming mode: causal; deployment verification should use `frames=1` unless
  the candidate is explicitly chunk-causal rather than frame-streaming.
- Mixture consistency: report whether it is trained as a loss, applied as a
  post-processing projection, both, or neither.
- Residual-source ablations may export fewer explicit NPU masks, but they must
  still return the fixed output order above.  For example, the 2-mask residual-SFX
  BandSFC RT+ recipe exports Speech/Music only and reconstructs `effects` as
  `mixture - speech - music` outside the core.
- Silence behavior: inactive stems should not hallucinate persistent energy;
  report silent-source penalty settings and inactive-stem metrics when available.

## Secondary Tracks

- MUSDB18-HQ four-stem music separation remains a quality/generalization track.
- Speech-only corpora can be used for ablations, but they do not replace the
  primary `speech`, `music`, `effects` contract.
- Prompted/unified models must still report a fixed three-stem run using the
  output order above before being compared as product candidates.

## Metric Contract

Primary local metrics:

- DnR: average SNR and SI-SDR for `speech`, `music`, and `effects`.
- MUSDB18-HQ: average cSDR/uSDR plus per-stem SDR for music experiments.
- Streaming consistency: max/mean numerical diff between chunk/full or
  sequence/cell execution where the architecture exposes both paths.
- Deployment: parameters, GMAC/s, fp16/fp32 state KiB, ONNX node count, forbidden
  op count, ONE import/optimize/quantize status, and `circle-verify` when run.

Subjective listening categories:

- Bass preservation.
- Drums/transient sharpness.
- Speech intelligibility.
- Speech leakage into music/effects.
- Music leakage into speech/effects.
- Effects leakage into speech/music.
- Musical noise or tonal warble.
- Pre-echo or smearing.
- Chunk-boundary artifacts.

## Result Files

Use these templates for local measurements:

- `docs/templates/audio_separation_results_manifest.csv`
- `docs/templates/audio_separation_listening_notes.csv`

Rules:

- Use `measurement_type=local` only for numbers produced in this repo with a
  known checkpoint, recipe, and command.
- Use `measurement_type=paper` for literature baselines and include the source
  citation or paper note in `source`.
- Leave unavailable metrics blank instead of copying numbers across datasets.
- Keep DnR three-stem and MUSDB four-stem rows separate.
- For residual-source ablations, keep `n_src=3` in the result row and describe
  `core_n_src=2` plus the residual policy in `variant`, `mixture_consistency`, or
  `notes`.

## Minimum Report Row For A Candidate

A deployable candidate is not considered validated until the result manifest has
one row with:

- `task=dnr_three_stem`.
- `checkpoint` set to the evaluated checkpoint path or artifact ID.
- `dnr_snr_avg` and `dnr_si_sdr_avg`, or an explicit reason in `notes`.
- `params`, `gmac_per_s`, `state_fp16_kib`, and `onnx_nodes`.
- `one_import`, `one_optimize`, and `one_quantize` marked `PASS` or `FAIL`.
- A matching `listening_sheet` ID in the listening-notes CSV.
