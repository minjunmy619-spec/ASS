# Quality Comparison Report — Band-SCNet-NPU

This document tracks SDR quality targets and (once trained) comparison results
for Band-SCNet-NPU against published baselines and existing in-repo models.

## Target Metrics

| Model | Dataset | SDR (all-stems avg) | Notes |
|-------|---------|--------------------:|-------|
| Band-SCNet (Interspeech 2025) | MUSDB18-HQ | 7.79 dB | Non-causal, 2.59M params, 92ms latency |
| Online SFC soft-band-query rt192k | DnR | TBD | Causal, ~120K params (untrained) |
| BandSCNetNPU rt192k | DnR | TBD | Causal, ~62K params, 191 KiB state |
| BandSCNetNPU rt192k_plus | DnR | TBD | Causal, ~73K params, 178 KiB state |
| BandSCNetNPU rt192k_plus (freq-preprocessed) | DnR | TBD | fp768keep512 variant |

## Evaluation Protocol

1. **Dataset**: DnR (Divide and Remaster) — 3-stem: Speech / Music / Effects.
2. **Metric**: SI-SDR improvement (dB), averaged across stems.
3. **Streaming mode**: frame-by-frame `forward_stream` (T=1) to verify
   causal consistency with training-mode `forward`.
4. **Baseline comparison**: use the same loss (ThresSNRLoss) and training
   schedule (150 epochs, AdamW 1e-3, WarmUpStepLR) across all candidates.

## How to Run (once checkpoints exist)

```bash
# Streaming evaluation on DnR validation set
./recipes/dnr/scripts/streaming_eval.sh \
    recipes/dnr/models/band-scnet-npu.rt192k 8 cuda

# SDR computation
python recipes/dnr/scripts/evaluate_sdr.py \
    --model-path recipes/dnr/models/band-scnet-npu.rt192k \
    --data-path recipes/dnr/hdf5/cv_unsegmented.hdf5
```

## Expected Ranking (hypothesis)

Given the 192 KiB DSP state constraint, we expect:

1. **rt192k_plus (freq-preprocessed)** — highest capacity per frame (C=56)
   plus the frequency-domain pre-filter allows more bins to be spent on
   the musically-important low-frequency range.
2. **rt192k_plus** — same wider separator, no frequency preprocessing.
3. **rt192k** — narrower but deeper (3 stages vs 2).
4. **edge_small** — minimal model for pipeline validation only.

If the DSP quota is ever relaxed beyond 192 KiB, raising `channels` to 64+
and `num_stages` to 4+ should close the gap toward Band-SCNet's 7.79 dB.

## Results (to be filled after training)

| Preset | Epochs | SI-SDRi (Speech) | SI-SDRi (Music) | SI-SDRi (Effects) | Avg |
|--------|-------:|-----------------:|----------------:|------------------:|----:|
| edge_small | — | — | — | — | — |
| rt192k | — | — | — | — | — |
| rt192k_plus | — | — | — | — | — |
| rt192k_plus (fp768keep512) | — | — | — | — | — |

---

*This file will be updated once training completes. See
`.kiro/specs/band-scnet-npu/tasks.md` Phase 12.3 for the tracking entry.*
