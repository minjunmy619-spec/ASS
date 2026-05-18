# USS Leakage Investigation: `modified_deft_uss_lite_6s_unified_all_features`

## Symptom

The unified all-features USS recipe trains without numerical issues, but
manual listening of the separated foreground stems reveals **audible
interference signals and partial content from other foreground stems** in the
matched-class slots. Some separated stems contain mixtures of multiple sources
rather than a clean target.

This document summarizes a deep trace of the model architecture, loss function,
training pipeline, and evaluation pipeline that identified the root causes,
ranked by severity and impact, with concrete fixes for each.

## Scope of investigation

Files and modules examined:

- Config: [`config/separation/modified_deft_uss_lite_6s_unified_all_features.yaml`](../config/separation/modified_deft_uss_lite_6s_unified_all_features.yaml)
- Model: [`src/models/deft/unified_uss.py`](../src/models/deft/unified_uss.py),
  [`src/models/deft/modified_deft_bf16_safe.py`](../src/models/deft/modified_deft_bf16_safe.py),
  [`src/models/deft/modified_deft.py`](../src/models/deft/modified_deft.py),
  [`src/models/deft/modified_deft_semantic_bridge.py`](../src/models/deft/modified_deft_semantic_bridge.py),
  [`src/models/deft/foa_spatial_features.py`](../src/models/deft/foa_spatial_features.py),
  [`src/models/deft/spatial_heads.py`](../src/models/deft/spatial_heads.py)
- Loss: [`src/training/loss/uss_loss.py`](../src/training/loss/uss_loss.py),
  [`src/training/loss/uss_bridge_loss.py`](../src/training/loss/uss_bridge_loss.py),
  [`src/training/loss/uss_residual_loss.py`](../src/training/loss/uss_residual_loss.py),
  [`src/training/loss/class_aware_pit.py`](../src/training/loss/class_aware_pit.py)
- Lightning module: [`src/training/lightningmodule/uss_bridge.py`](../src/training/lightningmodule/uss_bridge.py)
- Datamodule: [`src/datamodules/dataset.py`](../src/datamodules/dataset.py),
  [`src/datamodules/uss_dataset.py`](../src/datamodules/uss_dataset.py)
- Evaluation: [`src/evaluation/evaluate.py`](../src/evaluation/evaluate.py),
  [`src/evaluation/evaluate_stage.py`](../src/evaluation/evaluate_stage.py),
  [`src/evaluation/metrics/s5capi_metric.py`](../src/evaluation/metrics/s5capi_metric.py),
  [`src/evaluation/metrics/s5_validation_breakdown.py`](../src/evaluation/metrics/s5_validation_breakdown.py)
- Synthesis: SpAudSyn (external, https://github.com/nttcslab/SpAudSyn)
  via [`src/datamodules/dataset.py::_get_item_generate`](../src/datamodules/dataset.py)
- Reference baseline: [`config/separation/resunetk_capisdr.yaml`](../config/separation/resunetk_capisdr.yaml)

## Severity classification

| Severity | Meaning |
|---|---|
| **Critical** | Direct cause of the reported leakage. Likely a major contributor to perceived audio quality. Fix has high impact. |
| **High** | Substantial contributor to leakage or training inefficiency. Fix has clear benefit. |
| **Medium** | Suboptimal but not the dominant cause. Fix gives incremental improvement. |
| **Low** | Cleanup or design concern. Fix is more about code health than audio quality. |

---

## Issue Index

1. [Critical: Spatial conditioning train/eval mismatch](#1-critical-spatial-conditioning-traineval-mismatch)
2. [Critical: Plain sa-SDR converges to Wiener-floor leakage](#2-critical-plain-sa-sdr-converges-to-wiener-floor-leakage)
3. [Critical: Wet/dry mismatch in mixture-consistency loss](#3-critical-wetdry-mismatch-in-mixture-consistency-loss)
4. [High: Non-foreground supervision is one hundred times too weak](#4-high-non-foreground-supervision-is-one-hundred-times-too-weak)
5. [High: Inactive foreground slots become "junk drawers"](#5-high-inactive-foreground-slots-become-junk-drawers)
6. [High: Independent sigmoid masks allow slot energy overlap](#6-high-independent-sigmoid-masks-allow-slot-energy-overlap)
7. [Medium: Same-class duplicate disambiguation is undertrained](#7-medium-same-class-duplicate-disambiguation-is-undertrained)
8. [Medium: Two redundant DoA heads under shared supervision](#8-medium-two-redundant-doa-heads-under-shared-supervision)
9. [Medium: Dead `_residual_consistency_loss` clutter](#9-medium-dead-_residual_consistency_loss-clutter)
10. [Low: Naming collision on `doa_vector` input vs output keys](#10-low-naming-collision-on-doa_vector-input-vs-output-keys)

---

## 1. Critical: Spatial conditioning train/eval mismatch

### What is happening

[`uss_bridge.USSBridgeLightning._get_input_dict`](../src/training/lightningmodule/uss_bridge.py)
forwards an oracle DoA vector (built from `event_position` metadata) to the
model under the key `spatial_vector`. The model
[`UnifiedModifiedDeFTUSS`](../src/models/deft/unified_uss.py)
receives it via `_choose_spatial_condition`
([`modified_deft_semantic_bridge.py`](../src/models/deft/modified_deft_semantic_bridge.py))
and FiLM-modulates every TF feature with `(1 + γ(spatial_vec)) · x + β(spatial_vec)`
at full conditioning scale (`spatial_conditioning_scale: 1.0`).

The `predicted_spatial_prob` schedule in
[`spatial_conditioning_curriculum`](../src/training/callbacks/spatial_conditioning_curriculum.py)
warms from `0.25` to `1.0` over `warmup=25` and `anneal=150` epochs,
i.e. oracle DoA dominates conditioning for ~70% of training.

At evaluation time
([`evaluate_stage._evaluate_uss_batch`](../src/evaluation/evaluate_stage.py),
[`Kwon2025S5.predict_label_separate`](../src/models/s5/kwo2025.py)), the
forward call is `self.model({"mixture": mixture})` with no `spatial_vector`,
so the model falls back to `pred_doa_vector` from `doa_head`. That head is
weakly supervised (`lambda_doa: 0.05`, `lambda_bridge_doa: 0.05`), and its
outputs are not slot-permutation-consistent.

### Why this causes leakage

During training the model exploits the oracle DoA as a slot-identity shortcut:
"slot i is supposed to listen toward direction θ_i". The mask path becomes
dependent on FiLM modulation rather than learning to disambiguate slots from
the AIV/IPD spatial features in the encoder. At evaluation, the FiLM input
becomes noisy and slot-inconsistent, so the masks revert to a poorly
disambiguated state and energy from neighboring directions leaks into the
matched slot.

### Severity rationale

This is the dominant single cause of train/eval distribution shift in the
recipe. The encoder already receives strong mixture-derived spatial features
(AIV/IPD per TF bin) via
[`FOASpatialFeatureEncoder`](../src/models/deft/foa_spatial_features.py),
so disabling the FiLM oracle pathway loses no information; it only removes
the shortcut.

### Fix options

**Recommended: disable FiLM, keep DoA heads as auxiliaries**

Config in `model.args`:

```yaml
use_spatial_conditioning: false
use_doa_head: true
predicted_spatial_prob: 1.0
spatial_mix_fallback_prob: 0.0
```

Remove the `spatial_conditioning_curriculum` callback from `train.callbacks`.

In `loss.args`:

```yaml
lambda_doa: 0.1
lambda_bridge_doa: 0.1
```

**Alternative: always-predicted-with-detach**

Keep FiLM active but remove the oracle dependence:

```yaml
use_spatial_conditioning: true
predicted_spatial_prob: 1.0
detach_predicted_spatial_for_condition: true
spatial_conditioning_scale: 0.3
lambda_doa: 0.3
lambda_bridge_doa: 0.3
```

**Alternative: brief warmup with oracle, then predicted**

Compress curriculum to ~3-8 epochs before snapping to predicted-only:

```yaml
warmup_epochs: 3
anneal_epochs: 5
start_predicted_spatial_prob: 0.5
end_predicted_spatial_prob: 1.0
```

### Expected effects

- Eliminates train/eval distribution shift on the mask path.
- Loses no spatial information (AIV/IPD remain in the encoder).
- DoA heads still train as auxiliary tasks, useful for downstream TSE bridge.
- Should produce noticeably cleaner matched-slot audio within 20-30 epochs of
  fine-tuning.

### Diagnostic before retraining

Modify `_evaluate_uss_batch` to feed oracle DoA at eval and compare:

```python
input_dict = {"mixture": mixture}
if "foreground_doa" in batch:
    input_dict["spatial_vector"] = batch["foreground_doa"].to(self.device)
output = self.model(input_dict)
```

If feeding oracle DoA at eval makes matched-slot audio sound clearly cleaner,
the spatial mismatch hypothesis is confirmed and the recommended fix will help.

### Files affected

- `config/separation/<your_config>.yaml`: `model.args`, remove curriculum callback
- No code changes required

---

## 2. Critical: Plain sa-SDR converges to Wiener-floor leakage

### What is happening

[`pairwise_sa_sdr_loss`](../src/training/loss/class_aware_pit.py) computes:

```
sa-SDR = 10 log10(||target||² / ||pred − target||²)
```

This is **not** SI-SDR (no projection step) and **not** SDRi (no mixture
anchor). The loss optimum at any TF bin is the Wiener filter:

```
mask* = ||target||² / (||target||² + ||interference||²)
```

For a 6 dB SNR bin this is `mask* ≈ 0.8`, so the prediction contains
**40% of the original interference amplitude** at that bin. This is the
sa-SDR loss optimum, not a sub-optimum.

[`uss_loss.get_loss_func`](../src/training/loss/uss_loss.py) supports three
foreground assignment strategies:

- `global_pit` (default): plain sa-SDR + class CE tiebreaker
- `soft_capi`: same gradient as sa-SDR but adds class-confidence filter that
  marks low-confidence pairs invalid (penalty `invalid_class_cost`)
- `hard_capi`: like `soft_capi` but excludes invalid pairs entirely

The `capi_use_sdri` flag toggles between sa-SDR (False) and sa-SDRi (True) in
the pairwise cost. **`capi_use_sdri: false` is equivalent to `global_pit`
plus the class-confidence filter** — it does not change the gradient or the
local-minimum structure.

Current post-warmup config uses `soft_capi` with `capi_use_sdri: false`, so
the model is still in the sa-SDR Wiener-floor regime.

### Why this causes leakage

The sa-SDR gradient *can* push leakage out of a matched slot, but its
strength is proportional to the leakage level:

```
∂(sa-SDR loss)/∂(pred) ∝ (pred − target)
```

When leakage drops below the Wiener-floor level, the gradient drops to zero.
The system converges with non-trivial residual leakage that sa-SDR
**cannot detect as suboptimal**, because mixture-like outputs achieve a
genuinely good sa-SDR (e.g., 5-8 dB).

### How SDRi changes this

```
sa-SDRi = sa-SDR(pred, target) − sa-SDR(mixture, target)
```

The gradient is identical (mixture term doesn't depend on pred). But the
**values** differ: a mixture-like prediction scores `sa-SDRi ≈ 0` instead
of `sa-SDR ≈ 5-8 dB`. Combined with the class-confidence filter, mixture-like
slots get marked invalid, the `invalid_class_cost = 20` penalty pushes them
to "unmatched" status, and `lambda_inactive_foreground` then suppresses them.
This breaks the Wiener-floor equilibrium.

### Severity rationale

This is the single biggest perceptual-quality lever short of architectural
changes. SDRi PIT is a recognized fix for "model just attenuates the mixture"
behavior in source separation literature.

### Fix options

**Recommended: enable SDRi flip with safety knobs**

In `loss.args`:

```yaml
foreground_assignment: soft_capi
capi_use_sdri: true                    # ← THE flip
capi_ref_channel: 0
capi_confidence_threshold: 0.25        # was: 0.35; gentler during transition
capi_invalid_class_cost: 10.0          # was: 20.0; less brittle
lambda_class_pit: 0.3                  # was: 0.1; SDRi values are smaller, scale CE up
```

Apply only after warmup phase, when the model has reached at least
~3-5 dB sa-SDR; otherwise SDRi values are noisy and PIT thrashes.

**Alternative: hard_capi**

```yaml
foreground_assignment: hard_capi
capi_use_sdri: true
```

Stricter version that excludes invalid pairs entirely. More aggressive but
less stable; use only if `soft_capi` is insufficient.

**Conservative: keep sa-SDR but add stronger inactive suppression**

Skip the SDRi flip but combine items 4 and 5 below to make leakage costlier
through different mechanisms. Less effective than SDRi.

### Expected effects

- Mixture-like slot outputs become detectable as suboptimal.
- The Wiener-floor local minimum is destabilized.
- Should produce a clear improvement in matched-slot purity within 5-10
  epochs of fine-tuning.

### Monitoring during transition

Watch `epoch_train/loss_matched_valid_pair_rate`:
- Should briefly drop from ~0.95 to ~0.6 as the filter excludes mixture-like slots.
- Should climb back to ≥ 0.8 by epoch 10-15.
- If it stays below 0.4, lower `capi_confidence_threshold` to 0.15 or
  `capi_invalid_class_cost` to 5.0.

### Files affected

- `config/separation/<your_config>.yaml`: `loss.args`
- No code changes required

---

## 3. Critical: Wet/dry mismatch in mixture-consistency loss

### What is happening

The training data contract is:

| Source | In `mixture` | As training target |
|---|---|---|
| Foreground events | wet (full RIR) | **dry** (direct-path RIR ~6-50 ms window) |
| Interference events | wet (full RIR) | **dry** (direct-path RIR) |
| Background noise | wet (room recording) | wet |

This is determined by the SpAudSyn synthesizer
([`_synthesize_one_event`](https://github.com/nttcslab/SpAudSyn/blob/main/src/spatial_audio_synthesizer.py))
combined with `fg_return: {dry: true}`, `int_return: {dry: true}`,
`bg_return: {waveform: true}` in your config.

Two losses currently try to enforce mixture consistency:

- [`_mix_loss`](../src/training/loss/uss_residual_loss.py): asks
  `sum(predictions including residual_waveform) ≈ wet_mixture`.
  Active in your config with `lambda_mix: 0.05`.
- [`_residual_loss`](../src/training/loss/uss_residual_loss.py): asks the
  residual slot to match `wet_mixture − sum(dry_targets) − wet_noise`.
  Active in your config with `lambda_residual_slot: 0.05`.
  This residual reference is **the sum of reverb tails** of all foreground
  and interference events — physically meaningful but hard for one masked
  slot to predict.
- [`_residual_consistency_loss`](../src/training/loss/uss_loss.py) (note:
  different function): MSE form of `_mix_loss` without residual; gated by
  `lambda_residual` defaulting to 0. **Dead in your config.**

### Why this is broken

Sa-SDR PIT pulls foreground predictions toward dry targets (no reverb).
Mix loss pulls the predicted sum toward wet mixture (with reverb).
With dry foreground predictions and one residual slot, the equation
`sum(dry_predictions) + residual_pred ≈ wet_mixture` is satisfied only
when residual_pred captures the full reverb tail energy of all events,
which is too much for a single mask slot to absorb cleanly.

In practice the residual slot under-predicts, mix loss is unsatisfied,
and the gradient pulls foreground predictions to be slightly wet to close
the budget. This **degrades CAPI-SDRi** because the metric compares against
dry stems.

### Why `capi_use_sdri: true` cannot fix this

CAPI-SDRi at evaluation time computes:

```
SDR(pred, dry_ref) − SDR(wet_mixture, dry_ref)
```

So the metric requires dry predictions. Wet predictions are bounded above
by the dry-vs-wet energy gap, regardless of separation quality.

### Severity rationale

The current `lambda_mix=0.05` and `lambda_residual_slot=0.05` are too small
to actually distort training, so the impact today is limited. But raising
them to make mixture consistency meaningful (a recommended fix elsewhere)
would actively hurt CAPI-SDRi unless the wet/dry mismatch is corrected first.
This is therefore a **prerequisite** for several other improvements.

### Fix options

**Recommended: redefine `_mix_loss` to compare against dry-domain target**

Add a `target_mode` parameter to `_mix_loss` and pass `"dry"` from the
bridge factory:

```python
# src/training/loss/uss_residual_loss.py
def _mix_loss(output, target, ref_channel, target_mode="wet"):
    if target_mode == "dry":
        ref = sum(_sum_src(target[k], ref_channel)
                  for k in ("foreground_waveform", "interference_waveform", "noise_waveform")
                  if k in target).detach()
        pred_keys = ("foreground_waveform", "interference_waveform", "noise_waveform")
    elif target_mode == "wet":
        ref = _mix(target["mixture"], ref_channel)
        pred_keys = ("foreground_waveform", "interference_waveform", "noise_waveform", "residual_waveform")
    else:
        raise ValueError(...)
    recon = sum(_sum_src(output[k], ref_channel) for k in pred_keys if k in output)
    return F.l1_loss(recon.float(), ref.float())
```

Plumb `mix_loss_target_mode` through `uss_bridge_loss.get_loss_func` and add
to your YAML:

```yaml
mix_loss_target_mode: dry          # NEW (requires the code patch above)
lambda_mix: 0.3                    # was: 0.05; now meaningful
lambda_residual_slot: 0.0          # was: 0.05; residual slot has no role with dry mix
```

If you also drop the residual slot architecturally:

```yaml
# in model.args
enable_residual_slots: false
n_residual: 0
```

**Alternative: disable mixture-consistency entirely (match official baseline)**

```yaml
lambda_mix: 0.0
lambda_residual_slot: 0.0
lambda_residual: 0.0
enable_residual_slots: false
n_residual: 0
```

This matches `resunetk_capisdr.yaml`'s philosophy: trust PIT sa-SDR alone.
Loses mixture-consistency regularization but eliminates the wet/dry conflict.

**Alternative: switch dataset to wet targets (NOT recommended)**

```yaml
fg_return: {wet: true, ...}
int_return: {wet: true, ...}
```

Mathematically clean but **breaks CAPI-SDRi evaluation** (which expects dry
predictions) and breaks downstream TSE which expects dry sources.

### Expected effects

With the recommended fix:
- Mix loss becomes a meaningful regularizer (no asymmetric energy gap).
- `lambda_mix: 0.3` provides energy-budget supervision that prevents
  cross-source double-counting.
- Foreground predictions stay dry; CAPI-SDRi is preserved.
- Removes the implicit "absorb all reverb tails" demand on the residual slot.

### Verification logging

Add to validation:

```python
pred_sum = output["foreground_waveform"][:, :, 0].sum(dim=1) \
         + output["interference_waveform"][:, :, 0].sum(dim=1) \
         + output["noise_waveform"][:, 0, 0]
dry_sum  = target["foreground_waveform"][:, :, 0].sum(dim=1) \
         + target["interference_waveform"][:, :, 0].sum(dim=1) \
         + target["noise_waveform"][:, 0, 0]
wet_mix  = target["mixture"][:, 0]
self.log("epoch_valid/dry_mix_l1", (pred_sum - dry_sum).abs().mean())
self.log("epoch_valid/dry_wet_gap", (dry_sum - wet_mix).abs().mean())
```

After 10 epochs, `dry_mix_l1` should be < 0.01 (predictions close the dry
budget). `dry_wet_gap` is roughly constant (the reverb tail energy).

### Files affected

- `src/training/loss/uss_residual_loss.py` (modify `_mix_loss`)
- `src/training/loss/uss_bridge_loss.py` (plumb new parameter)
- `config/separation/<your_config>.yaml`

---

## 4. High: Non-foreground supervision is one hundred times too weak

### What is happening

In [`uss_loss.py`](../src/training/loss/uss_loss.py):

```python
loss = (
    loss_fg
    + lambda_non_foreground * (loss_int + loss_noise)  # 0.01 in your config
    + ...
)
```

So interference and noise reconstruction losses get weight `0.01` relative to
foreground (weight `1.0`). The interference slots and noise slot are
essentially unsupervised at typical training scales.

### Why this causes leakage

A non-foreground slot needs **two** kinds of force:
- An attractor pulling its target signal *into* it
- A penalty pushing other content *out* of it

`lambda_non_foreground = 0.01` makes the attractor 100× weaker than the
foreground attractor. Combined with `lambda_inactive_interference = 0.01`,
unmatched interference slots have neither strong "be silent" nor strong
"capture interference" forces.

The interference energy in the mixture has no strong destination slot.
Non-foreground slots remain near-zero (because the inactive penalty, even
at 0.01, dominates the unsupervised signal). The interference energy is
absorbed into foreground slots instead, because those slots have stronger
gradients and the foreground sa-SDR resistance is weak (Wiener floor).

### Severity rationale

This is a major contributor to interference leakage specifically. The fix is
config-only and stable. Not as fundamental as items 1-3 but compounds with
them.

### Fix options

**Recommended:**

```yaml
lambda_non_foreground: 0.2          # was: 0.01
```

This gives interference and noise reconstruction 20% the weight of foreground
sa-SDR. Strong enough to make non-foreground slots a real attractor without
overwhelming the foreground objective.

**Conservative: 0.1**

If 0.2 destabilizes early training, try 0.1. Less effective but safer.

**Aggressive: 0.5**

If 0.2 doesn't fully resolve interference leakage, try 0.5. Watch foreground
quality — at this weight non-foreground losses begin to compete meaningfully
with foreground separation.

### Expected effects

- Interference slots actively capture interference content.
- Foreground slots see less interference leakage.
- Synergistic with item 5 (inactive suppression): inactive int/noise slots
  also become better-suppressed.

### Files affected

- `config/separation/<your_config>.yaml`: `loss.args`

---

## 5. High: Inactive foreground slots become "junk drawers"

### What is happening

[`uss_loss.py`](../src/training/loss/uss_loss.py) `inactive_source_energy_loss`:

```python
def inactive_source_energy_loss(waveform, inactive_mask):
    energy = source_energy_loss(waveform)  # mean(waveform²) per slot
    return (energy * inactive_mask.float()).sum() / inactive_mask.float().sum().clamp_min(1.0)
```

Multiplied by `lambda_inactive_foreground: 0.2`. So an unmatched foreground
slot with mask amplitude 0.3 incurs penalty `0.2 × 0.3² = 0.018`. Small.

Three subtle forces pull energy *into* unmatched fg slots:

1. The shared encoder + slot-specific 1×1 conv (`object_conv`) cannot easily
   learn slot-specific suppression, because slot identity (which is unmatched)
   varies example-to-example.
2. PIT assignment can be unstable: a slot with partial leakage may end up
   matched to a target it partially contains, freeing some other slot to be
   "unmatched" and pushing the suppression onto a possibly cleaner slot.
3. Mix loss (when `lambda_mix > 0`) creates pressure for energy conservation,
   which can flow into unmatched slots if other slots can't absorb all the
   mixture energy.

### Why this causes leakage

Unmatched fg slots end up containing **low-amplitude broadband content**
(typically 5-20% of mixture amplitude) — leaked interference, room noise,
weak portions of foreground. At evaluation time, these slots are correctly
labeled as silence by the PIT-oracle assignment in
[`_pit_oracle_labels`](../src/evaluation/evaluate_stage.py) and skipped during
file output (`if label == "silence": continue`), so they do **not** appear in
the listened audio for oracle-label evaluation.

**However**, this issue still matters because:
- The shared encoder is trained to produce these "junk drawer" features for
  all slots, which leaks into matched slots' representations through the
  shared parameters.
- Without oracle labels, predicted-class evaluation will route some junk
  drawer outputs to matched class labels.

### Severity rationale

Significant for non-oracle evaluation, indirect contribution to matched-slot
quality even with oracle eval. Easy to fix.

### Fix options

**Recommended:**

```yaml
lambda_inactive_foreground: 1.0     # was: 0.2
lambda_inactive_interference: 0.1   # was: 0.01
lambda_inactive_noise: 0.1          # was: 0.01
```

At `1.0`, unmatched slot energy is penalized 5× more strongly. Combined with
`lambda_non_foreground: 0.2` (item 4), int and noise inactive penalties also
become meaningful.

**Aggressive: 2.0 for foreground**

If matched-slot quality is still affected, try `lambda_inactive_foreground: 2.0`.

### Expected effects

- Unmatched foreground slots converge to true-zero output.
- Shared encoder representations less polluted by "junk drawer" content.
- Some improvement in matched-slot purity through the shared encoder path.

### Eval-time silence gating diagnostic

If listening to raw `foreground_waveform` (without `_pit_oracle_labels`
filtering), gate the output at eval:

```python
silence_prob = torch.sigmoid(output["silence_logits"])  # [B, n_fg]
active_mask = (silence_prob > 0.5).float()
gated_fg = output["foreground_waveform"] * active_mask[:, :, None, None]
```

This is what
[`Kwon2025S5._force_silent_slots`](../src/models/s5/kwo2025.py) does
internally. With oracle-label `evaluate_stage`, silence gating is implicit
through PIT-oracle assignment.

### Files affected

- `config/separation/<your_config>.yaml`: `loss.args`

---

## 6. High: Independent sigmoid masks allow slot energy overlap

### What is happening

In [`modified_deft_bf16_safe.py::_spatial_mask_to_waveform`](../src/models/deft/modified_deft_bf16_safe.py):

```python
mask_mag = torch.sigmoid(mask[:, :, :, 0])  # per-slot, INDEPENDENT
```

Each of the 7 slots predicts its own sigmoid mask in [0, 1] without any
constraint that masks across slots sum to ≤ 1 at any given TF bin. The
optimal sa-SDR mask for each slot at a TF bin is the local Wiener filter
*independently*. At a TF bin where two foreground sources overlap, both
slots' masks can be 0.5-0.8, and **both slots grab most of the bin energy**.

### Why this causes leakage

This is the architectural enabler of item 2. Even if sa-SDR were a perfect
loss, independent sigmoid masks would still allow slot-overlap leakage at
the architectural level.

### Severity rationale

Architectural change, more invasive than config tweaks. Best as a follow-up
after items 1-5 if leakage persists.

### Fix options

**Recommended: opt-in softmax mask flag**

Add a `mask_softmax` flag to `UnifiedModifiedDeFTUSS` and switch
`_spatial_mask_to_waveform` to softmax over slots when enabled:

```python
mag_logits = mask[:, :, :, 0]
if getattr(self, "mask_softmax", False):
    temperature = float(getattr(self, "mask_softmax_temperature", 1.0))
    mask_mag = F.softmax(mag_logits / max(temperature, 1e-3), dim=1)
else:
    mask_mag = torch.sigmoid(mag_logits)
```

Wire `mask_softmax: bool = False` and `mask_softmax_temperature: float = 1.0`
through the model's `__init__` and store as instance attributes.

In your YAML:

```yaml
mask_softmax: true
mask_softmax_temperature: 1.0
```

**Required companion changes when enabling softmax:**
- Item 3 fix (dry-domain mix loss) is **mandatory**: softmax assumes
  `Σ_slots(mask) = 1` per TF bin, which requires wet-target consistency or
  an equivalent dry-target reformulation.
- Update `_residual_consistency_loss` to include `residual_waveform` if you
  keep residual slots.

**Alternative: hybrid (sigmoid magnitude × softmax routing)**

Predict both a sigmoid magnitude and a softmax routing weight per slot, then
multiply. Keeps some "mass not assigned" capacity while still constraining
inter-slot competition. Requires more architectural changes.

### Expected effects

- Slot energy at each TF bin partitions into a fixed budget summing to 1.
- Cross-source leakage at overlapping TF bins becomes architecturally
  impossible (in the limit).
- Slower convergence in early epochs because slots start at uniform 1/7.
- Risk of mode collapse where one slot dominates everywhere; mitigated by
  PIT and item 4/5 fixes.

### Caveats

- Dry-target sa-SDR with softmax masks is a research-grade configuration;
  expect 5-10 epochs of unstable training before recovery.
- Phase mask (real/imag with tanh) should NOT be softmaxed — it must remain
  per-slot independent for phase rotation.
- If you observe one slot dominating, add an entropy regularizer:
  `entropy = -(mask_mag * mask_mag.clamp_min(1e-8).log()).sum(dim=1).mean()`
  with weight ~0.001.

### Files affected

- `src/models/deft/modified_deft_bf16_safe.py` (modify `_spatial_mask_to_waveform`)
- `src/models/deft/unified_uss.py` (expose `mask_softmax` parameters)
- `src/training/loss/uss_loss.py` (update `_residual_consistency_loss` to
  include residual)
- `config/separation/<your_config>.yaml`

---

## 7. Medium: Same-class duplicate disambiguation is undertrained

### What is happening

Your config has `dupse_rate: 0.5`, so half of multi-source training scenes
contain same-class duplicates. The losses that disambiguate same-class slots
are:

- [`_spatial_diversity_loss`](../src/training/loss/uss_loss.py):
  `lambda_spatial_diversity: 0.02`
- [`_waveform_anticollapse_loss`](../src/training/loss/uss_loss.py):
  `lambda_waveform_anticollapse: 0.01`

Both operate only on same-class active pairs and use a margin-based hinge
loss. At weights 0.02 and 0.01, the gradient contribution is ~3 orders of
magnitude smaller than foreground sa-SDR.

### Why this causes leakage

For same-class duplicates, slot identity must be inferred from spatial or
waveform features. With weak disambiguation losses, two slots with the same
class label can converge to producing similar outputs — both emit a mix of
the two sources rather than separating them.

This is partially mitigated by the spatial conditioning FiLM (item 1), which
explains why turning off FiLM (item 1's recommended fix) needs item 7's
strengthening to maintain dupse case performance.

### Severity rationale

Important when item 1's fix is applied (no more FiLM shortcut). Otherwise
moderate. Easy to fix.

### Fix options

**Recommended:**

```yaml
lambda_spatial_diversity: 0.05          # was: 0.02
lambda_waveform_anticollapse: 0.05      # was: 0.01
spatial_diversity_margin: 0.15          # was: 0.2; tighter
waveform_anticollapse_margin: 0.2       # was: 0.3; tighter
```

**Aggressive: 0.1**

If dupse case CAPI-SDRi remains poor, try `0.1` for both.

### Expected effects

- Same-class slot outputs become more orthogonal in waveform and spatial-embedding domains.
- `capi_sdri_same_class_duplicate` (logged by validation breakdown metric)
  should improve.

### Files affected

- `config/separation/<your_config>.yaml`: `loss.args`

---

## 8. Medium: Two redundant DoA heads under shared supervision

### What is happening

The model produces two DoA predictions:

- `output["doa_vector"]` from
  [`ForegroundSpatialHead`](../src/models/deft/spatial_heads.py),
  computed **after** FiLM modulation. Supervised by
  [`_matched_doa_loss`](../src/training/loss/uss_loss.py) at
  `lambda_doa: 0.05`, compared against `target["foreground_doa"]`.
- `output["pred_doa_vector"]` from
  [`ObjectDoAHead`](../src/models/deft/modified_deft_semantic_bridge.py),
  computed **before** FiLM modulation. Supervised by
  [`_doa_loss`](../src/training/loss/uss_bridge_loss.py) at
  `lambda_bridge_doa: 0.05`, compared against `target["spatial_vector"]`
  (which is aliased to `target["foreground_doa"]` in your setup).

Both predict per-slot 3D unit vectors against the same oracle target with
similar architectures. Combined supervision strength is `0.05 + 0.05 = 0.1`,
8× weaker than `lambda_class_ce = 0.8`.

When FiLM is active, `pred_doa_vector` actually drives the FiLM input, so it
has architectural significance. `doa_vector` is a post-FiLM readout that's
trivially close to oracle (since FiLM injected oracle DoA).

### Why this is suboptimal

Two heads do nearly identical work. With FiLM disabled (item 1's recommended
fix), neither head feeds back into the main separator; both become pure
auxiliary tasks. The redundancy doesn't break anything but wastes parameters
and gradient signal.

### Severity rationale

Cleanup, not a leakage source. Low audio impact but improves code clarity.

### Fix options

**Recommended (with item 1's FiLM disable): mild bump and accept redundancy**

```yaml
lambda_doa: 0.1           # was: 0.05
lambda_bridge_doa: 0.1    # was: 0.05
```

Keep both heads; downstream TSE bridge consumes both outputs.

**Cleanup option: unify**

Edit `spatial_heads.py` to drop the `doa` linear from `ForegroundSpatialHead`,
keeping only `spatial_embedding`. Remove `output["doa_vector"]` from
`unified_uss._add_optional_heads`. Remove `_matched_doa_loss` from
`uss_loss.py`. About 40 lines of code changes; one DoA loss remains
(`lambda_bridge_doa`).

### Expected effects

- No direct leakage impact.
- Cleaner architecture if cleanup option is taken.
- Slightly stronger DoA supervision overall (helps downstream TSE).

### Files affected

- `config/separation/<your_config>.yaml`: `loss.args`
- (Optional) `src/models/deft/spatial_heads.py`,
  `src/models/deft/unified_uss.py`,
  `src/training/loss/uss_loss.py`

---

## 9. Medium: Dead `_residual_consistency_loss` clutter

### What is happening

[`uss_loss._residual_consistency_loss`](../src/training/loss/uss_loss.py) is
a near-duplicate of `_mix_loss` (MSE form, omits residual_waveform from
`recon`). It's gated by `lambda_residual` defaulting to `0.0` and is not
set in any current config. The function is computed every step and logged,
but contributes zero gradient.

### Why this is suboptimal

- Confusing: two near-identical losses with similar names but different
  inclusion rules for `residual_waveform`.
- `_residual_consistency_loss` does NOT include `residual_waveform` in
  `recon`, so it's actually inconsistent with the architecture as-shipped
  (which produces residual waveform when `enable_residual_slots: true`).
- Adds noise to logged metrics; makes loss interpretation harder.

### Severity rationale

Pure code cleanup. No audio impact.

### Fix options

**Recommended: remove from `uss_loss.py`**

Delete `_residual_consistency_loss` and its `lambda_residual` parameter.
`_mix_loss` (with item 3's `target_mode` parameter) is the canonical
mixture-consistency mechanism.

**Alternative: leave it for backward compatibility**

Some older configs may depend on it. If keeping it, add an inclusion of
`residual_waveform` to make it consistent with the architecture:

```python
recon = output["foreground_waveform"][:, :, 0].sum(dim=1) + ...
if "residual_waveform" in output:
    recon = recon + output["residual_waveform"][:, :, 0].sum(dim=1)
```

### Expected effects

- No audio impact.
- Cleaner loss code, fewer logged metrics.

### Files affected

- `src/training/loss/uss_loss.py`

---

## 10. Low: Naming collision on `doa_vector` input vs output keys

### What is happening

The key `doa_vector` is used for two unrelated things:

- **Input**: third fallback in
  [`_get_oracle_spatial_vector`](../src/models/deft/modified_deft_semantic_bridge.py)
  for oracle spatial conditioning. Lightning module passes it through if
  present in batch ([`uss_bridge.py`](../src/training/lightningmodule/uss_bridge.py)).
  Never set by the dataset in your run.
- **Output**: `output["doa_vector"]` from `spatial_head`, predicted DoA
  used by `_matched_doa_loss`.

The two never collide in flow because input and output dicts are separate
Python objects. But the dual semantics is confusing when reading code.

### Severity rationale

Code clarity issue. No functional impact.

### Fix options

**Recommended (deferred): rename**

A cleaner naming would be:
- Input oracle: `oracle_spatial_vector` (or just `spatial_vector`, dropping
  the duplicate aliases)
- Output of `spatial_head`: `pred_spatial_doa_vector`
- Output of `doa_head`: `pred_bridge_doa_vector` (current `pred_doa_vector`)

Rename requires careful handling because checkpoint state-dict keys may
embed these names, and external consumers (TSE, SC) may reference them.
Best done as a coordinated cleanup pass once leakage is resolved.

**Alternative: leave as-is**

Document the dual semantics in a comment near
[`_get_oracle_spatial_vector`](../src/models/deft/modified_deft_semantic_bridge.py)
and call it a day.

### Expected effects

- No audio impact.
- Less confusing code.

### Files affected

- (Optional) Rename across model, loss, lightning module, datamodule

---

## Recommended fix bundle: synergistic combinations

Issues 1, 2, 3, 4, 5 are the **leakage chain** and should be fixed together
for compounding effect. Issues 6, 7 are second-order improvements. Issues 8,
9, 10 are cleanup.

### Bundle A: Minimum viable fix (config-only, no code changes)

Smallest change that addresses the dominant leakage causes.

```yaml
# model.args
use_spatial_conditioning: false        # ← Item 1
predicted_spatial_prob: 1.0
spatial_mix_fallback_prob: 0.0

# loss.args
foreground_assignment: soft_capi
capi_use_sdri: true                    # ← Item 2
capi_confidence_threshold: 0.25
capi_invalid_class_cost: 10.0
lambda_class_pit: 0.3
lambda_non_foreground: 0.2             # ← Item 4
lambda_inactive_foreground: 1.0        # ← Item 5
lambda_inactive_interference: 0.1      # ← Item 5
lambda_inactive_noise: 0.1             # ← Item 5
lambda_mix: 0.0                        # ← Item 3 (disable rather than fix)
lambda_residual_slot: 0.0              # ← Item 3
```

Remove `spatial_conditioning_curriculum` callback. No code changes.

**Expected impact**: addresses the four largest leakage sources. Should give
clearly cleaner audio in 20-30 epochs of fine-tuning. Loses mixture-consistency
regularization (item 3 disabled rather than fixed).

### Bundle B: Recommended fix (config + small code patch)

Bundle A plus the dry-domain mix loss patch.

Code changes:
- `src/training/loss/uss_residual_loss.py`: add `target_mode` param to `_mix_loss`
- `src/training/loss/uss_bridge_loss.py`: plumb `mix_loss_target_mode` through factory

Config additions on top of Bundle A:

```yaml
mix_loss_target_mode: dry              # ← Item 3 (recommended fix)
lambda_mix: 0.3
lambda_spatial_diversity: 0.05         # ← Item 7
lambda_waveform_anticollapse: 0.05
lambda_doa: 0.1                        # ← Item 8 (mild bump)
lambda_bridge_doa: 0.1
```

**Expected impact**: Bundle A + meaningful mixture-consistency regularizer
without wet/dry conflict + better same-class disambiguation. Strongest
recommended option.

### Bundle C: Architectural fix (Bundle B + softmax masks)

Bundle B plus item 6's softmax mask architectural change.

Code changes (in addition to Bundle B):
- `src/models/deft/modified_deft_bf16_safe.py`: softmax option in `_spatial_mask_to_waveform`
- `src/models/deft/unified_uss.py`: expose `mask_softmax`, `mask_softmax_temperature`
- `src/training/loss/uss_loss.py`: include residual in `_residual_consistency_loss`
  (or delete it per item 9)

Config additions:

```yaml
mask_softmax: true                     # ← Item 6
mask_softmax_temperature: 1.0
```

**Expected impact**: Bundle B + architectural prevention of slot energy
overlap. Highest expected ceiling but more invasive; reserve for after
Bundle B is verified working.

### Sequencing recommendation

1. Run **Bundle A** as a 30-epoch fine-tune from your current checkpoint.
   Listen, measure CAPI-SDRi breakdown.
2. If clear improvement: apply **Bundle B** as a follow-up 30-epoch fine-tune.
3. If clear improvement and you want to push further: **Bundle C** as a
   from-scratch retrain (softmax masks change initialization dynamics).
4. **Items 8, 9, 10** are cleanup; do them as a separate PR after the leakage
   chain is resolved.

---

## Diagnostic tools

### `S5ValidationBreakdownMetric` for leakage measurement

[`src/evaluation/metrics/s5_validation_breakdown.py`](../src/evaluation/metrics/s5_validation_breakdown.py)
buckets soundscapes into `zero_target`, `one_target`, `distinct_class`, and
`same_class_duplicate`, then reports CAPI-SDRi and slot statistics per
bucket. Enable with `--validation_breakdown` in
[`evaluate.py`](../src/evaluation/evaluate.py) or
[`evaluate_stage.py`](../src/evaluation/evaluate_stage.py).

Key metrics for tracking leakage fixes:

| Metric | Target trajectory after fixes |
|---|---|
| `valid/foreground_leakage_energy_*` | Should drop, especially `_one_target` bucket |
| `valid/capi_sdri_zero_target_fp_rate` | Should drop toward 0 |
| `valid/capi_sdri_one_target` | Should rise (cleanest case) |
| `valid/capi_sdri_distinct_class` | Should rise (most common multi-source case) |
| `valid/capi_sdri_same_class_duplicate` | Should rise (indicates dupse disambiguation works) |
| `valid/silence_precision` and `valid/silence_recall` | Both should approach 1.0 |

### Pre-retrain spatial-mismatch probe (item 1)

Modify `_evaluate_uss_batch` once to feed oracle DoA at eval. If oracle-DoA
listening test sounds clearly cleaner than current eval, item 1's fix is
high-impact for your specific checkpoint.

### Mix-loss verification logging (item 3)

After applying Bundle B, log `epoch_valid/dry_mix_l1` and
`epoch_valid/dry_wet_gap` per the snippet in item 3. Confirms the dry-domain
mix loss is converging.

### Lambda monitoring during SDRi transition (item 2)

Watch `epoch_train/loss_matched_valid_pair_rate`. Healthy trajectory: dip
from ~0.95 to ~0.6, recover to ≥ 0.8 by epoch 10-15.

---

## Summary table

| # | Issue | Severity | Files affected (config / code) | Bundle |
|---|---|---|---|---|
| 1 | Spatial conditioning train/eval mismatch | Critical | config only | A, B, C |
| 2 | Plain sa-SDR Wiener-floor leakage | Critical | config only | A, B, C |
| 3 | Wet/dry mismatch in mix loss | Critical | config + 2 files | B (fixed), A (disabled) |
| 4 | Non-foreground supervision too weak | High | config only | A, B, C |
| 5 | Inactive slot junk drawer | High | config only | A, B, C |
| 6 | Independent sigmoid masks | High | config + 3 files | C |
| 7 | Same-class disambiguation undertrained | Medium | config only | B, C |
| 8 | Redundant DoA heads | Medium | config (or rename code) | B (mild bump) |
| 9 | Dead `_residual_consistency_loss` | Medium | code cleanup | (optional) |
| 10 | `doa_vector` naming collision | Low | code rename | (deferred) |

## Open questions for the user

Before applying any bundle, please confirm:

1. **Are you fine-tuning from an existing checkpoint or retraining from scratch?**
   - Fine-tune: Bundle A or B is appropriate; expect 20-50 epochs of
     adaptation.
   - From scratch: Bundle C is appropriate but expect 100+ epochs to surpass
     current checkpoint.
2. **Is downstream TSE / SC pipeline affected?**
   - If yes: keep `enable_semantic_bridge: true` and `use_doa_head: true`
     in all bundles (already specified).
   - If no: items 8 and 10 cleanup options become viable.
3. **Are you scoring on the official test set (CAPI-SDRi against dry stems)?**
   - Yes (default): Option B from item 3 is **forbidden** (wet targets break
     the metric).
   - If using a custom dry-vs-wet scoring rule: more flexibility on item 3.

