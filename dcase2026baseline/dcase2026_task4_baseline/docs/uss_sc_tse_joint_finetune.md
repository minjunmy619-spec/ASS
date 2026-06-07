# Full Joint USS + SC + TSE Finetuning

This document explains the full joint finetuning path for the three-model
pipeline:

```text
FOA mixture
  -> USS separator
  -> SC source classifier on USS foreground slots
  -> TSE refiner conditioned by USS slots + SC labels + USS hints
```

Main implementation:

```text
src/training/lightningmodule/uss_sc_tse_joint_model_parallel.py
```

Main starter config:

```text
config/separation/modified_deft_uss_sc_tse_joint_two_stage_spatial.yaml
```

The joint module is **standalone** and does not subclass the existing USS+SC
joint module. It uses single-process model parallelism, manual optimization,
three supervised anchor losses, and explicit gradient-routing controls.

---

## 1. Why this pipeline exists

The previous training stages are useful but limited:

1. SC can be finetuned on USS outputs.
2. TSE can be finetuned with frozen USS and SC teachers.
3. But USS, SC, and TSE do not adapt together to the final live pipeline.

The full joint pipeline addresses this by training all three models in the same
step:

```text
USS learns to produce separations useful for both source quality and downstream SC/TSE.
SC learns on current USS outputs and can optionally receive TSE-error feedback.
TSE learns on live USS outputs and live SC label distributions.
```

The important safety rule is that each model still has its own direct supervised
anchor:

```text
USS: oracle separation/class/activity losses
SC: oracle-matched class/silence loss on USS slots
TSE: oracle source waveform loss aligned into USS slot order
```

Do not rely on a single downstream loss to train all three models. That can make
upstream models drift toward artifacts that satisfy one downstream objective but
hurt final evaluation.

---

## 2. Runtime layout

This path is **model-parallel**, not DDP.

Example three-GPU placement:

```yaml
uss_device: cuda:0
sc_device: cuda:1
tse_device: cuda:2
loss_device: cuda:0
```

If only two GPUs are available, colocate two components manually, for example:

```yaml
uss_device: cuda:0
sc_device: cuda:1
tse_device: cuda:0
loss_device: cuda:0
```

The trainer should still use one Lightning process:

```yaml
trainer:
  args:
    accelerator: gpu
    devices: 1
    strategy: auto
```

Do **not** switch this mode to ordinary DDP. DDP would replicate all three models
on each GPU and usually increases memory usage.

---

## 3. Dataset contract

The joint module expects `USSDataset` batches, not the online-teacher TSE-only
batch wrapper.

Config:

```yaml
dataset:
  module: src.datamodules.uss_dataset
  main: USSDataset
```

Required or expected batch keys:

```text
mixture: [B, 4, T]
foreground_waveform: [B, S, 1, T]
interference_waveform: [B, I, 1, T]
noise_waveform: [B, N, 1, T]
class_index: [B, S]
is_silence: [B, S]
foreground_span_sec: optional [B, S, 2]
interference_span_sec: optional [B, I, 2]
noise_span_sec: optional [B, N, 2]
foreground_doa: optional [B, S, 3]
foreground_doa_mask: optional [B, S]
label_vector: optional [B, S, K] or flattened equivalent
```

TSE target labels are reconstructed from `label_vector` if present; otherwise
from `class_index` and `is_silence`.

---

## 4. Forward pass

Each training step performs:

```text
1. USS forward
   mixture -> uss_out

2. USS anchor loss
   uss_loss(uss_out, oracle USS targets)

3. SC-on-USS branch
   uss_out[foreground_waveform]
     -> oracle waveform matching
     -> class_index / is_silence / sample_weight targets
     -> SC forward/loss

4. TSE branch
   USS foreground slots -> TSE enrollment
   SC logits/probabilities -> TSE label_vector
   USS hints -> TSE query/temporal/spatial conditions
   TSE forward/loss against oracle sources aligned to USS slots

5. Optional SC-on-TSE branch
   TSE output waveforms -> SC loss on final refined waveform distribution
```

The integrated loss is:

```text
L_total =
    lambda_uss                * L_uss
  + lambda_sc_uss             * L_sc_uss
  + lambda_tse                * L_tse
  + lambda_sc_tse             * L_sc_tse
  + lambda_uss_sc_consistency * L_uss_sc_consistency
```

Starter config values:

```yaml
lambda_uss: 1.0
lambda_sc_uss: 0.05
lambda_tse: 1.0
lambda_sc_tse: 0.0
lambda_uss_sc_consistency: 0.01
```

---

## 5. SC energy threshold and silence gates

### 5.1 SC inference behavior

SC `predict()` applies energy thresholds:

```text
plain_logits -> top-1 class
energy = -logsumexp(plain_logits)
if energy > class/default threshold: label_vector = all zeros
```

This is an inference-time silence decision.

### 5.2 SC training loss behavior

During joint finetuning, the SC anchor loss uses `sc_model.forward(...)`, not
`sc_model.predict(...)`.

Therefore the SC anchor loss does **not** apply hard energy thresholds. Instead,
its active/silence targets come from oracle waveform matching:

```text
USS slot matches active oracle source -> active class target
unmatched or bad slot -> silence/low-weight target
```

This is intentional. The loss should teach SC calibration; it should not depend
on a fixed hard threshold that may be stale while USS is changing.

The SC loss can still include continuous energy supervision through:

```yaml
sc_loss:
  args:
    lambda_energy: 0.001
    m_in: -6.0
    m_out: -1.0
```

This uses the `energy` tensor continuously. It is not the same as applying an
inference threshold.

### 5.3 TSE label silence gates

The TSE query label is built from SC logits. The joint module supports:

```yaml
tse_label_silence_gate: none
tse_label_silence_gate: hard_energy
tse_label_silence_gate: soft_energy
```

#### `none`

```yaml
tse_label_silence_gate: none
```

No energy-threshold silence decision is applied to TSE labels. TSE receives SC
class probabilities for every slot.

Recommended default for early joint training.

Why:

- avoids zeroing useful labels due to stale thresholds;
- keeps TSE supervision alive when SC is uncertain;
- works well with soft labels;
- gives SC a chance to improve from direct SC loss and optional TSE-error
  feedback.

#### `hard_energy`

```yaml
tse_label_silence_gate: hard_energy
```

Uses SC energy thresholds to hard-zero labels for predicted-silence slots.

This best matches inference behavior, but it is risky during training:

```text
SC false negative -> label vector becomes zero -> TSE receives no useful query
```

Avoid it at the beginning. Use only as a late-stage ablation or deployment-match
experiment.

#### `soft_energy`

```yaml
tse_label_silence_gate: soft_energy
tse_label_silence_temperature: 0.5
```

Applies a differentiable-ish gate:

```text
active_prob = sigmoid((threshold - energy) / temperature)
label_vector = active_prob * softmax(plain_logits / label_temperature)
```

This is a compromise between no gate and hard thresholding. It can be useful
after the SC energy distribution has stabilized.

### 5.4 Recommendation

Use this first:

```yaml
tse_label_mode: soft_detached
tse_label_silence_gate: none
tse_label_grad_scale: 0.0
```

After the tri-loss run is stable, try:

```yaml
tse_label_mode: soft_grad
tse_label_silence_gate: none
tse_label_grad_scale: 0.02
```

Only later try:

```yaml
tse_label_mode: soft_grad
tse_label_silence_gate: soft_energy
tse_label_silence_temperature: 0.5
tse_label_grad_scale: 0.02
```

Avoid `hard_energy` unless you are specifically testing inference-like threshold
behavior.

---

## 6. Soft SC labels and TSE-error feedback to SC

The TSE model accepts continuous `label_vector` inputs because class conditioning
is implemented by linear layers. Therefore soft SC probabilities are valid TSE
queries.

Supported modes:

| Mode | Forward label to TSE | TSE loss updates SC? | Typical use |
| --- | --- | --- | --- |
| `hard_detached` | hard one-hot | no | deployment-like baseline |
| `soft_detached` | soft probabilities | no | safe joint default |
| `soft_grad` | soft probabilities | yes, scaled | let TSE error improve SC |
| `straight_through` | hard forward, soft backward | yes, scaled | deployment-like forward with gradient |

Config knobs:

```yaml
tse_label_mode: soft_detached
tse_label_temperature: 1.0
tse_label_grad_scale: 0.0
```

For TSE-to-SC feedback:

```yaml
tse_label_mode: soft_grad
tse_label_grad_scale: 0.02
```

This gives SC two gradient sources:

```text
SC gradient =
    lambda_sc_uss * grad(SC direct supervised loss)
  + lambda_tse * tse_label_grad_scale * grad(TSE waveform loss through SC labels)
```

Keep `tse_label_grad_scale` small at first. Values like `0.02` or `0.05` are
safer than `1.0`.

---

## 7. Requiring SC active/class match for TSE loss

Two flags can filter which slots contribute to the TSE loss:

```yaml
require_sc_active_for_tse_loss: false
require_sc_class_match_for_tse_loss: false
```

They are conceptually different from the SC loss. They only decide whether an
oracle-aligned USS slot should be supervised by TSE.

### 7.1 `require_sc_active_for_tse_loss`

If enabled:

```text
TSE supervised slot = oracle/USS matched slot AND SC predicted active
```

This can protect TSE when SC is a frozen teacher and false positives are common.
But it can also block useful supervision when SC is uncertain or miscalibrated.

### 7.2 `require_sc_class_match_for_tse_loss`

If enabled:

```text
TSE supervised slot = oracle/USS matched slot AND SC predicted the oracle class
```

This is stricter and often too restrictive during joint training.

If SC predicts the wrong class:

```text
SC wrong -> slot filtered out -> no TSE loss -> no TSE-to-SC correction
```

So this flag can prevent the pipeline from correcting SC mistakes.

### 7.3 Recommended settings by training mode

| Training mode | `require_sc_active_for_tse_loss` | `require_sc_class_match_for_tse_loss` | Why |
| --- | --- | --- | --- |
| Frozen SC teacher, hard labels | `true` often useful | optional | protects TSE from bad teacher rows |
| Joint training, soft labels | `false` | `false` | soft labels express uncertainty; keep supervision alive |
| Joint training, `soft_grad` | `false` | `false` | needed for TSE error to correct SC |
| Debug/ablation if unstable | `true` | `false` | intermediate safety filter |

Default in the joint config:

```yaml
require_sc_active_for_tse_loss: false
require_sc_class_match_for_tse_loss: false
```

This is the recommended default for full joint finetuning.

---

## 8. Match quality and target construction

TSE targets are not copied from SC. They are built by waveform matching:

```text
USS foreground slot -> matched to oracle foreground source by SA-SDR/SI-SDR
```

Important config:

```yaml
match_metric: sa_sdr
min_tse_match_score: -20.0
min_estimate_energy_db: -80.0
tse_use_match_quality_weight: false
```

The SC labels are used as TSE query conditions, but the TSE target waveform and
oracle label are still derived from oracle-aligned matching. This prevents wrong
SC predictions from becoming the target class.

If `tse_use_match_quality_weight: true`, the module applies the same clean /
uncertain / bad match logic to TSE target selection and logs:

```text
tse_match_quality_weight
```

This is useful for stricter training, but the current default keeps it disabled
so TSE sees more matched rows.

---

## 9. Gradient routing

The joint module separates forward values from backward gradient flow.

Gradient scaling uses the pattern:

```python
x_for_downstream = x.detach() + scale * (x - x.detach())
```

This keeps the forward tensor unchanged while scaling gradient into the upstream
producer.

Main knobs:

```yaml
sc_uss_to_uss_grad_scale: 0.0
tse_to_uss_enrollment_grad_scale: 0.0
tse_to_uss_condition_grad_scale: 0.0
sc_tse_to_tse_grad_scale: 0.0
tse_label_grad_scale: 0.0
```

Recommended initial defaults are all zero, with all three models still trainable
through their own supervised losses:

```text
USS updates from L_uss
SC updates from L_sc_uss
TSE updates from L_tse
```

After the pipeline is stable, enable controlled coupling one piece at a time:

```yaml
tse_label_mode: soft_grad
tse_label_grad_scale: 0.02
```

Then optionally:

```yaml
tse_to_uss_enrollment_grad_scale: 0.02  # or 0.05
```

Avoid enabling many cross-model gradients at once.

---

## 10. Recommended training schedule

### Stage A: stable tri-loss joint run

```yaml
tse_label_mode: soft_detached
tse_label_silence_gate: none
tse_label_grad_scale: 0.0
require_sc_active_for_tse_loss: false
require_sc_class_match_for_tse_loss: false
sc_uss_to_uss_grad_scale: 0.0
tse_to_uss_enrollment_grad_scale: 0.0
tse_to_uss_condition_grad_scale: 0.0
lambda_sc_tse: 0.0
```

This verifies that the pipeline, losses, device placement, and matching logic are
correct.

### Stage B: let TSE error update SC softly

```yaml
tse_label_mode: soft_grad
tse_label_grad_scale: 0.02
tse_label_silence_gate: none
```

Watch SC direct metrics and TSE loss. Increase to `0.05` only if stable.

### Stage C: optional final SC-on-TSE adaptation

```yaml
lambda_sc_tse: 0.01
sc_tse_to_tse_grad_scale: 0.0
```

This adapts SC to final TSE-refined waveforms without letting SC loss reshape TSE
outputs. Later, a tiny `sc_tse_to_tse_grad_scale` can be tested as an ablation.

### Stage D: optional soft energy gate

```yaml
tse_label_silence_gate: soft_energy
tse_label_silence_temperature: 0.5
```

Use only after SC energy calibration is stable.

---

## 11. LR scheduler, BF16, and readiness checklist

### 11.1 Warmup cosine LR scheduler

The joint module supports a lightweight built-in warmup-cosine scheduler for
manual optimization.  It is configured per optimizer:

```yaml
uss_lr_scheduler:
  interval: epoch
  frequency: 1
  scheduler:
    main: warmup_cosine
    args:
      warmup_epochs: 2
      max_epochs: 30
      start_lr_scale: 0.1
      min_lr_scale: 0.1
```

Equivalent `*_steps` names are also accepted for step-based schedules:

```yaml
scheduler:
  main: warmup_cosine
  args:
    warmup_steps: 1000
    max_steps: 50000
    start_lr_scale: 0.1
    min_lr_scale: 0.1
```

The starter config uses epoch-based schedules for USS, SC, and TSE:

```text
epoch 0        -> 10% of base LR
end warmup     -> 100% of base LR
after warmup   -> cosine decay
end of training -> 10% of base LR
```

The module steps schedulers manually because it uses
`automatic_optimization = False`.  Epoch schedulers are stepped in
`on_train_epoch_end`; step schedulers are stepped in `training_step`.

If you change `max_epochs`, update all three scheduler `max_epochs` values too.
If you use a step-based schedule, estimate `max_steps` from:

```text
num_training_batches * max_epochs / accumulate_grad_batches
```

### 11.2 BF16 mixed precision

The starter config uses:

```yaml
precision: bf16-mixed
```

This is expected to work with the current pipeline because:

- TSE STFT/ISTFT paths use explicit autocast-disabled full-precision blocks.
- SC soft labels are computed with `softmax(logits.float())` before casting back.
- Matching and quality decisions use detached tensors and mostly FP32 math.
- Manual gradient clipping is explicitly applied before optimizer steps.

Requirements and caveats:

- All assigned GPUs should support BF16 well, typically Ampere or newer.
- The SC PretrainedSED/M2D branches are the most likely place to expose
  precision-specific instability.
- If NaNs appear, first test a short debug run with:

```yaml
precision: 32-true
```

Then lower LR or disable cross-model gradient routes before changing the main
training recipe.

### 11.3 Readiness checklist

Before starting a real run, check:

1. **TSE checkpoint**

   The starter config leaves this empty:

   ```yaml
   tse_pretrained_ckpt:
   ```

   For true finetuning, set it to your best TSE checkpoint.  Leaving it empty
   means the two-stage TSE model starts from random initialization.

2. **GPU mapping**

   Make sure visible GPU IDs match the configured model devices:

   ```yaml
   uss_device: cuda:0
   sc_device: cuda:1
   tse_device: cuda:2
   ```

   The module re-places and asserts the submodel devices at fit/validation/test
   start.  If Lightning or the environment moves a model unexpectedly, training
   should fail early rather than silently OOMing one GPU.

3. **Safe labels and gates**

   Start with:

   ```yaml
   tse_label_mode: soft_detached
   tse_label_silence_gate: none
   require_sc_active_for_tse_loss: false
   require_sc_class_match_for_tse_loss: false
   ```

4. **Cross-model gradients**

   Start with all route scales at zero:

   ```yaml
   sc_uss_to_uss_grad_scale: 0.0
   tse_to_uss_enrollment_grad_scale: 0.0
   tse_to_uss_condition_grad_scale: 0.0
   sc_tse_to_tse_grad_scale: 0.0
   tse_label_grad_scale: 0.0
   ```

   Then enable only one route at a time after the tri-loss run is stable.

5. **Validation checks**

   Quick local checks that do not instantiate all heavyweight models:

   ```bash
   cd /home/cmj/works/ASS/dcase2026baseline

   .venv/bin/python -m py_compile \
     dcase2026_task4_baseline/src/training/lightningmodule/uss_sc_tse_joint_model_parallel.py

   PYTHONPATH=dcase2026_task4_baseline .venv/bin/python -c \
     "from src.utils import parse_yaml; cfg=parse_yaml('dcase2026_task4_baseline/config/separation/modified_deft_uss_sc_tse_joint_two_stage_spatial.yaml'); print(cfg['lightning_module']['main'])"
   ```

6. **Runtime smoke test**

   The final readiness check is a short run on the target GPU machine with the
   real checkpoints and data.  Use a tiny debug split or a low `limit_train_batches`
   if your training launcher supports it.  The syntax/config checks cannot prove
   memory fits.

---

## 12. Diagnostics to watch

Losses:

```text
epoch_train/loss
epoch_train/loss_uss
epoch_train/loss_sc_uss
epoch_train/loss_tse
epoch_train/loss_sc_tse
epoch_train/loss_uss_sc_consistency
```

SC-on-USS:

```text
sc_uss_top1
sc_uss_active_weight_mean
sc_uss_used_match_count
sc_uss_clean_match_count
sc_uss_uncertain_match_count
sc_uss_bad_match_count
sc_uss_unmatched_silence_count
```

TSE target/query health:

```text
tse_raw_matched_slots
tse_matched_slots
tse_match_score
tse_estimate_energy_db
tse_sc_active_rate
tse_sc_class_match_rate
tse_label_entropy
tse_match_quality_weight
```

Interpretation:

- `tse_raw_matched_slots` low: USS estimates are not matching oracle sources well.
- `tse_matched_slots` much lower than raw: gating/filtering is too strict.
- `tse_sc_class_match_rate` low: SC is wrong often; avoid class-match filtering.
- `tse_label_entropy` near zero: SC labels are too hard/confident.
- `sc_uss_used_match_count` low: SC direct loss has too few supervised rows.

---

## 13. Starter command

Run from the project root with the project virtual environment:

```bash
cd /home/cmj/works/ASS/dcase2026baseline/dcase2026_task4_baseline
source .venv/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python -m src.train \
  -c config/separation/modified_deft_uss_sc_tse_joint_two_stage_spatial.yaml \
  --workspace workspace/uss_sc_tse_joint_two_stage_spatial
```

For model-parallel GPU mapping, set visible devices before launch if needed:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python -m src.train \
  -c config/separation/modified_deft_uss_sc_tse_joint_two_stage_spatial.yaml \
  --workspace workspace/uss_sc_tse_joint_two_stage_spatial
```

---

## 14. Practical recommendations

1. Start with `soft_detached` and no silence gate.
2. Do not require SC active/class match in the first joint run.
3. Let SC learn from direct oracle-matched SC loss before enabling TSE-to-SC
   gradient.
4. Use small TSE-to-SC gradient scales (`0.02` to `0.05`).
5. Keep pretrained SC feature branches frozen unless there is strong evidence
   that the classifier head/late blocks are insufficient.
6. Do not enable hard energy gating early.
7. Do not enable many cross-model gradient paths at once.
8. Compare against the previous frozen-teacher TSE checkpoint using the same final
   evaluation config before adopting a joint checkpoint.
