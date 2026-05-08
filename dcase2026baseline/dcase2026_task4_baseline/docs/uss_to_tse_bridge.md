# USS -> TSE Semantic-Acoustic Bridge

This is an opt-in path. Existing USS/TSE configs keep their old behavior unless
they set a bridge/query conditioning dimension or enable S5 handoff.

## 1. Train bridge-aware USS

```bash
python -m src.train \
  -c config/separation/modified_deft_uss_temporal_bridge_min.yaml \
  -w workspace/uss_bridge
```

This uses:

- `src.datamodules.uss_spatial_dataset.USSSpatialDataset`
- `src.models.deft.modified_deft_semantic_bridge.SemanticBridgeModifiedDeFTUSSSpatialTemporal`
- `src.training.lightningmodule.uss_bridge.USSBridgeLightning`
- `src.training.loss.uss_bridge_loss.get_loss_func`

The USS bridge model emits extra proposal keys:

- `foreground_embedding`
- `foreground_audio_embedding`
- `prototype_logits`
- `pred_doa_vector`
- `used_spatial_vector`
- `tse_condition`

## 2. Export bridge features

```bash
python -m src.tools.export_uss_bridge_features \
  --config config/separation/modified_deft_uss_temporal_bridge_min.yaml \
  --checkpoint workspace/uss_bridge/checkpoints/last.ckpt \
  --soundscape_dir workspace/sc_finetune/soundscape \
  --output_dir workspace/sc_finetune/uss_bridge_features \
  --batch_size 4
```

The exporter writes one file per soundscape:

```text
workspace/sc_finetune/uss_bridge_features/<soundscape>.pt
```

Each file can contain `tse_condition`, embeddings, DoA vectors, and logits.

## 3. Fine-tune TSE with bridge features

```bash
python -m src.train \
  -c config/separation/modified_deft_tse_lite_6s_temporal_estimated_enrollment_bridge_min.yaml \
  -w workspace/tse_bridge
```

This uses:

- `src.datamodules.tse_bridge_dataset.BridgeEstimatedEnrollmentTSEDataset`
- `src.training.lightningmodule.tse_bridge.TSEBridgeLightning`
- `src.models.deft.modified_deft_tse_bridge.BridgeModifiedDeFTTSEMemoryEfficientTemporal`

The TSE wrapper preserves the old input contract. If `bridge_condition` is absent, it behaves like the base TSE model. If present, it projects the bridge feature into an additive residual over the original `label_vector` condition.

The base TSE classes now also have a direct query-condition receiver. Setting
`query_condition_dim > 0` on `ModifiedDeFTTSE`,
`ModifiedDeFTTSEMemoryEfficient`, `ModifiedDeFTTSETemporal`, or
`ModifiedDeFTTSEMemoryEfficientTemporal` adds a per-query FiLM projection for
`query_condition`, `tse_condition`, `bridge_condition`, or
`proposal_condition`. The conditioned estimated-enrollment configs use this
path:

```bash
python -m src.train \
  -c config/separation/modified_deft_tse_lite_6s_temporal_estimated_enrollment_uss_conditioned.yaml \
  -w workspace/separation

python -m src.train \
  -c config/separation/modified_deft_tse_lite_10s_temporal_estimated_enrollment_uss_conditioned.yaml \
  -w workspace/separation
```

For live S5 inference, enable the handoff explicitly:

```yaml
model:
  args:
    tse_uss_conditioning_enabled: true
    tse_config:
      args:
        query_condition_dim: 256
```

With this flag, `Kwon2025S5` and `Kwon2025TemporalS5` first use explicit USS
proposal tensors such as `tse_condition`; if they are absent, they synthesize a
compact condition from USS class logits, silence logits, count logits, spatial
embeddings, DoA vectors, activity summaries, and slot RMS. The ready-to-run
temporal sibling is
`src/evaluation/eval_configs/kwo2025_top1_like_lite_estimated_temporal_sc_uss_conditioned_tse.yaml`.

`src/evaluation.export_sc_finetune_cache --mode pseudo_s5` preserves that same
condition when the S5 output includes it. Alongside `soundscape/`,
`oracle_target/`, and `estimate_target/`, the exporter writes:

```text
workspace/sc_finetune/uss_bridge_features/<soundscape>.pt
```

with both `query_condition` and `tse_condition` keys. `evaluate_stage.py
--stage tse` forwards bridge/query/proposal tensors when they are present in the
stage dataset and warns when a query-conditioned TSE is evaluated without any
condition tensor. Full `evaluate.py` is still the correct check for live
USS-to-TSE handoff.

## Recommended schedule

For USS bridge training:

- start with `predicted_spatial_prob: 0.0` for a stable warmup if training is unstable;
- use `0.3` after warmup;
- increase to `0.5` only after DoA and class losses are stable.

For TSE bridge fine-tuning:

- start with `bridge_label_scale: 0.3` to avoid overpowering the class condition;
- use `0.5` once the exported USS features are reliable;
- keep `pretrained_model_strict: false` because the bridge wrapper adds new projection weights.
