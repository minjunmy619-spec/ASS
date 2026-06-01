# FOA Event-Query Prompted SFC Operation

Date: 2026-06-01

## Scope

This pass adds a strongest-performance Proposal D branch that is not constrained
by NPU compilation rules.

The new variant targets 4-channel FOA waveform input and uses sound-event class
queries as prompt conditions.  Each query can be a one-hot class vector or a soft
class-probability vector.  FOA is used as spatial evidence, but each separated
event output is mono.

## Added Variant

- Core: `FOAEventQueryPromptedAsymmetricSFC2D`
- STFT wrapper: `FOAEventQueryPromptedAsymmetricSFCModel`
- Builder:
  `build_prompted_asymmetric_sfc_foa_event_query_strong_system`
- Recipe scaffold:
  `recipes/foa/models/prompted-asymmetric-sfc.foa-event-query-strong/config.yaml`

## Design

- Input waveform contract: `[B, 4, samples]` FOA.
- Query condition contract: `[B, Q, C]`, where `Q` is the number of output event
  queries and `C` is the number of sound-event classes.
- Output waveform contract: `[B, Q, 1, samples]`.
- Supports condition modes:
  - `probability`: one-hot or softmax class probabilities are used directly.
  - `softmax` / `logits`: logits are normalized inside the model.
  - `onehot`: argmax class is converted to one-hot inside the model.
- Uses offline `ModelWrapper`, not `OnlineModelWrapper`, because this branch is
  performance-first and non-causal.
- Uses high-capacity non-NPU blocks:
  - cross-attention SFC query compression,
  - axial time and band Transformer attention,
  - Conformer-style depthwise 2D convolution branches,
  - event-query FiLM,
  - shared event-conditioned cross-attention decoder,
  - shared mono complex mask head over the FOA W-channel plus learned mono
    residual complex output head.

## Training Note

The recipe uses `EventConditionedSupTask`.  The datamodule must emit either:

```text
dict(wav=<FOA mixture>, ref=<mono targets>, event_condition=<B,Q,C>)
```

or:

```text
(wav, ref, event_condition)
```

The existing DnR mono datamodules are not suitable for this FOA event-query
recipe without adaptation.

## Verification Commands

```bash
PYTHONPYCACHEPREFIX=/tmp/opencode/pycache \
  ./.venv/bin/python -m py_compile \
  spectral_feature_compression/core/model/foa_event_query_prompted_sfc.py \
  spectral_feature_compression/core/model/proposed_separation_models.py \
  spectral_feature_compression/core/tasks/conditioned_sup_task.py \
  spectral_feature_compression/__init__.py \
  tests/test_proposed_separation_models.py

./.venv/bin/python -m ruff check \
  spectral_feature_compression/core/model/foa_event_query_prompted_sfc.py \
  spectral_feature_compression/core/model/proposed_separation_models.py \
  spectral_feature_compression/core/tasks/conditioned_sup_task.py \
  spectral_feature_compression/__init__.py \
  tests/test_proposed_separation_models.py

./.venv/bin/python -m pytest \
  tests/test_proposed_separation_models.py::test_foa_event_query_prompted_asymmetric_sfc_uses_class_queries
```

Verification results from this pass:

- `py_compile`: pass.
- `ruff check`: pass, with only the existing pyproject deprecation warning.
- `pytest tests/test_proposed_separation_models.py`: 13 passed.
- Recipe YAML parse: pass.
- Default strong core parameter count at `n_freq=1025`, `n_bands=128`,
  `d_model=192`, `8` encoder layers, `4` decoder layers: `11989394` params.
- Review follow-up: `EventConditionedSupTask` now segments/repeats
  `event_condition` alongside CSS validation mini-batches, so long validation
  recordings with batch size greater than one do not pass stale `[B,Q,C]`
  conditions into larger segmented batches.

## Pending

- Wire a real FOA datamodule that emits event-query conditions.
- Add event-query aware PIT or CAPI assignment if the dataset has duplicate
  same-class events.
- Add event-class conditioning from an external SED model if class probabilities
  are estimated rather than oracle labels.
