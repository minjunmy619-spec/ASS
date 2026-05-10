# Tasks: Band-SCNet-NPU

Tasks are ordered so the NPU pipeline is validated as early as possible (fail fast), with training-related work layered on top only after graph validity is proven. Each task references the relevant requirement(s) in `requirements.md` and the matching section(s) in `design.md`.

> Guideline: run every verification step inside the project's docker + `.venv` (per `AGENT.md`).

Legend: `[x]` implemented and tested in-repo · `[ ]` pending.

---

## Phase 1 — Scaffolding

- [x] **1.1 Create `BandSCNetNPU/` package skeleton** — _FR-12, NFR-6_
  - Files created: `__init__.py`, `band_scnet_npu.py`, `blocks.py`, `sparse_io.py`, `presets.py`, `streaming.py`, `test_band_scnet_npu.py`, `README.md`.
- [x] **1.2 Set up shared imports** — _FR-12_
  - Reuses `CausalConv2d`, `RMSNorm2d`, `_runtime_assert`, `pack_complex_stft_as_2d` from `spectral_feature_compression.core.model.online_sfc_2d`; and `flatten_tensor_tree` / `unflatten_tensor_tree` from `spectral_feature_compression.utils.onnx_streaming`.

---

## Phase 2 — Core Blocks

- [x] **2.1 Implement `GatedAct`** — _FR-5, design §4.8_ — `blocks.GatedAct`
- [x] **2.2 Implement `CrossBandBlock`** — _FR-2, FR-5, design §4.3_
- [x] **2.3 Implement `NarrowBandBlock` (no attention)** — _FR-2, FR-4, FR-5, design §4.4_
  - Streaming-vs-full max-abs-diff ≤ 1e-5 confirmed by `test_narrow_band_block_streaming_consistency_no_attn`.
- [x] **2.4 Implement `BoundedCausalAttn`** — _FR-2, FR-5, design §4.5_
  - Uses frequency-pooled KV + ring-buffer cache `[B, num_heads, W, 2*head_dim]` to stay under the 192 KiB budget.
  - Streaming-vs-full parity confirmed by `test_narrow_band_block_streaming_consistency_with_attn` for `T <= W`.
- [x] **2.5 Wire `use_attn` flag** — _FR-2_ — no attn submodule instantiated when `use_attn=False` (zero params, zero state).

---

## Phase 3 — Sparse Pyramid

- [x] **3.1 Implement band-split utility** — _FR-1, design §4.2_
  - `sparse_io.split_bands` + `pad_n_freq_for_split`; covered by `test_band_split_sums_to_n_freq` over `{128, 257, 513, 1025, 2049}`.
- [x] **3.2 Implement `SparseDownsampleEncoder`** — _FR-1, FR-5, FR-6, design §4.2_
  - Strides run FIRST (at full resolution is the input, not the internal state), so internal conv-block state scales with `F_branch / 2**strides`.
- [x] **3.3 Implement `SparseUpsampleDecoder`** — _FR-1, FR-5, FR-6, design §4.6_
  - Conv blocks run at the reduced resolution first, then the stride-2 `ConvTranspose2d` chain expands back.
  - Round-trip shape verified by `test_sparse_pyramid_round_trip_shape`.

---

## Phase 4 — Full Model Assembly

- [x] **4.1 Implement `BandSCNetNPU.__init__`** — _FR-3, FR-12, design §4 & §6_
- [x] **4.2 Implement `BandSCNetNPU.forward`** — _FR-3, FR-6, design §6_
- [x] **4.3 Implement `init_stream_state`** — _FR-4, FR-8, design §5 & §6_
  - Returns a `BandSCNetNPUState` named-tuple; flattens cleanly via `spectral_feature_compression.utils.onnx_streaming.flatten_tensor_tree`.
- [x] **4.4 Implement `forward_stream`** — _FR-4, design §6_
- [x] **4.5 Implement `state_size_bytes`** — _FR-8, design §5_
  - Budget check enforced by `test_state_budget_edge_small` (109 KiB) and `test_state_budget_rt192k` (191 KiB), both under the 192 KiB quota at `n_freq=2049`.
- [x] **4.6 Source mask head** — _FR-3, design §4.7_ — `_SourceMaskHead`, emits real-valued gains applied via `apply_source_gain_mask_4d`.

---

## Phase 5 — Presets

- [x] **5.1 `presets.edge_small()`** — _FR-14, design §3_ — C=16/8, L=2, Kt=5, no attn; ~10 k params, ~109 KiB state at n_freq=2049.
- [x] **5.2 `presets.rt192k()`** — _FR-14, design §3_ — C=40/8, L=3, Kt=3, bounded causal attn W=16; ~62 k params, ~191 KiB state at n_freq=2049.

> Note: The measured parameter counts are below the design-doc targets because the DSP 192 KiB streaming-state quota is the hard constraint. See `README.md` ["Known limitations"] for sizing trade-offs.

---

## Phase 6 — Streaming-Consistency Tests

- [x] **6.1 Non-streaming vs streaming equivalence** — _FR-4, Acceptance #2_ — `test_edge_small_streaming_matches_full`, `test_rt192k_streaming_matches_full`.
- [x] **6.2 Conv-constraint test** — _FR-7, Acceptance #4_ — `test_npu_conv_constraints_both_presets` walks every `Conv2d` / `ConvTranspose2d`.
- [x] **6.3 State-budget test** — _FR-8, Acceptance #3_ — `test_state_budget_{edge_small,rt192k}`.

---

## Phase 7 — ONNX Export

- [x] **7.1 Flat-state adapter** — _FR-9, design §6 & §7_ — `streaming.build_example_state_and_spec` / `restore_state_from_flat`, plus the `BandSCNetNPUStreamingExportWrapper` in `band_scnet_npu.py`.
- [ ] **7.2 Hook into `tools/online/export_onnx_online_model.py`** — _FR-9, FR-12_
  - Add `--target band-scnet-npu` with `--preset {edge_small,rt192k}`.
- [x] **7.3 ONNX checker** — _Acceptance #6_ — `test_streaming_onnx_export_edge_small` calls `onnx.checker.check_model`.
- [x] **7.4 ONNX op audit (local smoke)** — _Acceptance #7_
  - `test_streaming_onnx_export_edge_small` additionally rejects `{Tile, Expand, ConstantOfShape, ScatterND, If, Loop, Scan}`.
  - Full project-standard audit via `tools/online/audit_onnx_model.py` still pending task 7.2.

---

## Phase 8 — MLIR Verification

- [ ] **8.1 Run `export_verify_mlir.py --target band-scnet-npu --preset edge_small`** — _FR-10, Acceptance #8_
- [ ] **8.2 Run the same pipeline on `rt192k`** — _FR-10, Acceptance #8_
- [ ] **8.3 Measure NPU stats** — _NFR-5_ via `tools/online/measure_npu_model_stats.py`.

> These three require running inside the project's docker image (per `AGENT.md`) which has `onnx-mlir` installed.

---

## Phase 9 — Test Script

- [x] **9.1 Assemble `BandSCNetNPU/test_band_scnet_npu.py`** — _NFR-6, Acceptance #1–#10_
  - 12 tests total, all passing locally: block-level, model-level, NPU-constraint, streaming, budget, ONNX.

---

## Phase 10 — Training Recipe

- [ ] **10.1 `recipes/band_scnet_npu/config/model/band_scnet_npu_rt192k.yaml`** — _FR-13, Acceptance #11_
- [ ] **10.2 `recipes/band_scnet_npu/config/task/dnr_3stem_causal.yaml`** — _FR-13_
- [ ] **10.3 Launch scripts (`train.sh`, `eval_streaming.sh`)** — _FR-13_
- [ ] **10.4 1-step smoke-run of the training recipe** — _FR-13, Acceptance #11_

---

## Phase 11 — Documentation

- [x] **11.1 `BandSCNetNPU/README.md`** — _Acceptance #12, NFR-6_
  - Architecture diagram, preset table, op inventory, streaming / ONNX usage examples, paper-traceability table, known limitations.
- [ ] **11.2 Update the top-level `README.md`**

---

## Phase 12 — Stretch / Follow-up

- [ ] **12.1 Add `rt192k_plus` preset** when the DSP quota is relaxed.
- [ ] **12.2 Optional `FrequencyPreprocessedOnlineModel` pre-filter wrapper** (v2).
- [ ] **12.3 Quality comparison report** once trained checkpoints exist.

---

## Definition of Done

Still open (tracked above):

1. Tool integration (phase 7.2) — register `band-scnet-npu` in `tools/online/*.py`.
2. MLIR verification (phase 8) — requires docker + onnx-mlir.
3. Training recipe (phase 10) — requires access to DnR data + aiaccel stack.
4. Top-level README update (phase 11.2).

Already done (covered by passing local test suite):

- Phases 1–6 and 9: module scaffolding, all core blocks, full model, presets, streaming-consistency, NPU kernel/stride constraints, state budget, ONNX export + checker + forbidden-op audit.
- Phase 7.1 + 7.3 + 7.4 (local smoke subset).
- Phase 11.1 (BandSCNetNPU/README.md).
