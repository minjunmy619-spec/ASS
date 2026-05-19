# TIGER NPU Compilation Operation - 2026-05-17

## Scope

Verified TIGER recipe variants under `recipes/dnr/models/*tiger*.rt192k` with the ONE import -> optimize -> quantize flow. The `dcase2026baseline` subtree was not used.

## Tooling updates

- `tools/online/verify_npu_variants.py`
  - skips `onnxsim` for graphs containing `Tile` or `ConstantOfShape`, because those graphs can spend minutes folding large tensors before reaching ONE;
  - still runs `onnxsim --no-large-tensor` for other graphs;
  - classifies `failed to legalize operation` as an import-stage failure.
- `tools/online/export_onnx_online_model.py` and `tools/online/audit_onnx_model.py`
  - allow `Relu`, `Resize`, and `Squeeze` for the current edge NPU contract.

## Commands run

Single passing edge-v2 check:

```bash
BASE=/home/cmj/works/ASS
TMP=$BASE/logs/npu_debug_manual/tiger_recipe_roots_edge_v2_rerun
rm -rf "$TMP"
mkdir -p "$TMP"
ln -s "$BASE/recipes/dnr/models/tiger-npu-edge-v2.rt192k" "$TMP/tiger-npu-edge-v2.rt192k"
PYTHONUNBUFFERED=1 "$BASE/.venv/bin/python" "$BASE/tools/online/verify_npu_variants.py" \
  --mode recipe \
  --recipe-root "$TMP" \
  --run-name tiger_edge_v2_allowlist_20260517 \
  --quantize-layer-fallback
```

Per-variant TIGER sweep:

```bash
BASE=/home/cmj/works/ASS
TMP=$BASE/logs/npu_debug_manual/tiger_recipe_roots_single
rm -rf "$TMP"
mkdir -p "$TMP"
for v in tiger-ctx-deployable.rt192k tiger-ctx-tiger-like.rt192k tiger-deployable.rt192k tiger-npu-edge-v1.rt192k tiger-npu-edge-v2.rt192k tiger-npu-large.rt192k tiger-tiger-like.rt192k; do
  rm -f "$TMP"/*
  ln -s "$BASE/recipes/dnr/models/$v" "$TMP/$v"
  PYTHONUNBUFFERED=1 timeout 240s "$BASE/.venv/bin/python" "$BASE/tools/online/verify_npu_variants.py" \
    --mode recipe \
    --recipe-root "$TMP" \
    --run-name "tiger_single2_${v%.rt192k}_20260517" \
    --quantize-layer-fallback
done
```

## Results

| Variant | Result | Stage / blocker | Artifact root |
|---|---:|---|---|
| `tiger-npu-edge-v2.rt192k` | PASS | import, optimize, quantize completed; `model.q.circle` exists | `logs/npu_verify_general/tiger_edge_v2_allowlist_20260517/tiger-npu-edge-v2.rt192k/` |
| `tiger-ctx-deployable.rt192k` | FAIL | raw ONNX import fails: `loc("/separator/freq_path.1/Slice"): error: failed to legalize operation 'onnx.Slice'`; previous simplified run reached quantization and failed in `record-minmax` Transpose assert | `logs/npu_verify_general/tiger_single2_tiger-ctx-deployable_20260517/tiger-ctx-deployable.rt192k/` |
| `tiger-ctx-tiger-like.rt192k` | FAIL | raw ONNX import fails at `onnx.Slice`; previous simplified run reached quantization and failed in `record-minmax` Transpose assert | `logs/npu_verify_general/tiger_single2_tiger-ctx-tiger-like_20260517/tiger-ctx-tiger-like.rt192k/` |
| `tiger-deployable.rt192k` | FAIL | raw ONNX import fails: `loc("/separator/freq_path.1/Slice"): error: failed to legalize operation 'onnx.Slice'` | `logs/npu_verify_general/tiger_single2_tiger-deployable_20260517/tiger-deployable.rt192k/` |
| `tiger-npu-edge-v1.rt192k` | FAIL | raw ONNX import fails: `loc("/separator/stages.0/freq_attn/Slice"): error: failed to legalize operation 'onnx.Slice'` | `logs/npu_verify_general/tiger_single2_tiger-npu-edge-v1_20260517/tiger-npu-edge-v1.rt192k/` |
| `tiger-npu-large.rt192k` | FAIL | raw ONNX import fails: `loc("/separator/stages.0/freq_attn/Slice"): error: failed to legalize operation 'onnx.Slice'` | `logs/npu_verify_general/tiger_single2_tiger-npu-large_20260517/tiger-npu-large.rt192k/` |
| `tiger-tiger-like.rt192k` | FAIL | raw ONNX import fails: `loc("/separator/freq_path.1/Slice"): error: failed to legalize operation 'onnx.Slice'` | `logs/npu_verify_general/tiger_single2_tiger-tiger-like_20260517/tiger-tiger-like.rt192k/` |

## Passing edge-v2 details

- `model.circle`: 24 MiB
- `model.opt.circle`: 24 MiB
- `model.q.circle`: 7.1 MiB
- ONNX disallowed ops after allowlist update: `[]`
- ONNX key op counts: `Conv=95`, `MatMul=8`, `Softmax=4`, `Resize=14`, `Transpose=8`, `Slice=54`.

## Next fix target

`tiger-npu-edge-v2.rt192k` is the current compile-ready TIGER candidate. The older TIGER/ctx variants still need source-graph cleanup before they can be treated as NPU candidates:

1. Replace or pre-lower the unsupported ONNX `Slice` patterns in `freq_path` / `freq_attn`.
2. For ctx variants, after import is unblocked, re-check the older quantization failure from `logs/npu_verify_general/tiger_transpose_rewire_retry_20260516`, where `record-minmax` aborts in `luci-interpreter/src/kernels/Transpose.cpp` with `perm()->shape().dim(0) == dims`.

## API rerun - 2026-05-17

Re-ran the full TIGER recipe sweep with the current workspace state and artifact checks.

```bash
BASE="/home/cmj/works/ASS"
TMP="$BASE/logs/npu_debug_manual/tiger_recipe_roots_all_20260517_api"
rm -rf "$TMP"
mkdir -p "$TMP"
for v in tiger-ctx-deployable.rt192k tiger-ctx-tiger-like.rt192k tiger-deployable.rt192k tiger-npu-edge-v1.rt192k tiger-npu-edge-v2.rt192k tiger-npu-large.rt192k tiger-tiger-like.rt192k; do
  ln -s "$BASE/recipes/dnr/models/$v" "$TMP/$v"
done
PYTHONUNBUFFERED=1 "$BASE/.venv/bin/python" "$BASE/tools/online/verify_npu_variants.py" \
  --mode recipe \
  --recipe-root "$TMP" \
  --run-name tiger_all_api_20260517 \
  --quantize-layer-fallback
```

Result summary:

| Variant | Result | Stage / blocker |
|---|---:|---|
| `tiger-npu-edge-v2.rt192k` | PASS | import, optimize, quantize completed; `model.q.circle` exists |
| `tiger-ctx-deployable.rt192k` | FAIL | import: `loc("/separator/freq_path.1/Slice"): error: failed to legalize operation 'onnx.Slice'` |
| `tiger-ctx-tiger-like.rt192k` | FAIL | import: `loc("/separator/freq_path.1/Slice"): error: failed to legalize operation 'onnx.Slice'` |
| `tiger-deployable.rt192k` | FAIL | import: `loc("/separator/freq_path.1/Slice"): error: failed to legalize operation 'onnx.Slice'` |
| `tiger-tiger-like.rt192k` | FAIL | import: `loc("/separator/freq_path.1/Slice"): error: failed to legalize operation 'onnx.Slice'` |
| `tiger-npu-edge-v1.rt192k` | FAIL | import: `loc("/separator/stages.0/freq_attn/Slice"): error: failed to legalize operation 'onnx.Slice'` |
| `tiger-npu-large.rt192k` | FAIL | import: `loc("/separator/stages.0/freq_attn/Slice"): error: failed to legalize operation 'onnx.Slice'` |

Artifacts:

- Summary: `logs/npu_verify_general/tiger_all_api_20260517/summary.md`
- Passing model root: `logs/npu_verify_general/tiger_all_api_20260517/tiger-npu-edge-v2.rt192k/`
- `model.circle`: 24534244 bytes
- `model.opt.circle`: 24787776 bytes
- `model.q.circle`: 7377712 bytes

Interpretation: with the current verifier policy that avoids expensive simplification for `Tile` / `ConstantOfShape` graphs, the older TIGER variants fail earlier at raw ONNX import. The older `tiger_transpose_rewire_retry_20260516` quantization failure is still relevant for simplified ctx graphs, but it is behind the first structural blocker: dynamic Slice legalization in the older attention/state graph.
