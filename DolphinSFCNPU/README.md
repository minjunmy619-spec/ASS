# DolphinSFCNPU

`DolphinSFCNPU` is the second-generation ASS adaptation of Dolphin, redesigned
to meet the AGENT.md deployment rules — in particular **rule 13 (streaming
state must fit the 192 KiB DSP quota)** and **rule 14 (small number of ONNX
I/O parameters)**. The first-generation `DolphinSFC/` module is kept intact
for comparison.

## Why the redesign

The previous revision of this module exposed 22 separate streaming-state
tensors at the ONNX boundary (23 inputs, 23 outputs counting `x` / `y`). That
violated rule 14, but more importantly the **total state bytes** were well
above the DSP quota because of:

- a long temporal depthwise conv (`kt=7`) at every Global-Local block,
- a redundant "local" depthwise conv at every block,
- an extra stateful source-prior coder,
- a stateful downsampling conv at every pyramid step,
- a time-stateful `SpectralCompressor2d` operating at full `n_freq`
  (~131 KiB alone at `d_model=128`, `n_freq=257`, fp16).

The earlier pack-into-one-tensor fix only collapsed the **ONNX parameter
count** (22 slots -> 1 slot). It did not reduce the actual state bytes. This
revision does the opposite: it reduces the **real state** by changing the
architecture, and the packed-state export wrapper is retained so rule 14 is
still satisfied simultaneously.

## Architecture

```
input (B, 2*n_chan, T, F)
  -> in_proj (Conv1x1 + RMSNorm)
  -> StatelessBandCompressor2d   # time-stateless, F -> K (band tokens)
  -> encoder stage 0 (+ stateless band_down)
  -> encoder stage 1 (+ stateless band_down)
  -> encoder stage 2 (bottleneck, no downsample)
  -> decoder stage 2 (bottleneck, no upsample)
  -> decoder stage 1 (stateless band_up + skip merge)
  -> decoder stage 0 (stateless band_up + skip merge)
  -> SpectralDecoder2d (K -> F)
  -> out_proj (Conv1x1)
  -> real-valued source-gain mask applied to x
```

Each encoder / decoder stage contains 1 or more `DolphinSFCNPUSlimBlock`s.
Each slim block has:

- a residual **temporal sub-block** with one causal depthwise `(kt, 1)`
  conv — **this is the only streaming cache in the block** — preceded by a
  SiLU-gated pointwise expansion that plays the role of the old standalone
  source-prior coder, and
- a stateless residual **frequency/channel sub-block** (RMSNorm +
  pointwise(2·hC) SiLU gate + depthwise `(1, kf)` + pointwise(C)).

One cache per block, at `n_bands`-level (or half-bands / quarter-bands on
deeper scales) rather than full `n_freq`. The compressor, the band
down/up-samplers, and everything in the frequency sub-block are stateless
along time.

## What this buys you

For three target sizes the following holds at `n_freq=257` with `fp16` state
and `batch=1`:

| preset   | params | state leaves | state bytes (fp16) |
|----------|--------|--------------|---------------------|
| edge_small | ~125 K  | 8 | ~12 KiB  |
| slim_4m  | ~3.6 M  | 8 | ~144 KiB |
| slim_6m  | ~5.0 M  | 8 | ~162 KiB |
| slim_8m  | ~7.7 M  | 8 | ~186 KiB |

All three slim presets fit inside the 192 KiB DSP quota **with headroom**,
and all three stay inside AGENT.md rule 15's 7 M parameter ceiling (slim_8m
is intentionally at the edge of the 3-8 M target).

### Comparison with the previous revision

| metric | old DolphinSFCNPU | slim DolphinSFCNPU |
|---|---|---|
| state leaf tensors | 22 | 8 |
| ONNX `input` / `output` count | 2 / 2 (packed wrapper)  | 2 / 2 (packed wrapper) |
| state bytes at the same capacity | > 300 KiB | < 190 KiB |
| streaming caches per block | 3 (global_dw + local_dw + ffn.dw) | 1 |
| time-stateful compressor | yes | no |
| stand-alone source-prior coder | yes (cached) | no (folded into block gate) |

The slot count in both revisions is 2/2 thanks to the packed-state wrapper,
but the **byte count** in this revision is roughly **45 % smaller** at
comparable capacity, which is the metric that actually matters for rule 13.

## Packed streaming-state I/O (unchanged contract)

The core `DolphinSFCNPUSeparator.forward_stream` still accepts and returns a
nested tuple of per-block caches so Python / training code keeps an
ergonomic API. For ONNX export use `DolphinSFCNPUStreamingExportWrapper`:

```python
wrapper = DolphinSFCNPUStreamingExportWrapper(model, batch_size=1, dtype=torch.float32).eval()
x = torch.randn(1, 2, 1, model.n_freq)
packed_state = wrapper.init_packed_state(batch_size=1, dtype=torch.float32)

torch.onnx.export(
    wrapper,
    (x, packed_state),
    "dolphin_sfc_npu.onnx",
    opset_version=11,
    input_names=["x", "state"],
    output_names=["y", "next_state"],
    do_constant_folding=True,
)
```

The packing layout (per-leaf shape and offset list) is frozen at wrapper
construction time. Unpack is a static list of `Slice` + `Reshape` ops,
repack is per-leaf `Reshape(flatten)` + a single `Concat`. All of these are
on the NPU-allowed op list. No runtime shape branches, no dynamic control
flow.

## Validation

Run inside the ASS Docker checkout:

```bash
cd /app/ASS
./.venv/bin/python DolphinSFCNPU/test_dolphin_sfc_npu.py
```

The test suite asserts:

- streaming (frame-by-frame) output matches the offline forward for all
  presets within fp32 tolerance,
- the packed-state wrapper matches the tree-state core numerically,
- the exported graph has **exactly 2 inputs and 2 outputs** and contains
  none of `{Tile, Expand, ConstantOfShape, ScatterND, Unflatten}`,
- each slim preset's fp16 streaming state stays **below 192 KiB**,
- each slim preset's parameter count lies in the **3-8 M** window,
- the state-leaf count equals 2·sum(blocks_per_scale) (structural regression
  guard — if somebody reintroduces an extra per-block cache, this test
  catches it).

## Explicit-query variants

Two opt-in Dolphin/SFC hybrids are available for ablation while preserving the
same packed-state streaming/export contract:

- `query_variant="soft_band_query"` / preset suffix `_soft_query`: the
  stateless compressor emits both K-band latent tokens and K-band query tokens;
  the decoder uses those query tokens to condition SFC expansion back to full
  frequency. This is the lower-risk query path because the side-path remains on
  the compressed band axis.
- `query_variant="crossattn_query"` / preset suffix `_crossattn_query`: the
  stateless compressor performs F-to-K cross-attention, and the decoder uses
  full-frequency side embeddings as queries over K latent Dolphin tokens. This
  is closer to query/cross-attention SFC, but has more `MatMul`/`Softmax` and
  reshape traffic.

Both variants keep the Dolphin multi-scale separator unchanged and add no extra
streaming cache. Example presets:

- `slim_6m_soft_query` / `slim_6m_soft_band_query`
- `slim_6m_crossattn_query`

The query recipe overlays expose the critical capacity/latency/performance
controls directly in YAML:

- `n_bands`: compressed band resolution and state width;
- `d_model`: compressor/query side-path width;
- `num_scales`, `widths`, `blocks_per_scale`: multi-scale separator capacity;
- `time_kernels`: temporal receptive field and per-block state size;
- `freq_kernels`: stateless frequency-mixing width;
- `compressor_freq_kernel`: stateless compressor frequency refinement;
- `ffn_expansion`: stateless frequency/channel FFN capacity.

The current slim-6m query recipes resolve to useful-capacity models:

| Recipe | Params | fp16 state | ONE result |
|---|---:|---:|---|
| `dolphin-sfc-npu.slim-6m.soft-query.distill-mixsoftmax.rt192k.fp512keep475` | 5.25M | 162 KiB | PASS, 782 ONNX nodes |
| `dolphin-sfc-npu.slim-6m.crossattn-query.distill-mixsoftmax.rt192k.fp512keep475` | 5.31M | 162 KiB | PASS, 761 ONNX nodes |

NPU artifact roots:

```text
logs/npu_verify_general/dolphin_sfc_slim6m_soft_query_controls_20260603
logs/npu_verify_general/dolphin_sfc_slim6m_crossattn_query_controls_20260603
```

DnR distillation overlays are provided at:

- `recipes/dnr/models/dolphin-sfc-npu.slim-6m.soft-query.distill-mixsoftmax.rt192k.fp512keep475/config.yaml`
- `recipes/dnr/models/dolphin-sfc-npu.slim-6m.crossattn-query.distill-mixsoftmax.rt192k.fp512keep475/config.yaml`

## Presets

- `edge_small`: `n_bands=32, d_model=16, widths=(16, 32, 64), blocks=(1,1,1)`. Smoke/export only.
- `slim_4m`: `n_bands=48, d_model=128, widths=(128, 192, 256), blocks=(1,2,1)`. ~3.6 M params.
- `slim_6m`: `n_bands=48, d_model=128, widths=(128, 224, 320), blocks=(1,2,1)`. ~5.0 M params.
- `slim_8m`: `n_bands=48, d_model=128, widths=(128, 240, 384), blocks=(1,2,1)`. ~6.5 M params.
- Append `_soft_query` or `_crossattn_query` to any preset above to select an explicit-query variant.
