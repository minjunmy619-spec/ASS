# BandDualPathNPU: causal band-split dual-path separator for TV NPU

Date: 2026-09-28

## 1. Purpose

Design a speech / music / effects separator that is better aligned with the TV
NPU target than the existing student families:

- causal, one STFT frame per NPU call, unbounded temporal memory;
- ONE-compilable operator set (Conv2d, depthwise Conv2d, Sigmoid/Tanh,
  elementwise, optional BatchMatMul/Softmax), all tensors 4D with batch first;
- parameters < 6M, < 3 GMAC/s, and every per-call input *and* output
  (frame, masks, state in, state out) inside the 192 KiB DSP quota;
- quantization-friendly numerics (no BatchNorm folding, bounded I/O).

Files:

- `spectral_feature_compression/core/model/band_dualpath_npu.py` (model, host contract, export wrapper, presets, builder)
- `tools/online/export_band_dualpath_npu.py` (export + rule audit + ORT parity + calibration + onecc cfg/run)
- `tools/online/synthetic_stem_benchmark.py` (controlled CPU sanity benchmark)
- `tests/test_band_dualpath_npu.py` (18 tests)
- `recipes/dnr/models/band-dualpath-npu.{medium,medium-noattn,wide}.bs48.onfly.rt192k/config.yaml`

## 2. Existing model families (what is in the repo)

All deployable families share one pipeline: host STFT (`n_fft=2048, hop=512`
at 44.1 kHz, 1025 bins; Dolphin uses 4096/1024), packed real/imag
`[B, 2M, T, F]`, an NPU core that returns masks or masked spectra, host iSTFT.
`ModelWrapper` does the same for training; streaming export goes through
`StreamingStateIOWrapper` or family wrappers; `tools/online/verify_npu_variants.py`
runs export -> onnxsim -> ONE import/optimize/quantize.

| Family | Core structure | Temporal modelling | Norm | Evidence in repo |
|---|---|---|---|---|
| Offline SFC teacher (`bslocoformer`, `crossattn_enc_dec`) | SFC-CA encoder (64 learned band queries attend over 1025 bins), TF-Locoformer (freq + time self-attention, Conv1D SwiGLU FFN), SFC-CA decoder | non-causal attention | RMSGroupNorm | paper model, teacher only |
| Online SFC 2D (`online_*_sfc_2d`: soft-band, query, crossattn-query, hierarchical, FFI, dilated, ConvGRU, hard-band) | soft/hard band routing F->K, conv or ConvGRU separator, band -> bin decoder | causal convs / ConvGRU | RMS over bands | 0.02-1.5M params, too small |
| SFC-small NPU (`sfc_small_*`, Jul 2026) | exact SFC encoder/decoder as per-head MatMul+Softmax over 1025 bins; separator = Conv2d Loco/Macaron blocks at 36-64 bands | causal `(2,1)` convs (lrattn adds a fixed-decay EMA context) | BatchNorm2d folded into conv (cLN variants add running stats) | 1.0-3.8M, 130-360 Circle nodes, ONE PASS, ~2.9 GMAC/s |
| TVConv pyramid (`tvconv_pyramid_npu_separator_2d`) | stride-2 frequency pyramid to F=32, bottleneck temporal blocks, ConvTranspose decoder, speech/music masks + residual SFX | causal conv or ConvGRU/ConvLSTM at the bottleneck only | none (norm-free) | 1.8-4.2M, ~270 raw nodes |
| DolphinSFCNPU slim | stateless band compressor F->48 bands, 3-scale band U-Net, slim blocks = gated causal depthwise `(3,1)` + frequency GLU FFN | 8 blocks x `kt=3` at hop 1024 | RMS over bands | 5.17M, 314 nodes, ONE PASS |
| BandSFCNetNPU | adaptive SFC band compression + cross-attention/soft-query + narrow-band blocks | causal conv | mixed | 2-4M, 444-1514 nodes |
| BandSCNetNPU | SCNet sparse down/up-sampling (low/mid/high branches, stride-2 conv/TConv chains), cross-band + narrow-band blocks, windowed causal attention | `kt=3` + bounded attention W=16 | none/PReLU | 2.3-6.3M, 648-806 nodes, state near 192 KiB |
| EdgeFusionNPU | low-node stack of 1x1 token-capacity layers on 257 bins | causal conv, packed state | - | 5.3M, 168 nodes |
| Source-aware MelBand RoFormer / Loco-CNB students | mel-band compression, per-source heads, Loco blocks | causal conv / cumulative norm | cLN/PCEN | 1000-1800 raw nodes |
| TIGER NPU edge v1/v2, TF-MLPNet | TIGER multi-scale FFI cells / MLP mixers with KV cache | cache/windowed | - | compile references |

## 3. Why a new family (measured, not assumed)

1. **Short temporal context.**  Structural causal receptive field measured by
   the exact input-gradient support of the full training forward (zero-init
   heads randomized first so the probe sees the network, 4 s probe):

   | Model | structural causal context |
   |---|---|
   | `sfc-small-macaron-conv2d-bn` | 0.11 s |
   | `dolphin-sfc-npu.slim-6m.fp512` | 0.44 s |
   | `sfc-small-macaron-lrattn-bn` | unbounded (fixed-decay EMA, rank 2) |
   | **BandDualPathNPU medium** | unbounded (per-band GRU + scene GRU) |

   Speech/music/effects identity is decided over ~0.5-2 s (prosody, beats,
   event envelopes); 0.1-0.4 s of context is a structural quality ceiling.
2. **BatchNorm folding breaks quantization.**  `NPU_QUANT_ISSUE.md` documents
   the BN+ReLU model collapsing even at INT16 while the RMSNorm model is
   lossless; the latest NPU students (all `sfc_small_*_bn`) still use folded
   BN, and most families feed the raw (uncompressed) STFT with unbounded ReLU
   paths, which stretches per-tensor activation ranges.
3. **Compute spent on frequency transport.**  The exact SFC encoder/decoder
   project all 1025 bins every frame (e.g. `d_inner=64` K/V projection alone
   is ~8 MMAC/frame = 0.7 GMAC/s); the macaron separator gets only two blocks
   at 36 bands inside a 2.9 GMAC/s budget.
4. **Coarse low-frequency resolution** (musical36/48 bands) for speech F0 and
   formants, and Dolphin's 4096-sample window adds 93 ms algorithmic latency.

## 4. BandDualPathNPU design

```text
host (DSP):  STFT 2048/512 -> packed (re,im) -> power-law |X|^0.3 compression
             -> band layout: 3 regions [B, 2*M*W_r, 1, K_r]
               R0 bins   0-95   W=4  -> 24 bands ( 86 Hz)  0.00-2.07 kHz
               R1 bins  96-351  W=16 -> 16 bands (345 Hz)  2.07-7.58 kHz
               R2 bins 352-1023 W=84 ->  8 bands (1.8 kHz) 7.58-22.05 kHz
NPU:         per-region 1x1 Conv2d embed -> Concat(K=48) + band position emb
             N x DualPathBlock:
               BandGRU    : RMSNorm -> GRU over time per band (shared) -> 1x1 proj, residual
               SceneGRU   : (every 2nd block) RMSNorm -> band mean -> wide GRU -> FiLM(x*gamma+beta)
               BandAttn   : (every 3rd block) RMSNorm -> 4-head self-attention across the 48 bands of the frame
               BandConvGLU: RMSNorm -> depthwise (1,5) band conv -> value*sigmoid(gate) 1x1 FFN, residual
             RMSNorm -> Split(24,16,8) -> per-region GLU head -> tanh complex masks [B, 2*S*M*W_r, 1, K_r]
host:        unpack masks (bin 1024 reuses bin 1023) -> complex mask x uncompressed STFT -> iSTFT
```

Key decisions:

- **Band split in the host layout.**  The bins of a band are laid out on the
  channel axis by the host, so band-splitting is a plain 1x1 Conv2d (BSRNN
  band-split) with no Slice/Gather/pyramid/cross-attention on the NPU.
  `BandLayout.pack/unpack` is the executable host reference.
- **GRU trained with cuDNN, deployed as convs.**  Training runs `nn.GRU` over
  `B*K` band sequences (fast, exact).  `_gru_step_conv` evaluates the same
  weights as six 1x1 Conv2d + Sigmoid/Tanh + elementwise ops (no Split/Concat).
  Tested equal to `nn.GRU` and full-sequence == frame-streaming to ~5e-7.
- **Scene GRU.**  Band-split trunks spend `K` MACs per weight, so trunk
  parameters are capped by MACs.  The scene GRU runs on the band-mean
  (`[B,C,1,1]`) with a 384-wide state: ~0.6M params per instance for
  ~0.05 GMAC/s, adding long-term scene memory (commentary vs concert vs film).
  FiLM convs are zero-initialized (identity at start).
- **Quantization-friendly numerics.**  RMSNorm per (frame, band) over channels
  written as `x / sqrt(mean(x*x) + eps)`: ONE's `transform_sqrt_div_to_rsqrt_mul`
  plus `fuse_rmsnorm` can turn it into one `RmsNorm` op.  Norm gains and the
  attention `1/sqrt(d)` are folded into the following convs by
  `prepare_for_export_()`.  Inputs are power-law compressed (bounded), GRU
  states are tanh-bounded, masks are tanh-bounded; no BatchNorm anywhere, so
  EMA weights are safe in training.
- **Compact ABI.**  5 inputs / 5 outputs: `x_band0..2`, `band_state [1,H,6,48]`,
  `scene_state [1,384,2,1]` and the three masks plus next states.

## 5. Presets and measured budgets

Budgets from `macs_per_frame()` (convs + attention matmuls, 86.13 frames/s) and
the audit of the simplified ONNX (`model.sim.onnx`) that ONE imports.

| Preset | Params | GMAC/s | State (fp16) | All I/O per call (fp16; uint8 is half) | ONNX nodes | Memory ops | BMM/Softmax |
|---|---:|---:|---:|---:|---:|---:|---|
| `medium` | 1.98M | 2.55 | 46.5 KiB | 109.0 KiB | 340 | 24 | 4 / 2 |
| `medium_noattn` | 1.97M | 2.46 | 46.5 KiB | 109.0 KiB | 302 | 12 | none |
| `wide` | 4.08M | 2.84 | 39.0 KiB | 94.0 KiB | 295 | 23 | 4 / 2 |
| `small` | 0.95M | 1.33 | 31.0 KiB | 78.0 KiB | 304 | 23 | 4 / 2 |

Node counts include 17 unfused RMSNorm patterns (5 ops each); after ONE
`fuse_rmsnorm` each becomes one op (~270 nodes for `medium`).  Memory ops are
the 8 state-row Slices, 3 Concats, 1 Split and, with attention, 4 Transposes
and 8 Reshapes.

`medium` simplified-ONNX op histogram:

```text
Add 68, Concat 3, Conv 101, Div 17, MatMul 4, Mul 44, ReduceMean 19, Reshape 8,
Sigmoid 25, Slice 8, Softmax 2, Split 1, Sqrt 17, Sub 8, Tanh 11, Transpose 4
```

## 6. Rule compliance (AGENTS.md)

| Rule | Status |
|---|---|
| 1/2 basic ops, 2D only | Conv2d / depthwise Conv2d / MatMul(4D) / Softmax / Sigmoid / Tanh / elementwise / Reshape / Transpose / Concat / Split / Slice; every op has a circle-mlir converter |
| 3/4 rank <= 4, batch first | audited on every tensor of the simplified graph; training-only paths fold frames into batch, the export path never does |
| 5 kernel span <= 14 | band conv `(1,5)`; everything else 1x1 |
| 6/7 pooling / TConv / AdaptiveAvgPool | none (the MEAN-free fallback uses AvgPool 1x4 stride 4 and 1x12 stride 1) |
| 8 consts in forward | no generated tensors; only initializers and the RMSNorm eps |
| 9 ScatterND / unflatten | none; audit also rejects Tile, Expand, Gather, ConstantOfShape, Range, Where, Shape, Cast, Pow, Loop/If |
| 10/11 memory ops, node count | 12-24 memory ops, ~300-340 nodes |
| 12 causal | per-band GRU, scene GRU, in-frame attention; streaming == full sequence |
| 13 192 KiB quota | 39-55 KiB (uint8 ABI, default) / 78-109 KiB (fp16) for all inputs + outputs + state in/out; float32 ABI is rejected by the export tool |
| 14 few I/O | 5 in / 5 out |
| 15 < 7M params, < 3 GMAC/s | 0.95-4.08M, 1.33-2.84 GMAC/s |
| 16 ONE limits | softmax on last axis; no grouped (non-depthwise) conv; no strided conv; see section 8 |

## 7. Verification performed in this session

Environment: Python 3.11, torch 2.14 CPU, onnx 1.23, onnxsim 0.7.3,
onnxruntime 1.30.  The ONE toolchain could not be built here (see section 8).

```bash
# 17 tests: layout, budgets, GRU cell == nn.GRU, streaming == full forward
# (raw and folded), chunked state carry, waveform backward, mixture
# consistency, ONNX export + NPU audit + ORT parity, MEAN-free fallback.
python -m pytest -q tests/test_band_dualpath_npu.py

# Export + audit + ORT parity + real sequential calibration records + onecc cfg
python tools/online/export_band_dualpath_npu.py --preset medium --out-dir logs/band_dualpath_npu/medium
python tools/online/export_band_dualpath_npu.py --preset medium_noattn --out-dir logs/band_dualpath_npu/medium_noattn
```

Results: all presets export at opset 13 with `dynamo=False`, pass the checker,
onnxsim and the rule audit with zero violations; ONNX Runtime matches PyTorch
to <= 7.5e-7 over 16 sequential frames with carried state.

Structural receptive field (section 3) was measured with an exact
input-gradient probe on each model's training forward.

Synthetic CPU benchmark (identical data, loss, optimizer and steps for each
model; see section 9 for results and caveats):

```bash
python tools/online/synthetic_stem_benchmark.py --model band_dualpath_medium --steps 500 --batch 4 --seconds 1.5 --out logs/synth_bench
python tools/online/synthetic_stem_benchmark.py --model dolphin_slim6m_fp512 --steps 500 --batch 4 --seconds 1.5 --out logs/synth_bench
python tools/online/synthetic_stem_benchmark.py --model sfc_macaron_lrattn_bn --steps 500 --batch 4 --seconds 1.5 --out logs/synth_bench
```

## 8. ONE compile: to run on the WSL machine

Not executed here: the ONE checkout's latest commit (#16489) removed
`compiler/`, and GitHub archive downloads needed to build externals are
blocked in this container.  The export tool writes everything needed and runs
ONE when `onecc` is on `PATH`:

```bash
export ONE_CMDS=/home/cmj/works/ONE/build/compiler/one-cmds
source "$ONE_CMDS/venv/bin/activate"; export PATH="$ONE_CMDS:$PATH"
# (LD_LIBRARY_PATH as in OPERATION_MANUAL_PYTORCH_TO_ONE_NPU.md section 2.2)

.venv/bin/python tools/online/export_band_dualpath_npu.py \
  --config recipes/dnr/models/band-dualpath-npu.medium.bs48.onfly.rt192k/config.yaml \
  [--ckpt <lightning.ckpt>] [--calib-wav <44.1k mix.wav>] \
  --out-dir logs/band_dualpath_npu/medium --run-one
cat logs/band_dualpath_npu/medium/report.json   # "one": {"import": true, "optimize": true, "quantize": true}
```

The generated `onecc.cfg` imports `model.sim.onnx` and optimizes with
`convert_nchw_to_nhwc`, `transform_sqrt_div_to_rsqrt_mul`, `fuse_rmsnorm`,
`replace_non_const_fc_with_batch_matmul` and the repo's low-latency cleanup
flags, then quantizes uint8 per-channel with calibration records that carry
real sequential GRU state (not random state).

Boundary type (`--io-type`, default `uint8`, matching the repo onecc template):
the ABI bytes per call (frame inputs + masks + state in + state out) are

| io type | `medium` / `medium_noattn` | fits 192 KiB |
|---|---:|---|
| uint8 (default) | 55,808 B (54.5 KiB) | yes |
| int16 | 111,616 B (109 KiB) | yes |
| float32 | 223,232 B (218 KiB) | no - the tool reports a violation and exits 1 |

With uint8 I/O the carried GRU states are requantized every frame; they are
tanh-bounded to [-1, 1], so the step is ~1/128.  If long-context drift shows up
in the float-vs-quantized comparison, use `--io-type int16`.

Known risks and fallbacks:

- If `fuse_rmsnorm` or the quantizer rejects `RMS_NORM`, drop that flag; the
  pattern stays as Mul/Mean/Add/Rsqrt/Mul.
- If MEAN is slow or unsupported on the backend, export with
  `--rms-reduce conv` (channel mean as a fixed 1x1 Conv2d, band mean as staged
  AvgPool2d); numerically identical, tested.
- `luci-interpreter` cannot evaluate quantized BatchMatMul (known from the
  macaron runs); `medium_noattn` contains no BatchMatMul/Softmax/Transpose.

## 9. Synthetic benchmark results

See the section appended after the runs finish.

## 10. Next steps

1. Run section 8 on the WSL box for `medium` and `medium_noattn`; record Circle
   op counts and quantized-vs-float mask error with `circle-eval-diff`.
2. Train `medium` and `medium_noattn` with the TV on-the-fly profile recipe
   next to `dolphin-sfc-npu.slim-6m` and `tvconv-pyramid-sourceaware-sfclite-convgru`
   under the same data and validation manifest.
3. Host implementation: `compress_packed_spectrum`, `BandLayout.pack/unpack`
   and `BandDualPathNPUModel.apply_masks` are the reference; bin 1024 copies
   the bin-1023 mask.
4. Optional: teacher distillation (mask/waveform) with the existing
   `TeacherStudentDistillationTask`, and per-source loss weights.
