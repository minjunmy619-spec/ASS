#!/usr/bin/env python3
"""Export BandDualPathNPU to streaming ONNX, audit it against the NPU rules, and prepare ONE inputs.

Steps (all artifacts land in ``--out-dir``):

1. build the core from a preset / recipe config, optionally load a checkpoint;
2. fold RMSNorm gains and the attention scale into convolutions;
3. ``torch.onnx.export(..., dynamo=False)`` at opset 11-14 with batch 1, one frame;
4. onnxsim with fixed input shapes (``model.sim.onnx``);
5. rule audit (op allowlist, rank <= 4, batch-first, kernel span, strides, groups,
   softmax axis, static shapes, I/O bytes, node / memory-op counts);
6. ONNX Runtime vs PyTorch parity over sequential frames with carried state;
7. calibration records with *real sequential* state (``calib/``, ``calib_list.txt``)
   and a ``onecc`` config; ``--run-one`` runs ``one-create-quant-dataset`` + ``onecc``
   when the ONE tools are on PATH and checks every stage artifact.

Example::

    python tools/online/export_band_dualpath_npu.py --preset medium --out-dir logs/band_dualpath_npu/medium
    python tools/online/export_band_dualpath_npu.py \
        --config recipes/dnr/models/band-dualpath-npu.medium.onfly.rt192k/config.yaml \
        --ckpt path/to/epoch=xxxx.ckpt --calib-wav path/to/mix.wav --out-dir logs/band_dualpath_npu/trained --run-one
"""

from __future__ import annotations

from typing import Any

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import onnx  # noqa: E402
from onnx import numpy_helper, shape_inference  # noqa: E402

from spectral_feature_compression.core.model.band_dualpath_npu import (  # noqa: E402
    DEFAULT_REGIONS,
    PRESETS,
    REGIONS_SR24K_R5,
    BandDualPathNPUExportWrapper,
    BandDualPathNPUModel,
)

# Ops the model is allowed to contain after simplification.  All of them have a
# converter in ONE circle-mlir (compiler/circle-mlir/lib/pass/src/ops).
ALLOWED_OPS = {
    "Conv",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Sqrt",
    "ReduceMean",
    "Sigmoid",
    "Tanh",
    "MatMul",
    "Softmax",
    "Reshape",
    "Transpose",
    "Concat",
    "Split",
    "Slice",
    "AveragePool",
}
MEMORY_OPS = {"Reshape", "Transpose", "Concat", "Split", "Slice"}
FORBIDDEN_HINT = {
    "Tile", "Expand", "ConstantOfShape", "ScatterND", "Gather", "Range", "Where", "Loop", "If", "Scan",
    "GRU", "LSTM", "RNN", "Shape", "Cast", "Pow", "BatchNormalization", "Unsqueeze", "Squeeze",
}
# Boundary tensor types accepted by one-quantize (input_type / output_type).
IO_TYPES = {"uint8": torch.uint8, "int16": torch.int16, "float32": torch.float32}
DSP_QUOTA_BYTES = 192 * 1024
DEFAULT_ONE_OPTIMIZE_FLAGS = (
    "replace_non_const_fc_with_batch_matmul",
    "convert_nchw_to_nhwc",
    "nchw_to_nhwc_input_shape",
    "nchw_to_nhwc_output_shape",
    "transform_sqrt_div_to_rsqrt_mul",
    "fuse_rmsnorm",
    "fuse_add_with_conv",
    "fuse_mul_with_conv",
    "fuse_activation_function",
    "remove_redundant_transpose",
    "remove_unnecessary_transpose",
    "remove_redundant_reshape",
    "remove_unnecessary_reshape",
    "remove_unnecessary_slice",
    "remove_unnecessary_split",
    "remove_unnecessary_add",
    "remove_unnecessary_mul",
    "remove_duplicate_const",
    "common_subexpression_elimination",
)


REGION_CHOICES = {"default": DEFAULT_REGIONS, "sr24k_r5": REGIONS_SR24K_R5}


def build_model(args: argparse.Namespace) -> BandDualPathNPUModel:
    """Build the model; also sets ``args.fs``, ``args.n_fft`` and ``args.hop`` (from the recipe when given)."""
    if args.config:
        from omegaconf import OmegaConf

        cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=False)
        model_cfg = dict(cfg["task"]["model"])
        args.fs = int(model_cfg.get("fs", args.fs))
        args.n_fft = int(model_cfg.get("n_fft", args.n_fft))
        args.hop = int(model_cfg.get("hop_length", args.hop))
        preset = model_cfg.pop("preset", None)
        for key in ("_target_", "n_fft", "hop_length", "fs", "scaling", "css_segment_size", "css_shift_size",
                    "css_batch_size"):
            model_cfg.pop(key, None)
        kwargs = dict(PRESETS[preset]) if preset else {}
        kwargs.update(model_cfg)
        model = BandDualPathNPUModel(n_freq=args.n_fft // 2 + 1, **kwargs)
    else:
        model = BandDualPathNPUModel(
            n_freq=args.n_fft // 2 + 1,
            regions=REGION_CHOICES[args.regions],
            io_layout=args.io_layout,
            mask_points=args.mask_points,
            **PRESETS[args.preset],
        )
    if args.ckpt:
        state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        state = state.get("state_dict", state)
        prefixes = ("model.model.", "model.") if args.no_ema else ("ema_model.module.model.", "model.model.", "model.")
        prefix = next((p for p in prefixes if any(k.startswith(p + "core.") for k in state)), "")
        filtered = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix + "core.")}
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"Checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    return model.eval()


def stream_records(
    model: BandDualPathNPUModel, spec: torch.Tensor, n_records: int, stride: int
) -> list[tuple[torch.Tensor, ...]]:
    """Run the float core frame by frame and keep (inputs incl. carried state) every ``stride`` frames."""
    core = model.core
    feats = model.host_features(spec)
    state = core.init_stream_state(1)
    records = []
    with torch.no_grad():
        for t in range(spec.shape[-1]):
            parts = [f[:, :, t : t + 1] for f in feats]
            if t % stride == stride - 1:
                records.append((*parts, *(s for s in state if s is not None)))
                if len(records) >= n_records:
                    break
            _, state = core.forward_stream(parts, state)
    return records


def calibration_spectrum(args: argparse.Namespace, n_frames: int) -> torch.Tensor:
    window = torch.hann_window(args.n_fft)
    if args.calib_wav:
        import soundfile as sf

        wav, sr = sf.read(args.calib_wav, dtype="float32", always_2d=True)
        if sr != args.fs:
            raise ValueError(f"Calibration wav must be {args.fs} Hz (the model sample rate), got {sr}")
        wav = torch.from_numpy(wav.mean(axis=1))
    else:
        # Deterministic stand-in: harmonic tones + noise bursts at varying level.
        gen = torch.Generator().manual_seed(0)
        n = n_frames * args.hop + args.n_fft
        t = torch.arange(n) / args.fs
        env = 0.5 + 0.5 * torch.sin(2 * torch.pi * 0.7 * t)
        tones = sum(torch.sin(2 * torch.pi * f0 * k * t) / k for f0 in (110.0, 220.0, 330.0) for k in range(1, 6))
        burst = args.fs // 10
        wav = 0.05 * env * tones + 0.02 * torch.randn(n, generator=gen) * (
            torch.rand(n // burst + 1, generator=gen).repeat_interleave(burst)[:n]
        )
    spec = torch.stft(wav, args.n_fft, args.hop, window=window, return_complex=True)[..., :n_frames]
    return spec.reshape(1, 1, spec.shape[-2], spec.shape[-1])


def export_onnx(wrapper: BandDualPathNPUExportWrapper, path: Path, opset: int) -> None:
    inputs = wrapper.example_inputs(1)
    in_names, out_names = wrapper.io_names()
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            inputs,
            str(path),
            opset_version=opset,
            input_names=in_names,
            output_names=out_names,
            do_constant_folding=True,
            dynamo=False,
        )


def simplify(src: Path, dst: Path) -> None:
    from onnxsim import simplify as onnxsim_simplify

    model = onnx.load(str(src))
    simplified, ok = onnxsim_simplify(model)
    if not ok:
        raise RuntimeError("onnxsim could not validate the simplified model")
    onnx.save(simplified, str(dst))


def _attr(node: onnx.NodeProto, name: str, default: Any = None) -> Any:
    for attr in node.attribute:
        if attr.name == name:
            return onnx.helper.get_attribute_value(attr)
    return default


def audit(path: Path, io_bytes: int) -> dict[str, Any]:
    model = shape_inference.infer_shapes(onnx.load(str(path)), strict_mode=True)
    graph = model.graph
    initializers = {init.name: init for init in graph.initializer}
    shapes: dict[str, list[int | str]] = {}
    for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
        dims = [d.dim_value if d.HasField("dim_value") else (d.dim_param or "?") for d in vi.type.tensor_type.shape.dim]
        shapes[vi.name] = dims

    violations: list[str] = []
    ops = Counter(node.op_type for node in graph.node)
    for op in sorted(ops):
        if op not in ALLOWED_OPS:
            violations.append(f"op {op} x{ops[op]} is outside the allowlist")
    for name, dims in shapes.items():
        if any(not isinstance(d, int) or d <= 0 for d in dims):
            violations.append(f"tensor {name} has a non-static shape {dims}")
        if len(dims) > 4:
            violations.append(f"tensor {name} has rank {len(dims)} > 4")
        if len(dims) == 4 and dims[0] != 1:
            violations.append(f"4D tensor {name} does not keep batch=1 first: {dims}")

    for node in graph.node:
        if node.op_type == "Conv":
            kernel = _attr(node, "kernel_shape")
            dilation = _attr(node, "dilations", [1] * len(kernel))
            strides = _attr(node, "strides", [1] * len(kernel))
            group = _attr(node, "group", 1)
            for k, d in zip(kernel, dilation):
                if (k - 1) * d > 14:
                    violations.append(f"{node.name}: kernel {kernel} dilation {dilation} exceeds span 14")
            if any(s != 1 for s in strides):
                violations.append(f"{node.name}: conv stride {strides} (model is designed stride-free)")
            w = initializers.get(node.input[1])
            if w is None:
                violations.append(f"{node.name}: conv weight is not a constant initializer")
            elif group != 1:
                out_ch, in_per_group = w.dims[0], w.dims[1]
                if not (in_per_group == 1 and out_ch == group):
                    violations.append(f"{node.name}: grouped (non-depthwise) conv group={group} splits in ONE")
        elif node.op_type == "AveragePool":
            kernel = _attr(node, "kernel_shape")
            strides = _attr(node, "strides", [1] * len(kernel))
            if any(k > 15 for k in kernel) or any(st not in (1, 2, 4) for st in strides):
                violations.append(f"{node.name}: AveragePool kernel {kernel} stride {strides} outside NPU limits")
        elif node.op_type == "Softmax":
            axis = _attr(node, "axis", -1)
            rank = len(shapes.get(node.input[0], []))
            if axis not in (-1, rank - 1):
                violations.append(f"{node.name}: softmax axis {axis} is not the last axis")
        elif node.op_type == "MatMul":
            ranks = [len(shapes.get(i, initializers[i].dims if i in initializers else [])) for i in node.input]
            if any(r != 4 for r in ranks):
                violations.append(f"{node.name}: MatMul operand ranks {ranks} (expected 4D BatchMatMul)")
        elif node.op_type == "ReduceMean":
            axes = _attr(node, "axes")
            if axes is None and len(node.input) > 1 and node.input[1] in initializers:
                axes = numpy_helper.to_array(initializers[node.input[1]]).tolist()
            if _attr(node, "keepdims", 1) != 1:
                violations.append(f"{node.name}: ReduceMean must keep dims")
            if axes not in ([1], [3]):
                violations.append(f"{node.name}: ReduceMean axes {axes} (expected channel or band axis)")

    const_bytes = sum(int(np.prod(init.dims)) * 4 for init in graph.initializer)
    return {
        "nodes": len(graph.node),
        "ops": dict(sorted(ops.items())),
        "memory_ops": sum(ops[o] for o in MEMORY_OPS),
        "inputs": {vi.name: shapes[vi.name] for vi in graph.input},
        "outputs": {vi.name: shapes[vi.name] for vi in graph.output},
        "io_bytes": io_bytes,
        "initializer_bytes_fp32": const_bytes,
        "violations": violations,
        "forbidden_present": sorted(set(ops) & FORBIDDEN_HINT),
    }


def ort_parity(path: Path, wrapper: BandDualPathNPUExportWrapper, records: list[tuple[torch.Tensor, ...]]) -> float:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]
    worst = 0.0
    with torch.no_grad():
        for record in records:
            ref = wrapper(*record)
            got = sess.run(None, {n: t.numpy() for n, t in zip(names, record)})
            worst = max(worst, max(float(np.abs(r.numpy() - g).max()) for r, g in zip(ref, got)))
    return worst


def write_calibration(records: list[tuple[torch.Tensor, ...]], out_dir: Path) -> Path:
    calib_dir = out_dir / "calib"
    calib_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for ridx, record in enumerate(records):
        paths = []
        for iidx, tensor in enumerate(record):
            p = calib_dir / f"rec{ridx:04d}_in{iidx}.npy"
            np.save(p, tensor.numpy().astype(np.float32))
            paths.append(str(p.resolve()))
        lines.append(" ".join(paths))
    list_path = out_dir / "calib_list.txt"
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return list_path


def write_onecc_cfg(
    out_dir: Path, onnx_path: Path, calib_h5: Path, granularity: str, io_type: str = "uint8"
) -> Path:
    lines = [
        "[Environment]", 'ONECC_ENV="ONECC"', "",
        "[backend]", "target=", "",
        "[onecc]",
        "one-import-tf=False", "one-import-tflite=False", "one-import-bcq=False",
        "one-import-onnx=True", "one-optimize=True", "one-quantize=True",
        "one-partition=False", "one-pack=False", "one-codegen=False", "one-profile=False", "one-infer=False", "",
        "[one-import-onnx]", f"input_path={onnx_path.resolve()}", f"output_path={(out_dir / 'model.circle').resolve()}",
        "dynamic_batch_to_single_batch=True", "",
        "[one-optimize]", f"input_path={(out_dir / 'model.circle').resolve()}",
        f"output_path={(out_dir / 'model.opt.circle').resolve()}",
        *(f"{flag}=True" for flag in DEFAULT_ONE_OPTIMIZE_FLAGS), "",
        "[one-quantize]", f"input_path={(out_dir / 'model.opt.circle').resolve()}",
        f"output_path={(out_dir / 'model.q.circle').resolve()}",
        f"input_data={calib_h5.resolve()}", "input_data_format=h5",
        "quantized_dtype=uint8", f"granularity={granularity}", f"input_type={io_type}", f"output_type={io_type}",
    ]
    cfg = out_dir / "onecc.cfg"
    cfg.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return cfg


def run_one(out_dir: Path, list_path: Path, cfg: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    calib_h5 = out_dir / "calib.h5"
    for name, cmd in (
        ("dataset", ["one-create-quant-dataset", "-i", "numpy", "-l", str(list_path), "-p", str(calib_h5)]),
        ("onecc", ["onecc", "-C", str(cfg)]),
    ):
        proc = subprocess.run(cmd, cwd=out_dir, capture_output=True, text=True)
        (out_dir / f"{name}.log").write_text(proc.stdout + proc.stderr, encoding="utf-8")
        result[f"{name}_rc"] = proc.returncode
        if proc.returncode != 0:
            break
    for stage, artifact in (("import", "model.circle"), ("optimize", "model.opt.circle"), ("quantize", "model.q.circle")):
        result[stage] = (out_dir / artifact).exists()
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--preset", default="medium", choices=sorted(PRESETS))
    src.add_argument("--config", type=Path, help="recipe config.yaml with task.model")
    p.add_argument("--fs", type=int, default=44100, help="sample rate (taken from --config when given)")
    p.add_argument("--n-fft", type=int, default=2048, help="STFT size (taken from --config when given)")
    p.add_argument("--hop", type=int, default=512, help="STFT hop (taken from --config when given)")
    p.add_argument("--regions", default="default", choices=sorted(REGION_CHOICES), help="band layout for --preset")
    p.add_argument("--io-layout", default="regions", choices=["regions", "slots"], help="ABI layout for --preset")
    p.add_argument("--mask-points", type=int, default=16, help="mask points per band for --io-layout slots")
    p.add_argument("--ckpt", type=Path, help="Lightning checkpoint (EMA weights are used when present)")
    p.add_argument("--no-ema", action="store_true", help="use the raw instead of the EMA weights of --ckpt")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--opset", type=int, default=13, choices=range(11, 15))
    p.add_argument("--calib-wav", type=Path, help="44.1 kHz wav for calibration (default: synthetic signal)")
    p.add_argument("--calib-records", type=int, default=64)
    p.add_argument("--calib-stride", type=int, default=4, help="keep one record every N streamed frames")
    p.add_argument("--granularity", default="channel", choices=["channel", "layer"])
    p.add_argument(
        "--io-type", default="uint8", choices=sorted(IO_TYPES),
        help="quantized model boundary type; float32 I/O exceeds the 192 KiB quota for the medium presets",
    )
    p.add_argument(
        "--rms-reduce", default="mean", choices=["mean", "conv"],
        help="RMSNorm channel mean as ReduceMean (fusable to RmsNorm) or as a fixed 1x1 Conv2d fallback",
    )
    p.add_argument("--run-one", action="store_true", help="run one-create-quant-dataset + onecc if on PATH")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(args)
    model.core = copy.deepcopy(model.core).prepare_for_export_().set_rms_reduce_(args.rms_reduce)
    wrapper = BandDualPathNPUExportWrapper(model.core).eval()

    raw_path, sim_path = out_dir / "model.onnx", out_dir / "model.sim.onnx"
    export_onnx(wrapper, raw_path, args.opset)
    onnx.checker.check_model(onnx.load(str(raw_path)))
    simplify(raw_path, sim_path)

    n_frames = args.calib_records * args.calib_stride + 1
    records = stream_records(model, calibration_spectrum(args, n_frames), args.calib_records, args.calib_stride)
    io = model.core.io_size_bytes(dtype=IO_TYPES[args.io_type])
    report: dict[str, Any] = {
        "source": str(args.config or f"preset:{args.preset}"),
        "checkpoint": str(args.ckpt) if args.ckpt else None,
        "params": sum(t.numel() for t in model.core.parameters()),
        "stft": {"fs": args.fs, "n_fft": args.n_fft, "hop": args.hop},
        "io_layout": model.core.io_layout,
        # Host contract: region (start_bin, end_bin, bins_per_band); slots layout also needs slot_width / mask_points.
        "host_layout": {
            "regions": [[r.start, r.end, r.width] for r in model.core.layout.regions],
            "n_bands": model.core.n_bands,
            "slot_width": model.core.layout.slot_width,
            "mask_points": model.core.mask_points if model.core.io_layout == "slots" else None,
            "compress_exponent": model.compress_exponent,
        },
        "gmac_per_s": model.core.macs_per_frame() * args.fs / args.hop / 1e9,
        "io_type": args.io_type,
        "io_bytes": io,
        # model.sim.onnx is what ONE imports; the raw graph is only summarized.
        "raw_ops": dict(sorted(Counter(n.op_type for n in onnx.load(str(raw_path)).graph.node).items())),
        "sim": audit(sim_path, io["total"]),
        "ort_max_abs_err_raw": ort_parity(raw_path, wrapper, records[:16]),
        "ort_max_abs_err_sim": ort_parity(sim_path, wrapper, records[:16]),
    }
    list_path = write_calibration(records, out_dir)
    cfg = write_onecc_cfg(out_dir, sim_path, out_dir / "calib.h5", args.granularity, args.io_type)
    report["onecc_cfg"] = str(cfg)
    if args.run_one:
        if shutil.which("onecc") and shutil.which("one-create-quant-dataset"):
            report["one"] = run_one(out_dir, list_path, cfg)
        else:
            report["one"] = "skipped: onecc / one-create-quant-dataset not on PATH"

    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    sim = report["sim"]
    print(json.dumps({k: report[k] for k in ("source", "stft", "io_layout", "params", "gmac_per_s", "io_type", "io_bytes")}, indent=2))
    print(f"sim nodes={sim['nodes']} memory_ops={sim['memory_ops']} ops={sim['ops']}")
    print(f"ORT parity raw={report['ort_max_abs_err_raw']:.2e} sim={report['ort_max_abs_err_sim']:.2e}")
    if "one" in report:
        print(f"ONE: {report['one']}")
    violations = list(report["sim"]["violations"])
    if io["total"] > DSP_QUOTA_BYTES:
        violations.append(
            f"{args.io_type} ABI (frame + masks + state in/out) is {io['total']} B > {DSP_QUOTA_BYTES} B DSP quota"
        )
    for v in violations:
        print("VIOLATION:", v)
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
