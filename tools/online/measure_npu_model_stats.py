#!/usr/bin/env python3

from __future__ import annotations

import contextlib
import csv
from dataclasses import dataclass
from datetime import datetime
import io
import json
from pathlib import Path
import re
from argparse import ArgumentParser
import shutil
import subprocess
import sys
from typing import Any, Callable

import onnx
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spectral_feature_compression.utils.onnx_streaming import StreamingStateIOWrapper, flatten_tensor_tree
from tools.online.export_onnx_online_model import load_export_core


DEFAULT_ONNX_MLIR = Path("/workdir/onnx-mlir/build/Debug/bin/onnx-mlir")
MAC_OP_KEYS = ("conv", "mm", "bmm", "matmul", "addmm")


READY_SUITE = [
    {
        "target": "tiger-edge",
        "label": "TIGERNPUEdgeV1",
    },
    {
        "target": "dolphin",
        "label": "DolphinSFCNPU edge_small",
        "dolphin_preset": "edge_small",
        "n_chan": 1,
        "freqs": 1025,
    },
    {
        "target": "tf-mlpnet",
        "label": "TF-MLPNet TIGEREdgeMLP 24ch",
    },
    {
        "target": "online",
        "label": "SFC soft-query rt192k fp512keep475",
        "model_path": "recipes/musdb18hq/models/online-soft-band-query-sfc2d.rt192k.fp512keep475.causal24dim.6l.64b/config.yaml",
        "n_chan": 2,
    },
    {
        "target": "online",
        "label": "SFC crossattn-query rt192k fp512keep475",
        "model_path": "recipes/musdb18hq/models/online-crossattn-query-sfc2d.rt192k.fp512keep475.causal24dim.6l.64b/config.yaml",
        "n_chan": 2,
    },
    {
        "target": "online",
        "label": "SFC soft-dilated maxdil rt192k",
        "model_path": "recipes/musdb18hq/models/online-soft-band-dilated-sfc2d.rt192k.maxdil.causal16dim.6l.64b/config.yaml",
        "n_chan": 2,
    },
    {
        "target": "online",
        "label": "DNR parallel-FFI rt192k",
        "model_path": "recipes/dnr/models/online-hierarchical-soft-band-parallel-ffi-sfc2d.rt192k.speech-lowfreq-narrow.causal20dim.0-1-1l.128-96-48b/config.yaml",
        "n_chan": 1,
    },
]


SUMMARY_FIELDS = [
    "label",
    "target",
    "source",
    "params",
    "param_fp16_kib",
    "param_fp32_kib",
    "state_fp16_kib",
    "state_fp32_kib",
    "frames_per_call",
    "sample_rate",
    "hop_length",
    "frame_rate",
    "mac_per_call",
    "mac_per_frame",
    "gmac_per_s",
    "onnx_nodes",
    "onnx_unique_ops",
    "onnx_initializer_fp32_kib",
    "mlir_ok",
    "mlir_ops",
    "mlir_size_kib",
    "onnx_path",
    "mlir_path",
    "error",
]


@dataclass
class ExportedModel:
    label: str
    target: str
    source: str
    module: torch.nn.Module
    export_module: torch.nn.Module
    export_inputs: tuple[torch.Tensor, ...]
    onnx_path: Path
    input_names: list[str]
    output_names: list[str]
    frames_per_call: int
    state_fp16_bytes: int
    state_fp32_bytes: int
    param_count: int
    param_fp16_bytes: int
    param_fp32_bytes: int
    opset: int


def quiet_call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        return fn(*args, **kwargs)


def sanitize_filename(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return clean.strip("_") or "model"


def dtype_nbytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def tensor_tree_bytes(state: Any, dtype: torch.dtype) -> int:
    flat, _ = flatten_tensor_tree(state)
    return sum(int(t.numel()) * dtype_nbytes(dtype) for t in flat)


def parameter_count(module: torch.nn.Module) -> int:
    return sum(int(p.numel()) for p in module.parameters())


def parameter_bytes(module: torch.nn.Module, dtype: torch.dtype) -> int:
    element_size = dtype_nbytes(dtype)
    total = 0
    for tensor in list(module.parameters()) + list(module.buffers()):
        total += int(tensor.numel()) * element_size
    return total


def format_float(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        return str(value)
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def kib(num_bytes: int | float | None) -> float | None:
    if num_bytes is None:
        return None
    return float(num_bytes) / 1024.0


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p


def profile_macs(
    module: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    *,
    include_elementwise: bool = False,
) -> tuple[float, dict[str, int]]:
    module.eval()
    with torch.no_grad():
        module(*inputs)

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) as prof:
        with torch.no_grad():
            module(*inputs)

    flops = 0
    by_key: dict[str, int] = {}
    for event in prof.key_averages():
        event_flops = int(event.flops or 0)
        if not event_flops:
            continue
        key = event.key.lower()
        if include_elementwise or any(op_key in key for op_key in MAC_OP_KEYS):
            flops += event_flops
            by_key[event.key] = event_flops
    return flops / 2.0, by_key


def onnx_stats(path: Path) -> dict[str, Any]:
    model = onnx.load(str(path))
    op_counts: dict[str, int] = {}
    for node in model.graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
    initializer_bytes = 0
    for initializer in model.graph.initializer:
        dims = initializer.dims if initializer.dims else [1]
        count = 1
        for dim in dims:
            count *= int(dim)
        initializer_bytes += count * dtype_nbytes(torch.float32)
    return {
        "onnx_nodes": len(model.graph.node),
        "onnx_unique_ops": len(op_counts),
        "onnx_op_counts": dict(sorted(op_counts.items())),
        "onnx_top_ops": dict(sorted(op_counts.items(), key=lambda item: (-item[1], item[0]))[:12]),
        "onnx_initializer_fp32_bytes": initializer_bytes,
    }


def count_mlir_ops(path: Path) -> tuple[int, dict[str, int]]:
    op_re = re.compile(
        r'^\s*(?:%[\w\d_,\s:]+?\s*=\s*)?(?:"[A-Za-z0-9_.]+"|[A-Za-z][A-Za-z0-9_]*\.[A-Za-z0-9_]+)\b'
    )
    skip_prefixes = ("module ", "func.func ", "return")
    total = 0
    dialect_counts: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("//") or stripped.startswith(skip_prefixes):
            continue
        match = op_re.match(line)
        if match is None:
            continue
        token = match.group(0).split("=")[-1].strip().strip('"')
        dialect = token.split(".", 1)[0] if "." in token else token
        total += 1
        dialect_counts[dialect] = dialect_counts.get(dialect, 0) + 1
    return total, dict(sorted(dialect_counts.items(), key=lambda item: (-item[1], item[0])))


def resolve_onnx_mlir(cli_path: Path | None) -> Path | None:
    if cli_path is not None:
        return cli_path if cli_path.is_file() else None
    path_candidate = shutil.which("onnx-mlir")
    if path_candidate:
        return Path(path_candidate)
    if DEFAULT_ONNX_MLIR.is_file():
        return DEFAULT_ONNX_MLIR
    return None


def compile_mlir(onnx_path: Path, output_stub: Path, onnx_mlir: Path, timeout: int) -> dict[str, Any]:
    proc = subprocess.run(
        [str(onnx_mlir), "--EmitMLIR", str(onnx_path), "-o", str(output_stub)],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        return {
            "mlir_ok": False,
            "mlir_error": ((proc.stderr or "") + "\n" + (proc.stdout or ""))[-2000:],
        }

    mlir_path = output_stub.with_name(output_stub.name + ".onnx.mlir")
    if not mlir_path.is_file():
        blob = (proc.stdout or "") + "\n" + (proc.stderr or "")
        match = re.search(r'Generated\s+"([^"]+)"', blob)
        if match:
            mlir_path = Path(match.group(1))
    if not mlir_path.is_file():
        return {
            "mlir_ok": False,
            "mlir_error": "onnx-mlir returned success but the emitted MLIR file was not found.",
        }

    mlir_ops, dialect_counts = count_mlir_ops(mlir_path)
    return {
        "mlir_ok": True,
        "mlir_path": str(mlir_path),
        "mlir_ops": mlir_ops,
        "mlir_size_bytes": mlir_path.stat().st_size,
        "mlir_dialect_counts": dialect_counts,
        "mlir_top_dialects": dict(list(dialect_counts.items())[:10]),
    }


def export_to_onnx(exported: ExportedModel) -> None:
    exported.onnx_path.parent.mkdir(parents=True, exist_ok=True)
    exported.export_module.eval()
    with torch.no_grad():
        torch.onnx.export(
            exported.export_module,
            exported.export_inputs,
            str(exported.onnx_path),
            export_params=True,
            opset_version=exported.opset,
            do_constant_folding=True,
            input_names=exported.input_names,
            output_names=exported.output_names,
            dynamic_axes=None,
            dynamo=False,
        )
    onnx.checker.check_model(onnx.load(str(exported.onnx_path)))


def build_online_export(args: Any, spec: dict[str, Any], out_dir: Path) -> ExportedModel:
    model_path = resolve_path(spec.get("model_path") or args.model_path)
    label = spec.get("label") or args.label or model_path.parent.name
    frames = int(spec.get("frames") or args.frames)

    core, source_mode = quiet_call(load_export_core, model_path, args.device)
    core = core.to(args.device).eval()
    n_chan = int(spec.get("n_chan") or args.n_chan or getattr(core, "n_chan", 2))
    n_freq = getattr(core, "n_freq", None)
    if n_freq is None:
        raise RuntimeError(f"Could not infer n_freq from {type(core).__name__}; pass a supported online core.")

    state = core.init_stream_state(batch_size=1, device=args.device, dtype=torch.float32)
    flat_state, _ = flatten_tensor_tree(state)
    wrapper = StreamingStateIOWrapper(core, batch_size=1, device=args.device, dtype=torch.float32).to(args.device).eval()
    dummy = torch.randn(1, 2 * n_chan, frames, int(n_freq), device=args.device, dtype=torch.float32)
    inputs = (dummy, *flat_state)
    return ExportedModel(
        label=label,
        target="online",
        source=f"{source_mode}:{model_path.relative_to(REPO_ROOT) if model_path.is_relative_to(REPO_ROOT) else model_path}",
        module=core,
        export_module=wrapper,
        export_inputs=inputs,
        onnx_path=out_dir / f"{sanitize_filename(label)}.onnx",
        input_names=["x", *[f"state_{idx}" for idx in range(len(flat_state))]],
        output_names=["y", *[f"next_state_{idx}" for idx in range(len(flat_state))]],
        frames_per_call=frames,
        state_fp16_bytes=tensor_tree_bytes(state, torch.float16),
        state_fp32_bytes=tensor_tree_bytes(state, torch.float32),
        param_count=parameter_count(core),
        param_fp16_bytes=parameter_bytes(core, torch.float16),
        param_fp32_bytes=parameter_bytes(core, torch.float32),
        opset=args.online_opset,
    )


def build_dolphin_export(args: Any, spec: dict[str, Any], out_dir: Path) -> ExportedModel:
    from DolphinSFCNPU import DolphinSFCNPUStreamingExportWrapper, build_dolphin_sfc_npu_preset

    preset = spec.get("dolphin_preset") or args.dolphin_preset
    label = spec.get("label") or args.label or f"DolphinSFCNPU {preset}"
    n_freq = int(spec.get("freqs") or args.freqs)
    n_chan = int(spec.get("n_chan") or args.n_chan or 1)
    frames = int(spec.get("frames") or args.frames)
    model = build_dolphin_sfc_npu_preset(
        preset,
        n_freq=n_freq,
        n_fft=args.n_fft,
        sample_rate=args.sample_rate,
        n_chan=n_chan,
    ).to(args.device).eval()
    state = model.init_stream_state(batch_size=1, device=args.device, dtype=torch.float32)
    flat_state, _ = flatten_tensor_tree(state)
    wrapper = DolphinSFCNPUStreamingExportWrapper(model, batch_size=1, dtype=torch.float32).to(args.device).eval()
    dummy = torch.randn(1, 2 * n_chan, frames, model.n_freq, device=args.device, dtype=torch.float32)
    inputs = (dummy, *flat_state)
    return ExportedModel(
        label=label,
        target="dolphin",
        source=f"DolphinSFCNPU preset={preset}",
        module=model,
        export_module=wrapper,
        export_inputs=inputs,
        onnx_path=out_dir / f"{sanitize_filename(label)}.onnx",
        input_names=["x", *[f"state_{idx}" for idx in range(len(flat_state))]],
        output_names=["y", *[f"next_state_{idx}" for idx in range(len(flat_state))]],
        frames_per_call=frames,
        state_fp16_bytes=tensor_tree_bytes(state, torch.float16),
        state_fp32_bytes=tensor_tree_bytes(state, torch.float32),
        param_count=parameter_count(model),
        param_fp16_bytes=parameter_bytes(model, torch.float16),
        param_fp32_bytes=parameter_bytes(model, torch.float32),
        opset=args.online_opset,
    )


def build_tiger_edge_export(args: Any, spec: dict[str, Any], out_dir: Path) -> ExportedModel:
    from TIGER.tiger_npu_edge import NPUEdgeCtxExportWrapper, TIGERNPUEdgeV1

    label = spec.get("label") or args.label or "TIGERNPUEdgeV1"
    model = quiet_call(TIGERNPUEdgeV1).to(args.device).eval()
    state = model.init_streaming_state(batch_size=1, device=args.device, dtype=torch.float32)
    dummy = torch.zeros(1, 1, model.enc_dim * 2, 1, device=args.device, dtype=torch.float32)
    wrapper = NPUEdgeCtxExportWrapper(model).to(args.device).eval()
    inputs = (dummy, *state)
    return ExportedModel(
        label=label,
        target="tiger-edge",
        source="TIGER.tiger_npu_edge.TIGERNPUEdgeV1",
        module=model,
        export_module=wrapper,
        export_inputs=inputs,
        onnx_path=out_dir / f"{sanitize_filename(label)}.onnx",
        input_names=["subband_spec_RIs", "past_kvs", "past_valid_mask", "time_ctx"],
        output_names=["band_masked_output", "new_kv", "new_valid_mask", "new_time_ctx"],
        frames_per_call=1,
        state_fp16_bytes=tensor_tree_bytes(state, torch.float16),
        state_fp32_bytes=tensor_tree_bytes(state, torch.float32),
        param_count=parameter_count(model),
        param_fp16_bytes=parameter_bytes(model, torch.float16),
        param_fp32_bytes=parameter_bytes(model, torch.float32),
        opset=args.tiger_opset,
    )


def build_tf_mlpnet_export(args: Any, spec: dict[str, Any], out_dir: Path) -> ExportedModel:
    tf_root = REPO_ROOT / "TF-MLPNet"
    if str(tf_root) not in sys.path:
        sys.path.insert(0, str(tf_root))

    from tf_mlpnet.export_onnx import TIGEREdgeMLPCellExportWrapper, build_tiger_edge_mlp_dummy_inputs
    from tf_mlpnet.tiger_edge_mlp import TIGEREdgeMLP

    label = spec.get("label") or args.label or "TF-MLPNet TIGEREdgeMLP"
    model = quiet_call(
        TIGEREdgeMLP,
        out_channels=args.tf_out_channels,
        in_channels=args.tf_in_channels,
        num_blocks=args.tf_num_blocks,
        upsampling_depth=args.tf_upsampling_depth,
        num_sources=args.tf_num_sources,
        need_streaming=True,
        edge_hidden_channels=args.tf_edge_hidden_channels,
        edge_num_blocks=args.tf_edge_num_blocks,
    ).to(args.device).eval()
    wrapper = TIGEREdgeMLPCellExportWrapper(model).to(args.device).eval()
    inputs = tuple(t.to(args.device) for t in build_tiger_edge_mlp_dummy_inputs(model, batch_size=1, device=args.device))
    state = inputs[1:]
    return ExportedModel(
        label=label,
        target="tf-mlpnet",
        source="TF-MLPNet/tf_mlpnet/tiger_edge_mlp.py",
        module=model,
        export_module=wrapper,
        export_inputs=inputs,
        onnx_path=out_dir / f"{sanitize_filename(label)}.onnx",
        input_names=[
            "subband_spec_RIs",
            "past_kvs",
            "past_valid_mask",
            "prev_states_0",
            "prev_states_1",
            "prev_states_2",
            "prev_global_states",
        ],
        output_names=[
            "band_masked_output",
            "new_kvs",
            "new_valid_mask",
            "new_states_0",
            "new_states_1",
            "new_states_2",
            "new_global_states",
        ],
        frames_per_call=1,
        state_fp16_bytes=tensor_tree_bytes(state, torch.float16),
        state_fp32_bytes=tensor_tree_bytes(state, torch.float32),
        param_count=parameter_count(model),
        param_fp16_bytes=parameter_bytes(model, torch.float16),
        param_fp32_bytes=parameter_bytes(model, torch.float32),
        opset=args.tf_opset,
    )


BUILDERS = {
    "online": build_online_export,
    "dolphin": build_dolphin_export,
    "tiger-edge": build_tiger_edge_export,
    "tf-mlpnet": build_tf_mlpnet_export,
}


def evaluate_exported_model(args: Any, exported: ExportedModel, out_dir: Path, onnx_mlir: Path | None) -> dict[str, Any]:
    mac_per_call, mac_by_op = profile_macs(
        exported.export_module,
        exported.export_inputs,
        include_elementwise=args.include_elementwise_flops,
    )
    mac_per_frame = mac_per_call / max(exported.frames_per_call, 1)
    frame_rate = float(args.sample_rate) / float(args.hop_length)
    row: dict[str, Any] = {
        "label": exported.label,
        "target": exported.target,
        "source": exported.source,
        "params": exported.param_count,
        "param_fp16_bytes": exported.param_fp16_bytes,
        "param_fp32_bytes": exported.param_fp32_bytes,
        "state_fp16_bytes": exported.state_fp16_bytes,
        "state_fp32_bytes": exported.state_fp32_bytes,
        "frames_per_call": exported.frames_per_call,
        "sample_rate": args.sample_rate,
        "hop_length": args.hop_length,
        "frame_rate": frame_rate,
        "mac_per_call": mac_per_call,
        "mac_per_frame": mac_per_frame,
        "gmac_per_s": mac_per_frame * frame_rate / 1e9,
        "mac_by_op": mac_by_op,
        "onnx_path": str(exported.onnx_path),
    }

    export_to_onnx(exported)
    row.update(onnx_stats(exported.onnx_path))

    if args.skip_mlir:
        row["mlir_ok"] = None
        row["mlir_error"] = "skipped"
    elif onnx_mlir is None:
        row["mlir_ok"] = False
        row["mlir_error"] = "onnx-mlir not found; pass --onnx-mlir or run inside the ASS Docker toolchain."
    else:
        row.update(
            compile_mlir(
                exported.onnx_path,
                out_dir / f"{sanitize_filename(exported.label)}.emit",
                onnx_mlir,
                args.mlir_timeout,
            )
        )
    return row


def flatten_for_csv(row: dict[str, Any]) -> dict[str, Any]:
    flat = dict(row)
    for key in ("param_fp16_bytes", "param_fp32_bytes", "state_fp16_bytes", "state_fp32_bytes"):
        flat[key.replace("_bytes", "_kib")] = kib(row.get(key))
    flat["onnx_initializer_fp32_kib"] = kib(row.get("onnx_initializer_fp32_bytes"))
    flat["mlir_size_kib"] = kib(row.get("mlir_size_bytes"))
    for key, value in list(flat.items()):
        if isinstance(value, (dict, list)):
            flat[key] = json.dumps(value, sort_keys=True)
    return flat


def write_outputs(rows: list[dict[str, Any]], args: Any, out_dir: Path) -> tuple[Path, Path]:
    json_out = args.json_out or out_dir / "npu_model_stats.json"
    csv_out = args.csv_out or out_dir / "npu_model_stats.csv"
    json_out.parent.mkdir(parents=True, exist_ok=True)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")

    with csv_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(flatten_for_csv(row))
    return json_out, csv_out


def print_summary(rows: list[dict[str, Any]]) -> None:
    headers = ["label", "params", "state_fp16_kib", "gmac_per_s", "onnx_nodes", "mlir_ops", "status"]
    table_rows = []
    for row in rows:
        flat = flatten_for_csv(row)
        status = "ok"
        if row.get("error"):
            status = "error"
        elif row.get("mlir_ok") is False:
            status = "mlir-failed"
        elif row.get("mlir_ok") is None:
            status = "mlir-skipped"
        table_rows.append(
            [
                str(row.get("label", "")),
                str(row.get("params", "")),
                format_float(flat.get("state_fp16_kib"), 2),
                format_float(row.get("gmac_per_s"), 4),
                str(row.get("onnx_nodes", "")),
                str(row.get("mlir_ops", "")),
                status,
            ]
        )

    widths = [len(header) for header in headers]
    for row in table_rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]
    print(" | ".join(header.ljust(width) for header, width in zip(headers, widths)))
    print(" | ".join("-" * width for width in widths))
    for row in table_rows:
        print(" | ".join(cell.ljust(width) for cell, width in zip(row, widths)))


def build_specs(args: Any) -> list[dict[str, Any]]:
    if args.target == "ready-suite":
        specs = READY_SUITE
        if args.suite_filter:
            filters = tuple(item.lower() for item in args.suite_filter)
            specs = [
                spec
                for spec in specs
                if any(
                    token in str(spec.get("label", "")).lower()
                    or token in str(spec.get("target", "")).lower()
                    or token in str(spec.get("model_path", "")).lower()
                    for token in filters
                )
            ]
        return [dict(spec) for spec in specs]
    if args.target == "online" and args.model_path is None:
        raise SystemExit("--model-path is required when --target online.")
    return [
        {
            "target": args.target,
            "label": args.label,
            "model_path": args.model_path,
            "n_chan": args.n_chan,
            "frames": args.frames,
            "freqs": args.freqs,
            "dolphin_preset": args.dolphin_preset,
        }
    ]


def parse_args() -> Any:
    parser = ArgumentParser(
        description="Measure NPU deployment stats: params, state, MACs, ONNX nodes, and emitted MLIR ops."
    )
    parser.add_argument(
        "--target",
        choices=["ready-suite", "online", "dolphin", "tiger-edge", "tf-mlpnet"],
        default="ready-suite",
        help="Model family to measure. Use ready-suite for the current NPU candidate set.",
    )
    parser.add_argument("--model-path", type=Path, help="Online SFC recipe config, model dir, or checkpoint.")
    parser.add_argument("--label", type=str, help="Human-readable label used in outputs and file names.")
    parser.add_argument(
        "--suite-filter",
        action="append",
        default=[],
        help="Substring filter for --target ready-suite. Can be repeated.",
    )
    parser.add_argument("--out-dir", type=Path, help="Directory for ONNX, MLIR, JSON, and CSV outputs.")
    parser.add_argument("--json-out", type=Path, help="Optional JSON summary output path.")
    parser.add_argument("--csv-out", type=Path, help="Optional CSV summary output path.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--freqs", type=int, default=1025)
    parser.add_argument("--n-chan", type=int, default=None)
    parser.add_argument("--frames", type=int, default=1, help="Fixed T frames per exported streaming call.")
    parser.add_argument("--online-opset", type=int, default=11)
    parser.add_argument("--tiger-opset", type=int, default=14)
    parser.add_argument("--tf-opset", type=int, default=17)
    parser.add_argument("--skip-mlir", action="store_true", help="Skip onnx-mlir --EmitMLIR.")
    parser.add_argument("--onnx-mlir", type=Path, help="Path to onnx-mlir binary.")
    parser.add_argument("--mlir-timeout", type=int, default=600)
    parser.add_argument(
        "--include-elementwise-flops",
        action="store_true",
        help="Include profiler-reported elementwise FLOPs in MAC estimates. Default counts conv/mm/bmm-style ops only.",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dolphin-preset", default="edge_small")
    parser.add_argument("--tf-out-channels", type=int, default=24)
    parser.add_argument("--tf-in-channels", type=int, default=96)
    parser.add_argument("--tf-num-blocks", type=int, default=4)
    parser.add_argument("--tf-upsampling-depth", type=int, default=2)
    parser.add_argument("--tf-num-sources", type=int, default=2)
    parser.add_argument("--tf-edge-hidden-channels", type=int, default=24)
    parser.add_argument("--tf-edge-num-blocks", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = REPO_ROOT / "logs" / "npu_model_stats" / timestamp
    else:
        out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    onnx_mlir = None if args.skip_mlir else resolve_onnx_mlir(args.onnx_mlir)
    specs = build_specs(args)
    rows: list[dict[str, Any]] = []
    for spec in specs:
        target = spec["target"]
        row: dict[str, Any] = {
            "label": spec.get("label") or target,
            "target": target,
        }
        try:
            exported = BUILDERS[target](args, spec, out_dir)
            row.update(evaluate_exported_model(args, exported, out_dir, onnx_mlir))
        except Exception as exc:
            row["error"] = repr(exc)
            if args.fail_fast:
                raise
        rows.append(row)
        print(json.dumps(row, sort_keys=True))

    json_out, csv_out = write_outputs(rows, args, out_dir)
    print_summary(rows)
    print(f"Wrote JSON: {json_out}")
    print(f"Wrote CSV:  {csv_out}")


if __name__ == "__main__":
    main()
