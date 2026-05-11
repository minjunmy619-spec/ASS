#!/usr/bin/env python3

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any

import onnx


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.online.audit_onnx_model import get_allowed_ops
from tools.online.measure_npu_model_stats import (
    BUILDERS,
    count_mlir_ops,
    export_to_onnx,
    onnx_stats,
    resolve_onnx_mlir,
    sanitize_filename,
)


MLIR_FAIL_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bonnx\.If\b"), "onnx.If (dynamic branch)"),
    (re.compile(r"\bonnx\.Scan\b"), "onnx.Scan (explicit loop)"),
    (re.compile(r"\bonnx\.Loop\b"), "onnx.Loop"),
    (re.compile(r"\bonnx\.Sequence"), "onnx.Sequence*"),
    (re.compile(r"\bonnx\.NonZero\b"), "onnx.NonZero (dynamic indices)"),
    (re.compile(r"\bonnx\.RNN\b"), "onnx.RNN"),
    (re.compile(r"\bonnx\.LSTM\b"), "onnx.LSTM"),
    (re.compile(r"\bonnx\.GRU\b"), "onnx.GRU"),
    (re.compile(r"\bscf\.(if|for|while)\b"), "scf loop/branch"),
    (re.compile(r"\bcf\.cond_br\b"), "cf.cond_br"),
)
MATH_OP_RE = re.compile(r"\bmath\.([a-zA-Z0-9_]+)\b")


def relative_or_abs(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def resolve_input_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def parse_forbid_ops(values: list[str]) -> set[str]:
    ops: set[str] = set()
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                ops.add(part)
    return ops


def run_command(cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout)


def find_generated_mlir(stub: Path, proc: subprocess.CompletedProcess[str]) -> Path:
    blob = (proc.stdout or "") + "\n" + (proc.stderr or "")
    for line in blob.splitlines():
        match = re.search(r'Generated\s+"([^"]+)"', line)
        if match:
            generated = Path(match.group(1))
            if generated.is_file():
                return generated
    guess = stub.with_name(stub.name + ".onnx.mlir")
    if guess.is_file():
        return guess
    raise RuntimeError(f"onnx-mlir finished but emitted MLIR was not found. Tail:\n{blob[-2000:]}")


def run_emit_mlir(onnx_path: Path, stub: Path, tool: Path, timeout: int) -> dict[str, Any]:
    print(f"[mlir] emit textual MLIR -> {stub}.onnx.mlir")
    proc = run_command([str(tool), "--EmitMLIR", str(onnx_path), "-o", str(stub)], timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"onnx-mlir --EmitMLIR failed:\n{((proc.stderr or '') + (proc.stdout or ''))[-3000:]}")
    mlir_path = find_generated_mlir(stub, proc)
    mlir_ops, dialect_counts = count_mlir_ops(mlir_path)
    fails, math_ops = scan_mlir(mlir_path)
    return {
        "ok": True,
        "path": str(mlir_path),
        "size_bytes": mlir_path.stat().st_size,
        "ops": mlir_ops,
        "dialect_counts": dialect_counts,
        "red_flags": fails,
        "math_ops": dict(math_ops),
    }


def run_compile_shared_lib(onnx_path: Path, stub: Path, tool: Path, timeout: int) -> dict[str, Any]:
    print(f"[mlir] compile shared library -> {stub}.so")
    proc = run_command([str(tool), str(onnx_path), "-o", str(stub)], timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"onnx-mlir shared-library compile failed:\n{((proc.stderr or '') + (proc.stdout or ''))[-3000:]}")

    blob = (proc.stdout or "") + "\n" + (proc.stderr or "")
    generated: Path | None = None
    for line in blob.splitlines():
        match = re.search(r'Generated\s+"([^"]+)"', line)
        if match:
            candidate = Path(match.group(1))
            if candidate.is_file():
                generated = candidate
                break
    if generated is None:
        for candidate in (stub.with_suffix(".so"), stub.with_name(stub.name + ".so")):
            if candidate.is_file():
                generated = candidate
                break
    if generated is None:
        raise RuntimeError(f"onnx-mlir returned success but .so was not found. Tail:\n{blob[-2000:]}")
    return {
        "ok": True,
        "path": str(generated),
        "size_bytes": generated.stat().st_size,
    }


def scan_mlir(mlir_path: Path) -> tuple[list[str], Counter[str]]:
    text = mlir_path.read_text(encoding="utf-8", errors="replace")
    failures: list[str] = []
    for regex, label in MLIR_FAIL_PATTERNS:
        if regex.search(text):
            failures.append(label)
    math_ops: Counter[str] = Counter()
    for match in MATH_OP_RE.finditer(text):
        math_ops[match.group(1)] += 1
    return failures, math_ops


def build_exported_model(args: Any, out_dir: Path):
    spec = {
        "target": args.target,
        "label": args.label,
        "model_path": args.model_path,
        "n_chan": args.n_chan,
        "frames": args.frames,
        "freqs": args.freqs,
        "dolphin_preset": args.dolphin_preset,
        "band_scnet_npu_preset": args.band_scnet_npu_preset,
    }
    if args.target == "online" and args.model_path is None:
        raise ValueError("--model-path is required for --target online.")
    return BUILDERS[args.target](args, spec, out_dir)


def export_or_use_onnx(args: Any, out_dir: Path) -> tuple[Path, dict[str, Any]]:
    if args.onnx_in is not None:
        onnx_path = resolve_input_path(args.onnx_in)
        if onnx_path is None or not onnx_path.is_file():
            raise FileNotFoundError(f"ONNX input not found: {args.onnx_in}")
        print(f"[onnx] using existing {onnx_path}")
        onnx.checker.check_model(onnx.load(str(onnx_path)))
        return onnx_path, {
            "mode": "existing_onnx",
            "label": args.label or onnx_path.stem,
            "target": "onnx",
            "source": str(onnx_path),
        }

    exported = build_exported_model(args, out_dir)
    print(f"[onnx] export {exported.label} -> {exported.onnx_path}")
    export_to_onnx(exported)
    return exported.onnx_path, {
        "mode": "exported",
        "label": exported.label,
        "target": exported.target,
        "source": exported.source,
        "frames_per_call": exported.frames_per_call,
        "state_fp16_bytes": exported.state_fp16_bytes,
        "state_fp32_bytes": exported.state_fp32_bytes,
        "params": exported.param_count,
    }


def audit_onnx(onnx_path: Path, args: Any) -> dict[str, Any]:
    print("[onnx] checker")
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    stats = onnx_stats(onnx_path)

    allowed = get_allowed_ops(args.op_preset).union(args.allow_op)
    ops = set(stats["onnx_op_counts"])
    disallowed = sorted(op for op in ops if allowed and op not in allowed)
    forbidden = sorted(ops & parse_forbid_ops(args.forbid_op))
    control_flow_ops = sorted(op for op in ops if op in {"If", "Loop", "Scan", "SequenceConstruct", "SequenceAt"})
    stats.update(
        {
            "op_preset": args.op_preset,
            "disallowed_ops": disallowed,
            "forbidden_ops": forbidden,
            "control_flow_ops": control_flow_ops,
        }
    )
    if disallowed:
        print(f"[onnx] disallowed ops: {', '.join(disallowed)}")
    else:
        print("[onnx] disallowed ops: none")
    if forbidden:
        print(f"[onnx] forbidden ops: {', '.join(forbidden)}")
    if control_flow_ops:
        print(f"[onnx] control-flow ops: {', '.join(control_flow_ops)}")

    if args.fail_on_disallowed_ops and disallowed:
        raise RuntimeError(f"ONNX graph has disallowed ops: {', '.join(disallowed)}")
    if forbidden:
        raise RuntimeError(f"ONNX graph has forbidden ops: {', '.join(forbidden)}")
    if args.fail_on_control_flow and control_flow_ops:
        raise RuntimeError(f"ONNX graph has control-flow ops: {', '.join(control_flow_ops)}")
    return stats


def write_manifest(out_dir: Path, payload: dict[str, Any]) -> Path:
    manifest = out_dir / "export_verify_mlir_manifest.json"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def parse_args() -> Any:
    parser = ArgumentParser(description="Export ONNX and verify onnx-mlir conversion for NPU deployment bring-up.")
    parser.add_argument(
        "--target",
        choices=["online", "dolphin", "tiger-edge", "tiger-edge-v2", "tf-mlpnet", "band-scnet-npu"],
        default="online",
        help="Model exporter to use. Ignored when --onnx-in is passed.",
    )
    parser.add_argument("--model-path", type=Path, help="Online recipe config, trained directory, or checkpoint.")
    parser.add_argument("--onnx-in", type=Path, help="Verify an existing ONNX instead of exporting first.")
    parser.add_argument("--label", type=str, help="Name used for output files and manifest.")
    parser.add_argument("--out-dir", type=Path, help="Output directory for ONNX, MLIR, and manifest.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--n-chan", type=int, default=None)
    parser.add_argument("--freqs", type=int, default=1025)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--online-opset", type=int, default=11)
    parser.add_argument("--tiger-opset", type=int, default=14)
    parser.add_argument("--tf-opset", type=int, default=17)
    parser.add_argument("--dolphin-preset", default="edge_small")
    parser.add_argument("--band-scnet-npu-preset", default="rt192k")
    parser.add_argument("--tf-out-channels", type=int, default=24)
    parser.add_argument("--tf-in-channels", type=int, default=96)
    parser.add_argument("--tf-num-blocks", type=int, default=4)
    parser.add_argument("--tf-upsampling-depth", type=int, default=2)
    parser.add_argument("--tf-num-sources", type=int, default=2)
    parser.add_argument("--tf-edge-hidden-channels", type=int, default=24)
    parser.add_argument("--tf-edge-num-blocks", type=int, default=4)
    parser.add_argument("--op-preset", choices=["none", "edge_npu_recommended"], default="edge_npu_recommended")
    parser.add_argument("--allow-op", action="append", default=[])
    parser.add_argument("--forbid-op", action="append", default=[], help="Forbidden ONNX op, or comma-separated ops.")
    parser.add_argument("--fail-on-disallowed-ops", action="store_true")
    parser.add_argument("--fail-on-control-flow", action="store_true", default=True)
    parser.add_argument("--allow-control-flow", action="store_false", dest="fail_on_control_flow")
    parser.add_argument("--onnx-mlir", type=Path, help="Path to onnx-mlir binary.")
    parser.add_argument("--skip-emit-mlir", action="store_true")
    parser.add_argument("--compile-shared-lib", action="store_true", help="Also run onnx-mlir without --EmitMLIR.")
    parser.add_argument("--mlir-timeout", type=int, default=600)
    parser.add_argument("--fail-on-mlir-red-flags", action="store_true", default=True)
    parser.add_argument("--allow-mlir-red-flags", action="store_false", dest="fail_on_mlir_red_flags")
    parser.add_argument("--fail-on-math-ops", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    label = args.label or (args.model_path.parent.name if args.model_path else args.target)
    out_dir = args.out_dir or REPO_ROOT / "logs" / "onnx_mlir_verify" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "label": label,
        "out_dir": str(out_dir),
        "status": "started",
    }
    try:
        onnx_path, export_meta = export_or_use_onnx(args, out_dir)
        manifest["export"] = export_meta
        manifest["onnx_path"] = str(onnx_path)
        manifest["onnx"] = audit_onnx(onnx_path, args)

        tool = None if args.skip_emit_mlir and not args.compile_shared_lib else resolve_onnx_mlir(args.onnx_mlir)
        if tool is None and (not args.skip_emit_mlir or args.compile_shared_lib):
            raise FileNotFoundError("onnx-mlir not found. Pass --onnx-mlir or run inside /app/ASS Docker.")
        if tool is not None:
            manifest["onnx_mlir"] = str(tool)

        if not args.skip_emit_mlir:
            stub = out_dir / f"{sanitize_filename(label)}.emit"
            emit = run_emit_mlir(onnx_path, stub, tool, args.mlir_timeout)
            manifest["emit_mlir"] = emit
            if emit["red_flags"]:
                print(f"[mlir] red flags: {', '.join(emit['red_flags'])}")
                if args.fail_on_mlir_red_flags:
                    raise RuntimeError(f"MLIR red flags found: {', '.join(emit['red_flags'])}")
            else:
                print("[mlir] red flags: none")
            if emit["math_ops"]:
                print(f"[mlir] math ops: {emit['math_ops']}")
                if args.fail_on_math_ops:
                    raise RuntimeError(f"MLIR math ops found: {emit['math_ops']}")
            else:
                print("[mlir] math ops: none")

        if args.compile_shared_lib:
            stub = out_dir / f"{sanitize_filename(label)}.compile"
            manifest["compile_shared_lib"] = run_compile_shared_lib(onnx_path, stub, tool, args.mlir_timeout)

        manifest["status"] = "ok"
        manifest_path = write_manifest(out_dir, manifest)
        print(f"[ok] manifest -> {manifest_path}")
        return 0
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = repr(exc)
        manifest_path = write_manifest(out_dir, manifest)
        print(f"[failed] {exc}", file=sys.stderr)
        print(f"[failed] manifest -> {manifest_path}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
