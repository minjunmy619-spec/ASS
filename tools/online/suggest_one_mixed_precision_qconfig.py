#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
import re
import sys
import time
from typing import Any

ROOT = Path("/home/cmj/works/ASS")
ONE_ROOT = Path(os.environ.get("ONE_ROOT", "/home/cmj/works/ONE"))
ONE_CMDS = Path(os.environ.get("ONE_CMDS", str(ONE_ROOT / "build" / "compiler" / "one-cmds")))
DEFAULT_OUT_ROOT = ROOT / "logs" / "one_mixed_precision_suggestions"


def add_one_schema_paths() -> None:
    env_site = os.environ.get("ONE_CIRCLE_SCHEMA_SITE")
    candidates = [Path(env_site)] if env_site else []
    candidates.extend((ONE_CMDS / "venv" / "lib").glob("python*/site-packages"))
    for candidate in candidates:
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


add_one_schema_paths()

try:
    from circle_schema.v0_9.circle.Model import Model
    from circle_schema.v0_9.circle.BuiltinOperator import BuiltinOperator
    from circle_schema.v0_9.circle.TensorType import TensorType
except ImportError as exc:  # pragma: no cover - this is an environment error path.
    raise SystemExit(
        "Failed to import ONE circle_schema from the one-cmds virtualenv. "
        "Set up /home/cmj/works/ONE/build/compiler/one-cmds/venv or run with that python."
    ) from exc


QUALITY_BASE = {
    "BATCH_MATMUL": 92,
    "SOFTMAX": 88,
    "LOG_SOFTMAX": 84,
    "TRANSPOSE_CONV": 78,
    "CONV_2D": 68,
    "DEPTHWISE_CONV_2D": 60,
    "FULLY_CONNECTED": 58,
    "LOGISTIC": 56,
    "TANH": 52,
    "EXP": 50,
    "SQRT": 48,
    "RSQRT": 48,
    "DIV": 45,
    "MUL": 42,
    "ADD": 38,
    "SUB": 38,
    "MEAN": 36,
    "INSTANCE_NORM": 62,
    "RMS_NORM": 62,
    "PRELU": 42,
}

LOW_VALUE_MEMORY_OPS = {
    "RESHAPE",
    "TRANSPOSE",
    "CONCATENATION",
    "SPLIT",
    "SPLIT_V",
    "SLICE",
    "STRIDED_SLICE",
    "PACK",
    "UNPACK",
    "PAD",
    "PADV2",
    "SQUEEZE",
    "GATHER",
    "TILE",
    "EXPAND_DIMS",
}

ONE_MIXED_BOUNDARY_SPECIAL_CASES = {
    "TRANSPOSE",
    "FULLY_CONNECTED",
    "MUL",
    "BATCH_MATMUL",
}

NAME_HINTS = (
    "mask",
    "head",
    "out",
    "final",
    "logit",
    "softmax",
    "attn",
    "attention",
    "query",
    "key",
    "value",
    "sfc",
    "expand",
    "decoder",
)


@dataclass
class TensorInfo:
    index: int
    name: str
    dtype: str
    shape: list[int]
    producer: int | None = None
    consumers: list[int] = field(default_factory=list)


@dataclass
class OpInfo:
    index: int
    name: str
    op: str
    inputs: list[int]
    outputs: list[int]
    output_shape: list[int]
    depth: int = 1
    consumers: set[int] = field(default_factory=set)
    producers: set[int] = field(default_factory=set)
    rough_ops: float = 0.0
    score: float = 0.0
    latency_risk: float = 0.0
    boundary_risk: float = 0.0
    qerror: float = 0.0
    eligible: bool = True
    reasons: list[str] = field(default_factory=list)


def enum_reverse(cls: type) -> dict[int, str]:
    rev: dict[int, str] = {}
    for key, value in vars(cls).items():
        if key.startswith("_") or not isinstance(value, int):
            continue
        rev[value] = key
    return rev


BUILTIN_OP_NAMES = enum_reverse(BuiltinOperator)
TENSOR_TYPE_NAMES = enum_reverse(TensorType)


def decode_name(raw: bytes | None, fallback: str) -> str:
    if raw is None:
        return fallback
    return raw.decode("utf-8", errors="replace")


def tensor_shape(tensor: Any) -> list[int]:
    return [int(tensor.Shape(i)) for i in range(tensor.ShapeLength())]


def product(values: list[int]) -> int:
    result = 1
    for value in values:
        if value <= 0:
            continue
        result *= value
    return result


def read_circle(circle_path: Path) -> tuple[list[TensorInfo], list[OpInfo], set[int], set[int]]:
    model = Model.GetRootAs(circle_path.read_bytes(), 0)
    if model.SubgraphsLength() != 1:
        raise ValueError("Only single-subgraph Circle models are supported by this helper.")

    opcodes = []
    for i in range(model.OperatorCodesLength()):
        opcode = model.OperatorCodes(i)
        opcodes.append(BUILTIN_OP_NAMES.get(opcode.BuiltinCode(), f"OP_{opcode.BuiltinCode()}"))

    subgraph = model.Subgraphs(0)
    tensors: list[TensorInfo] = []
    for idx in range(subgraph.TensorsLength()):
        tensor = subgraph.Tensors(idx)
        tensors.append(
            TensorInfo(
                index=idx,
                name=decode_name(tensor.Name(), f"tensor_{idx}"),
                dtype=TENSOR_TYPE_NAMES.get(tensor.Type(), f"TYPE_{tensor.Type()}"),
                shape=tensor_shape(tensor),
            )
        )

    ops: list[OpInfo] = []
    for idx in range(subgraph.OperatorsLength()):
        operator = subgraph.Operators(idx)
        inputs = [int(operator.Inputs(i)) for i in range(operator.InputsLength()) if operator.Inputs(i) >= 0]
        outputs = [
            int(operator.Outputs(i)) for i in range(operator.OutputsLength()) if operator.Outputs(i) >= 0
        ]
        op_name = opcodes[operator.OpcodeIndex()]
        out_name = tensors[outputs[0]].name if outputs else f"{op_name}_{idx}"
        out_shape = tensors[outputs[0]].shape if outputs else []
        ops.append(
            OpInfo(
                index=idx,
                name=out_name,
                op=op_name,
                inputs=inputs,
                outputs=outputs,
                output_shape=out_shape,
            )
        )
        for output in outputs:
            tensors[output].producer = idx
        for input_idx in inputs:
            tensors[input_idx].consumers.append(idx)

    input_tensors = {int(subgraph.Inputs(i)) for i in range(subgraph.InputsLength())}
    output_tensors = {int(subgraph.Outputs(i)) for i in range(subgraph.OutputsLength())}

    for op in ops:
        for input_idx in op.inputs:
            producer = tensors[input_idx].producer
            if producer is not None:
                op.producers.add(producer)
                ops[producer].consumers.add(op.index)

    compute_depths(ops)
    for op in ops:
        op.rough_ops = estimate_rough_ops(op, tensors)

    return tensors, ops, input_tensors, output_tensors


def compute_depths(ops: list[OpInfo]) -> None:
    indegree = {op.index: len(op.producers) for op in ops}
    queue = deque(op.index for op in ops if indegree[op.index] == 0)
    while queue:
        idx = queue.popleft()
        op = ops[idx]
        pred_depth = [ops[pred].depth for pred in op.producers]
        op.depth = (max(pred_depth) + 1) if pred_depth else 1
        for consumer in op.consumers:
            indegree[consumer] -= 1
            if indegree[consumer] == 0:
                queue.append(consumer)


def estimate_rough_ops(op: OpInfo, tensors: list[TensorInfo]) -> float:
    output_elems = product(op.output_shape)
    if op.op == "BATCH_MATMUL" and len(op.inputs) >= 2:
        lhs = tensors[op.inputs[0]].shape
        rhs = tensors[op.inputs[1]].shape
        if len(lhs) >= 2 and len(rhs) >= 2:
            m = lhs[-2]
            k = lhs[-1]
            n = rhs[-1]
            batch = max(product(lhs[:-2]), product(rhs[:-2]), 1)
            return float(batch * m * n * k)
    if op.op in {"CONV_2D", "DEPTHWISE_CONV_2D", "TRANSPOSE_CONV"} and len(op.inputs) >= 2:
        weight_shape = tensors[op.inputs[1]].shape
        kernel = product(weight_shape[:3]) if len(weight_shape) >= 3 else product(weight_shape)
        return float(output_elems * max(kernel, 1))
    if op.op == "FULLY_CONNECTED" and len(op.inputs) >= 2:
        lhs = product(tensors[op.inputs[0]].shape)
        rhs = product(tensors[op.inputs[1]].shape)
        return float(max(lhs, rhs))
    return float(output_elems)


def load_visq_errors(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    errors: dict[str, float] = {}
    for item in data.get("error", []):
        if not isinstance(item, dict):
            continue
        for key, value in item.items():
            try:
                errors[key] = float(value)
            except (TypeError, ValueError):
                continue
    return errors


def load_qconfig_layers(path: Path | None) -> set[str]:
    if path is None:
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for item in data.get("layers", []):
        if not isinstance(item, dict) or item.get("dtype") != "int16":
            continue
        if item.get("name"):
            names.add(str(item["name"]))
        for name in item.get("names", []):
            names.add(str(name))
    return names


def regex_any(patterns: list[re.Pattern[str]], text: str) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def compile_regex(values: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(value) for value in values if value]


def parse_csv_strings(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def score_ops(
    ops: list[OpInfo],
    output_tensors: set[int],
    visq_errors: dict[str, float],
    ampq_layers: set[str],
    prefer_patterns: list[re.Pattern[str]],
    skip_patterns: list[re.Pattern[str]],
    exclude_patterns: list[re.Pattern[str]],
    exclude_ops: set[str],
) -> None:
    max_depth = max((op.depth for op in ops), default=1)
    max_log_ops = max((math.log10(op.rough_ops + 10.0) for op in ops), default=1.0)
    max_qerror = max(visq_errors.values(), default=0.0)

    output_names = {op.name for op in ops if set(op.outputs) & output_tensors}

    for op in ops:
        reasons: list[str] = []
        base = QUALITY_BASE.get(op.op, 10)
        if base >= 50:
            reasons.append(f"quality-sensitive {op.op}")

        if op.name in output_names:
            base += 25
            reasons.append("graph output path")

        lname = op.name.lower()
        matched_hints = [hint for hint in NAME_HINTS if hint in lname]
        if matched_hints:
            base += 8 + 3 * min(len(matched_hints), 4)
            reasons.append("name hint: " + ",".join(matched_hints[:4]))

        if op.depth >= 0.80 * max_depth:
            base += 10
            reasons.append("late-stage node")

        if op.name in ampq_layers:
            base += 18
            reasons.append("selected by AMPQ config")

        if regex_any(prefer_patterns, op.name):
            base += 20
            reasons.append("matched prefer-regex")

        if op.op in LOW_VALUE_MEMORY_OPS:
            base -= 25
            reasons.append("memory/shape op: avoid isolated int16")

        if regex_any(skip_patterns, op.name):
            base -= 100
            reasons.append("matched skip-regex")

        excluded = False
        if op.op in exclude_ops:
            excluded = True
            base -= 1000
            reasons.append("excluded op type")
        if regex_any(exclude_patterns, op.name):
            excluded = True
            base -= 1000
            reasons.append("matched exclude-regex")

        qerror = visq_errors.get(op.name, 0.0)
        op.qerror = qerror
        if qerror > 0 and max_qerror > 0:
            base += 35.0 * (qerror / max_qerror)
            reasons.append("high VISQ error")

        log_ops = math.log10(op.rough_ops + 10.0)
        latency_risk = 20.0 * (log_ops / max_log_ops)
        if op.op in {"BATCH_MATMUL", "CONV_2D", "DEPTHWISE_CONV_2D", "TRANSPOSE_CONV", "FULLY_CONNECTED"}:
            latency_risk += 8
        boundary_risk = 0.0
        if op.producers:
            boundary_risk += 2.5
        if op.consumers:
            boundary_risk += 2.5
        if op.op not in ONE_MIXED_BOUNDARY_SPECIAL_CASES:
            boundary_risk += 8
        if op.op in LOW_VALUE_MEMORY_OPS:
            boundary_risk += 12

        op.latency_risk = latency_risk
        op.boundary_risk = boundary_risk
        op.score = base - 0.55 * latency_risk - 0.75 * boundary_risk
        op.eligible = not excluded
        op.reasons = reasons or ["low-priority fallback"]


def qconfig_payload(
    circle_path: Path,
    names: list[str],
    default_dtype: str,
    default_granularity: str,
    target_dtype: str,
    target_granularity: str,
) -> dict[str, Any]:
    return {
        "default_quantization_dtype": default_dtype,
        "default_granularity": default_granularity,
        "model_path": str(circle_path),
        "layers": [
            {"name": name, "dtype": target_dtype, "granularity": target_granularity}
            for name in names
        ],
    }


def write_qconfig(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_depth_split(ops: list[OpInfo], fraction: float, front: bool) -> list[str]:
    max_depth = max((op.depth for op in ops), default=1)
    if front:
        cutoff = max_depth * fraction
        return [op.name for op in ops if op.depth <= cutoff and op.score > 0 and op.eligible]
    cutoff = max_depth * (1.0 - fraction)
    return [op.name for op in ops if op.depth >= cutoff and op.score > 0 and op.eligible]


def build_island(ops: list[OpInfo], center_index: int, size: int) -> list[str]:
    by_index = {op.index: op for op in ops}
    selected = {center_index}
    frontier = deque([center_index])
    while frontier and len(selected) < size:
        idx = frontier.popleft()
        neighbors = sorted(by_index[idx].producers | by_index[idx].consumers)
        neighbors.sort(key=lambda n: by_index[n].boundary_risk)
        for neighbor in neighbors:
            if neighbor in selected:
                continue
            if by_index[neighbor].score <= 0:
                continue
            if not by_index[neighbor].eligible:
                continue
            selected.add(neighbor)
            frontier.append(neighbor)
            if len(selected) >= size:
                break
    ordered = sorted((by_index[idx] for idx in selected), key=lambda op: op.index)
    return [op.name for op in ordered]


def write_nodes_csv(path: Path, ops: list[OpInfo]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "score",
                "name",
                "op",
                "depth",
                "rough_ops",
                "latency_risk",
                "boundary_risk",
                "qerror",
                "eligible",
                "output_shape",
                "producers",
                "consumers",
                "reasons",
            ],
        )
        writer.writeheader()
        for rank, op in enumerate(sorted(ops, key=lambda item: item.score, reverse=True), start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "score": f"{op.score:.3f}",
                    "name": op.name,
                    "op": op.op,
                    "depth": op.depth,
                    "rough_ops": f"{op.rough_ops:.0f}",
                    "latency_risk": f"{op.latency_risk:.3f}",
                    "boundary_risk": f"{op.boundary_risk:.3f}",
                    "qerror": f"{op.qerror:.8g}",
                    "eligible": int(op.eligible),
                    "output_shape": "x".join(str(v) for v in op.output_shape),
                    "producers": ";".join(str(v) for v in sorted(op.producers)),
                    "consumers": ";".join(str(v) for v in sorted(op.consumers)),
                    "reasons": "; ".join(op.reasons),
                }
            )


def write_summary(path: Path, ops: list[OpInfo], qconfigs: list[dict[str, Any]], args: argparse.Namespace) -> None:
    top = sorted(ops, key=lambda item: item.score, reverse=True)[: args.top_k]
    lines = [
        "# ONE Mixed Precision Suggestions",
        "",
        "This report uses stock ONE mixed-precision behavior. It does not modify ONE source.",
        "",
        "Generated qconfigs exclude memory/layout ops by default to avoid int16 islands made of "
        "reshape, slice, pad, concat, or transpose boundaries.",
        "",
        "## Top Candidates",
        "",
        "| rank | score | op | name | reasons |",
        "|---:|---:|---|---|---|",
    ]
    for rank, op in enumerate(top, start=1):
        lines.append(
            f"| {rank} | {op.score:.1f} | `{op.op}` | `{op.name}` | {'; '.join(op.reasons)} |"
        )

    lines.extend(["", "## Generated QConfigs", ""])
    for item in qconfigs:
        lines.append(f"- `{item['path'].name}`: {item['description']} ({len(item['names'])} layers)")

    lines.extend(
        [
            "",
            "## How To Test One Proposal",
            "",
            "```bash",
            "one-quantize \\",
            "  --input_path <model.opt.circle> \\",
            "  --output_path <model.mixed.q.circle> \\",
            "  --input_data <calib.h5-or-list.txt> \\",
            "  --input_data_format <h5-or-list> \\",
            "  --quantized_dtype uint8 \\",
            "  --granularity channel \\",
            "  --input_type uint8 \\",
            "  --output_type uint8 \\",
            "  --quant_config <one-of-the-json-files>",
            "```",
            "",
            "Prefer small contiguous islands first. Isolated int16 nodes can add Quantize boundaries.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Suggest stock ONE uint8/int16 mixed-precision qconfigs for a Circle model."
    )
    parser.add_argument("--circle", type=Path, required=True, help="Input optimized fp32 Circle model.")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--island-sizes", default="3,5,8")
    parser.add_argument("--depth-fractions", default="0.25,0.5")
    parser.add_argument("--visq-json", type=Path, help="Optional VISQ JSON generated by stock one-quantize AMPQ flow.")
    parser.add_argument("--ampq-config", type=Path, help="Optional FinalConfiguration.mpq.json to boost AMPQ-selected layers.")
    parser.add_argument("--prefer-regex", action="append", default=[])
    parser.add_argument("--skip-regex", action="append", default=[])
    parser.add_argument(
        "--exclude-regex",
        action="append",
        default=[],
        help="Hard-exclude matching node names from generated qconfigs.",
    )
    parser.add_argument(
        "--exclude-op",
        default=",".join(sorted(LOW_VALUE_MEMORY_OPS)),
        help=(
            "Comma-separated op types to hard-exclude from generated qconfigs. "
            "Defaults to memory/layout ops; pass an empty string to disable."
        ),
    )
    parser.add_argument("--default-dtype", default="uint8")
    parser.add_argument("--default-granularity", default="channel")
    parser.add_argument("--target-dtype", default="int16")
    parser.add_argument("--target-granularity", default="channel")
    args = parser.parse_args()

    if not args.circle.exists():
        parser.error(f"--circle not found: {args.circle}")
    if args.visq_json is not None and not args.visq_json.exists():
        parser.error(f"--visq-json not found: {args.visq_json}")
    if args.ampq_config is not None and not args.ampq_config.exists():
        parser.error(f"--ampq-config not found: {args.ampq_config}")
    if args.top_k <= 0:
        parser.error("--top-k must be greater than zero")

    island_sizes = [int(item.strip()) for item in args.island_sizes.split(",") if item.strip()]
    depth_fractions = [float(item.strip()) for item in args.depth_fractions.split(",") if item.strip()]
    if not island_sizes:
        parser.error("--island-sizes must contain at least one size")
    if any(size <= 0 for size in island_sizes):
        parser.error("--island-sizes values must be greater than zero")
    if not depth_fractions:
        parser.error("--depth-fractions must contain at least one value")
    if any(fraction <= 0.0 or fraction > 1.0 for fraction in depth_fractions):
        parser.error("--depth-fractions values must be in the range (0, 1]")
    exclude_ops = parse_csv_strings(args.exclude_op)

    out_dir = args.out_dir or (DEFAULT_OUT_ROOT / time.strftime("%Y%m%d-%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)

    tensors, ops, _input_tensors, output_tensors = read_circle(args.circle)
    score_ops(
        ops,
        output_tensors,
        load_visq_errors(args.visq_json),
        load_qconfig_layers(args.ampq_config),
        compile_regex(args.prefer_regex),
        compile_regex(args.skip_regex),
        compile_regex(args.exclude_regex),
        exclude_ops,
    )

    ranked = sorted(
        [op for op in ops if op.score > 0 and op.eligible],
        key=lambda item: item.score,
        reverse=True,
    )
    write_nodes_csv(out_dir / "nodes.csv", ops)

    qconfigs: list[dict[str, Any]] = []

    top_names = [op.name for op in ranked[: args.top_k]]
    if top_names:
        path = out_dir / f"qconfig_top{len(top_names)}_{args.target_dtype}.json"
        write_qconfig(
            path,
            qconfig_payload(
                args.circle,
                top_names,
                args.default_dtype,
                args.default_granularity,
                args.target_dtype,
                args.target_granularity,
            ),
        )
        qconfigs.append({"path": path, "description": "top-ranked individual nodes", "names": top_names})

    for size in island_sizes:
        if not ranked:
            break
        names = build_island(ops, ranked[0].index, size)
        path = out_dir / f"qconfig_best_island{size}_{args.target_dtype}.json"
        write_qconfig(
            path,
            qconfig_payload(
                args.circle,
                names,
                args.default_dtype,
                args.default_granularity,
                args.target_dtype,
                args.target_granularity,
            ),
        )
        qconfigs.append({"path": path, "description": "local graph island around best node", "names": names})

    for fraction in depth_fractions:
        for front in (True, False):
            names = build_depth_split(ops, fraction, front)
            if not names:
                continue
            side = "front" if front else "back"
            pct = int(round(fraction * 100))
            path = out_dir / f"qconfig_depth_{side}_{pct}_{args.target_dtype}.json"
            write_qconfig(
                path,
                qconfig_payload(
                    args.circle,
                    names,
                    args.default_dtype,
                    args.default_granularity,
                    args.target_dtype,
                    args.target_granularity,
                ),
            )
            qconfigs.append(
                {"path": path, "description": f"AMPQ-style depth {side} split at {fraction:g}", "names": names}
            )

    write_summary(out_dir / "summary.md", ops, qconfigs, args)

    metadata = {
        "circle": str(args.circle),
        "num_tensors": len(tensors),
        "num_ops": len(ops),
        "outputs": [tensors[idx].name for idx in sorted(output_tensors)],
        "exclude_ops": sorted(exclude_ops),
        "exclude_regex": args.exclude_regex,
        "qconfigs": [
            {"path": str(item["path"]), "description": item["description"], "layers": item["names"]}
            for item in qconfigs
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"[done] wrote {out_dir / 'nodes.csv'}")
    print(f"[done] wrote {out_dir / 'summary.md'}")
    print(f"[done] wrote {len(qconfigs)} qconfig proposal(s) under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
