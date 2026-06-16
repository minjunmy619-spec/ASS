#!/usr/bin/env python3

from __future__ import annotations

from typing import Any

from argparse import ArgumentParser
from collections import Counter, defaultdict
import json
from pathlib import Path
import subprocess

import numpy as np

import onnx
from onnx import numpy_helper

DEFAULT_CIRCLE_INSPECT = Path("/home/cmj/works/ONE/build/compiler/circle-inspect/circle-inspect")
MEMORY_OPS = (
    "Transpose",
    "Reshape",
    "Slice",
    "Split",
    "Concat",
    "Gather",
    "Expand",
    "Tile",
    "Pad",
    "Squeeze",
    "Unsqueeze",
)
SLOW_MATH_OPS = ("Div", "Sqrt", "ReduceMean", "ReduceSum", "Softmax", "Sigmoid", "Tanh", "Exp", "Log")


def _value_shape(value_info: onnx.ValueInfoProto) -> list[int | None] | None:
    if not value_info.type.HasField("tensor_type"):
        return None
    dims: list[int | None] = []
    for dim in value_info.type.tensor_type.shape.dim:
        dims.append(int(dim.dim_value) if dim.dim_value > 0 else None)
    return dims


def shape_map(model: onnx.ModelProto) -> dict[str, list[int | None]]:
    try:
        inferred = onnx.shape_inference.infer_shapes(model)
    except Exception:  # noqa: BLE001 - report should still work on partially inferable graphs
        inferred = model
    shapes: dict[str, list[int | None]] = {}
    for graph in (model.graph, inferred.graph):
        for group in (graph.input, graph.value_info, graph.output):
            for value_info in group:
                shape = _value_shape(value_info)
                if shape is not None:
                    shapes[value_info.name] = shape
    for initializer in model.graph.initializer:
        shapes[initializer.name] = [int(dim) for dim in initializer.dims]
    return shapes


def initializer_arrays(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {initializer.name: numpy_helper.to_array(initializer) for initializer in model.graph.initializer}


def constant_arrays(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    constants: dict[str, np.ndarray] = {}
    for node in model.graph.node:
        if node.op_type != "Constant" or not node.output:
            continue
        for attr in node.attribute:
            if attr.name == "value" and attr.HasField("t"):
                constants[node.output[0]] = numpy_helper.to_array(attr.t)
                break
    return constants


def static_array(name: str, initializers: dict[str, np.ndarray], constants: dict[str, np.ndarray]) -> np.ndarray | None:
    if not name:
        return None
    if name in initializers:
        return initializers[name]
    return constants.get(name)


def producer_map(model: onnx.ModelProto) -> dict[str, onnx.NodeProto]:
    producers: dict[str, onnx.NodeProto] = {}
    for node in model.graph.node:
        for output in node.output:
            producers[output] = node
    return producers


def consumer_map(model: onnx.ModelProto) -> dict[str, list[onnx.NodeProto]]:
    consumers: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    for node in model.graph.node:
        for input_name in node.input:
            consumers[input_name].append(node)
    return consumers


def node_name(node: onnx.NodeProto) -> str:
    return node.name or "/".join(node.output) or node.op_type


def append_sample(samples: dict[str, list[str]], key: str, node: onnx.NodeProto, limit: int) -> None:
    bucket = samples.setdefault(key, [])
    if len(bucket) < limit:
        bucket.append(node_name(node))


def get_attr_int(node: onnx.NodeProto, name: str, default: int) -> int:
    for attr in node.attribute:
        if attr.name == name:
            return int(attr.i)
    return default


def analyze_onnx(model: onnx.ModelProto, *, sample_limit: int = 8) -> dict[str, Any]:
    shapes = shape_map(model)
    initializers = initializer_arrays(model)
    constants = constant_arrays(model)
    producers = producer_map(model)
    consumers = consumer_map(model)
    op_counts = Counter(node.op_type for node in model.graph.node)
    samples: dict[str, list[str]] = {}

    counts: dict[str, int] = {
        "nodes": len(model.graph.node),
        "memory_ops_total": sum(op_counts.get(op, 0) for op in MEMORY_OPS),
        "slow_math_ops_total": sum(op_counts.get(op, 0) for op in SLOW_MATH_OPS),
        "div_by_static_const": 0,
        "div_by_sqrt": 0,
        "dynamic_or_activation_div": 0,
        "sigmoid_outputs_consumed_by_mul": 0,
        "split_to_sigmoid_gate": 0,
        "cat_slice_state_updates": 0,
        "non_depthwise_grouped_conv": 0,
        "depthwise_grouped_conv": 0,
        "rank_gt4_values": 0,
        "activation_matmul": 0,
    }

    for name, shape in shapes.items():
        if len(shape) > 4:
            counts["rank_gt4_values"] += 1
            if len(samples.setdefault("rank_gt4_values", [])) < sample_limit:
                samples["rank_gt4_values"].append(f"{name}:{shape}")

    for node in model.graph.node:
        if node.op_type == "Div" and len(node.input) >= 2:
            divisor = static_array(node.input[1], initializers, constants)
            pred = producers.get(node.input[1])
            if divisor is not None:
                counts["div_by_static_const"] += 1
                append_sample(samples, "div_by_static_const", node, sample_limit)
            elif pred is not None and pred.op_type == "Sqrt":
                counts["div_by_sqrt"] += 1
                append_sample(samples, "div_by_sqrt", node, sample_limit)
            else:
                counts["dynamic_or_activation_div"] += 1
                append_sample(samples, "dynamic_or_activation_div", node, sample_limit)

        elif node.op_type == "Sigmoid":
            if any(consumer.op_type == "Mul" for output in node.output for consumer in consumers.get(output, [])):
                counts["sigmoid_outputs_consumed_by_mul"] += 1
                append_sample(samples, "sigmoid_outputs_consumed_by_mul", node, sample_limit)
            pred = producers.get(node.input[0]) if node.input else None
            if pred is not None and pred.op_type == "Split":
                counts["split_to_sigmoid_gate"] += 1
                append_sample(samples, "split_to_sigmoid_gate", node, sample_limit)

        elif node.op_type == "Slice":
            pred = producers.get(node.input[0]) if node.input else None
            if (
                pred is not None
                and pred.op_type == "Concat"
                and any(out.startswith("next_state") for out in node.output)
            ):
                counts["cat_slice_state_updates"] += 1
                append_sample(samples, "cat_slice_state_updates", node, sample_limit)

        elif node.op_type == "Conv":
            group = get_attr_int(node, "group", 1)
            if group > 1:
                input_shape = shapes.get(node.input[0]) if node.input else None
                input_channels = input_shape[1] if input_shape is not None and len(input_shape) >= 2 else None
                if input_channels == group:
                    counts["depthwise_grouped_conv"] += 1
                else:
                    counts["non_depthwise_grouped_conv"] += 1
                    append_sample(samples, "non_depthwise_grouped_conv", node, sample_limit)

        elif node.op_type == "MatMul" and len(node.input) >= 2:
            lhs_shape = shapes.get(node.input[0])
            rhs_shape = shapes.get(node.input[1])
            lhs_const = node.input[0] in initializers or node.input[0] in constants
            rhs_const = node.input[1] in initializers or node.input[1] in constants
            if lhs_shape is not None and rhs_shape is not None and not lhs_const and not rhs_const:
                counts["activation_matmul"] += 1
                append_sample(samples, "activation_matmul", node, sample_limit)

    memory_counts = {op: op_counts.get(op, 0) for op in MEMORY_OPS if op_counts.get(op, 0)}
    slow_math_counts = {op: op_counts.get(op, 0) for op in SLOW_MATH_OPS if op_counts.get(op, 0)}

    return {
        "op_counts": dict(sorted(op_counts.items())),
        "top_ops": dict(sorted(op_counts.items(), key=lambda item: (-item[1], item[0]))[:16]),
        "memory_op_counts": memory_counts,
        "slow_math_op_counts": slow_math_counts,
        "rewrite_candidate_counts": counts,
        "samples": samples,
        "recommendations": recommendations(counts, memory_counts, slow_math_counts),
    }


def recommendations(
    counts: dict[str, int], memory_counts: dict[str, int], slow_math_counts: dict[str, int]
) -> list[str]:
    recs: list[str] = []
    if counts.get("div_by_static_const", 0):
        recs.append(
            "Rewrite `x / constant` as `x * reciprocal_constant` in PyTorch, or rely on the verifier's "
            "`div_static_const_to_mul` ONNX pre-import rewrite. This is mathematically safe for finite "
            "non-zero constants."
        )
    if counts.get("div_by_sqrt", 0):
        recs.append(
            "Most `Div` fed by `Sqrt` are RMSNorm-style reciprocal square-root patterns. Compile with "
            "`--low-latency-optimize` so ONE can apply `transform_sqrt_div_to_rsqrt_mul`."
        )
    if counts.get("dynamic_or_activation_div", 0):
        recs.append(
            "Dynamic `Div` nodes remain real slow ops. Consider model-level ablations: pre-normalize static bases, "
            "use Softmax-normalized weights, disable dynamic renormalization, or approximate with learned scale "
            "only if quality allows."
        )
    if counts.get("split_to_sigmoid_gate", 0) or counts.get("sigmoid_outputs_consumed_by_mul", 0):
        recs.append(
            "GLU/Sigmoid gates export as Split/Sigmoid/Mul and are not fused by ONE. Keep quality-critical gates, but "
            "try ReLU/ReLU6 or single-branch Conv blocks in low-value FFN/pooled mixers."
        )
    if counts.get("cat_slice_state_updates", 0):
        recs.append(
            "Concat+Slice state updates are streaming-cache memory ops. Reduce layer count/context/state tensors, "
            "or fuse multi-branch memory blocks so fewer caches are updated per frame."
        )
    if counts.get("non_depthwise_grouped_conv", 0):
        recs.append(
            "Avoid non-depthwise grouped Conv. ONE lowers it to Split -> Conv(s) -> Concat, which usually hurts "
            "latency. Use groups=1 or groups=in_channels."
        )
    if memory_counts.get("Transpose", 0) or memory_counts.get("Reshape", 0):
        recs.append(
            "High Transpose/Reshape counts indicate layout or MatMul transport overhead. Use `--low-latency-optimize`, "
            "inspect final Circle counts, and reduce source loops/SFC transitions when counts remain high."
        )
    if memory_counts.get("Split", 0) or memory_counts.get("Concat", 0):
        recs.append(
            "Split/Concat often comes from source loops, GLU gates, branch fusion, or streaming state. Prefer "
            "packed-channel vectorization and fewer small parallel branches."
        )
    if slow_math_counts.get("Softmax", 0):
        recs.append(
            "Keep Softmax only on the last dimension for ONE compatibility; avoid adding source/channel-axis Softmax."
        )
    if counts.get("rank_gt4_values", 0):
        recs.append(
            "Rank >4 tensors violate the strict NPU rule. Keep source/channel packed into C, not batch or extra rank."
        )
    return recs


def parse_circle_operator_counts(output: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for raw_line in output.splitlines():
        op = raw_line.strip()
        if not op or " " in op or op.startswith("["):
            continue
        counts[op] += 1
    return dict(sorted(counts.items()))


def circle_operator_counts(circle_path: Path, circle_inspect: Path = DEFAULT_CIRCLE_INSPECT) -> dict[str, int]:
    if not circle_path.exists() or not circle_inspect.exists():
        return {}
    p = subprocess.run([str(circle_inspect), "--operators", str(circle_path)], text=True, capture_output=True)
    if p.returncode != 0:
        return {}
    return parse_circle_operator_counts(p.stdout or "")


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# Latency Rewrite Candidate Report", ""]
    lines.extend(["## ONNX top operators", "", "```text"])
    for op, count in report["onnx"]["top_ops"].items():
        lines.append(f"{op}: {count}")
    lines.extend(["```", ""])

    lines.extend(["## ONNX memory operators", "", "```text"])
    for op, count in report["onnx"]["memory_op_counts"].items():
        lines.append(f"{op}: {count}")
    lines.extend(["```", ""])

    lines.extend(["## ONNX slow/math operators", "", "```text"])
    for op, count in report["onnx"]["slow_math_op_counts"].items():
        lines.append(f"{op}: {count}")
    lines.extend(["```", ""])

    lines.extend(["## Rewrite candidate counts", "", "```text"])
    for key, value in report["onnx"]["rewrite_candidate_counts"].items():
        lines.append(f"{key}: {value}")
    lines.extend(["```", ""])

    if report.get("circle_op_counts"):
        lines.extend(["## Circle operators", "", "```text"])
        for op, count in sorted(report["circle_op_counts"].items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"{op}: {count}")
        lines.extend(["```", ""])

    samples = report["onnx"].get("samples", {})
    if samples:
        lines.extend(["## Samples", ""])
        for key, values in samples.items():
            lines.append(f"### {key}")
            lines.append("")
            for value in values:
                lines.append(f"- `{value}`")
            lines.append("")

    recs = report["onnx"].get("recommendations", [])
    if recs:
        lines.extend(["## Recommendations", ""])
        for rec in recs:
            lines.append(f"- {rec}")
    return "\n".join(lines) + "\n"


def parse_args() -> Any:
    p = ArgumentParser(description="Report latency/node rewrite candidates for strict NPU ONNX/Circle graphs.")
    p.add_argument(
        "onnx", type=Path, help="ONNX model path, usually model.sim.onnx after verifier simplification/prep."
    )
    p.add_argument("--circle", type=Path, default=None, help="Optional Circle artifact for post-ONE operator counts.")
    p.add_argument("--circle-inspect", type=Path, default=DEFAULT_CIRCLE_INSPECT)
    p.add_argument("--json-out", type=Path, default=None)
    p.add_argument("--md-out", type=Path, default=None)
    p.add_argument("--sample-limit", type=int, default=8)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    model = onnx.load(str(args.onnx))
    report: dict[str, Any] = {
        "onnx_path": str(args.onnx),
        "onnx": analyze_onnx(model, sample_limit=args.sample_limit),
    }
    if args.circle is not None:
        report["circle_path"] = str(args.circle)
        report["circle_op_counts"] = circle_operator_counts(args.circle, args.circle_inspect)

    if args.json_out is not None:
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = render_markdown(report)
    if args.md_out is not None:
        args.md_out.write_text(md, encoding="utf-8")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
