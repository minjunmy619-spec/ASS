#!/usr/bin/env python3

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path

import onnx
from onnx import numpy_helper


def get_allowed_ops(preset: str) -> set[str]:
    presets = {
        "none": set(),
        "edge_npu_recommended": {
            "Add",
            "Cast",
            "Clip",
            "Concat",
            "Constant",
            "ConstantOfShape",
            "Conv",
            "ConvTranspose",
            "Div",
            "Equal",
            "Expand",
            "Gather",
            "Identity",
            "MatMul",
            "Mul",
            "Pad",
            "Range",
            "ReduceMean",
            "ReduceSum",
            "Reshape",
            "Shape",
            "Sigmoid",
            "Slice",
            "Softmax",
            "Split",
            "Sqrt",
            "Sub",
            "Tanh",
            "Tile",
            "Transpose",
            "Unsqueeze",
            "Where",
        },
    }
    if preset not in presets:
        raise ValueError(f"Unsupported op preset: {preset}")
    return presets[preset]


def dtype_bytes(name: str) -> int:
    mapping = {
        "fp16": 2,
        "fp32": 4,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype name: {name}")
    return mapping[name]


def format_bytes(num_bytes: int) -> str:
    kib = num_bytes / 1024.0
    mib = kib / 1024.0
    if mib >= 1.0:
        return f"{num_bytes} B ({mib:.2f} MiB)"
    return f"{num_bytes} B ({kib:.2f} KiB)"


def numel_from_shapes(shapes: list[list[int]]) -> int:
    total = 0
    for shape in shapes:
        count = 1
        for dim in shape:
            count *= int(dim)
        total += count
    return total


def estimate_initializer_bytes(model: onnx.ModelProto, element_size: int) -> int:
    total_numel = 0
    for initializer in model.graph.initializer:
        dims = initializer.dims if initializer.dims else [1]
        count = 1
        for dim in dims:
            count *= int(dim)
        total_numel += count
    return total_numel * element_size


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("onnx_path", type=Path)
    parser.add_argument(
        "--op-preset",
        type=str,
        default="edge_npu_recommended",
        choices=["none", "edge_npu_recommended"],
    )
    parser.add_argument("--allow-op", action="append", default=[])
    parser.add_argument("--fail-on-disallowed-ops", action="store_true")
    parser.add_argument(
        "--state-meta",
        type=Path,
        help="Optional JSON metadata emitted by export_onnx_online_model.py for streaming state and externalized constants.",
    )
    parser.add_argument("--budget-kib", type=int, default=192)
    parser.add_argument(
        "--budget-dtype",
        type=str,
        default="fp16",
        choices=["fp16", "fp32"],
        help="Dtype used for deployment-memory estimates.",
    )
    parser.add_argument(
        "--fail-on-budget",
        action="store_true",
        help="Exit with status 2 when the selected deployment-memory totals exceed the budget.",
    )
    args = parser.parse_args()

    model = onnx.load(args.onnx_path)
    ops = sorted({node.op_type for node in model.graph.node})
    op_counts: dict[str, int] = {}
    for node in model.graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1

    initializer_bytes = 0
    for initializer in model.graph.initializer:
        initializer_bytes += numpy_helper.to_array(initializer).nbytes

    allowed_ops = get_allowed_ops(args.op_preset).union(args.allow_op)
    disallowed = sorted(op for op in ops if allowed_ops and op not in allowed_ops)
    budget_bytes = args.budget_kib * 1024
    budget_element_size = dtype_bytes(args.budget_dtype)
    initializer_budget_bytes = estimate_initializer_bytes(model, budget_element_size)
    streaming_state_budget_bytes = 0
    externalized_constant_budget_bytes = 0

    if args.state_meta is not None:
        payload = json.loads(args.state_meta.read_text(encoding="utf-8"))
        if "streaming_state" in payload:
            streaming_state_budget_bytes = (
                numel_from_shapes(payload["streaming_state"].get("shapes", [])) * budget_element_size
            )
        if "externalized_band_constants" in payload:
            externalized_constant_budget_bytes = (
                numel_from_shapes(payload["externalized_band_constants"].get("shapes", [])) * budget_element_size
            )

    state_plus_initializers = streaming_state_budget_bytes + initializer_budget_bytes
    state_plus_export_payload = state_plus_initializers + externalized_constant_budget_bytes

    print(f"Model: {args.onnx_path}")
    print(f"Ops ({len(ops)}): {', '.join(ops)}")
    print("Op counts:")
    for op in sorted(op_counts):
        print(f"  {op}: {op_counts[op]}")
    print(f"Initializers: {len(model.graph.initializer)} tensors, {initializer_bytes} bytes")
    print(
        f"Initializers ({args.budget_dtype} estimate): "
        f"{format_bytes(initializer_budget_bytes)}"
    )
    if args.state_meta is not None:
        print(f"State metadata: {args.state_meta}")
        print(
            f"Streaming state ({args.budget_dtype} estimate): "
            f"{format_bytes(streaming_state_budget_bytes)}"
        )
        print(
            f"Externalized band constants ({args.budget_dtype} estimate): "
            f"{format_bytes(externalized_constant_budget_bytes)}"
        )
        print(f"Budget: {format_bytes(budget_bytes)}")
        print(
            f"State + ONNX initializers ({args.budget_dtype}) within budget: "
            f"{state_plus_initializers <= budget_bytes}"
        )
        print(
            f"State + all exported parameter payload ({args.budget_dtype}) within budget: "
            f"{state_plus_export_payload <= budget_bytes}"
        )
        print(
            f"State + ONNX initializers ({args.budget_dtype}): "
            f"{format_bytes(state_plus_initializers)}"
        )
        print(
            f"State + all exported parameter payload ({args.budget_dtype}): "
            f"{format_bytes(state_plus_export_payload)}"
        )
    if args.op_preset != "none":
        print(f"Op preset: {args.op_preset}")
        if disallowed:
            print(f"Disallowed ops: {', '.join(disallowed)}")
        else:
            print("Disallowed ops: none")

    if args.fail_on_disallowed_ops and disallowed:
        raise SystemExit(2)
    if args.fail_on_budget and args.state_meta is not None and state_plus_export_payload > budget_bytes:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
