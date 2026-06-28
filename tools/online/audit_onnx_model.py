#!/usr/bin/env python3

from __future__ import annotations

from typing import Any

from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np

import onnx
from onnx import numpy_helper


def get_allowed_ops(preset: str) -> set[str]:
    presets = {
        "none": set(),
        "edge_npu_recommended": {
            "Add",
            "BatchNormalization",
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
            "Resize",
            "Relu",
            "Shape",
            "Sigmoid",
            "Slice",
            "Softmax",
            "Split",
            "Sqrt",
            "Squeeze",
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


def _initializer_arrays(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {initializer.name: numpy_helper.to_array(initializer) for initializer in model.graph.initializer}


def _constant_node_arrays(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    constants: dict[str, np.ndarray] = {}
    for node in model.graph.node:
        if node.op_type != "Constant" or not node.output:
            continue
        for attr in node.attribute:
            if attr.name == "value" and attr.HasField("t"):
                constants[node.output[0]] = numpy_helper.to_array(attr.t)
                break
    return constants


def _static_array(
    name: str,
    initializers: dict[str, np.ndarray],
    constants: dict[str, np.ndarray],
) -> np.ndarray | None:
    if not name:
        return None
    if name in initializers:
        return initializers[name]
    if name in constants:
        return constants[name]
    return None


def _value_shape(value_info: onnx.ValueInfoProto) -> list[int | None] | None:
    if not value_info.type.HasField("tensor_type"):
        return None
    shape = value_info.type.tensor_type.shape
    dims: list[int | None] = []
    for dim in shape.dim:
        if dim.dim_value > 0:
            dims.append(int(dim.dim_value))
        else:
            dims.append(None)
    return dims


def _shape_map(model: onnx.ModelProto) -> dict[str, list[int | None]]:
    try:
        inferred = onnx.shape_inference.infer_shapes(model)
    except Exception:  # noqa: BLE001 - audit should still work on partially inferable graphs
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


def _normalize_axes(axes: np.ndarray | None, rank: int) -> set[int] | None:
    if axes is None:
        return set(range(rank))
    try:
        values = axes.astype(np.int64).reshape(-1).tolist()
    except Exception:  # noqa: BLE001
        return None
    normalized = set()
    for axis in values:
        axis = int(axis)
        if axis < 0:
            axis += rank
        if axis < 0 or axis >= rank:
            return None
        normalized.add(axis)
    return normalized


def _append_sample(samples: dict[str, list[str]], key: str, node: onnx.NodeProto, limit: int = 8) -> None:
    bucket = samples.setdefault(key, [])
    if len(bucket) >= limit:
        return
    bucket.append(node.name or "/".join(node.output) or node.op_type)


def audit_npu_risks(model: onnx.ModelProto, *, transpose_threshold: int = 500) -> dict[str, Any]:
    """Return TIGER/ONE-derived graph-risk counts for strict edge exports.

    This is intentionally conservative.  It flags patterns that historically
    caused ONE import or record-minmax failures in this repo, even if a specific
    graph might still compile after simplification or a custom rewrite.
    """

    initializers = _initializer_arrays(model)
    initializer_names = set(initializers)
    constants = _constant_node_arrays(model)
    shapes = _shape_map(model)
    op_counts: dict[str, int] = {}
    samples: dict[str, list[str]] = {}

    for node in model.graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1

    risk_counts: dict[str, int] = {
        "tile": op_counts.get("Tile", 0),
        "constant_of_shape": op_counts.get("ConstantOfShape", 0),
        "expand": op_counts.get("Expand", 0),
        "prelu": op_counts.get("PRelu", 0),
        "dynamic_slice_bounds": 0,
        "dynamic_slice_with_dynamic_non_axis_dims": 0,
        "scalar_gather": 0,
        "activation_matmul_rank_le3": 0,
        "matmul_rank3_rhs2_nonconst": 0,
        "rank_gt4_values": 0,
        "transpose": op_counts.get("Transpose", 0),
        "transpose_perm_not_int32": 0,
        "high_transpose_count": int(op_counts.get("Transpose", 0) > transpose_threshold),
    }

    for op_type, key in (
        ("Tile", "tile"),
        ("ConstantOfShape", "constant_of_shape"),
        ("Expand", "expand"),
        ("PRelu", "prelu"),
    ):
        for node in model.graph.node:
            if node.op_type == op_type:
                _append_sample(samples, key, node)

    for value_name, shape in shapes.items():
        if len(shape) > 4:
            risk_counts["rank_gt4_values"] += 1
            samples.setdefault("rank_gt4_values", [])
            if len(samples["rank_gt4_values"]) < 8:
                samples["rank_gt4_values"].append(f"{value_name}:{shape}")

    for node in model.graph.node:
        if node.op_type == "Slice":
            starts = _static_array(node.input[1], initializers, constants) if len(node.input) > 1 else None
            ends = _static_array(node.input[2], initializers, constants) if len(node.input) > 2 else None
            axes = _static_array(node.input[3], initializers, constants) if len(node.input) > 3 else None
            dynamic_bounds = starts is None or ends is None
            if dynamic_bounds:
                risk_counts["dynamic_slice_bounds"] += 1
                _append_sample(samples, "dynamic_slice_bounds", node)

            data_shape = shapes.get(node.input[0]) if node.input else None
            if dynamic_bounds and data_shape is not None:
                normalized_axes = _normalize_axes(axes, len(data_shape))
                has_dynamic_non_axis = normalized_axes is None or any(
                    dim is None for idx, dim in enumerate(data_shape) if idx not in normalized_axes
                )
                if has_dynamic_non_axis:
                    risk_counts["dynamic_slice_with_dynamic_non_axis_dims"] += 1
                    _append_sample(samples, "dynamic_slice_with_dynamic_non_axis_dims", node)

        elif node.op_type == "Gather" and len(node.input) > 1:
            indices = _static_array(node.input[1], initializers, constants)
            if indices is not None and indices.ndim == 0:
                risk_counts["scalar_gather"] += 1
                _append_sample(samples, "scalar_gather", node)

        elif node.op_type == "MatMul" and len(node.input) >= 2:
            lhs, rhs = node.input[0], node.input[1]
            lhs_shape = shapes.get(lhs)
            rhs_shape = shapes.get(rhs)
            lhs_rank = len(lhs_shape) if lhs_shape is not None else None
            rhs_rank = len(rhs_shape) if rhs_shape is not None else None
            lhs_is_const = lhs in initializer_names or lhs in constants
            rhs_is_const = rhs in initializer_names or rhs in constants
            if (
                lhs_rank is not None
                and rhs_rank is not None
                and max(lhs_rank, rhs_rank) <= 3
                and not lhs_is_const
                and not rhs_is_const
            ):
                risk_counts["activation_matmul_rank_le3"] += 1
                _append_sample(samples, "activation_matmul_rank_le3", node)
            if lhs_rank == 3 and rhs_rank == 2 and not rhs_is_const:
                risk_counts["matmul_rank3_rhs2_nonconst"] += 1
                _append_sample(samples, "matmul_rank3_rhs2_nonconst", node)

        elif node.op_type == "Transpose" and len(node.input) > 1 and node.input[1]:
            perm = _static_array(node.input[1], initializers, constants)
            if perm is not None and perm.dtype != np.int32:
                risk_counts["transpose_perm_not_int32"] += 1
                _append_sample(samples, "transpose_perm_not_int32", node)

    strict_fail_keys = {
        "tile",
        "constant_of_shape",
        "expand",
        "prelu",
        "dynamic_slice_bounds",
        "dynamic_slice_with_dynamic_non_axis_dims",
        "scalar_gather",
        "activation_matmul_rank_le3",
        "matmul_rank3_rhs2_nonconst",
        "rank_gt4_values",
        "transpose_perm_not_int32",
        "high_transpose_count",
    }
    return {
        "risk_profile": "tiger_one_strict_edge",
        "transpose_threshold": transpose_threshold,
        "risk_counts": risk_counts,
        "risk_samples": samples,
        "strict_fail_keys": sorted(strict_fail_keys),
        "has_strict_edge_risks": any(risk_counts[key] > 0 for key in strict_fail_keys),
    }


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
        help=(
            "Optional JSON metadata emitted by export_onnx_online_model.py for "
            "streaming state and externalized constants."
        ),
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
    parser.add_argument(
        "--risk-profile",
        type=str,
        default="tiger_one_strict_edge",
        choices=["none", "tiger_one_strict_edge"],
        help="Run repo-specific strict-edge graph risk checks learned from TIGER/ONE failures.",
    )
    parser.add_argument(
        "--transpose-threshold",
        type=int,
        default=500,
        help="Transpose count above which the strict-edge risk audit reports high_transpose_count.",
    )
    parser.add_argument(
        "--fail-on-risk",
        action="store_true",
        help="Exit with status 2 when the selected risk profile finds strict-edge graph risks.",
    )
    parser.add_argument(
        "--risk-json-out",
        type=Path,
        help="Optional path to write the repo-specific risk audit payload as JSON.",
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
    pcen_state_budget_bytes = 0
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
        if "pcen_preprocessing" in payload:
            pcen_shape = payload["pcen_preprocessing"].get("state_shape")
            if isinstance(pcen_shape, list):
                pcen_state_budget_bytes = numel_from_shapes([pcen_shape]) * budget_element_size

    deployment_state_budget_bytes = streaming_state_budget_bytes + pcen_state_budget_bytes
    state_plus_initializers = deployment_state_budget_bytes + initializer_budget_bytes
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
            f"Core streaming state ({args.budget_dtype} estimate): "
            f"{format_bytes(streaming_state_budget_bytes)}"
        )
        if pcen_state_budget_bytes > 0:
            print(
                f"PCEN preprocessing state ({args.budget_dtype} estimate): "
                f"{format_bytes(pcen_state_budget_bytes)}"
            )
        print(
            f"Total deployment streaming state ({args.budget_dtype} estimate): "
            f"{format_bytes(deployment_state_budget_bytes)}"
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

    risk_payload: dict[str, Any] | None = None
    if args.risk_profile != "none":
        risk_payload = audit_npu_risks(model, transpose_threshold=args.transpose_threshold)
        print(f"Risk profile: {risk_payload['risk_profile']}")
        print("Risk counts:")
        for key, value in risk_payload["risk_counts"].items():
            print(f"  {key}: {value}")
        if risk_payload["risk_samples"]:
            print("Risk samples:")
            for key, values in sorted(risk_payload["risk_samples"].items()):
                print(f"  {key}: {', '.join(values)}")
        print(f"Strict-edge risks: {risk_payload['has_strict_edge_risks']}")
        if args.risk_json_out is not None:
            args.risk_json_out.parent.mkdir(parents=True, exist_ok=True)
            args.risk_json_out.write_text(json.dumps(risk_payload, indent=2, sort_keys=True), encoding="utf-8")

    if args.fail_on_disallowed_ops and disallowed:
        raise SystemExit(2)
    if args.fail_on_budget and args.state_meta is not None and state_plus_export_payload > budget_bytes:
        raise SystemExit(2)
    if args.fail_on_risk and risk_payload is not None and risk_payload["has_strict_edge_risks"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
