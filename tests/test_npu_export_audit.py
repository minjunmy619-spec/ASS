from __future__ import annotations

import numpy as np

try:
    import onnx
    from onnx import TensorProto, helper, numpy_helper
except ModuleNotFoundError:  # pragma: no cover - optional lightweight test environment
    onnx = None
    TensorProto = None
    helper = None
    numpy_helper = None

if onnx is not None:
    from tools.online.audit_onnx_model import audit_npu_risks
else:
    audit_npu_risks = None


def test_npu_risk_audit_flags_tiger_failure_patterns() -> None:
    if onnx is None:
        return

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 6, 67, None])
    starts = helper.make_tensor_value_info("starts", TensorProto.INT64, [1])
    ends = helper.make_tensor_value_info("ends", TensorProto.INT64, [1])
    gather_x = helper.make_tensor_value_info("gather_x", TensorProto.FLOAT, [1, 4])
    matmul_a = helper.make_tensor_value_info("matmul_a", TensorProto.FLOAT, [2, 3, 4])
    matmul_b = helper.make_tensor_value_info("matmul_b", TensorProto.FLOAT, [2, 4, 5])

    nodes = [
        helper.make_node("Slice", ["x", "starts", "ends", "axes", "steps"], ["slice_y"], name="dynamic_slice"),
        helper.make_node("Gather", ["gather_x", "scalar_index"], ["gather_y"], name="scalar_gather"),
        helper.make_node("MatMul", ["matmul_a", "matmul_b"], ["matmul_y"], name="activation_matmul"),
        helper.make_node("Tile", ["slice_y", "tile_repeats"], ["tile_y"], name="tile_resize"),
        helper.make_node("ConstantOfShape", ["cos_shape"], ["cos_y"], name="const_shape"),
    ]
    initializers = [
        numpy_helper.from_array(np.array([1], dtype=np.int64), name="axes"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), name="steps"),
        numpy_helper.from_array(np.array(0, dtype=np.int64), name="scalar_index"),
        numpy_helper.from_array(np.array([1, 1, 1, 1], dtype=np.int64), name="tile_repeats"),
        numpy_helper.from_array(np.array([2, 2], dtype=np.int64), name="cos_shape"),
    ]
    graph = helper.make_graph(
        nodes,
        "risk_graph",
        [x, starts, ends, gather_x, matmul_a, matmul_b],
        [
            helper.make_tensor_value_info("tile_y", TensorProto.FLOAT, None),
            helper.make_tensor_value_info("gather_y", TensorProto.FLOAT, None),
            helper.make_tensor_value_info("matmul_y", TensorProto.FLOAT, None),
            helper.make_tensor_value_info("cos_y", TensorProto.FLOAT, None),
        ],
        initializers,
    )
    payload = audit_npu_risks(helper.make_model(graph), transpose_threshold=1)

    counts = payload["risk_counts"]
    assert counts["tile"] == 1
    assert counts["constant_of_shape"] == 1
    assert counts["dynamic_slice_bounds"] == 1
    assert counts["dynamic_slice_with_dynamic_non_axis_dims"] == 1
    assert counts["scalar_gather"] == 1
    assert counts["activation_matmul_rank_le3"] == 1
    assert payload["has_strict_edge_risks"] is True


def test_npu_risk_audit_accepts_simple_static_graph() -> None:
    if onnx is None:
        return

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 3, 4])
    graph = helper.make_graph([helper.make_node("Relu", ["x"], ["y"], name="relu")], "clean_graph", [x], [y])

    payload = audit_npu_risks(helper.make_model(graph))

    assert payload["has_strict_edge_risks"] is False
