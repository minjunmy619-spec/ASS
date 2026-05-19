from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch
from onnx import numpy_helper
from onnx import TensorProto, helper


ROOT = Path("/home/cmj/works/ASS")
ONE_CMDS = Path("/home/cmj/works/ONE/build/compiler/one-cmds")
DEFAULT_OUT_ROOT = ROOT / "logs" / "npu_verify_general"


@dataclass
class Variant:
    kind: str  # recipe | tf
    name: str
    recipe_cfg: Path | None = None
    tf_builder: str | None = None


def sh(cmd: str, env: dict[str, str] | None = None) -> tuple[int, str]:
    p = subprocess.run(cmd, shell=True, text=True, capture_output=True, env=env)
    out = (p.stdout or "") + (p.stderr or "")
    return p.returncode, out


def load_lib_dirs() -> list[str]:
    cands = [
        Path("/home/cmj/works/ONE/build/compiler"),
        Path("/home/cmj/works/ONE/build/compiler/compiler"),
    ]
    base = None
    for c in cands:
        if c.exists():
            base = c
            break
    if base is None:
        return []
    rels = [
        "loco",
        "logo-core",
        "locop",
        "safemain",
        "mio-circle08",
        "mio-circle",
        "crew",
        "foder",
        "luci/import",
        "luci/lang",
        "luci/logex",
        "luci/pass/src",
        "luci/partition",
        "luci/plan",
        "luci/service",
        "luci-interpreter/src",
        "dio-hdf5",
    ]
    return [str((base / r).resolve()) for r in rels if (base / r).exists()]


def first_error_stage(log_text: str) -> str:
    s = log_text.lower()
    if (
        "shape inference failed" in s
        or "failed to scan parameters successfully" in s
        or "incompatible broadcast" in s
        or "failed to legalize operation" in s
    ):
        return "import"
    if "one-import-onnx" in s or "onnx2circle" in s:
        return "import"
    if "one-optimize" in s or "circle2circle" in s:
        return "optimize"
    if "one-quantize" in s or "record-minmax" in s or "circle_quantizer" in s or "circle-quantizer" in s:
        return "quantize"
    if "one-codegen" in s:
        return "codegen"
    return "unknown"


def infer_n_chan(recipe_cfg: Path) -> int:
    text = recipe_cfg.read_text(encoding="utf-8")
    m = re.search(r"(?m)^\s*n_chan:\s*(\d+)\s*$", text)
    return int(m.group(1)) if m else 2


def discover_recipe_variants(recipe_root: Path, name_contains: str | None = None) -> list[Variant]:
    items = []
    for cfg in sorted(recipe_root.glob("*/config.yaml")):
        name = cfg.parent.name
        if name_contains and name_contains.lower() not in name.lower():
            continue
        items.append(Variant(kind="recipe", name=name, recipe_cfg=cfg))
    return items


def discover_tf_variants() -> list[Variant]:
    return [
        Variant(kind="tf", name="tf_mlpnet_v2_24ch_4blk", tf_builder="v2_24ch_4blk"),
        Variant(kind="tf", name="tf_mlpnet_v2_32ch_4blk", tf_builder="v2_32ch_4blk"),
        Variant(kind="tf", name="tf_mlpnet_v1_24ch_4blk", tf_builder="v1_24ch_4blk"),
    ]


def build_tf_model(builder_name: str):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if str(ROOT / "TF-MLPNet") not in sys.path:
        sys.path.insert(0, str(ROOT / "TF-MLPNet"))

    from tf_mlpnet import TIGEREdgeMLP, V1TIGEREdgeMLP

    common = dict(num_blocks=4, upsampling_depth=2, num_sources=2, need_streaming=True)
    if builder_name == "v2_24ch_4blk":
        return TIGEREdgeMLP(
            out_channels=24,
            in_channels=96,
            edge_hidden_channels=24,
            edge_num_blocks=4,
            **common,
        ).eval()
    if builder_name == "v2_32ch_4blk":
        return TIGEREdgeMLP(
            out_channels=32,
            in_channels=128,
            edge_hidden_channels=32,
            edge_num_blocks=4,
            **common,
        ).eval()
    if builder_name == "v1_24ch_4blk":
        return V1TIGEREdgeMLP(
            out_channels=24,
            in_channels=96,
            edge_hidden_channels=24,
            edge_num_blocks=4,
            **common,
        ).eval()
    raise ValueError(f"Unknown TF builder: {builder_name}")


def export_recipe_variant(
    variant: Variant,
    out_dir: Path,
    py: Path,
    env: dict[str, str],
) -> tuple[int, str, Path]:
    assert variant.recipe_cfg is not None
    export_script = ROOT / "tools" / "online" / "export_onnx_online_model.py"
    onnx_path = out_dir / "model.onnx"
    manifest_path = out_dir / "manifest.json"
    n_chan = infer_n_chan(variant.recipe_cfg)
    export_cmd = (
        f'"{py}" "{export_script}" "{variant.recipe_cfg}" '
        f'--out "{onnx_path}" --n-chan {n_chan} --frames 1 --opset 14 '
        f'--disable-masking --deploy-manifest-out "{manifest_path}"'
    )
    rc, out = sh(export_cmd, env=env)
    mismatch = "expected input[1, 4" in out and "to have 2 channels" in out
    if rc != 0 and mismatch and n_chan != 1:
        retry_cmd = (
            f'"{py}" "{export_script}" "{variant.recipe_cfg}" '
            f'--out "{onnx_path}" --n-chan 1 --frames 1 --opset 14 '
            f'--disable-masking --deploy-manifest-out "{manifest_path}"'
        )
        rrc, rout = sh(retry_cmd, env=env)
        out += "\n\n=== EXPORT RETRY (--n-chan 1) ===\n" + rout
        rc = rrc
    return rc, out, onnx_path


def export_tf_variant(variant: Variant, out_dir: Path) -> tuple[int, str, Path]:
    assert variant.tf_builder is not None
    from tf_mlpnet import TIGEREdgeMLPCellExportWrapper, build_tiger_edge_mlp_dummy_inputs

    onnx_path = out_dir / "model.onnx"
    model = build_tf_model(variant.tf_builder).eval()
    wrapper = TIGEREdgeMLPCellExportWrapper(model).eval()
    inputs = build_tiger_edge_mlp_dummy_inputs(model, batch_size=1, device="cpu")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            inputs,
            str(onnx_path),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            dynamo=False,
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
        )
    return 0, "export-ok", onnx_path


def input_value_infos(model: onnx.ModelProto):
    initializer_names = {x.name for x in model.graph.initializer}
    return [i for i in model.graph.input if i.name not in initializer_names]


def onnx_shape_from_vi(vi: onnx.ValueInfoProto) -> list[int]:
    dims: list[int] = []
    for d in vi.type.tensor_type.shape.dim:
        if d.dim_value and d.dim_value > 0:
            dims.append(int(d.dim_value))
        else:
            dims.append(1)
    return dims


def run_onnxsim(
    py: Path,
    onnx_in: Path,
    onnx_out: Path,
    env: dict[str, str],
    *,
    force_large_shape_ops: bool = False,
) -> tuple[int, str]:
    model = onnx.load(str(onnx_in))
    op_types = {node.op_type for node in model.graph.node}
    large_shape_ops = sorted(op_types & {"Tile", "ConstantOfShape"})
    if large_shape_ops and not force_large_shape_ops:
        onnx.save(model, str(onnx_out))
        return (
            0,
            "Skipped onnxsim because the graph contains large-shape ops that can make "
            f"simplification impractically slow: {', '.join(large_shape_ops)}\n"
            "Pass --force-onnxsim-large-shape-ops to run simplification anyway.\n",
        )
    vis = input_value_infos(model)
    if vis:
        first = vis[0]
        shape = ",".join(str(v) for v in onnx_shape_from_vi(first))
        cmd = (
            f'"{py}" -m onnxsim "{onnx_in}" "{onnx_out}" '
            f'--overwrite-input-shape {first.name}:{shape} --no-large-tensor'
        )
        rc, out = sh(cmd, env=env)
        if rc == 0:
            return rc, out
        fallback_cmd = f'"{py}" -m onnxsim "{onnx_in}" "{onnx_out}" --no-large-tensor'
        frc, fout = sh(fallback_cmd, env=env)
        return frc, out + "\n\n=== ONNXSIM RETRY (no overwrite-input-shape) ===\n" + fout
    cmd = f'"{py}" -m onnxsim "{onnx_in}" "{onnx_out}" --no-large-tensor'
    return sh(cmd, env=env)


def rewrite_clip_to_minmax(onnx_path: Path) -> int:
    model = onnx.load(str(onnx_path))
    graph = model.graph
    new_nodes = []
    touched = 0

    for n in graph.node:
        if n.op_type != "Clip":
            new_nodes.append(n)
            continue
        x = n.input[0]
        min_in = n.input[1] if len(n.input) > 1 else ""
        max_in = n.input[2] if len(n.input) > 2 else ""
        out = n.output[0]

        if min_in and max_in:
            t1 = out + "_clip_max"
            new_nodes.append(helper.make_node("Max", [x, min_in], [t1], name=n.name + "_max"))
            new_nodes.append(helper.make_node("Min", [t1, max_in], [out], name=n.name + "_min"))
            touched += 1
        elif min_in:
            new_nodes.append(helper.make_node("Max", [x, min_in], [out], name=n.name + "_max"))
            touched += 1
        elif max_in:
            new_nodes.append(helper.make_node("Min", [x, max_in], [out], name=n.name + "_min"))
            touched += 1
        else:
            new_nodes.append(n)

    if touched > 0:
        del graph.node[:]
        graph.node.extend(new_nodes)
        onnx.save(model, str(onnx_path))
    return touched


def _sanitize_onnx_identity(s: str) -> str:
    out = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in (s or "node"))
    if out and out[0].isdigit():
        out = "n_" + out
    return out or "npu"


def rewrite_transpose_perm_int64_to_int32(onnx_path: Path) -> tuple[int, int]:
    """
    luci-interpreter Transpose reads perm as int32 elements; INT64 constants are mis-decoded and
    `perm()->shape().dim(0) == input_rank` fails under record-minmax.

    Handles:
    - INT64 initializer at end of Cast/Identity chain: convert to INT32 and rewire Transpose to
      consume the initializer directly (bypasses Cast output that may still be INT64)
    - INT64 tensor inside a Constant node used as perm
    """
    model = onnx.load(str(onnx_path))
    graph = model.graph
    init_by_name = {t.name: t for t in graph.initializer}
    int64_tensors_fixed = 0
    perm_rewired_to_initializer = 0
    graph_modified = False

    def producer_of(tensor: str) -> onnx.NodeProto | None:
        for n in graph.node:
            if tensor in n.output:
                return n
        return None

    def int64_tp_to_i32(tp: onnx.TensorProto) -> onnx.TensorProto:
        arr = onnx.numpy_helper.to_array(tp).astype(np.int32, copy=False)
        return onnx.numpy_helper.from_array(arr.astype(np.int32), name=tp.name)

    def trace_perm_source(perm_out: str) -> tuple[str | None, onnx.NodeProto | None]:
        """
        Returns (initializer_name, constant_node).
        Exactly one of the two is set when trace succeeds; (None, None) if unsupported.
        """
        cur = perm_out
        seen: set[str] = set()
        while cur and cur not in seen:
            seen.add(cur)
            if cur in init_by_name:
                return cur, None
            pr = producer_of(cur)
            if pr is None:
                return None, None
            if pr.op_type == "Constant":
                return None, pr
            if pr.op_type in ("Cast", "Identity"):
                cur = pr.input[0]
                continue
            return None, None
        return None, None

    for node in list(graph.node):
        if node.op_type != "Transpose":
            continue
        if len(node.input) < 2 or not node.input[1]:
            continue
        perm_out = node.input[1]
        init_nm, const_n = trace_perm_source(perm_out)
        if init_nm is not None:
            init = init_by_name[init_nm]
            if init.data_type == TensorProto.INT64:
                new_t = int64_tp_to_i32(init)
                for i, t in enumerate(graph.initializer):
                    if t.name == init_nm:
                        graph.initializer.pop(i)
                        graph.initializer.insert(i, new_t)
                        break
                init_by_name[init_nm] = new_t
                int64_tensors_fixed += 1
                graph_modified = True
            # Always wire perm to the initializer when we can trace it, so all Transpose ops
            # bypass Cast/Identity chains that may widen dtype back to INT64.
            if perm_out != init_nm:
                perm_rewired_to_initializer += 1
                graph_modified = True
            node.input[1] = init_nm
        elif const_n is not None:
            val_attr = None
            for attr in const_n.attribute:
                if attr.name == "value":
                    val_attr = attr
                    break
            if val_attr is None or val_attr.t.data_type != TensorProto.INT64:
                continue
            new_inner = int64_tp_to_i32(val_attr.t)
            new_inner.name = val_attr.t.name
            val_attr.t.CopyFrom(new_inner)
            int64_tensors_fixed += 1
            graph_modified = True

    if graph_modified:
        onnx.save(model, str(onnx_path))
    return int64_tensors_fixed, perm_rewired_to_initializer


def rewrite_matmul_stack_lhs3_rhs2_flatten(onnx_path: Path) -> int:
    """
    ONNX rank-3 x rank-2 matmul stacks batch matmul over the last two LHS dims.
    After import, non-const FullyConnected + replace_non_const_fc_with_batch_matmul can
    produce BatchMatMul shapes that fail during record-minmax. Flatten LHS to 2D keeps
    ONNX math and often yields a simpler circle graph.

    Applies only when onnx shape inference exposes fully static (>0) ranks.
    """
    model = onnx.load(str(onnx_path))
    graph = model.graph
    try:
        inferred = onnx.shape_inference.infer_shapes(model)
    except Exception:
        return 0

    ig = inferred.graph
    shapes: dict[str, list[int]] = {}
    for init in graph.initializer:
        shapes[init.name] = list(init.dims)
    for group in (ig.input, ig.output, ig.value_info):
        for vi in group:
            if not vi.type.HasField("tensor_type"):
                continue
            dims: list[int] = []
            for d in vi.type.tensor_type.shape.dim:
                if not d.dim_value:
                    dims = []
                    break
                dims.append(int(d.dim_value))
            if dims:
                shapes[vi.name] = dims

    specs: list[tuple[int, onnx.NodeProto, tuple[int, int, int, int]]] = []
    for idx, node in enumerate(graph.node):
        if node.op_type != "MatMul" or len(node.input) < 2:
            continue
        a, b = node.input[0], node.input[1]
        sa, sb = shapes.get(a), shapes.get(b)
        if not sa or not sb or len(sa) != 3 or len(sb) != 2:
            continue
        b0, b1, k = sa
        rk, tout = sb
        if k != rk or min(b0, b1, k, rk, tout) <= 0:
            continue
        specs.append((idx, node, (b0, b1, k, tout)))

    existing_ini = {i.name for i in graph.initializer}
    touched = 0
    for run_id, (idx, node, (batch0, batch1, k, bout)) in enumerate(
        sorted(specs, key=lambda x: x[0], reverse=True)
    ):
        stem = _sanitize_onnx_identity(node.name or "matmul")
        a, b = node.input[0], node.input[1]
        out_name = node.output[0]
        flat_leading = batch0 * batch1

        lh_shape_name = f"{stem}_npu_lhsh_{idx}_{run_id}"
        oh_name = f"{stem}_npu_oush_{idx}_{run_id}"
        if lh_shape_name in existing_ini or oh_name in existing_ini:
            continue
        existing_ini.add(lh_shape_name)
        existing_ini.add(oh_name)

        lhs_flat = f"{stem}_nlxf_{idx}_{run_id}"
        flat_mm_o = f"{stem}_nfmm_{idx}_{run_id}"

        graph.initializer.extend(
            [
                onnx.numpy_helper.from_array(
                    np.array([flat_leading, k], dtype=np.int64), name=lh_shape_name
                ),
                onnx.numpy_helper.from_array(
                    np.array([batch0, batch1, bout], dtype=np.int64), name=oh_name
                ),
            ]
        )

        ra = helper.make_node(
            "Reshape",
            [a, lh_shape_name],
            [lhs_flat],
            name=f"{stem}_nrsh_in_{run_id}",
        )
        mm = helper.make_node(
            "MatMul",
            [lhs_flat, b],
            [flat_mm_o],
            name=f"{stem}_nm2d_{run_id}",
        )
        rb = helper.make_node(
            "Reshape",
            [flat_mm_o, oh_name],
            [out_name],
            name=f"{stem}_nrsh_ot_{run_id}",
        )

        del graph.node[idx]
        graph.node.insert(idx, ra)
        graph.node.insert(idx + 1, mm)
        graph.node.insert(idx + 2, rb)
        touched += 1

    if touched:
        onnx.save(model, str(onnx_path))
    return touched


def apply_npu_onnx_compat_for_circle_import(onnx_path: Path) -> dict[str, int]:
    """ONNX tweaks so stock ONE importer + interpreter survive record-minmax."""
    tp_i64, tp_rew = rewrite_transpose_perm_int64_to_int32(onnx_path)
    mm = rewrite_matmul_stack_lhs3_rhs2_flatten(onnx_path)
    return {
        "transpose_perm_i64_tensors": tp_i64,
        "transpose_perm_rewired_to_init": tp_rew,
        "matmul_flatten_lhs3_rhs2": mm,
    }


def dtype_to_numpy(elem_type: int):
    if elem_type == TensorProto.FLOAT:
        return np.float32
    if elem_type == TensorProto.DOUBLE:
        return np.float64
    if elem_type == TensorProto.FLOAT16:
        return np.float16
    if elem_type in (TensorProto.INT32, TensorProto.INT16, TensorProto.INT8):
        return np.int32
    if elem_type in (TensorProto.INT64,):
        return np.int64
    if elem_type in (TensorProto.UINT8, TensorProto.UINT16, TensorProto.UINT32, TensorProto.UINT64):
        return np.uint32
    if elem_type == TensorProto.BOOL:
        return np.bool_
    return np.float32


def generate_calibration_from_onnx(onnx_path: Path, out_dir: Path, env: dict[str, str]) -> tuple[int, str, Path]:
    model = onnx.load(str(onnx_path))
    vis = input_value_infos(model)
    if not vis:
        return 1, "No ONNX runtime inputs discovered.", out_dir / "calib.h5"

    npy_paths = []
    for idx, vi in enumerate(vis):
        shape = onnx_shape_from_vi(vi)
        np_dtype = dtype_to_numpy(vi.type.tensor_type.elem_type)
        if np_dtype in (np.float32, np.float64, np.float16):
            arr = np.random.randn(*shape).astype(np_dtype)
        elif np_dtype == np.bool_:
            arr = (np.random.rand(*shape) > 0.5).astype(np.bool_)
        else:
            arr = np.random.randint(0, 3, size=shape, dtype=np.int64).astype(np_dtype)
        p = out_dir / f"calib_sample000_input{idx:02d}.npy"
        np.save(p, arr)
        npy_paths.append(str(p))

    list_path = out_dir / "calib_list.txt"
    list_path.write_text(" ".join(npy_paths) + "\n", encoding="utf-8")
    calib_h5 = out_dir / "calib.h5"
    ds_cmd = (
        f'one-create-quant-dataset -i numpy '
        f'-l "{list_path}" '
        f'-p "{calib_h5}"'
    )
    rc, out = sh(ds_cmd, env=env)
    return rc, out, calib_h5


def write_onecc_cfg(
    out_dir: Path, onnx_path: Path, calib_h5: Path, granularity: str = "channel"
) -> Path:
    circle_path = out_dir / "model.circle"
    opt_path = out_dir / "model.opt.circle"
    q_path = out_dir / "model.q.circle"

    cfg_text = "\n".join(
        [
            "[Environment]",
            'ONECC_ENV="ONECC"',
            "",
            "[backend]",
            "target=",
            "",
            "[onecc]",
            "one-import-tf=False",
            "one-import-tflite=False",
            "one-import-bcq=False",
            "one-import-onnx=True",
            "one-optimize=True",
            "one-quantize=True",
            "one-partition=False",
            "one-pack=False",
            "one-codegen=False",
            "one-profile=False",
            "one-infer=False",
            "",
            "[one-import-onnx]",
            f"input_path={onnx_path}",
            f"output_path={circle_path}",
            "",
            "[one-optimize]",
            f"input_path={circle_path}",
            f"output_path={opt_path}",
            "replace_non_const_fc_with_batch_matmul=False",
            "convert_nchw_to_nhwc=True",
            "",
            "[one-quantize]",
            f"input_path={opt_path}",
            f"output_path={q_path}",
            f"input_data={calib_h5}",
            "input_data_format=h5",
            "quantized_dtype=uint8",
            f"granularity={granularity}",
            "input_type=uint8",
            "output_type=uint8",
        ]
    )
    onecc_cfg = out_dir / "config.cfg"
    onecc_cfg.write_text(cfg_text, encoding="utf-8")
    return onecc_cfg


def run_one_variant(
    variant: Variant,
    out_dir: Path,
    py: Path,
    env: dict[str, str],
    *,
    quantize_layer_fallback: bool = False,
    force_onnxsim_large_shape_ops: bool = False,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_parts: list[str] = []
    status = "PASS"
    fail_stage = ""
    export_rc = 0
    sim_rc = 0
    ds_rc = -1
    onecc_rc = -1
    quant_granularity_used = "channel"
    quant_retry_layer = False

    if variant.kind == "recipe":
        export_rc, export_out, onnx_path = export_recipe_variant(variant, out_dir, py, env)
    else:
        try:
            export_rc, export_out, onnx_path = export_tf_variant(variant, out_dir)
        except Exception as ex:  # noqa: BLE001
            export_rc, export_out, onnx_path = 1, f"{type(ex).__name__}: {ex}", out_dir / "model.onnx"

    log_parts.append("=== EXPORT ===\n")
    log_parts.append(export_out + "\n")
    if export_rc != 0 or not onnx_path.exists():
        status = "FAIL"
        fail_stage = "export"

    sim_path = out_dir / "model.sim.onnx"
    if status == "PASS":
        sim_rc, sim_out = run_onnxsim(
            py,
            onnx_path,
            sim_path,
            env,
            force_large_shape_ops=force_onnxsim_large_shape_ops,
        )
        log_parts.append("\n=== ONNXSIM ===\n")
        log_parts.append(sim_out + "\n")
        if sim_rc != 0 or not sim_path.exists():
            status = "FAIL"
            fail_stage = "onnxsim"

    if status == "PASS":
        rewritten = rewrite_clip_to_minmax(sim_path)
        log_parts.append("\n=== CLIP_REWRITE ===\n")
        log_parts.append(f"rewritten={rewritten}\n")

    if status == "PASS":
        prep = apply_npu_onnx_compat_for_circle_import(sim_path)
        log_parts.append("\n=== NPU_ONNX_IMPORT_PREP ===\n")
        log_parts.append(f"{prep}\n")

    calib_h5 = out_dir / "calib.h5"
    if status == "PASS":
        ds_rc, ds_out, calib_h5 = generate_calibration_from_onnx(sim_path, out_dir, env)
        log_parts.append("\n=== CALIB_DATASET ===\n")
        log_parts.append(ds_out + "\n")
        if ds_rc != 0 or not calib_h5.exists():
            status = "FAIL"
            fail_stage = "calibration"

    if status == "PASS":
        onecc_cfg = write_onecc_cfg(out_dir, sim_path, calib_h5, granularity="channel")
        onecc_rc, onecc_out = sh(f'cd "{out_dir}" && onecc -C "{onecc_cfg}"', env=env)
        log_parts.append("\n=== ONECC ===\n")
        log_parts.append(onecc_out + "\n")
        if (
            quantize_layer_fallback
            and onecc_rc != 0
            and "Non-channel dimension of const node must be 1" in onecc_out
        ):
            quant_retry_layer = True
            quant_granularity_used = "layer"
            onecc_cfg = write_onecc_cfg(out_dir, sim_path, calib_h5, granularity="layer")
            retry_rc, retry_out = sh(f'cd "{out_dir}" && onecc -C "{onecc_cfg}"', env=env)
            log_parts.append("\n=== ONECC RETRY (granularity=layer) ===\n")
            log_parts.append(retry_out + "\n")
            onecc_rc = retry_rc
            onecc_out = onecc_out + "\n\n=== RETRY ===\n" + retry_out

        if onecc_rc != 0:
            status = "FAIL"
            fail_stage = first_error_stage(onecc_out)
        else:
            if not (out_dir / "model.circle").exists():
                status = "FAIL"
                fail_stage = "import"
            elif not (out_dir / "model.opt.circle").exists():
                status = "FAIL"
                fail_stage = "optimize"
            elif not (out_dir / "model.q.circle").exists():
                status = "FAIL"
                fail_stage = "quantize"

    log_path = out_dir / "run.log"
    log_path.write_text("".join(log_parts), encoding="utf-8")
    return {
        "kind": variant.kind,
        "variant": variant.name,
        "status": status,
        "fail_stage": fail_stage,
        "export_rc": export_rc,
        "onnxsim_rc": sim_rc,
        "dataset_rc": ds_rc,
        "onecc_rc": onecc_rc,
        "quant_granularity_used": quant_granularity_used,
        "quant_retry_layer": quant_retry_layer,
        "log": str(log_path),
        "path": str(out_dir),
        "recipe_cfg": str(variant.recipe_cfg) if variant.recipe_cfg else "",
        "tf_builder": variant.tf_builder or "",
    }


def build_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = f"{ONE_CMDS}:{env.get('PATH', '')}"
    env["PYTHONPATH"] = f"{ROOT}:{ROOT / 'TF-MLPNet'}:{env.get('PYTHONPATH', '')}".strip(":")
    lib_dirs = load_lib_dirs()
    if lib_dirs:
        env["LD_LIBRARY_PATH"] = ":".join(lib_dirs + [env.get("LD_LIBRARY_PATH", "")]).strip(":")
    env["NUMBA_DISABLE_JIT"] = "1"
    env["NUMBA_CACHE_DIR"] = str((ROOT / "logs" / ".numba_cache").resolve())
    Path(env["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    return env


def write_summary(results: list[dict[str, Any]], out_root: Path) -> None:
    json_path = out_root / "summary.json"
    md_path = out_root / "summary.md"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    ok = sum(1 for r in results if r["status"] == "PASS")
    fail = len(results) - ok
    lines = [
        "# NPU Variant Verification Summary",
        "",
        f"- Total: {len(results)}",
        f"- PASS: {ok}",
        f"- FAIL: {fail}",
        "",
        "| Kind | Variant | Status | Fail Stage |",
        "|---|---|---|---|",
    ]
    for r in results:
        lines.append(f"| {r['kind']} | {r['variant']} | {r['status']} | {r['fail_stage'] or '-'} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="General end-to-end NPU verifier for recipe and TF-MLPNet variants."
    )
    p.add_argument("--mode", choices=["all", "recipe", "tf"], default="all")
    p.add_argument(
        "--recipe-root",
        type=Path,
        default=ROOT / "recipes" / "dnr" / "models",
        help="Recipe parent folder containing variant subfolders with config.yaml",
    )
    p.add_argument(
        "--recipe-name-contains",
        default=None,
        help="Optional keyword filter for recipe variant folder names (e.g. sfc, tiger)",
    )
    p.add_argument(
        "--run-name",
        default=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Subfolder name under output root",
    )
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--limit", type=int, default=0, help="Optional max number of variants (0 means no limit)")
    p.add_argument("--dry-run", action="store_true", help="Only discover and print variants")
    p.add_argument(
        "--quantize-layer-fallback",
        action="store_true",
        help=(
            "If channel-wise quantization fails with "
            "\"Non-channel dimension of const node must be 1\", retry once with "
            "granularity=layer (hurts accuracy vs channel-wise; prefer ONNX/model fixes)."
        ),
    )
    p.add_argument(
        "--force-onnxsim-large-shape-ops",
        action="store_true",
        help=(
            "Run onnxsim even when the graph contains Tile or ConstantOfShape. "
            "This can be slow but is useful for diagnosing older TIGER graphs."
        ),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    py = ROOT / ".venv" / "bin" / "python"
    if not py.exists():
        print(f"Python not found: {py}")
        return 2

    variants: list[Variant] = []
    if args.mode in ("all", "recipe"):
        variants.extend(discover_recipe_variants(args.recipe_root, args.recipe_name_contains))
    if args.mode in ("all", "tf"):
        variants.extend(discover_tf_variants())

    if args.limit > 0:
        variants = variants[: args.limit]

    if not variants:
        print("No variants discovered.")
        return 1

    out_root = args.output_root / args.run_name
    out_root.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print(f"Discovered variants: {len(variants)}")
        for v in variants:
            if v.kind == "recipe":
                print(f"[recipe] {v.name} :: {v.recipe_cfg}")
            else:
                print(f"[tf] {v.name} :: builder={v.tf_builder}")
        return 0

    env = build_env()
    results = []
    for v in variants:
        out_dir = out_root / v.name
        result = run_one_variant(
            v,
            out_dir,
            py,
            env,
            quantize_layer_fallback=args.quantize_layer_fallback,
            force_onnxsim_large_shape_ops=args.force_onnxsim_large_shape_ops,
        )
        results.append(result)
        status = result["status"]
        fail_stage = result["fail_stage"]
        print(f"[{status}] {v.kind}:{v.name}" + (f" ({fail_stage})" if fail_stage else ""))

    write_summary(results, out_root)
    print(f"\nWrote summary to: {out_root / 'summary.md'}")
    print(f"Wrote details to: {out_root / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
