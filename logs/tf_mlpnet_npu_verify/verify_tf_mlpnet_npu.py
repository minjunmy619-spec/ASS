from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import helper


ROOT = Path("/home/cmj/works/ASS")
ONE_CMDS = Path("/home/cmj/works/ONE/build/compiler/one-cmds")
OUT_ROOT = ROOT / "logs" / "tf_mlpnet_npu_verify"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "TF-MLPNet") not in sys.path:
    sys.path.insert(0, str(ROOT / "TF-MLPNet"))


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
    if "one-import-onnx" in s or "onnx2circle" in s:
        return "import"
    if "one-optimize" in s or "circle2circle" in s:
        return "optimize"
    if "one-quantize" in s or "record-minmax" in s:
        return "quantize"
    if "one-codegen" in s:
        return "codegen"
    return "unknown"


def build_variants():
    from tf_mlpnet import TIGEREdgeMLP, V1TIGEREdgeMLP

    common = dict(
        num_blocks=4,
        upsampling_depth=2,
        num_sources=2,
        need_streaming=True,
    )
    return [
        (
            "tf_mlpnet_v2_24ch_4blk",
            TIGEREdgeMLP(
                out_channels=24,
                in_channels=96,
                edge_hidden_channels=24,
                edge_num_blocks=4,
                **common,
            ).eval(),
        ),
        (
            "tf_mlpnet_v2_32ch_4blk",
            TIGEREdgeMLP(
                out_channels=32,
                in_channels=128,
                edge_hidden_channels=32,
                edge_num_blocks=4,
                **common,
            ).eval(),
        ),
        (
            "tf_mlpnet_v1_24ch_4blk",
            V1TIGEREdgeMLP(
                out_channels=24,
                in_channels=96,
                edge_hidden_channels=24,
                edge_num_blocks=4,
                **common,
            ).eval(),
        ),
    ]


def export_variant_to_onnx(model, out_dir: Path) -> tuple[int, str, Path, tuple[int, ...]]:
    from tf_mlpnet import (
        TIGEREdgeMLPCellExportWrapper,
        build_tiger_edge_mlp_dummy_inputs,
        precheck_tiger_edge_mlp_export,
    )

    onnx_path = out_dir / "model.onnx"
    model = model.eval()
    wrapper = TIGEREdgeMLPCellExportWrapper(model).eval()
    inputs = build_tiger_edge_mlp_dummy_inputs(model, batch_size=1, device="cpu")
    precheck_tiger_edge_mlp_export(wrapper, inputs)
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

    x_shape = tuple(int(v) for v in inputs[0].shape)

    # Build calibration files from exact input tuple.
    list_path = out_dir / "calib_list.txt"
    npy_paths = []
    for idx, t in enumerate(inputs):
        p = out_dir / f"calib_sample000_input{idx:02d}.npy"
        np.save(p, t.detach().cpu().numpy().astype(np.float32))
        npy_paths.append(str(p))
    list_path.write_text(" ".join(npy_paths) + "\n", encoding="utf-8")

    return 0, "export-ok", onnx_path, x_shape


def run_onnxsim(py: Path, onnx_in: Path, onnx_out: Path, x_shape: tuple[int, ...], env: dict[str, str]) -> tuple[int, str]:
    shape = ",".join(str(v) for v in x_shape)
    cmd = (
        f'"{py}" -m onnxsim "{onnx_in}" "{onnx_out}" '
        f'--overwrite-input-shape subband_spec_RIs:{shape}'
    )
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


def write_onecc_cfg(out_dir: Path, onnx_path: Path, calib_h5: Path) -> Path:
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
            "replace_non_const_fc_with_batch_matmul=True",
            "convert_nchw_to_nhwc=True",
            "",
            "[one-quantize]",
            f"input_path={opt_path}",
            f"output_path={q_path}",
            f"input_data={calib_h5}",
            "input_data_format=h5",
            "quantized_dtype=uint8",
            "granularity=channel",
            "input_type=uint8",
            "output_type=uint8",
        ]
    )
    cfg = out_dir / "config.cfg"
    cfg.write_text(cfg_text, encoding="utf-8")
    return cfg


def main() -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}:{ROOT / 'TF-MLPNet'}:{env.get('PYTHONPATH', '')}".strip(":")
    env["PATH"] = f"{ONE_CMDS}:{env.get('PATH', '')}"
    lib_dirs = load_lib_dirs()
    if lib_dirs:
        env["LD_LIBRARY_PATH"] = ":".join(lib_dirs + [env.get("LD_LIBRARY_PATH", "")]).strip(":")
    env["NUMBA_DISABLE_JIT"] = "1"
    env["NUMBA_CACHE_DIR"] = str((ROOT / "logs" / ".numba_cache").resolve())
    Path(env["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

    py = ROOT / ".venv" / "bin" / "python"
    results: list[dict[str, object]] = []

    for name, model in build_variants():
        out_dir = OUT_ROOT / name
        out_dir.mkdir(parents=True, exist_ok=True)
        log_parts = []

        status = "PASS"
        fail_stage = ""
        export_rc = 0
        sim_rc = 0
        ds_rc = -1
        onecc_rc = -1

        try:
            export_rc, export_msg, onnx_path, x_shape = export_variant_to_onnx(model, out_dir)
            log_parts.append("=== EXPORT ===\n")
            log_parts.append(export_msg + "\n")
            log_parts.append(f"input_shape={x_shape}\n")
        except Exception as ex:  # noqa: BLE001
            export_rc = 1
            status = "FAIL"
            fail_stage = "export"
            log_parts.append("=== EXPORT ===\n")
            log_parts.append(f"{type(ex).__name__}: {ex}\n")
            onnx_path = out_dir / "model.onnx"
            x_shape = (1, 1, 1, 1)

        sim_path = out_dir / "model.sim.onnx"
        if status == "PASS":
            sim_rc, sim_out = run_onnxsim(py, onnx_path, sim_path, x_shape, env)
            log_parts.append("\n=== ONNXSIM ===\n")
            log_parts.append(sim_out)
            if sim_rc != 0:
                status = "FAIL"
                fail_stage = "onnxsim"

        if status == "PASS":
            clip_n = rewrite_clip_to_minmax(sim_path)
            log_parts.append(f"\n=== CLIP_REWRITE ===\nrewritten={clip_n}\n")

        calib_h5 = out_dir / "calib.h5"
        if status == "PASS":
            ds_cmd = (
                f'one-create-quant-dataset -i numpy '
                f'-l "{out_dir / "calib_list.txt"}" '
                f'-p "{calib_h5}"'
            )
            ds_rc, ds_out = sh(ds_cmd, env=env)
            log_parts.append("\n=== CALIB_DATASET ===\n")
            log_parts.append(ds_out)
            if ds_rc != 0 or not calib_h5.exists():
                status = "FAIL"
                fail_stage = "calibration"

        if status == "PASS":
            onecc_cfg = write_onecc_cfg(out_dir, sim_path, calib_h5)
            onecc_rc, onecc_out = sh(f'cd "{out_dir}" && onecc -C "{onecc_cfg}"', env=env)
            log_parts.append("\n=== ONECC ===\n")
            log_parts.append(onecc_out)
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
        print(f"[{status}] {name}" + (f" ({fail_stage})" if fail_stage else ""))

        results.append(
            {
                "model": name,
                "status": status,
                "fail_stage": fail_stage,
                "export_rc": export_rc,
                "onnxsim_rc": sim_rc,
                "dataset_rc": ds_rc,
                "onecc_rc": onecc_rc,
                "log": str(log_path),
            }
        )

    json_path = OUT_ROOT / "summary.json"
    md_path = OUT_ROOT / "summary.md"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    ok = sum(1 for r in results if r["status"] == "PASS")
    fail = len(results) - ok
    lines = [
        "# TF-MLPNet NPU Verification Summary",
        "",
        f"- Total: {len(results)}",
        f"- PASS: {ok}",
        f"- FAIL: {fail}",
        "",
        "| Variant | Status | Fail Stage |",
        "|---|---|---|",
    ]
    for r in results:
        lines.append(f"| {r['model']} | {r['status']} | {r['fail_stage'] or '-'} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nWrote: {json_path}")
    print(f"Wrote: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
