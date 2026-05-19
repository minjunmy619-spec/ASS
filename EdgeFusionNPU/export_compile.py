#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import onnx
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from EdgeFusionNPU.edge_fusion_npu import (  # noqa: E402
    EdgeFusionNPUExportWrapper,
    build_edge_fusion_npu_preset,
    count_parameters,
)


ONE_CMDS = Path("/home/cmj/works/ONE/build/compiler/one-cmds")
ONE_LIB_ROOT = Path("/home/cmj/works/ONE/build/compiler")


def one_env() -> dict[str, str]:
    env = os.environ.copy()
    lib_dirs = [
        ONE_LIB_ROOT / "luci/import",
        ONE_LIB_ROOT / "luci/export",
        ONE_LIB_ROOT / "luci/pass",
        ONE_LIB_ROOT / "luci/service",
        ONE_LIB_ROOT / "luci/lang",
        ONE_LIB_ROOT / "luci/env",
        ONE_LIB_ROOT / "luci/profile",
        ONE_LIB_ROOT / "luci/plan",
        ONE_LIB_ROOT / "luci/log",
        ONE_LIB_ROOT / "luci/logex",
        ONE_LIB_ROOT / "luci-compute",
        ONE_LIB_ROOT / "luci-interpreter/src",
        ONE_LIB_ROOT / "dio-hdf5",
        ONE_LIB_ROOT / "loco",
    ]
    env["PATH"] = f"{ONE_CMDS}:{env.get('PATH', '')}"
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}".strip(":")
    env["LD_LIBRARY_PATH"] = ":".join(str(p) for p in lib_dirs if p.exists()) + ":" + env.get(
        "LD_LIBRARY_PATH", ""
    )
    return env


def run(cmd: list[str], *, cwd: Path, env: dict[str, str]) -> str:
    proc = subprocess.run(cmd, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError("Command failed:\n" + " ".join(cmd) + "\n\n" + proc.stdout)
    return proc.stdout


def export_onnx(args: argparse.Namespace, out_dir: Path) -> tuple[Path, dict[str, object]]:
    overrides: dict[str, object] = {
        "n_chan": args.n_chan,
        "n_src": args.n_src,
    }
    if args.n_freq is not None:
        overrides["n_freq"] = args.n_freq
    model = build_edge_fusion_npu_preset(args.preset, **overrides).eval()
    wrapper = EdgeFusionNPUExportWrapper(model).eval()
    torch.manual_seed(args.seed)
    x = torch.randn(1, 2 * args.n_chan, model.n_freq, 1)
    state = model.init_states(batch_size=1)

    onnx_path = out_dir / "model.onnx"
    input_names = ["x", "state"]
    output_names = ["mask", "next_state"]
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (x, state),
            str(onnx_path),
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
            input_names=input_names,
            output_names=output_names,
        )
    onnx.checker.check_model(onnx.load(str(onnx_path)))
    param_count = count_parameters(model)
    state_elems = state.numel()
    return onnx_path, {
        "preset": args.preset,
        "n_freq": model.n_freq,
        "n_chan": args.n_chan,
        "n_src": args.n_src,
        "params": param_count,
        "param_fp16_kib": param_count * 2 / 1024.0,
        "param_fp32_kib": param_count * 4 / 1024.0,
        "state_elems": state_elems,
        "state_fp16_kib": state_elems * 2 / 1024.0,
        "state_fp32_kib": state_elems * 4 / 1024.0,
        "input_names": input_names,
        "output_names": output_names,
    }


def write_calibration(onnx_path: Path, out_dir: Path, env: dict[str, str]) -> Path:
    model = onnx.load(str(onnx_path))
    initializer_names = {x.name for x in model.graph.initializer}
    runtime_inputs = [i for i in model.graph.input if i.name not in initializer_names]
    npy_paths: list[str] = []
    rng = np.random.default_rng(0)
    for idx, vi in enumerate(runtime_inputs):
        dims = [d.dim_value if d.dim_value > 0 else 1 for d in vi.type.tensor_type.shape.dim]
        if idx == 0:
            arr = rng.standard_normal(dims).astype(np.float32)
        else:
            arr = np.zeros(dims, dtype=np.float32)
        path = out_dir / f"calib_sample000_input{idx:02d}.npy"
        np.save(path, arr)
        npy_paths.append(str(path))
    list_path = out_dir / "calib_list.txt"
    list_path.write_text(" ".join(npy_paths) + "\n", encoding="utf-8")
    calib_h5 = out_dir / "calib.h5"
    run(
        ["one-create-quant-dataset", "-i", "numpy", "-l", str(list_path), "-p", str(calib_h5)],
        cwd=out_dir,
        env=env,
    )
    return calib_h5


def write_onecc_cfg(out_dir: Path, onnx_path: Path, calib_h5: Path) -> Path:
    cfg = out_dir / "onecc.cfg"
    cfg.write_text(
        "\n".join(
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
                f"output_path={out_dir / 'model.circle'}",
                "",
                "[one-optimize]",
                f"input_path={out_dir / 'model.circle'}",
                f"output_path={out_dir / 'model.opt.circle'}",
                "replace_non_const_fc_with_batch_matmul=False",
                "convert_nchw_to_nhwc=True",
                "",
                "[one-quantize]",
                f"input_path={out_dir / 'model.opt.circle'}",
                f"output_path={out_dir / 'model.q.circle'}",
                f"input_data={calib_h5}",
                "input_data_format=h5",
                "quantized_dtype=uint8",
                "granularity=channel",
                "input_type=uint8",
                "output_type=uint8",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return cfg


def op_counts(onnx_path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in onnx.load(str(onnx_path)).graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    return dict(sorted(counts.items()))


def runtime_io_names(onnx_path: Path) -> tuple[list[str], list[str]]:
    model = onnx.load(str(onnx_path))
    initializer_names = {x.name for x in model.graph.initializer}
    inputs = [i.name for i in model.graph.input if i.name not in initializer_names]
    outputs = [o.name for o in model.graph.output]
    return inputs, outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and ONE-compile EdgeFusionNPU.")
    parser.add_argument("--preset", default="tiny")
    parser.add_argument("--out-dir", type=Path, default=Path("logs/npu_verify_general/edge_fusion_npu"))
    parser.add_argument(
        "--n-freq",
        type=int,
        default=None,
        help="Override preset frequency bins. When omitted, the preset's native n_freq is used.",
    )
    parser.add_argument("--n-chan", type=int, default=1)
    parser.add_argument("--n-src", type=int, default=3)
    parser.add_argument("--opset", type=int, default=14)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    env = one_env()

    onnx_path, manifest = export_onnx(args, out_dir)
    manifest["onnx_path"] = str(onnx_path)
    manifest["onnx_op_counts"] = op_counts(onnx_path)
    runtime_inputs, runtime_outputs = runtime_io_names(onnx_path)
    manifest["runtime_inputs"] = runtime_inputs
    manifest["runtime_outputs"] = runtime_outputs
    if len(runtime_inputs) > 4 or len(runtime_outputs) > 4:
        raise RuntimeError(
            f"Export exceeds 4-input/output limit: inputs={runtime_inputs}, outputs={runtime_outputs}"
        )

    if args.compile:
        calib_h5 = write_calibration(onnx_path, out_dir, env)
        cfg = write_onecc_cfg(out_dir, onnx_path, calib_h5)
        onecc_log = run(["onecc", "-C", str(cfg)], cwd=out_dir, env=env)
        (out_dir / "onecc.log").write_text(onecc_log, encoding="utf-8")
        manifest["calib_h5"] = str(calib_h5)
        manifest["onecc_cfg"] = str(cfg)
        manifest["circle"] = str(out_dir / "model.circle")
        manifest["opt_circle"] = str(out_dir / "model.opt.circle")
        manifest["q_circle"] = str(out_dir / "model.q.circle")
        manifest["compiled"] = (out_dir / "model.q.circle").exists()

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
