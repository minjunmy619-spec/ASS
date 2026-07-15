#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT = Path("/home/cmj/works/ASS")
ONE_CMDS = Path(os.environ.get("ONE_CMDS", "/home/cmj/works/ONE/build/compiler/one-cmds"))
ONE_BUILD_COMPILER = Path(os.environ.get("ONE_BUILD_COMPILER", str(ONE_CMDS.parent)))
OUT_DIR = ROOT / "logs" / "npu_efficiency_audit" / "sfc_small_conv2d_bn_npu_20260715" / "nhwc_abi_rawmask"
CONFIG = ROOT / "recipes" / "dnr" / "models" / "sfc-small-conv2d-bn-npu.musical64.onfly.rt192k" / "config.yaml"
EXPORT = ROOT / "tools" / "online" / "export_onnx_online_model.py"
PYTHON = ROOT / ".venv" / "bin" / "python"

ONE_IMPORT_ONNX = ONE_CMDS / "one-import-onnx"
ONE_OPTIMIZE = ONE_CMDS / "one-optimize"
ONE_QUANTIZE = ONE_CMDS / "one-quantize"
ONE_CREATE_QUANT_DATASET = ONE_CMDS / "one-create-quant-dataset"
CIRCLE_INSPECT = ONE_BUILD_COMPILER / "circle-inspect" / "circle-inspect"


def one_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = f"{ONE_CMDS}:{env.get('PATH', '')}"
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
        "luci/export",
        "luci/pass",
        "luci/service",
        "luci/lang",
        "luci/env",
        "luci/profile",
        "luci/plan",
        "luci/log",
        "luci/logex",
        "luci-compute",
        "luci-interpreter/src",
        "dio-hdf5",
    ]
    lib_dirs = [str((ONE_BUILD_COMPILER / rel).resolve()) for rel in rels if (ONE_BUILD_COMPILER / rel).exists()]
    if lib_dirs:
        env["LD_LIBRARY_PATH"] = ":".join(lib_dirs + [env.get("LD_LIBRARY_PATH", "")]).strip(":")
    return env


def run(cmd: list[str], *, cwd: Path = ROOT) -> str:
    print("+ " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=cwd, env=one_env(), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(proc.stdout, end="", flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with rc={proc.returncode}: {' '.join(cmd)}")
    return proc.stdout


def count_ops(circle: Path) -> dict[str, int]:
    out = run([str(CIRCLE_INSPECT), "--operators", str(circle)])
    counts: dict[str, int] = {}
    for line in out.splitlines():
        op = line.strip()
        if op and not op.startswith("+ "):
            counts[op] = counts.get(op, 0) + 1
    return dict(sorted(counts.items()))


def inspect_shapes(circle: Path) -> str:
    return run([str(CIRCLE_INSPECT), "--tensor_shape", str(circle)])


def nhwc(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().permute(0, 2, 3, 1).contiguous().numpy().astype(np.float32)


def synthetic_stream_frame(record_idx: int, n_freq: int, *, device: torch.device) -> torch.Tensor:
    freq = torch.linspace(0.0, 1.0, n_freq, device=device)
    envelope = 1.0 / (1.0 + 7.5 * freq)
    phase = 0.071 * float(record_idx + 1)
    real = 0.35 * envelope * torch.sin((record_idx % 11 + 1) * torch.pi * freq + phase)
    imag = 0.35 * envelope * torch.cos((record_idx % 7 + 1) * torch.pi * freq + 0.5 * phase)
    noise_gen = torch.Generator(device=device)
    noise_gen.manual_seed(20260715 + record_idx)
    noise = 0.04 * torch.randn((2, n_freq), generator=noise_gen, device=device)
    frame = torch.stack((real, imag), dim=0) + noise
    return frame.unsqueeze(0).unsqueeze(2).to(torch.float32)


def build_nhwc_calibration(records: int, out_dir: Path) -> dict[str, object]:
    sys.path.insert(0, str(ROOT))
    from tools.online.export_onnx_online_model import build_model_system_from_recipe_config

    model_system = build_model_system_from_recipe_config(CONFIG).eval()
    core = model_system.model.core.eval()
    core.masking = False
    device = torch.device("cpu")
    state = core.init_stream_state(batch_size=1, device=device, dtype=torch.float32)
    n_freq = int(core.n_freq)

    npy_dir = out_dir / "calib_npy"
    npy_dir.mkdir(parents=True, exist_ok=True)
    list_path = out_dir / "calib_nhwc_list.txt"
    lines: list[str] = []
    x_absmax = 0.0
    state_absmax = 0.0

    with torch.no_grad():
        for record_idx in range(records):
            x = synthetic_stream_frame(record_idx, n_freq, device=device)
            flat_inputs = [x, *state]
            paths = []
            for input_idx, tensor in enumerate(flat_inputs):
                arr = nhwc(tensor)
                path = npy_dir / f"calib_sample{record_idx:03d}_input{input_idx:02d}.npy"
                np.save(path, arr)
                paths.append(str(path.resolve()))
                if input_idx == 0:
                    x_absmax = max(x_absmax, float(np.max(np.abs(arr))))
                else:
                    state_absmax = max(state_absmax, float(np.max(np.abs(arr))))
            lines.append(" ".join(paths))
            _y, state = core.forward_stream(x, state)

    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "records": records,
        "input_names": ["x", *[f"state_{idx}" for idx in range(len(state))]],
        "nchw_shapes": [[1, 2, 1, n_freq], *[list(t.shape) for t in state]],
        "nhwc_shapes": [[1, 1, n_freq, 2], *[list(nhwc(t).shape) for t in state]],
        "x_absmax": x_absmax,
        "state_absmax": state_absmax,
        "list_path": str(list_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build NHWC ABI raw-mask SFC-small Circle/calibration/quant artifacts.")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--records", type=int, default=64)
    parser.add_argument("--skip-quantize", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx = out_dir / "stream_rawmask.onnx"
    circle = out_dir / "stream_rawmask.circle"
    opt_circle = out_dir / "stream_rawmask.nhwc.opt.circle"
    q_circle = out_dir / "stream_rawmask.nhwc.opt.q.circle"
    calib_h5 = out_dir / "calib_nhwc.h5"
    state_json = out_dir / "stream_rawmask_state_nchw_export.json"

    run(
        [
            str(PYTHON),
            str(EXPORT),
            str(CONFIG),
            "--out",
            str(onnx),
            "--n-chan",
            "1",
            "--frames",
            "1",
            "--opset",
            "11",
            "--streaming",
            "--disable-masking",
            "--check",
            "--state-meta-out",
            str(state_json),
        ]
    )
    run([str(ONE_IMPORT_ONNX), "-i", str(onnx), "-o", str(circle), "--dynamic_batch_to_single_batch"])
    run(
        [
            str(ONE_OPTIMIZE),
            "-i",
            str(circle),
            "-o",
            str(opt_circle),
            "--convert_nchw_to_nhwc",
            "--nchw_to_nhwc_input_shape",
            "--nchw_to_nhwc_output_shape",
            "--fuse_batchnorm_with_conv",
            "--fuse_activation_function",
            "--remove_duplicate_const",
            "--remove_unnecessary_add",
            "--remove_unnecessary_slice",
            "--remove_unnecessary_strided_slice",
            "--remove_unnecessary_reshape",
            "--remove_unnecessary_transpose",
            "--remove_redundant_reshape",
            "--remove_redundant_transpose",
            "--forward_transpose_op",
            "--resolve_customop_matmul",
            "--resolve_customop_batchmatmul",
        ]
    )

    calib_meta = build_nhwc_calibration(args.records, out_dir)
    run(
        [
            str(ONE_CREATE_QUANT_DATASET),
            "-i",
            "numpy",
            "-l",
            str(out_dir / "calib_nhwc_list.txt"),
            "-p",
            str(calib_h5),
        ]
    )

    quantized = False
    if not args.skip_quantize:
        run(
            [
                str(ONE_QUANTIZE),
                "-i",
                str(opt_circle),
                "-d",
                str(calib_h5),
                "-f",
                "h5",
                "-o",
                str(q_circle),
                "--quantized_dtype",
                "uint8",
                "--granularity",
                "channel",
                "--input_type",
                "uint8",
                "--output_type",
                "uint8",
                "--mode",
                "percentile",
                "--min_percentile",
                "0.1",
                "--max_percentile",
                "99.9",
            ]
        )
        quantized = True

    opt_counts = count_ops(opt_circle)
    q_counts = count_ops(q_circle) if quantized else {}
    shape_text = inspect_shapes(opt_circle)
    (out_dir / "stream_rawmask.nhwc.opt.tensor_shape.txt").write_text(shape_text, encoding="utf-8")
    (out_dir / "stream_rawmask.nhwc.opt.operators.json").write_text(
        json.dumps(opt_counts, indent=2), encoding="utf-8"
    )
    if quantized:
        (out_dir / "stream_rawmask.nhwc.opt.q.operators.json").write_text(
            json.dumps(q_counts, indent=2), encoding="utf-8"
        )

    manifest = {
        "recipe": str(CONFIG),
        "mode": "raw-mask streaming NHWC ABI",
        "onnx": str(onnx),
        "circle": str(circle),
        "optimized_circle": str(opt_circle),
        "quantized_circle": str(q_circle) if quantized else None,
        "calibration_h5": str(calib_h5),
        "calibration": calib_meta,
        "optimized_operator_counts": opt_counts,
        "quantized_operator_counts": q_counts,
        "notes": [
            "ONNX export remains NCHW.",
            "Circle ABI is changed by --nchw_to_nhwc_input_shape and --nchw_to_nhwc_output_shape.",
            "Runtime and calibration must feed NHWC tensors: x [1,1,1025,2], states [1,1,64,160].",
            "Raw mask output is NHWC [1,1,1025,6].",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
