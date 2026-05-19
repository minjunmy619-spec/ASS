#!/usr/bin/env python3
"""
Verify: replacing nn.PReLU with relu(x) + alpha.view(1,-1,1,1) * minimum(x, 0)
fixes ONE channel-wise quantization vs baseline CirclePRelu path.

Run:
  ./.venv/bin/python logs/prelu_alpha_layout_experiment/verify_subgraph_prelu_fix.py

Writes SUBGRAPH_VERIFY.md in this folder.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

ROOT = Path("/home/cmj/works/ASS")
ONE_CMDS = Path("/home/cmj/works/ONE/build/compiler/one-cmds")
HERE = Path(__file__).resolve().parent


def sh(cmd: str, env: dict[str, str] | None = None) -> tuple[int, str]:
    p = subprocess.run(cmd, shell=True, text=True, capture_output=True, env=env)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def load_lib_dirs() -> list[str]:
    cands = [
        Path("/home/cmj/works/ONE/build/compiler"),
        Path("/home/cmj/works/ONE/build/compiler/compiler"),
    ]
    base = next((c for c in cands if c.exists()), None)
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


def build_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = f"{ONE_CMDS}:{env.get('PATH', '')}"
    env["PYTHONPATH"] = f"{ROOT}:{env.get('PYTHONPATH', '')}".strip(":")
    lib_dirs = load_lib_dirs()
    if lib_dirs:
        env["LD_LIBRARY_PATH"] = ":".join(lib_dirs + [env.get("LD_LIBRARY_PATH", "")]).strip(":")
    return env


class TinyPReluBuiltin(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.act = nn.PReLU(num_parameters=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)


class TinyPReluSubgraph(nn.Module):
    """y = relu(x) + alpha.view(1,-1,1,1) * minimum(x, 0) — same math as nn.PReLU."""

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.alpha.view(1, -1, 1, 1).to(dtype=x.dtype)
        neg = torch.minimum(x, torch.zeros_like(x))
        return torch.relu(x) + a * neg


def export_onnx(module: nn.Module, path: Path, c: int, h: int, w: int) -> None:
    module.eval()
    x = torch.randn(1, c, h, w)
    torch.onnx.export(
        module,
        x,
        str(path),
        input_names=["input"],
        output_names=["output"],
        opset_version=14,
        dynamo=False,
    )


def onnx_op_histogram(onnx_path: Path) -> dict[str, int]:
    model = onnx.load(str(onnx_path))
    counts: dict[str, int] = {}
    for n in model.graph.node:
        counts[n.op_type] = counts.get(n.op_type, 0) + 1
    return counts


def ort_run(onnx_path: Path, x: np.ndarray) -> tuple[bool, str, np.ndarray | None]:
    try:
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        out = sess.run(None, {"input": x.astype(np.float32)})[0]
        return True, "OK", out
    except Exception as ex:  # noqa: BLE001
        return False, f"{type(ex).__name__}: {ex}", None


def write_onecc_cfg(out_dir: Path, onnx_path: Path, calib_h5: Path) -> Path:
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
            f"output_path={out_dir / 'model.circle'}",
            "",
            "[one-optimize]",
            f"input_path={out_dir / 'model.circle'}",
            f"output_path={out_dir / 'model.opt.circle'}",
            "replace_non_const_fc_with_batch_matmul=True",
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
        ]
    )
    p = out_dir / "config.cfg"
    p.write_text(cfg_text, encoding="utf-8")
    return p


def make_calib(out_dir: Path, env: dict[str, str], shape: tuple[int, ...]) -> tuple[int, str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    arr = np.random.randn(*shape).astype(np.float32)
    npy = out_dir / "calib_sample000_input00.npy"
    np.save(npy, arr)
    lst = out_dir / "calib_list.txt"
    lst.write_text(str(npy) + "\n", encoding="utf-8")
    calib_h5 = out_dir / "calib.h5"
    rc, out = sh(
        f'one-create-quant-dataset -i numpy -l "{lst}" -p "{calib_h5}"',
        env=env,
    )
    return rc, out, calib_h5


def main() -> int:
    C, H, W = 8, 4, 5
    rng = np.random.default_rng(1)
    alpha = rng.standard_normal((C,), dtype=np.float32)
    x_np = rng.standard_normal((1, C, H, W), dtype=np.float32)

    HERE.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Subgraph `relu + alpha * min(x,0)` vs builtin `nn.PReLU` — ONE verify")
    lines.append("")
    lines.append(f"- Shapes: `x` `[1,{C},{H},{W}]`, same `alpha` vector length `{C}` (RNG seed 1).")
    lines.append("")

    builtin = TinyPReluBuiltin(C).eval()
    subgraph = TinyPReluSubgraph(C).eval()
    with torch.no_grad():
        builtin.act.weight.copy_(torch.from_numpy(alpha))
        subgraph.alpha.copy_(torch.from_numpy(alpha))

    xt = torch.from_numpy(x_np)
    with torch.no_grad():
        y_b = builtin(xt).numpy()
        y_s = subgraph(xt).numpy()
    pt_diff = float(np.max(np.abs(y_b - y_s)))
    lines.append("## PyTorch equivalence (builtin vs subgraph)")
    lines.append(f"- max abs diff = **`{pt_diff:.6e}`** (expect ~0).")
    lines.append("")

    path_sub = HERE / "tiny_prelu_subgraph.onnx"
    export_onnx(subgraph, path_sub, C, H, W)
    ops = onnx_op_histogram(path_sub)
    lines.append("## ONNX ops (subgraph model)")
    lines.append(f"- `{path_sub.name}` op counts: `{ops}`")
    lines.append(f"- Contains **`PRelu`**: **`{'PRelu' in ops}`**.")
    lines.append("")

    ok_ort, msg_ort, y_ort = ort_run(path_sub, x_np)
    lines.append("## ONNX Runtime")
    lines.append(f"- Session: **{'PASS' if ok_ort else 'FAIL'}** — {msg_ort}")
    if ok_ort:
        lines.append(f"- vs PyTorch subgraph max diff **`{float(np.max(np.abs(y_s - y_ort))):.6e}`**.")
    lines.append("")

    lines.append("## ONE `onecc` (same pipeline as verifier: nhwc + channel quant)")
    env = build_env()
    run_dir = HERE / "onecc_run_subgraph_min_relu_mul"
    run_dir.mkdir(parents=True, exist_ok=True)
    rc_cal, cal_out, calib_h5 = make_calib(run_dir, env, (1, C, H, W))
    cfg = write_onecc_cfg(run_dir, path_sub, calib_h5)
    rc_oc, oc_out = sh(f'cd "{run_dir}" && onecc -C "{cfg}"', env=env)
    (run_dir / "onecc.log").write_text(
        f"=== calib rc={rc_cal} ===\n{cal_out}\n\n=== onecc rc={rc_oc} ===\n{oc_out}",
        encoding="utf-8",
    )

    q_const_err = "Non-channel dimension of const node must be 1" in oc_out
    has_q = (run_dir / "model.q.circle").exists()
    lines.append(f"- `one-create-quant-dataset` rc={rc_cal}")
    lines.append(f"- `onecc` rc=`{rc_oc}`")
    lines.append(f"- **`model.q.circle` exists**: **`{has_q}`**")
    lines.append(f"- Log contains **`Non-channel dimension of const node must be 1`**: **`{q_const_err}`**")
    lines.append(f"- Full log: `{run_dir / 'onecc.log'}`")
    lines.append("")

    lines.append("## Conclusion")
    if has_q and rc_oc == 0:
        lines.append(
            "**Subgraph replacement succeeds** end-to-end with channel-wise quantization "
            "for this minimal model (no `CirclePRelu` / `[C,1,1]` alpha quant path)."
        )
    elif has_q:
        lines.append(
            "**Quantized Circle produced** (`model.q.circle` exists) but `onecc` returned non-zero rc; "
            "inspect log."
        )
    else:
        lines.append(
            "**Did not produce `model.q.circle`** — subgraph bypass did not fully fix the pipeline "
            "for this export; inspect `onecc.log` for the failing stage."
        )

    out_md = HERE / "SUBGRAPH_VERIFY.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (HERE / "subgraph_verify.json").write_text(
        json.dumps(
            {
                "pytorch_builtin_vs_subgraph_max_abs": pt_diff,
                "onnx_ops": ops,
                "has_prelu_op": "PRelu" in ops,
                "onecc_rc": rc_oc,
                "model_q_circle": has_q,
                "quant_const_layout_error_in_log": q_const_err,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(out_md.read_text(encoding="utf-8"))
    return 0 if has_q else 1


if __name__ == "__main__":
    sys.exit(main())
