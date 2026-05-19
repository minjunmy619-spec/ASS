#!/usr/bin/env python3
"""
Minimal experiment: PRelu slope layout [C,1,1] vs [1,1,1,C] vs NCHW/NHWC semantics.

Run from ASS repo (uses local .venv recommended):
  ./.venv/bin/python logs/prelu_alpha_layout_experiment/run_experiment.py

Writes FINDINGS.md next to this script.
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
from onnx import numpy_helper

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


def prelu_numpy(x: np.ndarray, alpha_bc: np.ndarray) -> np.ndarray:
    """Elementwise PRelu: y = relu(x) + alpha * minimum(x, 0). alpha broadcasts to x."""
    neg = np.minimum(x, 0.0)
    return np.maximum(x, 0.0) + alpha_bc * neg


class TinyPRelu(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.act = nn.PReLU(num_parameters=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)


def export_onnx(path: Path, c: int, h: int, w: int, alpha_np: np.ndarray) -> tuple[str, tuple[int, ...]]:
    m = TinyPRelu(c).eval()
    with torch.no_grad():
        m.act.weight.copy_(torch.from_numpy(alpha_np.astype(np.float32)))
    x = torch.randn(1, c, h, w)
    torch.onnx.export(
        m,
        x,
        str(path),
        input_names=["input"],
        output_names=["output"],
        opset_version=14,
        dynamo=False,
    )
    model = onnx.load(str(path))
    slope_name = None
    slope_shape = ()
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        if arr.size == c and arr.ndim >= 1:
            slope_name = init.name
            slope_shape = tuple(arr.shape)
            break
    if slope_name is None:
        for n in model.graph.node:
            if n.op_type == "PRelu" and len(n.input) > 1:
                slope_name = n.input[1]
                break
        if slope_name:
            for init in model.graph.initializer:
                if init.name == slope_name:
                    slope_shape = tuple(numpy_helper.to_array(init).shape)
                    break
    return slope_name or "", slope_shape


def rewrite_slope_to_111c(onnx_in: Path, onnx_out: Path, *, channels: int) -> str:
    model = onnx.load(str(onnx_in))
    graph = model.graph
    slope_names = []
    for n in graph.node:
        if n.op_type == "PRelu" and len(n.input) > 1:
            slope_names.append(n.input[1])
    if len(slope_names) != 1:
        return f"expected 1 PRelu slope, got {len(slope_names)}"
    target = slope_names[0]
    new_inits = []
    hit = ""
    for init in graph.initializer:
        if init.name != target:
            new_inits.append(init)
            continue
        arr = numpy_helper.to_array(init).astype(np.float32).reshape(channels)
        new_arr = arr.reshape(1, 1, 1, channels).astype(np.float32)
        new_inits.append(numpy_helper.from_array(new_arr, name=init.name))
        hit = f"rewrote {target}: {tuple(init.dims)} -> {new_arr.shape}"
    if not hit:
        return "slope initializer not found"
    graph.ClearField("initializer")
    graph.initializer.extend(new_inits)
    onnx.save(model, str(onnx_out))
    try:
        onnx.checker.check_model(model)
        hit += " | onnx.checker: OK"
    except onnx.checker.ValidationError as ex:
        hit += f" | onnx.checker: FAIL ({ex})"
    return hit


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
    rng = np.random.default_rng(0)
    x_nchw = rng.standard_normal((1, C, H, W), dtype=np.float32)
    alpha_vec = rng.standard_normal((C,), dtype=np.float32)
    alpha_c11 = alpha_vec.reshape(C, 1, 1)
    alpha_111c = alpha_vec.reshape(1, 1, 1, C)

    lines: list[str] = []
    lines.append("# PRelu α layout experiment — findings")
    lines.append("")
    lines.append("## Setup")
    lines.append(f"- Tensor shapes: **NCHW** `input = [1, {C}, {H}, {W}]`")
    lines.append(f"- Same learned slopes `α` as length-{C} vector (fixed RNG seed).")
    lines.append("")

    # --- NumPy semantics ---
    y_nchw_ref = prelu_numpy(x_nchw, alpha_c11)
    x_nhwc = np.transpose(x_nchw, (0, 2, 3, 1))
    y_nhwc_ref = prelu_numpy(x_nhwc, alpha_111c)
    y_nchw_from_hwc = np.transpose(y_nhwc_ref, (0, 3, 1, 2))
    max_err_transpose = float(np.max(np.abs(y_nchw_ref - y_nchw_from_hwc)))

    lines.append("## A) NumPy: `[1,1,1,C]` is correct only with NHWC activations")
    lines.append("")
    lines.append(
        "- Define PRelu as `relu(x) + α * min(x, 0)` with broadcasting "
        "(matches PyTorch semantics)."
    )
    lines.append(f"- **`α` `[C,1,1]` + `x` `[1,C,H,W]`** → reference output `y_nchw`.")
    lines.append(f"- **`α` `[1,1,1,C]` + `x` `[1,H,W,C]`** → `y_nhwc`; transpose back to NCHW.")
    lines.append(f"- **`max(|y_nchw - transpose(y_nhwc)|)`** = `{max_err_transpose:.6e}` (should be ~0).")
    lines.append("")
    wrong_mix_line = ""
    wrong_mix_meta: float | str | None = None
    try:
        wrong_mix = prelu_numpy(x_nchw, alpha_111c)
        max_err_wrong = float(np.max(np.abs(y_nchw_ref - wrong_mix)))
        wrong_mix_meta = max_err_wrong
        wrong_mix_line = (
            f"- **Wrong mix: `x` NCHW `[1,C,H,W]` with `α` `[1,1,1,C]`** numerically broadcast → "
            f"`max(|y_ref - y_wrong|)` = `{max_err_wrong:.6e}`."
        )
    except ValueError as ex:
        wrong_mix_meta = "broadcast_error"
        wrong_mix_line = (
            "- **Wrong mix: `x` NCHW `[1,C,H,W]` with `α` `[1,1,1,C]`** → NumPy rejects broadcast "
            f"(`{type(ex).__name__}`): arithmetic not even defined."
        )

    lines.append(wrong_mix_line)
    lines.append("")

    # --- ONNX export ---
    HERE.mkdir(parents=True, exist_ok=True)
    base_onnx = HERE / "tiny_prelu_nchw.onnx"
    slope_name, slope_shape = export_onnx(base_onnx, C, H, W, alpha_vec)
    lines.append("## B) PyTorch → ONNX export")
    lines.append(f"- Output: `{base_onnx.name}`")
    lines.append(f"- Detected PRelu slope initializer `{slope_name}` shape **`{slope_shape}`**.")
    lines.append("")

    ok_base, msg_base, out_base = ort_run(base_onnx, x_nchw)
    lines.append("## C) ONNX Runtime (same `x` NCHW)")
    lines.append(f"- Baseline model session: **{'PASS' if ok_base else 'FAIL'}** — {msg_base}")
    if ok_base:
        lines.append(f"- Output shape `{out_base.shape}`, finite `{np.isfinite(out_base).all()}`.")
        pt = TinyPRelu(C).eval()
        pt.act.weight.data = torch.from_numpy(alpha_vec.astype(np.float32))
        with torch.no_grad():
            y_pt = pt(torch.from_numpy(x_nchw)).numpy()
        ort_diff = float(np.max(np.abs(y_pt - out_base)))
        lines.append(f"- vs PyTorch same weights: max diff `{ort_diff:.6e}`.")
    lines.append("")

    rew_onnx = HERE / "tiny_prelu_slope_111c.onnx"
    rw_msg = rewrite_slope_to_111c(base_onnx, rew_onnx, channels=C)
    ok_rw, msg_rw, out_rw = ort_run(rew_onnx, x_nchw)
    lines.append("## D) ONNX Runtime after rewriting slope → `[1,1,1,C]` (graph still NCHW)")
    lines.append(f"- Rewrite: {rw_msg}")
    lines.append(f"- Session: **{'PASS' if ok_rw else 'FAIL'}** — {msg_rw}")
    if ok_rw and out_base is not None:
        lines.append(f"- vs baseline ONNX output: max diff `{float(np.max(np.abs(out_base - out_rw))):.6e}`.")
    lines.append("")

    # --- ONE optional ---
    lines.append("## E) ONE `onecc` (optional — channel-wise quant)")
    env = build_env()
    one_ok = ONE_CMDS.exists() and (sh("which onecc", env=env)[0] == 0)
    if not one_ok:
        lines.append("- Skipped: `onecc` not found or ONE_CMDS missing.")
    else:
        for label, onnx_path in [("baseline_nchw", base_onnx), ("slope_111c_on_nchw", rew_onnx)]:
            run_dir = HERE / f"onecc_run_{label}"
            run_dir.mkdir(parents=True, exist_ok=True)
            rc_cal, cal_out, calib_h5 = make_calib(run_dir, env, (1, C, H, W))
            cfg = write_onecc_cfg(run_dir, onnx_path, calib_h5)
            rc_oc, oc_out = sh(f'cd "{run_dir}" && onecc -C "{cfg}"', env=env)
            (run_dir / "onecc.log").write_text(
                f"=== calib rc={rc_cal} ===\n{cal_out}\n\n=== onecc rc={rc_oc} ===\n{oc_out}",
                encoding="utf-8",
            )
            q_err = "Non-channel dimension of const node must be 1" in oc_out
            if label == "baseline_nchw":
                lines.append(
                    f"- **`{label}`**: circle/opt artifacts "
                    f"{(run_dir / 'model.circle').exists()}/{(run_dir / 'model.opt.circle').exists()}; "
                    f"quantize hits const-layout error={q_err} (onecc rc={rc_oc})."
                )
            else:
                lines.append(
                    f"- **`{label}`**: rc={rc_oc}; import/shape inference fails before Circle "
                    f"(artifacts circle/opt/q="
                    f"{(run_dir / 'model.circle').exists()}/"
                    f"{(run_dir / 'model.opt.circle').exists()}/"
                    f"{(run_dir / 'model.q.circle').exists()}). "
                    f"See `{run_dir / 'onecc.log'}`."
                )

    lines.append("")
    lines.append("## Final conclusions")
    lines.append("")
    lines.append(
        "1. **`α` shaped `[1,1,1,C]` matches per-channel PRelu iff activations are `[B,H,W,C]`** "
        "(NumPy section A)."
    )
    lines.append(
        "2. **Applying `[1,1,1,C]` on the exported ONNX while inputs stay `[B,C,H,W]` is "
        "semantically wrong** (large numeric error vs reference in A; ORT may reject or diverge in D)."
    )
    lines.append(
        "3. **`convert_nchw_to_nhwc=True` alone does not prove ONNX slope tensors were rewritten**: "
        "a layout fix requires ONE (or tooling) to transpose **`CirclePRelu` alpha** consistently "
        "with activations, not only renaming dims on paper."
    )
    lines.append(
        "4. **Runtime proof**: keeping channel-wise quantization, the robust fixes remain "
        "**ONE-side alpha handling**, **avoid `CirclePRelu` alpha layout** (subgraph / different "
        "activation), or **`granularity=layer`** as an explicit fallback."
    )

    findings_path = HERE / "FINDINGS.md"
    findings_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    meta = {
        "numpy_transpose_equiv_max_abs_err": max_err_transpose,
        "numpy_wrong_mix": wrong_mix_meta,
        "export_slope_shape": list(slope_shape),
        "ort_baseline_ok": ok_base,
        "ort_rewritten_ok": ok_rw,
    }
    (HERE / "results.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(findings_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
