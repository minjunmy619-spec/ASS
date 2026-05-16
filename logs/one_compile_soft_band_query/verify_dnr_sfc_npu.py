from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path


ROOT = Path("/home/cmj/works/ASS")
ONE_CMDS = Path("/home/cmj/works/ONE/build/compiler/one-cmds")
OUT_ROOT = ROOT / "logs" / "dnr_sfc_npu_verify"
OUT_ROOT.mkdir(parents=True, exist_ok=True)


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


def infer_n_chan(cfg: Path) -> int:
    text = cfg.read_text(encoding="utf-8")
    m = re.search(r"(?m)^\s*n_chan:\s*(\d+)\s*$", text)
    if m:
        return int(m.group(1))
    return 2


def export_once(
    py: Path,
    export_script: Path,
    cfg: Path,
    onnx_path: Path,
    manifest_path: Path,
    n_chan: int,
    env: dict[str, str],
) -> tuple[int, str]:
    export_cmd = (
        f'"{py}" "{export_script}" "{cfg}" '
        f'--out "{onnx_path}" --n-chan {n_chan} --frames 1 --opset 14 '
        f'--disable-masking --deploy-manifest-out "{manifest_path}"'
    )
    return sh(export_cmd, env=env)


def main() -> int:
    cfgs = sorted(
        p
        for p in (ROOT / "recipes" / "dnr" / "models").glob("*/config.yaml")
        if "sfc" in p.parent.name.lower()
    )

    if not cfgs:
        print("No SFC configs found.")
        return 1

    env = os.environ.copy()
    env["PATH"] = f"{ONE_CMDS}:{env.get('PATH', '')}"
    lib_dirs = load_lib_dirs()
    if lib_dirs:
        env["LD_LIBRARY_PATH"] = ":".join(lib_dirs + [env.get("LD_LIBRARY_PATH", "")]).strip(":")
    env["NUMBA_DISABLE_JIT"] = "1"
    env["NUMBA_CACHE_DIR"] = str((ROOT / "logs" / ".numba_cache").resolve())
    Path(env["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

    py = ROOT / ".venv" / "bin" / "python"
    export_script = ROOT / "tools" / "online" / "export_onnx_online_model.py"

    results = []

    for cfg in cfgs:
        model_name = cfg.parent.name
        out_dir = OUT_ROOT / model_name
        out_dir.mkdir(parents=True, exist_ok=True)
        n_chan = infer_n_chan(cfg)

        onnx_path = out_dir / "model.onnx"
        manifest_path = out_dir / "manifest.json"
        circle_path = out_dir / "model.circle"
        opt_path = out_dir / "model.opt.circle"
        q_path = out_dir / "model.q.circle"
        code_prefix = out_dir / "model"
        calib = ROOT / "logs" / "one_compile_soft_band_query" / "calib_stream.h5"
        log_path = out_dir / "run.log"

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
                "",
                "[one-quantize]",
                f"input_path={opt_path}",
                f"output_path={q_path}",
                f"input_data={calib}",
                "input_data_format=h5",
                "quantized_dtype=uint8",
                "granularity=channel",
                "input_type=uint8",
                "output_type=uint8",
            ]
        )
        onecc_cfg = out_dir / "config.cfg"
        onecc_cfg.write_text(cfg_text, encoding="utf-8")

        full_log = []
        rc_e, out_e = export_once(py, export_script, cfg, onnx_path, manifest_path, n_chan, env)
        used_n_chan = n_chan
        mismatch = "expected input[1, 4" in out_e and "to have 2 channels" in out_e
        if rc_e != 0 and mismatch and n_chan != 1:
            rc_e, out_e_retry = export_once(py, export_script, cfg, onnx_path, manifest_path, 1, env)
            out_e += "\n\n=== EXPORT RETRY (--n-chan 1) ===\n" + out_e_retry
            used_n_chan = 1
        full_log.append("=== EXPORT ===\n")
        full_log.append(out_e)

        status = "PASS"
        stage = ""
        rc_c = -1
        if rc_e != 0:
            status = "FAIL"
            stage = "export"
        else:
            rc_c, out_c = sh(f'cd "{out_dir}" && onecc -C "{onecc_cfg}"', env=env)
            full_log.append("\n=== ONECC ===\n")
            full_log.append(out_c)
            if rc_c != 0:
                status = "FAIL"
                stage = first_error_stage(out_c)
            else:
                if not circle_path.exists():
                    status = "FAIL"
                    stage = "import"
                elif not opt_path.exists():
                    status = "FAIL"
                    stage = "optimize"
                elif not q_path.exists():
                    status = "FAIL"
                    stage = "quantize"

        log_path.write_text("".join(full_log), encoding="utf-8")
        results.append(
            {
                "model": model_name,
                "config": str(cfg),
                "status": status,
                "fail_stage": stage,
                "used_n_chan": used_n_chan,
                "export_rc": rc_e,
                "onecc_rc": rc_c,
                "log": str(log_path),
            }
        )
        print(f"[{status}] {model_name}" + (f" ({stage})" if stage else ""))

    json_path = OUT_ROOT / "summary.json"
    md_path = OUT_ROOT / "summary.md"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    ok = sum(1 for r in results if r["status"] == "PASS")
    fail = len(results) - ok
    lines = [
        "# DNR SFC NPU Verification Summary",
        "",
        f"- Total: {len(results)}",
        f"- PASS: {ok}",
        f"- FAIL: {fail}",
        "",
        "| Model | Status | Fail Stage |",
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
