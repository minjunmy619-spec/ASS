#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any

ROOT = Path("/home/cmj/works/ASS")
ONE_CMDS = Path(os.environ.get("ONE_CMDS", "/home/cmj/works/ONE/build/compiler/one-cmds"))
ONE_BUILD_COMPILER = Path(os.environ.get("ONE_BUILD_COMPILER", str(ONE_CMDS.parent)))
DEFAULT_ONE_QUANTIZE = ONE_CMDS / "one-quantize"
DEFAULT_CIRCLE_INSPECT = ONE_CMDS / "circle-inspect"
DEFAULT_OUT_ROOT = ROOT / "logs" / "one_stock_quant_sweep"


def load_lib_dirs() -> list[str]:
    base = ONE_BUILD_COMPILER
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
    return [str((base / rel).resolve()) for rel in rels if (base / rel).exists()]


def build_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = f"{ONE_CMDS}:{env.get('PATH', '')}"
    lib_dirs = load_lib_dirs()
    if lib_dirs:
        env["LD_LIBRARY_PATH"] = ":".join(lib_dirs + [env.get("LD_LIBRARY_PATH", "")]).strip(":")
    return env


def parse_csv_floats(value: str) -> list[float]:
    values: list[float] = []
    for raw in value.split(","):
        raw = raw.strip()
        if raw:
            values.append(float(raw))
    if not values:
        raise argparse.ArgumentTypeError("at least one value is required")
    return values


def parse_csv_ints(value: str) -> list[int]:
    values: list[int] = []
    for raw in value.split(","):
        raw = raw.strip()
        if raw:
            values.append(int(raw))
    if not values:
        raise argparse.ArgumentTypeError("at least one value is required")
    return values


def parse_modes(value: str) -> set[str]:
    modes = {item.strip() for item in value.split(",") if item.strip()}
    supported = {"percentile", "moving_average", "ampq"}
    unknown = modes - supported
    if unknown:
        raise argparse.ArgumentTypeError(f"unsupported mode(s): {', '.join(sorted(unknown))}")
    if not modes:
        raise argparse.ArgumentTypeError("at least one mode is required")
    return modes


def infer_input_data_format(path: Path) -> str:
    if path.is_dir():
        return "directory"

    suffix = path.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        return "h5"
    if suffix in {".txt", ".lst", ".list", ".filelist"}:
        return "list"

    return "h5"


def validate_input_data_format(parser: argparse.ArgumentParser, path: Path, data_format: str) -> None:
    if data_format in {"h5", "hdf5", "list", "filelist"} and not path.is_file():
        parser.error(f"--calib-data must be a file for input-data-format={data_format}: {path}")
    if data_format in {"dir", "directory"} and not path.is_dir():
        parser.error(f"--calib-data must be a directory for input-data-format={data_format}: {path}")


FLOAT_RE = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"


def parse_mse_by_output(log_text: str) -> dict[str, float]:
    values: dict[str, float] = {}
    pattern = re.compile(rf"^MSE for (.+?) is {FLOAT_RE}\s*$", re.MULTILINE)
    for match in pattern.finditer(log_text):
        values[match.group(1)] = float(match.group(2))
    return values


def summarize_mse(log_text: str) -> dict[str, Any]:
    by_output = parse_mse_by_output(log_text)
    if not by_output:
        return {"mse": None, "mse_primary": None, "mse_mean": None, "mse_by_output": {}}
    values = list(by_output.values())
    return {
        "mse": values[0],
        "mse_primary": values[0],
        "mse_mean": sum(values) / len(values),
        "mse_by_output": by_output,
    }


def parse_circle_ops(output: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for raw_line in output.splitlines():
        op = raw_line.strip()
        if not op or " " in op or op.startswith("["):
            continue
        counts[op] = counts.get(op, 0) + 1
    return dict(sorted(counts.items()))


def run_command(
    cmd: list[str],
    env: dict[str, str],
    timeout: int,
    *,
    stream_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    if not stream_output:
        return subprocess.run(cmd, text=True, capture_output=True, env=env, timeout=timeout, check=False)

    proc = subprocess.Popen(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        bufsize=1,
    )
    output: list[str] = []
    start = time.monotonic()
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            output.append(line)
            print(line, end="", flush=True)
            if time.monotonic() - start > timeout:
                proc.kill()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                raise subprocess.TimeoutExpired(cmd, timeout, output="".join(output))
        returncode = proc.wait()
    except BaseException:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        raise
    return subprocess.CompletedProcess(cmd, returncode, "".join(output), "")


def _numeric_h5_keys(keys) -> list[str]:
    return sorted(keys, key=lambda value: int(value) if str(value).isdigit() else str(value))


def write_limited_h5(input_path: Path, output_path: Path, record_limit: int) -> Path:
    if record_limit <= 0:
        raise ValueError(f"record_limit must be positive, got {record_limit}")
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise RuntimeError("--*-record-limit for H5 input requires h5py") from exc

    with h5py.File(input_path, "r") as src, h5py.File(output_path, "w") as dst:
        if "value" not in src:
            raise ValueError(f"Unsupported H5 calibration layout without /value group: {input_path}")
        src_value = src["value"]
        dst_value = dst.create_group("value")
        for key in _numeric_h5_keys(src_value.keys())[:record_limit]:
            src.copy(src_value[key], dst_value, name=key)
        for key, value in src.attrs.items():
            dst.attrs[key] = value
    return output_path


def write_limited_list(input_path: Path, output_path: Path, record_limit: int) -> Path:
    lines = input_path.read_text(encoding="utf-8").splitlines()
    output_path.write_text("\n".join(lines[:record_limit]) + "\n", encoding="utf-8")
    return output_path


def maybe_limit_input_data(
    *,
    path: Path,
    data_format: str,
    record_limit: int | None,
    output_dir: Path,
    stem: str,
) -> Path:
    if record_limit is None:
        return path
    if data_format in {"h5", "hdf5"}:
        return write_limited_h5(path, output_dir / f"{stem}_first{record_limit}.h5", record_limit)
    if data_format in {"list", "filelist"}:
        return write_limited_list(path, output_dir / f"{stem}_first{record_limit}.txt", record_limit)
    raise ValueError(f"--{stem.replace('_', '-')}-record-limit is not supported for input-data-format={data_format}")


def add_common_quant_args(cmd: list[str], args: argparse.Namespace, output_path: Path) -> None:
    cmd.extend(
        [
            "--input_path",
            str(args.input_circle),
            "--output_path",
            str(output_path),
            "--input_data",
            str(args.calib_data),
            "--input_data_format",
            args.input_data_format,
            "--quantized_dtype",
            args.quantized_dtype,
            "--granularity",
            args.granularity,
            "--input_type",
            args.input_type,
            "--output_type",
            args.output_type,
        ]
    )
    if args.quant_config is not None:
        cmd.extend(["--quant_config", str(args.quant_config)])
    if args.save_intermediate:
        cmd.append("--save_intermediate")
    if args.tf_style_maxpool:
        cmd.append("--TF-style_maxpool")
    if args.evaluate_result:
        cmd.append("--evaluate_result")
    if args.test_data is not None:
        cmd.extend(["--test_data", str(args.test_data)])
    if args.print_mse:
        cmd.append("--print_mse")
    for extra in args.extra_one_quantize_arg:
        cmd.append(extra)


def build_run_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    if "percentile" in args.modes:
        for min_percentile in args.min_percentiles:
            for max_percentile in args.max_percentiles:
                if min_percentile < max_percentile:
                    specs.append(
                        {
                            "kind": "percentile",
                            "mode": "percentile",
                            "min_percentile": min_percentile,
                            "max_percentile": max_percentile,
                        }
                    )

    if "moving_average" in args.modes:
        for batch in args.moving_avg_batches:
            for const in args.moving_avg_consts:
                specs.append(
                    {
                        "kind": "moving_average",
                        "mode": "moving_average",
                        "moving_avg_batch": batch,
                        "moving_avg_const": const,
                    }
                )

    if "ampq" in args.modes:
        for ratio in args.ampq_qerror_ratios:
            for bisection_type in args.ampq_bisection_types:
                specs.append(
                    {
                        "kind": "ampq",
                        "ampq_qerror_ratio": ratio,
                        "bisection_type": bisection_type,
                    }
                )

    return specs


def spec_tag(index: int, spec: dict[str, Any]) -> str:
    if spec["kind"] == "percentile":
        body = f"percentile_p{spec['min_percentile']:g}_{spec['max_percentile']:g}"
    elif spec["kind"] == "moving_average":
        body = f"moving_avg_b{spec['moving_avg_batch']}_c{spec['moving_avg_const']:g}"
    else:
        body = f"ampq_r{spec['ampq_qerror_ratio']:g}_{spec['bisection_type']}"
    return f"{index:02d}_{body}".replace(".", "p")


def build_command(args: argparse.Namespace, spec: dict[str, Any], output_path: Path) -> list[str]:
    cmd = [str(args.one_quantize)]
    add_common_quant_args(cmd, args, output_path)
    if spec["kind"] == "percentile":
        cmd.extend(
            [
                "--mode",
                "percentile",
                "--min_percentile",
                str(spec["min_percentile"]),
                "--max_percentile",
                str(spec["max_percentile"]),
            ]
        )
    elif spec["kind"] == "moving_average":
        cmd.extend(
            [
                "--mode",
                "moving_average",
                "--moving_avg_batch",
                str(spec["moving_avg_batch"]),
                "--moving_avg_const",
                str(spec["moving_avg_const"]),
            ]
        )
    else:
        cmd.extend(
            [
                "--ampq",
                "--ampq_algorithm",
                args.ampq_algorithm,
                "--ampq_qerror_ratio",
                str(spec["ampq_qerror_ratio"]),
                "--bisection_type",
                spec["bisection_type"],
            ]
        )
    return cmd


def inspect_circle(circle_path: Path, args: argparse.Namespace, env: dict[str, str]) -> dict[str, int]:
    if args.circle_inspect is None or not args.circle_inspect.exists() or not circle_path.exists():
        return {}
    proc = run_command([str(args.circle_inspect), "--operators", str(circle_path)], env, args.timeout)
    if proc.returncode != 0:
        return {}
    return parse_circle_ops(proc.stdout or "")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run stock ONE quantization sweeps and rank candidates with external metrics."
    )
    parser.add_argument("--input-circle", type=Path, required=True, help="Optimized fp32 .circle model.")
    parser.add_argument("--calib-data", type=Path, required=True, help="Representative calibration data.")
    parser.add_argument(
        "--calib-record-limit",
        type=int,
        help="Use only the first N calibration records by writing a limited H5/list in the output directory.",
    )
    parser.add_argument(
        "--input-data-format",
        default="auto",
        help="record-minmax data format: auto, h5/hdf5, list/filelist, or dir/directory.",
    )
    parser.add_argument("--one-quantize", type=Path, default=DEFAULT_ONE_QUANTIZE)
    parser.add_argument("--circle-inspect", type=Path, default=DEFAULT_CIRCLE_INSPECT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--modes", type=parse_modes, default=parse_modes("percentile,ampq"))
    parser.add_argument("--min-percentiles", type=parse_csv_floats, default=parse_csv_floats("0.1,0.5,1.0"))
    parser.add_argument("--max-percentiles", type=parse_csv_floats, default=parse_csv_floats("99.0,99.5,99.9"))
    parser.add_argument("--moving-avg-batches", type=parse_csv_ints, default=parse_csv_ints("8,16,32"))
    parser.add_argument("--moving-avg-consts", type=parse_csv_floats, default=parse_csv_floats("0.05,0.1,0.2"))
    parser.add_argument("--ampq-qerror-ratios", type=parse_csv_floats, default=parse_csv_floats("0.01,0.03,0.05"))
    parser.add_argument("--ampq-bisection-types", default="auto,i16_front,i16_back")
    parser.add_argument("--ampq-algorithm", default="bisection")
    parser.add_argument("--quantized-dtype", default="uint8")
    parser.add_argument("--granularity", default="channel")
    parser.add_argument("--input-type", default="uint8")
    parser.add_argument("--output-type", default="uint8")
    parser.add_argument("--quant-config", type=Path)
    parser.add_argument("--test-data", type=Path)
    parser.add_argument(
        "--test-record-limit",
        type=int,
        help="Use only the first N test records by writing a limited H5/list in the output directory.",
    )
    parser.add_argument("--evaluate-result", action="store_true")
    parser.add_argument("--print-mse", action="store_true")
    parser.add_argument("--save-intermediate", action="store_true")
    parser.add_argument(
        "--tf-style-maxpool",
        "--TF-style-maxpool",
        dest="tf_style_maxpool",
        action="store_true",
    )
    parser.add_argument("--extra-one-quantize-arg", action="append", default=[])
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--stream-output", action="store_true", help="Stream one-quantize output while also logging it.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.ampq_bisection_types = [
        item.strip() for item in args.ampq_bisection_types.split(",") if item.strip()
    ]
    if not args.ampq_bisection_types:
        parser.error("--ampq-bisection-types must contain at least one value")
    if not args.one_quantize.exists():
        parser.error(f"--one-quantize not found: {args.one_quantize}")
    if not args.input_circle.exists():
        parser.error(f"--input-circle not found: {args.input_circle}")
    if not args.calib_data.exists():
        parser.error(f"--calib-data not found: {args.calib_data}")
    if args.quant_config is not None and not args.quant_config.exists():
        parser.error(f"--quant-config not found: {args.quant_config}")
    if args.input_data_format == "auto":
        args.input_data_format = infer_input_data_format(args.calib_data)
    if args.input_data_format not in {"h5", "hdf5", "list", "filelist", "dir", "directory"}:
        parser.error("--input-data-format must be auto, h5/hdf5, list/filelist, or dir/directory")
    validate_input_data_format(parser, args.calib_data, args.input_data_format)
    if args.evaluate_result and args.test_data is None:
        parser.error("--evaluate-result requires --test-data")
    if args.test_data is not None and not args.test_data.exists():
        parser.error(f"--test-data not found: {args.test_data}")
    if args.quant_config is not None and "ampq" in args.modes:
        parser.error(
            "--quant-config cannot be combined with --modes ampq because stock one-quantize "
            "does not pass quant_config into circle-mpqsolver. Use percentile/moving_average "
            "for a fixed qconfig, or run AMPQ separately without --quant-config."
        )

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_root = args.output_dir or (DEFAULT_OUT_ROOT / stamp)
    out_root.mkdir(parents=True, exist_ok=True)

    try:
        args.calib_data = maybe_limit_input_data(
            path=args.calib_data,
            data_format=args.input_data_format,
            record_limit=args.calib_record_limit,
            output_dir=out_root,
            stem="calib",
        )
        if args.test_data is not None:
            args.test_data = maybe_limit_input_data(
                path=args.test_data,
                data_format=args.input_data_format,
                record_limit=args.test_record_limit,
                output_dir=out_root,
                stem="test",
            )
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    env = build_env()
    results: list[dict[str, Any]] = []
    for index, spec in enumerate(build_run_specs(args)):
        tag = spec_tag(index, spec)
        out_circle = out_root / f"{tag}.q.circle"
        log_path = out_root / f"{tag}.log"
        cmd = build_command(args, spec, out_circle)
        command_text = shlex.join(cmd)
        print(f"[run] {command_text}")

        result = {
            "tag": tag,
            "spec": spec,
            "output_circle": str(out_circle),
            "log": str(log_path),
            "command": command_text,
        }
        if args.dry_run:
            log_path.write_text(command_text + "\n", encoding="utf-8")
            result.update({"returncode": 0, "seconds": 0.0, "dry_run": True})
        else:
            start = time.monotonic()
            try:
                proc = run_command(cmd, env, args.timeout, stream_output=args.stream_output)
                seconds = time.monotonic() - start
                blob = (proc.stdout or "") + "\n" + (proc.stderr or "")
                log_path.write_text(command_text + "\n\n" + blob, encoding="utf-8")
                result.update(
                    {
                        "returncode": proc.returncode,
                        "seconds": seconds,
                        **summarize_mse(blob),
                    }
                )
                if proc.returncode == 0:
                    result["circle_operator_counts"] = inspect_circle(out_circle, args, env)
                print(
                    f"[result] {tag} rc={proc.returncode} seconds={seconds:.2f} "
                    f"mse_primary={result.get('mse_primary')}"
                )
            except subprocess.TimeoutExpired:
                result.update({"returncode": -1, "seconds": args.timeout, "timeout": True})
                log_path.write_text(command_text + "\n\nTimed out.\n", encoding="utf-8")

        results.append(result)
        with (out_root / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    failures = [item for item in results if item["returncode"] != 0]
    print(f"[done] wrote {out_root / 'summary.json'}")
    if failures:
        print(f"[done] {len(failures)} run(s) failed; inspect logs before comparing metrics.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
