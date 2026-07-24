#!/usr/bin/env python3
"""Search a small mixed-precision qconfig set with stock ONE tools."""

from __future__ import annotations

from typing import Any

import argparse
import csv
import json
from pathlib import Path
import shlex
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_one_stock_quant_sweep import (  # noqa: E402
    DEFAULT_CIRCLE_INSPECT,
    DEFAULT_ONE_QUANTIZE,
    ONE_CMDS,
    build_env,
    parse_circle_ops,
    run_command,
    summarize_mse,
)
from suggest_one_mixed_precision_qconfig import (  # noqa: E402
    OpInfo,
    qconfig_payload,
    read_circle,
    write_qconfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Greedily choose int16 nodes using stock ONE final output MSE.")
    parser.add_argument("--circle", type=Path, required=True)
    parser.add_argument("--calib-data", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument(
        "--candidate-csv",
        type=Path,
        required=True,
        help="nodes.csv from either mixed-precision candidate tool.",
    )
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--one-quantize", type=Path, default=DEFAULT_ONE_QUANTIZE)
    parser.add_argument(
        "--one-create-quant-dataset",
        type=Path,
        default=ONE_CMDS / "one-create-quant-dataset",
    )
    parser.add_argument("--circle-inspect", type=Path, default=DEFAULT_CIRCLE_INSPECT)
    parser.add_argument("--input-data-format", default="auto")
    parser.add_argument("--test-data-format", default="auto")
    parser.add_argument("--max-candidates", type=int, default=24)
    parser.add_argument("--max-int16", type=int, default=8)
    parser.add_argument(
        "--objective",
        choices=("primary", "mean", "output"),
        default="primary",
        help="Output MSE used for search. Primary is the first separation output.",
    )
    parser.add_argument(
        "--objective-output",
        help="Exact output name used when --objective=output.",
    )
    parser.add_argument("--min-mse-improvement", type=float, default=0.0)
    parser.add_argument(
        "--min-relative-improvement",
        type=float,
        default=1e-6,
        help="Reject equal or numerically insignificant MSE changes.",
    )
    parser.add_argument(
        "--latency-weight",
        type=float,
        default=0.0,
        help="Penalty on selected rough compute divided by whole-model rough compute.",
    )
    parser.add_argument(
        "--conversion-weight",
        "--boundary-weight",
        dest="conversion_weight",
        type=float,
        default=0.0,
        help="Penalty on actual extra QUANTIZE/DEQUANTIZE operators; boundary-weight is an alias.",
    )
    parser.add_argument("--min-percentile", type=float, default=1.0)
    parser.add_argument("--max-percentile", type=float, default=99.0)
    parser.add_argument("--quantized-dtype", choices=("uint8",), default="uint8")
    parser.add_argument("--granularity", choices=("layer", "channel"), default="channel")
    parser.add_argument("--input-type", choices=("uint8",), default="uint8")
    parser.add_argument("--output-type", choices=("uint8",), default="uint8")
    parser.add_argument("--default-dtype", choices=("uint8",), default="uint8")
    parser.add_argument("--default-granularity", choices=("layer", "channel"), default="channel")
    parser.add_argument("--target-dtype", choices=("int16",), default="int16")
    parser.add_argument("--target-granularity", choices=("layer", "channel"), default="channel")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--stream-output", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def infer_format(path: Path) -> str:
    if path.is_dir():
        return "directory"
    if path.suffix.lower() in {".txt", ".lst", ".list", ".filelist"}:
        return "list"
    if path.suffix.lower() in {".h5", ".hdf5"}:
        return "h5"
    return "list"


def parse_bool(value: str, *, default: bool = True) -> bool:
    text = value.strip().lower()
    if not text:
        return default
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"invalid boolean value: {value}")


def load_candidates(
    path: Path,
    max_candidates: int,
    ops: list[OpInfo],
) -> tuple[list[dict[str, Any]], list[str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    ops_by_name = {op.name: op for op in ops}
    candidates: list[dict[str, Any]] = []
    skipped: list[str] = []
    seen: set[str] = set()
    for row in rows:
        name = row.get("name", "").strip()
        if not name or name in seen:
            continue
        try:
            eligible = parse_bool(row.get("eligible", "true"))
            score = float(row.get("selection_score", row.get("score", "0")))
        except ValueError as error:
            skipped.append(f"{name}: {error}")
            continue
        if not eligible or score <= 0:
            continue
        op = ops_by_name.get(name)
        if op is None:
            skipped.append(f"{name}: not found in current Circle graph")
            continue
        candidate = dict(row)
        candidate.update(
            {
                "name": name,
                "selection_score": score,
                "rough_ops": float(op.rough_ops),
                "index": int(op.index),
            }
        )
        candidates.append(candidate)
        seen.add(name)
    candidates.sort(key=lambda row: row["selection_score"], reverse=True)
    return candidates[:max_candidates], skipped


def boundary_proxy(selected: set[int], ops: list[OpInfo]) -> int:
    count = 0
    for op in ops:
        if op.index in selected:
            count += sum(1 for neighbor in op.producers | op.consumers if neighbor not in selected)
    return count


def conversion_count(result: dict[str, Any]) -> int | None:
    if result.get("operator_inspect_returncode") != 0:
        return None
    operators = result.get("operators")
    if not isinstance(operators, dict):
        return None
    return int(operators.get("QUANTIZE", 0)) + int(operators.get("DEQUANTIZE", 0))


def quality_value(result: dict[str, Any], objective: str, output_name: str | None) -> float | None:
    if objective == "primary":
        value = result.get("mse_primary")
    elif objective == "mean":
        value = result.get("mse_mean")
    else:
        by_output = result.get("mse_by_output", {})
        value = by_output.get(output_name) if isinstance(by_output, dict) else None
    return float(value) if value is not None else None


def improvement_is_sufficient(
    previous: float,
    current: float,
    min_absolute: float,
    min_relative: float,
) -> bool:
    required = max(min_absolute, abs(previous) * min_relative)
    return previous - current > required


def make_command(args: argparse.Namespace, output: Path, qconfig: Path | None) -> list[str]:
    command = [
        str(args.one_quantize),
        "--input_path",
        str(args.circle),
        "--output_path",
        str(output),
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
        "--mode",
        "percentile",
        "--min_percentile",
        str(args.min_percentile),
        "--max_percentile",
        str(args.max_percentile),
        "--evaluate_result",
        "--test_data",
        str(args.test_data),
        "--print_mse",
    ]
    if qconfig is not None:
        command.extend(["--quant_config", str(qconfig)])
    return command


def package_list(
    path: Path,
    output: Path,
    tool: Path,
    env: dict[str, str],
    timeout: int,
) -> None:
    command = [
        str(tool),
        "--input_data_format",
        "rawdata",
        "--data_list",
        str(path),
        "--output_path",
        str(output),
    ]
    result = run_command(command, env, timeout)
    if result.returncode != 0:
        text = (result.stdout or "") + "\n" + (result.stderr or "")
        raise RuntimeError(f"one-create-quant-dataset failed:\n{text[-3000:]}")


def evaluate(
    args: argparse.Namespace,
    env: dict[str, str],
    out_dir: Path,
    tag: str,
    qconfig: Path | None,
) -> dict[str, Any]:
    output = out_dir / f"{tag}.q.circle"
    log_path = out_dir / f"{tag}.log"
    command = make_command(args, output, qconfig)
    text_command = shlex.join(command)
    print(f"[run] {text_command}")
    result: dict[str, Any] = {
        "tag": tag,
        "command": text_command,
        "qconfig": str(qconfig) if qconfig else None,
    }
    if args.dry_run:
        log_path.write_text(text_command + "\n", encoding="utf-8")
        result.update({"returncode": 0, "dry_run": True})
        return result
    started = time.monotonic()
    process = run_command(
        command,
        env,
        args.timeout,
        stream_output=args.stream_output,
    )
    blob = (process.stdout or "") + "\n" + (process.stderr or "")
    log_path.write_text(text_command + "\n\n" + blob, encoding="utf-8")
    result.update(
        {
            "returncode": process.returncode,
            "seconds": time.monotonic() - started,
            **summarize_mse(blob),
        }
    )
    if process.returncode != 0:
        result["error_tail"] = blob[-3000:]
    if output.exists() and args.circle_inspect.exists():
        inspected = run_command(
            [str(args.circle_inspect), "--operators", str(output)],
            env,
            args.timeout,
        )
        result["operator_inspect_returncode"] = inspected.returncode
        result["operators"] = parse_circle_ops(inspected.stdout or "") if inspected.returncode == 0 else {}
    return result


def validate_args(args: argparse.Namespace) -> None:
    for path in (args.circle, args.calib_data, args.test_data, args.candidate_csv):
        if not path.exists():
            raise SystemExit(f"input not found: {path}")
    if args.max_candidates <= 0 or args.max_int16 <= 0 or args.timeout <= 0:
        raise SystemExit("--max-candidates, --max-int16, and --timeout must be positive")
    if (
        min(
            args.min_mse_improvement,
            args.min_relative_improvement,
            args.latency_weight,
            args.conversion_weight,
        )
        < 0
    ):
        raise SystemExit("improvement thresholds and penalty weights must be non-negative")
    if args.objective == "output" and not args.objective_output:
        raise SystemExit("--objective-output is required with --objective=output")
    if not 0 <= args.min_percentile < args.max_percentile <= 100:
        raise SystemExit("percentiles must satisfy 0 <= min < max <= 100")
    if args.input_data_format == "auto":
        args.input_data_format = infer_format(args.calib_data)
    if args.test_data_format == "auto":
        args.test_data_format = infer_format(args.test_data)
    supported = {"h5", "hdf5", "list", "filelist", "dir", "directory"}
    if args.input_data_format not in supported or args.test_data_format not in supported:
        raise SystemExit("unsupported input or test data format")
    if args.input_data_format != args.test_data_format:
        raise SystemExit(
            "stock one-quantize uses one --input_data_format for calibration and test data; "
            "use the same format for both files"
        )
    if not args.one_quantize.exists():
        raise SystemExit(f"one-quantize not found: {args.one_quantize}")
    if args.conversion_weight > 0 and not args.circle_inspect.exists():
        raise SystemExit("--conversion-weight requires circle-inspect")


def main() -> int:
    args = parse_args()
    validate_args(args)
    out_dir = args.out_dir or (
        Path("/home/cmj/works/ASS/logs/one_mixed_precision_search") / time.strftime("%Y%m%d-%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    _, ops, _, _ = read_circle(args.circle)
    candidates, skipped_candidates = load_candidates(
        args.candidate_csv,
        args.max_candidates,
        ops,
    )
    if not candidates:
        detail = "\n".join(skipped_candidates[:10])
        raise SystemExit(
            "candidate CSV contains no eligible positive-score nodes in the current graph"
            + (f":\n{detail}" if detail else "")
        )
    for message in skipped_candidates:
        print(f"[skip] {message}")

    total_model_cost = max(sum(float(op.rough_ops) for op in ops), 1.0)
    env = build_env()
    if args.input_data_format in {"list", "filelist"}:
        if not args.one_create_quant_dataset.exists():
            raise SystemExit(f"one-create-quant-dataset not found: {args.one_create_quant_dataset}")
        packaged_calib = out_dir / "calibration_from_list.h5"
        print(f"[one] package calibration list -> {packaged_calib}")
        package_list(
            args.calib_data,
            packaged_calib,
            args.one_create_quant_dataset,
            env,
            args.timeout,
        )
        packaged_test = out_dir / "test_from_list.h5"
        if args.test_data.resolve() == args.calib_data.resolve():
            packaged_test = packaged_calib
        else:
            print(f"[one] package test list -> {packaged_test}")
            package_list(
                args.test_data,
                packaged_test,
                args.one_create_quant_dataset,
                env,
                args.timeout,
            )
        args.calib_data = packaged_calib
        args.test_data = packaged_test
        args.input_data_format = "h5"
        args.test_data_format = "h5"

    trials: list[dict[str, Any]] = []
    baseline = evaluate(args, env, out_dir, "00_baseline_u8", None)
    baseline_quality = quality_value(
        baseline,
        args.objective,
        args.objective_output,
    )
    trials.append(
        {
            "round": 0,
            "candidate": None,
            "selected": [],
            "quality": baseline_quality,
            **baseline,
        }
    )
    if args.dry_run:
        print(f"[done] dry-run baseline command written under {out_dir}")
        return 0
    if baseline.get("returncode") != 0 or baseline_quality is None:
        available = list(baseline.get("mse_by_output", {}))
        raise SystemExit(
            f"baseline quantization failed or objective output was absent; "
            f"available outputs={available}: {baseline.get('error_tail', '')}"
        )
    baseline_conversions = conversion_count(baseline)
    if args.conversion_weight > 0 and baseline_conversions is None:
        raise SystemExit("could not inspect baseline conversion operators")

    initial_quality = baseline_quality
    previous_quality = baseline_quality
    current_objective = 1.0
    selected: list[dict[str, Any]] = []
    selected_indices: set[int] = set()
    selected_names: set[str] = set()
    for round_index in range(1, args.max_int16 + 1):
        best: dict[str, Any] | None = None
        for candidate in candidates:
            if candidate["name"] in selected_names:
                continue
            trial_names = [row["name"] for row in selected] + [candidate["name"]]
            trial_id = f"r{round_index:02d}_n{candidate['index']:04d}"
            qconfig = out_dir / "qconfigs" / f"{trial_id}.json"
            qconfig.parent.mkdir(parents=True, exist_ok=True)
            write_qconfig(
                qconfig,
                qconfig_payload(
                    args.circle,
                    trial_names,
                    args.default_dtype,
                    args.default_granularity,
                    args.target_dtype,
                    args.target_granularity,
                ),
            )
            result = evaluate(args, env, out_dir, trial_id, qconfig)
            quality = quality_value(result, args.objective, args.objective_output)
            result.update(
                {
                    "round": round_index,
                    "candidate": candidate["name"],
                    "selected": trial_names,
                    "quality": quality,
                }
            )
            trials.append(result)
            if result.get("returncode") != 0 or quality is None:
                continue

            trial_indices = selected_indices | {candidate["index"]}
            selected_cost = sum(float(row["rough_ops"]) for row in selected) + float(candidate["rough_ops"])
            conversions = conversion_count(result)
            if args.conversion_weight > 0 and conversions is None:
                continue
            extra_conversions = max(
                0,
                (conversions or 0) - (baseline_conversions or 0),
            )
            objective = (
                quality / max(initial_quality, 1e-12)
                + args.latency_weight * selected_cost / total_model_cost
                + args.conversion_weight * extra_conversions / max(len(ops), 1)
            )
            result.update(
                {
                    "selected_rough_ops": selected_cost,
                    "selected_rough_ops_fraction": selected_cost / total_model_cost,
                    "boundary_proxy": boundary_proxy(trial_indices, ops),
                    "conversion_count": conversions,
                    "extra_conversion_count": extra_conversions,
                    "objective_value": objective,
                }
            )
            if best is None or objective < best["objective_value"]:
                best = {"candidate_row": candidate, **result}
        if best is None:
            break

        current_quality = float(best["quality"])
        if not improvement_is_sufficient(
            previous_quality,
            current_quality,
            args.min_mse_improvement,
            args.min_relative_improvement,
        ):
            print(f"[stop] best MSE {current_quality:.8g} does not improve {previous_quality:.8g} enough")
            break
        if float(best["objective_value"]) >= current_objective:
            print(
                f"[stop] best penalized objective {best['objective_value']:.8g} "
                f"does not improve {current_objective:.8g}"
            )
            break
        chosen = best["candidate_row"]
        selected.append(chosen)
        selected_names.add(chosen["name"])
        selected_indices.add(chosen["index"])
        previous_quality = current_quality
        current_objective = float(best["objective_value"])
        print(f"[select] {chosen['name']} mse={current_quality:.8g} objective={best['objective_value']:.8g}")

    final_names = [row["name"] for row in selected]
    final_qconfig = out_dir / "qconfig_best_greedy_int16.json"
    write_qconfig(
        final_qconfig,
        qconfig_payload(
            args.circle,
            final_names,
            args.default_dtype,
            args.default_granularity,
            args.target_dtype,
            args.target_granularity,
        ),
    )
    summary = {
        "circle": str(args.circle),
        "candidate_csv": str(args.candidate_csv),
        "objective": args.objective,
        "objective_output": args.objective_output,
        "baseline_quality": initial_quality,
        "final_quality": previous_quality,
        "final_penalized_objective": current_objective,
        "selected_names": final_names,
        "skipped_candidates": skipped_candidates,
        "final_qconfig": str(final_qconfig),
        "trials": trials,
        "policy": (
            "greedy stock one-quantize output MSE with optional whole-model rough-compute "
            "and actual conversion-operator penalties"
        ),
    }
    (out_dir / "search_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    selected_text = "\n".join(f"- `{name}`" for name in final_names) or "- None"
    (out_dir / "search_summary.md").write_text(
        "# ONE Mixed-Precision Greedy Search\n\n"
        f"Objective: `{args.objective}`\n\n"
        f"Baseline MSE: `{initial_quality:.8g}`\n\n"
        f"Final MSE: `{previous_quality:.8g}`\n\n"
        f"Selected int16 nodes ({len(final_names)}):\n\n{selected_text}\n\n"
        f"Final qconfig: `{final_qconfig}`\n\n"
        "Each trial was quantized and evaluated by stock `one-quantize`.\n",
        encoding="utf-8",
    )
    print(f"[done] wrote {final_qconfig}")
    print(f"[done] wrote {out_dir / 'search_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
