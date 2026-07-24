#!/usr/bin/env python3
"""Rank Circle nodes for stock ONE uint8/int16 mixed quantization.

ONNX Runtime supplies representative activation values. Stock record-minmax
and circle-quantizer supply the ranges and quantization parameters.
"""

from __future__ import annotations

from typing import Any

import argparse
import csv
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path("/home/cmj/works/ASS")
ONE_ROOT = Path(os.environ.get("ONE_ROOT", "/home/cmj/works/ONE"))
ONE_BUILD_COMPILER = Path(os.environ.get("ONE_BUILD_COMPILER", str(ONE_ROOT / "build" / "compiler")))
ONE_RECORD_MINMAX = Path(
    os.environ.get("ONE_RECORD_MINMAX", str(ONE_BUILD_COMPILER / "record-minmax" / "record-minmax"))
)
ONE_CIRCLE_QUANTIZER = Path(
    os.environ.get(
        "ONE_CIRCLE_QUANTIZER",
        str(ONE_BUILD_COMPILER / "circle-quantizer" / "circle-quantizer"),
    )
)
DEFAULT_OUT_ROOT = ROOT / "logs" / "one_mixed_precision_calibration"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_one_stock_quant_sweep import build_env  # noqa: E402
from suggest_one_mixed_precision_qconfig import (  # noqa: E402
    LOW_VALUE_MEMORY_OPS,
    OpInfo,
    qconfig_payload,
    read_circle,
    write_qconfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure stock-ONE uint8/int16 activation error and emit qconfigs.")
    parser.add_argument("--circle", type=Path, required=True, help="Optimized fp32 Circle model.")
    parser.add_argument("--onnx", type=Path, required=True, help="ONNX exported from the same graph.")
    parser.add_argument(
        "--calib-data",
        type=Path,
        required=True,
        help="ONE list/filelist: one raw file per line, or one file per input separated by spaces.",
    )
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--recorded-circle", type=Path, help="Existing record-minmax Circle.")
    parser.add_argument("--record-minmax", type=Path, default=ONE_RECORD_MINMAX)
    parser.add_argument("--circle-quantizer", type=Path, default=ONE_CIRCLE_QUANTIZER)
    parser.add_argument("--min-percentile", type=float, default=1.0)
    parser.add_argument("--max-percentile", type=float, default=99.0)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument(
        "--max-values-per-node",
        type=int,
        default=20000,
        help="Reservoir size used for local MSE; ranges still come from record-minmax.",
    )
    parser.add_argument(
        "--onnx-output-batch-size",
        type=int,
        default=16,
        help="Maximum mapped intermediate outputs exposed to ONNX Runtime at once.",
    )
    parser.add_argument("--sampling-seed", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--default-dtype", choices=("uint8",), default="uint8")
    parser.add_argument("--default-granularity", choices=("layer", "channel"), default="channel")
    parser.add_argument("--target-dtype", choices=("int16",), default="int16")
    parser.add_argument("--target-granularity", choices=("layer", "channel"), default="channel")
    parser.add_argument(
        "--include-op",
        action="append",
        default=[],
        help="Include an otherwise low-value op type. May be repeated.",
    )
    return parser.parse_args()


def run_checked(command: list[str], timeout: int, label: str) -> None:
    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
        env=build_env(),
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        text = (result.stdout + "\n" + result.stderr).strip()
        raise RuntimeError(f"{label} failed ({result.returncode}):\n{text[-4000:]}")


def run_record_minmax(args: argparse.Namespace, output: Path) -> None:
    if not args.record_minmax.exists():
        raise FileNotFoundError(f"record-minmax not found: {args.record_minmax}")
    run_checked(
        [
            str(args.record_minmax),
            "--input_model",
            str(args.circle),
            "--input_data",
            str(args.calib_data),
            "--input_data_format",
            "list",
            "--output_model",
            str(output),
            "--mode",
            "percentile",
            "--min_percentile",
            str(args.min_percentile),
            "--max_percentile",
            str(args.max_percentile),
        ],
        args.timeout,
        "record-minmax",
    )


def quantize_recorded(
    args: argparse.Namespace,
    recorded: Path,
    output: Path,
    dtype: str,
) -> None:
    if not args.circle_quantizer.exists():
        raise FileNotFoundError(f"circle-quantizer not found: {args.circle_quantizer}")
    run_checked(
        [
            str(args.circle_quantizer),
            "--quantize_with_minmax",
            "float32",
            dtype,
            args.default_granularity if dtype == "uint8" else args.target_granularity,
            "--input_type",
            "uint8",
            "--output_type",
            "uint8",
            str(recorded),
            str(output),
        ],
        args.timeout,
        f"circle-quantizer ({dtype})",
    )


def load_circle_ranges(path: Path) -> dict[str, tuple[float, float]]:
    from circle_schema.v0_9.circle.Model import Model

    model = Model.GetRootAs(path.read_bytes(), 0)
    subgraph = model.Subgraphs(0)
    ranges: dict[str, tuple[float, float]] = {}
    for index in range(subgraph.TensorsLength()):
        tensor = subgraph.Tensors(index)
        quant = tensor.Quantization()
        if quant is None or quant.MinLength() == 0 or quant.MaxLength() == 0:
            continue
        raw_name = tensor.Name()
        if raw_name is not None:
            ranges[raw_name.decode("utf-8", errors="replace")] = (
                float(quant.Min(0)),
                float(quant.Max(0)),
            )
    return ranges


def load_circle_qparams(path: Path) -> dict[str, tuple[float, int]]:
    from circle_schema.v0_9.circle.Model import Model

    model = Model.GetRootAs(path.read_bytes(), 0)
    subgraph = model.Subgraphs(0)
    params: dict[str, tuple[float, int]] = {}
    for index in range(subgraph.TensorsLength()):
        tensor = subgraph.Tensors(index)
        quant = tensor.Quantization()
        if quant is None or quant.ScaleLength() == 0:
            continue
        raw_name = tensor.Name()
        if raw_name is None:
            continue
        name = raw_name.decode("utf-8", errors="replace")
        zero = int(quant.ZeroPoint(0)) if quant.ZeroPointLength() else 0
        params[name] = (float(quant.Scale(0)), zero)
    return params


def clean_identifier(name: str) -> set[str]:
    result: set[str] = set()
    for part in name.split(";"):
        value = part.strip().lstrip("/")
        if not value:
            continue
        result.add(value)
        for suffix in ("/pre_tr", "/post_tr", "/pads"):
            if value.endswith(suffix):
                result.add(value[: -len(suffix)])
        if value.endswith("_output_0"):
            result.add(value[: -len("_output_0")])
        if value.endswith(":0"):
            result.add(value[:-2])
    return result


def by_alias(values: dict[str, Any]) -> dict[str, Any]:
    aliases: dict[str, Any] = {}
    for name, value in values.items():
        aliases.setdefault(name, value)
        for alias in clean_identifier(name):
            aliases.setdefault(alias, value)
    return aliases


def value_for_name(name: str, values: dict[str, Any], aliases: dict[str, Any]) -> Any | None:
    if name in values:
        return values[name]
    return next((aliases[alias] for alias in clean_identifier(name) if alias in aliases), None)


def infer_onnx_shapes(model: Any) -> tuple[Any, str | None]:
    from onnx import shape_inference

    try:
        return shape_inference.infer_shapes(model), None
    except Exception as error:  # ONNX shape inference is best-effort for custom graphs.
        return model, f"{type(error).__name__}: {error}"


def onnx_output_map(model: Any) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for node in model.graph.node:
        if not node.output:
            continue
        output = node.output[0]
        for identifier in clean_identifier(node.name):
            mapping.setdefault(identifier, output)
        for identifier in clean_identifier(output):
            mapping.setdefault(identifier, output)
    return mapping


def known_onnx_outputs(model: Any) -> set[str]:
    known = {value.name for value in list(model.graph.value_info) + list(model.graph.output) + list(model.graph.input)}
    known.update(initializer.name for initializer in model.graph.initializer)
    return known


def expose_onnx_outputs(model: Any, names: list[str]) -> Any:
    import onnx

    copy = onnx.load_from_string(model.SerializeToString())
    known = {
        value.name: value for value in list(copy.graph.value_info) + list(copy.graph.output) + list(copy.graph.input)
    }
    existing = {value.name for value in copy.graph.output}
    for name in names:
        if name not in existing:
            copy.graph.output.append(known[name])
    return copy


def input_dtype(type_name: str) -> np.dtype:
    types = {
        "tensor(float)": np.dtype(np.float32),
        "tensor(float16)": np.dtype(np.float16),
        "tensor(double)": np.dtype(np.float64),
        "tensor(int32)": np.dtype(np.int32),
        "tensor(int64)": np.dtype(np.int64),
        "tensor(uint8)": np.dtype(np.uint8),
        "tensor(int8)": np.dtype(np.int8),
        "tensor(bool)": np.dtype(np.bool_),
    }
    if type_name not in types:
        raise ValueError(f"Unsupported ONNX input type for raw list data: {type_name}")
    return types[type_name]


def static_shape(input_meta: Any) -> list[int] | None:
    shape = []
    for value in input_meta.shape:
        if not isinstance(value, int) or value <= 0:
            return None
        shape.append(value)
    return shape


def read_input(path: Path, meta: Any) -> np.ndarray:
    dtype = input_dtype(meta.type)
    if path.suffix.lower() == ".npy":
        value = np.load(path)
        return value.astype(dtype, copy=False)
    shape = static_shape(meta)
    if shape is None:
        raise ValueError(f"Raw input requires a static ONNX shape: {meta.name} {meta.shape}")
    value = np.fromfile(path, dtype=dtype)
    expected = int(np.prod(shape))
    if value.size != expected:
        raise ValueError(f"Wrong element count in {path}: got {value.size}, expected {expected}")
    return value.reshape(shape)


def read_calibration_inputs(
    path: Path,
    metas: list[Any],
    limit: int,
) -> list[dict[str, np.ndarray]]:
    records: list[dict[str, np.ndarray]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        files = [Path(item) for item in line.split()]
        if len(files) == 1 and len(metas) > 1:
            data = files[0].read_bytes()
            offset = 0
            arrays = []
            for meta in metas:
                dtype = input_dtype(meta.type)
                shape = static_shape(meta)
                if shape is None:
                    raise ValueError(f"Concatenated raw input requires static shape: {meta.name}")
                count = int(np.prod(shape))
                size = count * dtype.itemsize
                arrays.append(np.frombuffer(data[offset : offset + size], dtype=dtype).reshape(shape).copy())
                offset += size
            if offset != len(data):
                raise ValueError(f"Unused bytes in concatenated input file: {files[0]}")
        elif len(files) == len(metas):
            arrays = [read_input(file, meta) for file, meta in zip(files, metas)]
        else:
            raise ValueError(f"Calibration line has {len(files)} files, model has {len(metas)} inputs")
        records.append({meta.name: value for meta, value in zip(metas, arrays)})
        if len(records) >= limit:
            break
    if not records:
        raise ValueError(f"No calibration records found in {path}")
    return records


@dataclass
class PriorityReservoir:
    limit: int
    rng: np.random.Generator
    values: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    priorities: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    seen: int = 0

    def update(self, value: np.ndarray) -> None:
        flat = np.asarray(value).reshape(-1)
        if not np.issubdtype(flat.dtype, np.floating):
            return
        finite = flat[np.isfinite(flat)]
        self.seen += int(finite.size)
        chunk_size = max(self.limit * 4, 65536)
        for start in range(0, finite.size, chunk_size):
            chunk = finite[start : start + chunk_size].astype(np.float32, copy=False)
            priorities = self.rng.random(chunk.size)
            values = np.concatenate((self.values, chunk))
            keys = np.concatenate((self.priorities, priorities))
            if values.size > self.limit:
                keep = np.argpartition(keys, self.limit - 1)[: self.limit]
                values = values[keep]
                keys = keys[keep]
            self.values = values
            self.priorities = keys


def collect_stats(
    model: Any,
    records: list[dict[str, np.ndarray]],
    output_names: list[str],
    batch_size: int,
    limit: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    import onnxruntime as ort

    reservoirs = {
        name: PriorityReservoir(limit, np.random.default_rng(seed + index)) for index, name in enumerate(output_names)
    }
    for batch_start in range(0, len(output_names), batch_size):
        names = output_names[batch_start : batch_start + batch_size]
        exposed = expose_onnx_outputs(model, names)
        session = ort.InferenceSession(
            exposed.SerializeToString(),
            providers=["CPUExecutionProvider"],
        )
        batch_number = batch_start // batch_size + 1
        batch_count = (len(output_names) + batch_size - 1) // batch_size
        for sample_index, feeds in enumerate(records):
            outputs = session.run(names, feeds)
            for name, value in zip(names, outputs):
                reservoirs[name].update(np.asarray(value))
            print(
                f"[onnx] output batch {batch_number}/{batch_count}, sample {sample_index + 1}/{len(records)}",
                flush=True,
            )
    stats = {name: item.values for name, item in reservoirs.items() if item.values.size}
    seen = {name: item.seen for name, item in reservoirs.items()}
    return stats, seen


def round_away_from_zero(values: np.ndarray) -> np.ndarray:
    return np.where(values >= 0, np.floor(values + 0.5), np.ceil(values - 0.5))


def dequantize_error(
    values: np.ndarray,
    scale: float,
    zero: int,
    quant_min: int,
    quant_max: int,
) -> tuple[float, float, float]:
    finite = values[np.isfinite(values)].astype(np.float64, copy=False)
    if finite.size == 0 or not np.isfinite(scale) or scale <= 0:
        return float("nan"), float("nan"), 0.0
    q = round_away_from_zero(finite / scale + zero)
    clipped = np.count_nonzero((q < quant_min) | (q > quant_max)) / finite.size
    q = np.clip(q, quant_min, quant_max)
    restored = (q - zero) * scale
    error = finite - restored
    return float(np.mean(error * error)), float(np.mean(np.abs(error))), float(clipped)


def circle_range_for_op(
    op: OpInfo,
    ranges: dict[str, tuple[float, float]],
    aliases: dict[str, tuple[float, float]],
) -> tuple[float, float] | None:
    return value_for_name(op.name, ranges, aliases)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "index",
        "name",
        "op",
        "mapped_output",
        "depth",
        "rough_ops",
        "eligible",
        "values_sampled",
        "values_seen",
        "range_min",
        "range_max",
        "u8_scale",
        "u8_zero_point",
        "i16_scale",
        "i16_zero_point",
        "u8_mse",
        "i16_mse",
        "mse_reduction",
        "relative_reduction",
        "clip_u8",
        "clip_i16",
        "selection_score",
        "reasons",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def main() -> int:
    args = parse_args()
    for path in (args.circle, args.onnx, args.calib_data):
        if not path.exists():
            raise SystemExit(f"input not found: {path}")
    positive = (
        args.max_samples,
        args.max_values_per_node,
        args.onnx_output_batch_size,
        args.timeout,
        args.top_k,
    )
    if any(value <= 0 for value in positive):
        raise SystemExit("sample, reservoir, batch, timeout, and top-k values must be positive")
    if not 0 <= args.min_percentile < args.max_percentile <= 100:
        raise SystemExit("percentiles must satisfy 0 <= min < max <= 100")

    out_dir = args.out_dir or (DEFAULT_OUT_ROOT / time.strftime("%Y%m%d-%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    recorded = args.recorded_circle or (out_dir / "recorded.minmax.circle")
    if args.recorded_circle is None:
        print(f"[one] record-minmax -> {recorded}")
        run_record_minmax(args, recorded)

    full_u8 = out_dir / "full_uint8.q.circle"
    full_i16 = out_dir / "full_int16.q.circle"
    print(f"[one] stock uint8 qparams -> {full_u8}")
    quantize_recorded(args, recorded, full_u8, "uint8")
    print(f"[one] stock int16 qparams -> {full_i16}")
    quantize_recorded(args, recorded, full_i16, "int16")

    ranges = load_circle_ranges(recorded)
    range_aliases = by_alias(ranges)
    u8_params = load_circle_qparams(full_u8)
    i16_params = load_circle_qparams(full_i16)
    u8_aliases = by_alias(u8_params)
    i16_aliases = by_alias(i16_params)

    import onnx
    import onnxruntime as ort

    original_model = onnx.load(str(args.onnx))
    model, shape_inference_error = infer_onnx_shapes(original_model)
    base_session = ort.InferenceSession(
        original_model.SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    records = read_calibration_inputs(
        args.calib_data,
        list(base_session.get_inputs()),
        args.max_samples,
    )

    _tensors, ops, _input_tensors, _output_tensors = read_circle(args.circle)
    mapping = onnx_output_map(original_model)
    mapped_by_op = {
        op.index: next(
            (mapping[key] for key in clean_identifier(op.name) if key in mapping),
            "",
        )
        for op in ops
    }
    known_outputs = known_onnx_outputs(model)
    output_names = sorted({name for name in mapped_by_op.values() if name and name in known_outputs})
    stats, seen = collect_stats(
        model,
        records,
        output_names,
        args.onnx_output_batch_size,
        args.max_values_per_node,
        args.sampling_seed,
    )

    include_ops = set(args.include_op)
    rows: list[dict[str, Any]] = []
    unmatched: dict[str, int] = {}
    for op in ops:
        mapped_output = mapped_by_op[op.index]
        values = stats.get(mapped_output)
        u8_param = value_for_name(op.name, u8_params, u8_aliases)
        i16_param = value_for_name(op.name, i16_params, i16_aliases)
        if not mapped_output:
            reason = "no ONNX output mapping"
        elif mapped_output not in known_outputs:
            reason = "mapped ONNX output lacks inferred type/shape"
        elif values is None:
            reason = "mapped ONNX output has no floating samples"
        elif u8_param is None:
            reason = "stock uint8 model has no output qparam"
        elif i16_param is None:
            reason = "stock int16 model has no output qparam"
        else:
            reason = ""
        if reason:
            unmatched[reason] = unmatched.get(reason, 0) + 1
            continue

        range_pair = circle_range_for_op(op, ranges, range_aliases)
        if range_pair is None:
            range_pair = (
                float(np.percentile(values, args.min_percentile)),
                float(np.percentile(values, args.max_percentile)),
            )
        min_value, max_value = range_pair
        u8_scale, u8_zero = u8_param
        i16_scale, i16_zero = i16_param
        u8_mse, _, clip_u8 = dequantize_error(values, u8_scale, u8_zero, 0, 255)
        i16_mse, _, clip_i16 = dequantize_error(
            values,
            i16_scale,
            i16_zero,
            -32768,
            32767,
        )
        signal = max(float(np.mean(values.astype(np.float64) ** 2)), 1e-12)
        reduction = max(u8_mse - i16_mse, 0.0)
        relative = reduction / max(u8_mse, 1e-12)
        eligible = op.op not in LOW_VALUE_MEMORY_OPS or op.op in include_ops
        score = 100.0 * reduction / signal + 25.0 * relative
        if not eligible:
            score = -1.0
        rows.append(
            {
                "index": op.index,
                "name": op.name,
                "op": op.op,
                "mapped_output": mapped_output,
                "depth": op.depth,
                "rough_ops": f"{op.rough_ops:.0f}",
                "eligible": str(eligible).lower(),
                "values_sampled": values.size,
                "values_seen": seen.get(mapped_output, 0),
                "range_min": min_value,
                "range_max": max_value,
                "u8_scale": u8_scale,
                "u8_zero_point": u8_zero,
                "i16_scale": i16_scale,
                "i16_zero_point": i16_zero,
                "u8_mse": u8_mse,
                "i16_mse": i16_mse,
                "mse_reduction": reduction,
                "relative_reduction": relative,
                "clip_u8": clip_u8,
                "clip_i16": clip_i16,
                "selection_score": score,
                "reasons": "activation reconstruction error using stock Circle qparams",
            }
        )

    rows.sort(key=lambda row: float(row["selection_score"]), reverse=True)
    write_csv(out_dir / "nodes.csv", rows)
    top = [row["name"] for row in rows if row["eligible"] == "true" and float(row["selection_score"]) > 0][: args.top_k]
    if top:
        qconfig_path = out_dir / f"qconfig_calibration_top{len(top)}_int16.json"
        write_qconfig(
            qconfig_path,
            qconfig_payload(
                args.circle,
                top,
                args.default_dtype,
                args.default_granularity,
                args.target_dtype,
                args.target_granularity,
            ),
        )
    else:
        qconfig_path = None

    metadata = {
        "circle": str(args.circle),
        "onnx": str(args.onnx),
        "calib_data": str(args.calib_data),
        "recorded_circle": str(recorded),
        "full_uint8_circle": str(full_u8),
        "full_int16_circle": str(full_i16),
        "samples": len(records),
        "num_circle_ops": len(ops),
        "num_mapped_ops": sum(bool(value) for value in mapped_by_op.values()),
        "num_unique_onnx_outputs": len(output_names),
        "rows": len(rows),
        "unmatched_ops": len(ops) - len(rows),
        "unmatched_reasons": unmatched,
        "shape_inference_error": shape_inference_error,
        "qconfig": str(qconfig_path) if qconfig_path else None,
        "notes": [
            "Ranges come from stock record-minmax.",
            "Scales and zero points are read from stock full-uint8 and full-int16 Circle models.",
            "Local MSE uses deterministic reservoir samples from the complete selected calibration set.",
            "Use search_one_mixed_precision_qconfig.py for stock final-output evaluation.",
        ],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "summary.md").write_text(
        "# ONE Mixed-Precision Calibration Ranking\n\n"
        f"Ranked {len(rows)} of {len(ops)} Circle nodes over {len(records)} calibration records.\n\n"
        "The ranking measures local int16-versus-uint8 reconstruction error using qparams "
        "read from stock `circle-quantizer` outputs. It does not replace final output evaluation.\n\n"
        f"- Node table: `{out_dir / 'nodes.csv'}`\n"
        f"- Candidate qconfig: `{qconfig_path}`\n"
        f"- Unmatched operators: `{len(ops) - len(rows)}`\n"
        "- Next step: run the stock greedy search and inspect output MSE and conversion operators.\n",
        encoding="utf-8",
    )
    print(f"[done] wrote {out_dir / 'nodes.csv'}")
    if qconfig_path:
        print(f"[done] wrote {qconfig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
