#!/usr/bin/env python3
"""
Measure parameters and GMACs/s for every NPU model variant in the repo.

This is a lightweight wrapper around the accounting parts of
``tools/online/measure_npu_model_stats.py``: it imports the same ``BUILDERS``
registry and the same ``profile_macs`` helper, but it deliberately does **not**
export ONNX or emit MLIR.  That makes it:

- fast to run on a CPU in a few seconds per model,
- usable outside the project Docker image (no ``onnx-mlir`` required),
- suitable as a CI / pre-commit sanity check for AGENT.md rule 15
  (params < 7 M, GMACs/s < 3 per model).

Covered families (matches the ones enumerated in AGENT.md "Current Status"):

- TIGER-NPU-Edge v1 and v2           (``TIGER/``)
- TF-MLPNet TIGEREdgeMLP            (``TF-MLPNet/``)
- DolphinSFCNPU presets              (``DolphinSFCNPU/``)
- BandSCNetNPU presets               (``BandSCNetNPU/``)
- Online SFC recipes                 (``spectral_feature_compression/`` via
                                       ``tools/online/export_onnx_online_model.py``)

Usage:

    ./.venv/bin/python tools/online/measure_gmacs.py               # full suite
    ./.venv/bin/python tools/online/measure_gmacs.py --filter dolphin
    ./.venv/bin/python tools/online/measure_gmacs.py --assert-rule15
    ./.venv/bin/python tools/online/measure_gmacs.py \
        --target dolphin --dolphin-preset edge_small

Rule-15 mode: exits non-zero if any measured model exceeds either the 7 M
parameter ceiling or the 3 GMACs/s compute ceiling.  Use it in CI.
"""

from __future__ import annotations

from typing import Any

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import tempfile

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse the existing accounting primitives so we stay in lock-step with the
# full-audit tool.
from tools.online.measure_npu_model_stats import (  # noqa: E402
    BUILDERS,
    profile_macs,
)

# AGENT.md rule 15 budgets.
PARAM_LIMIT = 7_000_000
GMACS_LIMIT = 3.0

# Suite of model configurations covered by this tool.  Kept local (rather than
# reusing ``READY_SUITE`` from ``measure_npu_model_stats``) because this tool
# intentionally covers a broader set of presets per family while excluding the
# online recipes that require checkpoint files on disk.
GMACS_SUITE: list[dict[str, Any]] = [
    # TIGER family ----------------------------------------------------------
    {"target": "tiger-edge", "label": "TIGERNPUEdgeV1"},
    {"target": "tiger-edge-v2", "label": "TIGERNPUEdgeV2"},
    # TF-MLPNet --------------------------------------------------------------
    {"target": "tf-mlpnet", "label": "TF-MLPNet TIGEREdgeMLP 24ch"},
    # DolphinSFCNPU: all available presets ----------------------------------
    {
        "target": "dolphin",
        "label": "DolphinSFCNPU edge_small",
        "dolphin_preset": "edge_small",
        "n_chan": 1,
        "freqs": 1025,
    },
    {
        "target": "dolphin",
        "label": "DolphinSFCNPU large_6m",
        "dolphin_preset": "large_6m",
        "n_chan": 1,
        "freqs": 1025,
    },
    {
        "target": "dolphin",
        "label": "DolphinSFCNPU large_8m",
        "dolphin_preset": "large_8m",
        "n_chan": 1,
        "freqs": 1025,
    },
    # BandSCNetNPU: all five presets ----------------------------------------
    {
        "target": "band-scnet-npu",
        "label": "BandSCNetNPU edge_small",
        "band_scnet_npu_preset": "edge_small",
        "n_chan": 1,
        "freqs": 2049,
    },
    {
        "target": "band-scnet-npu",
        "label": "BandSCNetNPU rt192k",
        "band_scnet_npu_preset": "rt192k",
        "n_chan": 1,
        "freqs": 2049,
    },
    {
        "target": "band-scnet-npu",
        "label": "BandSCNetNPU rt192k_plus",
        "band_scnet_npu_preset": "rt192k_plus",
        "n_chan": 1,
        "freqs": 2049,
    },
    {
        "target": "band-scnet-npu",
        "label": "BandSCNetNPU rt192k_param2m",
        "band_scnet_npu_preset": "rt192k_param2m",
        "n_chan": 1,
        "freqs": 2049,
    },
    {
        "target": "band-scnet-npu",
        "label": "BandSCNetNPU rt192k_param6m",
        "band_scnet_npu_preset": "rt192k_param6m",
        "n_chan": 1,
        "freqs": 2049,
    },
]


# The existing ``BUILDERS`` read a handful of attributes off the parser
# namespace.  We synthesise a minimal namespace rather than reconstructing
# the full argparse surface in this lightweight tool.
@dataclass
class _BuilderArgs:
    device: str = "cpu"
    sample_rate: int = 44100
    hop_length: int = 512
    n_fft: int = 2048
    freqs: int = 1025
    n_chan: int | None = None
    frames: int = 1
    online_opset: int = 11
    tiger_opset: int = 14
    tf_opset: int = 17
    include_elementwise_flops: bool = False
    model_path: Path | None = None
    label: str | None = None
    dolphin_preset: str = "edge_small"
    band_scnet_npu_preset: str = "rt192k"
    tf_out_channels: int = 24
    tf_in_channels: int = 96
    tf_num_blocks: int = 4
    tf_upsampling_depth: int = 2
    tf_num_sources: int = 2
    tf_edge_hidden_channels: int = 24
    tf_edge_num_blocks: int = 4


def _parameter_count(module: torch.nn.Module) -> int:
    return sum(int(p.numel()) for p in module.parameters())


def measure_one(
    spec: dict[str, Any],
    *,
    sample_rate: int,
    hop_length: int,
    frames: int,
    include_elementwise_flops: bool,
) -> dict[str, Any]:
    """Build one model, measure MACs per call, and derive realtime GMACs/s.

    Returns a row dict. ``error`` is populated (and numeric fields left as
    ``None``) if the build or profile fails; we never raise out of this
    function so the caller can continue with the rest of the suite.
    """
    target = spec["target"]
    label = spec.get("label") or target

    builder = BUILDERS[target]
    args = _BuilderArgs(
        sample_rate=sample_rate,
        hop_length=hop_length,
        frames=int(spec.get("frames", frames)),
        n_chan=spec.get("n_chan"),
        freqs=int(spec.get("freqs", 1025)),
        model_path=spec.get("model_path"),
        label=label,
        dolphin_preset=spec.get("dolphin_preset", "edge_small"),
        band_scnet_npu_preset=spec.get("band_scnet_npu_preset", "rt192k"),
        include_elementwise_flops=include_elementwise_flops,
    )

    row: dict[str, Any] = {
        "label": label,
        "target": target,
        "params": None,
        "param_m": None,
        "frames_per_call": None,
        "mac_per_call": None,
        "mac_per_frame": None,
        "gmac_per_s": None,
        "frame_rate": None,
    }
    # Builder writes ONNX into a scratch path during export; it is not used
    # here but we provide a real tmpdir so the dataclass stays well-formed.
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            exported = builder(args, dict(spec), Path(tmpdir))
            mac_per_call, _ = profile_macs(
                exported.export_module,
                exported.export_inputs,
                include_elementwise=include_elementwise_flops,
            )
        except Exception as exc:  # builder or profiler failed; record and move on
            row["error"] = f"{type(exc).__name__}: {exc}"
            return row

    frames_per_call = max(exported.frames_per_call, 1)
    mac_per_frame = mac_per_call / frames_per_call
    frame_rate = float(sample_rate) / float(hop_length)
    params = _parameter_count(exported.module)

    row.update(
        {
            "params": params,
            "param_m": params / 1e6,
            "frames_per_call": frames_per_call,
            "mac_per_call": mac_per_call,
            "mac_per_frame": mac_per_frame,
            "gmac_per_s": mac_per_frame * frame_rate / 1e9,
            "frame_rate": frame_rate,
            "source": exported.source,
        }
    )
    return row


def _expand_filters(filters: list[str]) -> tuple[str, ...]:
    return tuple(f.lower() for f in filters)


def _spec_matches(spec: dict[str, Any], filters: tuple[str, ...]) -> bool:
    if not filters:
        return True
    haystack = " ".join(
        str(spec.get(key, "")).lower()
        for key in ("label", "target", "dolphin_preset", "band_scnet_npu_preset")
    )
    return any(token in haystack for token in filters)


def build_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.target == "suite":
        filters = _expand_filters(args.filter)
        return [dict(s) for s in GMACS_SUITE if _spec_matches(s, filters)]

    # Single-target run — construct a spec from CLI args.
    spec: dict[str, Any] = {
        "target": args.target,
        "label": args.label or args.target,
    }
    if args.n_chan is not None:
        spec["n_chan"] = args.n_chan
    if args.freqs is not None:
        spec["freqs"] = args.freqs
    if args.target == "dolphin":
        spec["dolphin_preset"] = args.dolphin_preset
    elif args.target == "band-scnet-npu":
        spec["band_scnet_npu_preset"] = args.band_scnet_npu_preset
    elif args.target == "online":
        if not args.model_path:
            raise SystemExit("--model-path is required when --target online.")
        spec["model_path"] = args.model_path
    return [spec]


def format_row(row: dict[str, Any]) -> list[str]:
    def fmt(value: Any, digits: int = 4) -> str:
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return str(value)

    status = "ok"
    params = row.get("params") or 0
    gmacs = row.get("gmac_per_s")
    if row.get("error"):
        status = "ERROR"
    elif params > PARAM_LIMIT or (gmacs is not None and gmacs > GMACS_LIMIT):
        status = "over-budget"

    return [
        str(row.get("label", "")),
        str(row.get("target", "")),
        fmt(row.get("param_m"), 3),
        fmt(gmacs, 3),
        fmt(row.get("mac_per_frame"), 0) if row.get("mac_per_frame") is not None else "-",
        status,
    ]


def print_table(rows: list[dict[str, Any]]) -> None:
    headers = ["model", "target", "params(M)", "GMACs/s", "MAC/frame", "status"]
    body = [format_row(row) for row in rows]
    widths = [len(h) for h in headers]
    for r in body:
        widths = [max(w, len(c)) for w, c in zip(widths, r)]
    line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(line)
    print(" | ".join("-" * w for w in widths))
    for r in body:
        print(" | ".join(c.ljust(w) for c, w in zip(r, widths)))

    # Print any error rows in full so users do not have to chase JSON.
    errored = [r for r in rows if r.get("error")]
    if errored:
        print()
        print("Errors:")
        for r in errored:
            print(f"  - {r.get('label')}: {r['error']}")


def rule15_failures(rows: list[dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    for row in rows:
        if row.get("error"):
            failures.append(f"{row.get('label')}: build failed ({row['error']})")
            continue
        params = row.get("params") or 0
        gmacs = row.get("gmac_per_s") or 0.0
        if params > PARAM_LIMIT:
            failures.append(
                f"{row.get('label')}: params={params:,} exceeds {PARAM_LIMIT:,} (rule 15)"
            )
        if gmacs > GMACS_LIMIT:
            failures.append(
                f"{row.get('label')}: GMACs/s={gmacs:.3f} exceeds {GMACS_LIMIT:.1f} (rule 15)"
            )
    return failures


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Measure parameter count and realtime GMACs/s for every NPU model "
            "variant in the repo. Reuses the accounting primitives from "
            "tools/online/measure_npu_model_stats.py but skips ONNX/MLIR so it "
            "runs quickly without the Docker toolchain."
        )
    )
    p.add_argument(
        "--target",
        choices=["suite", "tiger-edge", "tiger-edge-v2", "tf-mlpnet", "dolphin", "band-scnet-npu", "online"],
        default="suite",
        help="Which model(s) to measure. Default 'suite' covers every built-in preset.",
    )
    p.add_argument(
        "--filter",
        action="append",
        default=[],
        help="Substring filter for --target suite; repeat to match any of several.",
    )
    p.add_argument("--sample-rate", type=int, default=44100)
    p.add_argument("--hop-length", type=int, default=512)
    p.add_argument("--frames", type=int, default=1, help="Frames per streaming call.")
    p.add_argument("--n-chan", type=int, default=None)
    p.add_argument("--freqs", type=int, default=None)
    p.add_argument("--label", type=str, default=None)
    p.add_argument("--model-path", type=Path, default=None, help="Required for --target online.")
    p.add_argument("--dolphin-preset", default="edge_small")
    p.add_argument("--band-scnet-npu-preset", default="rt192k")
    p.add_argument(
        "--include-elementwise-flops",
        action="store_true",
        help="Include profiler-reported elementwise FLOPs. Default counts conv/mm/bmm only.",
    )
    p.add_argument("--json-out", type=Path, default=None)
    p.add_argument(
        "--assert-rule15",
        action="store_true",
        help=(
            f"Exit non-zero if any model exceeds AGENT.md rule 15 "
            f"(params > {PARAM_LIMIT:,} or GMACs/s > {GMACS_LIMIT})."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    specs = build_specs(args)
    if not specs:
        raise SystemExit("No specs matched; check --filter.")

    rows: list[dict[str, Any]] = []
    for spec in specs:
        row = measure_one(
            spec,
            sample_rate=args.sample_rate,
            hop_length=args.hop_length,
            frames=args.frames,
            include_elementwise_flops=args.include_elementwise_flops,
        )
        rows.append(row)

    print_table(rows)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nWrote JSON: {args.json_out}")

    if args.assert_rule15:
        failures = rule15_failures(rows)
        if failures:
            print("\nAGENT.md rule 15 violations:")
            for f in failures:
                print(f"  - {f}")
            sys.exit(1)
        print("\nAll measured models satisfy AGENT.md rule 15.")


if __name__ == "__main__":
    main()
