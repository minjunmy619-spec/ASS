#!/usr/bin/env python3
"""Inspect / estimate Band-SCNet-NPU preset configurations.

Three modes:

1. ``list`` - table of every registered preset with its key constructor
   arguments, trainable parameter count, and streaming-state size at a
   given ``n_freq`` and dtype. Use this to see what is already available.

2. ``show <name>`` - the full constructor kwargs for one registered preset
   (JSON + pretty print), plus the same metrics as ``list``. Use this to
   quickly grab the config of a specific preset (for example to drop into
   a recipe YAML or to branch from when designing a new variant).

3. ``estimate`` - build an ad-hoc ``BandSCNetNPU`` from CLI flags (channels,
   num_stages, time_kernel, pooled_mixer_hidden, attention knobs, ...)
   and report trainable params and streaming-state bytes without
   registering anything. Use this to sweep new preset candidates and
   check them against the 192 KiB DSP state budget before committing
   them to ``BandSCNetNPU/presets.py``.

The script does NOT modify ``presets.py``. It only inspects and estimates.

Examples (run from the repo root, inside the .venv as per AGENT.md):

    # List every registered preset at the standard STFT width
    ./.venv/bin/python tools/online/band_scnet_npu_presets.py list \
        --n-freq 2049

    # Dump a specific preset's full kwargs as JSON
    ./.venv/bin/python tools/online/band_scnet_npu_presets.py show rt192k \
        --n-freq 2049 --json

    # Estimate a new candidate preset (C=48, 3 stages, Kt=3, pooled=8192)
    ./.venv/bin/python tools/online/band_scnet_npu_presets.py estimate \
        --n-freq 2049 \
        --channels 48 --num-stages 3 --time-kernel 3 \
        --use-attn --attn-window 16 --num-heads 4 --head-dim 8 \
        --pooled-mixer-hidden 8192

    # Sweep a small grid over channels and pooled-mixer hidden size
    ./.venv/bin/python tools/online/band_scnet_npu_presets.py estimate \
        --n-freq 2049 \
        --sweep-channels 32,40,48,56 \
        --sweep-pooled 0,4096,8192,16384 \
        --num-stages 2 --time-kernel 3 --use-attn
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from BandSCNetNPU import BandSCNetNPU  # noqa: E402
from BandSCNetNPU import presets as _presets_mod  # noqa: E402


DTYPE_MAP: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


# ---------------------------------------------------------------------------
# preset registry access


def registered_presets() -> dict[str, Callable[..., BandSCNetNPU]]:
    """Return the preset-name -> factory mapping from ``BandSCNetNPU.presets``.

    Uses the private ``_PRESETS`` dict if present (current canonical source of
    truth), otherwise falls back to introspecting public module attributes.
    """
    registry = getattr(_presets_mod, "_PRESETS", None)
    if isinstance(registry, dict) and registry:
        return dict(registry)
    # Fallback: any module-level callable returning a BandSCNetNPU.
    fallback: dict[str, Callable[..., BandSCNetNPU]] = {}
    for name, obj in vars(_presets_mod).items():
        if name.startswith("_") or not callable(obj):
            continue
        if name == "build_band_scnet_npu_preset":
            continue
        fallback[name] = obj
    return fallback


# ---------------------------------------------------------------------------
# metric computation


@dataclass
class PresetMetrics:
    name: str
    params: int
    state_bytes: int
    n_freq: int
    n_freq_padded: int
    concat_width: int
    out_widths: tuple[int, int, int]
    dtype: str

    @property
    def state_kib(self) -> float:
        return self.state_bytes / 1024.0


def compute_metrics(
    model: BandSCNetNPU,
    *,
    name: str,
    dtype: torch.dtype,
) -> PresetMetrics:
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    state_bytes = model.state_size_bytes(dtype=dtype)
    dtype_name = str(dtype).replace("torch.", "")
    return PresetMetrics(
        name=name,
        params=int(params),
        state_bytes=int(state_bytes),
        n_freq=int(model.n_freq),
        n_freq_padded=int(model.n_freq_padded),
        concat_width=int(model.concat_width),
        out_widths=tuple(int(x) for x in model.out_widths),
        dtype=dtype_name,
    )


# ---------------------------------------------------------------------------
# kwargs extraction for ``show``


def reconstruct_preset_kwargs(
    factory: Callable[..., BandSCNetNPU],
    *,
    n_freq: int,
    n_src: int,
    n_chan: int,
    masking: bool,
) -> tuple[dict[str, Any], BandSCNetNPU]:
    """Build the preset with the caller's n_freq / n_src / n_chan / masking and
    read the resulting configuration back off the model.

    The factories in ``presets.py`` hardcode all the architecture knobs
    (channels, num_stages, time_kernel, ...). We round-trip through a real
    instance so the report stays correct even if a factory learns to compute
    something at runtime (e.g. picking num_stages from ratios)."""
    model = factory(
        n_freq=n_freq,
        n_src=n_src,
        n_chan=n_chan,
        masking=masking,
    )
    kwargs: dict[str, Any] = {}
    # Core architecture knobs read directly off the assembled module.
    direct_attrs = [
        "n_freq",
        "n_src",
        "n_chan",
        "channels",
        "pyramid_channels",
        "num_stages",
        "time_kernel",
        "freq_kernel",
        "pyramid_time_kernel",
        "pyramid_freq_kernel",
        "use_attn",
        "attn_window",
        "num_heads",
        "head_dim",
        "pooled_mixer_hidden",
        "ratios",
        "masking",
    ]
    for attr in direct_attrs:
        if hasattr(model, attr):
            value = getattr(model, attr)
            if isinstance(value, tuple):
                value = list(value)
            kwargs[attr] = value

    # Pyramid strides / conv blocks live on the encoder.
    enc = getattr(model, "encoder", None)
    for src_attr, dst_attr in [
        ("conv_blocks_per_branch", "pyramid_conv_blocks"),
        ("strides_per_branch", "pyramid_strides"),
    ]:
        if enc is not None and hasattr(enc, src_attr):
            value = getattr(enc, src_attr)
            if isinstance(value, tuple):
                value = list(value)
            kwargs[dst_attr] = value

    return kwargs, model


# ---------------------------------------------------------------------------
# table formatting


def format_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "(no rows)"
    widths = {c: len(c) for c in columns}
    string_rows: list[dict[str, str]] = []
    for row in rows:
        sr: dict[str, str] = {}
        for c in columns:
            v = row.get(c, "")
            if isinstance(v, float):
                s = f"{v:.2f}"
            else:
                s = str(v)
            widths[c] = max(widths[c], len(s))
            sr[c] = s
        string_rows.append(sr)
    header = "  ".join(c.ljust(widths[c]) for c in columns)
    sep = "  ".join("-" * widths[c] for c in columns)
    body = "\n".join("  ".join(r[c].ljust(widths[c]) for c in columns) for r in string_rows)
    return f"{header}\n{sep}\n{body}"


# ---------------------------------------------------------------------------
# subcommand: list


def cmd_list(args: argparse.Namespace) -> int:
    dtype = DTYPE_MAP[args.dtype]
    presets = registered_presets()
    if not presets:
        print("no presets registered", file=sys.stderr)
        return 1

    rows: list[dict[str, Any]] = []
    details: dict[str, PresetMetrics] = {}
    for name in sorted(presets):
        model = presets[name](
            n_freq=args.n_freq,
            n_src=args.n_src,
            n_chan=args.n_chan,
            masking=args.masking,
        ).eval()
        m = compute_metrics(model, name=name, dtype=dtype)
        details[name] = m
        rows.append({
            "preset": name,
            "channels": model.channels,
            "pyr_ch": model.pyramid_channels,
            "stages": model.num_stages,
            "Kt": model.time_kernel,
            "Kf": model.freq_kernel,
            "attn": "on" if model.use_attn else "off",
            "pooled_h": model.pooled_mixer_hidden,
            "params": f"{m.params:,}",
            f"state_kib_{args.dtype}": round(m.state_kib, 2),
            "budget_ok": "yes" if m.state_bytes <= args.budget_bytes else "NO",
        })

    if args.json:
        print(json.dumps({
            name: {
                "params": details[name].params,
                "state_bytes": details[name].state_bytes,
                "state_kib": round(details[name].state_kib, 3),
                "concat_width": details[name].concat_width,
                "out_widths": list(details[name].out_widths),
                "n_freq_padded": details[name].n_freq_padded,
            } for name in sorted(details)
        }, indent=2))
        return 0

    columns = [
        "preset", "channels", "pyr_ch", "stages", "Kt", "Kf",
        "attn", "pooled_h", "params", f"state_kib_{args.dtype}", "budget_ok",
    ]
    print(format_table(rows, columns))
    budget_kib = args.budget_bytes / 1024.0
    print(
        f"\nn_freq={args.n_freq}  n_src={args.n_src}  n_chan={args.n_chan}  "
        f"dtype={args.dtype}  budget={budget_kib:.1f} KiB"
    )
    return 0


# ---------------------------------------------------------------------------
# subcommand: show


def cmd_show(args: argparse.Namespace) -> int:
    dtype = DTYPE_MAP[args.dtype]
    presets = registered_presets()
    if args.name not in presets:
        available = ", ".join(sorted(presets))
        print(
            f"Unknown preset {args.name!r}. Available: {available}",
            file=sys.stderr,
        )
        return 2

    kwargs, model = reconstruct_preset_kwargs(
        presets[args.name],
        n_freq=args.n_freq,
        n_src=args.n_src,
        n_chan=args.n_chan,
        masking=args.masking,
    )
    model.eval()
    m = compute_metrics(model, name=args.name, dtype=dtype)

    report = {
        "preset": args.name,
        "constructor_kwargs": kwargs,
        "metrics": {
            "params": m.params,
            "state_bytes": m.state_bytes,
            "state_kib": round(m.state_kib, 3),
            "dtype": args.dtype,
            "n_freq": m.n_freq,
            "n_freq_padded": m.n_freq_padded,
            "concat_width": m.concat_width,
            "out_widths": list(m.out_widths),
        },
        "budget": {
            "limit_bytes": args.budget_bytes,
            "limit_kib": round(args.budget_bytes / 1024.0, 3),
            "within_budget": m.state_bytes <= args.budget_bytes,
        },
    }

    if args.json:
        print(json.dumps(report, indent=2, default=str))
        return 0

    print(f"preset : {args.name}")
    print(f"params : {m.params:,}")
    print(
        f"state  : {m.state_kib:.2f} KiB ({args.dtype})  "
        f"budget {'OK' if m.state_bytes <= args.budget_bytes else 'EXCEEDED'} "
        f"vs {args.budget_bytes / 1024.0:.1f} KiB"
    )
    print(
        f"freq   : n_freq={m.n_freq}  padded={m.n_freq_padded}  "
        f"concat_width={m.concat_width}  out_widths={m.out_widths}"
    )
    print()
    print("constructor kwargs:")
    for key in sorted(kwargs):
        print(f"  {key} = {kwargs[key]!r}")
    return 0 if m.state_bytes <= args.budget_bytes else 3


# ---------------------------------------------------------------------------
# subcommand: estimate


def _parse_int_list(text: str) -> list[int]:
    return [int(s) for s in text.replace(";", ",").split(",") if s.strip()]


def _parse_int_list_opt(text: str | None) -> list[int] | None:
    if text is None:
        return None
    return _parse_int_list(text)


def _build_estimate_model(args: argparse.Namespace, **overrides: Any) -> BandSCNetNPU:
    kwargs: dict[str, Any] = dict(
        n_freq=args.n_freq,
        n_src=args.n_src,
        n_chan=args.n_chan,
        channels=args.channels,
        num_stages=args.num_stages,
        time_kernel=args.time_kernel,
        freq_kernel=args.freq_kernel,
        use_attn=args.use_attn,
        attn_window=args.attn_window,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        pooled_mixer_hidden=args.pooled_mixer_hidden,
        masking=args.masking,
    )
    if args.pyramid_channels is not None:
        kwargs["pyramid_channels"] = args.pyramid_channels
    if args.pyramid_time_kernel is not None:
        kwargs["pyramid_time_kernel"] = args.pyramid_time_kernel
    if args.pyramid_freq_kernel is not None:
        kwargs["pyramid_freq_kernel"] = args.pyramid_freq_kernel
    if args.pyramid_conv_blocks is not None:
        cb = _parse_int_list(args.pyramid_conv_blocks)
        if len(cb) != 3:
            raise ValueError(f"--pyramid-conv-blocks expects 3 ints, got {cb}")
        kwargs["pyramid_conv_blocks"] = tuple(cb)
    if args.pyramid_strides is not None:
        st = _parse_int_list(args.pyramid_strides)
        if len(st) != 3:
            raise ValueError(f"--pyramid-strides expects 3 ints, got {st}")
        kwargs["pyramid_strides"] = tuple(st)

    kwargs.update(overrides)
    return BandSCNetNPU(**kwargs).eval()


def cmd_estimate(args: argparse.Namespace) -> int:
    dtype = DTYPE_MAP[args.dtype]

    sweep_channels = _parse_int_list_opt(args.sweep_channels)
    sweep_stages = _parse_int_list_opt(args.sweep_stages)
    sweep_time_kernel = _parse_int_list_opt(args.sweep_time_kernel)
    sweep_pooled = _parse_int_list_opt(args.sweep_pooled)

    # Sweep axes default to a single value so the grid always has one row.
    channels_axis = sweep_channels if sweep_channels else [args.channels]
    stages_axis = sweep_stages if sweep_stages else [args.num_stages]
    time_kernel_axis = sweep_time_kernel if sweep_time_kernel else [args.time_kernel]
    pooled_axis = sweep_pooled if sweep_pooled else [args.pooled_mixer_hidden]

    rows: list[dict[str, Any]] = []
    for ch in channels_axis:
        for ns in stages_axis:
            for kt in time_kernel_axis:
                for ph in pooled_axis:
                    try:
                        model = _build_estimate_model(
                            args,
                            channels=ch,
                            num_stages=ns,
                            time_kernel=kt,
                            pooled_mixer_hidden=ph,
                        )
                    except Exception as exc:  # noqa: BLE001
                        rows.append({
                            "channels": ch,
                            "stages": ns,
                            "Kt": kt,
                            "pooled_h": ph,
                            "params": "-",
                            f"state_kib_{args.dtype}": "-",
                            "budget_ok": "ERR",
                            "note": str(exc),
                        })
                        continue
                    m = compute_metrics(model, name="<estimate>", dtype=dtype)
                    rows.append({
                        "channels": ch,
                        "stages": ns,
                        "Kt": kt,
                        "pooled_h": ph,
                        "params": f"{m.params:,}",
                        f"state_kib_{args.dtype}": round(m.state_kib, 2),
                        "budget_ok": "yes" if m.state_bytes <= args.budget_bytes else "NO",
                        "note": "",
                    })

    # Single-point mode: also print constructor kwargs so the user can copy
    # them into a new factory in presets.py.
    single_point = len(rows) == 1
    if args.json:
        payload: dict[str, Any] = {
            "n_freq": args.n_freq,
            "dtype": args.dtype,
            "budget_bytes": args.budget_bytes,
            "rows": rows,
        }
        if single_point:
            payload["suggested_factory_body"] = _suggested_factory_body(args)
        print(json.dumps(payload, indent=2))
        return 0

    columns = [
        "channels", "stages", "Kt", "pooled_h",
        "params", f"state_kib_{args.dtype}", "budget_ok", "note",
    ]
    print(format_table(rows, columns))
    budget_kib = args.budget_bytes / 1024.0
    print(
        f"\nn_freq={args.n_freq}  n_src={args.n_src}  n_chan={args.n_chan}  "
        f"dtype={args.dtype}  budget={budget_kib:.1f} KiB"
    )
    if single_point:
        print()
        print("Equivalent factory body (drop into BandSCNetNPU/presets.py):")
        print(_suggested_factory_body(args))
    return 0


def _suggested_factory_body(args: argparse.Namespace) -> str:
    """Generate a copy-pastable factory function body for ``presets.py``.

    Emits only the architecture-level kwargs; leaves ``n_freq``, ``n_src``,
    ``n_chan`` and ``masking`` to be forwarded by the caller of the factory
    (matching the existing presets).
    """
    parts: list[str] = []
    parts.append("def my_preset(")
    parts.append("    n_freq: int,")
    parts.append("    *,")
    parts.append("    n_src: int = 3,")
    parts.append("    n_chan: int = 1,")
    parts.append("    masking: bool = True,")
    parts.append(") -> BandSCNetNPU:")
    parts.append("    return BandSCNetNPU(")
    parts.append("        n_freq=n_freq,")
    parts.append("        n_src=n_src,")
    parts.append("        n_chan=n_chan,")
    parts.append(f"        channels={args.channels},")
    if args.pyramid_channels is not None:
        parts.append(f"        pyramid_channels={args.pyramid_channels},")
    parts.append(f"        num_stages={args.num_stages},")
    parts.append(f"        time_kernel={args.time_kernel},")
    parts.append(f"        freq_kernel={args.freq_kernel},")
    if args.pyramid_time_kernel is not None:
        parts.append(f"        pyramid_time_kernel={args.pyramid_time_kernel},")
    if args.pyramid_freq_kernel is not None:
        parts.append(f"        pyramid_freq_kernel={args.pyramid_freq_kernel},")
    if args.pyramid_conv_blocks is not None:
        cb = _parse_int_list(args.pyramid_conv_blocks)
        parts.append(f"        pyramid_conv_blocks={tuple(cb)},")
    if args.pyramid_strides is not None:
        st = _parse_int_list(args.pyramid_strides)
        parts.append(f"        pyramid_strides={tuple(st)},")
    parts.append(f"        use_attn={args.use_attn},")
    if args.use_attn:
        parts.append(f"        attn_window={args.attn_window},")
        parts.append(f"        num_heads={args.num_heads},")
        parts.append(f"        head_dim={args.head_dim},")
    if args.pooled_mixer_hidden:
        parts.append(f"        pooled_mixer_hidden={args.pooled_mixer_hidden},")
    parts.append("        masking=masking,")
    parts.append("    )")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# argument parsing


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--n-freq", type=int, default=2049, help="STFT frequency bins (default 2049)")
    p.add_argument("--n-src", type=int, default=3)
    p.add_argument("--n-chan", type=int, default=1)
    p.add_argument(
        "--dtype",
        default="float16",
        choices=sorted(DTYPE_MAP),
        help="dtype used for state-size accounting (fp16 matches DSP quota)",
    )
    p.add_argument(
        "--budget-bytes",
        type=int,
        default=192 * 1024,
        help="streaming-state budget in bytes (default 192 KiB = 196608)",
    )
    p.add_argument(
        "--masking",
        dest="masking",
        action="store_true",
        default=True,
        help="build the model with source-gain masking enabled (default)",
    )
    p.add_argument(
        "--no-masking",
        dest="masking",
        action="store_false",
        help="build the model in logits-only mode (training loss mode)",
    )
    p.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of a text table")


def _add_estimate_args(p: argparse.ArgumentParser) -> None:
    # Single-point knobs (map 1:1 to BandSCNetNPU.__init__).
    p.add_argument("--channels", type=int, default=40)
    p.add_argument("--pyramid-channels", type=int, default=None)
    p.add_argument("--num-stages", type=int, default=3)
    p.add_argument("--time-kernel", type=int, default=3)
    p.add_argument("--freq-kernel", type=int, default=3)
    p.add_argument("--pyramid-time-kernel", type=int, default=None)
    p.add_argument("--pyramid-freq-kernel", type=int, default=None)
    p.add_argument(
        "--pyramid-conv-blocks",
        type=str,
        default=None,
        help="comma-separated 3-tuple, e.g. '1,1,1' (low,mid,high)",
    )
    p.add_argument(
        "--pyramid-strides",
        type=str,
        default=None,
        help="comma-separated 3-tuple, e.g. '2,2,4' (low,mid,high)",
    )
    p.add_argument("--use-attn", action="store_true")
    p.add_argument("--no-attn", dest="use_attn", action="store_false")
    p.set_defaults(use_attn=True)
    p.add_argument("--attn-window", type=int, default=16)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--head-dim", type=int, default=8)
    p.add_argument("--pooled-mixer-hidden", type=int, default=0)

    # Sweep axes; each is a comma-separated list that overrides the single
    # knob when provided.
    p.add_argument(
        "--sweep-channels",
        type=str,
        default=None,
        help="comma-separated list of channels to sweep, e.g. '16,32,40,56'",
    )
    p.add_argument(
        "--sweep-stages",
        type=str,
        default=None,
        help="comma-separated list of num_stages to sweep",
    )
    p.add_argument(
        "--sweep-time-kernel",
        type=str,
        default=None,
        help="comma-separated list of time_kernel values to sweep",
    )
    p.add_argument(
        "--sweep-pooled",
        type=str,
        default=None,
        help="comma-separated list of pooled_mixer_hidden values to sweep",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="list every registered preset")
    _add_common_args(p_list)
    p_list.set_defaults(func=cmd_list)

    p_show = sub.add_parser("show", help="show the full kwargs of one preset")
    p_show.add_argument("name", help="preset name (see `list` for choices)")
    _add_common_args(p_show)
    p_show.set_defaults(func=cmd_show)

    p_est = sub.add_parser(
        "estimate",
        help="estimate params + state size for an ad-hoc preset / sweep grid",
    )
    _add_common_args(p_est)
    _add_estimate_args(p_est)
    p_est.set_defaults(func=cmd_estimate)

    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
