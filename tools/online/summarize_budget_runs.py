#!/usr/bin/env python3

from __future__ import annotations

import csv
from argparse import ArgumentParser
from pathlib import Path
import sys
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spectral_feature_compression.core.model.frequency_preprocessing import resolve_preprocessed_n_freq
from spectral_feature_compression.core.model.online_hard_band_sfc_2d import OnlineHardBandSFC2D
from spectral_feature_compression.core.model.online_crossattn_query_sfc_2d import OnlineCrossAttnQuerySFC2D
from spectral_feature_compression.core.model.online_hierarchical_soft_band_ffi_sfc_2d import (
    OnlineHierarchicalSoftBandFFISFC2D,
)
from spectral_feature_compression.core.model.online_hierarchical_soft_band_parallel_ffi_sfc_2d import (
    OnlineHierarchicalSoftBandParallelFFISFC2D,
)
from spectral_feature_compression.core.model.online_hierarchical_soft_band_sfc_2d import OnlineHierarchicalSoftBandSFC2D
from spectral_feature_compression.core.model.online_sfc_2d import OnlineSFC2D
from spectral_feature_compression.core.model.online_soft_band_dilated_sfc_2d import OnlineSoftBandDilatedSFC2D
from spectral_feature_compression.core.model.online_soft_band_gru_sfc_2d import OnlineSoftBandGRUSFC2D
from spectral_feature_compression.core.model.online_soft_band_query_sfc_2d import OnlineSoftBandQuerySFC2D
from spectral_feature_compression.core.model.online_soft_band_sfc_2d import OnlineSoftBandSFC2D


FIELDNAMES = [
    "recipe",
    "family",
    "prior",
    "budget_class",
    "d_model",
    "n_layers",
    "n_bands",
    "context_frames",
    "layer_cache_fp16_kib",
    "train_wallclock_per_epoch",
    "best_val_loss",
    "best_checkpoint",
    "separation_metric_main",
    "separation_metric_notes",
    "onnx_export_ok",
    "onnx_runtime_latency_ms",
    "streaming_equivalence_ok",
    "notes",
]


def parse_args() -> Any:
    parser = ArgumentParser()
    parser.add_argument(
        "--recipe-root",
        type=Path,
        default=REPO_ROOT / "recipes" / "musdb18hq" / "models",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=REPO_ROOT / "docs" / "templates" / "online_budget_results.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "docs" / "templates" / "online_budget_results.summary.csv",
    )
    parser.add_argument(
        "--include-nonbudget",
        action="store_true",
        help="Include online recipe directories outside the rt128k/rt192k budget sets.",
    )
    return parser.parse_args()


def load_template_rows(template_path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    if not template_path.exists():
        return rows
    with template_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            recipe = row.get("recipe", "").strip()
            if recipe:
                rows[recipe] = {field: row.get(field, "") for field in FIELDNAMES}
    return rows


def discover_recipe_dirs(root: Path, include_nonbudget: bool) -> list[Path]:
    pattern = "online*"
    candidates = [p for p in root.glob(pattern) if p.is_dir()]
    selected: list[Path] = []
    for path in sorted(candidates):
        name = path.name
        if not (
            name.startswith("online-sfc2d")
            or name.startswith("online-soft-band-sfc2d")
            or name.startswith("online-soft-band-query-sfc2d")
            or name.startswith("online-crossattn-query-sfc2d")
            or name.startswith("online-soft-band-gru-sfc2d")
            or name.startswith("online-soft-band-dilated-sfc2d")
            or name.startswith("online-hierarchical-soft-band-sfc2d")
            or name.startswith("online-hierarchical-soft-band-ffi-sfc2d")
            or name.startswith("online-hierarchical-soft-band-parallel-ffi-sfc2d")
            or name.startswith("online-hard-band-sfc2d")
        ):
            continue
        if include_nonbudget or (".rt192k." in name or ".rt128k." in name):
            selected.append(path)
    return selected


def load_recipe_config(recipe_dir: Path) -> dict[str, Any]:
    config_path = recipe_dir / "config.yaml"
    needed_keys = {
        "sr",
        "n_fft",
        "n_src",
        "n_chan",
        "online_d_model",
        "online_n_layers",
        "online_n_bands",
        "online_kernel_t",
        "online_kernel_f",
        "online_causal",
        "online_masking",
        "online_band_config",
        "online_query_type",
        "online_routing_normalization",
        "online_freq_preprocess_enabled",
        "online_freq_preprocess_keep_bins",
        "online_freq_preprocess_target_bins",
        "online_freq_preprocess_mode",
        "online_dilation_cycle",
        "online_gru_band_kernel_size",
        "online_pre_bands",
        "online_mid_bands",
        "online_bottleneck_bands",
        "online_pre_layers",
        "online_mid_layers",
        "online_bottleneck_layers",
        "online_hierarchical_prior_mode",
        "online_time_branch_kernel_sizes",
        "online_time_branch_dilations",
    }

    def merge_base_chain(path: Path) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        parsed: dict[str, Any] = {}
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            if not raw_line or raw_line.startswith(" ") or raw_line.startswith("\t"):
                continue
            if ":" not in raw_line:
                continue
            key, value = raw_line.split(":", 1)
            key = key.strip()
            parsed[key] = parse_scalar(value.strip())

        base_value = parsed.get("_base_")
        if isinstance(base_value, str) and base_value and "${" not in base_value:
            base_path = (path.parent / base_value).resolve()
            if base_path.exists():
                merged.update(merge_base_chain(base_path))
        for key, value in parsed.items():
            if key == "_base_":
                continue
            merged[key] = value
        return merged

    merged = merge_base_chain(config_path)
    return {key: merged[key] for key in needed_keys if key in merged}


def parse_scalar(value: str) -> Any:
    if not value:
        return ""
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [parse_scalar(part.strip()) for part in inner.split(",")]
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def infer_family(recipe_name: str) -> str:
    if recipe_name.startswith("online-hierarchical-soft-band-parallel-ffi-sfc2d"):
        return "hierarchical-soft-parallel-ffi"
    if recipe_name.startswith("online-hierarchical-soft-band-ffi-sfc2d"):
        return "hierarchical-soft-ffi"
    if recipe_name.startswith("online-soft-band-gru-sfc2d"):
        return "soft-gru"
    if recipe_name.startswith("online-soft-band-query-sfc2d"):
        return "soft-query"
    if recipe_name.startswith("online-crossattn-query-sfc2d"):
        return "crossattn-query"
    if recipe_name.startswith("online-hierarchical-soft-band-sfc2d"):
        return "hierarchical-soft"
    if recipe_name.startswith("online-soft-band-dilated-sfc2d"):
        return "soft-dilated"
    if recipe_name.startswith("online-soft-band-sfc2d"):
        return "soft"
    if recipe_name.startswith("online-hard-band-sfc2d"):
        return "hard"
    return "plain"


def infer_budget_class(recipe_name: str) -> str:
    if ".rt192k." in recipe_name:
        return "rt192k"
    if ".rt128k." in recipe_name:
        return "rt128k"
    return "default"


def infer_prior(recipe_name: str, config: dict[str, Any]) -> str:
    if recipe_name.startswith("online-sfc2d"):
        return "none"
    return str(config.get("online_band_config", "musical"))


def build_model_from_config(recipe_name: str, config: dict[str, Any]):
    core_n_freq = resolve_preprocessed_n_freq(
        config["n_fft"] // 2 + 1,
        enabled=bool(config.get("online_freq_preprocess_enabled", False)),
        keep_bins=config.get("online_freq_preprocess_keep_bins"),
        target_bins=config.get("online_freq_preprocess_target_bins"),
    )
    common = dict(
        n_freq=core_n_freq,
        n_fft=config["n_fft"],
        sample_rate=config["sr"],
        n_src=config["n_src"],
        n_chan=config["n_chan"],
        d_model=config["online_d_model"],
        kernel_size=(config["online_kernel_t"], config["online_kernel_f"]),
        causal=config["online_causal"],
        masking=config["online_masking"],
    )
    family = infer_family(recipe_name)
    if family == "plain":
        return OnlineSFC2D(**common, n_bands=config["online_n_bands"], n_layers=config["online_n_layers"])
    if family == "soft-gru":
        return OnlineSoftBandGRUSFC2D(
            **common,
            n_bands=config["online_n_bands"],
            n_layers=config["online_n_layers"],
            band_config=config["online_band_config"],
            routing_normalization=config.get("online_routing_normalization", "softmax"),
            gru_band_kernel_size=config.get("online_gru_band_kernel_size", 3),
        )
    if family == "crossattn-query":
        return OnlineCrossAttnQuerySFC2D(
            **common,
            n_bands=config["online_n_bands"],
            n_layers=config["online_n_layers"],
            band_config=config["online_band_config"],
            query_type=config.get("online_query_type", "adaptive"),
            routing_normalization=config.get("online_routing_normalization", "softmax"),
        )
    if family == "soft-query":
        return OnlineSoftBandQuerySFC2D(
            **common,
            n_bands=config["online_n_bands"],
            n_layers=config["online_n_layers"],
            band_config=config["online_band_config"],
            routing_normalization=config.get("online_routing_normalization", "softmax"),
        )
    if family == "hierarchical-soft":
        return OnlineHierarchicalSoftBandSFC2D(
            **common,
            pre_bands=config["online_pre_bands"],
            mid_bands=config["online_mid_bands"],
            bottleneck_bands=config["online_bottleneck_bands"],
            band_config=config["online_band_config"],
            pre_layers=config["online_pre_layers"],
            mid_layers=config["online_mid_layers"],
            bottleneck_layers=config["online_bottleneck_layers"],
            routing_normalization=config.get("online_routing_normalization", "softmax"),
            dilation_cycle=config.get("online_dilation_cycle"),
            hierarchical_prior_mode=config.get("online_hierarchical_prior_mode", "inherited"),
        )
    if family == "hierarchical-soft-ffi":
        return OnlineHierarchicalSoftBandFFISFC2D(
            **common,
            pre_bands=config["online_pre_bands"],
            mid_bands=config["online_mid_bands"],
            bottleneck_bands=config["online_bottleneck_bands"],
            band_config=config["online_band_config"],
            pre_layers=config["online_pre_layers"],
            mid_layers=config["online_mid_layers"],
            bottleneck_layers=config["online_bottleneck_layers"],
            routing_normalization=config.get("online_routing_normalization", "softmax"),
            dilation_cycle=config.get("online_dilation_cycle"),
            hierarchical_prior_mode=config.get("online_hierarchical_prior_mode", "inherited"),
        )
    if family == "hierarchical-soft-parallel-ffi":
        return OnlineHierarchicalSoftBandParallelFFISFC2D(
            **common,
            pre_bands=config["online_pre_bands"],
            mid_bands=config["online_mid_bands"],
            bottleneck_bands=config["online_bottleneck_bands"],
            band_config=config["online_band_config"],
            pre_layers=config["online_pre_layers"],
            mid_layers=config["online_mid_layers"],
            bottleneck_layers=config["online_bottleneck_layers"],
            routing_normalization=config.get("online_routing_normalization", "softmax"),
            hierarchical_prior_mode=config.get("online_hierarchical_prior_mode", "inherited"),
            time_branch_kernel_sizes=config.get("online_time_branch_kernel_sizes", [3, 3]),
            time_branch_dilations=config.get("online_time_branch_dilations", [1, 6]),
        )
    if family == "soft-dilated":
        return OnlineSoftBandDilatedSFC2D(
            **common,
            n_bands=config["online_n_bands"],
            n_layers=config["online_n_layers"],
            band_config=config["online_band_config"],
            routing_normalization=config.get("online_routing_normalization", "softmax"),
            dilation_cycle=config.get("online_dilation_cycle"),
        )
    if family == "soft":
        return OnlineSoftBandSFC2D(
            **common,
            n_bands=config["online_n_bands"],
            n_layers=config["online_n_layers"],
            band_config=config["online_band_config"],
            routing_normalization=config.get("online_routing_normalization", "softmax"),
        )
    return OnlineHardBandSFC2D(
        **common,
        n_bands=config["online_n_bands"],
        n_layers=config["online_n_layers"],
        band_config=config["online_band_config"],
    )


def tensor_to_float(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return f"{float(value.item()):.6f}"
        return ""
    if isinstance(value, (float, int)):
        return f"{float(value):.6f}"
    return ""


def scan_best_checkpoint(recipe_dir: Path) -> tuple[str, str]:
    checkpoints_dir = recipe_dir / "checkpoints"
    if not checkpoints_dir.exists():
        return "", ""

    last_ckpt = checkpoints_dir / "last.ckpt"
    best_path = ""
    best_score = ""

    candidate_ckpts: list[Path] = []
    if last_ckpt.exists():
        candidate_ckpts.append(last_ckpt)
    candidate_ckpts.extend(sorted(p for p in checkpoints_dir.glob("*.ckpt") if p.name != "last.ckpt"))

    for ckpt_path in candidate_ckpts:
        try:
            state = torch.load(ckpt_path, map_location="cpu")
        except Exception:
            continue
        callbacks = state.get("callbacks", {})
        for callback_state in callbacks.values():
            if not isinstance(callback_state, dict):
                continue
            if "best_model_path" in callback_state:
                best_model_path = str(callback_state.get("best_model_path", "")).strip()
                if best_model_path:
                    best_path = best_model_path
                    best_score = tensor_to_float(callback_state.get("best_model_score"))
                    return best_path, best_score

    non_last = sorted(p for p in checkpoints_dir.glob("*.ckpt") if p.name != "last.ckpt")
    if non_last:
        return str(non_last[-1]), ""
    if last_ckpt.exists():
        return str(last_ckpt), ""
    return "", ""


def scan_onnx(recipe_dir: Path) -> str:
    onnx_files = sorted(recipe_dir.rglob("*.onnx"))
    return "yes" if onnx_files else ""


def summarize_recipe(recipe_dir: Path, template_rows: dict[str, dict[str, str]]) -> dict[str, str]:
    recipe_name = recipe_dir.name
    row = {field: "" for field in FIELDNAMES}
    row.update(template_rows.get(recipe_name, {}))

    config = load_recipe_config(recipe_dir)
    model = build_model_from_config(recipe_name, config)
    best_checkpoint, best_val_loss = scan_best_checkpoint(recipe_dir)

    row["recipe"] = recipe_name
    row["family"] = infer_family(recipe_name)
    row["prior"] = infer_prior(recipe_name, config)
    row["budget_class"] = infer_budget_class(recipe_name)
    row["d_model"] = str(config["online_d_model"])
    if row["family"] in {"hierarchical-soft", "hierarchical-soft-ffi", "hierarchical-soft-parallel-ffi"}:
        row["n_layers"] = f"{config['online_pre_layers']}/{config['online_mid_layers']}/{config['online_bottleneck_layers']}"
        row["n_bands"] = f"{config['online_pre_bands']}/{config['online_mid_bands']}/{config['online_bottleneck_bands']}"
    else:
        row["n_layers"] = str(config["online_n_layers"])
        row["n_bands"] = str(config["online_n_bands"])
    row["context_frames"] = str(model.stream_context_frames())
    row["layer_cache_fp16_kib"] = f"{model.state_size_bytes(dtype=torch.float16, mode='layer_cache') / 1024.0:.2f}"
    row["best_checkpoint"] = best_checkpoint
    row["best_val_loss"] = best_val_loss
    row["onnx_export_ok"] = scan_onnx(recipe_dir)

    merged_config = recipe_dir / "merged_config.yaml"
    notes = row.get("notes", "")
    if merged_config.exists() and "merged_config.yaml present" not in notes:
        notes = (notes + "; " if notes else "") + "merged_config.yaml present"
    row["notes"] = notes
    return row


def write_rows(output_path: Path, rows: list[dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    template_rows = load_template_rows(args.template)
    recipe_dirs = discover_recipe_dirs(args.recipe_root, args.include_nonbudget)
    rows = [summarize_recipe(recipe_dir, template_rows) for recipe_dir in recipe_dirs]
    write_rows(args.output, rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
