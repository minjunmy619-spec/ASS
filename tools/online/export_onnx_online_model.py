#!/usr/bin/env python3

from __future__ import annotations

import copy
import importlib
import json
import re
from argparse import ArgumentParser
from pathlib import Path
import sys

import numpy as np
import onnx
from onnx import numpy_helper
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spectral_feature_compression.utils.onnx_streaming import (
    ExternalizedConstantsWrapper,
    StreamingStateIOWrapper,
    collect_external_constant_bindings,
    flatten_tensor_tree,
    get_external_constant_tensors,
    tensor_tree_shapes,
)


INTERPOLATION_RE = re.compile(r"^\$\{([^}]+)\}$")


def import_object(target: str):
    module_name, _, attr_name = target.rpartition(".")
    if not module_name:
        raise ValueError(f"Invalid import target: {target}")
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def parse_scalar(value: str):
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


def get_concrete_base_path(path: Path) -> Path | None:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line or raw_line.startswith(" ") or raw_line.startswith("\t"):
            continue
        if ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
        if key.strip() != "_base_":
            continue
        base_value = parse_scalar(value.strip())
        if isinstance(base_value, str) and base_value and "${" not in base_value:
            base_path = (path.parent / base_value).resolve()
            if base_path.exists():
                return base_path
        return None
    return None


def merge_top_level_scalars(path: Path) -> dict[str, object]:
    merged: dict[str, object] = {}
    parsed: dict[str, object] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line or raw_line.startswith(" ") or raw_line.startswith("\t"):
            continue
        if ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
        parsed[key.strip()] = parse_scalar(value.strip())

    base_path = get_concrete_base_path(path)
    if base_path is not None:
        merged.update(merge_top_level_scalars(base_path))

    for key, value in parsed.items():
        if key == "_base_":
            continue
        merged[key] = value
    return merged


def merge_task_model_mapping(path: Path) -> dict[str, object]:
    merged: dict[str, object] = {}
    base_path = get_concrete_base_path(path)
    if base_path is not None:
        merged.update(merge_task_model_mapping(base_path))

    in_task = False
    in_model = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = raw_line.strip()
        if indent == 0:
            in_task = stripped.startswith("task:")
            in_model = False
            continue
        if in_task and indent == 2 and stripped.startswith("model:"):
            in_model = True
            continue
        if in_model:
            if indent < 4:
                in_model = False
                continue
            if indent == 4 and ":" in stripped:
                key, value = stripped.split(":", 1)
                merged[key.strip()] = parse_scalar(value.strip())
    return merged


def resolve_value(value, context: dict[str, object], *, stack: tuple[str, ...] = ()):
    if isinstance(value, list):
        return [resolve_value(item, context, stack=stack) for item in value]
    if isinstance(value, str):
        match = INTERPOLATION_RE.fullmatch(value)
        if match is None:
            return value
        ref = match.group(1)
        if ref in stack:
            raise ValueError(f"Cyclic interpolation detected: {' -> '.join((*stack, ref))}")
        if ref not in context:
            raise KeyError(f"Unknown interpolation key: {ref}")
        return resolve_value(context[ref], context, stack=(*stack, ref))
    return value


def build_model_system_from_recipe_config(config_path: Path):
    top_level = merge_top_level_scalars(config_path)
    model_cfg = merge_task_model_mapping(config_path)
    if "_target_" not in model_cfg:
        raise ValueError(f"Could not find task.model._target_ in {config_path}")

    resolved_model_cfg: dict[str, object] = {}
    for key, value in model_cfg.items():
        resolution_context = {**top_level, **model_cfg, **resolved_model_cfg}
        if isinstance(value, str):
            match = INTERPOLATION_RE.fullmatch(value)
            if match is not None and match.group(1) == key and key in top_level:
                resolved_model_cfg[key] = resolve_value(top_level[key], resolution_context, stack=(key,))
                continue
        resolved_model_cfg[key] = resolve_value(value, resolution_context, stack=(key,))
    target = resolved_model_cfg.pop("_target_")
    builder = import_object(target)
    return builder(**resolved_model_cfg)


def load_trained_task(model_path: Path, device: str):
    from aiaccel.config import load_config
    from hydra.utils import instantiate

    if model_path.is_dir():
        checkpoint_path = model_path / "checkpoints"
        config_path = model_path
    elif model_path.suffix in [".ckpt", ".pth", ".pt"]:
        checkpoint_path = model_path
        config_path = checkpoint_path.parent.parent
    else:
        raise ValueError("model_path must be either a model directory or a checkpoint file")

    config = load_config(config_path / "merged_config.yaml")
    task = instantiate(config.task).to(device)
    task.load_average_checkpoint(checkpoint_path)
    task.eval()
    return task


def infer_recipe_config_path(model_path: Path) -> Path | None:
    if model_path.is_file() and model_path.name in {"config.yaml", "merged_config.yaml"}:
        return model_path
    if model_path.is_dir():
        merged_config = model_path / "merged_config.yaml"
        config_yaml = model_path / "config.yaml"
        if merged_config.exists():
            return merged_config
        if config_yaml.exists():
            return config_yaml
        return None
    if model_path.is_file() and model_path.suffix in {".ckpt", ".pth", ".pt"}:
        candidate = model_path.parent.parent / "merged_config.yaml"
        return candidate if candidate.exists() else None
    return None


def load_frequency_preprocess_metadata(model_path: Path) -> dict[str, object] | None:
    config_path = infer_recipe_config_path(model_path)
    if config_path is None:
        return None
    top_level = merge_top_level_scalars(config_path)
    if not bool(top_level.get("online_freq_preprocess_enabled", False)):
        return None
    return {
        "enabled": True,
        "keep_bins": top_level.get("online_freq_preprocess_keep_bins"),
        "target_bins": top_level.get("online_freq_preprocess_target_bins"),
        "mode": top_level.get("online_freq_preprocess_mode", "triangular"),
        "full_n_freq": int(top_level.get("n_fft", 0)) // 2 + 1 if "n_fft" in top_level else None,
    }


def get_export_core(task):
    model_wrapper = task.ema_model.module if getattr(task, "use_ema_model", False) else task.model
    online_model = model_wrapper.model
    core = getattr(online_model, "core", None)
    if core is None:
        raise ValueError(f"Expected an online wrapper with a .core module, got {type(online_model).__name__}")
    return core


def get_export_core_from_model_system(model_system):
    online_model = getattr(model_system, "model", model_system)
    core = getattr(online_model, "core", None)
    if core is None:
        raise ValueError(
            f"Expected an online wrapper/model with a .core module, got {type(model_system).__name__}"
        )
    return core


def load_export_core(model_path: Path, device: str):
    if model_path.is_file() and model_path.suffix in {".ckpt", ".pth", ".pt"}:
        return get_export_core(load_trained_task(model_path, device)), "trained_checkpoint"

    if model_path.is_dir():
        merged_config = model_path / "merged_config.yaml"
        config_yaml = model_path / "config.yaml"
        checkpoint_dir = model_path / "checkpoints"
        if merged_config.exists() and checkpoint_dir.exists():
            return get_export_core(load_trained_task(model_path, device)), "trained_directory"
        if merged_config.exists():
            model_system = build_model_system_from_recipe_config(merged_config)
            return get_export_core_from_model_system(model_system), "config_only_merged"
        if config_yaml.exists():
            model_system = build_model_system_from_recipe_config(config_yaml)
            return get_export_core_from_model_system(model_system), "config_only_recipe"
        raise ValueError(
            f"{model_path} must contain merged_config.yaml or config.yaml, and checkpoints/ for trained export."
        )

    if model_path.is_file() and model_path.name in {"config.yaml", "merged_config.yaml"}:
        model_system = build_model_system_from_recipe_config(model_path)
        mode = "config_only_merged" if model_path.name == "merged_config.yaml" else "config_only_recipe"
        return get_export_core_from_model_system(model_system), mode

    raise ValueError(
        "model_path must be a checkpoint, a model directory, or a config.yaml / merged_config.yaml file."
    )


def get_allowed_ops(preset: str) -> set[str]:
    presets = {
        "none": set(),
        "edge_npu_recommended": {
            "Add",
            "Cast",
            "Clip",
            "Concat",
            "Constant",
            "ConstantOfShape",
            "Conv",
            "ConvTranspose",
            "Div",
            "Equal",
            "Expand",
            "Gather",
            "Identity",
            "MatMul",
            "Mul",
            "Pad",
            "Range",
            "ReduceMean",
            "ReduceSum",
            "Reshape",
            "Shape",
            "Sigmoid",
            "Slice",
            "Softmax",
            "Split",
            "Sqrt",
            "Sub",
            "Tanh",
            "Tile",
            "Transpose",
            "Unsqueeze",
            "Where",
        },
    }
    if preset not in presets:
        raise ValueError(f"Unsupported op preset: {preset}")
    return presets[preset]


def audit_onnx_graph(onnx_model, *, allowed_ops: set[str]) -> dict[str, object]:
    ops = sorted({node.op_type for node in onnx_model.graph.node})
    op_counts: dict[str, int] = {}
    for node in onnx_model.graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
    initializer_bytes = 0
    for initializer in onnx_model.graph.initializer:
        initializer_bytes += numpy_helper.to_array(initializer).nbytes
    disallowed = sorted(op for op in ops if allowed_ops and op not in allowed_ops)
    return {
        "ops": ops,
        "op_counts": op_counts,
        "initializer_count": len(onnx_model.graph.initializer),
        "initializer_bytes": initializer_bytes,
        "disallowed_ops": disallowed,
    }


def tensor_numel(shape: list[int]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def tensor_collection_bytes(shapes: list[list[int]], *, element_size: int) -> int:
    return sum(tensor_numel(shape) * element_size for shape in shapes)


@torch.inference_mode()
def main():
    parser = ArgumentParser()
    parser.add_argument(
        "model_path",
        type=Path,
        help="Model directory, checkpoint file, or config.yaml / merged_config.yaml.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output ONNX path")
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--n-chan", type=int, required=True, help="Number of audio channels M")
    parser.add_argument("--frames", type=int, default=64, help="Fixed number of frames T for export")
    parser.add_argument("--freqs", type=int, default=None, help="Fixed number of frequency bins F for export")
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version")
    parser.add_argument("--check", action="store_true", help="Run onnx.checker.check_model after export")
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Export the strict online forward_stream path with flattened state inputs/outputs.",
    )
    parser.add_argument(
        "--disable-masking",
        action="store_true",
        help="Export raw predicted masks/mappings and keep packed complex multiply outside the ONNX graph.",
    )
    parser.add_argument(
        "--state-meta-out",
        type=Path,
        help="Optional JSON path to write flattened state names and shapes for streaming export.",
    )
    parser.add_argument(
        "--op-preset",
        type=str,
        default="edge_npu_recommended",
        choices=["none", "edge_npu_recommended"],
        help="Optional ONNX op allowlist preset used for post-export auditing.",
    )
    parser.add_argument(
        "--allow-op",
        action="append",
        default=[],
        help="Extra ONNX op types to allow in addition to the selected preset.",
    )
    parser.add_argument(
        "--fail-on-disallowed-ops",
        action="store_true",
        help="Exit with an error if the exported graph contains ops outside the selected allowlist.",
    )
    parser.add_argument(
        "--keep-initializers-as-inputs",
        action="store_true",
        help="Expose ONNX initializers as graph inputs as a fallback for fragile converter toolchains.",
    )
    parser.add_argument(
        "--externalize-band-constants",
        action="store_true",
        help="Expose selected band/basis constant buffers as explicit graph inputs instead of ONNX initializers.",
    )
    parser.add_argument(
        "--constants-out",
        type=Path,
        help="Optional .npz output for selected externalized band/basis constants.",
    )
    parser.add_argument(
        "--deploy-manifest-out",
        type=Path,
        help="Optional JSON manifest that describes ONNX/state/constants packaging for device integration.",
    )

    args = parser.parse_args()

    core, source_mode = load_export_core(args.model_path, args.device)
    frequency_preprocess_meta = load_frequency_preprocess_metadata(args.model_path)
    core = core.to(args.device)
    if args.disable_masking and hasattr(core, "masking"):
        core = copy.deepcopy(core)
        core.masking = False
    core.eval()

    core_freqs = getattr(core, "n_freq", None)
    export_freqs = args.freqs if args.freqs is not None else core_freqs
    if export_freqs is None:
        raise ValueError("Could not infer export frequency bins. Please pass --freqs explicitly.")
    if core_freqs is not None and int(export_freqs) != int(core_freqs):
        raise ValueError(f"Export freqs mismatch: --freqs={export_freqs}, but core expects n_freq={core_freqs}.")

    dummy = torch.randn(1, 2 * args.n_chan, args.frames, export_freqs, dtype=torch.float32, device=args.device)
    model_to_export: torch.nn.Module = core
    export_args: tuple[torch.Tensor, ...] = (dummy,)
    input_names = ["x"]
    output_names = ["y"]
    state_meta: dict[str, object] | None = None
    all_constant_bindings = collect_external_constant_bindings(core)
    all_constant_tensors = get_external_constant_tensors(core, all_constant_bindings) if all_constant_bindings else ()
    constant_bindings = all_constant_bindings if args.externalize_band_constants else ()
    constant_tensors = all_constant_tensors if args.externalize_band_constants else ()
    constant_meta = {
        "count": len(constant_bindings),
        "input_names": [f"const_{idx}_{binding.qualified_name.replace('.', '_')}" for idx, binding in enumerate(constant_bindings)],
        "qualified_names": [binding.qualified_name for binding in constant_bindings],
        "shapes": [list(t.shape) for t in constant_tensors],
    } if constant_bindings else None
    embedded_constant_meta = {
        "count": len(all_constant_bindings),
        "qualified_names": [binding.qualified_name for binding in all_constant_bindings],
        "shapes": [list(t.shape) for t in all_constant_tensors],
    } if (all_constant_bindings and not args.externalize_band_constants) else None

    if args.streaming:
        wrapper = StreamingStateIOWrapper(
            core,
            batch_size=1,
            device=dummy.device,
            dtype=dummy.dtype,
            externalize_constants=args.externalize_band_constants,
        ).to(args.device)
        wrapper.eval()
        init_state = core.init_stream_state(batch_size=1, device=dummy.device, dtype=dummy.dtype)
        flat_state, _ = flatten_tensor_tree(init_state)
        state_input_names = [f"state_{idx}" for idx in range(len(flat_state))]
        state_output_names = [f"next_state_{idx}" for idx in range(len(flat_state))]
        model_to_export = wrapper
        export_args = (dummy, *flat_state, *constant_tensors)
        input_names = [
            "x",
            *state_input_names,
            *(constant_meta["input_names"] if constant_meta is not None else []),
        ]
        output_names = ["y", *state_output_names]
        state_meta = {
            "state_count": len(flat_state),
            "input_names": state_input_names,
            "output_names": state_output_names,
            "shapes": tensor_tree_shapes(init_state),
        }
    elif args.externalize_band_constants:
        wrapper = ExternalizedConstantsWrapper(core).to(args.device)
        wrapper.eval()
        model_to_export = wrapper
        export_args = (dummy, *constant_tensors)
        input_names = ["x", *(constant_meta["input_names"] if constant_meta is not None else [])]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model_to_export,
        export_args,
        str(args.out),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=None,
        keep_initializers_as_inputs=args.keep_initializers_as_inputs,
    )

    onnx_model = onnx.load(args.out)
    if args.check:
        onnx.checker.check_model(onnx_model)
    allowed_ops = get_allowed_ops(args.op_preset).union(args.allow_op)
    audit = audit_onnx_graph(onnx_model, allowed_ops=allowed_ops)

    if args.state_meta_out is not None and (state_meta is not None or constant_meta is not None):
        args.state_meta_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {}
        if state_meta is not None:
            payload["streaming_state"] = state_meta
        if constant_meta is not None:
            payload["externalized_band_constants"] = constant_meta
        args.state_meta_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.constants_out is not None:
        if not args.externalize_band_constants:
            raise ValueError("--constants-out requires --externalize-band-constants.")
        args.constants_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.constants_out,
            **{
                input_name: tensor.detach().cpu().numpy()
                for input_name, tensor in zip(constant_meta["input_names"], constant_tensors)
            },
        )

    if args.deploy_manifest_out is not None:
        args.deploy_manifest_out.parent.mkdir(parents=True, exist_ok=True)
        state_bytes_fp16 = (
            tensor_collection_bytes(state_meta["shapes"], element_size=2) if state_meta is not None else 0
        )
        state_bytes_fp32 = (
            tensor_collection_bytes(state_meta["shapes"], element_size=4) if state_meta is not None else 0
        )
        embedded_constant_bytes_fp16 = (
            tensor_collection_bytes(embedded_constant_meta["shapes"], element_size=2)
            if embedded_constant_meta is not None
            else 0
        )
        embedded_constant_bytes_fp32 = (
            tensor_collection_bytes(embedded_constant_meta["shapes"], element_size=4)
            if embedded_constant_meta is not None
            else 0
        )
        externalized_constant_bytes_fp16 = (
            tensor_collection_bytes(constant_meta["shapes"], element_size=2)
            if args.externalize_band_constants and constant_meta is not None
            else 0
        )
        externalized_constant_bytes_fp32 = (
            tensor_collection_bytes(constant_meta["shapes"], element_size=4)
            if args.externalize_band_constants and constant_meta is not None
            else 0
        )
        manifest = {
            "source": str(args.model_path),
            "source_mode": source_mode,
            "model_file": str(args.out),
            "core_type": type(core).__name__,
            "streaming": args.streaming,
            "masking_inside_graph": not args.disable_masking,
            "keep_initializers_as_inputs": args.keep_initializers_as_inputs,
            "frames": args.frames,
            "freqs": export_freqs,
            "n_chan": args.n_chan,
            "frequency_preprocessing": frequency_preprocess_meta,
            "input_names": input_names,
            "output_names": output_names,
            "dynamic_inputs": ["x", *(state_meta["input_names"] if state_meta is not None else [])],
            "dynamic_outputs": ["y", *(state_meta["output_names"] if state_meta is not None else [])],
            "state_meta_file": str(args.state_meta_out) if args.state_meta_out is not None else None,
            "state_meta_inline": state_meta,
            "band_constants_mode": (
                "external_inputs" if args.externalize_band_constants else "embedded_initializers"
            ),
            "embedded_band_constants": embedded_constant_meta,
            "persistent_constant_inputs": constant_meta if args.externalize_band_constants else None,
            "constants_file": str(args.constants_out) if args.constants_out is not None else None,
            "memory_estimates": {
                "streaming_state_bytes_fp16": state_bytes_fp16,
                "streaming_state_bytes_fp32": state_bytes_fp32,
                "embedded_band_constants_bytes_fp16": embedded_constant_bytes_fp16,
                "embedded_band_constants_bytes_fp32": embedded_constant_bytes_fp32,
                "externalized_band_constants_bytes_fp16": externalized_constant_bytes_fp16,
                "externalized_band_constants_bytes_fp32": externalized_constant_bytes_fp32,
                "onnx_initializers_bytes": int(audit["initializer_bytes"]),
            },
            "onnx_audit": audit,
        }
        args.deploy_manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Exported: {args.out}")
    print(f"Source: {args.model_path}")
    print(f"Source mode: {source_mode}")
    print(f"Core: {type(core).__name__}")
    print(f"Opset: {args.opset}")
    print(f"Streaming export: {args.streaming}")
    print(f"Masking inside graph: {not args.disable_masking}")
    print(f"Keep initializers as inputs: {args.keep_initializers_as_inputs}")
    print(f"Externalize band constants: {args.externalize_band_constants}")
    print(f"Input shape:  (1, {2 * args.n_chan}, {args.frames}, {export_freqs})")
    if args.streaming:
        print(f"Flattened state tensors: {state_meta['state_count'] if state_meta is not None else 0}")
    if constant_meta is not None:
        print(f"Externalized constant tensors: {constant_meta['count']}")
    elif embedded_constant_meta is not None:
        print(f"Embedded band/basis tensors: {embedded_constant_meta['count']}")
    print("Output shape: (1, 2*N*M, T, F) or raw masks if masking is disabled")
    print(f"ONNX ops: {', '.join(audit['ops'])}")
    print(
        "Initializers: "
        f"{audit['initializer_count']} tensors, {int(audit['initializer_bytes'])} bytes"
    )
    if args.check:
        print("ONNX checker: passed")
    if args.op_preset != "none":
        print(f"Op preset: {args.op_preset}")
        if audit["disallowed_ops"]:
            print(f"Disallowed ops: {', '.join(audit['disallowed_ops'])}")
        else:
            print("Disallowed ops: none")
    if args.state_meta_out is not None and state_meta is not None:
        print(f"State metadata: {args.state_meta_out}")
    if args.constants_out is not None:
        print(f"Constants package: {args.constants_out}")
    if args.deploy_manifest_out is not None:
        print(f"Deploy manifest: {args.deploy_manifest_out}")

    if args.fail_on_disallowed_ops and audit["disallowed_ops"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
