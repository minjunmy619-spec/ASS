#!/usr/bin/env python3

from __future__ import annotations

from argparse import ArgumentParser
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
import re
import sys
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

AIACCEL_ROOT = REPO_ROOT / "aiaccel"
if str(AIACCEL_ROOT) not in sys.path:
    sys.path.insert(0, str(AIACCEL_ROOT))

BASE_KEY = "_base_"
INTERPOLATION_RE = re.compile(r"\$\{([^}]+)\}")


def build_context(config_path: Path) -> dict[str, str]:
    """Return the same path variables used by the repo's aiaccel recipes."""
    resolved_config_path = config_path.resolve()
    return {
        "config_path": str(resolved_config_path),
        "working_directory": str(resolved_config_path.parent),
        "base_config_path": str(REPO_ROOT / "aiaccel" / "aiaccel" / "torch" / "apps" / "config"),
    }


def strip_base_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: strip_base_keys(item) for key, item in value.items() if key != BASE_KEY}
    if isinstance(value, list):
        return [strip_base_keys(item) for item in value]
    return value


def load_with_aiaccel(config_path: Path) -> dict[str, Any]:
    from aiaccel.config import load_config, resolve_inherit
    from omegaconf import OmegaConf

    config = load_config(config_path, build_context(config_path))
    config = resolve_inherit(config)
    data = OmegaConf.to_container(config, resolve=False)
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping config, got {type(data).__name__}: {config_path}")
    return strip_base_keys(data)


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping config, got {type(data).__name__}: {path}")
    return data


def interpolate_path(value: str, context: Mapping[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in context:
            raise KeyError(f"Unknown interpolation in base path: {match.group(0)}")
        return context[key]

    return INTERPOLATION_RE.sub(replace, value)


def resolve_base_path(base_ref: str, config_path: Path, context: Mapping[str, str]) -> Path:
    expanded = interpolate_path(base_ref, context)
    base_path = Path(expanded)
    if not base_path.is_absolute():
        base_path = config_path.parent / base_path
    return base_path.resolve()


def iter_base_refs(raw_base: Any) -> list[str]:
    if raw_base is None:
        return []
    if isinstance(raw_base, str):
        return [raw_base]
    if isinstance(raw_base, list):
        if not all(isinstance(item, str) for item in raw_base):
            raise TypeError(f"{BASE_KEY} list must contain only strings: {raw_base!r}")
        return raw_base
    raise TypeError(f"{BASE_KEY} must be a string or list of strings, got {type(raw_base).__name__}")


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in override.items():
        if key == BASE_KEY:
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_with_yaml(config_path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    config_path = config_path.resolve()
    if config_path in stack:
        chain = " -> ".join(str(path) for path in (*stack, config_path))
        raise ValueError(f"Cyclic {BASE_KEY} chain detected: {chain}")
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    config = load_yaml_mapping(config_path)
    context = build_context(config_path)
    merged: dict[str, Any] = {}
    for base_ref in iter_base_refs(config.get(BASE_KEY)):
        base_path = resolve_base_path(base_ref, config_path, context)
        merged = deep_merge(merged, load_with_yaml(base_path, (*stack, config_path)))
    return deep_merge(merged, config)


def lookup(root: Any, expression: str) -> Any:
    current = root
    for part in expression.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
            continue
        raise KeyError(expression)
    return current


def resolve_interpolations(value: Any, root: Mapping[str, Any], *, strict: bool, stack: tuple[str, ...] = ()) -> Any:
    if isinstance(value, dict):
        return {key: resolve_interpolations(item, root, strict=strict, stack=(*stack, str(key))) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_interpolations(item, root, strict=strict, stack=(*stack, str(index))) for index, item in enumerate(value)]
    if not isinstance(value, str) or "${" not in value:
        return value

    matches = list(INTERPOLATION_RE.finditer(value))
    if not matches:
        return value

    if len(matches) == 1 and matches[0].span() == (0, len(value)):
        expression = matches[0].group(1)
        if expression in stack:
            raise ValueError(f"Cyclic interpolation detected: {' -> '.join((*stack, expression))}")
        try:
            resolved = lookup(root, expression)
        except KeyError:
            if strict:
                raise
            return value
        return resolve_interpolations(resolved, root, strict=strict, stack=(*stack, expression))

    def replace(match: re.Match[str]) -> str:
        expression = match.group(1)
        try:
            resolved = lookup(root, expression)
        except KeyError:
            if strict:
                raise
            return match.group(0)
        return str(resolve_interpolations(resolved, root, strict=strict, stack=(*stack, expression)))

    return INTERPOLATION_RE.sub(replace, value)


def load_flat_config(config_path: Path) -> dict[str, Any]:
    try:
        return load_with_aiaccel(config_path)
    except Exception as aiaccel_error:
        try:
            return strip_base_keys(load_with_yaml(config_path))
        except Exception as yaml_error:
            raise RuntimeError(
                f"Failed to expand {config_path} with aiaccel ({aiaccel_error}) "
                f"and the fallback YAML loader ({yaml_error})"
            ) from yaml_error


def default_output_path(config_path: Path) -> Path:
    return config_path.with_name(f"{config_path.stem}.standalone{config_path.suffix}")


def parse_args() -> Any:
    parser = ArgumentParser(description="Expand an aiaccel recipe config into a standalone YAML file without _base_.")
    parser.add_argument("config", type=Path, help="Input recipe config, usually recipes/.../config.yaml")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output YAML path. Defaults to <input>.standalone.yaml next to the input config.",
    )
    parser.add_argument(
        "--keep-interpolations",
        action="store_true",
        help="Keep ${...} references instead of resolving values that are available in the merged config.",
    )
    parser.add_argument(
        "--strict-interpolations",
        action="store_true",
        help="Fail if any ${...} reference cannot be resolved. By default unresolved references are kept.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    output_path = args.output.resolve() if args.output is not None else default_output_path(config_path)

    flat_config = load_flat_config(config_path)
    if not args.keep_interpolations:
        flat_config = resolve_interpolations(flat_config, flat_config, strict=args.strict_interpolations)
    flat_config = strip_base_keys(flat_config)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(flat_config, handle, sort_keys=False, allow_unicode=True)

    print(f"Wrote standalone config: {output_path}")


if __name__ == "__main__":
    main()
