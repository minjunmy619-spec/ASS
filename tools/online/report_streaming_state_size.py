#!/usr/bin/env python3

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spectral_feature_compression.core.model.online_hard_band_sfc_2d import OnlineHardBandSFC2D
from spectral_feature_compression.core.model.online_crossattn_query_sfc_2d import OnlineCrossAttnQuerySFC2D
from spectral_feature_compression.core.model.frequency_preprocessing import resolve_preprocessed_n_freq
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
from spectral_feature_compression.utils.onnx_streaming import (
    collect_external_constant_bindings,
    get_external_constant_tensors,
)


def build_model(args):
    core_n_freq = resolve_preprocessed_n_freq(
        args.n_freq,
        enabled=args.freq_preprocess_enabled,
        keep_bins=args.freq_preprocess_keep_bins,
        target_bins=args.freq_preprocess_target_bins,
    )
    common = dict(
        n_freq=core_n_freq,
        n_fft=args.n_fft,
        sample_rate=args.sample_rate,
        band_config=args.band_config,
        n_src=args.n_src,
        n_chan=args.n_chan,
        d_model=args.d_model,
        kernel_size=(args.kernel_t, args.kernel_f),
        causal=True,
        masking=True,
    )
    if args.variant == "plain":
        return OnlineSFC2D(**common, n_bands=args.n_bands, n_layers=args.n_layers)
    if args.variant == "soft":
        return OnlineSoftBandSFC2D(
            **common,
            n_bands=args.n_bands,
            n_layers=args.n_layers,
            routing_normalization=args.routing_normalization,
        )
    if args.variant == "soft_query":
        return OnlineSoftBandQuerySFC2D(
            **common,
            n_bands=args.n_bands,
            n_layers=args.n_layers,
            routing_normalization=args.routing_normalization,
        )
    if args.variant == "crossattn_query":
        return OnlineCrossAttnQuerySFC2D(
            **common,
            n_bands=args.n_bands,
            n_layers=args.n_layers,
            query_type=args.query_type,
            routing_normalization=args.routing_normalization,
        )
    if args.variant == "soft_gru":
        return OnlineSoftBandGRUSFC2D(
            **common,
            n_bands=args.n_bands,
            n_layers=args.n_layers,
            routing_normalization=args.routing_normalization,
            gru_band_kernel_size=args.gru_band_kernel_size,
        )
    if args.variant == "soft_dilated":
        return OnlineSoftBandDilatedSFC2D(
            **common,
            n_bands=args.n_bands,
            n_layers=args.n_layers,
            routing_normalization=args.routing_normalization,
            dilation_cycle=tuple(args.dilation_cycle),
        )
    if args.variant == "hierarchical_soft":
        return OnlineHierarchicalSoftBandSFC2D(
            **common,
            pre_bands=args.pre_bands,
            mid_bands=args.mid_bands,
            bottleneck_bands=args.bottleneck_bands,
            pre_layers=args.pre_layers,
            mid_layers=args.mid_layers,
            bottleneck_layers=args.bottleneck_layers,
            routing_normalization=args.routing_normalization,
            dilation_cycle=tuple(args.dilation_cycle),
            hierarchical_prior_mode=args.hierarchical_prior_mode,
        )
    if args.variant == "hierarchical_soft_ffi":
        return OnlineHierarchicalSoftBandFFISFC2D(
            **common,
            pre_bands=args.pre_bands,
            mid_bands=args.mid_bands,
            bottleneck_bands=args.bottleneck_bands,
            pre_layers=args.pre_layers,
            mid_layers=args.mid_layers,
            bottleneck_layers=args.bottleneck_layers,
            routing_normalization=args.routing_normalization,
            dilation_cycle=tuple(args.dilation_cycle),
            hierarchical_prior_mode=args.hierarchical_prior_mode,
        )
    if args.variant == "hierarchical_soft_parallel_ffi":
        return OnlineHierarchicalSoftBandParallelFFISFC2D(
            **common,
            pre_bands=args.pre_bands,
            mid_bands=args.mid_bands,
            bottleneck_bands=args.bottleneck_bands,
            pre_layers=args.pre_layers,
            mid_layers=args.mid_layers,
            bottleneck_layers=args.bottleneck_layers,
            routing_normalization=args.routing_normalization,
            hierarchical_prior_mode=args.hierarchical_prior_mode,
            time_branch_kernel_sizes=tuple(args.time_branch_kernel_sizes),
            time_branch_dilations=tuple(args.time_branch_dilations),
        )
    if args.variant == "hard":
        return OnlineHardBandSFC2D(**common, n_bands=args.n_bands, n_layers=args.n_layers)
    raise ValueError(args.variant)


def format_bytes(num_bytes: int) -> str:
    kib = num_bytes / 1024.0
    mib = kib / 1024.0
    if mib >= 1.0:
        return f"{num_bytes} B ({mib:.2f} MiB)"
    return f"{num_bytes} B ({kib:.2f} KiB)"


def dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype name: {name}")
    return mapping[name]


def tensor_bytes(tensors: list[torch.Tensor] | tuple[torch.Tensor, ...], dtype: torch.dtype) -> int:
    element_size = torch.tensor([], dtype=dtype).element_size()
    return sum(int(t.numel()) * element_size for t in tensors)


def module_parameter_and_buffer_bytes(module: torch.nn.Module, dtype: torch.dtype) -> int:
    element_size = torch.tensor([], dtype=dtype).element_size()
    total_numel = sum(int(p.numel()) for p in module.parameters())
    total_numel += sum(int(b.numel()) for b in module.buffers())
    return total_numel * element_size


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=[
            "plain",
            "soft",
            "soft_query",
            "crossattn_query",
            "soft_gru",
            "soft_dilated",
            "hierarchical_soft",
            "hierarchical_soft_ffi",
            "hierarchical_soft_parallel_ffi",
            "hard",
        ],
        required=True,
    )
    parser.add_argument("--n-src", type=int, default=4)
    parser.add_argument("--n-chan", type=int, default=2)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--n-freq", type=int, default=1025)
    parser.add_argument("--freq-preprocess-enabled", action="store_true")
    parser.add_argument("--freq-preprocess-keep-bins", type=int, default=None)
    parser.add_argument("--freq-preprocess-target-bins", type=int, default=None)
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--n-bands", type=int, default=64)
    parser.add_argument("--band-config", type=str, default="musical")
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--n-layers", type=int, default=12)
    parser.add_argument("--kernel-t", type=int, default=3)
    parser.add_argument("--kernel-f", type=int, default=3)
    parser.add_argument("--routing-normalization", type=str, default="softmax")
    parser.add_argument("--query-type", type=str, default="adaptive", choices=["adaptive", "learnable"])
    parser.add_argument("--gru-band-kernel-size", type=int, default=3)
    parser.add_argument("--dilation-cycle", type=int, nargs="+", default=[1, 2, 1, 2])
    parser.add_argument("--pre-bands", type=int, default=128)
    parser.add_argument("--mid-bands", type=int, default=96)
    parser.add_argument("--bottleneck-bands", type=int, default=48)
    parser.add_argument("--pre-layers", type=int, default=1)
    parser.add_argument("--mid-layers", type=int, default=2)
    parser.add_argument("--bottleneck-layers", type=int, default=2)
    parser.add_argument("--hierarchical-prior-mode", type=str, default="inherited")
    parser.add_argument("--time-branch-kernel-sizes", type=int, nargs="+", default=[3, 3])
    parser.add_argument("--time-branch-dilations", type=int, nargs="+", default=[1, 6])
    parser.add_argument("--budget-kib", type=int, default=192)
    parser.add_argument(
        "--budget-dtype",
        type=str,
        default="fp16",
        choices=["fp16", "fp32"],
        help="Dtype used for combined DSP budget estimates.",
    )
    parser.add_argument(
        "--fail-on-budget",
        action="store_true",
        help="Exit with status 2 if the selected budget checks exceed the budget.",
    )

    args = parser.parse_args()

    model = build_model(args)
    budget_bytes = args.budget_kib * 1024
    budget_dtype = dtype_from_name(args.budget_dtype)

    layer_cache_fp16 = model.state_size_bytes(dtype=torch.float16, mode="layer_cache")
    layer_cache_fp32 = model.state_size_bytes(dtype=torch.float32, mode="layer_cache")
    input_history_fp16 = model.state_size_bytes(dtype=torch.float16, mode="input_history")
    input_history_fp32 = model.state_size_bytes(dtype=torch.float32, mode="input_history")
    externalized_constant_tensors = get_external_constant_tensors(model, collect_external_constant_bindings(model))
    externalized_constants_fp16 = tensor_bytes(externalized_constant_tensors, torch.float16)
    externalized_constants_fp32 = tensor_bytes(externalized_constant_tensors, torch.float32)
    model_params_and_buffers_fp16 = module_parameter_and_buffer_bytes(model, torch.float16)
    model_params_and_buffers_fp32 = module_parameter_and_buffer_bytes(model, torch.float32)
    layer_cache_budget_dtype = model.state_size_bytes(dtype=budget_dtype, mode="layer_cache")
    state_plus_externalized_constants = layer_cache_budget_dtype + tensor_bytes(externalized_constant_tensors, budget_dtype)
    state_plus_all_model_tensors = layer_cache_budget_dtype + module_parameter_and_buffer_bytes(model, budget_dtype)

    print(f"Variant: {args.variant}")
    print(f"Context frames: {model.stream_context_frames()}")
    print(f"Layer-cache fp16:   {format_bytes(layer_cache_fp16)}")
    print(f"Layer-cache fp32:   {format_bytes(layer_cache_fp32)}")
    print(f"Input-history fp16: {format_bytes(input_history_fp16)}")
    print(f"Input-history fp32: {format_bytes(input_history_fp32)}")
    print(f"Band/basis constants fp16: {format_bytes(externalized_constants_fp16)}")
    print(f"Band/basis constants fp32: {format_bytes(externalized_constants_fp32)}")
    print(f"All model params+buffers fp16: {format_bytes(model_params_and_buffers_fp16)}")
    print(f"All model params+buffers fp32: {format_bytes(model_params_and_buffers_fp32)}")
    print(f"Budget:             {format_bytes(budget_bytes)}")
    print(f"Layer-cache fp16 within budget:   {layer_cache_fp16 <= budget_bytes}")
    print(f"Input-history fp16 within budget: {input_history_fp16 <= budget_bytes}")
    print(
        f"State + band/basis constants ({args.budget_dtype}) within budget: "
        f"{state_plus_externalized_constants <= budget_bytes}"
    )
    print(
        f"State + all model params+buffers ({args.budget_dtype}) within budget: "
        f"{state_plus_all_model_tensors <= budget_bytes}"
    )
    print(
        f"State + band/basis constants ({args.budget_dtype}): "
        f"{format_bytes(state_plus_externalized_constants)}"
    )
    print(
        f"State + all model params+buffers ({args.budget_dtype}): "
        f"{format_bytes(state_plus_all_model_tensors)}"
    )

    if args.fail_on_budget and (
        layer_cache_fp16 > budget_bytes
        or input_history_fp16 > budget_bytes
        or state_plus_externalized_constants > budget_bytes
        or state_plus_all_model_tensors > budget_bytes
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
