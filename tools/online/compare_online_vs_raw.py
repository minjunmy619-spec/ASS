#!/usr/bin/env python3

from __future__ import annotations

from argparse import ArgumentParser

import torch

from spectral_feature_compression.core.model.bslocoformer import BSLocoformer
from spectral_feature_compression.core.model.crossattn_enc_dec import CrossAttnDecoder, CrossAttnEncoder
from spectral_feature_compression.core.model.online_wrapper import OnlineSFCModel


def count_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


@torch.inference_mode()
def estimate_flops(model: torch.nn.Module, x: torch.Tensor) -> int | None:
    """
    Returns FLOPs as reported by torch.profiler (may be None if unsupported).
    Note: torch profiler reports FLOPs (not strictly MACs). Many people use MACs ≈ FLOPs/2 for conv/linear.
    """
    try:
        from torch.profiler import ProfilerActivity, profile
    except Exception:
        return None

    try:
        with profile(activities=[ProfilerActivity.CPU], record_shapes=False, with_flops=True) as prof:
            _ = model(x)
        flops = 0
        for evt in prof.key_averages():
            # evt.flops can be None depending on op support
            if getattr(evt, "flops", None) is not None:
                flops += int(evt.flops)
        return flops if flops > 0 else None
    except Exception:
        return None


def build_raw_model(
    *,
    sr: int,
    n_fft: int,
    n_chan: int,
    n_src: int,
    emb_dim: int,
    d_inner: int,
    n_bands: int,
):
    encoder = CrossAttnEncoder(
        d_inner=d_inner,
        d_model=emb_dim,
        n_chan=n_chan,
        sample_rate=sr,
        n_fft=n_fft,
        n_bands=n_bands,
        band_config="musical",
        query_type="learnable",
        n_heads=4,
        slope=[1.0, 1.0, 1.0, 1.0],
        learnable_slope=False,
        learnable_pos_bias=True,
        mask_outside_bands=False,
        use_ffn=True,
    )
    decoder = CrossAttnDecoder(
        d_inner=d_inner,
        d_model=emb_dim,
        n_src=n_src,
        n_chan=n_chan,
        sample_rate=sr,
        n_fft=n_fft,
        n_bands=n_bands,
        band_config="musical",
        query_type="learnable",
        n_heads=4,
        slope=[1.0, 1.0, 1.0, 1.0],
        learnable_slope=False,
        learnable_pos_bias=True,
        mask_outside_bands=False,
        use_ffn=True,
    )

    raw = BSLocoformer(
        encoder=encoder,
        decoder=decoder,
        n_src=n_src,
        n_chan=n_chan,
        n_layers=4,
        emb_dim=emb_dim,
        norm_type="rmsgroupnorm",
        num_groups=4,
        tf_order="ft",
        n_heads=4,
        flash_attention=False,  # CPU profiling friendliness
        attention_dim=emb_dim,
        ffn_type=["swiglu_conv1d", "swiglu_conv1d"],
        ffn_hidden_dim=[128, 128],
        conv1d_kernel=8,
        conv1d_shift=1,
        dropout=0.0,
        masking=True,
        eps=1.0e-5,
        checkpointing=False,
    )
    return raw


def build_online_model(*, n_src: int, n_chan: int, d_model: int, n_layers: int, causal: bool):
    return OnlineSFCModel(
        n_src=n_src,
        n_chan=n_chan,
        d_model=d_model,
        n_layers=n_layers,
        kernel_size=(3, 3),
        causal=causal,
        masking=True,
    ).core  # compare deployable core graph


def main():
    p = ArgumentParser()
    p.add_argument("--device", type=str, default="cpu")

    # common shape
    p.add_argument("--n_chan", type=int, default=2)
    p.add_argument("--n_src", type=int, default=4)
    p.add_argument("--n_fft", type=int, default=2048)
    p.add_argument("--frames", type=int, default=64, help="T")
    p.add_argument("--sr", type=int, default=44100)

    # raw config (matches the provided musdb recipe defaults)
    p.add_argument("--raw_emb_dim", type=int, default=96)
    p.add_argument("--raw_d_inner", type=int, default=64)
    p.add_argument("--raw_n_bands", type=int, default=64)

    # online config
    p.add_argument("--online_d_model", type=int, default=96)
    p.add_argument("--online_n_layers", type=int, default=12)
    p.add_argument("--online_causal", action="store_true")

    args = p.parse_args()

    dev = torch.device(args.device)
    F = args.n_fft // 2 + 1
    T = args.frames

    raw = build_raw_model(
        sr=args.sr,
        n_fft=args.n_fft,
        n_chan=args.n_chan,
        n_src=args.n_src,
        emb_dim=args.raw_emb_dim,
        d_inner=args.raw_d_inner,
        n_bands=args.raw_n_bands,
    ).to(dev)
    raw.eval()

    online_core = build_online_model(
        n_src=args.n_src,
        n_chan=args.n_chan,
        d_model=args.online_d_model,
        n_layers=args.online_n_layers,
        causal=args.online_causal,
    ).to(dev)
    online_core.eval()

    # Inputs:
    # - raw expects complex STFT (B, M, F, T)
    x_raw = torch.randn(1, args.n_chan, F, T, device=dev, dtype=torch.complex64)
    # - online core expects packed float (B, 2*M, T, F)
    x_online = torch.randn(1, 2 * args.n_chan, T, F, device=dev, dtype=torch.float32)

    raw_params = count_params(raw)
    online_params = count_params(online_core)

    raw_flops = estimate_flops(raw, x_raw)
    online_flops = estimate_flops(online_core, x_online)

    print("=== Parameter count ===")
    print(f"raw:    {raw_params:,}")
    print(f"online: {online_params:,}")
    if raw_params > 0:
        print(f"ratio (online/raw): {online_params/raw_params:.3f}")

    print("\n=== FLOPs (torch.profiler, CPU) ===")
    print("Note: profiler FLOPs coverage depends on PyTorch op support; treat as approximate.")
    print(f"raw:    {raw_flops if raw_flops is not None else 'N/A'}")
    print(f"online: {online_flops if online_flops is not None else 'N/A'}")
    if raw_flops and online_flops:
        print(f"ratio (online/raw): {online_flops/raw_flops:.3f}")


if __name__ == "__main__":
    main()

