from __future__ import annotations

import torch

from spectral_feature_compression.core.model.online_hierarchical_soft_band_parallel_ffi_sfc_2d import (
    OnlineHierarchicalSoftBandParallelFFISFC2D,
)


def _build_model() -> OnlineHierarchicalSoftBandParallelFFISFC2D:
    return OnlineHierarchicalSoftBandParallelFFISFC2D(
        n_freq=1025,
        pre_bands=128,
        mid_bands=96,
        bottleneck_bands=48,
        n_fft=2048,
        sample_rate=44100,
        band_config="speech_lowfreq_narrow",
        n_src=3,
        n_chan=1,
        d_model=20,
        pre_layers=0,
        mid_layers=1,
        bottleneck_layers=1,
        kernel_size=(3, 3),
        causal=True,
        masking=True,
        hierarchical_prior_mode="inherited",
        time_branch_kernel_sizes=(3, 3),
        time_branch_dilations=(1, 6),
    ).eval()


def test_forward_stream_matches_forward() -> None:
    torch.manual_seed(0)
    model = _build_model()
    x = torch.randn(1, 2, 4, 1025)

    with torch.no_grad():
        y_full = model(x)
        state = model.init_stream_state(batch_size=1, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(x.shape[2]):
            y_t, state = model.forward_stream(x[:, :, t : t + 1, :], state)
            ys.append(y_t)
        y_stream = torch.cat(ys, dim=2)

    assert y_full.shape == y_stream.shape == (1, 6, 4, 1025)
    assert torch.allclose(y_full, y_stream, atol=1e-4, rtol=1e-4)


def test_rt192k_candidate_within_budget() -> None:
    model = _build_model()
    layer_cache_kib = model.state_size_bytes(dtype=torch.float16, mode="layer_cache") / 1024.0
    assert layer_cache_kib <= 192.0
