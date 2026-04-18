from __future__ import annotations

import torch

from spectral_feature_compression.core.model.online_hierarchical_soft_band_ffi_sfc_2d import (
    OnlineHierarchicalSoftBandFFISFC2D,
)
from spectral_feature_compression.core.model.online_soft_band_sfc_2d import SoftBandSpec2d


def _build_model() -> OnlineHierarchicalSoftBandFFISFC2D:
    return OnlineHierarchicalSoftBandFFISFC2D(
        n_freq=1025,
        pre_bands=128,
        mid_bands=96,
        bottleneck_bands=48,
        n_fft=2048,
        sample_rate=44100,
        band_config="speech_lowfreq_narrow",
        n_src=3,
        n_chan=1,
        d_model=24,
        pre_layers=1,
        mid_layers=1,
        bottleneck_layers=2,
        kernel_size=(3, 3),
        causal=True,
        masking=True,
        dilation_cycle=(1, 2, 4, 6),
        hierarchical_prior_mode="inherited",
    ).eval()


def test_speech_lowfreq_narrow_allocates_more_lowfreq_bands() -> None:
    spec = SoftBandSpec2d(
        n_freq=1025,
        n_bands=64,
        n_fft=2048,
        sample_rate=44100,
        band_config="speech_lowfreq_narrow",
    )
    widths = (spec.ends - spec.starts).to(torch.float32)
    assert widths[:8].mean() < widths[-8:].mean()


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
