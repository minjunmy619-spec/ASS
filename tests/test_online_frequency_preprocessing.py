from __future__ import annotations

import torch

from spectral_feature_compression.core.model.frequency_preprocessing import (
    FrequencyPreprocessedOnlineModel,
    build_frequency_preprocessor,
    resolve_preprocessed_n_freq,
)
from spectral_feature_compression.core.model.online_soft_band_query_sfc_2d import OnlineSoftBandQuerySFC2D
from spectral_feature_compression.core.model.online_sfc_2d import pack_complex_stft_as_2d


@torch.inference_mode()
def test_frequency_preprocessed_wrapper_matches_streaming():
    full_n_freq = 33
    keep_bins = 16
    target_bins = 20
    core_n_freq = resolve_preprocessed_n_freq(
        full_n_freq,
        enabled=True,
        keep_bins=keep_bins,
        target_bins=target_bins,
    )
    projector = build_frequency_preprocessor(
        full_n_freq,
        enabled=True,
        keep_bins=keep_bins,
        target_bins=target_bins,
        mode="triangular",
    )
    core = OnlineSoftBandQuerySFC2D(
        n_freq=core_n_freq,
        n_bands=8,
        n_fft=64,
        sample_rate=16000,
        band_config="musical",
        n_src=2,
        n_chan=1,
        d_model=8,
        n_layers=2,
        kernel_size=(3, 3),
        causal=True,
        masking=True,
    ).eval()
    model = FrequencyPreprocessedOnlineModel(core=core, n_src=2, n_chan=1, freq_preprocessor=projector).eval()

    x = torch.randn(1, 1, full_n_freq, 5, dtype=torch.complex64)
    y_full = model(x)
    assert y_full.shape == (1, 2, 1, full_n_freq, 5)

    x2d = pack_complex_stft_as_2d(x)
    state = model.init_stream_state(batch_size=1, device=x2d.device, dtype=x2d.dtype)
    parts = []
    for frame_idx in range(x2d.shape[2]):
        y_part, state = model.forward_stream(x2d[:, :, frame_idx : frame_idx + 1, :], state)
        parts.append(y_part)
    y_stream = torch.cat(parts, dim=2)

    expected = pack_complex_stft_as_2d(y_full.squeeze(2))
    assert y_stream.shape == expected.shape
    diff = (y_stream - expected).abs().max().item()
    assert diff < 1e-4


def test_build_frequency_preprocessor_keeps_requested_size():
    projector = build_frequency_preprocessor(
        1025,
        enabled=True,
        keep_bins=475,
        target_bins=512,
        mode="triangular",
    )
    assert projector is not None
    assert projector.n_freq_in == 1025
    assert projector.n_freq_out == 512
    assert projector.keep_bins == 475
