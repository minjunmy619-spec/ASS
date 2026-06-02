from __future__ import annotations

import torch

from .presets import build_band_sfc_net_npu_preset
from .training_wrapper import build_band_sfc_net_npu_system
from spectral_feature_compression.core.model.online_soft_band_sfc_2d import SoftBandSpec2d


def _assert_close(name: str, a: torch.Tensor, b: torch.Tensor, atol: float = 1e-5) -> None:
    diff = (a - b).abs().max().item()
    if diff > atol:
        raise AssertionError(f"{name} max diff {diff:.6g} > {atol}")


def test_safe_forward_shape() -> None:
    model = build_band_sfc_net_npu_preset("safe", n_freq=129, n_src=3, n_chan=1).eval()
    x = torch.randn(2, 2, 5, 129)
    y = model(x)
    assert tuple(y.shape) == (2, 6, 5, 129)


def test_safe_streaming_matches_full() -> None:
    torch.manual_seed(0)
    model = build_band_sfc_net_npu_preset("safe", n_freq=65, n_src=3, n_chan=1).eval()
    x = torch.randn(1, 2, 6, 65)
    with torch.no_grad():
        y_full = model(x)
        state = model.init_stream_state(batch_size=1, dtype=x.dtype)
        frames = []
        for t in range(x.shape[2]):
            y_t, state = model.forward_stream(x[:, :, t : t + 1, :], state)
            frames.append(y_t)
        y_stream = torch.cat(frames, dim=2)
    _assert_close("safe streaming parity", y_full, y_stream, atol=2e-5)


def test_quality_forward_shape() -> None:
    model = build_band_sfc_net_npu_preset("quality", n_freq=65, n_src=3, n_chan=1).eval()
    x = torch.randn(1, 2, 3, 65)
    with torch.no_grad():
        y = model(x)
    assert tuple(y.shape) == (1, 6, 3, 65)


def test_quality_initializes_to_mixture_split_mask() -> None:
    model = build_band_sfc_net_npu_preset("quality", n_freq=65, n_src=3, n_chan=1).eval()
    x = torch.randn(1, 2, 3, 65)
    with torch.no_grad():
        y = model(x)
    expected = x.repeat(1, 3, 1, 1) / 3.0
    _assert_close("quality initial mixture split", y, expected, atol=1e-6)


def test_rt_plus_initializes_to_mixture_split_mask_without_residual() -> None:
    model = build_band_sfc_net_npu_preset("rt_plus", n_freq=65, n_src=3, n_chan=1).eval()
    x = torch.randn(1, 2, 3, 65)
    with torch.no_grad():
        y = model(x)
    expected = x.repeat(1, 3, 1, 1) / 3.0
    _assert_close("rt_plus initial mixture split", y, expected, atol=1e-6)


def test_safe_initializes_to_mixture_split_mask() -> None:
    model = build_band_sfc_net_npu_preset("safe", n_freq=65, n_src=3, n_chan=1).eval()
    x = torch.randn(1, 2, 3, 65)
    with torch.no_grad():
        y = model(x)
    expected = x.repeat(1, 3, 1, 1) / 3.0
    _assert_close("safe initial mixture split", y, expected, atol=1e-6)


def test_training_builder_threads_core_band_prior_metadata() -> None:
    system = build_band_sfc_net_npu_system(
        n_fft=2048,
        hop_length=512,
        fs=44100,
        n_src=3,
        n_chan=1,
        preset="quality",
        freq_preprocess_enabled=True,
        freq_preprocess_keep_bins=475,
        freq_preprocess_target_bins=512,
        freq_preprocess_mode="triangular",
    )
    core = system.model.core
    expected = SoftBandSpec2d(
        n_freq=512,
        n_bands=64,
        n_fft=1022,
        sample_rate=44100,
        band_config="musical",
    ).expansion_basis()
    fallback = SoftBandSpec2d(n_freq=512, n_bands=64, band_config="musical").expansion_basis()
    torch.testing.assert_close(core.encoder.query_basis, expected)
    assert not torch.allclose(core.encoder.query_basis, fallback)


@torch.inference_mode()
def test_css_validation_ignores_reference_kwarg() -> None:
    system = build_band_sfc_net_npu_system(
        n_fft=64,
        hop_length=16,
        fs=64,
        n_src=3,
        n_chan=1,
        preset="safe",
        freq_preprocess_enabled=True,
        freq_preprocess_keep_bins=16,
        freq_preprocess_target_bins=20,
        freq_preprocess_mode="triangular",
        css_segment_size=1,
        css_shift_size=1,
    ).eval()
    wav = torch.randn(1, 1, 96)
    ref = torch.randn(1, 3, 1, 96)
    est = system.css(wav, ref=ref)
    assert tuple(est.shape) == tuple(ref.shape)


def test_state_budget_safe_fp512() -> None:
    model = build_band_sfc_net_npu_preset("safe", n_freq=512, n_src=3, n_chan=1).eval()
    state_kib = model.state_size_bytes(dtype=torch.float16) / 1024.0
    assert state_kib < 192.0, f"safe fp512 state too large: {state_kib:.2f} KiB"


def main() -> None:
    tests = [
        test_safe_forward_shape,
        test_safe_streaming_matches_full,
        test_quality_forward_shape,
        test_quality_initializes_to_mixture_split_mask,
        test_rt_plus_initializes_to_mixture_split_mask_without_residual,
        test_safe_initializes_to_mixture_split_mask,
        test_training_builder_threads_core_band_prior_metadata,
        test_css_validation_ignores_reference_kwarg,
        test_state_budget_safe_fp512,
    ]
    for test in tests:
        test()
        print(f"[pass] {test.__name__}")
    print("all BandSFCNetNPU smoke tests passed")


if __name__ == "__main__":
    main()
