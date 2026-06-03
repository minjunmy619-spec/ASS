from __future__ import annotations

import torch

from spectral_feature_compression.core.model.online_soft_band_sfc_2d import SoftBandSpec2d

from .band_sfc_net_npu import CausalCNBBlock, CausalFSMNBandMixer
from .presets import build_band_sfc_net_npu_preset, quality_soft_query, safe_soft_query
from .training_wrapper import build_band_sfc_net_npu_system


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


def test_query_variant_factories_accept_positional_n_freq() -> None:
    soft = safe_soft_query(65, n_src=3, n_chan=1).eval()
    quality = quality_soft_query(65, n_src=3, n_chan=1).eval()
    assert soft.transport == "soft_query"
    assert quality.transport == "soft_query"


def test_query_variant_forward_stream_matches_full() -> None:
    torch.manual_seed(0)
    for preset in (
        "safe_soft_band_query",
        "safe_crossattn_query",
        "balanced_soft_band_query",
        "balanced_crossattn_query",
        "quality_soft_band_query",
        "quality_crossattn_query",
    ):
        model = build_band_sfc_net_npu_preset(preset, n_freq=65, n_src=3, n_chan=1).eval()
        x = torch.randn(1, 2, 4, 65)
        with torch.no_grad():
            y_full = model(x)
            state = model.init_stream_state(batch_size=1, dtype=x.dtype)
            frames = []
            for t in range(x.shape[2]):
                y_t, state = model.forward_stream(x[:, :, t : t + 1, :], state)
                frames.append(y_t)
            y_stream = torch.cat(frames, dim=2)
        assert tuple(y_full.shape) == (1, 6, 4, 65)
        _assert_close(f"{preset} streaming parity", y_full, y_stream, atol=3e-5)


def test_query_variant_deployable_state_budgets_fp512() -> None:
    for preset in (
        "safe_soft_band_query",
        "safe_crossattn_query",
        "balanced_soft_band_query",
        "balanced_crossattn_query",
        "quality_soft_band_query",
        "quality_crossattn_query",
        "rt_plus_soft_band_query",
        "rt_plus_crossattn_query",
        "causal_cnb_soft_band_query",
        "causal_cnb_crossattn_query",
    ):
        model = build_band_sfc_net_npu_preset(preset, n_freq=512, n_src=3, n_chan=1).eval()
        params = sum(p.numel() for p in model.parameters())
        state_kib = model.state_size_bytes(dtype=torch.float16) / 1024.0
        assert params < 7_000_000, f"{preset} params too large: {params}"
        assert state_kib < 192.0, f"{preset} fp512 state too large: {state_kib:.2f} KiB"


def test_causal_cnb_query_variants_forward_stream_matches_full() -> None:
    torch.manual_seed(0)
    for preset in ("causal_cnb_soft_band_query", "causal_cnb_crossattn_query"):
        model = build_band_sfc_net_npu_preset(preset, n_freq=65, n_src=3, n_chan=1).eval()
        x = torch.randn(1, 2, 4, 65)
        with torch.no_grad():
            y_full = model(x)
            state = model.init_stream_state(batch_size=1, dtype=x.dtype)
            frames = []
            for t in range(x.shape[2]):
                y_t, state = model.forward_stream(x[:, :, t : t + 1, :], state)
                frames.append(y_t)
            y_stream = torch.cat(frames, dim=2)
        assert tuple(y_full.shape) == (1, 6, 4, 65)
        _assert_close(f"{preset} CNB streaming parity", y_full, y_stream, atol=4e-5)
        assert model.stage_type == "causal_cnb"
        assert isinstance(model.stages[0], CausalCNBBlock)
        assert model.stages[0].narrow_band.dilation_schedule == (1, 2, 3)


def test_causal_cnb_literal_document_schedule_is_rejected_by_npu_validator() -> None:
    try:
        CausalFSMNBandMixer(24, kernel_t=5, dilation_schedule=(1, 2, 4))
    except ValueError as exc:
        assert ">= 14" in str(exc)
    else:
        raise AssertionError("Expected kernel_t=5 dilation=4 to violate the current NPU validator")


def test_causal_fsmn_zero_context_keeps_empty_stream_state() -> None:
    mixer = CausalFSMNBandMixer(24, kernel_t=1, dilation_schedule=(1,)).eval()
    x = torch.randn(1, 24, 1, 8)
    with torch.no_grad():
        _y, state = mixer.forward_stream(x, None)
    assert tuple(state.shape) == (1, 24, 0, 8)


def test_balanced_query_variants_have_useful_capacity() -> None:
    for preset in ("balanced_soft_band_query", "balanced_crossattn_query"):
        model = build_band_sfc_net_npu_preset(preset, n_freq=512, n_src=3, n_chan=1).eval()
        params = sum(p.numel() for p in model.parameters())
        state_kib = model.state_size_bytes(dtype=torch.float16) / 1024.0
        assert 2_000_000 <= params <= 7_000_000, f"{preset} params outside useful target: {params}"
        assert state_kib < 192.0, f"{preset} fp512 state too large: {state_kib:.2f} KiB"
        assert model.channels == 40
        assert model.n_bands == 64
        assert len(model.stages) == 4


def test_training_builder_threads_capacity_overrides() -> None:
    system = build_band_sfc_net_npu_system(
        n_fft=128,
        hop_length=32,
        fs=16000,
        n_src=3,
        n_chan=1,
        preset="balanced_soft_band_query",
        channels=36,
        time_kernel=5,
        freq_kernel=5,
        dilation_cycle=[1, 1, 2, 3],
        routing_normalization="softmax",
        use_attn=True,
        attn_window=8,
        num_heads=4,
        head_dim=6,
        pooled_mixer_hidden=6144,
        pooled_mixer_hidden_schedule=[4096, 6144, 8192, 6144],
        freq_preprocess_enabled=False,
        css_segment_size=1,
        css_shift_size=1,
    )
    core = system.model.core
    assert core.transport == "soft_query"
    assert core.channels == 36
    assert core.time_kernel == 5
    assert core.freq_kernel == 5
    assert core.dilation_schedule == (1, 1, 2, 3)
    assert core.use_attn is True
    assert core.attn_window == 8
    assert core.num_heads == 4
    assert core.head_dim == 6
    assert core.pooled_mixer_hidden_schedule == (4096, 6144, 8192, 6144)
    assert core.stages[0].pooled_mixer.hidden_channels == 4096
    assert core.stages[1].pooled_mixer.hidden_channels == 6144


def test_training_builder_threads_query_type() -> None:
    system = build_band_sfc_net_npu_system(
        n_fft=128,
        hop_length=32,
        fs=16000,
        n_src=3,
        n_chan=1,
        preset="safe_crossattn_query",
        query_type="learnable",
        freq_preprocess_enabled=False,
        css_segment_size=1,
        css_shift_size=1,
    )
    assert system.model.core.transport == "crossattn_query"
    assert system.model.core.encoder.query_type == "learnable"


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
        test_query_variant_factories_accept_positional_n_freq,
        test_query_variant_forward_stream_matches_full,
        test_query_variant_deployable_state_budgets_fp512,
        test_causal_cnb_query_variants_forward_stream_matches_full,
        test_causal_cnb_literal_document_schedule_is_rejected_by_npu_validator,
        test_causal_fsmn_zero_context_keeps_empty_stream_state,
        test_balanced_query_variants_have_useful_capacity,
        test_training_builder_threads_capacity_overrides,
        test_training_builder_threads_query_type,
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
