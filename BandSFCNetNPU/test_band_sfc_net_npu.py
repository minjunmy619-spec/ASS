from __future__ import annotations

import torch

from spectral_feature_compression.core.model.adaptive_mel_sfc_2d import AdaptiveMelBandSpec2d
from spectral_feature_compression.core.model.frequency_preprocessing import build_hybrid_frequency_bin_frequencies
from spectral_feature_compression.core.model.online_soft_band_sfc_2d import SoftBandSpec2d

from .band_sfc_net_npu import CausalCNBBlock, CausalFSMNBandMixer, CausalLocoCNBBlock
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
        "causal_cnb_balanced_soft_band_query",
        "causal_cnb_balanced_crossattn_query",
        "adaptive_mel_loco_cnb_soft_band_query",
        "adaptive_mel_loco_cnb_crossattn_query",
        "adaptive_mel_loco_cnb_stable_soft_band_query",
        "adaptive_mel_loco_cnb_stable_crossattn_query",
        "adaptive_mel_loco_cnb_band56_soft_band_query",
        "adaptive_mel_loco_cnb_clean_soft_band_query",
    ):
        model = build_band_sfc_net_npu_preset(
            preset, n_freq=512, n_fft=1022, sample_rate=44100, n_src=3, n_chan=1
        ).eval()
        params = sum(p.numel() for p in model.parameters())
        state_kib = model.state_size_bytes(dtype=torch.float16) / 1024.0
        assert params < 7_000_000, f"{preset} params too large: {params}"
        assert state_kib < 192.0, f"{preset} fp512 state too large: {state_kib:.2f} KiB"


def test_causal_cnb_query_variants_forward_stream_matches_full() -> None:
    torch.manual_seed(0)
    for preset in (
        "causal_cnb_soft_band_query",
        "causal_cnb_crossattn_query",
        "causal_cnb_balanced_soft_band_query",
        "causal_cnb_balanced_crossattn_query",
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


def test_causal_cnb_balanced_variants_have_useful_capacity() -> None:
    for preset in ("causal_cnb_balanced_soft_band_query", "causal_cnb_balanced_crossattn_query"):
        model = build_band_sfc_net_npu_preset(preset, n_freq=512, n_src=3, n_chan=1).eval()
        params = sum(p.numel() for p in model.parameters())
        state_kib = model.state_size_bytes(dtype=torch.float16) / 1024.0
        assert 2_000_000 <= params <= 7_000_000, f"{preset} params outside useful target: {params}"
        assert state_kib < 192.0, f"{preset} fp512 state too large: {state_kib:.2f} KiB"
        assert model.stage_type == "causal_cnb"
        assert model.channels == 32
        assert model.n_bands == 48
        assert len(model.stages) == 5
        assert model.stages[0].pooled_mixer.hidden_channels == 8192


def test_adaptive_mel_loco_cnb_variants_have_strong_capacity_and_state_budget() -> None:
    for preset in ("adaptive_mel_loco_cnb_soft_band_query", "adaptive_mel_loco_cnb_crossattn_query"):
        model = build_band_sfc_net_npu_preset(
            preset,
            n_freq=512,
            n_fft=1022,
            sample_rate=44100,
            n_src=3,
            n_chan=1,
        ).eval()
        params = sum(p.numel() for p in model.parameters())
        state_kib = model.state_size_bytes(dtype=torch.float16) / 1024.0
        assert 5_000_000 <= params <= 7_000_000, f"{preset} params outside strong target: {params}"
        assert state_kib < 192.0, f"{preset} fp512 state too large: {state_kib:.2f} KiB"
        assert model.stage_type == "loco_cnb"
        assert model.band_spec_type == "adaptive_mel"
        assert isinstance(model.band_spec, AdaptiveMelBandSpec2d)
        assert model.channels == 32
        assert model.n_bands == 48
        assert len(model.stages) == 5
        assert isinstance(model.stages[0], CausalLocoCNBBlock)
        assert model.stages[0].narrow_band.kernel_t == 4
        assert model.stages[0].narrow_band.dilation_schedule == (1, 2, 3)
        assert model.stages[0].local.time_kernel == 3
        assert model.encoder_capacity_mixer_layers == 2
        assert model.decoder_capacity_mixer_layers == 2
        assert model.residual_head is True


def test_adaptive_mel_loco_cnb_stability_fix_presets() -> None:
    stable = build_band_sfc_net_npu_preset(
        "adaptive_mel_loco_cnb_stable_soft_band_query",
        n_freq=512,
        n_fft=1022,
        sample_rate=44100,
        n_src=3,
        n_chan=1,
    ).eval()
    stable_params = sum(p.numel() for p in stable.parameters())
    assert 2_000_000 <= stable_params < 4_000_000
    assert stable.state_size_bytes(dtype=torch.float16) / 1024.0 < 192.0
    assert stable.channels == 36
    assert stable.n_bands == 48
    assert stable.residual_head is False
    assert stable.stages[0].pooled_mixer.hidden_channels == 2048
    assert stable.stages[1].pooled_mixer.hidden_channels == 4096
    assert stable.encoder_capacity_mixers[0].hidden_channels == 2048
    assert stable.decoder_capacity_mixers[0].hidden_channels == 2048

    band56 = build_band_sfc_net_npu_preset(
        "adaptive_mel_loco_cnb_band56_soft_band_query",
        n_freq=512,
        n_fft=1022,
        sample_rate=44100,
        n_src=3,
        n_chan=1,
    ).eval()
    band56_params = sum(p.numel() for p in band56.parameters())
    assert 2_000_000 <= band56_params < 4_000_000
    assert band56.state_size_bytes(dtype=torch.float16) / 1024.0 < 192.0
    assert band56.channels == 28
    assert band56.n_bands == 56
    assert band56.residual_head is False

    clean = build_band_sfc_net_npu_preset(
        "adaptive_mel_loco_cnb_clean_soft_band_query",
        n_freq=512,
        n_fft=1022,
        sample_rate=44100,
        n_src=3,
        n_chan=1,
    ).eval()
    clean_params = sum(p.numel() for p in clean.parameters())
    assert 500_000 <= clean_params < 1_500_000
    assert clean.state_size_bytes(dtype=torch.float16) / 1024.0 < 192.0
    assert clean.channels == 36
    assert clean.n_bands == 48
    assert clean.stage_mixer_type == "pointwise"
    assert clean.stages[0].pooled_mixer.__class__.__name__ == "PointwiseChannelMixer"
    assert clean.stages[0].pooled_mixer.hidden_channels == 512
    assert len(clean.encoder_capacity_mixers) == 0
    assert len(clean.decoder_capacity_mixers) == 0
    assert clean.loco_ffn_expansion == 16
    assert clean.residual_head is False


def test_adaptive_mel_loco_cnb_streaming_matches_full() -> None:
    torch.manual_seed(0)
    for preset in (
        "adaptive_mel_loco_cnb_soft_band_query",
        "adaptive_mel_loco_cnb_crossattn_query",
        "adaptive_mel_loco_cnb_stable_soft_band_query",
        "adaptive_mel_loco_cnb_band56_soft_band_query",
        "adaptive_mel_loco_cnb_clean_soft_band_query",
    ):
        model = build_band_sfc_net_npu_preset(
            preset,
            n_freq=65,
            n_fft=128,
            sample_rate=8000,
            n_src=3,
            n_chan=1,
        ).eval()
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
        _assert_close(f"{preset} Loco-CNB streaming parity", y_full, y_stream, atol=5e-5)


def test_adaptive_mel_loco_cnb_stream_state_omits_empty_encoder_cache() -> None:
    model = build_band_sfc_net_npu_preset(
        "adaptive_mel_loco_cnb_soft_band_query",
        n_freq=512,
        n_fft=1022,
        sample_rate=44100,
        n_src=3,
        n_chan=1,
    ).eval()
    state = model.init_stream_state(batch_size=1, dtype=torch.float16)
    flat = []

    def collect(tree) -> None:
        if isinstance(tree, torch.Tensor):
            flat.append(tree)
            return
        for item in tree:
            collect(item)

    collect(state)
    assert len(flat) == 10
    assert all(t.numel() > 0 for t in flat)
    assert model.state_size_bytes(dtype=torch.float16) == 168960


def test_training_builder_threads_loco_cnb_overrides() -> None:
    system = build_band_sfc_net_npu_system(
        n_fft=128,
        hop_length=32,
        fs=16000,
        n_src=3,
        n_chan=1,
        preset="adaptive_mel_loco_cnb_soft_band_query",
        transport="soft_band_query",
        band_spec_type="adaptive_mel",
        low_freq_hz=900.0,
        low_freq_band_fraction=0.5,
        overlap_factor=1.25,
        low_freq_overlap_factor=2.25,
        encoder_capacity_mixer_hidden=1024,
        encoder_capacity_mixer_layers=1,
        decoder_capacity_mixer_hidden=1024,
        decoder_capacity_mixer_layers=1,
        loco_expansion=1,
        loco_ffn_expansion=1,
        loco_time_kernel=3,
        loco_band_kernel=5,
        loco_time_dilation=1,
        residual_head=True,
        freq_preprocess_enabled=False,
        css_segment_size=1,
        css_shift_size=1,
    )
    core = system.model.core
    assert core.stage_type == "loco_cnb"
    assert core.band_spec_type == "adaptive_mel"
    assert isinstance(core.band_spec, AdaptiveMelBandSpec2d)
    assert core.band_spec.low_freq_hz == 900.0
    assert core.band_spec.low_freq_band_fraction == 0.5
    assert core.band_spec.overlap_factor == 1.25
    assert core.band_spec.low_freq_overlap_factor == 2.25
    assert core.encoder_capacity_mixer_layers == 1
    assert core.decoder_capacity_mixer_layers == 1
    assert core.stages[0].local.hidden == core.channels
    assert core.stages[0].local.ffn_hidden == core.channels
    assert core.stages[0].local.band_kernel == 5
    assert core.residual_head is True


def test_training_builder_uses_hybrid_bin_frequencies_for_adaptive_mel_prior() -> None:
    system = build_band_sfc_net_npu_system(
        n_fft=2048,
        hop_length=512,
        fs=44100,
        n_src=3,
        n_chan=1,
        preset="adaptive_mel_loco_cnb_soft_band_query",
        channels=8,
        n_bands=12,
        num_stages=1,
        pooled_mixer_hidden=0,
        pooled_mixer_hidden_schedule=[0],
        encoder_capacity_mixer_hidden=16,
        encoder_capacity_mixer_layers=1,
        decoder_capacity_mixer_hidden=16,
        decoder_capacity_mixer_layers=1,
        freq_preprocess_enabled=True,
        freq_preprocess_keep_bins=475,
        freq_preprocess_target_bins=512,
        css_segment_size=1,
        css_shift_size=1,
    )
    core = system.model.core
    assert core.n_freq == 512
    assert isinstance(core.band_spec, AdaptiveMelBandSpec2d)
    assert core.band_spec.explicit_bin_frequencies is True
    # 1000 Hz is around original 2048-FFT bin 46, not projected-axis bin 23.
    freqs = core.band_spec.bin_frequencies_hz
    assert float(freqs[46]) < 1000.0 < float(freqs[47])
    assert float(freqs[23]) < 600.0

    expected = build_hybrid_frequency_bin_frequencies(
        1025,
        keep_bins=475,
        target_bins=512,
        n_fft=2048,
        sample_rate=44100,
        mode="triangular",
    )
    torch.testing.assert_close(freqs, expected)


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
        test_causal_cnb_balanced_variants_have_useful_capacity,
        test_adaptive_mel_loco_cnb_variants_have_strong_capacity_and_state_budget,
        test_adaptive_mel_loco_cnb_stability_fix_presets,
        test_adaptive_mel_loco_cnb_streaming_matches_full,
        test_adaptive_mel_loco_cnb_stream_state_omits_empty_encoder_cache,
        test_training_builder_threads_loco_cnb_overrides,
        test_training_builder_uses_hybrid_bin_frequencies_for_adaptive_mel_prior,
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
