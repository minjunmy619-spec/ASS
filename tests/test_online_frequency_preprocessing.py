from __future__ import annotations

import torch

from BandSCNetNPU.training_wrapper import build_band_scnet_npu_system
from BandSFCNetNPU.training_wrapper import build_band_sfc_net_npu_system
from spectral_feature_compression.core.model.frequency_preprocessing import (
    FrequencyPreprocessedOnlineModel,
    LearnableQueryFrequencyProjector2d,
    PCENGainNormalizer2d,
    build_frequency_preprocessor,
    build_hybrid_frequency_bin_frequencies,
    build_hybrid_frequency_matrices,
    build_pcen_preprocessor,
    resolve_frequency_input_n_freq,
    resolve_preprocessed_n_freq,
)
from spectral_feature_compression.core.model.online_sfc_2d import pack_complex_stft_as_2d, unpack_2d_to_complex_stft
from spectral_feature_compression.core.model.online_soft_band_query_sfc_2d import OnlineSoftBandQuerySFC2D
from spectral_feature_compression.core.model.online_soft_band_sfc_2d import build_online_soft_band_sfc_system


class _RepeatPackedCore(torch.nn.Module):
    def __init__(self, *, n_freq: int, n_src: int, n_chan: int):
        super().__init__()
        self.n_freq = n_freq
        self.n_src = n_src
        self.n_chan = n_chan

    def forward(self, x2d: torch.Tensor) -> torch.Tensor:
        return x2d.repeat(1, self.n_src, 1, 1)

    def init_stream_state(self, batch_size: int = 1, *, device=None, dtype=None):
        return torch.zeros(batch_size, 0, 0, 0, device=device, dtype=dtype)

    def forward_stream(self, x2d: torch.Tensor, state=None):
        return self(x2d), state

    def stream_context_frames(self) -> int:
        return 0

    def forward_stream_recompute(self, x2d: torch.Tensor, history=None):
        return self(x2d), None


class _OneFrameContextPackedCore(torch.nn.Module):
    def __init__(self, *, n_freq: int, n_src: int, n_chan: int):
        super().__init__()
        self.n_freq = n_freq
        self.n_src = n_src
        self.n_chan = n_chan

    def _repeat(self, x2d: torch.Tensor) -> torch.Tensor:
        return x2d.repeat(1, self.n_src, 1, 1)

    def forward(self, x2d: torch.Tensor) -> torch.Tensor:
        previous = torch.nn.functional.pad(x2d[:, :, :-1, :], (0, 0, 1, 0))
        return self._repeat(x2d + 0.5 * previous)

    def init_stream_state(self, batch_size: int = 1, *, device=None, dtype=None):
        return torch.zeros(batch_size, 2 * self.n_chan, 1, self.n_freq, device=device, dtype=dtype)

    def forward_stream(self, x2d: torch.Tensor, state=None):
        if state is None:
            state = self.init_stream_state(batch_size=x2d.shape[0], device=x2d.device, dtype=x2d.dtype)
        full = torch.cat([state, x2d], dim=2)
        previous = full[:, :, :-1, :][:, :, -x2d.shape[2] :, :]
        new_state = full[:, :, -1:, :]
        return self._repeat(x2d + 0.5 * previous), new_state

    def stream_context_frames(self) -> int:
        return 1

    def forward_stream_recompute(self, x2d: torch.Tensor, history=None):
        if history is None:
            history = self.init_stream_state(batch_size=x2d.shape[0], device=x2d.device, dtype=x2d.dtype)
        full = torch.cat([history, x2d], dim=2)
        previous = full[:, :, :-1, :][:, :, -x2d.shape[2] :, :]
        new_history = full[:, :, -1:, :]
        return self._repeat(x2d + 0.5 * previous), new_history


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


def test_hybrid_log_and_piecewise_high_modes_have_valid_nonuniform_coverage():
    n_freq = 1025
    keep_bins = 475
    target_bins = 512
    n_fft = 2048
    sample_rate = 24000

    for mode in ("hybrid_log_high", "hybrid_piecewise_high"):
        analysis, synthesis = build_hybrid_frequency_matrices(
            n_freq,
            keep_bins=keep_bins,
            target_bins=target_bins,
            mode=mode,
        )
        assert tuple(analysis.shape) == (target_bins, n_freq)
        assert tuple(synthesis.shape) == (n_freq, target_bins)
        torch.testing.assert_close(analysis[:keep_bins, :keep_bins], torch.eye(keep_bins))
        torch.testing.assert_close(synthesis[:keep_bins, :keep_bins], torch.eye(keep_bins))
        torch.testing.assert_close(analysis[keep_bins:].sum(dim=1), torch.ones(target_bins - keep_bins))
        torch.testing.assert_close(synthesis[keep_bins:].sum(dim=1), torch.ones(n_freq - keep_bins))

        projector = build_frequency_preprocessor(
            n_freq,
            enabled=True,
            keep_bins=keep_bins,
            target_bins=target_bins,
            mode=mode,
        )
        assert projector is not None
        assert projector.manifest()["mode"] == mode

    log_freqs = build_hybrid_frequency_bin_frequencies(
        n_freq,
        keep_bins=keep_bins,
        target_bins=target_bins,
        n_fft=n_fft,
        sample_rate=sample_rate,
        mode="hybrid_log_high",
    )
    log_steps = log_freqs[keep_bins + 1 :] - log_freqs[keep_bins:-1]
    assert log_steps[:8].mean() < log_steps[-8:].mean()

    piecewise_freqs = build_hybrid_frequency_bin_frequencies(
        n_freq,
        keep_bins=keep_bins,
        target_bins=target_bins,
        n_fft=n_fft,
        sample_rate=sample_rate,
        mode="hybrid_piecewise_high",
    )
    piecewise_steps = piecewise_freqs[keep_bins + 1 :] - piecewise_freqs[keep_bins:-1]
    assert piecewise_steps[:16].mean() < piecewise_steps[-8:].mean()


def test_build_frequency_preprocessor_can_bypass_dc_bin():
    assert resolve_frequency_input_n_freq(1025, dc_bypass_enabled=True) == 1024
    assert (
        resolve_preprocessed_n_freq(
            1025,
            enabled=True,
            keep_bins=475,
            target_bins=512,
            dc_bypass_enabled=True,
        )
        == 512
    )
    projector = build_frequency_preprocessor(
        1025,
        enabled=True,
        keep_bins=475,
        target_bins=512,
        mode="triangular",
        dc_bypass_enabled=True,
    )
    assert projector is not None
    assert projector.n_freq_in == 1024
    assert projector.n_freq_out == 512


def test_learnable_query_frequency_preprocessor_uses_tied_decoder_query():
    torch.manual_seed(0)
    projector = build_frequency_preprocessor(
        33,
        enabled=True,
        keep_bins=16,
        target_bins=21,
        mode="learnable_query",
    )
    assert isinstance(projector, LearnableQueryFrequencyProjector2d)
    assert projector.frequency_query.requires_grad
    assert tuple(projector.frequency_query.shape) == (21, 33)

    x = torch.randn(2, 3, 4, 33)
    y = projector.analysis(x)
    z = projector.synthesis(y)
    flat_y = y.reshape(-1, 21)
    expected = (flat_y @ projector.frequency_query).reshape(2, 3, 4, 33)
    torch.testing.assert_close(z, expected, rtol=1e-6, atol=1e-6)

    loss = z.square().mean()
    loss.backward()
    assert projector.frequency_query.grad is not None
    assert projector.manifest()["type"] == "sfc_lite_learnable_query"
    assert projector.manifest()["tied_synthesis_query"] is True


@torch.inference_mode()
def test_pcen_gain_normalizer_streaming_matches_full_forward():
    torch.manual_seed(0)
    n_src = 2
    n_chan = 1
    n_freq = 33
    n_frames = 5
    core = _RepeatPackedCore(n_freq=n_freq, n_src=n_src, n_chan=n_chan)
    pcen = PCENGainNormalizer2d(
        n_chan=n_chan,
        smooth_coef=0.8,
        alpha=0.5,
        delta=2.0,
        root=0.5,
        gain_floor=0.05,
        gain_ceiling=10.0,
    )
    model = FrequencyPreprocessedOnlineModel(
        core=core,
        n_src=n_src,
        n_chan=n_chan,
        pcen_preprocessor=pcen,
    ).eval()

    x = torch.randn(1, n_chan, n_freq, n_frames, dtype=torch.complex64)
    y_full = model(x)
    expected = x.unsqueeze(1).expand(-1, n_src, -1, -1, -1)
    torch.testing.assert_close(y_full, expected, rtol=1e-5, atol=1e-5)

    x2d = pack_complex_stft_as_2d(x)
    state = model.init_stream_state(batch_size=1, device=x2d.device, dtype=x2d.dtype)
    parts = []
    for frame_idx in range(x2d.shape[2]):
        y_part, state = model.forward_stream(x2d[:, :, frame_idx : frame_idx + 1, :], state)
        parts.append(y_part)
    y_stream = torch.cat(parts, dim=2)

    expected_packed = pack_complex_stft_as_2d(y_full.reshape(1, n_src * n_chan, n_freq, n_frames))
    torch.testing.assert_close(y_stream, expected_packed, rtol=1e-5, atol=1e-5)
    assert model.pcen_preprocess_manifest()["type"] == "pcen_gain_normalizer_2d"


@torch.inference_mode()
def test_pcen_gain_normalizer_recompute_carries_pcen_state():
    torch.manual_seed(0)
    n_src = 2
    n_chan = 1
    n_freq = 33
    n_frames = 6
    core = _OneFrameContextPackedCore(n_freq=n_freq, n_src=n_src, n_chan=n_chan)
    pcen = PCENGainNormalizer2d(
        n_chan=n_chan,
        smooth_coef=0.8,
        alpha=0.5,
        delta=2.0,
        root=0.5,
        gain_floor=0.05,
        gain_ceiling=10.0,
    )
    model = FrequencyPreprocessedOnlineModel(
        core=core,
        n_src=n_src,
        n_chan=n_chan,
        pcen_preprocessor=pcen,
    ).eval()

    x = torch.randn(1, n_chan, n_freq, n_frames, dtype=torch.complex64)
    x2d = pack_complex_stft_as_2d(x)
    stream_state = model.init_stream_state(batch_size=1, device=x2d.device, dtype=x2d.dtype)
    recompute_history = None
    stream_parts = []
    recompute_parts = []
    for frame_idx in range(x2d.shape[2]):
        frame = x2d[:, :, frame_idx : frame_idx + 1, :]
        y_stream, stream_state = model.forward_stream(frame, stream_state)
        y_recompute, recompute_history = model.forward_stream_recompute(frame, recompute_history)
        stream_parts.append(y_stream)
        recompute_parts.append(y_recompute)

    torch.testing.assert_close(
        torch.cat(recompute_parts, dim=2),
        torch.cat(stream_parts, dim=2),
        rtol=1e-5,
        atol=1e-5,
    )


def test_build_pcen_preprocessor_is_opt_in():
    assert build_pcen_preprocessor(n_chan=1, enabled=False) is None
    assert build_pcen_preprocessor(n_chan=1, enabled=True) is not None


def test_pcen_gain_normalizer_rejects_wrong_state_shape():
    pcen = PCENGainNormalizer2d(n_chan=1)
    x = torch.randn(1, 2, 1, 8)
    bad_state = torch.zeros(1, 1, 1, 7)
    try:
        pcen.forward_with_gain(x, bad_state)
    except ValueError as exc:
        assert "PCEN state shape" in str(exc)
    else:
        raise AssertionError("Expected wrong PCEN state shape to fail")


def test_frequency_preprocessed_wrapper_rejects_core_frequency_mismatch():
    core = _RepeatPackedCore(n_freq=20, n_src=2, n_chan=1)
    projector = build_frequency_preprocessor(33, enabled=True, keep_bins=16, target_bins=21)
    try:
        FrequencyPreprocessedOnlineModel(core=core, n_src=2, n_chan=1, freq_preprocessor=projector)
    except ValueError as exc:
        assert "Core n_freq" in str(exc)
    else:
        raise AssertionError("Expected wrapper/core n_freq mismatch to fail")


@torch.inference_mode()
def test_frequency_preprocessed_wrapper_can_append_residual_source():
    torch.manual_seed(0)
    n_src = 3
    explicit_n_src = 2
    n_chan = 1
    n_freq = 17
    n_frames = 4
    core = _RepeatPackedCore(n_freq=n_freq, n_src=explicit_n_src, n_chan=n_chan)
    model = FrequencyPreprocessedOnlineModel(
        core=core,
        n_src=n_src,
        n_chan=n_chan,
        residual_source_enabled=True,
        residual_source_index=2,
    ).eval()

    x = torch.randn(1, n_chan, n_freq, n_frames, dtype=torch.complex64)
    y = model(x)
    assert tuple(y.shape) == (1, n_src, n_chan, n_freq, n_frames)
    torch.testing.assert_close(y.sum(dim=1), x, rtol=1e-5, atol=1e-5)
    assert model.residual_source_manifest()["explicit_n_src"] == explicit_n_src

    x2d = pack_complex_stft_as_2d(x)
    y2d, _state = model.forward_stream(x2d, model.init_stream_state(batch_size=1, dtype=x2d.dtype))
    y_stream = unpack_2d_to_complex_stft(y2d, n_src=n_src, n_chan=n_chan)
    torch.testing.assert_close(y_stream.sum(dim=1), x, rtol=1e-5, atol=1e-5)


def test_npu_and_online_builders_accept_unified_preprocessing_flags():
    bandsc = build_band_scnet_npu_system(
        n_fft=64,
        hop_length=16,
        fs=8000,
        n_src=2,
        n_chan=1,
        preset="edge_small",
        freq_preprocess_enabled=True,
        freq_preprocess_keep_bins=16,
        freq_preprocess_target_bins=20,
        dc_bypass_enabled=True,
        pcen_preprocess_enabled=True,
        css_segment_size=1,
        css_shift_size=1,
    )
    assert bandsc.model.core.n_freq == 20
    assert bandsc.model.body_input_n_freq == 32
    assert bandsc.model.pcen_preprocess_manifest()["type"] == "pcen_gain_normalizer_2d"
    assert bandsc.model.dc_bypass_manifest()["body_input_n_freq"] == 32

    online = build_online_soft_band_sfc_system(
        n_fft=64,
        hop_length=16,
        fs=8000,
        n_src=2,
        n_chan=1,
        n_bands=8,
        d_model=8,
        n_layers=1,
        freq_preprocess_enabled=True,
        freq_preprocess_keep_bins=16,
        freq_preprocess_target_bins=20,
        dc_bypass_enabled=True,
        pcen_preprocess_enabled=True,
        css_segment_size=1,
        css_shift_size=1,
    )
    assert online.model.core.n_freq == 20
    assert online.model.body_input_n_freq == 32
    assert online.model.pcen_preprocess_manifest()["type"] == "pcen_gain_normalizer_2d"
    assert online.model.dc_bypass_manifest()["body_input_n_freq"] == 32


def test_band_sfc_builder_supports_two_mask_residual_sfx_contract():
    system = build_band_sfc_net_npu_system(
        n_fft=64,
        hop_length=16,
        fs=8000,
        n_src=3,
        n_chan=1,
        core_n_src=2,
        preset="safe",
        freq_preprocess_enabled=True,
        freq_preprocess_keep_bins=16,
        freq_preprocess_target_bins=20,
        residual_source_enabled=True,
        residual_source_index=2,
        css_segment_size=1,
        css_shift_size=1,
    )
    assert system.model.n_src == 3
    assert system.model.core.n_src == 2
    assert system.model.residual_source_manifest()["mode"] == "mixture_minus_explicit_sources"


@torch.inference_mode()
def test_dc_bypass_zero_policy_restores_full_frequency_shape_and_streaming():
    torch.manual_seed(0)
    n_src = 2
    n_chan = 1
    full_n_freq = 33
    n_frames = 5
    core = _RepeatPackedCore(n_freq=full_n_freq - 1, n_src=n_src, n_chan=n_chan)
    model = FrequencyPreprocessedOnlineModel(
        core=core,
        n_src=n_src,
        n_chan=n_chan,
        dc_bypass_enabled=True,
        dc_policy="zero",
    ).eval()

    x = torch.randn(1, n_chan, full_n_freq, n_frames, dtype=torch.complex64)
    y_full = model(x)
    assert y_full.shape == (1, n_src, n_chan, full_n_freq, n_frames)
    torch.testing.assert_close(y_full[..., 0:1, :], torch.zeros_like(y_full[..., 0:1, :]))
    expected_body = x[:, None, :, 1:, :].expand(-1, n_src, -1, -1, -1)
    torch.testing.assert_close(y_full[..., 1:, :], expected_body, rtol=1e-5, atol=1e-5)

    x2d = pack_complex_stft_as_2d(x)
    state = model.init_stream_state(batch_size=1, device=x2d.device, dtype=x2d.dtype)
    parts = []
    for frame_idx in range(x2d.shape[2]):
        y_part, state = model.forward_stream(x2d[:, :, frame_idx : frame_idx + 1, :], state)
        parts.append(y_part)
    y_stream = torch.cat(parts, dim=2)
    expected_packed = pack_complex_stft_as_2d(y_full.reshape(1, n_src * n_chan, full_n_freq, n_frames))
    torch.testing.assert_close(y_stream, expected_packed, rtol=1e-5, atol=1e-5)
    assert model.dc_bypass_manifest()["body_input_n_freq"] == full_n_freq - 1
