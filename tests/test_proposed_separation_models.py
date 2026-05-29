from __future__ import annotations

import torch

from BandSFCNetNPU.presets import build_band_sfc_net_npu_preset
from spectral_feature_compression.core.loss.composite_separation import CompositeSeparationSpectralLoss
from spectral_feature_compression.core.model.proposed_separation_models import (
    build_edgefusion_sfc_distilled_system,
    build_hierarchical_sfc_ffi_lite_system,
    build_sfc_locoformer_lite_plus_system,
)
from spectral_feature_compression.core.tasks.distillation_task import TeacherStudentDistillationTask


def test_band_sfc_rt_plus_forward_and_streaming_shape() -> None:
    torch.manual_seed(0)
    model = build_band_sfc_net_npu_preset("rt_plus", n_freq=65, n_src=3, n_chan=1).eval()
    x = torch.randn(1, 2, 4, 65)
    with torch.no_grad():
        y = model(x)
        state = model.init_stream_state(batch_size=1, dtype=x.dtype)
        frames = []
        for frame_idx in range(x.shape[2]):
            frame, state = model.forward_stream(x[:, :, frame_idx : frame_idx + 1, :], state)
            frames.append(frame)
        y_stream = torch.cat(frames, dim=2)
    assert tuple(y.shape) == (1, 6, 4, 65)
    assert tuple(y_stream.shape) == tuple(y.shape)


def test_proposal_teacher_builder_tiny_forward() -> None:
    torch.manual_seed(0)
    model = build_sfc_locoformer_lite_plus_system(
        n_fft=64,
        hop_length=16,
        fs=8000,
        n_src=2,
        n_chan=1,
        n_bands=8,
        d_inner=8,
        d_model=16,
        n_layers=1,
        n_heads=2,
        attention_dim=16,
        ffn_hidden_dim=(16, 16),
        flash_attention=False,
        scaling=False,
    ).eval()
    wav = torch.randn(1, 1, 256)
    with torch.no_grad():
        est = model(wav)
    assert tuple(est.shape) == (1, 2, 1, 256)


def test_middle_and_edge_proposal_builders_expose_waveform_wrappers() -> None:
    middle = build_hierarchical_sfc_ffi_lite_system(
        n_fft=64,
        hop_length=16,
        fs=8000,
        n_src=2,
        n_chan=1,
        pre_bands=12,
        mid_bands=8,
        bottleneck_bands=4,
        d_model=8,
        freq_preprocess_enabled=False,
        css_segment_size=1,
        css_shift_size=1,
    )
    edge = build_edgefusion_sfc_distilled_system(
        n_fft=64,
        hop_length=16,
        fs=8000,
        n_src=2,
        n_chan=1,
        preset="tiny",
        freq_preprocess_enabled=False,
        css_segment_size=1,
        css_shift_size=1,
    )
    assert middle.model.n_src == 2
    assert edge.model.n_src == 2


def test_distillation_task_requires_teacher_when_weighted() -> None:
    student = torch.nn.Identity()
    loss = torch.nn.L1Loss()
    optimizer_config = object()
    try:
        TeacherStudentDistillationTask(
            model=student,
            loss=loss,
            n_fft=64,
            hop_length=16,
            optimizer_config=optimizer_config,  # type: ignore[arg-type]
            teacher_loss_weight=0.1,
        )
    except ValueError as exc:
        assert "teacher_model" in str(exc)
    else:
        raise AssertionError("Expected teacher_loss_weight without teacher_model to fail")


def test_composite_separation_spectral_loss_is_differentiable() -> None:
    torch.manual_seed(0)
    loss_fn = CompositeSeparationSpectralLoss(
        n_fft=32,
        hop_length=8,
        complex_ri_weight=0.5,
        log_magnitude_weight=0.2,
        multi_resolution_stft_weight=0.3,
        multi_resolution_stft_resolutions=((16, 4), (32, 8)),
        transient_weight=0.1,
    )
    est = torch.randn(2, 3, 1, 128, requires_grad=True)
    ref = torch.randn(2, 3, 1, 128)

    loss, components = loss_fn(est, ref)
    loss.backward()

    assert loss.ndim == 0
    assert set(components) == {"complex_ri", "log_magnitude", "multi_resolution_stft", "transient"}
    assert est.grad is not None
    assert torch.isfinite(est.grad).all()
