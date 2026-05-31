from __future__ import annotations

import torch

from BandSFCNetNPU.presets import build_band_sfc_net_npu_preset
from spectral_feature_compression.core.loss.composite_separation import CompositeSeparationSpectralLoss
from spectral_feature_compression.core.model.adaptive_mel_sfc_2d import AdaptiveMelBandSpec2d, OnlineAdaptiveMelSFC2D
from spectral_feature_compression.core.model.proposed_separation_models import (
    build_adaptive_mel_sfc_ablation_system,
    build_edgefusion_sfc_distilled_system,
    build_hierarchical_sfc_ffi_lite_system,
    build_sfc_locoformer_lite_plus_system,
    build_sfc_residual_refinement_system,
    build_sfc_sepreformer_multistem_system,
    build_sparse_unet_mel_sfc_music_system,
)
from spectral_feature_compression.core.model.residual_refinement_sfc_2d import OnlineResidualRefinementSFC2D
from spectral_feature_compression.core.model.source_split_sfc_2d import OnlineSourceSplitSFC2D
from spectral_feature_compression.core.model.sparse_unet_mel_sfc_2d import SparseUNetMelSFC2D
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
    torch.testing.assert_close(y_stream, y, rtol=1e-5, atol=1e-5)


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


def test_sparse_unet_mel_sfc_builder_forward_and_streaming_shape() -> None:
    torch.manual_seed(0)
    model = build_sparse_unet_mel_sfc_music_system(
        n_fft=64,
        hop_length=16,
        fs=8000,
        n_src=2,
        n_chan=1,
        d_model=8,
        branch_bands=(4, 4, 4),
        bottleneck_layers=1,
        freq_preprocess_enabled=False,
        css_segment_size=1,
        css_shift_size=1,
    ).eval()
    wav = torch.randn(1, 1, 256)
    with torch.no_grad():
        est = model(wav)
    assert tuple(est.shape) == (1, 2, 1, 256)

    core = SparseUNetMelSFC2D(
        n_freq=33,
        sample_rate=8000,
        n_src=2,
        n_chan=1,
        d_model=8,
        branch_bands=(4, 4, 4),
        bottleneck_layers=1,
    ).eval()
    x = torch.randn(1, 2, 4, 33)
    with torch.no_grad():
        y = core(x)
        state = core.init_stream_state(batch_size=1, dtype=x.dtype)
        frames = []
        for frame_idx in range(x.shape[2]):
            frame, state = core.forward_stream(x[:, :, frame_idx : frame_idx + 1, :], state)
            frames.append(frame)
        y_stream = torch.cat(frames, dim=2)
    assert tuple(y.shape) == (1, 4, 4, 33)
    assert tuple(y_stream.shape) == tuple(y.shape)
    torch.testing.assert_close(y_stream, y, rtol=1e-5, atol=1e-5)
    assert core.state_size_bytes(dtype=torch.float16) < 32 * 1024


def test_sfc_sepreformer_source_split_builder_forward_and_streaming_shape() -> None:
    torch.manual_seed(0)
    model = build_sfc_sepreformer_multistem_system(
        n_fft=64,
        hop_length=16,
        fs=8000,
        n_src=2,
        n_chan=1,
        n_bands=8,
        d_model=8,
        n_shared_layers=1,
        n_source_layers=1,
        freq_preprocess_enabled=False,
        css_segment_size=1,
        css_shift_size=1,
    ).eval()
    wav = torch.randn(1, 1, 256)
    with torch.no_grad():
        est = model(wav)
    assert tuple(est.shape) == (1, 2, 1, 256)

    core = OnlineSourceSplitSFC2D(
        n_freq=33,
        n_bands=8,
        sample_rate=8000,
        n_src=2,
        n_chan=1,
        d_model=8,
        n_shared_layers=1,
        n_source_layers=1,
    ).eval()
    x = torch.randn(1, 2, 4, 33)
    with torch.no_grad():
        y = core(x)
        state = core.init_stream_state(batch_size=1, dtype=x.dtype)
        frames = []
        for frame_idx in range(x.shape[2]):
            frame, state = core.forward_stream(x[:, :, frame_idx : frame_idx + 1, :], state)
            frames.append(frame)
        y_stream = torch.cat(frames, dim=2)
    assert tuple(y.shape) == (1, 4, 4, 33)
    assert tuple(y_stream.shape) == tuple(y.shape)
    torch.testing.assert_close(y_stream, y, rtol=1e-5, atol=1e-5)
    assert tuple(core.source_splitter(torch.randn(1, 8, 2, 8)).shape) == (1, 16, 2, 8)
    assert core.state_size_bytes(dtype=torch.float16) < 32 * 1024


def test_sfc_residual_refinement_builder_forward_and_streaming_shape() -> None:
    torch.manual_seed(0)
    model = build_sfc_residual_refinement_system(
        n_fft=64,
        hop_length=16,
        fs=8000,
        n_src=2,
        n_chan=1,
        n_bands=8,
        d_model=8,
        n_layers=1,
        long_branch_layers=1,
        refinement_layers=1,
        freq_preprocess_enabled=False,
        css_segment_size=1,
        css_shift_size=1,
    ).eval()
    wav = torch.randn(1, 1, 256)
    with torch.no_grad():
        est = model(wav)
    assert tuple(est.shape) == (1, 2, 1, 256)

    core = OnlineResidualRefinementSFC2D(
        n_freq=33,
        n_bands=8,
        sample_rate=8000,
        n_src=2,
        n_chan=1,
        d_model=8,
        n_layers=1,
        long_branch_layers=1,
        refinement_layers=1,
    ).eval()
    x = torch.randn(1, 2, 4, 33)
    with torch.no_grad():
        y = core(x)
        state = core.init_stream_state(batch_size=1, dtype=x.dtype)
        frames = []
        for frame_idx in range(x.shape[2]):
            frame, state = core.forward_stream(x[:, :, frame_idx : frame_idx + 1, :], state)
            frames.append(frame)
        y_stream = torch.cat(frames, dim=2)
    assert tuple(y.shape) == (1, 4, 4, 33)
    assert tuple(y_stream.shape) == tuple(y.shape)
    torch.testing.assert_close(y_stream, y, rtol=1e-5, atol=1e-5)
    assert core.correction_head.correction_scale.requires_grad
    assert core.long_temporal_refiner.delta_scale.requires_grad
    assert core.state_size_bytes(dtype=torch.float16) < 32 * 1024


def test_adaptive_mel_sfc_builder_exposes_low_frequency_overlap_controls() -> None:
    torch.manual_seed(0)
    spec = AdaptiveMelBandSpec2d(
        n_freq=33,
        n_bands=12,
        sample_rate=8000,
        low_freq_hz=1000.0,
        low_freq_band_fraction=0.5,
        overlap_factor=1.2,
        low_freq_overlap_factor=2.0,
    )
    assert tuple(spec.basis.shape) == (1, 12, 1, 33)
    assert spec.manifest()["n_bands"] == 12
    low_bins = 9
    overlap_count = (spec.basis[0, :, 0, :] > 0.0).sum(dim=0).to(torch.float32)
    assert float(overlap_count[:low_bins].mean()) > float(overlap_count[low_bins:].mean())

    model = build_adaptive_mel_sfc_ablation_system(
        n_fft=64,
        hop_length=16,
        fs=8000,
        n_src=2,
        n_chan=1,
        n_bands=12,
        d_model=8,
        n_layers=1,
        low_freq_hz=1000.0,
        low_freq_band_fraction=0.5,
        freq_preprocess_enabled=False,
        css_segment_size=1,
        css_shift_size=1,
    ).eval()
    wav = torch.randn(1, 1, 256)
    with torch.no_grad():
        est = model(wav)
    assert tuple(est.shape) == (1, 2, 1, 256)

    core = OnlineAdaptiveMelSFC2D(
        n_freq=33,
        n_bands=12,
        sample_rate=8000,
        n_src=2,
        n_chan=1,
        d_model=8,
        n_layers=1,
        low_freq_hz=1000.0,
        low_freq_band_fraction=0.5,
    ).eval()
    x = torch.randn(1, 2, 4, 33)
    with torch.no_grad():
        y = core(x)
        state = core.init_stream_state(batch_size=1, dtype=x.dtype)
        frames = []
        for frame_idx in range(x.shape[2]):
            frame, state = core.forward_stream(x[:, :, frame_idx : frame_idx + 1, :], state)
            frames.append(frame)
        y_stream = torch.cat(frames, dim=2)
    assert tuple(y.shape) == (1, 4, 4, 33)
    assert tuple(y_stream.shape) == tuple(y.shape)
    torch.testing.assert_close(y_stream, y, rtol=1e-5, atol=1e-5)
    assert core.state_size_bytes(dtype=torch.float16) < 32 * 1024


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


class _ToyLatentSeparator(torch.nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.latent = torch.nn.Conv1d(1, 1, kernel_size=1)
        self.scale = scale

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        _ = self.latent(wav)
        return torch.stack([wav * self.scale, wav * (1.0 - self.scale)], dim=1)


def test_distillation_task_supports_activity_mask_logit_and_latent_losses() -> None:
    torch.manual_seed(0)
    student = _ToyLatentSeparator(scale=0.4)
    teacher = _ToyLatentSeparator(scale=0.6)
    task = TeacherStudentDistillationTask(
        model=student,
        teacher_model=teacher,
        loss=torch.nn.L1Loss(),
        n_fft=32,
        hop_length=8,
        optimizer_config=object(),  # type: ignore[arg-type]
        teacher_mask_loss_weight=0.1,
        teacher_logit_loss_weight=0.1,
        source_activity_loss_weight=0.1,
        latent_distillation_weight=0.1,
        student_latent_modules=("latent",),
        teacher_latent_modules=("latent",),
    )
    wav = torch.randn(2, 1, 128)
    ref = torch.stack([wav * 0.7, wav * 0.3], dim=1)

    task._student_latents.clear()
    est, student_aux = task._forward_model(task.model, wav, ref, "training", css_validation=False)
    student_aux["latents"] = dict(task._student_latents)
    teacher_est, teacher_aux = task._teacher_forward(wav, ref=ref, log_prefix="training")

    losses = [
        task._source_activity_l1(est, ref),
        task._mask_or_logit_loss(est, teacher_est, wav, student_aux, teacher_aux, logit=False),
        task._mask_or_logit_loss(est, teacher_est, wav, student_aux, teacher_aux, logit=True),
        task._latent_distillation_loss(student_aux, teacher_aux),
    ]
    assert all(loss.ndim == 0 and torch.isfinite(loss) for loss in losses)
