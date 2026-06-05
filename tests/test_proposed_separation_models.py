from __future__ import annotations

from pathlib import Path

import torch

from BandSFCNetNPU.presets import build_band_sfc_net_npu_preset
from spectral_feature_compression.core.loss.composite_separation import CompositeSeparationSpectralLoss
from spectral_feature_compression.core.model.adaptive_mel_locoformer_lite_sfc_2d import (
    OnlineAdaptiveMelLocoformerLiteSFC2D,
)
from spectral_feature_compression.core.model.adaptive_mel_sfc_2d import AdaptiveMelBandSpec2d, OnlineAdaptiveMelSFC2D
from spectral_feature_compression.core.model.foa_event_query_prompted_sfc import (
    FOAEventQueryPromptedAsymmetricSFC2D,
)
from spectral_feature_compression.core.model.prompted_asymmetric_sfc_2d import OnlinePromptedAsymmetricSFC2D
from spectral_feature_compression.core.model.proposed_separation_models import (
    build_adaptive_mel_loco_cnb_npu_system,
    build_adaptive_mel_locoformer_lite_system,
    build_adaptive_mel_sfc_ablation_system,
    build_edgefusion_sfc_distilled_system,
    build_hierarchical_sfc_ffi_lite_system,
    build_prompted_asymmetric_sfc_foa_event_query_strong_system,
    build_prompted_asymmetric_sfc_unified_system,
    build_sfc_locoformer_lite_plus_system,
    build_sfc_residual_refinement_system,
    build_sfc_sepreformer_multistem_system,
    build_sparse_unet_mel_sfc_music_system,
)
from spectral_feature_compression.core.model.residual_refinement_sfc_2d import OnlineResidualRefinementSFC2D
from spectral_feature_compression.core.model.source_split_sfc_2d import OnlineSourceSplitSFC2D
from spectral_feature_compression.core.model.sparse_unet_mel_sfc_2d import SparseUNetMelSFC2D
from spectral_feature_compression.core.tasks.distillation_task import TeacherStudentDistillationTask
from tools.online.export_onnx_online_model import (
    build_model_system_from_recipe_config,
    merge_task_model_mapping,
    merge_top_level_scalars,
    resolve_value,
)


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
        fullband_capacity_layers=0,
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
        fullband_capacity_layers=0,
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

    deploy_core = SparseUNetMelSFC2D(
        n_freq=512,
        sample_rate=44100,
        n_src=3,
        n_chan=1,
        d_model=64,
        branch_bands=(24, 32, 24),
        bottleneck_layers=2,
        fullband_capacity_hidden=8192,
        fullband_capacity_layers=1,
    ).eval()
    deploy_params = sum(p.numel() for p in deploy_core.parameters())
    assert 2_000_000 <= deploy_params <= 7_000_000
    assert deploy_core.state_size_bytes(dtype=torch.float16) < 192 * 1024


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
        shared_capacity_layers=0,
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
        shared_capacity_layers=0,
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

    deploy_core = OnlineSourceSplitSFC2D(
        n_freq=512,
        n_bands=64,
        n_src=3,
        n_chan=1,
        d_model=32,
        n_shared_layers=1,
        n_source_layers=2,
        shared_capacity_hidden=6144,
        shared_capacity_layers=4,
    ).eval()
    deploy_params = sum(p.numel() for p in deploy_core.parameters())
    assert 2_000_000 <= deploy_params <= 7_000_000
    assert deploy_core.state_size_bytes(dtype=torch.float16) < 192 * 1024


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
        capacity_mixer_layers=0,
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
        capacity_mixer_layers=0,
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

    deploy_core = OnlineResidualRefinementSFC2D(
        n_freq=512,
        n_bands=64,
        sample_rate=44100,
        n_src=3,
        n_chan=1,
        d_model=24,
        n_layers=2,
        long_branch_layers=1,
        refinement_layers=1,
        capacity_mixer_hidden=8192,
        capacity_mixer_layers=4,
    ).eval()
    deploy_params = sum(p.numel() for p in deploy_core.parameters())
    assert 2_000_000 <= deploy_params <= 7_000_000
    assert deploy_core.state_size_bytes(dtype=torch.float16) < 192 * 1024


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
        capacity_mixer_layers=0,
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
        capacity_mixer_layers=0,
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

    deploy_core = OnlineAdaptiveMelSFC2D(
        n_freq=512,
        n_bands=80,
        sample_rate=44100,
        n_src=3,
        n_chan=1,
        d_model=24,
        n_layers=6,
        capacity_mixer_hidden=8192,
        capacity_mixer_layers=4,
    ).eval()
    deploy_params = sum(p.numel() for p in deploy_core.parameters())
    assert 2_000_000 <= deploy_params <= 7_000_000
    assert deploy_core.state_size_bytes(dtype=torch.float16) < 192 * 1024


def test_adaptive_mel_loco_cnb_builder_forward_shape() -> None:
    torch.manual_seed(0)
    model = build_adaptive_mel_loco_cnb_npu_system(
        n_fft=64,
        hop_length=16,
        fs=8000,
        n_src=2,
        n_chan=1,
        n_bands=8,
        channels=8,
        num_stages=1,
        time_kernel=1,
        freq_kernel=3,
        dilation_cycle=(1,),
        stage_type="loco_cnb",
        cnb_kernel=4,
        cnb_dilation_schedule=(1, 2, 3),
        routing_normalization="softmax",
        use_attn=False,
        attn_window=8,
        num_heads=2,
        head_dim=4,
        pooled_mixer_hidden=0,
        pooled_mixer_hidden_schedule=(0,),
        encoder_capacity_mixer_hidden=16,
        encoder_capacity_mixer_layers=1,
        decoder_capacity_mixer_hidden=16,
        decoder_capacity_mixer_layers=1,
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
    assert model.model.core.stage_type == "loco_cnb"
    assert model.model.core.band_spec_type == "adaptive_mel"
    assert model.model.core.encoder_capacity_mixer_layers == 1
    assert model.model.core.decoder_capacity_mixer_layers == 1
    assert model.model.core.cnb_kernel == 4
    assert model.model.core.num_heads == 2
    assert model.model.core.head_dim == 4


def test_adaptive_mel_loco_cnb_recipe_config_resolves_and_instantiates() -> None:
    config_path = Path(
        "recipes/dnr/models/band-sfc-net-npu.adaptive-mel-loco-cnb.soft-query.rt192k.fp512keep475/config.yaml"
    )
    top = merge_top_level_scalars(config_path)
    model_cfg = merge_task_model_mapping(config_path)
    context = {**top, **model_cfg}
    assert resolve_value(model_cfg["preset"], context) == "adaptive_mel_loco_cnb_soft_band_query"
    assert resolve_value(model_cfg["stage_type"], context) == "loco_cnb"
    assert resolve_value(model_cfg["band_spec_type"], context) == "adaptive_mel"
    assert resolve_value(model_cfg["freq_preprocess_keep_bins"], context) == 475
    assert resolve_value(model_cfg["freq_preprocess_target_bins"], context) == 512

    system = build_model_system_from_recipe_config(config_path).eval()
    core = system.model.core
    assert core.n_freq == 512
    assert core.stage_type == "loco_cnb"
    assert core.band_spec_type == "adaptive_mel"
    assert core.band_spec.explicit_bin_frequencies is True
    assert sum(p.numel() for p in core.parameters()) < 7_000_000
    assert core.state_size_bytes(dtype=torch.float16) < 192 * 1024


def test_adaptive_mel_loco_cnb_stability_fix_recipes_resolve_and_instantiate() -> None:
    checks = [
        (
            "recipes/dnr/models/"
            "band-sfc-net-npu.adaptive-mel-loco-cnb.stable-soft-query.rt192k.fp512keep475/config.yaml",
            "adaptive_mel_loco_cnb_stable_soft_band_query",
            36,
            48,
        ),
        (
            "recipes/dnr/models/"
            "band-sfc-net-npu.adaptive-mel-loco-cnb.stable-crossattn-query.rt192k.fp512keep475/config.yaml",
            "adaptive_mel_loco_cnb_stable_crossattn_query",
            36,
            48,
        ),
        (
            "recipes/dnr/models/"
            "band-sfc-net-npu.adaptive-mel-loco-cnb.band56-soft-query.rt192k.fp512keep475/config.yaml",
            "adaptive_mel_loco_cnb_band56_soft_band_query",
            28,
            56,
        ),
    ]
    for raw_path, preset, channels, n_bands in checks:
        config_path = Path(raw_path)
        top = merge_top_level_scalars(config_path)
        model_cfg = merge_task_model_mapping(config_path)
        context = {**top, **model_cfg}
        assert resolve_value(model_cfg["preset"], context) == preset
        assert resolve_value(model_cfg["channels"], context) == channels
        assert resolve_value(model_cfg["n_bands"], context) == n_bands
        assert resolve_value(model_cfg["residual_head"], context) is False

        system = build_model_system_from_recipe_config(config_path).eval()
        core = system.model.core
        assert core.stage_type == "loco_cnb"
        assert core.band_spec_type == "adaptive_mel"
        assert core.channels == channels
        assert core.n_bands == n_bands
        assert core.residual_head is False
        assert sum(p.numel() for p in core.parameters()) < 4_000_000
        assert core.state_size_bytes(dtype=torch.float16) < 192 * 1024


def test_adaptive_mel_loco_cnb_distill_recipe_declares_teacher_task() -> None:
    config_path = Path(
        "recipes/dnr/models/band-sfc-net-npu.adaptive-mel-loco-cnb.soft-query.distill.rt192k.fp512keep475/config.yaml"
    )
    task_cfg = merge_task_model_mapping(config_path)
    top = merge_top_level_scalars(config_path)
    assert top["teacher_checkpoint_path"] is None
    # `merge_task_model_mapping` intentionally extracts only task.model.  Check
    # the raw file to avoid instantiating a distillation task without a teacher.
    text = config_path.read_text(encoding="utf-8")
    assert "TeacherStudentDistillationTask" in text
    assert "require_teacher_checkpoint: true" in text
    assert "build_sfc_locoformer_lite_plus_system" in text
    assert resolve_value(task_cfg["preset"], {**top, **task_cfg}) == "adaptive_mel_loco_cnb_soft_band_query"


def test_adaptive_mel_locoformer_lite_builder_forward_and_streaming_shape() -> None:
    torch.manual_seed(0)
    model = build_adaptive_mel_locoformer_lite_system(
        n_fft=64,
        hop_length=16,
        fs=8000,
        n_src=2,
        n_chan=1,
        n_bands=12,
        d_model=8,
        n_layers=2,
        dilation_cycle=(1, 2),
        capacity_mixer_layers=0,
        encoder_capacity_mixer_hidden=16,
        encoder_capacity_mixer_layers=1,
        decoder_capacity_mixer_hidden=16,
        decoder_capacity_mixer_layers=1,
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

    core = OnlineAdaptiveMelLocoformerLiteSFC2D(
        n_freq=33,
        n_bands=12,
        sample_rate=8000,
        n_src=2,
        n_chan=1,
        d_model=8,
        n_layers=2,
        dilation_cycle=(1, 2),
        capacity_mixer_layers=0,
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
    assert core.band_spec.manifest()["type"] == "adaptive_overlapped_mel"
    assert core.state_size_bytes(dtype=torch.float16) < 32 * 1024

    deploy_core = OnlineAdaptiveMelLocoformerLiteSFC2D(
        n_freq=512,
        n_bands=80,
        d_model=32,
        n_layers=4,
        capacity_mixer_hidden=6144,
        capacity_mixer_layers=4,
    ).eval()
    deploy_params = sum(p.numel() for p in deploy_core.parameters())
    assert 2_000_000 <= deploy_params <= 7_000_000
    assert deploy_core.state_size_bytes(dtype=torch.float16) < 192 * 1024

    io_balanced_core = OnlineAdaptiveMelLocoformerLiteSFC2D(
        n_freq=512,
        n_bands=80,
        d_model=40,
        n_layers=4,
        capacity_mixer_hidden=2048,
        capacity_mixer_layers=2,
        encoder_capacity_mixer_hidden=4096,
        encoder_capacity_mixer_layers=2,
        decoder_capacity_mixer_hidden=4096,
        decoder_capacity_mixer_layers=2,
    ).eval()
    io_params = sum(p.numel() for p in io_balanced_core.parameters())
    encoder_params = sum(p.numel() for p in io_balanced_core.encoder_capacity_mixers.parameters())
    separator_capacity_params = sum(p.numel() for p in io_balanced_core.capacity_mixers.parameters())
    decoder_params = sum(p.numel() for p in io_balanced_core.decoder_capacity_mixers.parameters())
    assert 2_000_000 <= io_params <= 7_000_000
    assert io_balanced_core.state_size_bytes(dtype=torch.float16) < 192 * 1024
    assert encoder_params > separator_capacity_params
    assert decoder_params > separator_capacity_params


def test_prompted_asymmetric_sfc_builder_forward_and_streaming_shape() -> None:
    torch.manual_seed(0)
    model = build_prompted_asymmetric_sfc_unified_system(
        n_fft=64,
        hop_length=16,
        fs=8000,
        n_src=2,
        n_chan=1,
        n_bands=8,
        d_model=8,
        n_encoder_layers=1,
        n_decoder_layers=1,
        encoder_capacity_layers=0,
        source_capacity_layers=0,
        prompt_labels=("speech", "music"),
        freq_preprocess_enabled=False,
        css_segment_size=1,
        css_shift_size=1,
    ).eval()
    wav = torch.randn(1, 1, 256)
    with torch.no_grad():
        est = model(wav)
    assert tuple(est.shape) == (1, 2, 1, 256)

    core = OnlinePromptedAsymmetricSFC2D(
        n_freq=33,
        n_bands=8,
        sample_rate=8000,
        n_src=2,
        n_chan=1,
        d_model=8,
        n_encoder_layers=1,
        n_decoder_layers=1,
        encoder_capacity_layers=0,
        source_capacity_layers=0,
        prompt_labels=("speech", "music"),
    ).eval()
    x = torch.randn(1, 2, 4, 33)
    external_prompts = torch.randn(1, 2, 8)
    with torch.no_grad():
        y = core(x, prompt_embeddings=external_prompts)
        state = core.init_stream_state(batch_size=1, dtype=x.dtype)
        frames = []
        for frame_idx in range(x.shape[2]):
            frame, state = core.forward_stream(
                x[:, :, frame_idx : frame_idx + 1, :],
                state,
                prompt_embeddings=external_prompts,
            )
            frames.append(frame)
        y_stream = torch.cat(frames, dim=2)
    assert tuple(y.shape) == (1, 4, 4, 33)
    assert tuple(y_stream.shape) == tuple(y.shape)
    torch.testing.assert_close(y_stream, y, rtol=1e-5, atol=1e-5)
    assert core.prompt_manifest()["labels"] == ["speech", "music"]
    assert tuple(core.prompt_splitter(torch.randn(1, 8, 2, 8), core._default_prompts()).shape) == (1, 16, 2, 8)
    assert core.state_size_bytes(dtype=torch.float16) < 32 * 1024

    deploy_core = OnlinePromptedAsymmetricSFC2D(
        n_freq=512,
        n_bands=64,
        d_model=32,
        n_encoder_layers=3,
        n_decoder_layers=1,
        encoder_capacity_hidden=6144,
        encoder_capacity_layers=3,
        source_capacity_hidden=2048,
        source_capacity_layers=1,
    ).eval()
    deploy_params = sum(p.numel() for p in deploy_core.parameters())
    assert 2_000_000 <= deploy_params <= 7_000_000
    assert deploy_core.state_size_bytes(dtype=torch.float16) < 192 * 1024
    assert deploy_core.prompt_manifest()["static_export_prompts"] is True


def test_foa_event_query_prompted_asymmetric_sfc_uses_class_queries() -> None:
    torch.manual_seed(0)
    model = build_prompted_asymmetric_sfc_foa_event_query_strong_system(
        n_fft=64,
        hop_length=16,
        fs=8000,
        n_queries=2,
        n_src=2,
        n_chan=4,
        output_n_chan=1,
        n_event_classes=5,
        n_bands=8,
        d_model=16,
        n_heads=4,
        n_encoder_layers=1,
        n_decoder_layers=1,
        ffn_mult=2,
        dropout=0.0,
        conv_kernel_size=(3, 3),
        routing_kernel_size=(3, 3),
        scaling=False,
        css_segment_size=1,
        css_shift_size=1,
    ).eval()
    wav = torch.randn(1, 4, 256)
    event_condition = torch.zeros(1, 2, 5)
    event_condition[0, 0, 1] = 1.0
    event_condition[0, 1, 3] = 1.0
    with torch.no_grad():
        est = model(wav, event_condition=event_condition)
    assert tuple(est.shape) == (1, 2, 1, 256)

    core = FOAEventQueryPromptedAsymmetricSFC2D(
        n_freq=33,
        n_bands=8,
        sample_rate=8000,
        n_queries=2,
        n_event_classes=5,
        n_chan=4,
        output_n_chan=1,
        d_model=16,
        n_heads=4,
        n_encoder_layers=1,
        n_decoder_layers=1,
        ffn_mult=2,
        dropout=0.0,
        conv_kernel_size=(3, 3),
        routing_kernel_size=(3, 3),
    ).eval()
    x = torch.randn(1, 8, 4, 33)
    soft_condition = torch.softmax(torch.randn(1, 2, 5), dim=-1)
    alt_condition = torch.zeros(1, 2, 5)
    alt_condition[0, 0, 4] = 1.0
    alt_condition[0, 1, 0] = 1.0
    with torch.no_grad():
        y = core(x, event_condition=soft_condition)
        y_alt = core(x, event_condition=alt_condition, condition_mode="onehot")
    assert tuple(y.shape) == (1, 4, 4, 33)
    assert tuple(y_alt.shape) == tuple(y.shape)
    assert not torch.allclose(y, y_alt)
    manifest = core.event_query_manifest()
    assert manifest["foa_channels"] == 4
    assert manifest["output_channels"] == 1
    assert manifest["sfc_query_compression"] == "cross_attention"
    assert manifest["n_event_classes"] == 5
    assert manifest["npu_target"] is False


def test_event_conditioned_sup_task_css_repeats_conditions_per_segment() -> None:
    from pathlib import Path
    import sys

    aiaccel_path = str(Path(__file__).resolve().parents[1] / "aiaccel")
    if aiaccel_path not in sys.path:
        sys.path.insert(0, aiaccel_path)
    from spectral_feature_compression.core.tasks.conditioned_sup_task import EventConditionedSupTask

    class _ConditionCheckingWrapper(torch.nn.Module):
        fs = 8
        css_segment_size = 4
        css_shift_size = 2
        css_batch_size = 2

        def forward(self, wav: torch.Tensor, *, event_condition: torch.Tensor):
            assert event_condition.shape[0] == wav.shape[0]
            return wav.new_zeros(wav.shape[0], 2, 1, wav.shape[-1])

    task = object.__new__(EventConditionedSupTask)
    wav = torch.randn(2, 4, 80)
    event_condition = torch.randn(2, 2, 5)
    out = task._css_with_event_condition(
        _ConditionCheckingWrapper(),
        wav,
        ref=None,
        event_condition=event_condition,
    )
    assert tuple(out.shape) == (2, 2, 1, 80)


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
