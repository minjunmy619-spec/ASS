from __future__ import annotations

from pathlib import Path
import tempfile

from omegaconf import OmegaConf

import torch

import pytest

from BandSFCNetNPU.presets import build_band_sfc_net_npu_preset
from spectral_feature_compression.core.loss.composite_separation import CompositeSeparationSpectralLoss
from spectral_feature_compression.core.model.adaptive_mel_locoformer_lite_sfc_2d import (
    OnlineAdaptiveMelLocoformerLiteSFC2D,
)
from spectral_feature_compression.core.model.adaptive_mel_sfc_2d import AdaptiveMelBandSpec2d, OnlineAdaptiveMelSFC2D
from spectral_feature_compression.core.model.foa_event_query_prompted_sfc import (
    FOAEventQueryPromptedAsymmetricSFC2D,
)
from spectral_feature_compression.core.model.online_sfc_2d import ChannelAffine2d, RMSNorm2d
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
    build_source_aware_melband_loco_cnb_student_npu_system,
    build_source_aware_melband_roformer_strong_student_npu_system,
    build_source_aware_melband_roformer_student_npu_system,
    build_source_aware_melband_roformer_teacher_system,
    build_source_aware_residual_sfc_system,
    build_sparse_unet_mel_sfc_music_system,
    build_tvconv_pyramid_convgru_npu_separator_system,
    build_tvconv_pyramid_convlstm_npu_separator_system,
    build_tvconv_pyramid_npu_separator_system,
    build_tvconv_pyramid_sfclite_query_npu_separator_system,
    build_tvconv_pyramid_sourceaware_sfclite_convgru_npu_separator_system,
)
from spectral_feature_compression.core.model.residual_refinement_sfc_2d import OnlineResidualRefinementSFC2D
from spectral_feature_compression.core.model.source_aware_melband_loco_cnb_student_sfc_2d import (
    OnlineSourceAwareMelBandLocoCNBStudentSFC2D,
)
from spectral_feature_compression.core.model.source_aware_melband_roformer import SourceAwareMelBandRoformer2D
from spectral_feature_compression.core.model.source_aware_melband_strong_student_sfc_2d import (
    OnlineSourceAwareMelBandStrongStudentSFC2D,
)
from spectral_feature_compression.core.model.source_aware_melband_student_sfc_2d import (
    OnlineSourceAwareMelBandStudentSFC2D,
)
from spectral_feature_compression.core.model.source_aware_residual_sfc_2d import OnlineSourceAwareResidualSFC2D
from spectral_feature_compression.core.model.source_split_sfc_2d import OnlineSourceSplitSFC2D
from spectral_feature_compression.core.model.sparse_unet_mel_sfc_2d import SparseUNetMelSFC2D
from spectral_feature_compression.core.model.tvconv_pyramid_npu_separator_2d import TVConvPyramidNPUSeparator2D
from spectral_feature_compression.core.tasks.distillation_task import (
    TeacherStudentDistillationTask,
    _load_model_checkpoint,
)
from tools.online.export_onnx_online_model import (
    build_model_system_from_recipe_config,
    load_pcen_preprocess_metadata,
    merge_task_model_mapping,
    merge_top_level_scalars,
    resolve_value,
)


def test_source_aware_melband_public_lazy_exports() -> None:
    import spectral_feature_compression as sfc

    assert sfc.SourceAwareMelBandRoformer2D is SourceAwareMelBandRoformer2D
    assert sfc.OnlineSourceAwareMelBandStudentSFC2D is OnlineSourceAwareMelBandStudentSFC2D
    assert sfc.OnlineSourceAwareMelBandStrongStudentSFC2D is OnlineSourceAwareMelBandStrongStudentSFC2D
    assert sfc.OnlineSourceAwareMelBandLocoCNBStudentSFC2D is OnlineSourceAwareMelBandLocoCNBStudentSFC2D
    assert sfc.TVConvPyramidNPUSeparator2D is TVConvPyramidNPUSeparator2D


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


def test_tvconv_pyramid_npu_forward_streaming_and_recipe_budget() -> None:
    torch.manual_seed(0)
    core = TVConvPyramidNPUSeparator2D(
        n_freq=64,
        n_src=2,
        n_chan=1,
        base_channels=16,
        bottleneck_channels=32,
        n_down=3,
        n_blocks=2,
        expansion=2,
        time_kernel=3,
        time_dilation_cycle=(1, 2),
        band_kernel=3,
        mask_hidden=16,
        real_mask_scale=1.5,
    ).eval()
    x = torch.randn(1, 2, 5, 64)
    with torch.no_grad():
        y, aux = core(x, return_aux=True)
        state = core.init_stream_state(batch_size=1, dtype=x.dtype)
        frames = []
        for frame_idx in range(x.shape[2]):
            frame, state = core.forward_stream(x[:, :, frame_idx : frame_idx + 1, :], state)
            frames.append(frame)
        y_stream = torch.cat(frames, dim=2)
    assert tuple(y.shape) == (1, 4, 5, 64)
    assert tuple(aux["mask"].shape) == (1, 4, 5, 64)
    assert tuple(aux["mask_logits"].shape) == (1, 4, 5, 64)
    assert core.source_head.real_mask_scale == pytest.approx(1.5)
    torch.testing.assert_close(y_stream, y, rtol=1e-5, atol=1e-5)

    config_path = Path(
        "recipes/dnr/models/tvconv-pyramid-npu.speech-music-residual-sfx.robust-distill.rt192k.fp512keep475/config.yaml"
    )
    config = OmegaConf.load(config_path)
    assert str(config.datamodule._target_).endswith("OnTheFlyStemDataModule")
    assert len(config.datamodule.synthesis.synthesis_profiles) == 4
    assert config.task.mixture_consistency_weight == 0.0
    assert config.task.request_model_aux is True
    assert config.task.require_model_aux is True
    assert config.task.teacher_logit_loss_weight == pytest.approx(0.05)
    assert config.task.distillation_band_mapping == "linear"
    assert config.task.mask_aux_alignment == "shared_prefix"
    assert config.task.model.postprocess_enabled is False
    assert config.task.model.postprocess_mixture_consistency == "power"
    assert config.task.model.postprocess_final_mixture_consistency == "power"
    assert config.task.model.postprocess_wiener_blend == pytest.approx(0.25)
    assert config.task.model.postprocess_leakage_gate_enabled is False
    assert config.task.model.postprocess_leakage_gate_threshold_db == pytest.approx(12.0)
    assert config.task.model.postprocess_leakage_gate_attenuation_db == pytest.approx(6.0)
    assert config.task.model.postprocess_residual_source_index == 2
    assert config.task.model.postprocess_misi_iterations == 0
    assert config.task.model.postprocess_misi_eps == pytest.approx(1.0e-8)
    assert config.task.teacher_model.postprocess_enabled is False
    assert config.task.teacher_model.postprocess_mixture_consistency == "power"
    assert config.task.teacher_model.postprocess_final_mixture_consistency == "power"
    assert config.task.teacher_model.postprocess_wiener_blend == pytest.approx(0.25)
    assert config.task.teacher_model.postprocess_leakage_gate_enabled is False
    assert config.task.teacher_model.postprocess_residual_source_index is None
    assert config.task.teacher_model.postprocess_misi_iterations == 0
    assert config.task.source_loss_weight_normalization == "full_mean"
    assert config.task.source_weighted_snr_loss_weight > 0.0
    assert config.task.residual_source_loss_weight > 0.0

    top = merge_top_level_scalars(config_path)
    model_cfg = merge_task_model_mapping(config_path)
    context = {**top, **model_cfg}
    assert str(resolve_value(model_cfg["_target_"], context)).endswith("build_tvconv_pyramid_npu_separator_system")
    assert resolve_value(model_cfg["n_down"], context) == 4
    assert resolve_value(model_cfg["capacity_hidden"], context) == 0

    system = build_model_system_from_recipe_config(config_path).eval()
    recipe_core = system.model.core
    assert recipe_core.n_freq == 512
    assert recipe_core.n_down == 4
    assert recipe_core.n_blocks == 6
    assert recipe_core.source_head.real_mask_scale == pytest.approx(1.5)
    params = sum(p.numel() for p in recipe_core.parameters())
    assert 2_000_000 <= params <= 8_000_000
    assert recipe_core.state_size_bytes(dtype=torch.float16) < 192 * 1024
    assert system.model.residual_source_enabled is True
    assert system.postprocessor is None
    assert system.phase_consistency is None


def test_tvconv_pyramid_mask_logit_smoothing_streaming_and_recipe_budget() -> None:
    torch.manual_seed(0)
    core = TVConvPyramidNPUSeparator2D(
        n_freq=64,
        n_src=2,
        n_chan=1,
        base_channels=16,
        bottleneck_channels=32,
        n_down=3,
        n_blocks=2,
        expansion=2,
        time_kernel=3,
        time_dilation_cycle=(1, 2),
        band_kernel=3,
        source_head_type="folded_source_aware",
        source_mixer_layers=1,
        mask_hidden=16,
        mask_logit_smoothing_kernel=3,
        mask_logit_smoothing_blend=0.25,
    ).eval()
    x = torch.randn(1, 2, 5, 64)

    with torch.no_grad():
        y, aux = core(x, return_aux=True)
        state = core.init_stream_state(batch_size=1, dtype=x.dtype)
        frames = []
        for frame_idx in range(x.shape[2]):
            frame, state = core.forward_stream(x[:, :, frame_idx : frame_idx + 1, :], state)
            frames.append(frame)
        y_stream = torch.cat(frames, dim=2)

    assert tuple(y.shape) == (1, 4, 5, 64)
    assert aux["mask_logits_smoothing_kernel"] == 3
    assert aux["mask_logits_smoothing_blend"] == pytest.approx(0.25)
    assert core.source_head.has_stream_state is True
    assert len(state) == 3
    assert tuple(state[-1].shape) == (1, 4, 2, 64)
    assert core.stream_context_frames() == 8
    torch.testing.assert_close(y_stream, y, rtol=1e-5, atol=1e-5)

    config_path = Path(
        "recipes/dnr/models/"
        "tvconv-pyramid-sourceaware-sfclite-convgru-smoothlogit-npu."
        "speech-music-residual-sfx.robust-distill.rt192k.fp512keep475/"
        "config.yaml"
    )
    top = merge_top_level_scalars(config_path)
    model_cfg = merge_task_model_mapping(config_path)
    context = {**top, **model_cfg}
    assert str(resolve_value(model_cfg["_target_"], context)).endswith(
        "build_tvconv_pyramid_sourceaware_sfclite_convgru_npu_separator_system"
    )
    assert resolve_value(model_cfg["mask_logit_smoothing_kernel"], context) == 3
    assert resolve_value(model_cfg["mask_logit_smoothing_blend"], context) == pytest.approx(0.25)

    system = build_model_system_from_recipe_config(config_path).eval()
    recipe_core = system.model.core
    assert recipe_core.source_head_type == "folded_source_aware"
    assert recipe_core.source_head.logit_smoother.enabled is True
    assert recipe_core.source_head.logit_smoother.kernel_size == 3
    assert recipe_core.source_head.logit_smoother.blend == pytest.approx(0.25)
    assert recipe_core.state_size_bytes(dtype=torch.float16) < 192 * 1024
    params = sum(p.numel() for p in recipe_core.parameters())
    assert 3_000_000 <= params <= 8_000_000


def test_tvconv_pyramid_resize_conv_upsampling_streaming_and_recipe_budget() -> None:
    torch.manual_seed(0)
    core = TVConvPyramidNPUSeparator2D(
        n_freq=65,
        n_src=2,
        n_chan=1,
        base_channels=16,
        bottleneck_channels=32,
        n_down=3,
        n_blocks=2,
        expansion=2,
        time_kernel=3,
        time_dilation_cycle=(1, 2),
        band_kernel=3,
        upsample_mode="resize_conv",
        source_head_type="folded_source_aware",
        source_mixer_layers=1,
        mask_hidden=16,
        mask_logit_smoothing_kernel=3,
        mask_logit_smoothing_blend=0.25,
    ).eval()
    x = torch.randn(1, 2, 5, 65)

    with torch.no_grad():
        y, aux = core(x, return_aux=True)
        state = core.init_stream_state(batch_size=1, dtype=x.dtype)
        frames = []
        for frame_idx in range(x.shape[2]):
            frame, state = core.forward_stream(x[:, :, frame_idx : frame_idx + 1, :], state)
            frames.append(frame)
        y_stream = torch.cat(frames, dim=2)

    assert tuple(y.shape) == (1, 4, 5, 65)
    assert core.upsample_mode == "resize_conv"
    assert aux["mask_logits_smoothing_kernel"] == 3
    assert not any(isinstance(module, torch.nn.ConvTranspose2d) for module in core.modules())
    torch.testing.assert_close(y_stream, y, rtol=1e-5, atol=1e-5)

    config_path = Path(
        "recipes/dnr/models/"
        "tvconv-pyramid-sourceaware-sfclite-convgru-smoothup-smoothlogit-npu."
        "speech-music-residual-sfx.robust-distill.rt192k.fp512keep475/"
        "config.yaml"
    )
    top = merge_top_level_scalars(config_path)
    model_cfg = merge_task_model_mapping(config_path)
    context = {**top, **model_cfg}
    assert str(resolve_value(model_cfg["_target_"], context)).endswith(
        "build_tvconv_pyramid_sourceaware_sfclite_convgru_npu_separator_system"
    )
    assert resolve_value(model_cfg["upsample_mode"], context) == "resize_conv"
    assert resolve_value(model_cfg["mask_logit_smoothing_kernel"], context) == 3
    assert resolve_value(model_cfg["mask_logit_smoothing_blend"], context) == pytest.approx(0.25)

    system = build_model_system_from_recipe_config(config_path).eval()
    recipe_core = system.model.core
    assert recipe_core.upsample_mode == "resize_conv"
    assert recipe_core.source_head.logit_smoother.enabled is True
    assert recipe_core.state_size_bytes(dtype=torch.float16) < 192 * 1024
    assert not any(isinstance(module, torch.nn.ConvTranspose2d) for module in recipe_core.modules())
    params = sum(p.numel() for p in recipe_core.parameters())
    assert 3_000_000 <= params <= 8_000_000


def test_tvconv_pyramid_waveform_wrapper_returns_aux_masks() -> None:
    torch.manual_seed(0)
    model = build_tvconv_pyramid_npu_separator_system(
        n_fft=64,
        hop_length=16,
        fs=8000,
        n_src=3,
        n_chan=1,
        core_n_src=2,
        base_channels=8,
        bottleneck_channels=16,
        n_down=2,
        n_blocks=1,
        expansion=2,
        time_kernel=3,
        time_dilation_cycle=(1,),
        band_kernel=3,
        mask_hidden=8,
        real_mask_scale=1.5,
        residual_source_enabled=True,
        residual_source_index=2,
        freq_preprocess_enabled=False,
        css_segment_size=1,
        css_shift_size=1,
        postprocess_enabled=True,
        postprocess_final_mixture_consistency="uniform",
        postprocess_leakage_gate_enabled=True,
        postprocess_residual_source_index=2,
        postprocess_misi_iterations=1,
    ).eval()
    wav = torch.randn(1, 1, 512)

    with torch.no_grad():
        est, aux = model(wav, return_aux=True)

    assert tuple(est.shape) == (1, 3, 1, 512)
    assert model.postprocessor is not None
    assert model.phase_consistency is not None
    assert set(aux) == {
        "mask",
        "mask_domain",
        "mask_logits",
        "mask_logits_domain",
        "mask_logits_transform",
        "mask_logits_real_scale",
        "mask_logits_imag_scale",
    }
    assert aux["mask_domain"] == "packed_complex_mask"
    assert aux["mask_logits_domain"] == "tvconv_pyramid_complex_mask_logits"
    assert aux["mask_logits_transform"] == "sigmoid_tanh_complex_mask"
    assert aux["mask_logits_real_scale"] == pytest.approx(1.5)
    assert aux["mask_logits_imag_scale"] == pytest.approx(0.12)
    assert aux["mask"].ndim == 4
    assert aux["mask"].shape[1] == 4
    assert aux["mask"].shape[-1] == 33


@pytest.mark.parametrize(
    ("config_path", "target_suffix", "expected_type"),
    [
        (
            Path(
                "recipes/dnr/models/"
                "tvconv-pyramid-convgru-npu.speech-music-residual-sfx.robust-distill.rt192k.fp512keep475/"
                "config.yaml"
            ),
            "build_tvconv_pyramid_convgru_npu_separator_system",
            "gru",
        ),
        (
            Path(
                "recipes/dnr/models/"
                "tvconv-pyramid-convlstm-npu.speech-music-residual-sfx.robust-distill.rt192k.fp512keep475/"
                "config.yaml"
            ),
            "build_tvconv_pyramid_convlstm_npu_separator_system",
            "lstm",
        ),
    ],
)
def test_tvconv_pyramid_recurrent_recipe_variants_parse_and_build(
    config_path: Path,
    target_suffix: str,
    expected_type: str,
) -> None:
    top = merge_top_level_scalars(config_path)
    model_cfg = merge_task_model_mapping(config_path)
    context = {**top, **model_cfg}
    assert str(resolve_value(model_cfg["_target_"], context)).endswith(target_suffix)
    assert resolve_value(model_cfg["recurrent_layers"], context) == 1
    assert resolve_value(model_cfg["recurrent_band_kernel"], context) == 3
    assert resolve_value(model_cfg["recurrent_replace_blocks"], context) == 2

    system = build_model_system_from_recipe_config(config_path).eval()
    core = system.model.core
    assert core.recurrent_type == expected_type
    assert len(core.temporal_blocks) == 4
    assert len(core.recurrent_blocks) == 1
    assert core.state_size_bytes(dtype=torch.float16) < 192 * 1024


def test_tvconv_pyramid_recurrent_replace_blocks_requires_recurrent_type() -> None:
    with pytest.raises(ValueError, match="recurrent_replace_blocks"):
        TVConvPyramidNPUSeparator2D(n_freq=64, recurrent_replace_blocks=1)


@pytest.mark.parametrize(
    ("recurrent_type", "state_limit"),
    [
        ("gru", 16 * 1024),
        ("lstm", 20 * 1024),
    ],
)
def test_tvconv_pyramid_recurrent_bottleneck_streaming_matches_forward(
    recurrent_type: str,
    state_limit: int,
) -> None:
    torch.manual_seed(0)
    core = TVConvPyramidNPUSeparator2D(
        n_freq=64,
        n_src=2,
        n_chan=1,
        base_channels=16,
        bottleneck_channels=32,
        n_down=3,
        n_blocks=4,
        expansion=2,
        time_kernel=3,
        time_dilation_cycle=(1, 2, 1, 2),
        band_kernel=3,
        recurrent_type=recurrent_type,
        recurrent_layers=1,
        recurrent_band_kernel=3,
        recurrent_replace_blocks=2,
        mask_hidden=8,
    ).eval()
    x = torch.randn(1, 2, 6, 64)

    with torch.no_grad():
        y = core(x)
        state = core.init_stream_state(batch_size=1, dtype=x.dtype)
        frames = []
        for frame_idx in range(x.shape[2]):
            frame, state = core.forward_stream(x[:, :, frame_idx : frame_idx + 1, :], state)
            frames.append(frame)
        y_stream = torch.cat(frames, dim=2)

    assert tuple(y.shape) == (1, 4, 6, 64)
    torch.testing.assert_close(y_stream, y, rtol=1e-5, atol=1e-5)
    assert core.recurrent_type == recurrent_type
    assert len(core.temporal_blocks) == 2
    assert len(core.recurrent_blocks) == 1
    assert core.state_size_bytes(dtype=torch.float16) < state_limit
    with pytest.raises(RuntimeError, match="not exact for recurrent"):
        core.forward_stream_recompute(x[:, :, :1])


@pytest.mark.parametrize(
    ("builder", "expected_type"),
    [
        (build_tvconv_pyramid_convgru_npu_separator_system, "gru"),
        (build_tvconv_pyramid_convlstm_npu_separator_system, "lstm"),
    ],
)
def test_tvconv_pyramid_recurrent_waveform_builders(builder, expected_type: str) -> None:
    torch.manual_seed(0)
    model = builder(
        n_fft=64,
        hop_length=16,
        fs=8000,
        n_src=3,
        n_chan=1,
        core_n_src=2,
        base_channels=8,
        bottleneck_channels=16,
        n_down=2,
        n_blocks=3,
        expansion=2,
        time_kernel=3,
        time_dilation_cycle=(1, 2, 1),
        band_kernel=3,
        recurrent_layers=1,
        recurrent_band_kernel=3,
        recurrent_replace_blocks=1,
        mask_hidden=8,
        residual_source_enabled=True,
        residual_source_index=2,
        freq_preprocess_enabled=False,
        css_segment_size=1,
        css_shift_size=1,
    ).eval()
    wav = torch.randn(1, 1, 512)

    with torch.no_grad():
        est, aux = model(wav, return_aux=True)

    assert tuple(est.shape) == (1, 3, 1, 512)
    assert model.model.core.recurrent_type == expected_type
    assert model.model.core.state_size_bytes(dtype=torch.float16) < 192 * 1024
    assert aux["mask_domain"] == "packed_complex_mask"
    assert aux["mask_logits_domain"] == "tvconv_pyramid_complex_mask_logits"


def test_tvconv_pyramid_sfclite_query_recipe_and_builder() -> None:
    config_path = Path(
        "recipes/dnr/models/"
        "tvconv-pyramid-sfclite-query-npu.speech-music-residual-sfx.robust-distill.rt192k.fp512keep475/"
        "config.yaml"
    )
    top = merge_top_level_scalars(config_path)
    model_cfg = merge_task_model_mapping(config_path)
    context = {**top, **model_cfg}
    assert str(resolve_value(model_cfg["_target_"], context)).endswith(
        "build_tvconv_pyramid_sfclite_query_npu_separator_system"
    )
    assert resolve_value(model_cfg["freq_preprocess_mode"], context) == "learnable_query"

    system = build_model_system_from_recipe_config(config_path).eval()
    manifest = system.model.frequency_preprocess_manifest()
    assert manifest is not None
    assert manifest["type"] == "sfc_lite_learnable_query"
    assert manifest["tied_synthesis_query"] is True
    assert system.model.freq_preprocessor.frequency_query.requires_grad

    torch.manual_seed(0)
    tiny = build_tvconv_pyramid_sfclite_query_npu_separator_system(
        n_fft=64,
        hop_length=16,
        fs=8000,
        n_src=3,
        n_chan=1,
        core_n_src=2,
        base_channels=8,
        bottleneck_channels=16,
        n_down=2,
        n_blocks=2,
        expansion=2,
        time_kernel=3,
        time_dilation_cycle=(1, 2),
        band_kernel=3,
        mask_hidden=8,
        residual_source_enabled=True,
        residual_source_index=2,
        freq_preprocess_keep_bins=16,
        freq_preprocess_target_bins=20,
        css_segment_size=1,
        css_shift_size=1,
    ).eval()
    wav = torch.randn(1, 1, 512)

    with torch.no_grad():
        est, aux = tiny(wav, return_aux=True)

    assert tuple(est.shape) == (1, 3, 1, 512)
    assert tiny.model.frequency_preprocess_manifest()["type"] == "sfc_lite_learnable_query"
    assert aux["mask_domain"] == "packed_complex_mask"
    assert aux["mask_logits_domain"] == "tvconv_pyramid_complex_mask_logits"


def test_tvconv_pyramid_sourceaware_sfclite_convgru_recipe_and_builder() -> None:
    config_path = Path(
        "recipes/dnr/models/"
        "tvconv-pyramid-sourceaware-sfclite-convgru-npu.speech-music-residual-sfx.robust-distill.rt192k.fp512keep475/"
        "config.yaml"
    )
    top = merge_top_level_scalars(config_path)
    model_cfg = merge_task_model_mapping(config_path)
    context = {**top, **model_cfg}
    assert str(resolve_value(model_cfg["_target_"], context)).endswith(
        "build_tvconv_pyramid_sourceaware_sfclite_convgru_npu_separator_system"
    )
    assert resolve_value(model_cfg["freq_preprocess_mode"], context) == "learnable_query"
    assert resolve_value(model_cfg["source_head_type"], context) == "folded_source_aware"
    assert resolve_value(model_cfg["source_mixer_layers"], context) == 1
    assert resolve_value(model_cfg["mask_hidden"], context) == 384

    system = build_model_system_from_recipe_config(config_path).eval()
    core = system.model.core
    total_params = sum(p.numel() for p in system.model.parameters())
    assert core.recurrent_type == "gru"
    assert core.source_head_type == "folded_source_aware"
    assert len(core.recurrent_blocks) == 1
    assert system.model.frequency_preprocess_manifest()["type"] == "sfc_lite_learnable_query"
    assert 3_000_000 <= total_params <= 5_000_000
    assert core.state_size_bytes(dtype=torch.float16) < 192 * 1024

    torch.manual_seed(0)
    tiny = build_tvconv_pyramid_sourceaware_sfclite_convgru_npu_separator_system(
        n_fft=64,
        hop_length=16,
        fs=8000,
        n_src=3,
        n_chan=1,
        core_n_src=2,
        base_channels=8,
        bottleneck_channels=16,
        n_down=2,
        n_blocks=3,
        expansion=2,
        time_kernel=3,
        time_dilation_cycle=(1, 2, 1),
        band_kernel=3,
        recurrent_layers=1,
        recurrent_band_kernel=3,
        recurrent_replace_blocks=1,
        source_mixer_layers=1,
        mask_hidden=16,
        residual_source_enabled=True,
        residual_source_index=2,
        freq_preprocess_keep_bins=16,
        freq_preprocess_target_bins=20,
        css_segment_size=1,
        css_shift_size=1,
    ).eval()
    wav = torch.randn(1, 1, 512)

    with torch.no_grad():
        est, aux = tiny(wav, return_aux=True)

    assert tuple(est.shape) == (1, 3, 1, 512)
    assert aux["mask_domain"] == "packed_complex_mask"
    assert aux["mask_logits_domain"] == "tvconv_pyramid_folded_source_aware_complex_mask_logits"


def test_source_aware_melband_roformer_teacher_forward_and_recipe() -> None:
    torch.manual_seed(0)
    model = build_source_aware_melband_roformer_teacher_system(
        n_fft=64,
        hop_length=16,
        fs=8000,
        n_src=2,
        n_chan=1,
        n_bands=8,
        d_model=16,
        n_heads=4,
        source_attention_heads=1,
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
    wav = torch.randn(1, 1, 256)
    with torch.no_grad():
        est = model(wav)
        aux_est, aux = model(wav, return_aux=True)
    assert tuple(est.shape) == (1, 2, 1, 256)
    torch.testing.assert_close(aux_est, est, rtol=1e-5, atol=1e-5)
    assert set(aux) == {"mask", "mask_domain", "mask_logits", "mask_logits_domain"}
    assert aux["mask_domain"] == "packed_complex_mask"
    assert aux["mask_logits_domain"] == "source_aware_melband_roformer_complex_mask_logits"
    assert aux["mask"].ndim == 4
    assert aux["mask"].shape[1] == 4
    torch.testing.assert_close(aux["mask_logits"], aux["mask"])

    core = SourceAwareMelBandRoformer2D(
        n_freq=33,
        sample_rate=8000,
        n_src=2,
        n_chan=1,
        n_bands=8,
        d_model=16,
        n_heads=4,
        source_attention_heads=1,
        n_encoder_layers=1,
        n_decoder_layers=1,
        ffn_mult=2,
        dropout=0.0,
        conv_kernel_size=(3, 3),
        routing_kernel_size=(3, 3),
    ).eval()
    x = torch.randn(1, 2, 4, 33)
    with torch.no_grad():
        y = core(x)
    assert tuple(y.shape) == (1, 4, 4, 33)
    y_sources = y.reshape(1, 2, 2, 4, 33)
    torch.testing.assert_close(y_sources.sum(dim=1), x, rtol=1e-5, atol=1e-5)

    raw_core = SourceAwareMelBandRoformer2D(
        n_freq=33,
        sample_rate=8000,
        n_src=2,
        n_chan=1,
        n_bands=8,
        d_model=16,
        n_heads=4,
        source_attention_heads=1,
        n_encoder_layers=1,
        n_decoder_layers=1,
        ffn_mult=2,
        dropout=0.0,
        conv_kernel_size=(3, 3),
        routing_kernel_size=(3, 3),
        masking=False,
    ).eval()
    with torch.no_grad():
        masks = raw_core(x)
    assert tuple(masks.shape) == (1, 4, 4, 33)

    config_path = Path("recipes/dnr/models/source-aware-melband-roformer.teacher/config.yaml")
    student_config_path = Path(
        "recipes/dnr/models/tvconv-pyramid-npu.speech-music-residual-sfx.robust-distill.rt192k.fp512keep475/config.yaml"
    )
    raw_config = OmegaConf.load(config_path)
    student_config = OmegaConf.load(student_config_path)
    assert "_base_" not in raw_config
    assert str(raw_config.datamodule._target_).endswith("OnTheFlyStemDataModule")
    assert raw_config.datamodule.batch_size == 1
    assert raw_config.datamodule.val_batch_size == student_config.datamodule.val_batch_size
    assert raw_config.datamodule.test_batch_size == student_config.datamodule.test_batch_size
    assert OmegaConf.to_container(raw_config.datamodule.synthesis, resolve=True) == OmegaConf.to_container(
        student_config.datamodule.synthesis,
        resolve=True,
    )
    assert raw_config.trainer._target_ == "lightning.Trainer"
    assert raw_config.task.loss._target_.endswith("ThresSNRLossWithInactiveSource")

    top = merge_top_level_scalars(config_path)
    model_cfg = merge_task_model_mapping(config_path)
    context = {**top, **model_cfg}
    assert resolve_value(model_cfg["_target_"], context).endswith("build_source_aware_melband_roformer_teacher_system")
    assert resolve_value(model_cfg["d_model"], context) == 192
    assert resolve_value(model_cfg["n_bands"], context) == 128
    assert resolve_value(model_cfg["online_wrapper"], context) is False


def test_source_aware_melband_roformer_student_npu_forward_streaming_and_recipe() -> None:
    torch.manual_seed(0)
    model = build_source_aware_melband_roformer_student_npu_system(
        n_fft=64,
        hop_length=16,
        fs=8000,
        n_src=2,
        n_chan=1,
        n_bands=8,
        d_model=8,
        n_encoder_layers=1,
        n_decoder_layers=1,
        long_branch_layers=1,
        correction_layers=1,
        correction_channels=4,
        decoder_fusion_hidden=8,
        kernel_size=(3, 3),
        routing_kernel_size=(1, 3),
        freq_preprocess_enabled=False,
        css_segment_size=1,
        css_shift_size=1,
    ).eval()
    wav = torch.randn(1, 1, 256)
    with torch.no_grad():
        est = model(wav)
    assert tuple(est.shape) == (1, 2, 1, 256)

    core = OnlineSourceAwareMelBandStudentSFC2D(
        n_freq=33,
        sample_rate=8000,
        n_src=2,
        n_chan=1,
        n_bands=8,
        d_model=8,
        n_encoder_layers=1,
        n_decoder_layers=1,
        long_branch_layers=1,
        correction_layers=1,
        correction_channels=4,
        decoder_fusion_hidden=8,
        kernel_size=(3, 3),
        routing_kernel_size=(1, 3),
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
    y_sources = y.reshape(1, 2, 2, 4, 33)
    torch.testing.assert_close(y_sources.sum(dim=1), x, rtol=1e-5, atol=1e-5)
    assert tuple(core.source_splitter(torch.randn(1, 8, 2, 8)).shape) == (1, 16, 2, 8)
    assert len(core.init_stream_state(batch_size=1, dtype=x.dtype)) == 3
    assert core.state_size_bytes(dtype=torch.float16) < 32 * 1024

    raw_core = OnlineSourceAwareMelBandStudentSFC2D(
        n_freq=33,
        sample_rate=8000,
        n_src=2,
        n_chan=1,
        n_bands=8,
        d_model=8,
        n_encoder_layers=1,
        n_decoder_layers=1,
        long_branch_layers=1,
        correction_layers=1,
        correction_channels=4,
        decoder_fusion_hidden=8,
        kernel_size=(3, 3),
        routing_kernel_size=(1, 3),
        masking=False,
    ).eval()
    with torch.no_grad():
        masks = raw_core(x)
    assert tuple(masks.shape) == (1, 4, 4, 33)

    deploy_core = OnlineSourceAwareMelBandStudentSFC2D(
        n_freq=512,
        sample_rate=44100,
        n_src=3,
        n_chan=1,
        n_bands=80,
        d_model=40,
        n_encoder_layers=2,
        n_decoder_layers=3,
        long_branch_layers=1,
        correction_layers=1,
        correction_channels=16,
        decoder_fusion_hidden=96,
        decoder_kernel_size=(1, 3),
        encoder_dilation_cycle=(1, 2),
    ).eval()
    deploy_params = sum(p.numel() for p in deploy_core.parameters())
    assert 100_000 <= deploy_params <= 2_000_000
    assert deploy_core.state_size_bytes(dtype=torch.float16) < 192 * 1024

    config_path = Path("recipes/dnr/models/source-aware-melband-roformer.student-npu.rt192k.fp512keep475/config.yaml")
    top = merge_top_level_scalars(config_path)
    model_cfg = merge_task_model_mapping(config_path)
    context = {**top, **model_cfg}
    assert str(resolve_value(model_cfg["_target_"], context)).endswith(
        "build_source_aware_melband_roformer_student_npu_system"
    )
    assert resolve_value(model_cfg["d_model"], context) == 40
    assert resolve_value(model_cfg["n_bands"], context) == 80
    assert resolve_value(model_cfg["decoder_fusion_hidden"], context) == 96
    assert resolve_value(model_cfg["decoder_kernel_size"], context) == [1, 3]

    system = build_model_system_from_recipe_config(config_path).eval()
    recipe_core = system.model.core
    assert recipe_core.n_freq == 512
    assert recipe_core.n_bands == 80
    assert recipe_core.d_model == 40
    assert recipe_core.state_size_bytes(dtype=torch.float16) < 192 * 1024

    distill_path = Path(
        "recipes/dnr/models/source-aware-melband-roformer.student-npu.distill.rt192k.fp512keep475/config.yaml"
    )
    distill_top = merge_top_level_scalars(distill_path)
    distill_model_cfg = merge_task_model_mapping(distill_path)
    distill_context = {**distill_top, **distill_model_cfg}
    assert distill_top["teacher_checkpoint_path"] is None
    assert str(resolve_value(distill_model_cfg["_target_"], distill_context)).endswith(
        "build_source_aware_melband_roformer_student_npu_system"
    )
    distill_text = distill_path.read_text(encoding="utf-8")
    assert "TeacherStudentDistillationTask" in distill_text
    assert "require_teacher_checkpoint: true" in distill_text
    assert "teacher_css_validation: true" in distill_text
    assert "distillation_band_mapping: mel_centers" in distill_text
    assert "build_source_aware_melband_roformer_teacher_system" in distill_text


def test_source_aware_melband_roformer_student_npu_onnx_audit_smoke() -> None:
    onnx = pytest.importorskip("onnx")

    from spectral_feature_compression.utils.onnx_streaming import (
        StreamingStateIOWrapper,
        flatten_tensor_tree,
        get_external_constant_tensors,
    )
    from tools.online.audit_onnx_model import audit_npu_risks, get_allowed_ops

    torch.manual_seed(0)
    core = OnlineSourceAwareMelBandStudentSFC2D(
        n_freq=33,
        sample_rate=8000,
        n_src=2,
        n_chan=1,
        n_bands=8,
        d_model=8,
        n_encoder_layers=1,
        n_decoder_layers=1,
        long_branch_layers=1,
        correction_layers=1,
        correction_channels=4,
        decoder_fusion_hidden=8,
    ).eval()
    wrapper = StreamingStateIOWrapper(core, batch_size=1, dtype=torch.float32, externalize_constants=True).eval()
    state = core.init_stream_state(batch_size=1, dtype=torch.float32)
    flat_state, _ = flatten_tensor_tree(state)
    constants = get_external_constant_tensors(core, wrapper.constant_bindings)
    x = torch.randn(1, 2, 1, 33)

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "source_aware_melband_student_stream.onnx"
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                (x, *flat_state, *constants),
                str(out),
                opset_version=14,
                input_names=[
                    "x",
                    *[f"state_{idx}" for idx in range(len(flat_state))],
                    *[f"const_{idx}" for idx in range(len(constants))],
                ],
                output_names=["y", *[f"next_state_{idx}" for idx in range(len(flat_state))]],
                do_constant_folding=True,
                dynamo=False,
            )
        model = onnx.load(str(out))

    onnx.checker.check_model(model)
    allowed_ops = get_allowed_ops("edge_npu_recommended")
    assert sorted({node.op_type for node in model.graph.node} - allowed_ops) == []
    audit = audit_npu_risks(model)
    assert audit["has_strict_edge_risks"] is False
    assert audit["risk_counts"]["rank_gt4_values"] == 0


def test_source_aware_melband_roformer_strong_student_npu_forward_streaming_and_recipe() -> None:
    torch.manual_seed(0)
    model = build_source_aware_melband_roformer_strong_student_npu_system(
        n_fft=64,
        hop_length=16,
        fs=8000,
        n_src=2,
        n_chan=1,
        n_bands=8,
        d_model=8,
        n_encoder_layers=1,
        n_source_layers=1,
        correction_layers=1,
        source_fusion_hidden=16,
        source_seed_hidden=16,
        expander_hidden=16,
        mask_hidden=16,
        correction_channels=4,
        kernel_size=(3, 3),
        routing_kernel_size=(1, 3),
        freq_preprocess_enabled=False,
        css_segment_size=1,
        css_shift_size=1,
    ).eval()
    wav = torch.randn(1, 1, 256)
    with torch.no_grad():
        est = model(wav)
    assert tuple(est.shape) == (1, 2, 1, 256)

    core = OnlineSourceAwareMelBandStrongStudentSFC2D(
        n_freq=33,
        sample_rate=8000,
        n_src=2,
        n_chan=1,
        n_bands=8,
        d_model=8,
        n_encoder_layers=1,
        n_source_layers=1,
        correction_layers=1,
        source_fusion_hidden=16,
        source_seed_hidden=16,
        expander_hidden=16,
        mask_hidden=16,
        correction_channels=4,
        kernel_size=(3, 3),
        routing_kernel_size=(1, 3),
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
    y_sources = y.reshape(1, 2, 2, 4, 33)
    torch.testing.assert_close(y_sources.sum(dim=1), x, rtol=1e-5, atol=1e-5)
    assert len(core.init_stream_state(batch_size=1, dtype=x.dtype)) == 2
    assert core.state_size_bytes(dtype=torch.float16) < 32 * 1024

    raw_core = OnlineSourceAwareMelBandStrongStudentSFC2D(
        n_freq=33,
        sample_rate=8000,
        n_src=2,
        n_chan=1,
        n_bands=8,
        d_model=8,
        n_encoder_layers=1,
        n_source_layers=1,
        correction_layers=1,
        source_fusion_hidden=16,
        source_seed_hidden=16,
        expander_hidden=16,
        mask_hidden=16,
        correction_channels=4,
        kernel_size=(3, 3),
        routing_kernel_size=(1, 3),
        masking=False,
    ).eval()
    with torch.no_grad():
        masks = raw_core(x)
    assert tuple(masks.shape) == (1, 4, 4, 33)

    stateless_core = OnlineSourceAwareMelBandStrongStudentSFC2D(
        n_freq=33,
        sample_rate=8000,
        n_src=2,
        n_chan=1,
        n_bands=8,
        d_model=8,
        n_encoder_layers=1,
        n_source_layers=1,
        correction_layers=1,
        source_fusion_hidden=16,
        source_seed_hidden=16,
        expander_hidden=16,
        mask_hidden=16,
        correction_channels=4,
        kernel_size=(1, 3),
        routing_kernel_size=(1, 3),
    ).eval()
    with torch.no_grad():
        y_stateless = stateless_core(x)
        state = stateless_core.init_stream_state(batch_size=1, dtype=x.dtype)
        frames = []
        for frame_idx in range(x.shape[2]):
            frame, state = stateless_core.forward_stream(x[:, :, frame_idx : frame_idx + 1, :], state)
            frames.append(frame)
        y_stateless_stream = torch.cat(frames, dim=2)
    assert len(stateless_core.init_stream_state(batch_size=1, dtype=x.dtype)) == 0
    torch.testing.assert_close(y_stateless_stream, y_stateless, rtol=1e-5, atol=1e-5)

    deploy_core = OnlineSourceAwareMelBandStrongStudentSFC2D(
        n_freq=512,
        sample_rate=44100,
        n_src=3,
        n_chan=1,
        n_bands=80,
        d_model=48,
        n_encoder_layers=2,
        n_source_layers=5,
        correction_layers=1,
        source_fusion_hidden=192,
        source_seed_hidden=192,
        expander_hidden=128,
        mask_hidden=192,
        correction_channels=24,
        kernel_size=(3, 5),
        encoder_dilation_cycle=(1, 2),
    ).eval()
    deploy_params = sum(p.numel() for p in deploy_core.parameters())
    assert 1_000_000 <= deploy_params <= 2_500_000
    assert deploy_core.state_size_bytes(dtype=torch.float16) < 192 * 1024

    config_path = Path(
        "recipes/dnr/models/source-aware-melband-roformer.student-npu-strong.rt192k.fp512keep475/config.yaml"
    )
    top = merge_top_level_scalars(config_path)
    model_cfg = merge_task_model_mapping(config_path)
    context = {**top, **model_cfg}
    assert str(resolve_value(model_cfg["_target_"], context)).endswith(
        "build_source_aware_melband_roformer_strong_student_npu_system"
    )
    assert resolve_value(model_cfg["d_model"], context) == 48
    assert resolve_value(model_cfg["n_bands"], context) == 80
    assert resolve_value(model_cfg["n_source_layers"], context) == 5
    assert resolve_value(model_cfg["source_fusion_hidden"], context) == 192

    system = build_model_system_from_recipe_config(config_path).eval()
    recipe_core = system.model.core
    assert recipe_core.n_freq == 512
    assert recipe_core.n_bands == 80
    assert recipe_core.d_model == 48
    assert recipe_core.n_source_layers == 5
    assert sum(p.numel() for p in recipe_core.parameters()) == deploy_params
    assert recipe_core.state_size_bytes(dtype=torch.float16) < 192 * 1024

    distill_path = Path(
        "recipes/dnr/models/source-aware-melband-roformer.student-npu-strong.distill.rt192k.fp512keep475/config.yaml"
    )
    distill_top = merge_top_level_scalars(distill_path)
    distill_model_cfg = merge_task_model_mapping(distill_path)
    distill_context = {**distill_top, **distill_model_cfg}
    assert distill_top["teacher_checkpoint_path"] is None
    assert str(resolve_value(distill_model_cfg["_target_"], distill_context)).endswith(
        "build_source_aware_melband_roformer_strong_student_npu_system"
    )
    distill_text = distill_path.read_text(encoding="utf-8")
    assert "TeacherStudentDistillationTask" in distill_text
    assert "require_teacher_checkpoint: true" in distill_text
    assert "teacher_css_validation: true" in distill_text
    assert "distillation_band_mapping: mel_centers" in distill_text
    assert "build_source_aware_melband_roformer_teacher_system" in distill_text

    relaxed_path = Path(
        "recipes/dnr/models/"
        "source-aware-melband-roformer.student-npu-strong-relaxed.distill.tv-fp512keep475/"
        "config.yaml"
    )
    relaxed_system = build_model_system_from_recipe_config(relaxed_path).eval()
    relaxed_core = relaxed_system.model.core
    relaxed_params = sum(p.numel() for p in relaxed_core.parameters())
    assert relaxed_core.n_freq == 512
    assert relaxed_core.n_bands == 96
    assert relaxed_core.d_model == 72
    assert relaxed_core.n_encoder_layers == 3
    assert relaxed_core.n_source_layers == 8
    assert relaxed_core.correction_layers == 2
    assert relaxed_core.mixture_consistency is False
    assert 6_000_000 <= relaxed_params <= 7_000_000
    assert relaxed_core.state_size_bytes(dtype=torch.float16) == 600 * 1024
    assert relaxed_core.state_size_bytes(dtype=torch.float16) > 192 * 1024

    from aiaccel.config import load_config, resolve_inherit

    relaxed_config = resolve_inherit(
        load_config(
            relaxed_path,
            {
                "config_path": str(relaxed_path),
                "working_directory": str(relaxed_path.parent.resolve()),
                "base_config_path": str((Path.cwd() / "aiaccel" / "aiaccel" / "torch" / "apps" / "config").resolve()),
            },
        )
    )
    assert str(relaxed_config.datamodule._target_).endswith("OnTheFlyStemDataModule")
    assert len(relaxed_config.datamodule.synthesis.synthesis_profiles) == 4
    assert relaxed_config.task.request_model_aux is False
    assert relaxed_config.task.require_model_aux is False
    assert relaxed_config.task.source_loss_weights.speech == pytest.approx(1.6)
    assert relaxed_config.task.residual_source_loss_weight == pytest.approx(0.0)
    assert relaxed_config.task.distillation_band_mapping == "mel_centers"

    nodelite_path = Path(
        "recipes/dnr/models/"
        "source-aware-melband-roformer.student-npu-strong-nodelite.distill.tv-fp512keep475/"
        "config.yaml"
    )
    nodelite_system = build_model_system_from_recipe_config(nodelite_path).eval()
    nodelite_core = nodelite_system.model.core
    nodelite_params = sum(p.numel() for p in nodelite_core.parameters())
    assert nodelite_core.n_freq == 512
    assert nodelite_core.n_bands == 80
    assert nodelite_core.d_model == 48
    assert nodelite_core.n_encoder_layers == 2
    assert nodelite_core.n_source_layers == 5
    assert nodelite_core.correction_layers == 1
    assert nodelite_core.mixture_consistency is False
    assert 6_000_000 <= nodelite_params <= 6_500_000
    assert nodelite_core.state_size_bytes(dtype=torch.float16) == 186 * 1024

    nodelite_config = resolve_inherit(
        load_config(
            nodelite_path,
            {
                "config_path": str(nodelite_path),
                "working_directory": str(nodelite_path.parent.resolve()),
                "base_config_path": str((Path.cwd() / "aiaccel" / "aiaccel" / "torch" / "apps" / "config").resolve()),
            },
        )
    )
    assert str(nodelite_config.datamodule._target_).endswith("OnTheFlyStemDataModule")
    assert len(nodelite_config.datamodule.synthesis.synthesis_profiles) == 4
    assert nodelite_config.task.source_loss_weights.speech == pytest.approx(1.6)
    assert nodelite_config.task.request_model_aux is False
    assert nodelite_config.task.require_model_aux is False


def test_source_aware_melband_loco_cnb_student_npu_forward_streaming_and_recipe() -> None:
    torch.manual_seed(0)
    model = build_source_aware_melband_loco_cnb_student_npu_system(
        n_fft=64,
        hop_length=16,
        fs=8000,
        n_src=2,
        n_chan=1,
        n_bands=8,
        state_channels=8,
        source_channels=8,
        n_loco_layers=1,
        n_source_layers=1,
        source_fusion_hidden=16,
        source_seed_hidden=16,
        expander_hidden=16,
        mask_hidden=16,
        correction_channels=4,
        correction_kernel_size=(1, 3),
        cnb_kernel=3,
        cnb_dilation_schedule=(1, 2),
        cnb_num_heads=2,
        cnb_head_dim=4,
        pooled_mixer_hidden_schedule=(16,),
        freq_preprocess_enabled=False,
        css_segment_size=1,
        css_shift_size=1,
    ).eval()
    wav = torch.randn(1, 1, 256)
    with torch.no_grad():
        est = model(wav)
        est_aux, wav_aux = model(wav, return_aux=True)
    assert tuple(est.shape) == (1, 2, 1, 256)
    torch.testing.assert_close(est_aux, est, rtol=1e-5, atol=1e-5)
    assert wav_aux["mask_domain"] == "packed_complex_mask"
    assert wav_aux["mask_logits_domain"] == "source_aware_melband_loco_cnb_complex_mask_logits"

    core = OnlineSourceAwareMelBandLocoCNBStudentSFC2D(
        n_freq=33,
        sample_rate=8000,
        n_src=2,
        n_chan=1,
        n_bands=8,
        state_channels=8,
        source_channels=8,
        n_loco_layers=1,
        n_source_layers=1,
        source_fusion_hidden=16,
        source_seed_hidden=16,
        expander_hidden=16,
        mask_hidden=16,
        correction_channels=4,
        correction_kernel_size=(1, 3),
        cnb_kernel=3,
        cnb_dilation_schedule=(1, 2),
        cnb_num_heads=2,
        cnb_head_dim=4,
        pooled_mixer_hidden_schedule=(16,),
    ).eval()
    x = torch.randn(1, 2, 4, 33)
    with torch.no_grad():
        y = core(x)
        y_aux, aux = core(x, return_aux=True)
        state = core.init_stream_state(batch_size=1, dtype=x.dtype)
        frames = []
        for frame_idx in range(x.shape[2]):
            frame, state = core.forward_stream(x[:, :, frame_idx : frame_idx + 1, :], state)
            frames.append(frame)
        y_stream = torch.cat(frames, dim=2)
    assert tuple(y.shape) == (1, 4, 4, 33)
    torch.testing.assert_close(y_aux, y, rtol=1e-5, atol=1e-5)
    assert aux["mask_domain"] == "packed_complex_mask"
    assert aux["mask_logits_domain"] == "source_aware_melband_loco_cnb_complex_mask_logits"
    assert tuple(aux["mask"].shape) == (1, 4, 4, 33)
    torch.testing.assert_close(aux["mask_logits"], aux["mask"])
    assert tuple(y_stream.shape) == tuple(y.shape)
    torch.testing.assert_close(y_stream, y, rtol=1e-5, atol=1e-5)
    y_sources = y.reshape(1, 2, 2, 4, 33)
    torch.testing.assert_close(y_sources.sum(dim=1), x, rtol=1e-5, atol=1e-5)
    assert len(core.init_stream_state(batch_size=1, dtype=x.dtype)) == 1
    assert core.state_size_bytes(dtype=torch.float16) < 32 * 1024

    raw_core = OnlineSourceAwareMelBandLocoCNBStudentSFC2D(
        n_freq=33,
        sample_rate=8000,
        n_src=2,
        n_chan=1,
        n_bands=8,
        state_channels=8,
        source_channels=8,
        n_loco_layers=1,
        n_source_layers=1,
        source_fusion_hidden=16,
        source_seed_hidden=16,
        expander_hidden=16,
        mask_hidden=16,
        correction_channels=4,
        correction_kernel_size=(1, 3),
        cnb_kernel=3,
        cnb_dilation_schedule=(1, 2),
        cnb_num_heads=2,
        cnb_head_dim=4,
        pooled_mixer_hidden_schedule=(16,),
        masking=False,
    ).eval()
    with torch.no_grad():
        masks = raw_core(x)
    assert tuple(masks.shape) == (1, 4, 4, 33)

    merged_core = OnlineSourceAwareMelBandLocoCNBStudentSFC2D(
        n_freq=33,
        sample_rate=8000,
        n_src=2,
        n_chan=1,
        n_bands=8,
        state_channels=8,
        source_channels=8,
        n_loco_layers=1,
        n_source_layers=1,
        source_fusion_hidden=16,
        source_seed_hidden=16,
        expander_hidden=16,
        mask_hidden=16,
        correction_channels=4,
        correction_kernel_size=(1, 3),
        cnb_kernel=3,
        cnb_dilation_schedule=(1, 2),
        cnb_merge_dilations=True,
        cnb_num_heads=2,
        cnb_head_dim=4,
        pooled_mixer_hidden_schedule=(16,),
    ).eval()
    with torch.no_grad():
        y_merged = merged_core(x)
        merged_state = merged_core.init_stream_state(batch_size=1, dtype=x.dtype)
        merged_frames = []
        for frame_idx in range(x.shape[2]):
            frame, merged_state = merged_core.forward_stream(x[:, :, frame_idx : frame_idx + 1, :], merged_state)
            merged_frames.append(frame)
        y_merged_stream = torch.cat(merged_frames, dim=2)
    assert merged_core.cnb_merge_dilations is True
    assert merged_core.backbone[0].narrow_band.merge_dilations is True
    assert tuple(y_merged_stream.shape) == tuple(y_merged.shape)
    torch.testing.assert_close(y_merged_stream, y_merged, rtol=1e-5, atol=1e-5)

    deploy_core = OnlineSourceAwareMelBandLocoCNBStudentSFC2D(
        n_freq=512,
        sample_rate=44100,
        n_src=3,
        n_chan=1,
        n_bands=56,
        state_channels=36,
        source_channels=48,
        n_loco_layers=4,
        n_source_layers=4,
        source_fusion_hidden=192,
        source_seed_hidden=192,
        expander_hidden=128,
        mask_hidden=160,
        correction_channels=16,
        correction_kernel_size=(1, 5),
        routing_kernel_size=(1, 3),
        cnb_num_heads=4,
        cnb_head_dim=8,
        pooled_mixer_hidden_schedule=(2048, 4096, 4096, 2048),
    ).eval()
    deploy_params = sum(p.numel() for p in deploy_core.parameters())
    assert 2_000_000 <= deploy_params <= 4_000_000
    assert deploy_core.state_size_bytes(dtype=torch.float16) == 177_408
    assert deploy_core.state_size_bytes(dtype=torch.float16) < 192 * 1024

    config_path = Path("recipes/dnr/models/source-aware-melband-loco-cnb.student-npu.rt192k.fp512keep475/config.yaml")
    top = merge_top_level_scalars(config_path)
    model_cfg = merge_task_model_mapping(config_path)
    context = {**top, **model_cfg}
    assert str(resolve_value(model_cfg["_target_"], context)).endswith(
        "build_source_aware_melband_loco_cnb_student_npu_system"
    )
    assert resolve_value(model_cfg["n_bands"], context) == 56
    assert resolve_value(model_cfg["state_channels"], context) == 36
    assert resolve_value(model_cfg["source_channels"], context) == 48
    assert resolve_value(model_cfg["cnb_merge_dilations"], context) is False
    assert resolve_value(model_cfg["pooled_mixer_hidden_schedule"], context) == [2048, 4096, 4096, 2048]

    system = build_model_system_from_recipe_config(config_path).eval()
    recipe_core = system.model.core
    assert recipe_core.n_freq == 512
    assert recipe_core.n_bands == 56
    assert recipe_core.state_channels == 36
    assert recipe_core.source_channels == 48
    assert sum(p.numel() for p in recipe_core.parameters()) == deploy_params
    assert recipe_core.state_size_bytes(dtype=torch.float16) < 192 * 1024

    residual_path = Path(
        "recipes/dnr/models/source-aware-melband-loco-cnb.student-npu-residual-sfx.rt192k.fp512keep475/config.yaml"
    )
    residual_top = merge_top_level_scalars(residual_path)
    residual_model_cfg = merge_task_model_mapping(residual_path)
    residual_context = {**residual_top, **residual_model_cfg}
    assert resolve_value(residual_model_cfg["mixture_consistency"], residual_context) is False
    residual_system = build_model_system_from_recipe_config(residual_path).eval()
    assert residual_system.model.residual_source_enabled is True
    assert residual_system.model.core.n_src == 2
    assert residual_system.model.n_src == 3
    assert residual_system.model.core.mixture_consistency is False
    assert residual_system.model.core.state_size_bytes(dtype=torch.float16) == 177_408

    with pytest.raises(ValueError, match="residual_source_enabled requires mixture_consistency=False"):
        build_source_aware_melband_loco_cnb_student_npu_system(
            n_fft=64,
            hop_length=16,
            fs=8000,
            n_src=3,
            n_chan=1,
            core_n_src=2,
            n_bands=8,
            state_channels=8,
            source_channels=8,
            n_loco_layers=1,
            n_source_layers=1,
            residual_source_enabled=True,
            residual_source_index=2,
            mixture_consistency=True,
            freq_preprocess_enabled=False,
        )

    lowlat_path = Path(
        "recipes/dnr/models/source-aware-melband-loco-cnb.student-npu-residual-sfx-lowlat.rt192k.fp512keep475/config.yaml"
    )
    lowlat_top = merge_top_level_scalars(lowlat_path)
    lowlat_model_cfg = merge_task_model_mapping(lowlat_path)
    lowlat_context = {**lowlat_top, **lowlat_model_cfg}
    assert resolve_value(lowlat_model_cfg["cnb_merge_dilations"], lowlat_context) is True
    assert resolve_value(lowlat_model_cfg["mixture_consistency"], lowlat_context) is False
    lowlat_system = build_model_system_from_recipe_config(lowlat_path).eval()
    assert lowlat_system.model.core.cnb_merge_dilations is True
    assert lowlat_system.model.core.mixture_consistency is False
    assert lowlat_system.model.core.state_size_bytes(dtype=torch.float16) == 177_408
    assert sum(p.numel() for p in lowlat_system.model.core.parameters()) < sum(
        p.numel() for p in residual_system.model.core.parameters()
    )

    distill_path = Path(
        "recipes/dnr/models/source-aware-melband-loco-cnb.student-npu-residual-sfx.distill.rt192k.fp512keep475/config.yaml"
    )
    distill_top = merge_top_level_scalars(distill_path)
    distill_model_cfg = merge_task_model_mapping(distill_path)
    distill_context = {**distill_top, **distill_model_cfg}
    assert distill_top["teacher_checkpoint_path"] is None
    assert str(resolve_value(distill_model_cfg["_target_"], distill_context)).endswith(
        "build_source_aware_melband_loco_cnb_student_npu_system"
    )
    distill_text = distill_path.read_text(encoding="utf-8")
    assert "TeacherStudentDistillationTask" in distill_text
    assert "require_teacher_checkpoint: true" in distill_text
    assert "teacher_css_validation: true" in distill_text
    assert "distillation_band_mapping: mel_centers" in distill_text
    assert "build_source_aware_melband_roformer_teacher_system" in distill_text

    lowlat_distill_path = Path(
        "recipes/dnr/models/source-aware-melband-loco-cnb.student-npu-residual-sfx-lowlat.distill.rt192k.fp512keep475/config.yaml"
    )
    lowlat_distill_top = merge_top_level_scalars(lowlat_distill_path)
    lowlat_distill_model_cfg = merge_task_model_mapping(lowlat_distill_path)
    lowlat_distill_context = {**lowlat_distill_top, **lowlat_distill_model_cfg}
    assert resolve_value(lowlat_distill_model_cfg["cnb_merge_dilations"], lowlat_distill_context) is True
    assert resolve_value(lowlat_distill_model_cfg["mixture_consistency"], lowlat_distill_context) is False

    tv_path = Path(
        "recipes/dnr/models/source-aware-melband-loco-cnb.tv-stems.robust-lowlat.distill.rt192k.fp512keep475/config.yaml"
    )
    tv_text = tv_path.read_text(encoding="utf-8")
    assert "source_loss_weights" in tv_text
    assert "robust_label_loss_weight" in tv_text
    tv_system = build_model_system_from_recipe_config(tv_path).eval()
    assert tv_system.model.core.cnb_merge_dilations is True
    assert tv_system.model.residual_source_enabled is True

    fixed_tv_path = Path(
        "recipes/dnr/models/source-aware-melband-loco-cnb.tvfix-nopool.robust-lowlat.distill.rt192k.fp512keep475/config.yaml"
    )
    fixed_tv_cfg = OmegaConf.load(fixed_tv_path)
    assert str(fixed_tv_cfg.datamodule._target_).endswith("OnTheFlyStemDataModule")
    assert len(fixed_tv_cfg.datamodule.synthesis.synthesis_profiles) == 4
    assert fixed_tv_cfg.task.request_model_aux is True
    assert fixed_tv_cfg.task.require_model_aux is True
    assert fixed_tv_cfg.task.mask_aux_alignment == "shared_prefix"
    assert fixed_tv_cfg.task.distillation_band_mapping == "linear"
    assert fixed_tv_cfg.task.teacher_logit_loss_weight == 0.0
    fixed_tv_top = merge_top_level_scalars(fixed_tv_path)
    fixed_tv_model_cfg = merge_task_model_mapping(fixed_tv_path)
    fixed_tv_context = {**fixed_tv_top, **fixed_tv_model_cfg}
    assert resolve_value(fixed_tv_model_cfg["pooled_mixer_hidden_schedule"], fixed_tv_context) == [0, 0, 0, 0]
    fixed_tv_system = build_model_system_from_recipe_config(fixed_tv_path).eval()
    assert all(block.pooled_mixer.__class__.__name__ == "Identity" for block in fixed_tv_system.model.core.backbone)

    strong_tv_path = Path(
        "recipes/dnr/models/source-aware-melband-loco-cnb.tvfix-strong-nopool.robust-lowlat.distill.rt192k.fp512keep475/config.yaml"
    )
    strong_tv_system = build_model_system_from_recipe_config(strong_tv_path).eval()
    strong_core = strong_tv_system.model.core
    assert strong_core.source_channels == 80
    assert strong_core.n_source_layers == 5
    assert all(block.pooled_mixer.__class__.__name__ == "Identity" for block in strong_core.backbone)
    assert 3_000_000 <= sum(p.numel() for p in strong_core.parameters()) <= 4_000_000
    assert strong_core.state_size_bytes(dtype=torch.float16) == fixed_tv_system.model.core.state_size_bytes(
        dtype=torch.float16
    )

    capacity_sup_path = Path(
        "recipes/dnr/models/source-aware-melband-loco-cnb.tvfix-capacity-nopool.sup.rt192k.fp512keep475/config.yaml"
    )
    from aiaccel.config import load_config, resolve_inherit

    capacity_sup_cfg = resolve_inherit(
        load_config(
            capacity_sup_path,
            {
                "config_path": str(capacity_sup_path),
                "working_directory": str(capacity_sup_path.parent.resolve()),
                "base_config_path": str((Path.cwd() / "aiaccel" / "aiaccel" / "torch" / "apps" / "config").resolve()),
            },
        )
    )
    assert capacity_sup_cfg.task.teacher_model is None
    assert capacity_sup_cfg.task.teacher_loss_weight == pytest.approx(0.0)
    assert capacity_sup_cfg.task.teacher_mask_loss_weight == pytest.approx(0.0)
    assert capacity_sup_cfg.task.teacher_logit_loss_weight == pytest.approx(0.0)
    assert capacity_sup_cfg.task.request_model_aux is False
    capacity_sup_system = build_model_system_from_recipe_config(capacity_sup_path).eval()
    capacity_core = capacity_sup_system.model.core
    assert capacity_core.source_channels == 112
    assert capacity_core.n_source_layers == 5
    assert all(block.pooled_mixer.__class__.__name__ == "Identity" for block in capacity_core.backbone)
    assert 6_000_000 <= sum(p.numel() for p in capacity_core.parameters()) <= 7_000_000
    assert capacity_core.state_size_bytes(dtype=torch.float16) == fixed_tv_system.model.core.state_size_bytes(
        dtype=torch.float16
    )

    pcen_normlite_path = Path(
        "recipes/dnr/models/source-aware-melband-loco-cnb.tvfix-capacity-pcen-normlite.sup.rt192k.fp512keep475/config.yaml"
    )
    pcen_normlite_top = merge_top_level_scalars(pcen_normlite_path)
    pcen_normlite_model_cfg = merge_task_model_mapping(pcen_normlite_path)
    pcen_normlite_context = {**pcen_normlite_top, **pcen_normlite_model_cfg}
    assert resolve_value(pcen_normlite_model_cfg["norm_type"], pcen_normlite_context) == "affine"
    assert resolve_value(pcen_normlite_model_cfg["include_magnitude_features"], pcen_normlite_context) is False
    assert resolve_value(pcen_normlite_model_cfg["pcen_preprocess_enabled"], pcen_normlite_context) is True
    pcen_normlite_meta = load_pcen_preprocess_metadata(pcen_normlite_path)
    assert pcen_normlite_meta is not None
    assert pcen_normlite_meta["type"] == "pcen_gain_normalizer_2d"
    assert pcen_normlite_meta["placement"] == "after_frequency_preprocessing_before_core"
    assert pcen_normlite_meta["state_shape"] == [1, 1, 1, 512]
    pcen_normlite_system = build_model_system_from_recipe_config(pcen_normlite_path).eval()
    pcen_normlite_core = pcen_normlite_system.model.core
    assert pcen_normlite_system.model.pcen_preprocessor is not None
    assert pcen_normlite_core.norm_type == "affine"
    assert pcen_normlite_core.include_magnitude_features is False
    assert not any(isinstance(module, RMSNorm2d) for module in pcen_normlite_core.modules())
    assert any(isinstance(module, ChannelAffine2d) for module in pcen_normlite_core.modules())
    assert 6_000_000 <= sum(p.numel() for p in pcen_normlite_core.parameters()) <= 7_000_000
    assert pcen_normlite_core.state_size_bytes(dtype=torch.float16) == capacity_core.state_size_bytes(
        dtype=torch.float16
    )


def test_source_aware_melband_loco_cnb_student_npu_onnx_audit_smoke() -> None:
    onnx = pytest.importorskip("onnx")

    from spectral_feature_compression.utils.onnx_streaming import (
        StreamingStateIOWrapper,
        flatten_tensor_tree,
        get_external_constant_tensors,
    )
    from tools.online.audit_onnx_model import audit_npu_risks, get_allowed_ops

    torch.manual_seed(0)
    core = OnlineSourceAwareMelBandLocoCNBStudentSFC2D(
        n_freq=33,
        sample_rate=8000,
        n_src=2,
        n_chan=1,
        n_bands=8,
        state_channels=8,
        source_channels=8,
        n_loco_layers=1,
        n_source_layers=1,
        source_fusion_hidden=16,
        source_seed_hidden=16,
        expander_hidden=16,
        mask_hidden=16,
        correction_channels=4,
        correction_kernel_size=(1, 3),
        cnb_kernel=3,
        cnb_dilation_schedule=(1, 2),
        cnb_num_heads=2,
        cnb_head_dim=4,
        pooled_mixer_hidden_schedule=(16,),
    ).eval()
    wrapper = StreamingStateIOWrapper(core, batch_size=1, dtype=torch.float32, externalize_constants=True).eval()
    state = core.init_stream_state(batch_size=1, dtype=torch.float32)
    flat_state, _ = flatten_tensor_tree(state)
    constants = get_external_constant_tensors(core, wrapper.constant_bindings)
    x = torch.randn(1, 2, 1, 33)

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "source_aware_melband_loco_cnb_student_stream.onnx"
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                (x, *flat_state, *constants),
                str(out),
                opset_version=14,
                input_names=[
                    "x",
                    *[f"state_{idx}" for idx in range(len(flat_state))],
                    *[f"const_{idx}" for idx in range(len(constants))],
                ],
                output_names=["y", *[f"next_state_{idx}" for idx in range(len(flat_state))]],
                do_constant_folding=True,
                dynamo=False,
            )
        model = onnx.load(str(out))

    onnx.checker.check_model(model)
    allowed_ops = get_allowed_ops("edge_npu_recommended")
    assert sorted({node.op_type for node in model.graph.node} - allowed_ops) == []
    audit = audit_npu_risks(model)
    assert audit["has_strict_edge_risks"] is False
    assert audit["risk_counts"]["rank_gt4_values"] == 0
    assert audit["risk_counts"]["activation_matmul_rank_le3"] == 0


def test_source_aware_melband_roformer_strong_student_npu_onnx_audit_smoke() -> None:
    onnx = pytest.importorskip("onnx")

    from spectral_feature_compression.utils.onnx_streaming import (
        StreamingStateIOWrapper,
        flatten_tensor_tree,
        get_external_constant_tensors,
    )
    from tools.online.audit_onnx_model import audit_npu_risks, get_allowed_ops

    torch.manual_seed(0)
    core = OnlineSourceAwareMelBandStrongStudentSFC2D(
        n_freq=33,
        sample_rate=8000,
        n_src=2,
        n_chan=1,
        n_bands=8,
        d_model=8,
        n_encoder_layers=1,
        n_source_layers=1,
        correction_layers=1,
        source_fusion_hidden=16,
        source_seed_hidden=16,
        expander_hidden=16,
        mask_hidden=16,
        correction_channels=4,
        kernel_size=(3, 3),
    ).eval()
    wrapper = StreamingStateIOWrapper(core, batch_size=1, dtype=torch.float32, externalize_constants=True).eval()
    state = core.init_stream_state(batch_size=1, dtype=torch.float32)
    flat_state, _ = flatten_tensor_tree(state)
    constants = get_external_constant_tensors(core, wrapper.constant_bindings)
    x = torch.randn(1, 2, 1, 33)

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "source_aware_melband_strong_student_stream.onnx"
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                (x, *flat_state, *constants),
                str(out),
                opset_version=14,
                input_names=[
                    "x",
                    *[f"state_{idx}" for idx in range(len(flat_state))],
                    *[f"const_{idx}" for idx in range(len(constants))],
                ],
                output_names=["y", *[f"next_state_{idx}" for idx in range(len(flat_state))]],
                do_constant_folding=True,
                dynamo=False,
            )
        model = onnx.load(str(out))

    onnx.checker.check_model(model)
    allowed_ops = get_allowed_ops("edge_npu_recommended")
    assert sorted({node.op_type for node in model.graph.node} - allowed_ops) == []
    audit = audit_npu_risks(model)
    assert audit["has_strict_edge_risks"] is False
    assert audit["risk_counts"]["rank_gt4_values"] == 0


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


def test_source_aware_residual_sfc_builder_forward_streaming_and_recipe() -> None:
    torch.manual_seed(0)
    model = build_source_aware_residual_sfc_system(
        n_fft=64,
        hop_length=16,
        fs=8000,
        n_src=2,
        n_chan=1,
        n_bands=8,
        d_model=8,
        n_shared_layers=1,
        n_source_layers=1,
        long_branch_layers=1,
        correction_layers=1,
        correction_channels=4,
        shared_capacity_layers=0,
        freq_preprocess_enabled=False,
        css_segment_size=1,
        css_shift_size=1,
    ).eval()
    wav = torch.randn(1, 1, 256)
    with torch.no_grad():
        est = model(wav)
    assert tuple(est.shape) == (1, 2, 1, 256)

    core = OnlineSourceAwareResidualSFC2D(
        n_freq=33,
        n_bands=8,
        sample_rate=8000,
        n_src=2,
        n_chan=1,
        d_model=8,
        n_shared_layers=1,
        n_source_layers=1,
        long_branch_layers=1,
        correction_layers=1,
        correction_channels=4,
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
    assert len(core.init_stream_state(batch_size=1, dtype=x.dtype)) == 5
    assert core.state_size_bytes(dtype=torch.float16) < 32 * 1024

    deploy_core = OnlineSourceAwareResidualSFC2D(
        n_freq=512,
        n_bands=56,
        sample_rate=44100,
        n_src=3,
        n_chan=1,
        d_model=28,
        n_shared_layers=2,
        n_source_layers=2,
        long_branch_layers=1,
        correction_layers=1,
        correction_channels=12,
        shared_capacity_hidden=8192,
        shared_capacity_layers=4,
    ).eval()
    deploy_params = sum(p.numel() for p in deploy_core.parameters())
    assert 2_000_000 <= deploy_params <= 7_000_000
    assert deploy_core.state_size_bytes(dtype=torch.float16) < 192 * 1024

    config_path = Path("recipes/dnr/models/source-aware-residual-sfc.rt192k.fp512keep475/config.yaml")
    top = merge_top_level_scalars(config_path)
    model_cfg = merge_task_model_mapping(config_path)
    context = {**top, **model_cfg}
    assert resolve_value(model_cfg["d_model"], context) == 28
    assert resolve_value(model_cfg["n_bands"], context) == 56
    assert resolve_value(model_cfg["correction_channels"], context) == 12

    system = build_model_system_from_recipe_config(config_path).eval()
    recipe_core = system.model.core
    assert recipe_core.n_freq == 512
    assert recipe_core.n_bands == 56
    assert recipe_core.d_model == 28
    assert recipe_core.correction_channels == 12
    assert sum(p.numel() for p in recipe_core.parameters()) == deploy_params
    assert recipe_core.state_size_bytes(dtype=torch.float16) < 192 * 1024

    distill_path = Path("recipes/dnr/models/source-aware-residual-sfc.distill.rt192k.fp512keep475/config.yaml")
    distill_top = merge_top_level_scalars(distill_path)
    distill_model_cfg = merge_task_model_mapping(distill_path)
    assert distill_top["teacher_checkpoint_path"] is None
    assert str(resolve_value(distill_model_cfg["_target_"], {**distill_top, **distill_model_cfg})).endswith(
        "build_source_aware_residual_sfc_system"
    )
    distill_text = distill_path.read_text(encoding="utf-8")
    assert "TeacherStudentDistillationTask" in distill_text
    assert "require_teacher_checkpoint: true" in distill_text
    assert "build_sfc_locoformer_lite_plus_system" in distill_text


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
            "pooled",
        ),
        (
            "recipes/dnr/models/"
            "band-sfc-net-npu.adaptive-mel-loco-cnb.stable-crossattn-query.rt192k.fp512keep475/config.yaml",
            "adaptive_mel_loco_cnb_stable_crossattn_query",
            36,
            48,
            "pooled",
        ),
        (
            "recipes/dnr/models/"
            "band-sfc-net-npu.adaptive-mel-loco-cnb.band56-soft-query.rt192k.fp512keep475/config.yaml",
            "adaptive_mel_loco_cnb_band56_soft_band_query",
            28,
            56,
            "pooled",
        ),
        (
            "recipes/dnr/models/"
            "band-sfc-net-npu.adaptive-mel-loco-cnb.clean-soft-query.rt192k.fp512keep475/config.yaml",
            "adaptive_mel_loco_cnb_clean_soft_band_query",
            36,
            48,
            "pointwise",
        ),
        (
            "recipes/dnr/models/"
            "band-sfc-net-npu.adaptive-mel-loco-cnb.stable-soft-query.residual-sfx.rt192k.fp512keep475/config.yaml",
            "adaptive_mel_loco_cnb_stable_soft_band_query",
            36,
            48,
            "pooled",
        ),
    ]
    for raw_path, preset, channels, n_bands, stage_mixer_type in checks:
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
        assert core.stage_mixer_type == stage_mixer_type
        assert sum(p.numel() for p in core.parameters()) < 4_000_000
        assert core.state_size_bytes(dtype=torch.float16) < 192 * 1024
        if "residual-sfx" in raw_path:
            assert system.model.residual_source_enabled is True
            assert system.model.n_src == 3
            assert core.n_src == 2
            wav = torch.randn(1, 1, 4096)
            with torch.no_grad():
                est = system(wav)
            assert tuple(est.shape) == (1, 3, 1, 4096)
        else:
            assert core.n_src == 3


def test_adaptive_mel_loco_cnb_distill_recipe_declares_teacher_task() -> None:
    checks = [
        (
            "recipes/dnr/models/"
            "band-sfc-net-npu.adaptive-mel-loco-cnb.soft-query.distill.rt192k.fp512keep475/config.yaml",
            "adaptive_mel_loco_cnb_soft_band_query",
        ),
        (
            "recipes/dnr/models/"
            "band-sfc-net-npu.adaptive-mel-loco-cnb.stable-soft-query.residual-sfx.distill.rt192k.fp512keep475/config.yaml",
            "adaptive_mel_loco_cnb_stable_soft_band_query",
        ),
    ]
    for raw_path, preset in checks:
        config_path = Path(raw_path)
        task_cfg = merge_task_model_mapping(config_path)
        top = merge_top_level_scalars(config_path)
        assert top["teacher_checkpoint_path"] is None
        # `merge_task_model_mapping` intentionally extracts only task.model.  Check
        # the raw file to avoid instantiating a distillation task without a teacher.
        text = config_path.read_text(encoding="utf-8")
        assert "TeacherStudentDistillationTask" in text
        assert "require_teacher_checkpoint: true" in text
        assert "build_sfc_locoformer_lite_plus_system" in text
        assert resolve_value(task_cfg["preset"], {**top, **task_cfg}) == preset


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


def test_distillation_checkpoint_loader_prefers_ema_weights(tmp_path: Path) -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    checkpoint_path = tmp_path / "teacher.ckpt"
    torch.save(
        {
            "state_dict": {
                "model.weight": torch.tensor([[1.0]]),
                "ema_model.module.weight": torch.tensor([[2.0]]),
            }
        },
        checkpoint_path,
    )

    _load_model_checkpoint(model, checkpoint_path)

    torch.testing.assert_close(model.weight, torch.tensor([[2.0]]))


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


def test_distillation_task_source_weights_prioritize_speech_losses() -> None:
    est = torch.zeros(1, 3, 1, 8)
    ref = torch.zeros_like(est)
    ref[:, 0] = 1.0

    base_task = TeacherStudentDistillationTask(
        model=torch.nn.Identity(),
        loss=torch.nn.L1Loss(),
        n_fft=32,
        hop_length=8,
        optimizer_config=object(),  # type: ignore[arg-type]
        robust_label_loss_weight=0.1,
    )
    speech_task = TeacherStudentDistillationTask(
        model=torch.nn.Identity(),
        loss=torch.nn.L1Loss(),
        n_fft=32,
        hop_length=8,
        optimizer_config=object(),  # type: ignore[arg-type]
        source_loss_weights={"speech": 3.0, "music": 1.0, "effects": 0.5},
        robust_label_loss_weight=0.1,
    )

    assert speech_task._robust_label_loss(est, ref) > base_task._robust_label_loss(est, ref)
    assert speech_task._source_activity_l1(est, ref) > base_task._source_activity_l1(est, ref)


def test_distillation_task_penalizes_frame_local_leakage_inside_active_source() -> None:
    task = TeacherStudentDistillationTask(
        model=torch.nn.Identity(),
        loss=torch.nn.L1Loss(),
        n_fft=32,
        hop_length=8,
        optimizer_config=object(),  # type: ignore[arg-type]
        fs=1000,
        frame_silent_source_weight=0.1,
        frame_silent_source_db=-50.0,
        frame_silent_window_ms=100.0,
        frame_silent_hop_ms=100.0,
    )
    ref = torch.zeros(1, 3, 1, 1000)
    est = torch.zeros_like(ref)
    ref[:, 0, :, :500] = 0.5
    est[:, 0, :, 500:] = 0.1

    assert task._silent_source_penalty(est, ref) == 0.0
    assert task._frame_silent_source_penalty(est, ref) > 0.0


def test_distillation_frame_silence_threshold_keeps_quiet_valid_reference_active() -> None:
    task = TeacherStudentDistillationTask(
        model=torch.nn.Identity(),
        loss=torch.nn.L1Loss(),
        n_fft=32,
        hop_length=8,
        optimizer_config=object(),  # type: ignore[arg-type]
        fs=1000,
        frame_silent_source_weight=0.1,
        frame_silent_source_db=-80.0,
        frame_silent_window_ms=500.0,
        frame_silent_hop_ms=500.0,
    )
    ref = torch.zeros(1, 1, 1, 1000)
    est = torch.zeros_like(ref)
    ref[..., :500] = 1.0e-3  # -60 dBFS: quiet, but not a silent target frame.
    est[..., :500] = 0.7
    est[..., 500:] = 0.1

    torch.testing.assert_close(task._frame_silent_source_penalty(est, ref), torch.tensor(0.01))


def test_distillation_task_rejects_unknown_source_weight_name() -> None:
    with pytest.raises(ValueError, match="unknown sources"):
        TeacherStudentDistillationTask(
            model=torch.nn.Identity(),
            loss=torch.nn.L1Loss(),
            n_fft=32,
            hop_length=8,
            optimizer_config=object(),  # type: ignore[arg-type]
            source_loss_weights={"dialog": 1.0},
        )


def test_distillation_task_rejects_source_weight_length_mismatch() -> None:
    with pytest.raises(ValueError, match="source_order has 3"):
        TeacherStudentDistillationTask(
            model=torch.nn.Identity(),
            loss=torch.nn.L1Loss(),
            n_fft=32,
            hop_length=8,
            optimizer_config=object(),  # type: ignore[arg-type]
            source_loss_weights=(1.0, 0.5),
        )


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


class _ToyBandSpecSeparator(torch.nn.Module):
    def __init__(self, n_bands: int, scale: float) -> None:
        super().__init__()
        self.band_spec = AdaptiveMelBandSpec2d(n_freq=33, n_bands=n_bands, sample_rate=16000)
        self.scale = scale

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return torch.stack([wav * self.scale, wav * (1.0 - self.scale)], dim=1)


class _ToyBasisSpec(torch.nn.Module):
    def __init__(self, basis: torch.Tensor) -> None:
        super().__init__()
        self.n_bands = int(basis.shape[0])
        self.n_freq = int(basis.shape[1])
        self.register_buffer("basis", basis.view(1, self.n_bands, 1, self.n_freq))


class _ToyBasisSpecSeparator(torch.nn.Module):
    def __init__(self, basis: torch.Tensor, scale: float) -> None:
        super().__init__()
        self.band_spec = _ToyBasisSpec(basis)
        self.scale = scale

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return torch.stack([wav * self.scale, wav * (1.0 - self.scale)], dim=1)


class _ToyFrequencyProjector(torch.nn.Module):
    def __init__(self, analysis_matrix: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("analysis_matrix", analysis_matrix)
        self.register_buffer("synthesis_matrix", analysis_matrix.transpose(0, 1))


class _ToyFrequencyProjectorSeparator(torch.nn.Module):
    def __init__(self, analysis_matrix: torch.Tensor, scale: float) -> None:
        super().__init__()
        self.freq_preprocessor = _ToyFrequencyProjector(analysis_matrix)
        self.scale = scale

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return torch.stack([wav * self.scale, wav * (1.0 - self.scale)], dim=1)


class _ToyAuxSeparator(torch.nn.Module):
    def __init__(
        self,
        scale: float,
        *,
        mask_channels: int,
        mask_frames: int,
        mask_bins: int,
        logit_domain: str | None = "toy_logits",
    ) -> None:
        super().__init__()
        self.scale = scale
        self.mask_channels = mask_channels
        self.mask_frames = mask_frames
        self.mask_bins = mask_bins
        self.logit_domain = logit_domain

    def forward(self, wav: torch.Tensor, *, return_aux: bool = False) -> torch.Tensor | tuple[torch.Tensor, dict]:
        est = torch.stack([wav * self.scale, wav * (1.0 - self.scale)], dim=1)
        if not return_aux:
            return est
        mask = wav.new_full((wav.shape[0], self.mask_channels, self.mask_frames, self.mask_bins), self.scale)
        aux = {
            "mask": mask,
            "mask_domain": "packed_complex_mask",
            "mask_logits": mask,
        }
        if self.logit_domain is not None:
            aux["mask_logits_domain"] = self.logit_domain
        return est, aux


class _NoAuxSeparator(torch.nn.Module):
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return torch.stack([wav, wav * 0.0], dim=1)


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


def test_distillation_task_requests_aux_and_applies_source_specific_losses() -> None:
    student = _ToyAuxSeparator(scale=0.4, mask_channels=4, mask_frames=3, mask_bins=5)
    teacher = _ToyAuxSeparator(scale=0.6, mask_channels=6, mask_frames=4, mask_bins=7)
    task = TeacherStudentDistillationTask(
        model=student,
        teacher_model=teacher,
        loss=torch.nn.L1Loss(),
        n_fft=32,
        hop_length=8,
        optimizer_config=object(),  # type: ignore[arg-type]
        request_model_aux=True,
        teacher_mask_loss_weight=0.1,
        teacher_logit_loss_weight=0.1,
        source_order=("speech", "effects"),
        source_loss_weights=(1.6, 0.85),
        source_weighted_snr_loss_weight=0.1,
        explicit_source_loss_weight=0.1,
        residual_source_loss_weight=0.1,
        residual_source_index=1,
        mask_aux_alignment="shared_prefix",
        mask_aux_max_frame_mismatch=1,
        distillation_band_mapping="linear",
    )
    wav = torch.randn(2, 1, 128)
    ref = torch.stack([wav * 0.7, wav * 0.3], dim=1)

    est, student_aux = task._forward_model(task.model, wav, ref, "training", css_validation=False)
    teacher_est, teacher_aux = task._teacher_forward(wav, ref=ref, log_prefix="training")
    losses = [
        task._source_weighted_snr_loss(est, ref),
        task._robust_source_subset_loss(est, ref, (0,)),
        task._robust_source_subset_loss(est, ref, (1,)),
        task._mask_or_logit_loss(est, teacher_est, wav, student_aux, teacher_aux, logit=False),
        task._mask_or_logit_loss(est, teacher_est, wav, student_aux, teacher_aux, logit=True),
    ]

    assert set(student_aux) == {"mask", "mask_domain", "mask_logits", "mask_logits_domain"}
    assert student_aux["mask"].shape == (2, 4, 3, 5)
    assert all(loss.ndim == 0 and torch.isfinite(loss) for loss in losses)


def test_distillation_task_requires_model_aux_when_configured() -> None:
    task = TeacherStudentDistillationTask(
        model=_NoAuxSeparator(),
        teacher_model=_NoAuxSeparator(),
        loss=torch.nn.L1Loss(),
        n_fft=32,
        hop_length=8,
        optimizer_config=object(),  # type: ignore[arg-type]
        request_model_aux=True,
        require_model_aux=True,
    )
    wav = torch.randn(1, 1, 64)
    ref = torch.stack([wav, wav * 0.0], dim=1)

    with pytest.raises(RuntimeError, match="does not accept return_aux"):
        task._forward_model(task.model, wav, ref, "training", css_validation=False)


def test_distillation_task_logit_aux_requires_matching_domains() -> None:
    task = TeacherStudentDistillationTask(
        model=torch.nn.Identity(),
        teacher_model=torch.nn.Identity(),
        loss=torch.nn.L1Loss(),
        n_fft=32,
        hop_length=8,
        optimizer_config=object(),  # type: ignore[arg-type]
    )
    wav = torch.ones(1, 1, 64)
    est = torch.stack([wav, wav], dim=1)
    teacher_est = est.clone()
    student_logits = torch.zeros(1, 2, 3, 4)
    teacher_logits = torch.full_like(student_logits, 100.0)

    mismatch_loss = task._mask_or_logit_loss(
        est,
        teacher_est,
        wav,
        {"mask_logits": student_logits, "mask_logits_domain": "student_raw"},
        {"mask_logits": teacher_logits, "mask_logits_domain": "teacher_raw"},
        logit=True,
    )
    matched_loss = task._mask_or_logit_loss(
        est,
        teacher_est,
        wav,
        {"mask_logits": student_logits, "mask_logits_domain": "shared_raw"},
        {"mask_logits": teacher_logits, "mask_logits_domain": "shared_raw"},
        logit=True,
    )

    assert mismatch_loss.item() == pytest.approx(0.0)
    assert matched_loss > 10.0


def test_distillation_task_converts_teacher_mask_to_student_logit_domain() -> None:
    task = TeacherStudentDistillationTask(
        model=torch.nn.Identity(),
        teacher_model=torch.nn.Identity(),
        loss=torch.nn.L1Loss(),
        n_fft=32,
        hop_length=8,
        optimizer_config=object(),  # type: ignore[arg-type]
    )
    wav = torch.ones(1, 1, 64)
    est = torch.stack([wav, wav], dim=1)
    teacher_est = est.clone()
    student_logits = torch.tensor([[[[-1.0, 0.5]], [[0.25, -0.75]]]])
    real_scale = 1.5
    imag_scale = 0.2
    teacher_mask = student_logits.clone()
    teacher_mask[:, 0::2] = torch.sigmoid(student_logits[:, 0::2]) * real_scale
    teacher_mask[:, 1::2] = torch.tanh(student_logits[:, 1::2]) * imag_scale

    loss = task._mask_or_logit_loss(
        est,
        teacher_est,
        wav,
        {
            "mask_logits": student_logits,
            "mask_logits_domain": "tvconv_pyramid_complex_mask_logits",
            "mask_logits_transform": "sigmoid_tanh_complex_mask",
            "mask_logits_real_scale": real_scale,
            "mask_logits_imag_scale": imag_scale,
        },
        {
            "mask": teacher_mask,
            "mask_domain": "packed_complex_mask",
            "mask_logits": torch.full_like(student_logits, 100.0),
            "mask_logits_domain": "source_aware_melband_roformer_complex_mask_logits",
        },
        logit=True,
    )

    assert loss.item() == pytest.approx(0.0, abs=1.0e-5)


def test_distillation_task_mask_aux_alignment_is_explicit() -> None:
    strict_task = TeacherStudentDistillationTask(
        model=torch.nn.Identity(),
        teacher_model=torch.nn.Identity(),
        loss=torch.nn.L1Loss(),
        n_fft=32,
        hop_length=8,
        optimizer_config=object(),  # type: ignore[arg-type]
        distillation_band_mapping="linear",
    )
    shared_task = TeacherStudentDistillationTask(
        model=torch.nn.Identity(),
        teacher_model=torch.nn.Identity(),
        loss=torch.nn.L1Loss(),
        n_fft=32,
        hop_length=8,
        optimizer_config=object(),  # type: ignore[arg-type]
        mask_aux_alignment="shared_prefix",
        mask_aux_max_frame_mismatch=1,
        distillation_band_mapping="linear",
    )
    wav = torch.randn(2, 1, 128)
    est = torch.stack([wav * 0.7, wav * 0.3], dim=1)
    teacher_est = est.clone()
    student_aux = {
        "mask": torch.zeros(2, 4, 3, 5),
        "mask_domain": "packed_complex_mask",
    }
    teacher_aux = {
        "mask": torch.ones(2, 6, 4, 7),
        "mask_domain": "packed_complex_mask",
    }

    with pytest.raises(ValueError, match="shape mismatch"):
        strict_task._mask_or_logit_loss(est, teacher_est, wav, student_aux, teacher_aux, logit=False)
    loss = shared_task._mask_or_logit_loss(est, teacher_est, wav, student_aux, teacher_aux, logit=False)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_distillation_task_latent_allow_missing_handles_parameterless_models() -> None:
    task = TeacherStudentDistillationTask(
        model=torch.nn.Identity(),
        teacher_model=torch.nn.Identity(),
        loss=torch.nn.L1Loss(),
        n_fft=32,
        hop_length=8,
        optimizer_config=object(),  # type: ignore[arg-type]
        latent_distillation_weight=0.1,
        latent_allow_missing=True,
    )

    loss = task._latent_distillation_loss({}, {})

    assert loss.ndim == 0
    assert loss.item() == pytest.approx(0.0)


def test_distillation_task_maps_teacher_band_tensors_to_student_grid() -> None:
    student = _ToyBandSpecSeparator(n_bands=4, scale=0.4)
    teacher = _ToyBandSpecSeparator(n_bands=6, scale=0.6)
    task = TeacherStudentDistillationTask(
        model=student,
        teacher_model=teacher,
        loss=torch.nn.L1Loss(),
        n_fft=32,
        hop_length=8,
        optimizer_config=object(),  # type: ignore[arg-type]
        teacher_mask_loss_weight=0.1,
        teacher_logit_loss_weight=0.1,
        latent_distillation_weight=0.1,
        distillation_band_mapping="mel_centers",
    )
    wav = torch.randn(2, 1, 128)
    est = student(wav)
    teacher_est = teacher(wav)
    student_band = torch.zeros(2, 2, 3, 4)
    teacher_band = torch.ones(2, 2, 3, 6)
    student_aux = {
        "mask": student_band,
        "mask_logits": student_band,
        "latents": {"z": student_band},
    }
    teacher_aux = {
        "mask": teacher_band,
        "mask_logits": teacher_band,
        "latents": {"z": teacher_band},
    }

    _, mapped_teacher = task._align_distillation_tensors(student_band, teacher_band, name="test")
    losses = [
        task._mask_or_logit_loss(est, teacher_est, wav, student_aux, teacher_aux, logit=False),
        task._mask_or_logit_loss(est, teacher_est, wav, student_aux, teacher_aux, logit=True),
        task._latent_distillation_loss(student_aux, teacher_aux),
    ]

    assert mapped_teacher.shape == student_band.shape
    assert all(loss.ndim == 0 and torch.isfinite(loss) for loss in losses)


def test_distillation_task_uses_overlap_basis_for_band_mapping() -> None:
    student_basis = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ]
    )
    teacher_basis = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ]
    )
    student = _ToyBasisSpecSeparator(student_basis, scale=0.4)
    teacher = _ToyBasisSpecSeparator(teacher_basis, scale=0.6)
    task = TeacherStudentDistillationTask(
        model=student,
        teacher_model=teacher,
        loss=torch.nn.L1Loss(),
        n_fft=32,
        hop_length=8,
        optimizer_config=object(),  # type: ignore[arg-type]
        distillation_band_mapping="mel_centers",
    )
    student_band = torch.zeros(1, 1, 1, 3)
    teacher_band = torch.tensor([[[[2.0, 10.0]]]])

    _, mapped_teacher = task._align_distillation_tensors(student_band, teacher_band, name="test")

    expected = torch.tensor([[[[2.0, 6.0, 10.0]]]])
    torch.testing.assert_close(mapped_teacher, expected)


def test_distillation_task_uses_frequency_projector_for_dense_aux_mapping() -> None:
    analysis_matrix = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.25, 0.75, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    student = _ToyFrequencyProjectorSeparator(analysis_matrix, scale=0.4)
    teacher = torch.nn.Identity()
    task = TeacherStudentDistillationTask(
        model=student,
        teacher_model=teacher,
        loss=torch.nn.L1Loss(),
        n_fft=32,
        hop_length=8,
        optimizer_config=object(),  # type: ignore[arg-type]
        distillation_band_mapping="linear",
    )
    student_dense = torch.zeros(1, 1, 1, 3)
    teacher_dense = torch.tensor([[[[1.0, 2.0, 6.0, 20.0, 30.0]]]])

    _, mapped_teacher = task._align_distillation_tensors(student_dense, teacher_dense, name="test")

    expected = torch.tensor([[[[1.0, 5.0, 30.0]]]])
    torch.testing.assert_close(mapped_teacher, expected)


def test_distillation_task_mel_mapping_falls_back_for_dense_frequency_masks() -> None:
    student = _ToyBandSpecSeparator(n_bands=4, scale=0.4)
    teacher = _ToyBandSpecSeparator(n_bands=6, scale=0.6)
    task = TeacherStudentDistillationTask(
        model=student,
        teacher_model=teacher,
        loss=torch.nn.L1Loss(),
        n_fft=32,
        hop_length=8,
        optimizer_config=object(),  # type: ignore[arg-type]
        mask_aux_alignment="shared_prefix",
        mask_aux_max_frame_mismatch=1,
        distillation_band_mapping="mel_centers",
    )
    wav = torch.randn(2, 1, 128)
    est = student(wav)
    teacher_est = teacher(wav)
    student_aux = {
        "mask": torch.zeros(2, 4, 3, 5),
        "mask_domain": "packed_complex_mask",
    }
    teacher_aux = {
        "mask": torch.ones(2, 6, 4, 7),
        "mask_domain": "packed_complex_mask",
    }

    loss = task._mask_or_logit_loss(est, teacher_est, wav, student_aux, teacher_aux, logit=False)

    assert loss.ndim == 0 and torch.isfinite(loss)
