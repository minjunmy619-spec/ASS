from __future__ import annotations

from pathlib import Path

import torch

import pytest

from spectral_feature_compression.core.model.proposed_separation_models import (
    build_sfc_locoformer_conv2d_32b_npu_system,
)
from spectral_feature_compression.core.model.sfc_locoformer_conv2d_32b_npu import (
    FoldedFullBandSourceAwareComplexHead2d,
    OnlineSFCLocoformerConv2D32BNPU2D,
)
from tools.online.export_onnx_online_model import (
    build_model_system_from_recipe_config,
    merge_task_model_mapping,
    merge_top_level_scalars,
    resolve_value,
)


def test_sfc_locoformer_conv2d_32b_public_lazy_exports() -> None:
    import spectral_feature_compression as sfc

    assert sfc.OnlineSFCLocoformerConv2D32BNPU2D is OnlineSFCLocoformerConv2D32BNPU2D
    assert sfc.FoldedFullBandSourceAwareComplexHead2d is FoldedFullBandSourceAwareComplexHead2d


def test_sfc_locoformer_conv2d_core_aux_and_streaming_match() -> None:
    torch.manual_seed(0)
    core = OnlineSFCLocoformerConv2D32BNPU2D(
        n_freq=64,
        n_bands=8,
        sample_rate=8000,
        n_src=2,
        n_chan=1,
        d_model=32,
        n_loco_layers=2,
        dilation_cycle=(1, 2),
        source_head_channels=16,
        source_refine_layers=1,
        source_kernel_size=3,
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
    assert aux["mask_domain"] == "packed_complex_mask"
    assert aux["mask_logits_domain"] == "sfc_locoformer_conv2d_32b_complex_mask_logits"
    assert aux["mask_logits_transform"] == "sigmoid_tanh_complex_mask"
    assert aux["mask_logits_real_scale"] == pytest.approx(1.5)
    assert aux["mask_logits_imag_scale"] == pytest.approx(0.12)
    torch.testing.assert_close(y_stream, y, rtol=1e-5, atol=1e-5)


def test_sfc_locoformer_conv2d_folded_fullband_head_streaming_match() -> None:
    torch.manual_seed(0)
    core = OnlineSFCLocoformerConv2D32BNPU2D(
        n_freq=64,
        n_bands=8,
        sample_rate=8000,
        n_src=2,
        n_chan=1,
        d_model=24,
        n_loco_layers=2,
        dilation_cycle=(1, 2),
        expansion=1,
        ffn_expansion=3,
        source_head_type="folded_fullband",
        source_head_channels=8,
        source_bottleneck_channels=4,
        source_refine_layers=1,
        source_kernel_size=3,
        fullres_skip_enabled=True,
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
    assert isinstance(core.source_head, FoldedFullBandSourceAwareComplexHead2d)
    assert core.fullres_skip_enabled is True
    assert aux["mask_logits_source_head_type"] == "folded_fullband"
    assert aux["mask_logits_fullres_skip_enabled"] is True
    torch.testing.assert_close(y_stream, y, rtol=1e-5, atol=1e-5)


def test_sfc_locoformer_conv2d_waveform_wrapper_residual_sfx_aux() -> None:
    torch.manual_seed(0)
    model = build_sfc_locoformer_conv2d_32b_npu_system(
        n_fft=64,
        hop_length=16,
        fs=8000,
        n_src=3,
        n_chan=1,
        core_n_src=2,
        n_bands=8,
        d_model=32,
        n_loco_layers=2,
        dilation_cycle=(1, 2),
        source_head_channels=16,
        source_refine_layers=1,
        source_kernel_size=3,
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
    assert model.model.residual_source_enabled is True
    assert model.model.core.n_src == 2
    assert aux["mask_domain"] == "packed_complex_mask"
    assert aux["mask_logits_domain"] == "sfc_locoformer_conv2d_32b_complex_mask_logits"
    assert aux["mask"].shape[1] == 4


def test_sfc_locoformer_conv2d_recipe_builds_with_deploy_budget() -> None:
    config_path = Path(
        "recipes/dnr/models/"
        "sfc-locoformer-conv2d-32b-npu.2l.sourceaware-residual-sfx.distill.rt192k.fp512keep475/"
        "config.yaml"
    )
    top = merge_top_level_scalars(config_path)
    model_cfg = merge_task_model_mapping(config_path)
    context = {**top, **model_cfg}
    assert str(resolve_value(model_cfg["_target_"], context)).endswith("build_sfc_locoformer_conv2d_32b_npu_system")
    assert resolve_value(model_cfg["n_bands"], context) == 32
    assert resolve_value(model_cfg["d_model"], context) == 192
    assert resolve_value(model_cfg["n_loco_layers"], context) == 2
    assert resolve_value(model_cfg["dilation_cycle"], context) == [1, 2]

    system = build_model_system_from_recipe_config(config_path).eval()
    core = system.model.core
    assert core.n_freq == 512
    assert core.n_bands == 32
    assert core.d_model == 192
    assert len(core.separator) == 2
    assert core.state_size_bytes(dtype=torch.float16) < 192 * 1024
    assert system.model.residual_source_enabled is True
    grouped = [
        name
        for name, module in core.named_modules()
        if isinstance(module, torch.nn.Conv2d) and int(module.groups) != 1
    ]
    assert grouped == ["compressor.dw.conv"]
    params = sum(p.numel() for p in core.parameters())
    assert 3_000_000 <= params <= 8_000_000


def test_sfc_locoformer_conv2d_v2_recipe_builds_with_deploy_budget() -> None:
    config_path = Path(
        "recipes/dnr/models/"
        "sfc-locoformer-conv2d-48b-npu.4l.fullskip-folded-residual-sfx.distill.rt192k.fp512keep475/"
        "config.yaml"
    )
    top = merge_top_level_scalars(config_path)
    model_cfg = merge_task_model_mapping(config_path)
    context = {**top, **model_cfg}
    assert str(resolve_value(model_cfg["_target_"], context)).endswith("build_sfc_locoformer_conv2d_32b_npu_system")
    assert resolve_value(model_cfg["n_bands"], context) == 48
    assert resolve_value(model_cfg["d_model"], context) == 160
    assert resolve_value(model_cfg["n_loco_layers"], context) == 4
    assert resolve_value(model_cfg["dilation_cycle"], context) == [1, 2, 1, 2]
    assert resolve_value(model_cfg["expansion"], context) == 1
    assert resolve_value(model_cfg["ffn_expansion"], context) == 5
    assert resolve_value(model_cfg["source_head_type"], context) == "folded_fullband"
    assert resolve_value(model_cfg["fullres_skip_enabled"], context) is True

    system = build_model_system_from_recipe_config(config_path).eval()
    core = system.model.core
    assert core.n_freq == 512
    assert core.n_bands == 48
    assert core.d_model == 160
    assert len(core.separator) == 4
    assert core.source_head_type == "folded_fullband"
    assert core.fullres_skip_enabled is True
    assert isinstance(core.source_head, FoldedFullBandSourceAwareComplexHead2d)
    assert core.state_size_bytes(dtype=torch.float16) < 192 * 1024
    assert system.model.residual_source_enabled is True
    grouped = [
        name
        for name, module in core.named_modules()
        if isinstance(module, torch.nn.Conv2d) and int(module.groups) != 1
    ]
    assert grouped == ["compressor.dw.conv"]
    params = sum(p.numel() for p in core.parameters())
    assert 3_000_000 <= params <= 8_000_000
