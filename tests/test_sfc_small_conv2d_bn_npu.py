from __future__ import annotations

from collections import Counter
from pathlib import Path
import tempfile

import torch

import pytest

from spectral_feature_compression.core.model.sfc_small_conv2d_bn_npu import (
    SFCSmallConv2DBNNPUCore,
    SFCSmallConv2DBNNPUModel,
)
from spectral_feature_compression.utils.onnx_streaming import StreamingStateIOWrapper, flatten_tensor_tree
from tools.online.export_onnx_online_model import build_model_system_from_recipe_config


def test_sfc_small_conv2d_bn_public_lazy_exports() -> None:
    import spectral_feature_compression as sfc

    assert sfc.SFCSmallConv2DBNNPUCore is SFCSmallConv2DBNNPUCore
    assert sfc.SFCSmallConv2DBNNPUModel is SFCSmallConv2DBNNPUModel


def test_sfc_small_conv2d_bn_core_streaming_matches_full() -> None:
    torch.manual_seed(0)
    model = SFCSmallConv2DBNNPUCore(
        n_freq=65,
        n_bands=8,
        n_src=2,
        n_chan=1,
        d_inner=16,
        d_model=32,
        n_separator_layers=3,
        time_kernel_size=2,
        dilation_cycle=(1,),
    ).eval()
    x = torch.randn(1, 2, 6, 65)

    with torch.no_grad():
        y_full, mask = model(x, return_mask=True)
        state = model.init_stream_state(batch_size=1, dtype=x.dtype)
        frames = []
        for frame_idx in range(x.shape[2]):
            y_frame, state = model.forward_stream(x[:, :, frame_idx : frame_idx + 1, :], state)
            frames.append(y_frame)
        y_stream = torch.cat(frames, dim=2)

    assert tuple(y_full.shape) == (1, 4, 6, 65)
    assert tuple(mask.shape) == (1, 4, 6, 65)
    torch.testing.assert_close(y_stream, y_full, rtol=1e-5, atol=1e-5)


def test_sfc_small_conv2d_bn_uses_batchnorm_without_rmsnorm() -> None:
    model = SFCSmallConv2DBNNPUCore(n_freq=129, n_bands=16, d_inner=32, d_model=48, n_separator_layers=2)
    modules = list(model.modules())
    assert any(isinstance(module, torch.nn.BatchNorm2d) for module in modules)
    assert not any(module.__class__.__name__.lower().startswith("rms") for module in modules)
    assert not any(isinstance(module, torch.nn.MultiheadAttention) for module in modules)


def test_sfc_small_conv2d_bn_uses_sfc_query_transport() -> None:
    model = SFCSmallConv2DBNNPUCore(
        n_freq=129,
        n_bands=16,
        d_inner=32,
        d_model=48,
        n_separator_layers=2,
        n_sfc_heads=4,
    )
    assert tuple(model.encoder.query.shape) == (4, 16, 8)
    assert tuple(model.encoder.pos_bias.shape) == (4, 16, 129)
    assert tuple(model.decoder.query.shape) == (4, 129, 8)
    assert tuple(model.decoder.pos_bias.shape) == (4, 129, 16)
    assert not any(isinstance(module, torch.nn.ConvTranspose2d) for module in model.modules())


def test_sfc_small_conv2d_bn_default_budget() -> None:
    model = SFCSmallConv2DBNNPUCore(n_freq=1025, n_bands=64).eval()
    params = sum(param.numel() for param in model.parameters())
    state_bytes = model.state_size_bytes(dtype=torch.float16)
    assert 3_000_000 <= params <= 4_000_000
    assert state_bytes < 192 * 1024
    assert model.d_model == 160
    assert len(model.separator) == 8
    assert model.dilation_schedule == (1,) * 8
    assert len(model.init_stream_state(batch_size=1, dtype=torch.float16)) == 8


def test_sfc_small_conv2d_bn_waveform_wrapper_shape() -> None:
    torch.manual_seed(0)
    model = SFCSmallConv2DBNNPUModel(
        n_freq=65,
        n_bands=8,
        n_src=3,
        n_chan=1,
        d_inner=16,
        d_model=32,
        n_separator_layers=2,
    ).eval()
    stft = torch.randn(1, 1, 65, 4, dtype=torch.complex64)
    with torch.no_grad():
        y = model(stft)
    assert tuple(y.shape) == (1, 3, 1, 65, 4)


def test_sfc_small_conv2d_bn_recipe_builds_onfly_system() -> None:
    config_path = Path("recipes/dnr/models/sfc-small-conv2d-bn-npu.musical64.onfly.rt192k/config.yaml")
    system = build_model_system_from_recipe_config(config_path).eval()
    core = system.model.core
    assert isinstance(core, SFCSmallConv2DBNNPUCore)
    assert core.n_freq == 1025
    assert core.n_bands == 64
    assert core.d_inner == 64
    assert core.d_model == 160
    assert core.state_size_bytes(dtype=torch.float16) < 192 * 1024


def test_sfc_small_conv2d_bn_streaming_onnx_export_uses_npu_friendly_ops() -> None:
    onnx = pytest.importorskip("onnx")

    model = SFCSmallConv2DBNNPUCore(
        n_freq=65,
        n_bands=8,
        n_src=2,
        n_chan=1,
        d_inner=16,
        d_model=32,
        n_separator_layers=2,
    ).eval()
    wrapper = StreamingStateIOWrapper(model, batch_size=1, dtype=torch.float32)
    state = model.init_stream_state(batch_size=1, dtype=torch.float32)
    flat_state, _ = flatten_tensor_tree(state)
    x = torch.randn(1, 2, 1, 65)

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "sfc_small_conv2d_bn_npu.onnx"
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                (x, *flat_state),
                str(out),
                opset_version=11,
                input_names=["x", *[f"state_{idx}" for idx in range(len(flat_state))]],
                output_names=["y", *[f"next_state_{idx}" for idx in range(len(flat_state))]],
                do_constant_folding=True,
                dynamo=False,
            )
        graph = onnx.load(str(out))
        onnx.checker.check_model(graph)

    ops = {node.op_type for node in graph.graph.node}
    op_counts = Counter(node.op_type for node in graph.graph.node)
    assert "Conv" in ops
    assert "MatMul" in ops
    assert "Softmax" in ops
    assert op_counts["MatMul"] == 4
    assert op_counts["Softmax"] == 2
    assert not {"Gemm", "LayerNormalization", "RMSNormalization"} & ops
    assert not {"Pad", "ConstantOfShape", "Expand", "Tile"} & ops
