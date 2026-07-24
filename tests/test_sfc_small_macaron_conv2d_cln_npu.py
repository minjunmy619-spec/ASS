from __future__ import annotations

from collections import Counter
from pathlib import Path
import tempfile

import torch

from spectral_feature_compression.core.model.sfc_small_macaron_conv2d_cln_npu import (
    CumulativeLayerNorm2D,
    SFCSmallMacaronConv2DCLNNPUCore,
)
from spectral_feature_compression.utils.onnx_streaming import StreamingStateIOWrapper, flatten_tensor_tree
from tools.online.export_onnx_online_model import build_model_system_from_recipe_config


RECIPE = Path("recipes/dnr/models/sfc-small-macaron-conv2d-cln-npu.musical36.2l.onfly.rt192k/config.yaml")


def _small_core() -> SFCSmallMacaronConv2DCLNNPUCore:
    return SFCSmallMacaronConv2DCLNNPUCore(
        n_freq=65,
        n_fft=128,
        n_bands=36,
        n_src=2,
        d_inner=8,
        d_model=8,
        ffn_hidden=12,
        n_separator_layers=2,
        n_sfc_heads=2,
        frequency_kernel_size=15,
        time_kernel_size=2,
        dilation_cycle=(1,),
        decoder_ffn_hidden=4,
    )


def test_cumulative_layer_norm_matches_direct_reference() -> None:
    torch.manual_seed(0)
    norm = CumulativeLayerNorm2D(8).eval()
    x = torch.randn(2, 8, 5, 36)
    observed = norm(x)

    values = x.permute(0, 2, 1, 3).reshape(2, 5, -1)
    cumulative_sum = values.sum(dim=-1).cumsum(dim=1)
    cumulative_square = values.square().sum(dim=-1).cumsum(dim=1)
    count = torch.arange(1, 6, dtype=x.dtype).view(1, 5) * values.shape[-1]
    mean = (cumulative_sum / count).view(2, 1, 5, 1)
    second = (cumulative_square / count).view(2, 1, 5, 1)
    expected = (x - mean) * torch.rsqrt(torch.relu(second - mean.square()) + norm.eps)
    torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-5)


def test_cumulative_macaron_streaming_matches_full_eval() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = _small_core().eval()
    x = torch.randn(1, 2, 5, 65)
    with torch.no_grad():
        full = model(x)
        state = model.init_stream_state(dtype=x.dtype)
        frames = []
        for frame_idx in range(x.shape[2]):
            frame, state = model.forward_stream(x[:, :, frame_idx : frame_idx + 1], state)
            frames.append(frame)
    torch.testing.assert_close(torch.cat(frames, dim=2), full, rtol=1e-5, atol=1e-5)


def test_cumulative_macaron_state_abi_and_validation() -> None:
    model = _small_core().eval()
    state = model.init_stream_state()
    assert len(state) == 31
    assert tuple(state[0].shape) == (1, 1, 1, 1)
    assert sum(tuple(tensor.shape) == (1, 8, 1, 36) for tensor in state) == 6

    x = torch.randn(1, 2, 1, 65)
    for invalid in (state[:-1], (*state, state[0])):
        try:
            model.forward_stream(x, invalid)
        except RuntimeError as error:
            assert "Expected 31 separator states" in str(error)
        else:
            raise AssertionError("Invalid cumulative-LN state count was accepted")

    try:
        model.forward_stream(torch.randn(1, 2, 2, 65), state)
    except RuntimeError as error:
        assert "streaming expects one frame" in str(error)
    else:
        raise AssertionError("Multi-frame cumulative-LN streaming input was accepted")


def test_cumulative_macaron_recipe_budget_and_normalization_layout() -> None:
    system = build_model_system_from_recipe_config(RECIPE).eval()
    model = system.model.core.to(dtype=torch.float32)
    assert sum(parameter.numel() for parameter in model.parameters()) == 995_190
    assert len(model.init_stream_state()) == 31
    assert sum(isinstance(module, CumulativeLayerNorm2D) for module in model.separator.modules()) == 12
    assert not any(isinstance(module, torch.nn.BatchNorm2d) for module in model.separator.modules())
    assert model.state_size_bytes(dtype=torch.float16) == 55_346


def test_cumulative_macaron_streaming_onnx_avoids_reduction_layout_ops() -> None:
    import onnx

    torch.set_num_threads(1)
    model = _small_core().eval()
    model.masking = False
    state = model.init_stream_state(dtype=torch.float32)
    wrapper = StreamingStateIOWrapper(model, batch_size=1, dtype=torch.float32).eval()
    x = torch.randn(1, 2, 1, 65)
    flat_state, _ = flatten_tensor_tree(state)

    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "macaron_cln.onnx"
        torch.onnx.export(
            wrapper,
            (x, *flat_state),
            output,
            opset_version=14,
            do_constant_folding=True,
            dynamo=False,
        )
        graph = onnx.load(output)
        onnx.checker.check_model(graph)

    counts = Counter(node.op_type for node in graph.graph.node)
    assert counts["AveragePool"] == 48
    assert counts["ReduceMean"] == 0
    assert counts["ReduceSum"] == 0
    assert counts["Slice"] == 0
    assert counts["Split"] == 0
    assert counts["Transpose"] == 2
    assert counts["Reshape"] == 6
    assert len(graph.graph.input) == 32
    assert len(graph.graph.output) == 32
