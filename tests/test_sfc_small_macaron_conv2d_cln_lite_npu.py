from __future__ import annotations

from collections import Counter
from pathlib import Path
import tempfile

import torch

from spectral_feature_compression.core.model.sfc_small_macaron_conv2d_cln_lite_npu import (
    SFCSmallMacaronConv2DCLNLiteNPUCore,
    SharedCumulativeStatistics2D,
)
from spectral_feature_compression.utils.onnx_streaming import StreamingStateIOWrapper, flatten_tensor_tree
from tools.online.export_onnx_online_model import build_model_system_from_recipe_config


RECIPE = Path(
    "recipes/dnr/models/sfc-small-macaron-conv2d-cln-lite-npu.musical36.2l.onfly.rt192k/config.yaml"
)


def _small_core() -> SFCSmallMacaronConv2DCLNLiteNPUCore:
    return SFCSmallMacaronConv2DCLNLiteNPUCore(
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


def test_cln_lite_streaming_matches_full_eval() -> None:
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


def test_cln_lite_default_budget_and_state_reduction() -> None:
    model = build_model_system_from_recipe_config(RECIPE).model.core.to(dtype=torch.float32)
    state = model.init_stream_state()
    assert sum(parameter.numel() for parameter in model.parameters()) == 995_190
    assert len(state) == 15
    assert model.state_size_bytes(dtype=torch.float16) == 55_314
    assert sum(isinstance(module, SharedCumulativeStatistics2D) for module in model.modules()) == 4
    assert not any(isinstance(module, torch.nn.BatchNorm2d) for module in model.separator.modules())
    assert sum(tuple(tensor.shape) == (1, 128, 1, 36) for tensor in state) == 6


def test_cln_lite_rejects_invalid_stream_contract() -> None:
    model = _small_core().eval()
    state = model.init_stream_state()
    x = torch.randn(1, 2, 1, 65)
    for invalid in (state[:-1], (*state, state[0])):
        try:
            model.forward_stream(x, invalid)
        except RuntimeError as error:
            assert "Expected 15 separator states" in str(error)
        else:
            raise AssertionError("Invalid cLN-lite state count was accepted")

    try:
        model.forward_stream(torch.randn(1, 2, 2, 65), state)
    except RuntimeError as error:
        assert "expects one frame" in str(error)
    else:
        raise AssertionError("Multi-frame cLN-lite streaming input was accepted")


def test_cln_lite_streaming_onnx_reduces_normalization_nodes() -> None:
    import onnx

    torch.set_num_threads(1)
    model = _small_core().eval()
    model.masking = False
    state = model.init_stream_state(dtype=torch.float32)
    wrapper = StreamingStateIOWrapper(model, batch_size=1, dtype=torch.float32).eval()
    x = torch.randn(1, 2, 1, 65)
    flat_state, _ = flatten_tensor_tree(state)

    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "macaron_cln_lite.onnx"
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
    assert counts["AveragePool"] == 16
    assert counts["ReduceMean"] == 0
    assert counts["ReduceSum"] == 0
    assert counts["Slice"] == 0
    assert counts["Split"] == 0
    assert counts["Transpose"] == 2
    assert counts["Reshape"] == 6
    assert len(graph.graph.input) == 16
    assert len(graph.graph.output) == 16
