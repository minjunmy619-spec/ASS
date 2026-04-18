from __future__ import annotations

from pathlib import Path
import tempfile

import torch

from spectral_feature_compression.core.model.online_hard_band_sfc_2d import OnlineHardBandSFC2D
from spectral_feature_compression.core.model.online_crossattn_query_sfc_2d import OnlineCrossAttnQuerySFC2D
from spectral_feature_compression.core.model.online_sfc_2d import OnlineSFC2D
from spectral_feature_compression.core.model.online_soft_band_dilated_sfc_2d import OnlineSoftBandDilatedSFC2D
from spectral_feature_compression.core.model.online_soft_band_gru_sfc_2d import OnlineSoftBandGRUSFC2D
from spectral_feature_compression.core.model.online_soft_band_query_sfc_2d import OnlineSoftBandQuerySFC2D
from spectral_feature_compression.core.model.online_soft_band_sfc_2d import OnlineSoftBandSFC2D
from spectral_feature_compression.utils.onnx_streaming import (
    ExternalizedConstantsWrapper,
    StreamingStateIOWrapper,
    collect_external_constant_bindings,
    flatten_tensor_tree,
    get_external_constant_tensors,
)

try:
    import pytest
except ModuleNotFoundError:  # pragma: no cover - fallback for lightweight environments
    pytest = None


def _build_models():
    common = dict(
        n_freq=257,
        n_fft=512,
        sample_rate=16000,
        n_bands=32,
        n_src=2,
        n_chan=1,
        d_model=8,
        causal=True,
        masking=True,
    )
    return {
        "plain": OnlineSFC2D(**common, n_layers=2),
        "soft": OnlineSoftBandSFC2D(**common, n_layers=2),
        "soft_query": OnlineSoftBandQuerySFC2D(**common, n_layers=2),
        "crossattn_query": OnlineCrossAttnQuerySFC2D(**common, n_layers=2),
        "hard": OnlineHardBandSFC2D(**common, n_layers=2),
        "soft_dilated": OnlineSoftBandDilatedSFC2D(**common, n_layers=2, dilation_cycle=(1, 2)),
        "soft_gru": OnlineSoftBandGRUSFC2D(**common, n_layers=2),
    }


if pytest is not None:
    @pytest.mark.parametrize("name", ["plain", "soft", "soft_query", "crossattn_query", "hard", "soft_dilated", "soft_gru"])
    def test_forward_stream_matches_forward(name: str) -> None:
        torch.manual_seed(0)
        model = _build_models()[name].eval()
        x = torch.randn(1, 2, 5, 257)

        with torch.no_grad():
            y_full = model(x)
            state = model.init_stream_state(batch_size=1, device=x.device, dtype=x.dtype)
            ys = []
            for t in range(x.shape[2]):
                y_t, state = model.forward_stream(x[:, :, t : t + 1, :], state)
                ys.append(y_t)
            y_stream = torch.cat(ys, dim=2)

        assert y_full.shape == y_stream.shape == (1, 4, 5, 257)
        assert torch.allclose(y_full, y_stream, atol=1e-5, rtol=1e-5)


def test_streaming_export_wrapper_flattens_state_tree() -> None:
    model = _build_models()["plain"].eval()
    wrapper = StreamingStateIOWrapper(model, batch_size=1, dtype=torch.float32)
    state = model.init_stream_state(batch_size=1, dtype=torch.float32)
    flat_state, _ = flatten_tensor_tree(state)
    x = torch.randn(1, 2, 1, 257)

    with torch.no_grad():
        outputs = wrapper(x, *flat_state)

    assert len(outputs) == 1 + len(flat_state)
    assert outputs[0].shape == (1, 4, 1, 257)
    assert all(isinstance(t, torch.Tensor) for t in outputs[1:])


def test_externalized_constants_wrapper_matches_core_forward() -> None:
    model = _build_models()["soft"].eval()
    wrapper = ExternalizedConstantsWrapper(model)
    const_tensors = get_external_constant_tensors(model, wrapper.constant_bindings)
    x = torch.randn(1, 2, 5, 257)

    with torch.no_grad():
        y_core = model(x)
        y_wrapper = wrapper(x, *const_tensors)

    assert collect_external_constant_bindings(model)
    assert torch.allclose(y_core, y_wrapper, atol=1e-6, rtol=1e-6)


def test_streaming_wrapper_with_externalized_constants_matches_core() -> None:
    model = _build_models()["soft"].eval()
    wrapper = StreamingStateIOWrapper(model, batch_size=1, dtype=torch.float32, externalize_constants=True)
    state = model.init_stream_state(batch_size=1, dtype=torch.float32)
    flat_state, _ = flatten_tensor_tree(state)
    const_tensors = get_external_constant_tensors(model, wrapper.constant_bindings)
    x = torch.randn(1, 2, 1, 257)

    with torch.no_grad():
        y_core, new_state = model.forward_stream(x, state)
        flat_new_state, _ = flatten_tensor_tree(new_state)
        outputs = wrapper(x, *flat_state, *const_tensors)

    assert torch.allclose(y_core, outputs[0], atol=1e-6, rtol=1e-6)
    assert len(outputs) == 1 + len(flat_new_state)
    for expected, actual in zip(flat_new_state, outputs[1:]):
        assert torch.allclose(expected, actual, atol=1e-6, rtol=1e-6)


def test_streaming_wrapper_onnx_export_smoke() -> None:
    if pytest is not None:
        onnx = pytest.importorskip("onnx")
    else:
        import onnx

    model = _build_models()["plain"].eval()
    wrapper = StreamingStateIOWrapper(model, batch_size=1, dtype=torch.float32)
    x = torch.randn(1, 2, 1, 257)
    state = model.init_stream_state(batch_size=1, dtype=torch.float32)
    flat_state, _ = flatten_tensor_tree(state)

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "plain_stream.onnx"
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                (x, *flat_state),
                str(out),
                opset_version=11,
                input_names=["x", *[f"state_{i}" for i in range(len(flat_state))]],
                output_names=["y", *[f"next_state_{i}" for i in range(len(flat_state))]],
                do_constant_folding=True,
            )
        onnx.checker.check_model(onnx.load(str(out)))


def test_externalized_constants_wrapper_onnx_export_smoke() -> None:
    if pytest is not None:
        onnx = pytest.importorskip("onnx")
    else:
        import onnx

    model = _build_models()["soft"].eval()
    wrapper = ExternalizedConstantsWrapper(model)
    x = torch.randn(1, 2, 5, 257)
    const_tensors = get_external_constant_tensors(model, wrapper.constant_bindings)

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "soft_externalized.onnx"
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                (x, *const_tensors),
                str(out),
                opset_version=11,
                input_names=["x", *[f"const_{i}" for i in range(len(const_tensors))]],
                output_names=["y"],
                do_constant_folding=True,
            )
        onnx.checker.check_model(onnx.load(str(out)))
