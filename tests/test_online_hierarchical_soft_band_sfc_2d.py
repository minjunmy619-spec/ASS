from __future__ import annotations

from pathlib import Path
import tempfile

import torch

from spectral_feature_compression.core.model.online_hierarchical_soft_band_sfc_2d import (
    OnlineHierarchicalSoftBandSFC2D,
)

try:
    import pytest
except ModuleNotFoundError:  # pragma: no cover - fallback for lightweight environments
    pytest = None


def _build_model(prior_mode: str = "inherited", d_model: int = 20) -> OnlineHierarchicalSoftBandSFC2D:
    return OnlineHierarchicalSoftBandSFC2D(
        n_freq=1025,
        pre_bands=128,
        mid_bands=96,
        bottleneck_bands=48,
        n_src=4,
        n_chan=2,
        d_model=d_model,
        pre_layers=1,
        mid_layers=1,
        bottleneck_layers=3,
        kernel_size=(3, 3),
        causal=True,
        masking=True,
        dilation_cycle=(1, 2, 4, 6),
        hierarchical_prior_mode=prior_mode,
    ).eval()


if pytest is not None:
    @pytest.mark.parametrize("prior_mode", ["inherited", "uniform"])
    def test_forward_stream_matches_forward(prior_mode: str) -> None:
        torch.manual_seed(0)
        model = _build_model(prior_mode=prior_mode)
        x = torch.randn(1, 4, 4, 1025)

        with torch.no_grad():
            y_full = model(x)
            state = model.init_stream_state(batch_size=1, device=x.device, dtype=x.dtype)
            ys = []
            for t in range(x.shape[2]):
                y_t, state = model.forward_stream(x[:, :, t : t + 1, :], state)
                ys.append(y_t)
            y_stream = torch.cat(ys, dim=2)

        assert y_full.shape == y_stream.shape == (1, 16, 4, 1025)
        assert torch.allclose(y_full, y_stream, atol=1e-4, rtol=1e-4)
else:
    def test_forward_stream_matches_forward() -> None:
        torch.manual_seed(0)
        model = _build_model(prior_mode="inherited")
        x = torch.randn(1, 4, 4, 1025)

        with torch.no_grad():
            y_full = model(x)
            state = model.init_stream_state(batch_size=1, device=x.device, dtype=x.dtype)
            ys = []
            for t in range(x.shape[2]):
                y_t, state = model.forward_stream(x[:, :, t : t + 1, :], state)
                ys.append(y_t)
            y_stream = torch.cat(ys, dim=2)

        assert y_full.shape == y_stream.shape == (1, 16, 4, 1025)
        assert torch.allclose(y_full, y_stream, atol=1e-4, rtol=1e-4)


def test_rt192k_layer_cache_budget() -> None:
    model = _build_model(prior_mode="inherited", d_model=20)
    layer_cache_kib = model.state_size_bytes(dtype=torch.float16, mode="layer_cache") / 1024.0
    assert model.stream_context_frames() == 24
    assert layer_cache_kib <= 192.0


def test_onnx_export_opset11_smoke() -> None:
    if pytest is not None:
        onnx = pytest.importorskip("onnx")
    else:
        import onnx
    model = _build_model(prior_mode="inherited", d_model=20)
    x = torch.randn(1, 4, 4, 1025)
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "hierarchical.onnx"
        with torch.no_grad():
            torch.onnx.export(
                model,
                x,
                str(out),
                opset_version=11,
                input_names=["x"],
                output_names=["y"],
                do_constant_folding=True,
            )
        onnx.checker.check_model(onnx.load(str(out)))
