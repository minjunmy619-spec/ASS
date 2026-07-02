from __future__ import annotations

from pathlib import Path

import torch

import onnx

from spectral_feature_compression.core.model.conv2d_rnn_compat import Conv2dGRU, Conv2dLSTM


def test_conv2d_gru_matches_torch_gru_batch_first() -> None:
    torch.manual_seed(0)
    native = torch.nn.GRU(
        input_size=5,
        hidden_size=7,
        num_layers=2,
        bias=True,
        batch_first=True,
        dropout=0.0,
    ).eval()
    conv = Conv2dGRU.from_torch(native).eval()
    x = torch.randn(3, 4, 5)
    h0 = torch.randn(2, 3, 7)

    with torch.no_grad():
        y_native, h_native = native(x, h0)
        y_conv, h_conv = conv(x, h0)

    torch.testing.assert_close(y_conv, y_native, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(h_conv, h_native, rtol=1e-6, atol=1e-6)


def test_conv2d_gru_matches_torch_gru_unbatched() -> None:
    torch.manual_seed(1)
    native = torch.nn.GRU(input_size=4, hidden_size=6, num_layers=1, bias=False).eval()
    conv = Conv2dGRU.from_torch(native).eval()
    x = torch.randn(5, 4)
    h0 = torch.randn(1, 6)

    with torch.no_grad():
        y_native, h_native = native(x, h0)
        y_conv, h_conv = conv(x, h0)

    assert tuple(y_conv.shape) == (5, 6)
    assert tuple(h_conv.shape) == (1, 6)
    torch.testing.assert_close(y_conv, y_native, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(h_conv, h_native, rtol=1e-6, atol=1e-6)


def test_conv2d_lstm_matches_torch_lstm_batch_first() -> None:
    torch.manual_seed(2)
    native = torch.nn.LSTM(
        input_size=5,
        hidden_size=7,
        num_layers=2,
        bias=True,
        batch_first=True,
        dropout=0.0,
    ).eval()
    conv = Conv2dLSTM.from_torch(native).eval()
    x = torch.randn(3, 4, 5)
    h0 = torch.randn(2, 3, 7)
    c0 = torch.randn(2, 3, 7)

    with torch.no_grad():
        y_native, (h_native, c_native) = native(x, (h0, c0))
        y_conv, (h_conv, c_conv) = conv(x, (h0, c0))

    torch.testing.assert_close(y_conv, y_native, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(h_conv, h_native, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(c_conv, c_native, rtol=1e-6, atol=1e-6)


def test_conv2d_lstm_matches_torch_lstm_unbatched() -> None:
    torch.manual_seed(3)
    native = torch.nn.LSTM(input_size=4, hidden_size=6, num_layers=1, bias=False).eval()
    conv = Conv2dLSTM.from_torch(native).eval()
    x = torch.randn(5, 4)
    h0 = torch.randn(1, 6)
    c0 = torch.randn(1, 6)

    with torch.no_grad():
        y_native, (h_native, c_native) = native(x, (h0, c0))
        y_conv, (h_conv, c_conv) = conv(x, (h0, c0))

    assert tuple(y_conv.shape) == (5, 6)
    assert tuple(h_conv.shape) == (1, 6)
    assert tuple(c_conv.shape) == (1, 6)
    torch.testing.assert_close(y_conv, y_native, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(h_conv, h_native, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(c_conv, c_native, rtol=1e-6, atol=1e-6)


def test_conv2d_gru_onnx_export_uses_basic_ops(tmp_path: Path) -> None:
    torch.manual_seed(4)
    model = Conv2dGRU(input_size=3, hidden_size=4, num_layers=1, batch_first=True).eval()
    x = torch.randn(2, 3, 3)
    h0 = torch.randn(1, 2, 4)
    out = tmp_path / "conv2d_gru.onnx"

    torch.onnx.export(
        model,
        (x, h0),
        out,
        opset_version=14,
        input_names=["x", "h0"],
        output_names=["y", "hn"],
        dynamo=False,
    )
    graph = onnx.load(out)
    ops = {node.op_type for node in graph.graph.node}
    assert "GRU" not in ops
    assert "LSTM" not in ops
    assert "Conv" in ops


def test_conv2d_lstm_onnx_export_uses_basic_ops(tmp_path: Path) -> None:
    torch.manual_seed(5)
    model = Conv2dLSTM(input_size=3, hidden_size=4, num_layers=1, batch_first=True).eval()
    x = torch.randn(2, 3, 3)
    h0 = torch.randn(1, 2, 4)
    c0 = torch.randn(1, 2, 4)
    out = tmp_path / "conv2d_lstm.onnx"

    torch.onnx.export(
        model,
        (x, (h0, c0)),
        out,
        opset_version=14,
        input_names=["x", "h0", "c0"],
        output_names=["y", "hn", "cn"],
        dynamo=False,
    )
    graph = onnx.load(out)
    ops = {node.op_type for node in graph.graph.node}
    assert "GRU" not in ops
    assert "LSTM" not in ops
    assert "Conv" in ops
