"""Smoke tests for Band-SCNet-NPU.

Covers:
- block-level streaming consistency
- full-model streaming-vs-full-sequence parity
- Conv2d / ConvTranspose2d kernel + stride NPU constraints
- streaming-state byte budget (192 KiB fp16)
- ONNX export + checker + op allowlist

Run with ``python -m BandSCNetNPU.test_band_scnet_npu`` (or from pytest).
"""
from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from BandSCNetNPU import (  # noqa: E402
    BandSCNetNPU,
    BandSCNetNPUStreamingExportWrapper,
    CrossBandBlock,
    NarrowBandBlock,
    SparseDownsampleEncoder,
    SparseUpsampleDecoder,
    build_band_scnet_npu_preset,
    split_bands,
)
from spectral_feature_compression.utils.onnx_streaming import flatten_tensor_tree  # noqa: E402


EDGE_PRESET = "edge_small"
RT_PRESET = "rt192k"
N_FREQ_SMALL = 257  # for fast smoke tests


# --- helpers ---------------------------------------------------------------


def _walk_convs(model: nn.Module) -> Iterable[nn.Conv2d | nn.ConvTranspose2d]:
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            yield m


def _assert_conv_constraints(model: nn.Module) -> None:
    for conv in _walk_convs(model):
        kt, kf = (conv.kernel_size if isinstance(conv.kernel_size, tuple) else (conv.kernel_size, conv.kernel_size))
        dt, df = (conv.dilation if isinstance(conv.dilation, tuple) else (conv.dilation, conv.dilation))
        if (kt - 1) * dt > 14:
            raise AssertionError(f"time (k-1)*d = {(kt - 1) * dt} > 14 in {type(conv).__name__}")
        if (kf - 1) * df > 14:
            raise AssertionError(f"freq (k-1)*d = {(kf - 1) * df} > 14 in {type(conv).__name__}")
        if isinstance(conv, nn.ConvTranspose2d):
            st, sf = (conv.stride if isinstance(conv.stride, tuple) else (conv.stride, conv.stride))
            if st not in (1, 2):
                raise AssertionError(f"ConvTranspose2d stride_t must be 1 or 2, got {st}")
            if sf != 2:
                raise AssertionError(f"ConvTranspose2d stride_f must be 2, got {sf}")


def _streaming_equivalence(
    model: BandSCNetNPU,
    x: torch.Tensor,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> float:
    with torch.no_grad():
        full = model(x)
        state = model.init_stream_state(batch_size=x.shape[0], dtype=x.dtype)
        chunks = []
        for t in range(x.shape[2]):
            y, state = model.forward_stream(x[:, :, t : t + 1, :], state)
            chunks.append(y)
        streamed = torch.cat(chunks, dim=2)
    diff = (full - streamed).abs().max().item()
    if not torch.allclose(full, streamed, atol=atol, rtol=rtol):
        raise AssertionError(
            f"streaming mismatch: max|diff|={diff}, atol={atol}, rtol={rtol}"
        )
    return diff


# --- block-level tests ------------------------------------------------------


def test_cross_band_block_shape() -> None:
    blk = CrossBandBlock(channels=8, freq_kernel=3).eval()
    x = torch.randn(2, 8, 5, 17)
    with torch.no_grad():
        y = blk(x)
    assert y.shape == x.shape


def test_narrow_band_block_streaming_consistency_no_attn() -> None:
    torch.manual_seed(0)
    blk = NarrowBandBlock(channels=8, time_kernel=5, use_attn=False).eval()
    x = torch.randn(1, 8, 7, 13)
    with torch.no_grad():
        full = blk(x)
        state = blk.init_stream_state(1, freq_bins=13, dtype=x.dtype)
        chunks = []
        for t in range(x.shape[2]):
            y, state = blk.forward_stream(x[:, :, t : t + 1, :], state)
            chunks.append(y)
        streamed = torch.cat(chunks, dim=2)
    assert torch.allclose(full, streamed, atol=1e-5, rtol=1e-5), (
        f"no-attn streaming mismatch: {(full - streamed).abs().max()}"
    )


def test_narrow_band_block_streaming_consistency_with_attn() -> None:
    torch.manual_seed(0)
    blk = NarrowBandBlock(
        channels=8,
        time_kernel=5,
        use_attn=True,
        attn_window=4,
        num_heads=2,
        head_dim=4,
    ).eval()
    x = torch.randn(1, 8, 4, 13)  # T=W=4 so the causal mask == bounded mask
    with torch.no_grad():
        full = blk(x)
        state = blk.init_stream_state(1, freq_bins=13, dtype=x.dtype)
        chunks = []
        for t in range(x.shape[2]):
            y, state = blk.forward_stream(x[:, :, t : t + 1, :], state)
            chunks.append(y)
        streamed = torch.cat(chunks, dim=2)
    assert torch.allclose(full, streamed, atol=1e-5, rtol=1e-5), (
        f"with-attn streaming mismatch: {(full - streamed).abs().max()}"
    )


def test_sparse_pyramid_round_trip_shape() -> None:
    from BandSCNetNPU import pad_n_freq_for_split

    n_freq = pad_n_freq_for_split(N_FREQ_SMALL)
    enc = SparseDownsampleEncoder(
        n_freq=n_freq,
        in_channels=2,
        channels=8,
        time_kernel=5,
        freq_kernel=3,
    ).eval()
    dec = SparseUpsampleDecoder(
        n_freq=n_freq,
        channels=8,
        time_kernel=5,
        freq_kernel=3,
    ).eval()
    x = torch.randn(1, 2, 3, n_freq)
    with torch.no_grad():
        low, mid, high = enc(x)
        assert low.shape[-1] == enc.bands.low
        assert mid.shape[-1] == enc.bands.mid // 4
        assert high.shape[-1] == enc.bands.high // 16
        y = dec(low, mid, high, low, mid, high)
    assert y.shape[-1] == n_freq


# --- model-level tests ------------------------------------------------------


def test_band_split_sums_to_n_freq() -> None:
    from BandSCNetNPU import pad_n_freq_for_split

    for n_freq in (128, 257, 513, 1025, 2049):
        padded = pad_n_freq_for_split(n_freq)
        bands = split_bands(padded)
        assert bands.low + bands.mid + bands.high == padded, (n_freq, padded, bands)
        assert bands.high % 16 == 0, (n_freq, bands)
        assert bands.mid % 4 == 0, (n_freq, bands)
        assert bands.low % 4 == 0, (n_freq, bands)
        assert padded >= n_freq and padded - n_freq < 32, (n_freq, padded)


def test_edge_small_forward_shape() -> None:
    model = build_band_scnet_npu_preset(EDGE_PRESET, n_freq=N_FREQ_SMALL).eval()
    x = torch.randn(1, 2, 5, N_FREQ_SMALL)
    with torch.no_grad():
        y = model(x)
    # masking=True -> [B, 2*n_src*n_chan, T, F]
    assert y.shape == (1, 2 * model.n_src * model.n_chan, 5, N_FREQ_SMALL)


def test_edge_small_streaming_matches_full() -> None:
    torch.manual_seed(0)
    model = build_band_scnet_npu_preset(EDGE_PRESET, n_freq=N_FREQ_SMALL).eval()
    x = torch.randn(1, 2, 6, N_FREQ_SMALL)
    _streaming_equivalence(model, x)


def test_rt192k_streaming_matches_full() -> None:
    torch.manual_seed(0)
    model = build_band_scnet_npu_preset(RT_PRESET, n_freq=N_FREQ_SMALL).eval()
    # attention is causal-window; choose T <= W so causal==windowed for parity
    T = min(model.attn_window, 6)
    x = torch.randn(1, 2, T, N_FREQ_SMALL)
    _streaming_equivalence(model, x, atol=2e-5, rtol=2e-5)


def test_npu_conv_constraints_both_presets() -> None:
    for name in (EDGE_PRESET, RT_PRESET):
        model = build_band_scnet_npu_preset(name, n_freq=N_FREQ_SMALL).eval()
        _assert_conv_constraints(model)


def test_state_budget_edge_small() -> None:
    model = build_band_scnet_npu_preset(EDGE_PRESET, n_freq=2049).eval()
    bytes_fp16 = model.state_size_bytes(dtype=torch.float16)
    # edge_small should easily fit 192 KiB
    assert bytes_fp16 <= 196_608, f"edge_small state {bytes_fp16} B exceeds 192 KiB"


def test_state_budget_rt192k() -> None:
    model = build_band_scnet_npu_preset(RT_PRESET, n_freq=2049).eval()
    bytes_fp16 = model.state_size_bytes(dtype=torch.float16)
    assert bytes_fp16 <= 196_608, f"rt192k state {bytes_fp16} B exceeds 192 KiB"


# --- ONNX export smoke ------------------------------------------------------


FORBIDDEN_EXPORT_OPS = {
    "Tile",
    "Expand",
    "ConstantOfShape",
    "ScatterND",
    "If",
    "Loop",
    "Scan",
}


def _collect_ops(model_proto) -> set[str]:
    return {node.op_type for node in model_proto.graph.node}


def test_streaming_onnx_export_edge_small() -> None:
    import onnx

    model = build_band_scnet_npu_preset(EDGE_PRESET, n_freq=N_FREQ_SMALL).eval()
    wrapper = BandSCNetNPUStreamingExportWrapper(model, batch_size=1, dtype=torch.float32).eval()
    x = torch.randn(1, 2, 1, model.n_freq)
    flat_state, _ = flatten_tensor_tree(tuple(model.init_stream_state(batch_size=1, dtype=torch.float32)))

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "band_scnet_npu_edge_small.onnx"
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
        model_proto = onnx.load(str(out))
        onnx.checker.check_model(model_proto)
        bad = FORBIDDEN_EXPORT_OPS & _collect_ops(model_proto)
        assert not bad, f"forbidden ops in exported graph: {sorted(bad)}"


# --- runner ----------------------------------------------------------------


def _collect_tests() -> list:
    return [
        (name, obj)
        for name, obj in globals().items()
        if name.startswith("test_") and callable(obj)
    ]


def main() -> int:
    failures: list[tuple[str, BaseException]] = []
    for name, fn in _collect_tests():
        try:
            fn()
            print(f"[pass] {name}")
        except BaseException as exc:  # pylint: disable=broad-except
            failures.append((name, exc))
            print(f"[FAIL] {name}: {exc!r}")
    if failures:
        print(f"\n{len(failures)} failure(s)")
        return 1
    print(f"\nall {len(_collect_tests())} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
