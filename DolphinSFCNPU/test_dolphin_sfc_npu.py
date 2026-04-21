from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from DolphinSFCNPU import DolphinSFCNPUSeparator, DolphinSFCNPUStreamingExportWrapper, build_dolphin_sfc_npu_preset
from spectral_feature_compression.utils.onnx_streaming import flatten_tensor_tree


FORBIDDEN_EXPORT_OPS = {
    "ConstantOfShape",
    "Expand",
    "Tile",
}


def build_model() -> DolphinSFCNPUSeparator:
    return build_dolphin_sfc_npu_preset(
        "edge_small",
        n_freq=257,
        n_fft=512,
        sample_rate=16000,
    )


def export_streaming_onnx(model: DolphinSFCNPUSeparator, out_path: Path):
    import onnx

    wrapper = DolphinSFCNPUStreamingExportWrapper(model, batch_size=1, dtype=torch.float32).eval()
    x = torch.randn(1, 2, 1, model.n_freq)
    state = model.init_stream_state(batch_size=1, dtype=torch.float32)
    flat_state, _ = flatten_tensor_tree(state)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (x, *flat_state),
            str(out_path),
            opset_version=11,
            input_names=["x", *[f"state_{idx}" for idx in range(len(flat_state))]],
            output_names=["y", *[f"next_state_{idx}" for idx in range(len(flat_state))]],
            do_constant_folding=True,
            dynamo=False,
        )

    model_proto = onnx.load(str(out_path))
    onnx.checker.check_model(model_proto)
    return model_proto, flat_state


def collect_ops(model_proto) -> set[str]:
    return {node.op_type for node in model_proto.graph.node}


def test_forward_stream_matches_forward() -> None:
    torch.manual_seed(0)
    model = build_model().eval()
    x = torch.randn(1, 2, 5, model.n_freq)
    with torch.no_grad():
        full = model(x)
        state = model.init_stream_state(batch_size=1, dtype=x.dtype)
        chunks = []
        for t in range(x.shape[2]):
            y, state = model.forward_stream(x[:, :, t : t + 1, :], state)
            chunks.append(y)
        streamed = torch.cat(chunks, dim=2)

    assert full.shape == streamed.shape == (1, 6, 5, model.n_freq)
    assert torch.allclose(full, streamed, atol=1e-5, rtol=1e-5)


def test_large_presets_forward_stream_match() -> None:
    torch.manual_seed(0)
    for preset in ("large_6m", "large_8m"):
        model = build_dolphin_sfc_npu_preset(preset, n_freq=257, n_fft=512, sample_rate=16000).eval()
        x = torch.randn(1, 2, 3, model.n_freq)
        with torch.no_grad():
            full = model(x)
            state = model.init_stream_state(batch_size=1, dtype=x.dtype)
            chunks = []
            for t in range(x.shape[2]):
                y, state = model.forward_stream(x[:, :, t : t + 1, :], state)
                chunks.append(y)
            streamed = torch.cat(chunks, dim=2)
        assert torch.allclose(full, streamed, atol=3e-5, rtol=3e-5), preset


def test_streaming_onnx_export_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model_proto, _ = export_streaming_onnx(build_model().eval(), Path(tmpdir) / "edge_small.onnx")
    assert "Tile" not in collect_ops(model_proto)


def test_large_presets_onnx_op_audit() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        for preset in ("large_6m", "large_8m"):
            model = build_dolphin_sfc_npu_preset(preset, n_freq=257, n_fft=512, sample_rate=16000).eval()
            model_proto, flat_state = export_streaming_onnx(model, Path(tmpdir) / f"{preset}.onnx")
            ops = collect_ops(model_proto)
            forbidden = sorted(ops & FORBIDDEN_EXPORT_OPS)
            assert not forbidden, f"{preset} exported forbidden ops: {forbidden}"
            assert "MatMul" in ops, "The adaptive bmm band routing should still export as MatMul."
            assert len(flat_state) == 22


if __name__ == "__main__":
    test_forward_stream_matches_forward()
    test_large_presets_forward_stream_match()
    try:
        test_streaming_onnx_export_smoke()
        test_large_presets_onnx_op_audit()
    except ModuleNotFoundError as exc:
        print(f"[skip] ONNX export smoke skipped: {exc}")

    for preset in ("edge_small", "large_6m", "large_8m"):
        model = build_dolphin_sfc_npu_preset(preset, n_freq=1025, n_fft=2048, sample_rate=44100)
        print(f"[ok] {preset} params={sum(p.numel() for p in model.parameters())}")
        print(f"[ok] {preset} fp16_state_bytes={model.state_size_bytes(dtype=torch.float16)}")
