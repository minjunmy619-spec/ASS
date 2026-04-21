from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from DolphinSFC import DolphinSFCSeparator, DolphinSFCStreamingExportWrapper, build_dolphin_sfc_preset
from spectral_feature_compression.utils.onnx_streaming import flatten_tensor_tree


def build_model() -> DolphinSFCSeparator:
    return DolphinSFCSeparator(
        n_freq=257,
        n_fft=512,
        sample_rate=16000,
        n_bands=32,
        n_src=3,
        d_model=16,
        num_scales=3,
        masking=True,
    )


def test_forward_stream_matches_forward() -> None:
    torch.manual_seed(0)
    model = build_model().eval()
    x = torch.randn(1, 2, 5, 257)
    with torch.no_grad():
        full = model(x)
        state = model.init_stream_state(batch_size=1, dtype=x.dtype)
        chunks = []
        for t in range(x.shape[2]):
            y, state = model.forward_stream(x[:, :, t : t + 1, :], state)
            chunks.append(y)
        streamed = torch.cat(chunks, dim=2)

    assert full.shape == streamed.shape == (1, 6, 5, 257)
    assert torch.allclose(full, streamed, atol=1e-5, rtol=1e-5)


def test_streaming_onnx_export_smoke() -> None:
    import onnx

    model = build_model().eval()
    wrapper = DolphinSFCStreamingExportWrapper(model, batch_size=1, dtype=torch.float32)
    x = torch.randn(1, 2, 1, 257)
    state = model.init_stream_state(batch_size=1, dtype=torch.float32)
    flat_state, _ = flatten_tensor_tree(state)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "dolphin_sfc_stream.onnx"
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
        onnx.checker.check_model(onnx.load(str(out_path)))


if __name__ == "__main__":
    test_forward_stream_matches_forward()
    try:
        test_streaming_onnx_export_smoke()
    except ModuleNotFoundError as exc:
        print(f"[skip] ONNX export smoke skipped: {exc}")

    model = build_model()
    large = build_dolphin_sfc_preset("large_8m", n_freq=1025, n_fft=2048, sample_rate=44100)
    print(f"[ok] params={sum(p.numel() for p in model.parameters())}")
    print(f"[ok] fp16 state bytes={model.state_size_bytes(dtype=torch.float16)}")
    print(f"[ok] large_8m params={sum(p.numel() for p in large.parameters())}")
    print(f"[ok] large_8m fp16 state bytes={large.state_size_bytes(dtype=torch.float16)}")
