from __future__ import annotations

import torch

from .presets import build_band_sfc_net_npu_preset


def _assert_close(name: str, a: torch.Tensor, b: torch.Tensor, atol: float = 1e-5) -> None:
    diff = (a - b).abs().max().item()
    if diff > atol:
        raise AssertionError(f"{name} max diff {diff:.6g} > {atol}")


def test_safe_forward_shape() -> None:
    model = build_band_sfc_net_npu_preset("safe", n_freq=129, n_src=3, n_chan=1).eval()
    x = torch.randn(2, 2, 5, 129)
    y = model(x)
    assert tuple(y.shape) == (2, 6, 5, 129)


def test_safe_streaming_matches_full() -> None:
    torch.manual_seed(0)
    model = build_band_sfc_net_npu_preset("safe", n_freq=65, n_src=3, n_chan=1).eval()
    x = torch.randn(1, 2, 6, 65)
    with torch.no_grad():
        y_full = model(x)
        state = model.init_stream_state(batch_size=1, dtype=x.dtype)
        frames = []
        for t in range(x.shape[2]):
            y_t, state = model.forward_stream(x[:, :, t : t + 1, :], state)
            frames.append(y_t)
        y_stream = torch.cat(frames, dim=2)
    _assert_close("safe streaming parity", y_full, y_stream, atol=2e-5)


def test_quality_forward_shape() -> None:
    model = build_band_sfc_net_npu_preset("quality", n_freq=65, n_src=3, n_chan=1).eval()
    x = torch.randn(1, 2, 3, 65)
    with torch.no_grad():
        y = model(x)
    assert tuple(y.shape) == (1, 6, 3, 65)


def test_state_budget_safe_fp512() -> None:
    model = build_band_sfc_net_npu_preset("safe", n_freq=512, n_src=3, n_chan=1).eval()
    state_kib = model.state_size_bytes(dtype=torch.float16) / 1024.0
    assert state_kib < 192.0, f"safe fp512 state too large: {state_kib:.2f} KiB"


def main() -> None:
    tests = [
        test_safe_forward_shape,
        test_safe_streaming_matches_full,
        test_quality_forward_shape,
        test_state_budget_safe_fp512,
    ]
    for test in tests:
        test()
        print(f"[pass] {test.__name__}")
    print("all BandSFCNetNPU smoke tests passed")


if __name__ == "__main__":
    main()
