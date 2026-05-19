from __future__ import annotations

import torch

from EdgeFusionNPU import build_edge_fusion_npu_preset


def test_edge_fusion_npu_tiny_smoke() -> None:
    model = build_edge_fusion_npu_preset("tiny").eval()
    x = torch.randn(1, 2, 1025, 1)
    state = model.init_states(batch_size=1)
    with torch.no_grad():
        mask, next_state = model(x, state)
    assert mask.shape == (1, 3, 1025, 1)
    assert next_state.shape == state.shape
    assert state.dim() == 4
    assert float(mask.min()) >= 0.0
    assert float(mask.max()) <= float(model.mask_scale)


def test_edge_fusion_npu_v2_variants_keep_packed_state_contract() -> None:
    cases = [
        ("compact-v2-ssmlite", 513),
        ("compact-v2-bandtoken", 513),
        ("compact-v2-hybrid", 513),
        ("balanced-v2-hybrid", 257),
    ]
    for preset, n_freq in cases:
        model = build_edge_fusion_npu_preset(preset, n_freq=n_freq).eval()
        x = torch.randn(1, 2, n_freq, 1)
        state = model.init_states(batch_size=1)
        with torch.no_grad():
            mask, next_state = model(x, state)
        assert mask.shape == (1, 3, n_freq, 1)
        assert next_state.shape == state.shape
        assert state.dim() == 4
        assert float(mask.min()) >= 0.0
        assert float(mask.max()) <= float(model.mask_scale)


def test_ssm_lite_state_update_is_convex() -> None:
    model = build_edge_fusion_npu_preset("compact-v2-ssmlite").eval()
    block = model.blocks[0]
    assert block.memory_mode == "ssm_lite"

    x = torch.ones(1, model.hidden_channels, model.n_freq, 1)
    state = torch.full((1, model.hidden_channels, model.n_freq, model.context_size), 2.0)
    with torch.no_grad():
        _, next_state = block(x, state)
    latest = next_state[:, :, :, -1:]
    assert float(latest.min()) >= 1.0
    assert float(latest.max()) <= 2.0


def test_export_defaults_use_preset_frequency() -> None:
    model = build_edge_fusion_npu_preset("compact")
    assert model.n_freq == 513
    model = build_edge_fusion_npu_preset("balanced-v2-hybrid")
    assert model.n_freq == 257


def test_rejects_bad_state_shape() -> None:
    model = build_edge_fusion_npu_preset("tiny").eval()
    x = torch.randn(1, 2, 1025, 1)
    bad_state = torch.zeros(1, model.state_channels, 1024, model.context_size)
    try:
        model(x, bad_state)
    except ValueError as exc:
        assert "state frequency bins" in str(exc)
    else:
        raise AssertionError("bad state frequency should fail")
