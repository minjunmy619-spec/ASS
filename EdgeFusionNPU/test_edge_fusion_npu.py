from __future__ import annotations

import torch

from EdgeFusionNPU import build_edge_fusion_npu_preset
from EdgeFusionNPU.edge_fusion_npu import count_parameters
from EdgeFusionNPU.training_wrapper import EdgeFusionNPUOnlineModel, build_edge_fusion_npu_system
from spectral_feature_compression.core.model.frequency_preprocessing import build_frequency_preprocessor


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
        ("big-v2-hybrid-2m", 257),
        ("large-v2-hybrid-5m", 257),
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


def test_core_chunk_forward_matches_manual_frames() -> None:
    model = build_edge_fusion_npu_preset("tiny").eval()
    x = torch.randn(1, 2, model.n_freq, 4)
    state = model.init_states(batch_size=1)

    with torch.no_grad():
        chunk_mask, chunk_state = model(x, state)

        manual_state = state
        manual_masks = []
        for frame_idx in range(x.shape[-1]):
            mask, manual_state = model(x[:, :, :, frame_idx : frame_idx + 1], manual_state)
            manual_masks.append(mask)
        manual_mask = torch.cat(manual_masks, dim=-1)

    assert chunk_mask.shape == (1, 3, model.n_freq, 4)
    torch.testing.assert_close(chunk_mask, manual_mask)
    torch.testing.assert_close(chunk_state, manual_state)


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


def test_big_variants_are_in_requested_parameter_range() -> None:
    cases = [
        ("big-v2-hybrid-2m", 2_000_000, 3_000_000),
        ("large-v2-hybrid-5m", 5_000_000, 6_000_000),
    ]
    for preset, lower, upper in cases:
        model = build_edge_fusion_npu_preset(preset)
        params = count_parameters(model)
        assert lower <= params <= upper
        state = model.init_states(batch_size=1)
        assert state.dim() == 4
        assert state.shape[1] == model.state_channels


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


def test_training_wrapper_processes_chunk_and_returns_state() -> None:
    core = build_edge_fusion_npu_preset("tiny").eval()
    model = EdgeFusionNPUOnlineModel(core=core, n_src=3, n_chan=1).eval()
    x = torch.complex(torch.randn(1, 1, core.n_freq, 4), torch.randn(1, 1, core.n_freq, 4))

    with torch.no_grad():
        chunk_out, chunk_state = model(x, return_state=True)

        state = core.init_states(batch_size=1)
        manual_frames = []
        for frame_idx in range(x.shape[-1]):
            frame = x[:, :, :, frame_idx : frame_idx + 1]
            packed = torch.cat([frame.real, frame.imag], dim=1)
            mask, state = core(packed, state)
            mask = mask.reshape(1, 3, 1, core.n_freq, 1)
            manual_frames.append(frame.unsqueeze(1) * mask)
        manual_out = torch.cat(manual_frames, dim=-1)

    assert chunk_out.shape == (1, 3, 1, core.n_freq, 4)
    assert chunk_state.shape == core.init_states(batch_size=1).shape
    torch.testing.assert_close(chunk_out, manual_out)
    torch.testing.assert_close(chunk_state, state)


def test_training_wrapper_frequency_preprocessing_shape() -> None:
    system = build_edge_fusion_npu_system(
        n_fft=1024,
        hop_length=256,
        fs=16000,
        preset="tiny",
        freq_preprocess_enabled=True,
        freq_preprocess_keep_bins=192,
        freq_preprocess_target_bins=257,
    ).eval()
    x = torch.complex(torch.randn(1, 1, 513, 3), torch.randn(1, 1, 513, 3))
    with torch.no_grad():
        y = system.model(x)
    assert y.shape == (1, 3, 1, 513, 3)
    assert system.model.input_n_freq == 513
    assert system.model.core_n_freq == 257


def test_frequency_preprocessor_preserves_bfloat16_dtype() -> None:
    projector = build_frequency_preprocessor(
        513,
        enabled=True,
        keep_bins=192,
        target_bins=257,
    )
    x = torch.randn(1, 2, 3, 513, dtype=torch.bfloat16)
    y = projector.analysis(x)
    z = projector.synthesis(y)
    assert y.dtype == torch.bfloat16
    assert z.dtype == torch.bfloat16
    assert y.shape == (1, 2, 3, 257)
    assert z.shape == x.shape


def test_frequency_preprocessed_training_wrapper_streaming_split_matches_clip() -> None:
    system = build_edge_fusion_npu_system(
        n_fft=1024,
        hop_length=256,
        fs=16000,
        preset="tiny",
        freq_preprocess_enabled=True,
        freq_preprocess_keep_bins=192,
        freq_preprocess_target_bins=257,
    ).eval()
    model = system.model
    x2d = torch.randn(1, 2, 5, 513)

    with torch.no_grad():
        clip_out, clip_state = model.forward_stream(x2d)
        chunk_a, state_a = model.forward_stream(x2d[:, :, :2, :])
        chunk_b, state_b = model.forward_stream(x2d[:, :, 2:, :], state_a)
        split_out = torch.cat([chunk_a, chunk_b], dim=2)

    assert clip_out.shape == (1, 6, 5, 513)
    assert clip_state.shape[2] == 257
    torch.testing.assert_close(split_out, clip_out)
    torch.testing.assert_close(state_b, clip_state)


def test_training_wrapper_split_chunks_match_single_clip() -> None:
    core = build_edge_fusion_npu_preset("tiny").eval()
    model = EdgeFusionNPUOnlineModel(core=core, n_src=3, n_chan=1).eval()
    x = torch.complex(torch.randn(1, 1, core.n_freq, 5), torch.randn(1, 1, core.n_freq, 5))

    with torch.no_grad():
        clip_out, clip_state = model(x, return_state=True)
        chunk_a, state_a = model(x[..., :2], return_state=True)
        chunk_b, state_b = model(x[..., 2:], initial_state=state_a, return_state=True)
        split_out = torch.cat([chunk_a, chunk_b], dim=-1)

    torch.testing.assert_close(split_out, clip_out)
    torch.testing.assert_close(state_b, clip_state)


def test_training_wrapper_chunk_backward() -> None:
    core = build_edge_fusion_npu_preset("tiny")
    model = EdgeFusionNPUOnlineModel(core=core, n_src=3, n_chan=1)
    x = torch.complex(
        torch.randn(1, 1, core.n_freq, 3, requires_grad=True),
        torch.randn(1, 1, core.n_freq, 3, requires_grad=True),
    )

    out = model(x)
    loss = out.abs().mean()
    loss.backward()

    assert core.input_proj.weight.grad is not None
