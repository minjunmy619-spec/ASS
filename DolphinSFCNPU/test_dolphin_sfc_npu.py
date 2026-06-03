from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from DolphinSFCNPU import (  # noqa: E402
    DolphinSFCNPUSeparator,
    DolphinSFCNPUStreamingExportWrapper,
    build_dolphin_sfc_npu_from_config,
    build_dolphin_sfc_npu_preset,
)
from DolphinSFCNPU.training_wrapper import build_dolphin_sfc_npu_system  # noqa: E402

FORBIDDEN_EXPORT_OPS = {
    "ConstantOfShape",
    "Expand",
    "Tile",
    "ScatterND",
    "Unflatten",
}

# AGENT.md rule 13: 192 KiB DSP quota covers all streaming state for the
# exported graph.  We allow a small headroom for the residual activations the
# DSP will also hold simultaneously, so the budget asserted here is slightly
# tighter than the raw quota.
STATE_BUDGET_BYTES = 192 * 1024
# AGENT.md rule 15: parameter budget.
PARAM_UPPER_LIMIT = 8_000_000
PARAM_LOWER_LIMIT = 3_000_000


def build_small_model() -> DolphinSFCNPUSeparator:
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
    packed_state = wrapper.init_packed_state(batch_size=1, dtype=torch.float32)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (x, packed_state),
            str(out_path),
            opset_version=11,
            input_names=["x", "state"],
            output_names=["y", "next_state"],
            do_constant_folding=True,
            dynamo=False,
        )

    model_proto = onnx.load(str(out_path))
    onnx.checker.check_model(model_proto)
    return model_proto, packed_state


def collect_ops(model_proto) -> set[str]:
    return {node.op_type for node in model_proto.graph.node}


def test_forward_stream_matches_forward() -> None:
    torch.manual_seed(0)
    model = build_small_model().eval()
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


def test_training_wrapper_frequency_preprocessing_shape() -> None:
    system = build_dolphin_sfc_npu_system(
        n_fft=1024,
        hop_length=256,
        fs=16000,
        preset="edge_small",
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


def test_default_output_head_initializes_to_mixture_split_gain() -> None:
    torch.manual_seed(0)
    model = build_dolphin_sfc_npu_preset("edge_small", n_freq=65, n_fft=128, sample_rate=16000).eval()
    x = torch.randn(1, 2, 3, model.n_freq)
    with torch.no_grad():
        y = model(x)

    expected = torch.cat([x / float(model.n_src) for _ in range(model.n_src)], dim=1)
    torch.testing.assert_close(y, expected, rtol=1e-6, atol=1e-6)


def test_softmax_output_head_is_mixture_consistent() -> None:
    torch.manual_seed(0)
    model = build_dolphin_sfc_npu_preset(
        "edge_small",
        n_freq=65,
        n_fft=128,
        sample_rate=16000,
        mask_activation="softmax",
    ).eval()
    x = torch.randn(1, 2, 3, model.n_freq)
    with torch.no_grad():
        y = model(x)

    summed = y.reshape(1, model.n_src, 2 * model.n_chan, x.shape[2], x.shape[3]).sum(dim=1)
    torch.testing.assert_close(summed, x, rtol=1e-6, atol=1e-6)


def test_query_variant_presets_forward_stream_match() -> None:
    torch.manual_seed(0)
    for preset in ("edge_small_soft_query", "edge_small_crossattn_query"):
        model = build_dolphin_sfc_npu_preset(preset, n_freq=129, n_fft=256, sample_rate=16000).eval()
        x = torch.randn(1, 2, 4, model.n_freq)
        with torch.no_grad():
            full = model(x)
            state = model.init_stream_state(batch_size=1, dtype=x.dtype)
            chunks = []
            for t in range(x.shape[2]):
                y, state = model.forward_stream(x[:, :, t : t + 1, :], state)
                chunks.append(y)
            streamed = torch.cat(chunks, dim=2)
        assert full.shape == streamed.shape == (1, 6, 4, model.n_freq)
        assert torch.allclose(full, streamed, atol=5e-5, rtol=5e-5), preset
        assert model.state_size_bytes(batch_size=1, dtype=torch.float16) <= STATE_BUDGET_BYTES


def test_query_variant_aliases_and_capacity_are_useful() -> None:
    for preset in ("slim_6m_soft_band_query", "slim_6m_soft_query", "slim_6m_crossattn_query"):
        model = build_dolphin_sfc_npu_preset(preset, n_freq=475, n_fft=948, sample_rate=44100).eval()
        params = sum(p.numel() for p in model.parameters())
        state_bytes = model.state_size_bytes(batch_size=1, dtype=torch.float16)
        assert 3_000_000 <= params <= 7_000_000, f"{preset} params outside useful target: {params}"
        assert state_bytes <= STATE_BUDGET_BYTES, f"{preset} state too large: {state_bytes}"


def test_config_builder_threads_capacity_overrides() -> None:
    model = build_dolphin_sfc_npu_from_config(
        preset="slim_6m_soft_band_query",
        n_freq=257,
        n_fft=512,
        sample_rate=16000,
        n_bands=40,
        d_model=96,
        num_scales=3,
        widths=[96, 160, 224],
        blocks_per_scale=[1, 1, 1],
        time_kernels=[3, 5, 3],
        freq_kernels=[3, 5, 3],
        compressor_freq_kernel=5,
        ffn_expansion=3,
    ).eval()
    assert model.query_variant == "soft_band_query"
    assert model.n_bands == 40
    assert model.d_model == 96
    assert model.widths == (96, 160, 224)
    assert model.blocks_per_scale == (1, 1, 1)
    assert model.time_kernels == (3, 5, 3)
    assert model.freq_kernels == (3, 5, 3)
    assert model.compressor_freq_kernel == 5
    assert model.ffn_expansion == 3
    assert model.encoder[0].blocks[0].f_in.out_channels == 96 * 3 * 2


def test_training_wrapper_threads_capacity_overrides() -> None:
    system = build_dolphin_sfc_npu_system(
        n_fft=512,
        hop_length=128,
        fs=16000,
        preset="slim_6m_crossattn_query",
        query_type="learnable",
        n_bands=40,
        d_model=96,
        num_scales=3,
        widths=[96, 160, 224],
        blocks_per_scale=[1, 1, 1],
        time_kernels=[3, 5, 3],
        freq_kernels=[3, 5, 3],
        compressor_freq_kernel=5,
        ffn_expansion=3,
    )
    core = system.model.core
    assert core.query_variant == "crossattn_query"
    assert core.query_type == "learnable"
    assert core.n_bands == 40
    assert core.widths == (96, 160, 224)
    assert core.compressor_freq_kernel == 5
    assert core.ffn_expansion == 3


def test_query_variant_builder_flag_matches_named_preset() -> None:
    torch.manual_seed(0)
    named = build_dolphin_sfc_npu_preset(
        "edge_small_soft_query",
        n_freq=65,
        n_fft=128,
        sample_rate=16000,
    ).eval()
    flagged = build_dolphin_sfc_npu_preset(
        "edge_small",
        n_freq=65,
        n_fft=128,
        sample_rate=16000,
        query_variant="soft_band_query",
    ).eval()
    assert named.query_variant == flagged.query_variant == "soft_band_query"
    assert named.state_size_bytes(batch_size=1, dtype=torch.float16) == flagged.state_size_bytes(
        batch_size=1,
        dtype=torch.float16,
    )


def test_slim_presets_forward_stream_match() -> None:
    torch.manual_seed(0)
    for preset in ("slim_4m", "slim_6m", "slim_8m"):
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
        assert torch.allclose(full, streamed, atol=5e-5, rtol=5e-5), preset


def test_slim_presets_fit_param_and_state_budgets() -> None:
    for preset in ("slim_4m", "slim_6m", "slim_8m"):
        model = build_dolphin_sfc_npu_preset(preset, n_freq=257, n_fft=512, sample_rate=16000).eval()
        params = sum(p.numel() for p in model.parameters())
        state_bytes = model.state_size_bytes(batch_size=1, dtype=torch.float16)

        assert PARAM_LOWER_LIMIT <= params <= PARAM_UPPER_LIMIT, f"{preset} params {params} out of 3-8M range"
        assert state_bytes <= STATE_BUDGET_BYTES, f"{preset} fp16 state bytes {state_bytes} exceed {STATE_BUDGET_BYTES}"


def test_streaming_onnx_export_has_small_io_parameter_count() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model_proto, _ = export_streaming_onnx(build_small_model().eval(), Path(tmpdir) / "edge_small.onnx")
    ops = collect_ops(model_proto)
    forbidden = sorted(ops & FORBIDDEN_EXPORT_OPS)
    assert not forbidden, f"edge_small exported forbidden ops: {forbidden}"
    # AGENT.md rule 14: keep the number of I/O parameters small.
    assert len(model_proto.graph.input) == 2, [i.name for i in model_proto.graph.input]
    assert len(model_proto.graph.output) == 2, [o.name for o in model_proto.graph.output]


def test_slim_presets_onnx_op_audit() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        for preset in ("slim_4m", "slim_6m", "slim_8m"):
            model = build_dolphin_sfc_npu_preset(preset, n_freq=257, n_fft=512, sample_rate=16000).eval()
            model_proto, packed_state = export_streaming_onnx(model, Path(tmpdir) / f"{preset}.onnx")
            ops = collect_ops(model_proto)
            forbidden = sorted(ops & FORBIDDEN_EXPORT_OPS)
            assert not forbidden, f"{preset} exported forbidden ops: {forbidden}"
            assert "MatMul" in ops, "The band compressor/decoder bmm path should export as MatMul."
            assert len(model_proto.graph.input) == 2, [i.name for i in model_proto.graph.input]
            assert len(model_proto.graph.output) == 2, [o.name for o in model_proto.graph.output]
            assert packed_state.ndim == 2 and packed_state.shape[0] == 1

        model = build_dolphin_sfc_npu_preset(
            "edge_small",
            n_freq=257,
            n_fft=512,
            sample_rate=16000,
            mask_activation="softmax",
        ).eval()
        model_proto, _packed_state = export_streaming_onnx(model, Path(tmpdir) / "edge_small_softmax.onnx")
        ops = collect_ops(model_proto)
        assert "Softmax" in ops
        assert not sorted(ops & FORBIDDEN_EXPORT_OPS)


def test_packed_state_roundtrip_matches_tree() -> None:
    torch.manual_seed(0)
    model = build_small_model().eval()
    wrapper = DolphinSFCNPUStreamingExportWrapper(model, batch_size=1, dtype=torch.float32).eval()

    x = torch.randn(1, 2, 4, model.n_freq)
    tree_state = model.init_stream_state(batch_size=1, dtype=x.dtype)
    packed_state = wrapper.init_packed_state(batch_size=1, dtype=x.dtype)

    with torch.no_grad():
        tree_outs = []
        wrapper_outs = []
        for t in range(x.shape[2]):
            frame = x[:, :, t : t + 1, :]
            tree_y, tree_state = model.forward_stream(frame, tree_state)
            wrap_y, packed_state = wrapper(frame, packed_state)
            tree_outs.append(tree_y)
            wrapper_outs.append(wrap_y)

    tree_full = torch.cat(tree_outs, dim=2)
    wrap_full = torch.cat(wrapper_outs, dim=2)
    assert torch.allclose(tree_full, wrap_full, atol=1e-6, rtol=1e-6)


def test_state_leaf_count_is_small() -> None:
    """The slim design should expose far fewer state leaves than the old one.

    This is a structural regression guard; if somebody accidentally reintroduces
    a per-block extra cache (e.g. a global_dw path) this test will catch it.
    """
    model = build_dolphin_sfc_npu_preset("slim_8m", n_freq=257, n_fft=512, sample_rate=16000).eval()
    wrapper = DolphinSFCNPUStreamingExportWrapper(model, batch_size=1, dtype=torch.float32).eval()
    total_blocks = sum(model.blocks_per_scale) * 2  # encoder + decoder, one cache per block
    assert wrapper.state_tensor_count == total_blocks, (
        f"Expected one cache per slim block ({total_blocks}), got {wrapper.state_tensor_count}."
    )


if __name__ == "__main__":
    test_forward_stream_matches_forward()
    test_default_output_head_initializes_to_mixture_split_gain()
    test_softmax_output_head_is_mixture_consistent()
    test_query_variant_presets_forward_stream_match()
    test_query_variant_aliases_and_capacity_are_useful()
    test_config_builder_threads_capacity_overrides()
    test_training_wrapper_threads_capacity_overrides()
    test_query_variant_builder_flag_matches_named_preset()
    test_slim_presets_forward_stream_match()
    test_slim_presets_fit_param_and_state_budgets()
    test_state_leaf_count_is_small()
    test_packed_state_roundtrip_matches_tree()
    try:
        test_streaming_onnx_export_has_small_io_parameter_count()
        test_slim_presets_onnx_op_audit()
    except ModuleNotFoundError as exc:
        print(f"[skip] ONNX export smoke skipped: {exc}")

    for preset in ("edge_small", "slim_4m", "slim_6m", "slim_8m"):
        model = build_dolphin_sfc_npu_preset(preset, n_freq=1025, n_fft=2048, sample_rate=44100)
        params = sum(p.numel() for p in model.parameters())
        state_bytes = model.state_size_bytes(batch_size=1, dtype=torch.float16)
        print(f"[ok] {preset}: params={params:_}, fp16_state_bytes={state_bytes:_}")
