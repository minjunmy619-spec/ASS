"""Smoke tests for ``TIGEREdgeMLP`` (v1 / v2 / v3 TF-MLPNet edge backbone)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Package root: ``TF-MLPNet/`` (parent of ``tf_mlpnet/``)
_TF_ROOT = Path(__file__).resolve().parent.parent
_ASS_ROOT = _TF_ROOT.parent
if str(_ASS_ROOT) not in sys.path:
    sys.path.insert(0, str(_ASS_ROOT))
if str(_TF_ROOT) not in sys.path:
    sys.path.insert(0, str(_TF_ROOT))

from tf_mlpnet.export_onnx import (  # noqa: E402
    TIGEREdgeMLPCellExportWrapper,
    build_tiger_edge_mlp_dummy_inputs,
    export_tiger_edge_mlp_to_onnx,
    precheck_tiger_edge_mlp_export,
)
from tf_mlpnet.npu_utils import assert_conv_receptive_field, streaming_state_bytes_fp32  # noqa: E402
from tf_mlpnet.legacy_v1 import V1TIGEREdgeMLP  # noqa: E402
from tf_mlpnet.tiger_edge_mlp import TIGEREdgeMLP  # noqa: E402
from tf_mlpnet.tiger_edge_mlp_v3 import (  # noqa: E402
    TIGEREdgeMLPV3,
    V3_PRESETS,
    build_tiger_edge_mlp_v3,
)


# ---------------------------------------------------------------------------
# v2 tests (unchanged; retained as regressions)
# ---------------------------------------------------------------------------


def test_forward_cell_and_streaming_budget():
    model = TIGEREdgeMLP(
        out_channels=32,
        in_channels=128,
        num_blocks=4,
        upsampling_depth=2,
        num_sources=2,
        need_streaming=True,
        edge_hidden_channels=32,
        edge_num_blocks=4,
        edge_time_dilations=(1, 2, 4),
    )
    model.eval()
    assert_conv_receptive_field(model, limit=14)

    states = model.init_streaming_state(batch_size=1)
    nbytes = streaming_state_bytes_fp32(states)
    assert nbytes <= 192 * 1024, f"streaming state {nbytes} B exceeds 192 KiB"

    ri = torch.randn(1, 1, model.enc_dim * 2, 1)
    out = model.forward_cell(ri, *states)
    assert len(out) == 7
    assert out[0].dim() == 4


def test_sequence_matches_unrolled_cell():
    torch.manual_seed(0)
    model = TIGEREdgeMLP(
        out_channels=24,
        in_channels=96,
        num_blocks=4,
        upsampling_depth=2,
        num_sources=2,
        need_streaming=True,
        edge_hidden_channels=24,
        edge_num_blocks=4,
    )
    model.eval()
    T = 5
    ri = torch.randn(1, 1, model.enc_dim * 2, T)

    seq_out, *_ = model.forward_sequence(ri, detach_state=False)

    s = model.init_streaming_state(1, device=ri.device, dtype=ri.dtype)
    past_kvs, past_valid, p0, p1, p2, pg = s
    frames = []
    for t in range(T):
        fo, past_kvs, past_valid, p0, p1, p2, pg = model.forward_cell(
            ri[:, :, :, t : t + 1],
            past_kvs=past_kvs,
            past_valid_mask=past_valid,
            prev_states_0=p0,
            prev_states_1=p1,
            prev_states_2=p2,
            prev_global_states=pg,
        )
        frames.append(fo)
    cell_cat = torch.cat(frames, dim=-1)

    assert torch.allclose(seq_out, cell_cat, atol=1e-5, rtol=1e-4)


def test_v1_forward_cell_smoke():
    model = V1TIGEREdgeMLP(
        out_channels=24,
        in_channels=96,
        num_blocks=4,
        upsampling_depth=2,
        num_sources=2,
        need_streaming=True,
        edge_hidden_channels=24,
        edge_num_blocks=4,
    )
    model.eval()
    states = model.init_streaming_state(batch_size=1)
    ri = torch.randn(1, 1, model.enc_dim * 2, 1)
    out = model.forward_cell(ri, *states)
    assert len(out) == 7
    assert out[0].dim() == 4


def test_onnx_export_roundtrip_smoke(tmp_path: Path):
    model = TIGEREdgeMLP(
        out_channels=24,
        in_channels=96,
        num_blocks=4,
        upsampling_depth=2,
        num_sources=2,
        need_streaming=True,
        edge_hidden_channels=24,
        edge_num_blocks=4,
    )
    model.eval()
    wrapper = TIGEREdgeMLPCellExportWrapper(model)
    inputs = build_tiger_edge_mlp_dummy_inputs(model, batch_size=1)
    precheck_tiger_edge_mlp_export(wrapper, inputs)
    out_onnx = tmp_path / "tiger_edge_mlp_cell.onnx"
    export_tiger_edge_mlp_to_onnx(wrapper, inputs, out_onnx, opset_version=17)
    assert out_onnx.is_file()

    import onnx

    onnx.checker.check_model(onnx.load(str(out_onnx)))


# ---------------------------------------------------------------------------
# v3 tests
# ---------------------------------------------------------------------------


def _small_v3_model() -> TIGEREdgeMLPV3:
    """Small v3 model used by most smoke tests (~hundreds of KB of params)."""
    return TIGEREdgeMLPV3(
        out_channels=24,
        in_channels=96,
        num_blocks=4,
        upsampling_depth=2,
        num_sources=2,
        need_streaming=True,
        edge_hidden_channels=32,
        edge_num_blocks=4,
        edge_freq_kernel_size=5,
        edge_time_kernel_size=3,
        edge_time_dilations=(1, 2, 4),
    ).eval()


def test_v3_npu_op_audit():
    """Every Conv2d / AvgPool2d in v3 honours (k-1)*d < 14."""
    model = _small_v3_model()
    assert_conv_receptive_field(model, limit=14)


def test_v3_forward_cell_shape_and_state_budget():
    """v3 cell forward returns the same 7-tuple layout and fits the quota."""
    model = _small_v3_model()
    states = model.init_streaming_state(batch_size=1)
    assert len(states) == 6
    # Budget check on the small model; v3 presets in V3_PRESETS are checked
    # separately below so we can read the per-preset numbers.
    nbytes = streaming_state_bytes_fp32(states)
    assert nbytes <= 192 * 1024, f"v3 small streaming state {nbytes} B exceeds 192 KiB"

    ri = torch.randn(1, 1, model.enc_dim * 2, 1)
    out = model.forward_cell(ri, *states)
    assert len(out) == 7
    band_masked, *state_outs = out
    assert band_masked.dim() == 4
    # Sanity: output matches the same layout TIGER's mask decoder produces
    assert band_masked.shape[-1] == 1
    # The 6 returned state tensors must match the input shapes exactly.
    for old, new in zip(states, state_outs):
        assert old.shape == new.shape, (
            f"v3 state shape changed: {old.shape} -> {new.shape}"
        )


def test_v3_sequence_matches_unrolled_cell():
    """forward_sequence must be bit-identical to a manual frame-by-frame loop."""
    torch.manual_seed(0)
    model = _small_v3_model()
    T = 5
    ri = torch.randn(1, 1, model.enc_dim * 2, T)

    seq_out, *_ = model.forward_sequence(ri, detach_state=False)

    s = model.init_streaming_state(1, device=ri.device, dtype=ri.dtype)
    past_kvs, past_valid, p0, p1, p2, pg = s
    frames = []
    for t in range(T):
        fo, past_kvs, past_valid, p0, p1, p2, pg = model.forward_cell(
            ri[:, :, :, t : t + 1],
            past_kvs=past_kvs,
            past_valid_mask=past_valid,
            prev_states_0=p0,
            prev_states_1=p1,
            prev_states_2=p2,
            prev_global_states=pg,
        )
        frames.append(fo)
    cell_cat = torch.cat(frames, dim=-1)

    assert torch.allclose(seq_out, cell_cat, atol=1e-5, rtol=1e-4)


def test_v3_state_carries_memory():
    """EMA global state should differ between first and later frames."""
    torch.manual_seed(1)
    model = _small_v3_model()
    s = model.init_streaming_state(1)

    ri = torch.randn(1, 1, model.enc_dim * 2, 1)
    out1 = model.forward_cell(ri, *s)
    # global_state is the last element of the 7-tuple
    _, _, _, _, _, _, g1 = out1

    # Feed a second (different) frame and check the global state actually moves
    ri2 = torch.randn(1, 1, model.enc_dim * 2, 1)
    out2 = model.forward_cell(ri2, *out1[1:])
    _, _, _, _, _, _, g2 = out2
    assert not torch.allclose(g1, g2), "v3 global EMA state should update frame-to-frame"


def test_v3_onnx_export_roundtrip_smoke(tmp_path: Path):
    model = _small_v3_model()
    wrapper = TIGEREdgeMLPCellExportWrapper(model)
    inputs = build_tiger_edge_mlp_dummy_inputs(model, batch_size=1)
    precheck_tiger_edge_mlp_export(wrapper, inputs)
    out_onnx = tmp_path / "tiger_edge_mlp_v3_cell.onnx"
    export_tiger_edge_mlp_to_onnx(wrapper, inputs, out_onnx, opset_version=17)
    assert out_onnx.is_file()

    import onnx

    model_proto = onnx.load(str(out_onnx))
    onnx.checker.check_model(model_proto)

    # Audit that the exported graph sticks to the NPU-friendly op set.
    forbidden = {"BatchMatMul", "MatMul", "LayerNormalization", "GroupNormalization"}
    offending = sorted({n.op_type for n in model_proto.graph.node} & forbidden)
    assert not offending, f"v3 ONNX graph contains NPU-unfriendly ops: {offending}"


def test_v3_presets_fit_budget():
    """Every preset in ``V3_PRESETS`` fits the 192 KiB fp16 DSP state quota
    at a typical DnR configuration (win=2048, num_sources=3)."""
    for name in V3_PRESETS:
        model = build_tiger_edge_mlp_v3(
            name,
            num_sources=3,
            sample_rate=44100,
            win=2048,
            stride=512,
        ).eval()
        states = model.init_streaming_state(batch_size=1)
        nbytes_fp32 = streaming_state_bytes_fp32(states)
        # fp16 deployment halves the footprint.
        nbytes_fp16 = nbytes_fp32 // 2
        assert nbytes_fp16 <= 192 * 1024, (
            f"v3 preset {name!r} streaming state {nbytes_fp16} B exceeds 192 KiB (fp16)"
        )
        # Also audit conv receptive fields.
        assert_conv_receptive_field(model, limit=14)


def test_v3_preset_parameter_counts():
    """Each v3 preset should be in the advertised 2M-10M total range."""
    expected = {
        "v3-small":   (2_000_000,  5_000_000),
        "v3-balance": (4_000_000,  8_000_000),
        "v3-large":   (7_000_000, 11_000_000),
    }
    for name, (lo, hi) in expected.items():
        model = build_tiger_edge_mlp_v3(
            name, num_sources=3, sample_rate=44100, win=2048, stride=512,
        )
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert lo <= n <= hi, (
            f"v3 preset {name!r} has {n:,} params, expected in [{lo:,}, {hi:,}]"
        )


if __name__ == "__main__":
    import tempfile

    test_forward_cell_and_streaming_budget()
    test_sequence_matches_unrolled_cell()
    test_v1_forward_cell_smoke()
    test_onnx_export_roundtrip_smoke(Path(tempfile.mkdtemp()))

    test_v3_npu_op_audit()
    test_v3_forward_cell_shape_and_state_budget()
    test_v3_sequence_matches_unrolled_cell()
    test_v3_state_carries_memory()
    test_v3_onnx_export_roundtrip_smoke(Path(tempfile.mkdtemp()))
    test_v3_presets_fit_budget()
    test_v3_preset_parameter_counts()
    print("[ok] TF-MLPNet smoke tests (v1/v2/v3) passed")
