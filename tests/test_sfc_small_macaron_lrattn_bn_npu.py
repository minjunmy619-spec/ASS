from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import tempfile

import torch

from spectral_feature_compression.core.model.bandit_split import get_band_specs
from spectral_feature_compression.core.model.crossattn_enc_dec import prepare_bandit_position_bias
from spectral_feature_compression.core.model.online_model_wrapper import OnlineModelWrapper
from spectral_feature_compression.core.model.sfc_small_macaron_lrattn_bn_npu import (
    LowRankAxialAttention2D,
    LowRankMacaronAxisPath2D,
    LowRankNPUTFLocoformerBlock2D,
    SFCSmallMacaronLRAttnBNNPUCore,
)
from spectral_feature_compression.utils.onnx_streaming import (
    StreamingStateIOWrapper,
    flatten_tensor_tree,
)
from tools.online.export_onnx_online_model import (
    audit_onnx_graph,
    build_model_system_from_recipe_config,
    get_allowed_ops,
)

RECIPE = Path(
    "recipes/dnr/models/"
    "sfc-small-macaron-lrattn-bn-npu.musical36.2l.r2d64g560.onfly.rt192k/config.yaml"
)


def _small_core() -> SFCSmallMacaronLRAttnBNNPUCore:
    return SFCSmallMacaronLRAttnBNNPUCore(
        n_freq=65,
        n_fft=128,
        n_bands=12,
        n_src=2,
        d_inner=16,
        d_model=24,
        ffn_hidden=32,
        n_separator_layers=2,
        n_sfc_heads=4,
        attention_rank=2,
        attention_value_channels=20,
        frequency_context_hidden_channels=24,
        frequency_kernel_size=15,
        time_kernel_size=2,
        dilation_cycle=(1,),
        decoder_ffn_hidden=8,
    )


def _stream_macs_per_frame(model: SFCSmallMacaronLRAttnBNNPUCore) -> int:
    conv_macs = defaultdict(int)
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):

            def hook(module, _inputs, output, name=name):
                kernel_t, kernel_f = module.kernel_size
                conv_macs[name] += (
                    output.numel()
                    * (module.in_channels // module.groups)
                    * kernel_t
                    * kernel_f
                )

            handles.append(module.register_forward_hook(hook))

    x = torch.randn(1, 2 * model.n_chan, 1, model.n_freq)
    state = model.init_stream_state(dtype=x.dtype)
    with torch.no_grad():
        model.forward_stream(x, state)
    for handle in handles:
        handle.remove()

    head_dim = model.d_inner // model.n_sfc_heads
    attention_macs = 4 * model.n_sfc_heads * model.n_bands * model.n_freq * head_dim
    return sum(conv_macs.values()) + attention_macs


def test_lrattn_uses_exact_sfc_position_bias() -> None:
    model = _small_core()
    bands, _, _ = get_band_specs("musical", 128, 44100, n_bands=12)
    official = prepare_bandit_position_bias(bands, q_len=12, kv_len=65, n_heads=4)[0]
    torch.testing.assert_close(model.encoder.pos_bias, official, rtol=0, atol=0)
    torch.testing.assert_close(model.decoder.pos_bias, official.transpose(1, 2), rtol=0, atol=0)


def test_lrattn_preserves_macaron_skeleton_and_global_mixers() -> None:
    model = _small_core()
    assert len(model.separator.blocks) == 2
    for block in model.separator.blocks:
        assert isinstance(block, LowRankNPUTFLocoformerBlock2D)
        assert isinstance(block.freq_path, LowRankMacaronAxisPath2D)
        assert isinstance(block.frame_path, LowRankMacaronAxisPath2D)
        assert isinstance(block.freq_path.attention, LowRankAxialAttention2D)
        assert isinstance(block.frame_path.attention, LowRankAxialAttention2D)
        assert block.freq_path.attention.axis == "frequency"
        assert block.frame_path.attention.axis == "time"
        assert block.freq_path.pre_ffn is not block.freq_path.post_ffn
        assert block.frame_path.pre_ffn is not block.frame_path.post_ffn


def test_lrattn_frequency_output_depends_on_distant_band() -> None:
    torch.manual_seed(0)
    attention = LowRankAxialAttention2D(
        8,
        6,
        axis="frequency",
        rank=2,
        n_bands=36,
        temporal_decay=0.995,
        frequency_context_hidden_channels=12,
    ).eval()
    x = torch.randn(1, 8, 1, 36, requires_grad=True)
    attention(x)[:, :, :, 0].sum().backward()
    assert x.grad is not None
    assert torch.count_nonzero(x.grad[:, :, :, -1]).item() > 0


def test_lrattn_temporal_output_depends_on_entire_causal_history() -> None:
    torch.manual_seed(0)
    attention = LowRankAxialAttention2D(
        8,
        6,
        axis="time",
        rank=2,
        n_bands=12,
        temporal_decay=0.995,
        frequency_context_hidden_channels=12,
    ).eval()
    x = torch.randn(1, 8, 12, 12, requires_grad=True)
    attention(x)[:, :, -1:, :].sum().backward()
    assert x.grad is not None
    assert torch.count_nonzero(x.grad[:, :, :1, :]).item() > 0


def test_lrattn_streaming_matches_full_eval() -> None:
    torch.manual_seed(0)
    model = _small_core().eval()
    x = torch.randn(1, 2, 7, 65)
    with torch.no_grad():
        full = model(x)
        state = model.init_stream_state(dtype=x.dtype)
        frames = []
        for frame_index in range(x.shape[2]):
            frame, state = model.forward_stream(
                x[:, :, frame_index : frame_index + 1],
                state,
            )
            frames.append(frame)
    torch.testing.assert_close(torch.cat(frames, dim=2), full, rtol=1e-5, atol=1e-5)


def test_lrattn_rejects_non_frame_streaming_input() -> None:
    model = _small_core().eval()
    x = torch.randn(1, 2, 2, 65)
    state = model.init_stream_state(dtype=x.dtype)
    try:
        model.forward_stream(x, state)
    except RuntimeError as error:
        assert "exactly one frame" in str(error)
    else:
        raise AssertionError("Expected multi-frame streaming input to be rejected")


def test_lrattn_default_compute_and_state_budget() -> None:
    system = build_model_system_from_recipe_config(RECIPE).eval()
    model = system.model.core
    params = sum(parameter.numel() for parameter in model.parameters())
    macs_per_second = _stream_macs_per_frame(model) * 44100 / 512
    state_bytes = model.state_size_bytes(dtype=torch.float16)
    frame_input_bytes = 2 * model.n_chan * model.n_freq * 2
    frame_output_bytes = 2 * model.n_src * model.n_chan * model.n_freq * 2
    total_abi_bytes = 2 * state_bytes + frame_input_bytes + frame_output_bytes

    assert params == 2_556_198
    assert macs_per_second < 3_000_000_000
    assert state_bytes == 55_296
    assert total_abi_bytes == 126_992
    assert total_abi_bytes < 192 * 1024
    assert isinstance(system, OnlineModelWrapper)
    assert system.stft[0].center is False
    assert len(model.init_stream_state()) == 8


def test_lrattn_lazy_package_export() -> None:
    import spectral_feature_compression as sfc

    assert sfc.SFCSmallMacaronLRAttnBNNPUCore is SFCSmallMacaronLRAttnBNNPUCore


def test_lrattn_streaming_onnx_has_no_separator_transport_or_quadratic_ops() -> None:
    import onnx

    torch.manual_seed(0)
    model = _small_core().eval()
    model.masking = False
    state = model.init_stream_state(dtype=torch.float32)
    wrapper = StreamingStateIOWrapper(model, batch_size=1, dtype=torch.float32).eval()
    x = torch.randn(1, 2, 1, 65)
    flat_state, _ = flatten_tensor_tree(state)

    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "lrattn.onnx"
        torch.onnx.export(
            wrapper,
            (x, *flat_state),
            output,
            opset_version=14,
            do_constant_folding=True,
            dynamo=False,
        )
        graph = onnx.load(output)
        onnx.checker.check_model(graph)

    counts = Counter(node.op_type for node in graph.graph.node)
    audit = audit_onnx_graph(
        graph,
        allowed_ops=get_allowed_ops("edge_npu_recommended"),
    )
    assert audit["disallowed_ops"] == []
    assert counts["Resize"] == 0
    assert counts["ConvTranspose"] == 0
    assert counts["Split"] == 0
    assert counts["Slice"] == 0
    assert counts["Pad"] == 0
    assert counts["CumSum"] == 0
    assert counts["Transpose"] == 2
    assert counts["Reshape"] == 6
    assert counts["MatMul"] == 4
    assert counts["Softmax"] == 2
    assert counts["AveragePool"] == 4
    assert counts["Div"] == 0
