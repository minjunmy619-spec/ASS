from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import tempfile

import torch

from spectral_feature_compression.core.model.bandit_split import get_band_specs
from spectral_feature_compression.core.model.crossattn_enc_dec import prepare_bandit_position_bias
from spectral_feature_compression.core.model.online_model_wrapper import OnlineModelWrapper
from spectral_feature_compression.core.model.sfc_small_macaron_conv2d_bn_npu import (
    FactorizedAxisSwiGLUFFN2D,
    MacaronAxisPath2D,
    NPUTFLocoformerBlock2D,
    SFCSmallMacaronConv2DBNNPUCore,
)
from spectral_feature_compression.utils.onnx_streaming import StreamingStateIOWrapper, flatten_tensor_tree
from tools.online.export_onnx_online_model import build_model_system_from_recipe_config


RECIPE = Path("recipes/dnr/models/sfc-small-macaron-conv2d-bn-npu.musical36.2l.onfly.rt192k/config.yaml")


def _small_core() -> SFCSmallMacaronConv2DBNNPUCore:
    return SFCSmallMacaronConv2DBNNPUCore(
        n_freq=65,
        n_fft=128,
        n_bands=12,
        n_src=2,
        d_inner=16,
        d_model=24,
        ffn_hidden=32,
        n_separator_layers=2,
        n_sfc_heads=4,
        frequency_kernel_size=15,
        time_kernel_size=2,
        dilation_cycle=(1,),
        decoder_ffn_hidden=8,
    )


def _stream_macs_per_frame(model: SFCSmallMacaronConv2DBNNPUCore) -> int:
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


def test_macaron_sfc_uses_exact_official_position_bias() -> None:
    model = _small_core()
    bands, _, _ = get_band_specs("musical", 128, 44100, n_bands=12)
    official = prepare_bandit_position_bias(bands, q_len=12, kv_len=65, n_heads=4)[0]
    torch.testing.assert_close(model.encoder.pos_bias, official, rtol=0, atol=0)
    torch.testing.assert_close(model.decoder.pos_bias, official.transpose(1, 2), rtol=0, atol=0)


def test_macaron_separator_preserves_official_tf_block_skeleton() -> None:
    model = _small_core()
    assert len(model.separator.blocks) == 2
    for block in model.separator.blocks:
        assert isinstance(block, NPUTFLocoformerBlock2D)
        assert isinstance(block.freq_path, MacaronAxisPath2D)
        assert isinstance(block.frame_path, MacaronAxisPath2D)
        for path in (block.freq_path, block.frame_path):
            assert isinstance(path.pre_ffn, FactorizedAxisSwiGLUFFN2D)
            assert isinstance(path.post_ffn, FactorizedAxisSwiGLUFFN2D)
            assert path.pre_ffn is not path.post_ffn

    frequency_ffns = [
        module
        for block in model.separator.blocks
        for module in (block.freq_path.pre_ffn, block.freq_path.post_ffn)
    ]
    temporal_ffns = [
        module
        for block in model.separator.blocks
        for module in (block.frame_path.pre_ffn, block.frame_path.post_ffn)
    ]
    assert len(frequency_ffns) == len(temporal_ffns) == 4


def test_macaron_separator_keeps_all_sfc_bands() -> None:
    model = _small_core().eval()
    observed_shapes = []
    handles = [
        block.register_forward_hook(
            lambda _module, inputs, output: observed_shapes.append(
                (tuple(inputs[0].shape), tuple(output.shape))
            )
        )
        for block in model.separator.blocks
    ]
    x = torch.randn(1, 2, 5, 65)
    with torch.no_grad():
        encoded = model.encoder(x)
        separated = model.separator(encoded)
    for handle in handles:
        handle.remove()

    assert encoded.shape == separated.shape == (1, 24, 5, 12)
    assert all(input_shape[-1] == output_shape[-1] == 12 for input_shape, output_shape in observed_shapes)


def test_macaron_streaming_matches_full_eval() -> None:
    torch.manual_seed(0)
    model = _small_core().eval()
    x = torch.randn(1, 2, 5, 65)
    with torch.no_grad():
        full = model(x)
        state = model.init_stream_state(dtype=x.dtype)
        frames = []
        for frame_idx in range(x.shape[2]):
            frame, state = model.forward_stream(x[:, :, frame_idx : frame_idx + 1], state)
            frames.append(frame)
    torch.testing.assert_close(torch.cat(frames, dim=2), full, rtol=1e-5, atol=1e-5)


def test_macaron_default_compute_and_state_budget() -> None:
    system = build_model_system_from_recipe_config(RECIPE).eval()
    model = system.model.core
    params = sum(parameter.numel() for parameter in model.parameters())
    macs_per_second = _stream_macs_per_frame(model) * 44100 / 512
    state_bytes = model.state_size_bytes(dtype=torch.float16)
    frame_input_bytes = 2 * model.n_chan * model.n_freq * 2
    frame_output_bytes = 2 * model.n_src * model.n_chan * model.n_freq * 2
    total_abi_bytes = 2 * state_bytes + frame_input_bytes + frame_output_bytes

    assert params == 1_003_894
    assert macs_per_second < 3_000_000_000
    assert total_abi_bytes < 192 * 1024
    assert isinstance(system, OnlineModelWrapper)
    assert system.stft[0].center is False
    assert len(model.init_stream_state()) == 6
    assert all(tuple(state.shape[-2:]) == (1, 36) for state in model.init_stream_state())


def test_macaron_lazy_package_export() -> None:
    import spectral_feature_compression as sfc

    assert sfc.SFCSmallMacaronConv2DBNNPUCore is SFCSmallMacaronConv2DBNNPUCore


def test_macaron_streaming_onnx_preserves_gating_without_transport_ops() -> None:
    import onnx

    torch.manual_seed(0)
    model = _small_core().eval()
    model.masking = False
    state = model.init_stream_state(dtype=torch.float32)
    wrapper = StreamingStateIOWrapper(model, batch_size=1, dtype=torch.float32).eval()
    x = torch.randn(1, 2, 1, 65)
    flat_state, _ = flatten_tensor_tree(state)

    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "macaron.onnx"
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
    assert counts["Resize"] == 0
    assert counts["ConvTranspose"] == 0
    assert counts["Split"] == 0
    assert counts["Slice"] == 0
    assert counts["Pad"] == 0
    assert counts["Sigmoid"] == 8
    assert counts["Mul"] == 16
    assert counts["Transpose"] == 2
    assert counts["Reshape"] == 6
    assert counts["MatMul"] == 4
    assert counts["Softmax"] == 2
