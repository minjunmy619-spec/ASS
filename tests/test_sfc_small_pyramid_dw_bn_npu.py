from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import tempfile

import torch

from spectral_feature_compression.core.model.bandit_split import get_band_specs
from spectral_feature_compression.core.model.crossattn_enc_dec import prepare_bandit_position_bias
from spectral_feature_compression.core.model.online_model_wrapper import OnlineModelWrapper
from spectral_feature_compression.core.model.sfc_small_pyramid_dw_bn_npu import (
    CausalDepthwiseConv2dBNAct,
    SFCSmallPyramidDWBNNPUCore,
)
from tools.online.export_onnx_online_model import build_model_system_from_recipe_config
from spectral_feature_compression.utils.onnx_streaming import StreamingStateIOWrapper, flatten_tensor_tree


RECIPE = Path("recipes/dnr/models/sfc-small-pyramid-dw-bn-npu.musical64.onfly.rt192k/config.yaml")


def _small_core() -> SFCSmallPyramidDWBNNPUCore:
    return SFCSmallPyramidDWBNNPUCore(
        n_freq=65,
        n_fft=128,
        n_bands=16,
        n_src=2,
        d_inner=16,
        d_model=24,
        n_separator_layers=4,
        n_sfc_heads=4,
        pyramid_channels=(24, 32, 40, 48),
        dilation_cycle=(1, 2),
        decoder_ffn_hidden=8,
    )


def _stream_macs_per_frame(model: SFCSmallPyramidDWBNNPUCore) -> int:
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


def test_pyramid_sfc_uses_exact_official_position_bias() -> None:
    model = _small_core()
    bands, _, _ = get_band_specs("musical", 128, 44100, n_bands=16)
    official = prepare_bandit_position_bias(bands, q_len=16, kv_len=65, n_heads=4)[0]
    torch.testing.assert_close(model.encoder.pos_bias, official, rtol=0, atol=0)
    torch.testing.assert_close(model.decoder.pos_bias, official.transpose(1, 2), rtol=0, atol=0)


def test_pyramid_sfc_streaming_matches_full_eval() -> None:
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


def test_pyramid_sfc_default_budget_and_state_abi() -> None:
    system = build_model_system_from_recipe_config(RECIPE).eval()
    model = system.model.core
    params = sum(parameter.numel() for parameter in model.parameters())
    macs_per_second = _stream_macs_per_frame(model) * 44100 / 512
    state_bytes = model.state_size_bytes(dtype=torch.float16)
    frame_input_bytes = 2 * model.n_chan * model.n_freq * 2
    frame_output_bytes = 2 * model.n_src * model.n_chan * model.n_freq * 2
    total_abi_bytes = 2 * state_bytes + frame_input_bytes + frame_output_bytes

    assert 3_000_000 <= params <= 4_000_000
    assert macs_per_second < 3_000_000_000
    assert total_abi_bytes < 192 * 1024
    assert isinstance(system, OnlineModelWrapper)
    assert system.stft[0].center is False
    assert any(
        isinstance(module, CausalDepthwiseConv2dBNAct)
        and module.conv.groups == module.conv.in_channels
        for module in model.modules()
    )
    assert not any(module.__class__.__name__.lower().startswith("rms") for module in model.modules())


def test_pyramid_sfc_public_lazy_exports() -> None:
    import spectral_feature_compression as sfc

    assert sfc.SFCSmallPyramidDWBNNPUCore is SFCSmallPyramidDWBNNPUCore


def test_pyramid_sfc_streaming_onnx_has_compact_layout_contract() -> None:
    onnx = __import__("onnx")
    model = _small_core().eval()
    model.separator = type(model.separator)(
        model.d_model,
        n_bands=model.n_bands,
        pyramid_channels=(24, 32, 40, 48),
        n_blocks=4,
        time_kernel_size=2,
        freq_kernel_size=3,
        ffn_expansion=2,
        dilation_cycle=(1,),
    ).eval()
    model.masking = False
    wrapper = StreamingStateIOWrapper(model, batch_size=1, dtype=torch.float32)
    state, _ = flatten_tensor_tree(model.init_stream_state(dtype=torch.float32))
    x = torch.randn(1, 2, 1, 65)

    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "pyramid_sfc.onnx"
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                (x, *state),
                str(output),
                opset_version=11,
                do_constant_folding=True,
                dynamo=False,
            )
        graph = onnx.load(str(output))
        onnx.checker.check_model(graph)

    counts = defaultdict(int)
    for node in graph.graph.node:
        counts[node.op_type] += 1
    assert counts["MatMul"] == 4
    assert counts["Softmax"] == 2
    assert counts["Transpose"] == 2
    assert counts["Reshape"] == 6
    assert counts["Resize"] == 4
    assert counts["Slice"] == 0
    assert counts["Pad"] == 0
