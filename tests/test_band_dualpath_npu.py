from __future__ import annotations

import copy
from pathlib import Path
import sys

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from spectral_feature_compression.core.model.band_dualpath_npu import (  # noqa: E402
    DEFAULT_REGIONS,
    PRESETS,
    BandDualPathNPUCore,
    BandDualPathNPUExportWrapper,
    BandDualPathNPUModel,
    BandLayout,
    RMSNorm2d,
    SceneGRU,
    _gru_step_conv,
    build_band_dualpath_npu_system,
)

FRAMES_PER_SECOND = 44100 / 512


def _randomize_identity_params(model: torch.nn.Module) -> None:
    """Make norm gains and zero-initialized FiLM non-trivial so every path is exercised."""
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, RMSNorm2d):
                module.weight.uniform_(0.5, 1.5)
            if isinstance(module, SceneGRU):
                for conv in (module.gamma, module.beta):
                    conv.weight.normal_(0.0, 0.05)
                    conv.bias.normal_(0.0, 0.05)


def _stream(core: BandDualPathNPUCore, feats: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    state = core.init_stream_state(feats[0].shape[0])
    outs: list[list[torch.Tensor]] = [[] for _ in feats]
    for t in range(feats[0].shape[2]):
        masks, state = core.forward_stream([f[:, :, t : t + 1] for f in feats], state)
        for idx, mask in enumerate(masks):
            outs[idx].append(mask)
    return tuple(torch.cat(o, dim=2) for o in outs)


def test_layout_covers_spectrum_and_roundtrips() -> None:
    layout = BandLayout(1025, DEFAULT_REGIONS)
    assert layout.n_bands == 48
    assert layout.band_counts == (24, 16, 8)
    x = torch.randn(2, 6, 5, 1025)
    parts = layout.pack(x)
    assert [tuple(p.shape) for p in parts] == [(2, 24, 5, 24), (2, 96, 5, 16), (2, 504, 5, 8)]
    y = layout.unpack(parts, 6)
    assert torch.equal(y[..., :1024], x[..., :1024])
    assert torch.equal(y[..., 1024], x[..., 1023])  # Nyquist bin reuses the last mask
    # Channel index is c * W + w: bin 5 of channel 1 lives in band 1 (4-bin bands), slot w=1.
    assert torch.equal(parts[0][:, 1 * 4 + 1, :, 1], x[:, 1, :, 5])


@pytest.mark.parametrize("regions", [((0, 96, 4), (96, 350, 16)), ((1, 96, 4),), ((0, 1000, 8),)])
def test_layout_rejects_invalid_regions(regions) -> None:
    with pytest.raises(ValueError):
        BandLayout(1025, regions)


@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_preset_budgets(preset: str) -> None:
    core = BandDualPathNPUCore(**PRESETS[preset])
    params = sum(p.numel() for p in core.parameters())
    gmacs = core.macs_per_frame() * FRAMES_PER_SECOND / 1e9
    io = core.io_size_bytes(dtype=torch.float16)
    assert params < 6_000_000
    assert gmacs < 3.0
    # All per-call inputs and outputs (frame, masks, state in and out) fit the 192 KiB DSP quota.
    assert io["total"] < 192 * 1024


def test_gru_conv_step_matches_torch_gru() -> None:
    torch.manual_seed(0)
    gru = torch.nn.GRU(12, 7, batch_first=True)
    x = torch.randn(2, 12, 1, 5)
    h = torch.randn(2, 7, 1, 5)
    ref, _ = gru(x.permute(0, 3, 2, 1).reshape(10, 1, 12), h.permute(2, 0, 3, 1).reshape(1, 10, 7))
    got = _gru_step_conv(gru, x, h)
    torch.testing.assert_close(got.permute(0, 3, 2, 1).reshape(10, 1, 7), ref, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("preset", ["medium", "medium_noattn"])
def test_streaming_matches_full_sequence_and_export_folding(preset: str) -> None:
    torch.manual_seed(0)
    model = BandDualPathNPUModel(**PRESETS[preset]).eval()
    _randomize_identity_params(model)
    spec = torch.randn(2, 1, 1025, 24, dtype=torch.complex64) * 3
    feats = model.host_features(spec)
    with torch.no_grad():
        full = model.core(feats)
        streamed = _stream(model.core, feats)
        folded = _stream(copy.deepcopy(model.core).prepare_for_export_(), feats)
    for a, b, c in zip(full, streamed, folded):
        torch.testing.assert_close(b, a, atol=2e-5, rtol=1e-4)
        torch.testing.assert_close(c, a, atol=2e-5, rtol=1e-4)
        assert a.abs().max() <= 1.0  # tanh-bounded complex masks


def test_chunked_forward_with_state_matches_single_pass() -> None:
    torch.manual_seed(0)
    model = BandDualPathNPUModel(**PRESETS["small"]).eval()
    _randomize_identity_params(model)
    feats = model.host_features(torch.randn(1, 1, 1025, 30, dtype=torch.complex64))
    with torch.no_grad():
        full = model.core(feats)
        first, state = model.core([f[:, :, :17] for f in feats], return_state=True)
        second = model.core([f[:, :, 17:] for f in feats], state=state)
    for whole, a, b in zip(full, first, second):
        torch.testing.assert_close(torch.cat((a, b), dim=2), whole, atol=2e-5, rtol=1e-4)


def test_training_wrapper_waveform_backward() -> None:
    torch.manual_seed(0)
    net = build_band_dualpath_npu_system(n_fft=2048, hop_length=512, fs=44100, preset="small").train()
    wav = torch.randn(2, 1, 44100) * 0.1
    est = net(wav)
    assert est.shape == (2, 3, 1, 44100)
    est.square().mean().backward()
    for name, param in net.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name


def test_mixture_consistency_restores_mixture() -> None:
    model = BandDualPathNPUModel(mixture_consistency=True, **PRESETS["small"]).eval()
    spec = torch.randn(1, 1, 1025, 6, dtype=torch.complex64)
    with torch.no_grad():
        est = model(spec)
    torch.testing.assert_close(est.sum(dim=1), spec, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("preset", ["medium", "medium_noattn"])
def test_onnx_export_passes_npu_audit(preset: str, tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxsim")
    ort = pytest.importorskip("onnxruntime")
    from tools.online.export_band_dualpath_npu import audit, export_onnx, simplify

    torch.manual_seed(0)
    model = BandDualPathNPUModel(**PRESETS[preset]).eval()
    _randomize_identity_params(model)
    core = copy.deepcopy(model.core).prepare_for_export_()
    wrapper = BandDualPathNPUExportWrapper(core).eval()
    raw, sim = tmp_path / "model.onnx", tmp_path / "model.sim.onnx"
    export_onnx(wrapper, raw, opset=13)
    simplify(raw, sim)
    report = audit(sim, core.io_size_bytes()["total"])
    assert report["violations"] == []
    assert report["forbidden_present"] == []
    assert report["nodes"] < 400
    n_inputs = len(core.layout.regions) + 2
    assert len(report["inputs"]) == n_inputs and len(report["outputs"]) == n_inputs
    if preset == "medium_noattn":
        assert not {"MatMul", "Softmax", "Transpose", "Reshape"} & set(report["ops"])

    sess = ort.InferenceSession(str(sim), providers=["CPUExecutionProvider"])
    inputs = wrapper.example_inputs(1)
    with torch.no_grad():
        ref = wrapper(*inputs)
    got = sess.run(None, {i.name: t.numpy() for i, t in zip(sess.get_inputs(), inputs)})
    for r, g in zip(ref, got):
        assert abs(r.numpy() - g).max() < 1e-4


def test_mean_free_rmsnorm_fallback_is_equivalent() -> None:
    torch.manual_seed(0)
    model = BandDualPathNPUModel(**PRESETS["medium"]).eval()
    _randomize_identity_params(model)
    feats = model.host_features(torch.randn(1, 1, 1025, 8, dtype=torch.complex64))
    with torch.no_grad():
        ref = _stream(model.core, feats)
        conv_core = copy.deepcopy(model.core).prepare_for_export_().set_rms_reduce_("conv")
        got = _stream(conv_core, feats)
    for a, b in zip(ref, got):
        torch.testing.assert_close(b, a, atol=2e-5, rtol=1e-4)
