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
    REGIONS_SR24K_R5,
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
FRAMES_PER_SECOND_24K = 24000 / 512


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
    # All per-call inputs and outputs (frame, masks, state in and out) fit the 192 KiB DSP quota
    # for fp16 and for the uint8 quantized ABI written into onecc.cfg by default.
    assert io["total"] < 192 * 1024
    assert core.io_size_bytes(dtype=torch.uint8)["total"] < 192 * 1024


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


def test_onecc_cfg_quantizes_the_boundary_by_default(tmp_path: Path) -> None:
    from tools.online.export_band_dualpath_npu import write_onecc_cfg

    text = write_onecc_cfg(tmp_path, tmp_path / "m.onnx", tmp_path / "c.h5", "channel").read_text()
    assert "input_type=uint8" in text and "output_type=uint8" in text
    # float32 boundary tensors would not fit the DSP quota for the medium preset.
    assert BandDualPathNPUCore(**PRESETS["medium"]).io_size_bytes(dtype=torch.float32)["total"] > 192 * 1024


# ---------------------------------------------------------------- 24 kHz "slots" I/O layout


def _slots_model(preset: str = "medium_noattn") -> BandDualPathNPUModel:
    return BandDualPathNPUModel(regions=REGIONS_SR24K_R5, io_layout="slots", mask_points=16, **PRESETS[preset]).eval()


def test_sr24k_layout_doubles_band_width_per_region() -> None:
    layout = BandLayout(1025, REGIONS_SR24K_R5)
    assert layout.band_counts == (24, 12, 12, 8, 6)
    assert layout.n_bands == 62 and layout.slot_width == 4 + 8 + 16 + 32 + 64
    widths = [r.width for r in layout.regions]
    assert all(b == 2 * a for a, b in zip(widths, widths[1:]))


def test_pack_slots_places_each_band_in_its_region_slot() -> None:
    layout = BandLayout(1025, REGIONS_SR24K_R5)
    x = torch.randn(2, 2, 3, 1025)
    slots = layout.pack_slots(x)
    parts = layout.pack(x)
    assert slots.shape == (2, 2 * 124, 3, 62)
    channel_offset = band_offset = 0
    for part in parts:
        rows = slice(channel_offset, channel_offset + part.shape[1])
        cols = slice(band_offset, band_offset + part.shape[3])
        assert torch.equal(slots[:, rows, :, cols], part)
        outside = slots[:, rows].clone()
        outside[..., cols] = 0
        assert torch.count_nonzero(outside) == 0  # a region's slot is zero for every other band
        channel_offset += part.shape[1]
        band_offset += part.shape[3]


def test_unpack_points_exact_for_narrow_bands_and_linear_for_wide_bands() -> None:
    layout = BandLayout(1025, REGIONS_SR24K_R5)
    points = torch.randn(1, 6 * 16, 2, 62)
    bins = layout.unpack_points(points, 6, 16)
    pts = points.reshape(1, 6, 16, 2, 62)
    assert bins.shape == (1, 6, 2, 1025)
    # Band 0 (W=4) and the first 16-bin band (bins 192..207, band 36) are copied exactly.
    torch.testing.assert_close(bins[..., 0:4], pts[:, :, :4, :, 0].permute(0, 1, 3, 2))
    torch.testing.assert_close(bins[..., 192:208], pts[:, :, :, :, 36].permute(0, 1, 3, 2))
    # First 64-bin band (bins 640..703, band 56): end points hit the first/last point, linear in between.
    torch.testing.assert_close(bins[..., 640], pts[:, :, 0, :, 56])
    torch.testing.assert_close(bins[..., 703], pts[:, :, 15, :, 56])
    pos = 21 * 15 / 63
    lo = int(pos)
    expected = (1 - (pos - lo)) * pts[:, :, lo, :, 56] + (pos - lo) * pts[:, :, lo + 1, :, 56]
    torch.testing.assert_close(bins[..., 640 + 21], expected)
    torch.testing.assert_close(bins[..., 1024], bins[..., 1023])  # Nyquist tail


def test_slots_embedding_equals_per_region_embeddings() -> None:
    torch.manual_seed(0)
    regions_core = BandDualPathNPUCore(regions=REGIONS_SR24K_R5, **PRESETS["medium_noattn"]).eval()
    slots_core = BandDualPathNPUCore(
        regions=REGIONS_SR24K_R5, io_layout="slots", mask_points=16, **PRESETS["medium_noattn"]
    ).eval()
    with torch.no_grad():
        slots_core.embed[0].weight.zero_()
        offset = 0
        for conv in regions_core.embed:
            slots_core.embed[0].weight[:, offset : offset + conv.in_channels] = conv.weight
            offset += conv.in_channels
        slots_core.embed[0].bias.zero_()
        region_bias = torch.cat(
            [conv.bias.view(1, -1, 1, 1).expand(1, -1, 1, r.n_bands) for conv, r in
             zip(regions_core.embed, regions_core.layout.regions)],
            dim=3,
        )
        slots_core.band_pos.copy_(regions_core.band_pos + region_bias)
        x = torch.randn(1, 2, 4, 1025)
        torch.testing.assert_close(
            slots_core._embed([slots_core.layout.pack_slots(x)]), regions_core._embed(regions_core.layout.pack(x)),
            atol=1e-5, rtol=1e-5,
        )


@pytest.mark.parametrize("preset", ["medium", "medium_noattn"])
def test_slots_streaming_matches_full_sequence_and_folding(preset: str) -> None:
    torch.manual_seed(0)
    model = _slots_model(preset)
    _randomize_identity_params(model)
    spec = torch.randn(1, 1, 1025, 16, dtype=torch.complex64) * 3
    feats = model.host_features(spec)
    assert len(feats) == 1 and feats[0].shape == (1, 248, 16, 62)
    with torch.no_grad():
        full = model.core(feats)
        streamed = _stream(model.core, feats)
        folded = _stream(copy.deepcopy(model.core).prepare_for_export_(), feats)
        est = model(spec)
    assert full[0].shape == (1, 96, 16, 62)
    torch.testing.assert_close(streamed[0], full[0], atol=2e-5, rtol=1e-4)
    torch.testing.assert_close(folded[0], full[0], atol=2e-5, rtol=1e-4)
    assert est.shape == (1, 3, 1, 1025, 16)


@pytest.mark.parametrize("preset", ["medium", "medium_noattn"])
def test_slots_sr24k_budgets(preset: str) -> None:
    core = _slots_model(preset).core
    assert sum(p.numel() for p in core.parameters()) < 6_000_000
    assert core.macs_per_frame() * FRAMES_PER_SECOND_24K / 1e9 < 3.0
    assert core.io_size_bytes(dtype=torch.uint8)["total"] < 192 * 1024


@pytest.mark.parametrize("preset,max_memory_ops", [("medium_noattn", 10), ("medium", 22)])
def test_slots_export_is_three_in_three_out_without_extra_memory_ops(
    preset: str, max_memory_ops: int, tmp_path: Path
) -> None:
    pytest.importorskip("onnxsim")
    ort = pytest.importorskip("onnxruntime")
    from tools.online.export_band_dualpath_npu import audit, export_onnx, simplify

    torch.manual_seed(0)
    model = _slots_model(preset)
    _randomize_identity_params(model)
    wrapper = BandDualPathNPUExportWrapper(copy.deepcopy(model.core).prepare_for_export_()).eval()
    raw, sim = tmp_path / "model.onnx", tmp_path / "model.sim.onnx"
    export_onnx(wrapper, raw, opset=13)
    simplify(raw, sim)
    report = audit(sim, 0)
    assert report["violations"] == [] and report["forbidden_present"] == []
    assert list(report["inputs"]) == ["spectrum", "band_state", "scene_state"]
    assert list(report["outputs"]) == ["mask_points", "next_band_state", "next_scene_state"]
    assert report["inputs"]["spectrum"] == [1, 248, 1, 62] and report["outputs"]["mask_points"] == [1, 96, 1, 62]
    assert report["memory_ops"] <= max_memory_ops
    assert "Split" not in report["ops"]  # no per-region split/concat around the embedding or heads

    sess = ort.InferenceSession(str(sim), providers=["CPUExecutionProvider"])
    inputs = wrapper.example_inputs(1)
    with torch.no_grad():
        ref = wrapper(*inputs)
    got = sess.run(None, {i.name: t.numpy() for i, t in zip(sess.get_inputs(), inputs)})
    for r, g in zip(ref, got):
        assert abs(r.numpy() - g).max() < 1e-4


def test_numpy_host_reference_matches_training_wrapper() -> None:
    import numpy as np

    from tools.online.band_dualpath_host_reference import HostTables, apply_masks, expand_masks, pack_frame

    torch.manual_seed(0)
    model = _slots_model()
    tables = HostTables.build(REGIONS_SR24K_R5, mask_points=16)
    spec = torch.randn(1, 1, 1025, 1, dtype=torch.complex64) * 2
    np.testing.assert_allclose(
        pack_frame(spec[0, :, :, 0].numpy(), tables), model.host_features(spec)[0].numpy(), atol=1e-5, rtol=1e-4
    )
    points = torch.randn(1, 96, 1, 62)
    ref = model.apply_masks(spec, (points,))[0, ..., 0].numpy()
    got = apply_masks(spec[0, :, :, 0].numpy(), expand_masks(points.numpy(), tables))
    np.testing.assert_allclose(got, ref, atol=1e-4, rtol=1e-4)
