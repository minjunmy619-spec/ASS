import torch
import torch.nn.functional as F

from src.models.deft.unified_uss import UnifiedModifiedDeFTUSS
from src.models.deft.modified_deft import ModifiedDeFTTSEMemoryEfficientTemporal
from src.models.s5.kwo2025 import Kwon2025S5
from src.models.s5.kwo2025_temporal import Kwon2025TemporalS5
from src.training.lightningmodule.online_teacher_tse import OnlineTeacherTSELightning
from src.training.lightningmodule.uss_bridge import USSBridgeLightning
from src.training.loss.uss_bridge_loss import _doa_loss
from src.training.loss.uss_bridge_loss import get_loss_func


def _tiny_unified_uss():
    return UnifiedModifiedDeFTUSS(
        input_channels=4,
        output_channels=1,
        hidden_channels=8,
        n_deft_blocks=1,
        n_heads=1,
        n_foreground=3,
        n_interference=1,
        n_classes=18,
        window_size=64,
        hop_size=16,
        time_window_size=16,
        freq_group_size=16,
        shift_windows=False,
        sample_rate=16000,
        inference_chunk_seconds=None,
        enable_foa_spatial_features=True,
        enable_temporal_activity=True,
        enable_count_head=True,
        count_hidden_dim=16,
        max_count=3,
        enable_spatial_head=True,
        spatial_embedding_dim=8,
        enable_residual_slots=True,
        n_residual=1,
        enable_semantic_bridge=True,
        embedding_dim=16,
        prototype_scale=4.0,
        spatial_dim=3,
        tse_condition_dim=12,
    )


def _target(batch_size=2, samples=512):
    spans = torch.tensor(
        [
            [[0.0, 0.25], [0.1, 0.5], [-1.0, -1.0]],
            [[0.0, 0.5], [-1.0, -1.0], [-1.0, -1.0]],
        ],
        dtype=torch.float32,
    )
    foreground_doa = F.normalize(torch.randn(batch_size, 3, 3), dim=-1)
    is_silence = torch.tensor([[False, False, True], [False, True, True]])
    return {
        "mixture": torch.randn(batch_size, 4, samples),
        "foreground_waveform": torch.randn(batch_size, 3, 1, samples),
        "interference_waveform": torch.randn(batch_size, 1, 1, samples),
        "noise_waveform": torch.randn(batch_size, 1, 1, samples),
        "class_index": torch.randint(0, 18, (batch_size, 3)),
        "is_silence": is_silence,
        "foreground_span_sec": spans,
        "interference_span_sec": torch.tensor([[[0.0, 0.5]], [[-1.0, -1.0]]], dtype=torch.float32),
        "noise_span_sec": torch.tensor([[[0.0, 0.5]], [[0.0, 0.5]]], dtype=torch.float32),
        "foreground_doa": foreground_doa,
        "foreground_doa_mask": ~is_silence,
        "spatial_vector": foreground_doa,
    }


class _PretrainedSEDFusionPredictStub:
    def predict(self, input_dict):
        waveform = input_dict["waveform"]
        n_items = waveform.shape[0]
        class_indices = torch.arange(n_items, device=waveform.device) % 18
        labels = F.one_hot(class_indices, num_classes=18).float()
        return {
            "label_vector": labels,
            "raw_label_vector": labels.clone(),
            "class_indices": class_indices,
            "probabilities": torch.full((n_items,), 0.9, device=waveform.device),
            "energy": torch.zeros(n_items, device=waveform.device),
            "silence": torch.zeros(n_items, device=waveform.device, dtype=torch.bool),
        }


def _tiny_conditioned_tse(query_condition_dim=12):
    return ModifiedDeFTTSEMemoryEfficientTemporal(
        mixture_channels=4,
        enrollment_channels=1,
        hidden_channels=8,
        n_deft_blocks=1,
        n_heads=1,
        label_dim=18,
        window_size=64,
        hop_size=16,
        time_window_size=16,
        freq_group_size=16,
        shift_windows=False,
        sample_rate=16000,
        inference_chunk_seconds=None,
        query_condition_dim=query_condition_dim,
        query_condition_hidden_dim=16,
        require_query_condition=True,
    )


def test_unified_uss_emits_all_opt_in_feature_keys():
    model = _tiny_unified_uss()
    model.eval()
    mixture = torch.randn(2, 4, 512)

    with torch.no_grad():
        out = model({"mixture": mixture})

    assert out["foreground_waveform"].shape[:3] == (2, 3, 1)
    assert out["interference_waveform"].shape[:3] == (2, 1, 1)
    assert out["residual_waveform"].shape[:3] == (2, 1, 1)
    assert out["count_logits"].shape == (2, 4)
    assert out["foreground_activity_logits"].shape[:2] == (2, 3)
    assert out["spatial_embedding"].shape == (2, 3, 8)
    assert out["doa_vector"].shape == (2, 3, 3)
    assert out["pred_doa_vector"].shape == (2, 3, 3)
    assert out["prototype_logits"].shape == (2, 3, 18)
    assert out["tse_condition"].shape == (2, 3, 12)
    assert torch.isfinite(out["foreground_waveform"]).all()
    assert torch.isfinite(out["tse_condition"]).all()


def test_unified_uss_outputs_feed_fusion_sc_and_conditioned_tse_contracts():
    uss = _tiny_unified_uss().eval()
    tse = _tiny_conditioned_tse(query_condition_dim=12).eval()
    s5 = object.__new__(Kwon2025S5)
    torch.nn.Module.__init__(s5)
    s5.sc = _PretrainedSEDFusionPredictStub()
    s5.tse = tse
    s5.labels = [f"class_{idx}" for idx in range(18)]
    s5.duplicate_recall_enabled = False
    s5.tse_uss_conditioning_enabled = True
    mixture = torch.randn(2, 4, 512)

    with torch.no_grad():
        uss_out = uss({"mixture": mixture})
        enrollment = uss_out["foreground_waveform"]
        labels, probabilities, label_vector = s5._classify_sources(enrollment)
        query_condition = s5._build_tse_query_condition(uss_out, enrollment)
        tse_out = s5._run_tse(mixture, enrollment, label_vector, query_condition)

    assert enrollment.shape == (2, 3, 1, 512)
    assert label_vector.shape == (2, 3, 18)
    assert probabilities.shape == (2, 3)
    assert labels[0] == ["class_0", "class_1", "class_2"]
    assert query_condition.shape == (2, 3, 12)
    assert tse_out.shape == (2, 3, 1, 512)


def test_unified_uss_outputs_feed_temporal_s5_final_contract():
    uss = _tiny_unified_uss().eval()
    tse = _tiny_conditioned_tse(query_condition_dim=12).eval()
    s5 = object.__new__(Kwon2025TemporalS5)
    torch.nn.Module.__init__(s5)
    s5.uss = uss
    s5.sc = _PretrainedSEDFusionPredictStub()
    s5.tse = tse
    s5.labels = [f"class_{idx}" for idx in range(18)]
    s5.onehots = torch.eye(18)
    s5.duplicate_recall_enabled = False
    s5.tse_uss_conditioning_enabled = True
    s5.activity_threshold = 0.0
    s5.temporal_conditioning_enabled = True
    s5.activity_gating_enabled = False
    mixture = torch.randn(2, 4, 512)

    with torch.no_grad():
        output = s5.predict_label_separate(mixture)

    assert output["waveform"].shape == (2, 3, 1, 512)
    assert output["probabilities"].shape == (2, 3)
    assert output["query_condition"].shape == (2, 3, 12)
    assert len(output["label"]) == 2
    assert len(output["label"][0]) == 3


def test_online_teacher_accepts_unified_uss_and_fusion_sc_predict_contracts():
    module = object.__new__(OnlineTeacherTSELightning)
    torch.nn.Module.__init__(module)
    module.query_condition_enabled = True
    module.query_condition_key = None
    module.sc_model = _PretrainedSEDFusionPredictStub()
    uss = _tiny_unified_uss().eval()
    mixture = torch.randn(2, 4, 512)

    with torch.no_grad():
        uss_out = uss({"mixture": mixture})
        enrollment = uss_out["foreground_waveform"]
        query_condition = module._build_query_condition(uss_out, enrollment)
        sc_out = module._teacher_sc_predict(enrollment)

    assert query_condition.shape == (2, 3, 12)
    assert sc_out["label_vector"].shape == (2, 3, 18)
    assert sc_out["raw_label_vector"].shape == (2, 3, 18)
    assert sc_out["probabilities"].shape == (2, 3)


def test_unified_uss_with_features_disabled_keeps_base_output_contract():
    model = UnifiedModifiedDeFTUSS(
        input_channels=4,
        output_channels=1,
        hidden_channels=8,
        n_deft_blocks=1,
        n_heads=1,
        n_foreground=3,
        n_interference=1,
        n_classes=18,
        window_size=64,
        hop_size=16,
        time_window_size=16,
        freq_group_size=16,
        shift_windows=False,
        sample_rate=16000,
        inference_chunk_seconds=None,
    )
    model.eval()

    with torch.no_grad():
        out = model({"mixture": torch.randn(2, 4, 512)})

    assert {"foreground_waveform", "interference_waveform", "noise_waveform", "class_logits", "silence_logits"} <= set(out)
    for key in ("count_logits", "foreground_activity_logits", "spatial_embedding", "residual_waveform", "tse_condition"):
        assert key not in out


def test_unified_uss_chunked_eval_preserves_residual_activity_slots():
    model = UnifiedModifiedDeFTUSS(
        input_channels=4,
        output_channels=1,
        hidden_channels=8,
        n_deft_blocks=1,
        n_heads=1,
        n_foreground=3,
        n_interference=1,
        n_classes=18,
        window_size=64,
        hop_size=16,
        time_window_size=16,
        freq_group_size=16,
        shift_windows=False,
        sample_rate=16,
        inference_chunk_seconds=16,
        inference_chunk_hop_seconds=8,
        enable_temporal_activity=True,
        enable_residual_slots=True,
        n_residual=1,
    )
    model.eval()

    with torch.no_grad():
        out = model({"mixture": torch.randn(2, 4, 512)})

    assert out["interference_waveform"].shape[1] == 1
    assert out["residual_waveform"].shape[1] == 1
    assert out["interference_activity_logits"].shape[:2] == (2, 1)
    assert out["residual_activity_logits"].shape[:2] == (2, 1)


def test_bridge_doa_loss_ignores_unavailable_foreground_doa_targets():
    output = {
        "class_logits": torch.zeros(1, 2, 18),
        "pred_doa_vector": torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]),
    }
    target = {
        "spatial_vector": torch.tensor([[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]]),
        "foreground_doa_mask": torch.tensor([[True, False]]),
    }
    best_perm = torch.tensor([[0, 1]])
    active = torch.tensor([[True, True]])

    loss = _doa_loss(output, target, best_perm, active)

    assert torch.isclose(loss, torch.tensor(0.0))


def test_unified_uss_outputs_are_consumed_by_bridge_loss():
    model = _tiny_unified_uss()
    model.eval()
    target = _target()

    with torch.no_grad():
        out = model({"mixture": target["mixture"], "spatial_vector": target["spatial_vector"]})

    loss_func = get_loss_func(
        lambda_count=0.1,
        lambda_activity_foreground=0.01,
        lambda_activity_interference=0.01,
        lambda_activity_noise=0.01,
        lambda_doa=0.01,
        lambda_spatial_diversity=0.01,
        lambda_bridge_proto=0.01,
        lambda_bridge_supcon=0.01,
        lambda_bridge_infonce=0.01,
        lambda_bridge_doa=0.01,
        lambda_residual_slot=0.01,
        lambda_mix=0.01,
        residual_stft_fft_sizes=(64, 128),
    )
    loss = loss_func(out, target)

    for key in ("loss", "loss_bridge", "loss_residual_slot", "loss_mix", "loss_count"):
        assert key in loss
        assert torch.isfinite(loss[key])


def test_uss_bridge_lightning_uses_foreground_doa_as_spatial_vector_fallback():
    module = object.__new__(USSBridgeLightning)
    foreground_doa = torch.randn(2, 3, 3)
    batch = {
        "mixture": torch.randn(2, 4, 256),
        "foreground_waveform": torch.randn(2, 3, 1, 256),
        "interference_waveform": torch.randn(2, 1, 1, 256),
        "noise_waveform": torch.randn(2, 1, 1, 256),
        "class_index": torch.zeros(2, 3, dtype=torch.long),
        "is_silence": torch.zeros(2, 3, dtype=torch.bool),
        "foreground_doa": foreground_doa,
    }

    input_dict = module._get_input_dict(batch)
    target_dict = module._get_target_dict(batch)

    assert input_dict["spatial_vector"] is foreground_doa
    assert target_dict["spatial_vector"] is foreground_doa
