import torch

from src.models.deft.uss_count_head import ForegroundCountHead
from src.models.deft.modified_deft import _match_query_condition_shape
from src.training.loss.uss_loss import get_loss_func
from src.models.s5.kwo2025 import Kwon2025S5


def _dummy_uss_output(batch_size=2, n_foreground=3, n_classes=18, samples=64):
    return {
        "foreground_waveform": torch.randn(batch_size, n_foreground, 1, samples),
        "interference_waveform": torch.randn(batch_size, 2, 1, samples),
        "noise_waveform": torch.randn(batch_size, 1, 1, samples),
        "class_logits": torch.randn(batch_size, n_foreground, n_classes),
        "silence_logits": torch.randn(batch_size, n_foreground),
    }


def _dummy_uss_target(batch_size=2, n_foreground=3, samples=64):
    return {
        "mixture": torch.randn(batch_size, 4, samples),
        "foreground_waveform": torch.randn(batch_size, n_foreground, 1, samples),
        "interference_waveform": torch.randn(batch_size, 2, 1, samples),
        "noise_waveform": torch.randn(batch_size, 1, 1, samples),
        "class_index": torch.randint(0, 18, (batch_size, n_foreground)),
        "is_silence": torch.tensor([[False, True, True], [False, False, True]]),
    }


def test_foreground_count_head_shape():
    output = _dummy_uss_output()
    head = ForegroundCountHead(n_foreground=3, n_classes=18, hidden_dim=16, max_count=3)
    count_logits = head(output)
    assert count_logits.shape == (2, 4)
    assert torch.isfinite(count_logits).all()


def test_uss_loss_accepts_optional_count_logits():
    output = _dummy_uss_output(samples=128)
    target = _dummy_uss_target(samples=128)
    head = ForegroundCountHead(n_foreground=3, n_classes=18, hidden_dim=16, max_count=3)
    output["count_logits"] = head(output)

    loss_func = get_loss_func(lambda_count=0.2, lambda_inactive_foreground=0.2)
    loss = loss_func(output, target)

    assert "loss_count" in loss
    assert torch.isfinite(loss["loss_count"])
    assert torch.isfinite(loss["loss"])


def test_uss_loss_accepts_capi_foreground_assignment_modes():
    for mode in ("soft_capi", "hard_capi"):
        output = _dummy_uss_output(samples=128)
        target = _dummy_uss_target(samples=128)
        head = ForegroundCountHead(n_foreground=3, n_classes=18, hidden_dim=16, max_count=3)
        output["count_logits"] = head(output)

        loss_func = get_loss_func(
            foreground_assignment=mode,
            capi_use_sdri=True,
            capi_confidence_threshold=0.0,
            capi_invalid_class_cost=10.0,
            lambda_class_pit=1.0,
            lambda_count=0.2,
            lambda_inactive_foreground=0.2,
        )
        loss = loss_func(output, target)

        assert torch.isfinite(loss["loss"])
        assert torch.isfinite(loss["loss_fg_match"])
        assert torch.isfinite(loss["loss_matched_target_class_nll"])
        assert torch.isfinite(loss["loss_matched_valid_pair_rate"])


def test_uss_loss_accepts_spatial_supervision_and_anticollapse_terms():
    output = _dummy_uss_output(samples=128)
    target = _dummy_uss_target(samples=128)
    target["class_index"] = torch.tensor([[0, 0, 1], [2, 2, 3]])
    target["is_silence"] = torch.tensor([[False, False, True], [False, False, True]])
    target["foreground_doa"] = torch.nn.functional.normalize(torch.randn(2, 3, 3), dim=-1)
    target["foreground_doa_mask"] = ~target["is_silence"]
    output["spatial_embedding"] = torch.nn.functional.normalize(torch.randn(2, 3, 8), dim=-1)
    output["doa_vector"] = torch.nn.functional.normalize(torch.randn(2, 3, 3), dim=-1)
    head = ForegroundCountHead(n_foreground=3, n_classes=18, hidden_dim=16, max_count=3)
    output["count_logits"] = head(output)

    loss_func = get_loss_func(
        foreground_assignment="soft_capi",
        capi_use_sdri=True,
        capi_confidence_threshold=0.0,
        capi_invalid_class_cost=10.0,
        lambda_class_pit=1.0,
        lambda_count=0.2,
        lambda_inactive_foreground=0.2,
        lambda_doa=0.1,
        lambda_spatial_diversity=0.05,
        lambda_waveform_anticollapse=0.02,
    )
    loss = loss_func(output, target)

    for key in ("loss", "loss_doa", "loss_spatial_diversity", "loss_waveform_anticollapse"):
        assert key in loss
        assert torch.isfinite(loss[key])


def test_kwon_uss_gate_suppresses_count_zero_slots():
    obj = object.__new__(Kwon2025S5)
    obj.uss_gate_enabled = True
    obj.uss_gate_count0_threshold = 0.65
    obj.uss_gate_slot_active_threshold = 0.45
    obj.uss_gate_slot_energy_threshold = 1e-4

    waveforms = torch.ones(2, 3, 1, 16)
    labels = [["Speech", "Dog", "Cat"], ["Speech", "Dog", "Cat"]]
    probs = torch.ones(2, 3)
    label_vector = torch.ones(2, 3, 18)
    uss_out = {
        "count_logits": torch.tensor([[5.0, 0.0, 0.0, 0.0], [0.0, 5.0, 0.0, 0.0]]),
        "silence_logits": torch.ones(2, 3),
    }

    gated_waveforms, gated_labels, gated_probs, gated_vectors = obj._apply_uss_gate_to_stage1(
        uss_out, waveforms, labels, probs, label_vector
    )

    assert torch.allclose(gated_waveforms[0], torch.zeros_like(gated_waveforms[0]))
    assert gated_labels[0] == ["silence", "silence", "silence"]
    assert torch.all(gated_probs[0] == 0)
    assert torch.all(gated_vectors[0] == 0)
    assert torch.all(gated_waveforms[1] != 0)


def test_tse_query_condition_shape_matching_pads_slots_and_features():
    label_vector = torch.zeros(2, 3, 18)
    query_condition = torch.ones(2, 2, 5)

    matched = _match_query_condition_shape(query_condition, label_vector, condition_dim=8)

    assert matched.shape == (2, 3, 8)
    assert torch.allclose(matched[:, :2, :5], torch.ones(2, 2, 5))
    assert torch.all(matched[:, 2] == 0)
    assert torch.all(matched[..., 5:] == 0)


def test_tse_query_condition_shape_matching_broadcasts_clip_condition():
    label_vector = torch.zeros(2, 3, 18)
    query_condition = torch.randn(2, 12)

    matched = _match_query_condition_shape(query_condition, label_vector, condition_dim=8)

    assert matched.shape == (2, 3, 8)
    assert torch.allclose(matched[:, 0], query_condition[:, :8])
    assert torch.allclose(matched[:, 1], query_condition[:, :8])
    assert torch.allclose(matched[:, 2], query_condition[:, :8])


def test_kwon_build_tse_query_condition_prefers_explicit_uss_condition():
    obj = object.__new__(Kwon2025S5)
    obj.tse_uss_conditioning_enabled = True

    stage_waveform = torch.randn(2, 3, 1, 32)
    explicit = torch.randn(2, 2, 7)
    condition = obj._build_tse_query_condition({"tse_condition": explicit}, stage_waveform)

    assert condition.shape == (2, 3, 7)
    assert torch.allclose(condition[:, :2], explicit)
    assert torch.all(condition[:, 2] == 0)


def test_kwon_build_tse_query_condition_broadcasts_clip_condition():
    obj = object.__new__(Kwon2025S5)
    obj.tse_uss_conditioning_enabled = True

    stage_waveform = torch.randn(2, 3, 1, 32)
    explicit = torch.randn(2, 7)
    condition = obj._build_tse_query_condition({"tse_condition": explicit}, stage_waveform)

    assert condition.shape == (2, 3, 7)
    assert torch.allclose(condition[:, 0], explicit)
    assert torch.allclose(condition[:, 1], explicit)
    assert torch.allclose(condition[:, 2], explicit)


def test_kwon_build_tse_query_condition_from_uss_outputs():
    obj = object.__new__(Kwon2025S5)
    obj.tse_uss_conditioning_enabled = True

    stage_waveform = torch.randn(2, 3, 1, 32)
    uss_out = {
        "class_logits": torch.randn(2, 3, 18),
        "silence_logits": torch.randn(2, 3),
        "count_logits": torch.randn(2, 4),
        "spatial_embedding": torch.randn(2, 3, 6),
        "foreground_activity_logits": torch.randn(2, 3, 5),
    }

    condition = obj._build_tse_query_condition(uss_out, stage_waveform)

    assert condition.shape == (2, 3, 32)
    assert torch.isfinite(condition).all()


def test_kwon_build_tse_query_condition_is_opt_in():
    obj = object.__new__(Kwon2025S5)
    obj.tse_uss_conditioning_enabled = False

    stage_waveform = torch.randn(2, 3, 1, 32)

    assert obj._build_tse_query_condition({"class_logits": torch.randn(2, 3, 18)}, stage_waveform) is None
