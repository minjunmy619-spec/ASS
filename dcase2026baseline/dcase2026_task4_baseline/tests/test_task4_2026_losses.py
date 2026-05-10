import sys
from types import MethodType, SimpleNamespace
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.s5.kwo2025 import Kwon2025S5
from src.models.s5.kwo2025_temporal import Kwon2025TemporalS5
from src.models.m2dat.m2d_sc import M2DSingleClassifierTemporalStrong, ArcMarginProduct
from src.models.deft.modified_deft import (
    ChunkedModifiedDeFTUSSSpatialTemporal,
    ModifiedDeFTTSEMemoryEfficientTemporal,
    ModifiedDeFTTSETemporal,
    ModifiedDeFTUSSSpatialTemporal,
    ModifiedDeFTUSSTemporal,
)
from src.models.deft.deft_tse import DeFTTSELikeSpatialTemporal
from src.datamodules.source_classifier_dataset import SourceClassifierDataset
from src.datamodules.tse_dataset import EstimatedEnrollmentTSEDataset
from src.temporal import event_to_span_sec, spans_to_frame_targets
from src.training.loss.class_aware_pit import class_aware_pit_loss
from src.training.loss.m2d_sc_arcface import get_loss_func as get_sc_loss_func
from src.training.loss.masked_snr import get_loss_func as get_tse_loss_func
from src.training.loss.uss_loss import get_loss_func as get_uss_loss_func
from src.training.lightningmodule.online_teacher_tse import OnlineTeacherTSELightning
from src.evaluation.metrics.s5capi_metric import (
    S5ClassAwareMetric,
    S5ClassAwareMetricAssignmentComparison,
    S5ClassAwareMetricSDRiAssignment,
)
from src.evaluation.calibrate_sc_energy_thresholds import choose_threshold


def pairwise_mse_loss(waveform_pred, waveform_target):
    pred = waveform_pred.flatten(start_dim=2)
    target = waveform_target.flatten(start_dim=2)
    return (pred.unsqueeze(1) - target.unsqueeze(2)).pow(2).mean(dim=-1)


class _DummyBaseDataset:
    n_sources = 3
    sr = 1
    collate_fn = None

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        dry_sources = torch.zeros(3, 1, 10)
        est_dry_sources = torch.zeros(3, 1, 10)
        dry_sources[0, 0, 0:1] = 1.0
        est_dry_sources[0, 0, 0:1] = 1.0
        dry_sources[1, 0, 3:5] = 1.0
        est_dry_sources[1, 0, 3:5] = 1.0
        label_vector = torch.eye(18)[:3]
        return {
            "mixture": torch.zeros(4, 10),
            "dry_sources": dry_sources,
            "est_dry_sources": est_dry_sources,
            "label": ["AlarmClock", "BicycleBell", "silence"],
            "est_label": ["AlarmClock", "BicycleBell", "silence"],
            "label_vector": label_vector,
            "est_label_vector": label_vector.clone(),
            "span_sec": torch.tensor([[0.0, 1.0], [3.0, 5.0], [-1.0, -1.0]]),
            "est_span_sec": torch.tensor([[0.0, 1.0], [3.0, 5.0], [-1.0, -1.0]]),
        }


class _DummyUSS(torch.nn.Module):
    def __init__(self, enrollment):
        super().__init__()
        self.enrollment = enrollment

    def forward(self, input_dict):
        batch_size, n_sources = self.enrollment.shape[:2]
        class_logits = self.enrollment.new_zeros(batch_size, n_sources, 4)
        class_logits[..., 0] = 2.0
        silence_logits = self.enrollment.new_ones(batch_size, n_sources)
        count_logits = self.enrollment.new_zeros(batch_size, 4)
        count_logits[:, 2] = 3.0
        return {
            "foreground_waveform": self.enrollment,
            "class_logits": class_logits,
            "silence_logits": silence_logits,
            "count_logits": count_logits,
        }


class _DummySC(torch.nn.Module):
    def __init__(self, label_vector):
        super().__init__()
        self.label_vector = label_vector

    def predict(self, input_dict):
        flat = self.label_vector.reshape(-1, self.label_vector.shape[-1])
        active = flat.abs().sum(dim=-1) > 0
        return {
            "label_vector": flat,
            "raw_label_vector": flat,
            "class_indices": torch.argmax(flat, dim=-1),
            "probabilities": active.float(),
            "energy": flat.new_zeros(flat.shape[0]),
            "silence": ~active,
        }


class _RecordingTwoPassTSE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enrollments = []

    def forward(self, input_dict):
        self.enrollments.append(input_dict["enrollment"].detach().clone())
        activity_logits = input_dict["enrollment"].new_zeros(input_dict["enrollment"].shape[:2] + (4,))
        return {
            "waveform": input_dict["enrollment"] + 0.5,
            "activity_logits": activity_logits,
        }


class _SecondPassSilentSC(torch.nn.Module):
    def __init__(self, first_label_vector, second_label_vector):
        super().__init__()
        self.calls = 0
        self.first_label_vector = first_label_vector
        self.second_label_vector = second_label_vector

    def predict(self, input_dict):
        self.calls += 1
        label_vector = self.first_label_vector if self.calls == 1 else self.second_label_vector
        flat = label_vector.reshape(-1, label_vector.shape[-1])
        active = flat.abs().sum(dim=-1) > 0
        return {
            "label_vector": flat,
            "raw_label_vector": flat,
            "class_indices": torch.argmax(flat, dim=-1),
            "probabilities": active.float(),
            "energy": flat.new_zeros(flat.shape[0]),
            "silence": ~active,
        }


def _simple_tse_loss(output_dict, target_dict):
    return {"loss": (output_dict["waveform"] - target_dict["waveform"]).abs().mean()}


def _online_teacher_shell():
    module = object.__new__(OnlineTeacherTSELightning)
    module.uss_output_key = "foreground_waveform"
    module.match_metric = "sa_sdr"
    module.min_match_score = -100.0
    module.min_estimate_energy_db = -100.0
    module.require_sc_active_for_loss = True
    module.require_sc_class_match_for_loss = False
    module.label_source = "sc"
    module.query_condition_enabled = True
    module.query_condition_key = None
    module.temporal_conditioning_source = "auto"
    module.tse_refinement_passes = 1
    module.second_pass_loss_weight = 1.0
    module.second_pass_detach_enrollment = True
    return module


def test_online_teacher_aligns_oracle_targets_to_uss_slots():
    module = _online_teacher_shell()
    target_waveform = torch.zeros(1, 3, 1, 8)
    target_waveform[0, 0, 0, 0:2] = 1.0
    target_waveform[0, 1, 0, 3:5] = 1.0
    enrollment = target_waveform[:, [1, 0, 2]].clone()
    label_vector = torch.zeros(1, 3, 4)
    label_vector[0, 0, 0] = 1.0
    label_vector[0, 1, 1] = 1.0

    aligned = module._align_oracle_to_estimate_slots(
        enrollment,
        {
            "waveform": target_waveform,
            "label_vector": label_vector,
            "active_mask": torch.tensor([[True, True, False]]),
            "span_sec": torch.tensor([[[0.0, 0.25], [0.375, 0.625], [-1.0, -1.0]]]),
        },
    )

    assert aligned["active_mask"].tolist() == [[True, True, False]]
    assert torch.allclose(aligned["waveform"][0, 0], target_waveform[0, 1])
    assert torch.allclose(aligned["waveform"][0, 1], target_waveform[0, 0])
    assert aligned["label_vector"][0, 0, 1] == 1.0
    assert aligned["label_vector"][0, 1, 0] == 1.0


def test_online_teacher_builds_tse_inputs_from_frozen_uss_and_sc():
    module = _online_teacher_shell()
    target_waveform = torch.zeros(1, 3, 1, 8)
    target_waveform[0, 0, 0, 0:2] = 1.0
    target_waveform[0, 1, 0, 3:5] = 1.0
    enrollment = target_waveform[:, [1, 0, 2]].clone()
    sc_labels = torch.zeros(1, 3, 4)
    sc_labels[0, 0, 1] = 1.0
    sc_labels[0, 1, 0] = 1.0
    label_vector = torch.zeros(1, 3, 4)
    label_vector[0, 0, 0] = 1.0
    label_vector[0, 1, 1] = 1.0
    object.__setattr__(module, "uss_model", _DummyUSS(enrollment))
    object.__setattr__(module, "sc_model", _DummySC(sc_labels))

    input_dict, target_dict, diagnostics = module._build_teacher_batch({
        "mixture": torch.zeros(1, 4, 8),
        "waveform": target_waveform,
        "label_vector": label_vector,
        "active_mask": torch.tensor([[True, True, False]]),
        "span_sec": torch.tensor([[[0.0, 0.25], [0.375, 0.625], [-1.0, -1.0]]]),
    })

    assert torch.allclose(input_dict["enrollment"], enrollment)
    assert torch.allclose(input_dict["label_vector"], sc_labels)
    assert input_dict["query_condition"].shape[:2] == (1, 3)
    assert target_dict["active_mask"].tolist() == [[True, True, False]]
    assert torch.allclose(target_dict["waveform"][0, 0], target_waveform[0, 1])
    assert target_dict["label_vector"][0, 0, 1] == 1.0
    assert diagnostics["teacher_matched_slots"].item() == 2.0


def test_online_teacher_two_pass_uses_first_tse_output_as_second_enrollment():
    module = _online_teacher_shell()
    module.tse_refinement_passes = 2
    module.second_pass_loss_weight = 0.25
    module.metric_func = None
    target_waveform = torch.zeros(1, 3, 1, 8)
    target_waveform[0, 0, 0, 0:2] = 1.0
    target_waveform[0, 1, 0, 3:5] = 1.0
    enrollment = target_waveform[:, [1, 0, 2]].clone()
    sc_labels = torch.zeros(1, 3, 4)
    sc_labels[0, 0, 1] = 1.0
    sc_labels[0, 1, 0] = 1.0
    label_vector = torch.zeros(1, 3, 4)
    label_vector[0, 0, 0] = 1.0
    label_vector[0, 1, 1] = 1.0
    tse = _RecordingTwoPassTSE()
    object.__setattr__(module, "model", tse)
    object.__setattr__(module, "loss_func", _simple_tse_loss)
    object.__setattr__(module, "uss_model", _DummyUSS(enrollment))
    object.__setattr__(module, "sc_model", _DummySC(sc_labels))

    loss_dict = module._step(
        {
            "mixture": torch.zeros(1, 4, 8),
            "waveform": target_waveform,
            "label_vector": label_vector,
            "active_mask": torch.tensor([[True, True, False]]),
            "span_sec": torch.tensor([[[0.0, 0.25], [0.375, 0.625], [-1.0, -1.0]]]),
        },
        "train",
    )

    assert len(tse.enrollments) == 2
    assert torch.allclose(tse.enrollments[0], enrollment)
    assert torch.allclose(tse.enrollments[1], enrollment + 0.5)
    assert "first_pass_loss" in loss_dict
    assert "second_pass_loss" in loss_dict
    assert loss_dict["teacher_second_pass_enabled"].item() == 1.0


def test_online_teacher_two_pass_masks_second_loss_with_second_sc_output():
    module = _online_teacher_shell()
    module.tse_refinement_passes = 2
    module.metric_func = None
    target_waveform = torch.zeros(1, 3, 1, 8)
    target_waveform[0, 0, 0, 0:2] = 1.0
    target_waveform[0, 1, 0, 3:5] = 1.0
    enrollment = target_waveform[:, [1, 0, 2]].clone()
    first_sc_labels = torch.zeros(1, 3, 4)
    first_sc_labels[0, 0, 1] = 1.0
    first_sc_labels[0, 1, 0] = 1.0
    second_sc_labels = first_sc_labels.clone()
    second_sc_labels[0, 1] = 0.0
    label_vector = torch.zeros(1, 3, 4)
    label_vector[0, 0, 0] = 1.0
    label_vector[0, 1, 1] = 1.0
    object.__setattr__(module, "uss_model", _DummyUSS(enrollment))
    object.__setattr__(module, "sc_model", _SecondPassSilentSC(first_sc_labels, second_sc_labels))
    first_input, target_dict, _ = module._build_teacher_batch(
        {
            "mixture": torch.zeros(1, 4, 8),
            "waveform": target_waveform,
            "label_vector": label_vector,
            "active_mask": torch.tensor([[True, True, False]]),
            "span_sec": torch.tensor([[[0.0, 0.25], [0.375, 0.625], [-1.0, -1.0]]]),
        }
    )
    first_output = {"waveform": enrollment + 0.5}

    _, second_target_dict, diagnostics = module._build_second_pass_input(first_input, first_output, target_dict)

    assert second_target_dict["active_mask"].tolist() == [[True, False, False]]
    assert torch.allclose(second_target_dict["waveform"][0, 1], torch.zeros_like(second_target_dict["waveform"][0, 1]))
    assert diagnostics["teacher_second_pass_matched_slots"].item() == 1.0


def test_estimated_enrollment_crop_deactivates_out_of_window_sources():
    dataset = EstimatedEnrollmentTSEDataset(
        _DummyBaseDataset(),
        crop_seconds=6.0,
        random_crop=False,
        require_estimate_for_active=True,
    )

    item = dataset[0]

    assert item["active_mask"].tolist() == [False, True, False]
    assert item["waveform"][0].abs().sum() == 0
    assert item["label_vector"][0].abs().sum() == 0
    assert item["label_vector"][1].abs().sum() > 0


def test_temporal_sc_eval_activity_is_stitched_across_crops():
    model = object.__new__(M2DSingleClassifierTemporalStrong)
    activity_chunks = [
        torch.ones(1, 5),
        torch.ones(1, 5) * 2.0,
        torch.ones(1, 5) * 3.0,
    ]

    stitched = model._stitch_eval_activity(
        activity_chunks,
        crop_starts=[0, 2, 5],
        total_samples=10,
        crop_samples=5,
    )

    assert stitched.shape == (1, 10)
    assert torch.allclose(stitched[:, :2], torch.ones(1, 2))
    assert torch.allclose(stitched[:, 2:5], torch.full((1, 3), 1.5))
    assert torch.allclose(stitched[:, 5:7], torch.full((1, 2), 2.5))
    assert torch.allclose(stitched[:, 7:], torch.full((1, 3), 3.0))


def test_duplicate_same_class_target_swap_gives_low_loss():
    pred = torch.tensor([[[[0.0, 1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 1.0, 0.0]]]])
    target = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]], [[0.0, 1.0, 0.0, 0.0]], [[0.0, 0.0, 1.0, 0.0]]]])
    label = torch.zeros(1, 3, 4)
    label[0, 0, 1] = 1.0
    label[0, 1, 1] = 1.0
    label[0, 2, 2] = 1.0

    loss, best_perm, _ = class_aware_pit_loss(pred, target, label, pairwise_loss_func=pairwise_mse_loss)

    assert torch.allclose(loss, torch.zeros_like(loss))
    assert best_perm[0].tolist() == [1, 0, 2]


def test_distinct_labels_cannot_swap():
    pred = torch.tensor([[[[0.0, 1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]]])
    target = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]], [[0.0, 1.0, 0.0, 0.0]]]])
    label = torch.zeros(1, 2, 4)
    label[0, 0, 1] = 1.0
    label[0, 1, 2] = 1.0

    loss, best_perm, _ = class_aware_pit_loss(pred, target, label, pairwise_loss_func=pairwise_mse_loss)

    assert loss.item() > 0.1
    assert best_perm[0].tolist() == [0, 1]


def test_all_silence_tse_target_has_nonzero_output_gradient():
    loss_func = get_tse_loss_func(lambda_inactive=1.0)
    output = {"waveform": torch.randn(2, 3, 1, 32, requires_grad=True)}
    target = {
        "waveform": torch.zeros(2, 3, 1, 32),
        "label_vector": torch.zeros(2, 3, 4),
        "active_mask": torch.zeros(2, 3, dtype=torch.bool),
    }

    loss = loss_func(output, target)["loss"]
    loss.backward()

    assert loss.item() > 0.0
    assert output["waveform"].grad.abs().sum().item() > 0.0


def test_capi_raw_and_sdri_assignment_match_for_equal_same_class_counts():
    raw_metric = S5ClassAwareMetric()
    sdri_metric = S5ClassAwareMetricSDRiAssignment()
    ref = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    est = torch.tensor([[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    mixture = ref.sum(dim=0)

    raw_output = raw_metric._pi_metric(est_wf=est, ref_wf=ref, mixture=mixture)
    sdri_output = sdri_metric._pi_metric(est_wf=est, ref_wf=ref, mixture=mixture)

    assert raw_output["ref_perm"] == sdri_output["ref_perm"]
    assert raw_output["est_perm"] == sdri_output["est_perm"]
    assert torch.allclose(raw_output["metric_i"], sdri_output["metric_i"], equal_nan=True)


def test_capi_comparison_exposes_subset_selection_difference():
    def fake_metric(pred, target):
        if torch.all(pred == 0):
            return target.new_tensor([[0.0], [100.0]])
        return target.new_tensor([[10.0], [20.0]])

    est = torch.tensor([[1.0]])
    ref = torch.tensor([[1.0], [2.0]])
    mixture = torch.zeros(1)

    raw_metric = S5ClassAwareMetric()
    raw_metric.metric_func = fake_metric
    raw_output = raw_metric._pi_metric(est_wf=est, ref_wf=ref, mixture=mixture)
    assert raw_output["ref_perm"] == [1]
    sdri_metric = S5ClassAwareMetricSDRiAssignment()
    sdri_metric.metric_func = fake_metric
    sdri_output = sdri_metric._pi_metric(est_wf=est, ref_wf=ref, mixture=mixture)
    assert sdri_output["ref_perm"] == [0]

    comparison = S5ClassAwareMetricAssignmentComparison()
    comparison.metric_func = fake_metric
    value = comparison.compute_sample(
        est_lb=["dog"],
        est_wf=est,
        ref_lb=["dog", "dog"],
        ref_wf=ref,
        mixture=mixture,
    )
    assert value["raw_sdr_assignment"] == -40.0
    assert value["sdri_assignment"] == 5.0
    assert value["delta_sdri_minus_raw"] == 45.0


class DummyUSS(torch.nn.Module):
    def forward(self, input_dict):
        mixture = input_dict["mixture"]
        return {"foreground_waveform": torch.ones(mixture.shape[0], 3, 1, mixture.shape[-1], device=mixture.device)}


class SilentSC(torch.nn.Module):
    def predict(self, input_dict):
        batch = input_dict["waveform"].shape[0]
        return {
            "label_vector": torch.zeros(batch, 18, device=input_dict["waveform"].device),
            "probabilities": torch.zeros(batch, device=input_dict["waveform"].device),
        }


class DuplicateCandidateSC(torch.nn.Module):
    def predict(self, input_dict):
        batch = input_dict["waveform"].shape[0]
        label_vector = torch.zeros(batch, 18, device=input_dict["waveform"].device)
        raw_label_vector = torch.zeros_like(label_vector)
        probabilities = torch.full((batch,), 0.1, device=input_dict["waveform"].device)

        for start in range(0, batch, 3):
            label_vector[start, 1] = 1.0
            raw_label_vector[start, 1] = 1.0
            probabilities[start] = 0.9

            raw_label_vector[start + 1, 1] = 1.0
            probabilities[start + 1] = 0.8

            raw_label_vector[start + 2, 2] = 1.0
            probabilities[start + 2] = 0.8

        return {
            "label_vector": label_vector,
            "raw_label_vector": raw_label_vector,
            "probabilities": probabilities,
        }


class DuplicateCandidateTemporalSC(DuplicateCandidateSC):
    def __init__(self, slot_support):
        super().__init__()
        self.slot_support = slot_support

    def predict(self, input_dict):
        out = super().predict(input_dict)
        batch = input_dict["waveform"].shape[0]
        support = torch.tensor(self.slot_support, device=input_dict["waveform"].device, dtype=torch.float32)
        support = support.repeat(batch // len(self.slot_support))
        out["activity_probabilities"] = support[:, None].expand(-1, 4)
        return out


class FailingTSE(torch.nn.Module):
    def forward(self, input_dict):
        raise AssertionError("TSE should be skipped when all sources are predicted as silence")


class HallucinatingTSE(torch.nn.Module):
    def forward(self, input_dict):
        return {"waveform": torch.ones_like(input_dict["enrollment"]) * 7.0}


class RecordingTemporalTSE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conditioning = []

    def forward(self, input_dict):
        self.conditioning.append(input_dict.get("temporal_conditioning"))
        activity = input_dict.get("temporal_conditioning")
        if activity is None:
            activity = torch.ones(input_dict["enrollment"].shape[:2] + (4,), device=input_dict["enrollment"].device)
        logits = torch.logit(torch.clamp(activity, min=1e-4, max=1.0 - 1e-4))
        return {
            "waveform": torch.ones_like(input_dict["enrollment"]) * 7.0,
            "activity_logits": logits,
        }


class PartialThenHallucinatingSC(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def predict(self, input_dict):
        self.calls += 1
        batch = input_dict["waveform"].shape[0]
        label_vector = torch.zeros(batch, 18, device=input_dict["waveform"].device)
        probabilities = torch.zeros(batch, device=input_dict["waveform"].device)
        for start in range(0, batch, 3):
            if self.calls == 1:
                label_vector[start, 1] = 1.0
                probabilities[start] = 0.9
            else:
                label_vector[start : start + 3, 1] = 1.0
                probabilities[start : start + 3] = 0.9
        return {
            "label_vector": label_vector,
            "probabilities": probabilities,
        }


class TemporalSupportSC(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def predict(self, input_dict):
        self.calls += 1
        batch = input_dict["waveform"].shape[0]
        label_vector = torch.zeros(batch, 18, device=input_dict["waveform"].device)
        probabilities = torch.full((batch,), 0.9, device=input_dict["waveform"].device)
        activity = torch.full((batch, 4), 0.05, device=input_dict["waveform"].device)
        for start in range(0, batch, 3):
            label_vector[start : start + 3, 1] = 1.0
            activity[start] = 0.95
        return {
            "label_vector": label_vector,
            "raw_label_vector": label_vector.clone(),
            "probabilities": probabilities,
            "activity_probabilities": activity,
        }


def test_s5_returns_all_silence_without_tse_hallucination():
    model = Kwon2025S5.__new__(Kwon2025S5)
    torch.nn.Module.__init__(model)
    model.uss = DummyUSS()
    model.sc = SilentSC()
    model.tse = FailingTSE()
    model.labels = [f"class_{idx}" for idx in range(18)]
    model.onehots = torch.eye(18)

    output = model.predict_label_separate(torch.zeros(2, 4, 64))

    assert output["label"] == [["silence", "silence", "silence"], ["silence", "silence", "silence"]]
    assert torch.allclose(output["waveform"], torch.zeros_like(output["waveform"]))
    assert torch.allclose(output["probabilities"], torch.zeros_like(output["probabilities"]))


def test_s5_forces_partially_silent_slots_after_tse_hallucination():
    model = Kwon2025S5.__new__(Kwon2025S5)
    torch.nn.Module.__init__(model)
    model.uss = DummyUSS()
    model.sc = PartialThenHallucinatingSC()
    model.tse = HallucinatingTSE()
    model.labels = [f"class_{idx}" for idx in range(18)]
    model.onehots = torch.eye(18)
    model.duplicate_recall_enabled = False

    output = model.predict_label_separate(torch.zeros(1, 4, 64))

    assert output["label"] == [["class_1", "silence", "silence"]]
    assert output["probabilities"][0, 0].item() == pytest.approx(0.9)
    assert output["probabilities"][0, 1:].sum().item() == 0.0
    assert output["waveform"][0, 0].abs().sum().item() > 0.0
    assert output["waveform"][0, 1:].abs().sum().item() == 0.0


def test_s5_duplicate_recall_recovers_silenced_same_class_slot_only_when_enabled():
    model = Kwon2025S5.__new__(Kwon2025S5)
    torch.nn.Module.__init__(model)
    model.sc = DuplicateCandidateSC()
    model.labels = [f"class_{idx}" for idx in range(18)]
    model.duplicate_recall_min_probability = 0.35
    model.duplicate_recall_min_waveform_rms = 1e-4

    waveforms = torch.ones(1, 3, 1, 32)
    model.duplicate_recall_enabled = False
    labels, _, label_vector = model._classify_sources(waveforms)
    assert labels == [["class_1", "silence", "silence"]]
    assert label_vector[0, 1].sum().item() == 0.0

    model.duplicate_recall_enabled = True
    labels, _, label_vector = model._classify_sources(waveforms)
    assert labels == [["class_1", "class_1", "silence"]]
    assert label_vector[0, 1, 1].item() == 1.0
    assert label_vector[0, 2].sum().item() == 0.0


def test_temporal_s5_duplicate_recall_requires_temporal_support():
    model = Kwon2025TemporalS5.__new__(Kwon2025TemporalS5)
    torch.nn.Module.__init__(model)
    model.labels = [f"class_{idx}" for idx in range(18)]
    model.duplicate_recall_enabled = True
    model.duplicate_recall_min_probability = 0.35
    model.duplicate_recall_min_waveform_rms = 1e-4
    model.activity_threshold = 0.5

    waveforms = torch.ones(1, 3, 1, 32)
    model.sc = DuplicateCandidateTemporalSC([1.0, 0.0, 1.0])
    labels, _, label_vector, _ = model._classify_sources_temporal(waveforms)
    assert labels == [["class_1", "silence", "silence"]]
    assert label_vector[0, 1].sum().item() == 0.0

    model.sc = DuplicateCandidateTemporalSC([1.0, 1.0, 1.0])
    labels, _, label_vector, _ = model._classify_sources_temporal(waveforms)
    assert labels == [["class_1", "class_1", "silence"]]
    assert label_vector[0, 1, 1].item() == 1.0


def test_temporal_s5_gates_inactive_slots_and_conditions_tse():
    class TemporalUSS(torch.nn.Module):
        def forward(self, input_dict):
            mixture = input_dict["mixture"]
            activity = torch.tensor([[[4.0, 4.0, 4.0, 4.0], [-4.0, -4.0, -4.0, -4.0], [-4.0, -4.0, -4.0, -4.0]]])
            return {
                "foreground_waveform": torch.ones(mixture.shape[0], 3, 1, mixture.shape[-1], device=mixture.device),
                "foreground_activity_logits": activity.to(mixture.device),
            }

    model = Kwon2025TemporalS5.__new__(Kwon2025TemporalS5)
    torch.nn.Module.__init__(model)
    model.uss = TemporalUSS()
    model.sc = TemporalSupportSC()
    model.tse = RecordingTemporalTSE()
    model.labels = [f"class_{idx}" for idx in range(18)]
    model.onehots = torch.eye(18)
    model.duplicate_recall_enabled = False
    model.activity_threshold = 0.5
    model.temporal_conditioning_enabled = True
    model.activity_gating_enabled = True

    output = model.predict_label_separate(torch.zeros(1, 4, 64))

    assert output["label"] == [["class_1", "silence", "silence"]]
    assert output["waveform"][0, 0].abs().sum().item() > 0.0
    assert output["waveform"][0, 1:].abs().sum().item() == 0.0
    assert len(model.tse.conditioning) == 2
    assert model.tse.conditioning[0].shape == (1, 3, 4)
    assert model.tse.conditioning[0][0, 0].amin().item() > 0.5
    assert model.tse.conditioning[0][0, 1:].amax().item() < 0.5


def test_temporal_s5_can_stop_after_one_tse_refinement_pass():
    class TemporalUSS(torch.nn.Module):
        def forward(self, input_dict):
            mixture = input_dict["mixture"]
            activity = torch.tensor([[[4.0, 4.0, 4.0, 4.0], [-4.0, -4.0, -4.0, -4.0], [-4.0, -4.0, -4.0, -4.0]]])
            return {
                "foreground_waveform": torch.ones(mixture.shape[0], 3, 1, mixture.shape[-1], device=mixture.device),
                "foreground_activity_logits": activity.to(mixture.device),
            }

    model = Kwon2025TemporalS5.__new__(Kwon2025TemporalS5)
    torch.nn.Module.__init__(model)
    model.uss = TemporalUSS()
    model.sc = TemporalSupportSC()
    model.tse = RecordingTemporalTSE()
    model.labels = [f"class_{idx}" for idx in range(18)]
    model.onehots = torch.eye(18)
    model.duplicate_recall_enabled = False
    model.activity_threshold = 0.5
    model.temporal_conditioning_enabled = True
    model.activity_gating_enabled = True
    model.tse_refinement_passes = 1

    output = model.predict_label_separate(torch.zeros(1, 4, 64))

    assert output["label"] == [["class_1", "silence", "silence"]]
    assert len(model.tse.conditioning) == 1


def test_energy_threshold_calibration_prefers_low_energy_true_positives():
    best = choose_threshold(
        energies=[-8.0, -7.5, -2.0, -1.5],
        positives=[True, True, False, False],
        beta=0.5,
    )

    assert best["threshold"] == pytest.approx(-7.5)
    assert best["tp"] == 2
    assert best["fp"] == 0
    assert best["fn"] == 0
    assert best["precision"] == pytest.approx(1.0)
    assert best["recall"] == pytest.approx(1.0)


def test_event_span_targets_keep_silence_inactive():
    span = event_to_span_sec({"metadata": {"event_time": 1.0, "event_duration": 2.0}})
    assert span == (1.0, 3.0)

    targets = spans_to_frame_targets(
        torch.tensor([[1.0, 3.0], [-1.0, -1.0]]),
        num_frames=5,
        duration_sec=torch.tensor([4.0, 4.0]),
    )
    assert targets[0].tolist() == [0.0, 1.0, 1.0, 1.0, 0.0]
    assert targets[1].sum().item() == 0.0


class FakeSCBaseDataset(torch.utils.data.Dataset):
    n_sources = 2
    labels = ["dog", "alarm"]
    sr = 4
    collate_fn = None

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {
            "dry_sources": torch.tensor([[[0.0, 1.0, 1.0, 0.0]], [[0.0, 0.0, 0.0, 0.0]]]),
            "label": ["dog", "silence"],
            "span_sec": torch.tensor([[0.25, 0.75], [-1.0, -1.0]]),
        }


class FakeDuplicateSCBaseDataset(FakeSCBaseDataset):
    n_sources = 3

    def __getitem__(self, idx):
        return {
            "dry_sources": torch.tensor([
                [[0.0, 1.0, 1.0, 0.0]],
                [[1.0, 0.0, 0.0, 1.0]],
                [[0.0, 0.0, 0.0, 0.0]],
            ]),
            "label": ["dog", "dog", "silence"],
            "span_sec": torch.tensor([[0.25, 0.75], [0.0, 1.0], [-1.0, -1.0]]),
        }


class FakeEstimatedEnrollmentBaseDataset(torch.utils.data.Dataset):
    n_sources = 3
    sr = 4
    collate_fn = None

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        label_vector = torch.zeros(3, 4)
        label_vector[0, 1] = 1.0
        label_vector[1, 1] = 1.0
        est_label_vector = label_vector.clone()
        est_label_vector[1] = 0.0
        return {
            "mixture": torch.arange(40, dtype=torch.float32).view(1, 40),
            "dry_sources": torch.stack([
                torch.ones(1, 40),
                torch.ones(1, 40) * 2.0,
                torch.zeros(1, 40),
            ]),
            "est_dry_sources": torch.stack([
                torch.ones(1, 40) * 10.0,
                torch.zeros(1, 40),
                torch.zeros(1, 40),
            ]),
            "label": ["dog", "dog", "silence"],
            "est_label": ["dog", "silence", "silence"],
            "label_vector": label_vector,
            "est_label_vector": est_label_vector,
            "span_sec": torch.tensor([[0.0, 5.0], [2.0, 8.0], [-1.0, -1.0]]),
            "est_span_sec": torch.tensor([[0.0, 5.0], [-1.0, -1.0], [-1.0, -1.0]]),
        }


def test_source_classifier_dataset_emits_span_sec():
    dataset = SourceClassifierDataset(FakeSCBaseDataset())
    batch = dataset._collate_fn([dataset[0], dataset[1]])

    assert torch.allclose(batch["span_sec"][0], torch.tensor([0.25, 0.75]))
    assert torch.allclose(batch["span_sec"][1], torch.tensor([-1.0, -1.0]))


def test_source_classifier_dataset_marks_duplicate_classes():
    dataset = SourceClassifierDataset(FakeDuplicateSCBaseDataset())
    batch = dataset._collate_fn([dataset[0], dataset[1], dataset[2]])

    assert batch["duplicate_class_count"].tolist() == [2, 2, 0]
    assert batch["is_duplicate_class"].tolist() == [True, True, False]


def test_estimated_enrollment_tse_dataset_uses_estimates_as_enrollment_and_masks_missing_slots():
    dataset = EstimatedEnrollmentTSEDataset(
        FakeEstimatedEnrollmentBaseDataset(),
        crop_seconds=6.0,
        random_crop=False,
        require_estimate_for_active=True,
    )

    item = dataset[0]

    assert item["mixture"].shape[-1] == 24
    assert torch.allclose(item["enrollment"][0], torch.ones(1, 24) * 10.0)
    assert torch.allclose(item["waveform"][0], torch.ones(1, 24))
    assert item["active_mask"].tolist() == [True, False, False]
    assert item["label_vector"][0, 1].item() == 1.0
    assert item["label_vector"][1].sum().item() == 0.0
    assert item["waveform"][1].sum().item() == 0.0
    assert torch.allclose(item["span_sec"][0], torch.tensor([0.0, 3.0]))
    assert torch.allclose(item["span_sec"][1], torch.tensor([0.0, 6.0]))


def test_previous_sc_loss_ignores_new_span_field_by_default():
    output = {
        "logits": torch.randn(3, 4, requires_grad=True),
        "plain_logits": torch.randn(3, 4, requires_grad=True),
        "energy": torch.randn(3, requires_grad=True),
    }
    target = {
        "class_index": torch.tensor([1, 2, 0]),
        "is_silence": torch.tensor([False, False, True]),
        "span_sec": torch.tensor([[0.0, 1.0], [0.5, 1.5], [-1.0, -1.0]]),
    }
    loss = get_sc_loss_func(lambda_energy=0.0)(output, target)["loss"]
    loss.backward()
    assert torch.isfinite(loss)


def test_sc_duplicate_recall_loss_pushes_duplicate_active_energy():
    output = {
        "logits": torch.randn(3, 4, requires_grad=True),
        "plain_logits": torch.randn(3, 4, requires_grad=True),
        "energy": torch.tensor([-5.0, -5.0, 0.0], requires_grad=True),
    }
    target = {
        "class_index": torch.tensor([1, 1, 0]),
        "is_silence": torch.tensor([False, False, True]),
        "is_duplicate_class": torch.tensor([True, True, False]),
    }
    loss_dict = get_sc_loss_func(lambda_duplicate_recall=0.5, duplicate_m_in=-8.0)(output, target)
    loss = loss_dict["loss"]
    loss.backward()

    assert loss_dict["loss_duplicate_recall"].item() > 0.0
    assert output["energy"].grad[:2].gt(0).all()
    assert output["energy"].grad[2].item() == 0.0


def test_sc_loss_truncation_downweights_high_loss_active_labels():
    logits = torch.tensor(
        [
            [8.0, -8.0, -8.0],
            [-8.0, 8.0, -8.0],
            [8.0, -8.0, -8.0],
        ],
        requires_grad=True,
    )
    output = {
        "logits": logits,
        "plain_logits": logits,
        "energy": torch.zeros(3, requires_grad=True),
    }
    target = {
        "class_index": torch.tensor([0, 1, 2]),
        "is_silence": torch.tensor([False, False, False]),
        "current_epoch": 2,
        "is_training": True,
    }

    plain = get_sc_loss_func(lambda_energy=0.0)(output, target)
    robust = get_sc_loss_func(
        lambda_energy=0.0,
        robust_loss_mode="truncation",
        truncation_quantile=0.5,
        truncation_drop_weight=0.0,
        min_keep_ratio=0.5,
    )(output, target)

    assert robust["loss_arcface"].item() < plain["loss_arcface"].item()
    assert robust["loss_arcface_raw"].item() == pytest.approx(plain["loss_arcface"].item())
    assert robust["loss_truncation_kept_ratio"].item() == pytest.approx(2.0 / 3.0)


def test_sc_loss_truncation_respects_warmup_and_validation_default():
    output = {
        "logits": torch.tensor([[8.0, -8.0], [-8.0, 8.0], [8.0, -8.0]], requires_grad=True),
        "plain_logits": torch.tensor([[8.0, -8.0], [-8.0, 8.0], [8.0, -8.0]], requires_grad=True),
        "energy": torch.zeros(3, requires_grad=True),
    }
    target = {
        "class_index": torch.tensor([0, 1, 1]),
        "is_silence": torch.tensor([False, False, False]),
        "current_epoch": 0,
        "is_training": True,
    }

    warmup = get_sc_loss_func(
        robust_loss_mode="truncation",
        truncation_quantile=0.5,
        truncation_warmup_epochs=2,
        truncation_drop_weight=0.0,
    )(output, target)
    target["current_epoch"] = 3
    target["is_training"] = False
    validation = get_sc_loss_func(
        robust_loss_mode="truncation",
        truncation_quantile=0.5,
        truncation_warmup_epochs=2,
        truncation_drop_weight=0.0,
    )(output, target)

    assert warmup["loss_truncation_kept_ratio"].item() == 1.0
    assert validation["loss_truncation_kept_ratio"].item() == 1.0


def test_sc_duplicate_recall_shares_truncation_weights():
    output = {
        "logits": torch.tensor([[8.0, -8.0], [8.0, -8.0], [-8.0, 8.0]], requires_grad=True),
        "plain_logits": torch.tensor([[8.0, -8.0], [8.0, -8.0], [-8.0, 8.0]], requires_grad=True),
        "energy": torch.tensor([-5.0, -3.0, -5.0], requires_grad=True),
    }
    target = {
        "class_index": torch.tensor([0, 1, 1]),
        "is_silence": torch.tensor([False, False, False]),
        "is_duplicate_class": torch.tensor([True, True, False]),
        "current_epoch": 2,
        "is_training": True,
    }

    plain = get_sc_loss_func(lambda_duplicate_recall=1.0, duplicate_m_in=-8.0)(output, target)
    robust = get_sc_loss_func(
        lambda_duplicate_recall=1.0,
        duplicate_m_in=-8.0,
        robust_loss_mode="truncation",
        truncation_quantile=0.5,
        truncation_drop_weight=0.0,
        min_keep_ratio=0.5,
    )(output, target)

    assert plain["loss_duplicate_recall"].item() == pytest.approx(17.0)
    assert robust["loss_duplicate_recall"].item() == pytest.approx(9.0)


def test_temporal_sc_forward_and_activity_loss():
    model = M2DSingleClassifierTemporalStrong.__new__(M2DSingleClassifierTemporalStrong)
    torch.nn.Module.__init__(model)
    model.cfg = SimpleNamespace(feature_d=4, sample_rate=8)
    model.num_classes = 3
    model.ref_channel = None
    model.energy_thresholds = {}
    model.eval_crop_seconds = None
    model.eval_crop_hop_seconds = None
    model.activity_temperature = 1.0
    model.activity_head = torch.nn.Linear(4, 1)
    model.embedding = torch.nn.Linear(16, 6)
    model.arc_head = ArcMarginProduct(6, out_features=3)

    def encode(self, waveform, average_per_time_frame=False):
        base = waveform[:, :4].unsqueeze(-1).repeat(1, 1, 4)
        return base

    model.encode = MethodType(encode, model)
    batch = {
        "waveform": torch.randn(2, 1, 8),
        "class_index": torch.tensor([1, 0]),
        "is_silence": torch.tensor([False, True]),
        "span_sec": torch.tensor([[0.0, 0.5], [-1.0, -1.0]]),
    }
    output = model({"waveform": batch["waveform"], "class_index": batch["class_index"]})
    assert output["activity_logits"].shape == (2, 4)

    loss_func = get_sc_loss_func(lambda_activity=0.5)
    loss = loss_func(output, batch)["loss"]
    loss.backward()

    assert torch.isfinite(loss)
    assert model.activity_head.weight.grad is not None


def test_temporal_tse_loss_uses_activity_logits():
    loss_func = get_tse_loss_func(lambda_activity=0.5)
    output = {
        "waveform": torch.randn(1, 2, 1, 32, requires_grad=True),
        "activity_logits": torch.randn(1, 2, 5, requires_grad=True),
        "duration_sec": torch.tensor([1.0]),
    }
    target = {
        "waveform": torch.randn(1, 2, 1, 32),
        "label_vector": torch.eye(2).unsqueeze(0),
        "active_mask": torch.tensor([[True, False]]),
        "span_sec": torch.tensor([[[0.0, 0.5], [-1.0, -1.0]]]),
    }
    loss = loss_func(output, target)["loss"]
    loss.backward()
    assert torch.isfinite(loss)
    assert output["activity_logits"].grad is not None


def test_temporal_uss_loss_uses_activity_logits():
    loss_func = get_uss_loss_func(
        lambda_activity_foreground=0.2,
        lambda_activity_interference=0.1,
        lambda_activity_noise=0.1,
    )
    output = {
        "foreground_waveform": torch.randn(1, 2, 1, 32, requires_grad=True),
        "interference_waveform": torch.randn(1, 1, 1, 32, requires_grad=True),
        "noise_waveform": torch.randn(1, 1, 1, 32, requires_grad=True),
        "class_logits": torch.randn(1, 2, 3, requires_grad=True),
        "silence_logits": torch.randn(1, 2, requires_grad=True),
        "foreground_activity_logits": torch.randn(1, 2, 5, requires_grad=True),
        "interference_activity_logits": torch.randn(1, 1, 5, requires_grad=True),
        "noise_activity_logits": torch.randn(1, 1, 5, requires_grad=True),
        "duration_sec": torch.tensor([1.0]),
    }
    target = {
        "mixture": torch.randn(1, 2, 32),
        "foreground_waveform": torch.randn(1, 2, 1, 32),
        "interference_waveform": torch.randn(1, 1, 1, 32),
        "noise_waveform": torch.randn(1, 1, 1, 32),
        "class_index": torch.tensor([[1, 0]]),
        "is_silence": torch.tensor([[False, True]]),
        "foreground_span_sec": torch.tensor([[[0.0, 0.5], [-1.0, -1.0]]]),
        "interference_span_sec": torch.tensor([[[0.25, 0.75]]]),
        "noise_span_sec": torch.tensor([[[0.0, 1.0]]]),
    }
    loss = loss_func(output, target)["loss"]
    loss.backward()
    assert torch.isfinite(loss)
    assert output["foreground_activity_logits"].grad is not None


def test_non_memory_temporal_deft_variants_forward():
    mixture = torch.randn(1, 2, 512)
    uss_kwargs = dict(
        input_channels=2,
        hidden_channels=8,
        n_deft_blocks=1,
        n_heads=1,
        n_foreground=2,
        n_interference=1,
        n_classes=4,
        window_size=64,
        hop_size=32,
        sample_rate=32000,
    )
    for cls in (ModifiedDeFTUSSTemporal, ModifiedDeFTUSSSpatialTemporal, ChunkedModifiedDeFTUSSSpatialTemporal):
        kwargs = dict(uss_kwargs)
        if cls is not ModifiedDeFTUSSTemporal:
            kwargs["output_channels"] = 1
        if cls is ChunkedModifiedDeFTUSSSpatialTemporal:
            kwargs["inference_chunk_seconds"] = None
        model = cls(**kwargs)
        out = model({"mixture": mixture})
        assert out["foreground_waveform"].shape == (1, 2, 1, 512)
        assert out["foreground_activity_logits"].shape[:2] == (1, 2)

    label_vector = torch.eye(4)[:2].unsqueeze(0)
    tse = ModifiedDeFTTSETemporal(
        mixture_channels=2,
        enrollment_channels=1,
        hidden_channels=8,
        n_deft_blocks=1,
        n_heads=1,
        label_dim=4,
        window_size=64,
        hop_size=32,
        sample_rate=32000,
    )
    out = tse({"mixture": mixture, "enrollment": torch.randn(1, 2, 1, 512), "label_vector": label_vector})
    assert out["waveform"].shape == (1, 2, 1, 512)
    assert out["activity_logits"].shape[:2] == (1, 2)
    conditioned = tse({
        "mixture": mixture,
        "enrollment": torch.randn(1, 2, 1, 512),
        "label_vector": label_vector,
        "temporal_conditioning": torch.tensor([[[1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]]]),
    })
    assert conditioned["waveform"].shape == (1, 2, 1, 512)

    like = DeFTTSELikeSpatialTemporal(
        input_channels=2,
        output_channels=1,
        target_sources_num=2,
        label_len=8,
        window_size=64,
        hop_size=32,
        base_channels=8,
        n_blocks=1,
        n_heads=1,
        sample_rate=32000,
    )
    out = like({"mixture": mixture, "label_vector": label_vector})
    assert out["waveform"].shape == (1, 2, 1, 512)
    assert out["activity_logits"].shape[:2] == (1, 2)


if __name__ == "__main__":
    test_duplicate_same_class_target_swap_gives_low_loss()
    test_distinct_labels_cannot_swap()
    test_all_silence_tse_target_has_nonzero_output_gradient()
    test_capi_raw_and_sdri_assignment_match_for_equal_same_class_counts()
    test_capi_comparison_exposes_subset_selection_difference()
    test_s5_returns_all_silence_without_tse_hallucination()
    test_s5_forces_partially_silent_slots_after_tse_hallucination()
    test_s5_duplicate_recall_recovers_silenced_same_class_slot_only_when_enabled()
    test_temporal_s5_duplicate_recall_requires_temporal_support()
    test_temporal_s5_gates_inactive_slots_and_conditions_tse()
    test_energy_threshold_calibration_prefers_low_energy_true_positives()
    test_event_span_targets_keep_silence_inactive()
    test_source_classifier_dataset_emits_span_sec()
    test_source_classifier_dataset_marks_duplicate_classes()
    test_estimated_enrollment_tse_dataset_uses_estimates_as_enrollment_and_masks_missing_slots()
    test_previous_sc_loss_ignores_new_span_field_by_default()
    test_sc_duplicate_recall_loss_pushes_duplicate_active_energy()
    test_temporal_sc_forward_and_activity_loss()
    test_temporal_tse_loss_uses_activity_logits()
    test_temporal_uss_loss_uses_activity_logits()
    test_non_memory_temporal_deft_variants_forward()
    print("Task4 2026 loss smoke tests passed.")
