import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics.s5capi_metric import (
    S5ClassAwareMetric,
    S5ClassAwareMetricAssignmentComparison,
    S5ClassAwareMetricSDRiAssignment,
)


def fake_assignment_metric(pred, target):
    if torch.all(pred == 0):
        return target.new_tensor([[0.0], [100.0]])
    return target.new_tensor([[10.0], [20.0]])


def test_default_metric_locks_official_raw_sdr_assignment():
    metric = S5ClassAwareMetric()
    metric.metric_func = fake_assignment_metric

    output = metric._pi_metric(
        est_wf=torch.tensor([[1.0]]),
        ref_wf=torch.tensor([[1.0], [2.0]]),
        mixture=torch.zeros(1),
    )

    assert output["ref_perm"] == [1]
    assert output["est_perm"] == [0]
    assert torch.allclose(output["metric"], torch.tensor([20.0, 0.0]))
    assert torch.allclose(output["metric_i"], torch.tensor([-80.0, 0.0]))


def test_paper_definition_diagnostic_uses_sdri_assignment():
    metric = S5ClassAwareMetricSDRiAssignment()
    metric.metric_func = fake_assignment_metric

    output = metric._pi_metric(
        est_wf=torch.tensor([[1.0]]),
        ref_wf=torch.tensor([[1.0], [2.0]]),
        mixture=torch.zeros(1),
    )

    assert output["ref_perm"] == [0]
    assert output["est_perm"] == [0]
    assert torch.allclose(output["metric"], torch.tensor([10.0, 0.0]))
    assert torch.allclose(output["metric_i"], torch.tensor([10.0, 0.0]))


def test_assignment_comparison_reports_official_and_paper_delta():
    comparison = S5ClassAwareMetricAssignmentComparison()
    comparison.metric_func = fake_assignment_metric

    value = comparison.compute_sample(
        est_lb=["dog"],
        est_wf=torch.tensor([[1.0]]),
        ref_lb=["dog", "dog"],
        ref_wf=torch.tensor([[1.0], [2.0]]),
        mixture=torch.zeros(1),
    )

    assert value["raw_sdr_assignment"] == -40.0
    assert value["sdri_assignment"] == 5.0
    assert value["delta_sdri_minus_raw"] == 45.0


def test_true_zero_target_silence_is_excluded_from_average():
    metric = S5ClassAwareMetric()
    value = metric.compute_sample(
        est_lb=["silence"],
        est_wf=torch.zeros(1, 4),
        ref_lb=["silence"],
        ref_wf=torch.zeros(1, 4),
        mixture=torch.zeros(4),
    )

    assert value is None
    metric.metric_values = [value]
    assert metric.compute() is None


def test_zero_target_false_positive_is_zero_penalty_and_included():
    metric = S5ClassAwareMetric()
    value = metric.compute_sample(
        est_lb=["dog"],
        est_wf=torch.ones(1, 4),
        ref_lb=["silence"],
        ref_wf=torch.zeros(1, 4),
        mixture=torch.zeros(4),
    )

    assert value == 0.0
    metric.metric_values = [None, value]
    assert metric.compute()["mean"] == 0.0


def test_false_negative_is_zero_penalty_and_included():
    metric = S5ClassAwareMetric()
    value = metric.compute_sample(
        est_lb=["silence"],
        est_wf=torch.zeros(1, 4),
        ref_lb=["dog"],
        ref_wf=torch.ones(1, 4),
        mixture=torch.zeros(4),
    )

    assert value == 0.0
    metric.metric_values = [None, value]
    assert metric.compute()["mean"] == 0.0


def test_reference_mix_sdr_is_independent_of_estimated_labels():
    ref_wf = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ])
    mixture = ref_wf[:2].sum(dim=0)
    ref_labels = ["dog", "cat", "silence"]

    oracle_like = S5ClassAwareMetric()
    oracle_value = oracle_like.compute_sample(
        est_lb=["dog", "cat", "silence"],
        est_wf=ref_wf.clone(),
        ref_lb=ref_labels,
        ref_wf=ref_wf,
        mixture=mixture,
    )
    oracle_like.metric_values = [oracle_value]
    oracle_summary = oracle_like.compute()

    predicted_like = S5ClassAwareMetric()
    predicted_value = predicted_like.compute_sample(
        est_lb=["silence", "wrong_class", "silence"],
        est_wf=ref_wf.clone(),
        ref_lb=ref_labels,
        ref_wf=ref_wf,
        mixture=mixture,
    )
    predicted_like.metric_values = [predicted_value]
    predicted_summary = predicted_like.compute()

    assert "reference_mix_sdr" in oracle_summary
    assert "reference_mix_sdr" in predicted_summary
    assert oracle_summary["reference_mix_sdr"] == predicted_summary["reference_mix_sdr"]
    assert oracle_summary["mix_sdr"] != predicted_summary.get("mix_sdr")
