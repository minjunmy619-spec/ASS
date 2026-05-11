import json
import os

import pytest
import torch
import yaml

from src.evaluation.evaluate import Evaluator


class DummyS5:
    def predict_label_separate(self, mixture):
        batch_size = mixture.shape[0]
        waveform = torch.zeros(batch_size, 3, 1, mixture.shape[-1])
        waveform[:, 0, 0, 0] = 0.5
        waveform[:, 2, 0, 1] = -0.25
        return {
            "label": [["AlarmClock", "silence", "Speech"] for _ in range(batch_size)],
            "waveform": waveform,
            "probabilities": torch.tensor([[0.9, 0.0, 0.7]]).expand(batch_size, 3),
        }


def _bare_evaluator(tmp_path, inference_only=True):
    evaluator = object.__new__(Evaluator)
    evaluator.filename = "dummy_submission"
    evaluator.result_dir = str(tmp_path / "results")
    evaluator.waveform_output_dir = str(tmp_path / "waveforms")
    os.makedirs(evaluator.waveform_output_dir, exist_ok=True)
    evaluator.use_cpu = True
    evaluator.inference_only = inference_only
    evaluator.use_generated_waveform = False
    evaluator.metric_funcs = []
    evaluator.model = DummyS5()
    evaluator.sr = 32000
    evaluator.dataloader = [
        {
            "mixture": torch.zeros(1, 4, 16),
            "soundscape": ["scene_a"],
        }
    ]
    return evaluator


def test_inference_only_writes_predictions_without_oracle_targets(tmp_path):
    evaluator = _bare_evaluator(tmp_path, inference_only=True)

    evaluator.evaluate()

    results_path = tmp_path / "results" / "dummy_submission_results.json"
    summary_path = tmp_path / "results" / "dummy_submission_summary.json"
    assert results_path.exists()
    assert summary_path.exists()

    results = json.loads(results_path.read_text())
    summary = json.loads(summary_path.read_text())
    assert results[0]["soundscape"] == "scene_a"
    assert results[0]["est_labels"] == ["AlarmClock", "silence", "Speech"]
    assert len(results[0]["waveform_files"]) == 2
    assert all(os.path.exists(path) for path in results[0]["waveform_files"])
    assert summary["mode"] == "inference_only"
    assert summary["num_soundscapes"] == 1
    assert summary["num_non_silence_predictions"] == 2


def test_validation_mode_requires_oracle_targets(tmp_path):
    evaluator = _bare_evaluator(tmp_path, inference_only=False)
    evaluator.waveform_output_dir = ""
    evaluator.result_dir = ""

    with pytest.raises(KeyError, match="requires oracle"):
        evaluator.evaluate()


def test_submission_config_omits_oracle_targets():
    with open("src/evaluation/eval_configs/kwo2025_top1_like_lite_estimated_temporal_sc_uss_conditioned_tse_submission.yaml") as f:
        cfg = yaml.safe_load(f)

    ds_args = cfg["dataset"]["args"]
    assert "oracle_target_dir" not in ds_args["config"]
    assert ds_args["return_source"] is False
    assert ds_args["config"]["mode"] == "waveform"
