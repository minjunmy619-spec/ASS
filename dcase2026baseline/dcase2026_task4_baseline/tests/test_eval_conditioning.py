import torch

from src.evaluation.evaluate_stage import StageEvaluator
from src.evaluation.export_sc_finetune_cache import _save_query_conditions


class RecordingTSE(torch.nn.Module):
    label_dim = 18

    def __init__(self):
        super().__init__()
        self.last_input = None

    def forward(self, input_dict):
        self.last_input = input_dict
        return {"waveform": torch.zeros_like(input_dict["enrollment"])}


def test_evaluate_stage_tse_forwards_bridge_conditions():
    evaluator = object.__new__(StageEvaluator)
    evaluator.device = torch.device("cpu")
    evaluator.model = RecordingTSE()
    evaluator.tse_condition_keys_seen = set()

    batch = {
        "mixture": torch.randn(2, 4, 32),
        "enrollment": torch.randn(2, 3, 1, 32),
        "label_vector": torch.randn(2, 3, 18),
        "bridge_condition": torch.randn(2, 3, 256),
        "label": [["a", "b", "silence"], ["c", "silence", "silence"]],
    }

    evaluator._evaluate_tse_batch(batch)

    assert "bridge_condition" in evaluator.model.last_input
    assert evaluator.model.last_input["bridge_condition"].shape == (2, 3, 256)
    assert evaluator.tse_condition_keys_seen == {"bridge_condition"}


def test_export_cache_saves_query_conditions_for_tse_bridge_training(tmp_path):
    batch = {"soundscape": ["scene_a", "scene_b"]}
    condition = torch.randn(2, 3, 64)

    _save_query_conditions(batch, {"query_condition": condition}, str(tmp_path), overwrite=False)

    saved = torch.load(tmp_path / "uss_bridge_features" / "scene_a.pt", map_location="cpu")
    assert torch.allclose(saved["query_condition"], condition[0])
    assert torch.allclose(saved["tse_condition"], condition[0])
