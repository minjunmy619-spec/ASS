import torch

from src.evaluation.evaluate_stage import StageEvaluator
from src.evaluation.export_sc_finetune_cache import (
    _ensure_soundscape_ids,
    _pit_oracle_labels,
    _pit_oracle_labels_with_manifest,
    _save_query_conditions,
    _uss_labels,
)


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


def test_export_cache_saves_tse_condition_alias_for_unified_uss(tmp_path):
    batch = {"soundscape": ["scene_a"]}
    condition = torch.randn(1, 3, 64)

    _save_query_conditions(batch, {"tse_condition": condition}, str(tmp_path), overwrite=False)

    saved = torch.load(tmp_path / "uss_bridge_features" / "scene_a.pt", map_location="cpu")
    assert torch.allclose(saved["query_condition"], condition[0])
    assert torch.allclose(saved["tse_condition"], condition[0])


def test_export_cache_generates_soundscape_ids_for_dynamic_dataset_batches():
    batch = {"mixture": torch.randn(2, 4, 32)}

    named = _ensure_soundscape_ids(batch, "train", offset=4)

    assert named["soundscape"] == ["train_generated_00000004", "train_generated_00000005"]
    assert "soundscape" not in batch


def test_export_cache_pit_oracle_labels_match_estimates_to_reference_slots():
    est_waveforms = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]])
    ref_waveforms = torch.tensor([[[0.0, 2.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    ref_labels = [["dog", "cat", "silence"]]

    labels = _pit_oracle_labels(est_waveforms, ref_waveforms, ref_labels)

    assert labels == [["cat", "dog"]]


def test_export_cache_pit_oracle_labels_reject_low_energy_matches():
    est_waveforms = torch.zeros(1, 2, 8)
    ref_waveforms = torch.zeros(1, 2, 8)
    ref_waveforms[0, 0, :2] = 1.0
    ref_waveforms[0, 1, 4:6] = 1.0
    ref_labels = [["dog", "cat"]]

    labels, rows = _pit_oracle_labels_with_manifest(
        est_waveforms,
        ref_waveforms,
        ref_labels,
        soundscapes=["scene_a"],
        include_unmatched=True,
    )

    assert labels == [["silence", "silence"]]
    assert rows
    assert all(not row.saved for row in rows)
    assert {row.quality_group for row in rows if row.oracle_slot >= 0} == {"bad"}


def test_export_cache_pit_oracle_labels_save_uncertain_is_opt_in():
    est_waveforms = torch.tensor([[[1.0, 0.0, 0.0]]])
    ref_waveforms = torch.tensor([[[1.0, 0.0, 0.0]]])
    ref_labels = [["dog"]]

    labels, rows = _pit_oracle_labels_with_manifest(
        est_waveforms,
        ref_waveforms,
        ref_labels,
        clean_match_score=100.0,
        save_uncertain=False,
    )
    assert labels == [["silence"]]
    assert rows[0].quality_group == "uncertain"
    assert rows[0].sample_weight == 0.35
    assert not rows[0].saved

    labels, rows = _pit_oracle_labels_with_manifest(
        est_waveforms,
        ref_waveforms,
        ref_labels,
        clean_match_score=100.0,
        save_uncertain=True,
    )
    assert labels == [["dog"]]
    assert rows[0].saved


def test_export_cache_uss_pseudo_labels_respect_silence_logits():
    output = {
        "class_logits": torch.tensor([[[0.0, 4.0], [5.0, 0.0], [0.0, 3.0]]]),
        "silence_logits": torch.tensor([[2.0, -1.0, 0.5]]),
    }

    labels = _uss_labels(output, ["cat", "dog"])

    assert labels == [["dog", "silence", "dog"]]
