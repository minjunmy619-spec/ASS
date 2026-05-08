import csv
import json

from src.evaluation.calibrate_sc_energy_thresholds import energy_diagnostics, write_outputs


def test_energy_diagnostics_reports_requested_buckets():
    records = [
        {"energy": -20.0, "pred_class": 1, "target_class": 1, "is_silence": False, "is_estimated_source": False},
        {"energy": -10.0, "pred_class": 2, "target_class": 1, "is_silence": False, "is_estimated_source": False},
        {"energy": -2.0, "pred_class": 2, "target_class": 0, "is_silence": True, "is_estimated_source": False},
        {"energy": -4.0, "pred_class": 2, "target_class": 2, "is_silence": False, "is_estimated_source": True},
    ]

    rows = {row["bucket"]: row for row in energy_diagnostics(records)}

    assert rows["true_active_correct_predictions"]["n"] == 2
    assert rows["true_active_correct_predictions"]["q00"] == -20.0
    assert rows["true_active_correct_predictions"]["q100"] == -4.0
    assert rows["true_active_wrong_predictions"]["n"] == 1
    assert rows["true_active_wrong_predictions"]["q50"] == -10.0
    assert rows["true_silence_slots"]["n"] == 1
    assert rows["true_silence_slots"]["q50"] == -2.0
    assert rows["estimated_source_slots"]["n"] == 1
    assert rows["estimated_source_slots"]["q50"] == -4.0


def test_write_outputs_writes_energy_diagnostics(tmp_path):
    diagnostics = energy_diagnostics([
        {"energy": -5.0, "pred_class": 0, "target_class": 0, "is_silence": False, "is_estimated_source": True},
    ])

    paths = write_outputs(
        tmp_path,
        thresholds={0: -0.5},
        stats=[
            {
                "class_index": 0,
                "class_name": "AlarmClock",
                "threshold": -0.5,
                "n_predicted_as_class": 1,
                "n_pos": 1,
                "n_neg": 0,
                "tp": 1,
                "fp": 0,
                "fn": 0,
                "tn": 0,
                "precision": 1.0,
                "recall": 1.0,
                "fpr": 0.0,
                "fbeta": 1.0,
                "used_fallback": False,
            }
        ],
        default_threshold=-0.5,
        diagnostics=diagnostics,
    )

    assert tmp_path / "energy_diagnostics.csv" in paths
    assert tmp_path / "energy_diagnostics.json" in paths
    with (tmp_path / "energy_diagnostics.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["bucket"] == "true_active_correct_predictions"
    assert rows[0]["n"] == "1"
    with (tmp_path / "energy_diagnostics.json").open() as f:
        json_rows = json.load(f)
    assert json_rows[-1]["bucket"] == "estimated_source_slots"
    assert json_rows[-1]["q50"] == -5.0
