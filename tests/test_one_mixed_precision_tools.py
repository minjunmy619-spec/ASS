from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "online"
sys.path.insert(0, str(TOOLS))

import analyze_one_mixed_precision_calibration as analyzer  # noqa: E402
import search_one_mixed_precision_qconfig as search  # noqa: E402


def test_priority_reservoir_samples_later_calibration_values() -> None:
    reservoir = analyzer.PriorityReservoir(20, np.random.default_rng(7))

    reservoir.update(np.zeros(100, dtype=np.float32))
    reservoir.update(np.ones(100, dtype=np.float32))

    assert reservoir.seen == 200
    assert reservoir.values.size == 20
    assert np.any(reservoir.values == 0)
    assert np.any(reservoir.values == 1)


def test_dequantize_error_uses_away_from_zero_rounding_and_native_int16_range() -> None:
    values = np.array([-32768.0, -0.5, 0.5, 32767.0], dtype=np.float32)

    mse, _mae, clipped = analyzer.dequantize_error(
        values,
        scale=1.0,
        zero=0,
        quant_min=-32768,
        quant_max=32767,
    )

    assert mse == pytest.approx(0.125)
    assert clipped == 0.0


def test_load_candidates_accepts_legacy_boolean_and_resolves_live_graph_index(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("name", "eligible", "score", "rough_ops"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "name": "node",
                "eligible": "1",
                "score": "3.5",
                "rough_ops": "999999",
            }
        )
    op = search.OpInfo(
        index=17,
        name="node",
        op="CONV_2D",
        inputs=[],
        outputs=[],
        output_shape=[1, 4, 4, 8],
        rough_ops=123.0,
    )

    candidates, skipped = search.load_candidates(path, 8, [op])

    assert skipped == []
    assert candidates[0]["index"] == 17
    assert candidates[0]["rough_ops"] == 123.0
    assert candidates[0]["selection_score"] == 3.5


def test_load_candidates_reports_stale_names(tmp_path: Path) -> None:
    path = tmp_path / "stale.csv"
    path.write_text(
        "name,eligible,selection_score\nmissing,true,1.0\n",
        encoding="utf-8",
    )

    candidates, skipped = search.load_candidates(path, 8, [])

    assert candidates == []
    assert skipped == ["missing: not found in current Circle graph"]


def test_quality_value_defaults_to_primary_separation_output() -> None:
    result = {
        "mse_primary": 1.0,
        "mse_mean": 4.0,
        "mse_by_output": {"separation": 1.0, "state": 7.0},
    }

    assert search.quality_value(result, "primary", None) == 1.0
    assert search.quality_value(result, "mean", None) == 4.0
    assert search.quality_value(result, "output", "state") == 7.0
    assert search.quality_value(result, "output", "missing") is None


def test_equal_or_tiny_mse_change_is_not_accepted() -> None:
    assert not search.improvement_is_sufficient(1.0, 1.0, 0.0, 0.0)
    assert not search.improvement_is_sufficient(1.0, 0.9999995, 0.0, 1e-6)
    assert search.improvement_is_sufficient(1.0, 0.999, 0.0, 1e-6)
