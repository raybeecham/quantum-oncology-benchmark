from __future__ import annotations

import pytest

from quantum_oncology_benchmark.calibration import build_calibration_diagnostics


def _prediction_rows() -> list[dict[str, object]]:
    return [
        {
            "outer_fold": 0,
            "search_profile": "reference-v1",
            "model": "example",
            "sample_index_hash": "a" * 64,
            "y_true": 0,
            "y_pred": 0,
            "y_score": 0.1,
        },
        {
            "outer_fold": 0,
            "search_profile": "reference-v1",
            "model": "example",
            "sample_index_hash": "b" * 64,
            "y_true": 1,
            "y_pred": 0,
            "y_score": 0.4,
        },
        {
            "outer_fold": 1,
            "search_profile": "reference-v1",
            "model": "example",
            "sample_index_hash": "c" * 64,
            "y_true": 1,
            "y_pred": 1,
            "y_score": 0.6,
        },
        {
            "outer_fold": 1,
            "search_profile": "reference-v1",
            "model": "example",
            "sample_index_hash": "d" * 64,
            "y_true": 0,
            "y_pred": 1,
            "y_score": 0.9,
        },
    ]


def test_calibration_diagnostics_compute_fixed_bin_contract() -> None:
    diagnostics = build_calibration_diagnostics(_prediction_rows(), bins=2)

    summary = diagnostics["summary"][0]
    assert summary["samples"] == 4
    assert summary["prevalence"] == 0.5
    assert summary["mean_predicted_probability"] == 0.5
    assert summary["calibration_in_the_large"] == 0.0
    assert summary["expected_calibration_error"] == pytest.approx(0.25)
    assert summary["maximum_calibration_error"] == pytest.approx(0.25)
    assert summary["pooled_out_of_fold_brier_score"] == pytest.approx(0.335)
    assert summary["false_negative_count"] == 1
    assert summary["false_positive_count"] == 1

    bins = diagnostics["reliability_bins"]
    assert len(bins) == 2
    assert sum(int(row["samples"]) for row in bins) == 4
    assert bins[0]["mean_predicted_probability"] == pytest.approx(0.25)
    assert bins[0]["observed_positive_rate"] == pytest.approx(0.5)

    distribution = diagnostics["probability_distribution"]
    assert len(distribution) == 4
    assert sum(int(row["samples"]) for row in distribution) == 4

    errors = diagnostics["classification_errors"]
    assert {row["error_type"] for row in errors} == {"false_negative", "false_positive"}
    assert all(len(str(row["sample_index_hash"])) == 64 for row in errors)


def test_calibration_diagnostics_reject_invalid_probability() -> None:
    rows = _prediction_rows()
    rows[0]["y_score"] = 1.1

    with pytest.raises(ValueError, match="probabilities"):
        build_calibration_diagnostics(rows, bins=2)
