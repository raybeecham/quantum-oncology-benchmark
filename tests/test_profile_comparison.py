from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from quantum_oncology_benchmark.profile_comparison import compare_nested_profiles


def _payload(profile: str, *, candidate: bool = False) -> dict[str, Any]:
    predictions = []
    models = ("logistic_regression", "rbf_svm", "hist_gradient_boosting")
    truth = [0, 0, 1, 1]
    values = {
        "logistic_regression": [0, 0, 1, 1],
        "rbf_svm": [0, 1, 1, 1] if candidate else [0, 0, 1, 1],
        "hist_gradient_boosting": [0, 0, 1, 1] if candidate else [0, 1, 1, 1],
    }
    scores = [0.1, 0.6, 0.8, 0.9]
    for model in models:
        for index, label in enumerate(truth):
            predictions.append(
                {
                    "outer_fold": index % 2,
                    "model": model,
                    "sample_index_hash": f"sample-{index}",
                    "y_true": label,
                    "y_pred": values[model][index],
                    "y_score": scores[index],
                    "search_profile": profile,
                }
            )

    summary_values = {
        "logistic_regression": 0.95,
        "rbf_svm": 0.94 if candidate else 0.95,
        "hist_gradient_boosting": 0.93 if candidate else 0.92,
    }
    outer_rows = []
    for fold in range(2):
        outer_rows.extend(
            [
                {
                    "outer_fold": fold,
                    "model": "logistic_regression",
                    "best_params": {"model__C": 1.0},
                },
                {
                    "outer_fold": fold,
                    "model": "rbf_svm",
                    "best_params": {"model__C": 100.0 if candidate else 10.0},
                },
                {
                    "outer_fold": fold,
                    "model": "hist_gradient_boosting",
                    "best_params": {"model__max_leaf_nodes": 7 if candidate else 15},
                },
            ]
        )

    return {
        "schema_version": "nested-cv-1.1",
        "config": {
            "dataset": "breast-cancer",
            "features": 8,
            "seed": 42,
            "outer_folds": 2,
            "inner_folds": 2,
            "models": list(models),
            "primary_metric": "balanced_accuracy",
            "max_samples": None,
            "search_profile": profile,
        },
        "dataset": {"fingerprint": "fingerprint"},
        "methodology": {
            "model_search_spaces": {
                "logistic_regression": {"model__C": [0.1, 1.0, 10.0]},
                "rbf_svm": {
                    "model__C": [1.0, 10.0, 100.0]
                    if candidate
                    else [0.1, 1.0, 10.0],
                    "model__gamma": ["scale", 0.01, 0.1],
                },
                "hist_gradient_boosting": {
                    "model__max_leaf_nodes": [7, 15, 31]
                    if candidate
                    else [15, 31]
                },
            }
        },
        "summary": [
            {"model": model, "balanced_accuracy_mean": summary_values[model]}
            for model in models
        ],
        "outer_fold_predictions": predictions,
        "outer_fold_results": outer_rows,
    }


def _write(directory: Path, payload: dict[str, Any]) -> None:
    directory.mkdir(parents=True)
    (directory / "nested_experiment.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_compare_profiles_writes_paired_protocol_freeze_artifacts(tmp_path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    output = tmp_path / "comparison"
    _write(reference, _payload("reference-v1"))
    _write(candidate, _payload("sensitivity-v1", candidate=True))

    payload = compare_nested_profiles(reference, candidate, output)

    assert payload["schema_version"] == "profile-comparison-1.0"
    assert payload["protocol_freeze"]["primary_classical_comparator"] == (
        "logistic_regression"
    )
    rows = {row["model"]: row for row in payload["comparison_summary"]}
    assert rows["logistic_regression"]["changed_predictions"] == 0
    assert rows["rbf_svm"]["reference_only_correct"] == 1
    assert rows["hist_gradient_boosting"]["candidate_only_correct"] == 1

    decisions = {
        row["model"]: row["decision"]
        for row in payload["protocol_freeze"]["model_recommendations"]
    }
    assert decisions["logistic_regression"] == "retain_reference_control"
    assert decisions["rbf_svm"] == "retain_reference_grid"
    assert decisions["hist_gradient_boosting"] == (
        "retain_candidate_expansion_for_future_protocol"
    )

    assert len(payload["paired_predictions"]) == 12
    assert (output / "profile_comparison.json").exists()
    assert (output / "cross_profile_summary.csv").exists()
    assert (output / "cross_profile_predictions.csv").exists()
    assert (output / "parameter_boundary_summary.csv").exists()
    assert (output / "PROTOCOL_FREEZE_REPORT.md").exists()


def test_compare_profiles_rejects_different_partitions(tmp_path) -> None:
    reference_payload = _payload("reference-v1")
    candidate_payload = _payload("sensitivity-v1", candidate=True)
    candidate_payload["outer_fold_predictions"][0]["outer_fold"] = 1
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write(reference, reference_payload)
    _write(candidate, candidate_payload)

    with pytest.raises(ValueError, match="outer-fold assignment differs"):
        compare_nested_profiles(reference, candidate, tmp_path / "output")
