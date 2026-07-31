from __future__ import annotations

from sklearn.pipeline import Pipeline

from quantum_oncology_benchmark.config import NestedCVConfig
from quantum_oncology_benchmark.nested_cv import (
    build_nested_model_specs,
    run_nested_cv,
)
from quantum_oncology_benchmark.nested_reporting import (
    normalize_nested_experiment_for_reproducibility,
)


def test_nested_model_specs_keep_reference_profile_immutable() -> None:
    specs = build_nested_model_specs(
        seed=42,
        feature_count=4,
        models=(
            "logistic_regression",
            "rbf_svm",
            "random_forest",
            "hist_gradient_boosting",
        ),
        calibration_folds=3,
        search_profile="reference-v1",
    )

    assert list(specs) == [
        "logistic_regression",
        "rbf_svm",
        "random_forest",
        "hist_gradient_boosting",
    ]
    for spec in specs.values():
        assert isinstance(spec.estimator, Pipeline)
        assert list(spec.estimator.named_steps) == ["imputer", "scaler", "selector", "model"]

    assert specs["logistic_regression"].param_grid == {"model__C": [0.1, 1.0, 10.0]}
    assert specs["rbf_svm"].param_grid["model__C"] == [0.1, 1.0, 10.0]
    assert specs["rbf_svm"].param_grid["model__gamma"] == ["scale", 0.01, 0.1]
    assert specs["random_forest"].param_grid["model__n_estimators"] == [200, 500]
    assert specs["hist_gradient_boosting"].param_grid["model__max_leaf_nodes"] == [15, 31]


def test_nested_sensitivity_profile_expands_only_boundary_grids() -> None:
    specs = build_nested_model_specs(
        seed=42,
        feature_count=4,
        models=(
            "logistic_regression",
            "rbf_svm",
            "random_forest",
            "hist_gradient_boosting",
        ),
        calibration_folds=3,
        search_profile="sensitivity-v1",
    )

    assert specs["logistic_regression"].param_grid == {"model__C": [0.1, 1.0, 10.0]}
    assert specs["rbf_svm"].param_grid["model__C"] == [1.0, 10.0, 100.0]
    assert specs["rbf_svm"].param_grid["model__gamma"] == ["scale", 0.01, 0.1]
    assert specs["random_forest"].param_grid["model__n_estimators"] == [200, 500]
    assert specs["hist_gradient_boosting"].param_grid["model__max_leaf_nodes"] == [7, 15, 31]


def test_nested_cv_writes_complete_diagnostic_artifacts(tmp_path) -> None:
    output = tmp_path / "nested"
    config = NestedCVConfig(
        features=4,
        outer_folds=2,
        inner_folds=2,
        models=("logistic_regression", "hist_gradient_boosting"),
        calibration_bins=5,
        max_samples=80,
        output_dir=str(output),
    )

    payload = run_nested_cv(config)

    assert payload["schema_version"] == "nested-cv-1.1"
    assert payload["methodology"]["primary_endpoint_locked"] is True
    assert payload["methodology"]["search_profile"] == "reference-v1"
    assert payload["methodology"]["outer_test_usage"] == (
        "single_final_evaluation_after_inner_selection"
    )
    assert payload["methodology"]["calibration_diagnostics"][
        "each_sample_predicted_once_per_model"
    ] is True
    assert len(payload["outer_fold_results"]) == 4
    assert len(payload["summary"]) == 2
    assert len(payload["pairwise_comparisons"]) == 2
    assert len(payload["pairwise_summary"]) == 1
    assert len(payload["inner_search_results"]) == 14
    assert len(payload["calibration_summary"]) == 2
    assert len(payload["probability_distribution"]) == 20

    for row in payload["outer_fold_results"]:
        assert row["search_profile"] == "reference-v1"
        assert len(row["train_index_hash"]) == 64
        assert len(row["test_index_hash"]) == 64
        assert row["train_index_hash"] != row["test_index_hash"]
        assert len(row["selected_features"]) == 4
        assert sum(row["training_class_counts"].values()) == row["train_samples"]
        assert sum(row["test_class_counts"].values()) == row["test_samples"]

    for model in config.models:
        model_predictions = [
            row for row in payload["outer_fold_predictions"] if row["model"] == model
        ]
        assert len(model_predictions) == 80
        assert len({row["sample_index_hash"] for row in model_predictions}) == 80

        reliability_rows = [
            row for row in payload["calibration_bins"] if row["model"] == model
        ]
        assert sum(row["samples"] for row in reliability_rows) == 80

    expected_errors = sum(
        row["false_positive_count"] + row["false_negative_count"]
        for row in payload["calibration_summary"]
    )
    assert len(payload["classification_errors"]) == expected_errors

    selected_candidates = [row for row in payload["inner_search_results"] if row["selected"]]
    assert len(selected_candidates) == config.outer_folds * len(config.models)

    expected_files = (
        "nested_experiment.json",
        "outer_fold_results.csv",
        "outer_fold_predictions.csv",
        "inner_search_results.csv",
        "nested_summary.csv",
        "nested_pairwise_comparisons.csv",
        "calibration_summary.csv",
        "calibration_bins.csv",
        "probability_distribution.csv",
        "classification_errors.csv",
        "NESTED_CV_REPORT.md",
    )
    assert all((output / name).exists() for name in expected_files)

    report = (output / "NESTED_CV_REPORT.md").read_text(encoding="utf-8")
    assert "## Locked Evaluation Protocol" in report
    assert "## Out-of-Fold Calibration Diagnostics" in report
    assert "single_final_evaluation_after_inner_selection" in report
    assert "No pooled p-value" in report


def test_nested_svm_calibrates_scores_only_on_outer_training(tmp_path) -> None:
    config = NestedCVConfig(
        features=4,
        outer_folds=2,
        inner_folds=2,
        models=("rbf_svm",),
        calibration_bins=5,
        max_samples=80,
        output_dir=str(tmp_path / "svm"),
    )

    payload = run_nested_cv(config, write_output=False)

    assert payload["methodology"]["svm_calibration_changes_class_predictions"] is False
    assert payload["methodology"]["svm_probability_calibration"] == (
        "sigmoid_cv_on_outer_training_after_selection"
    )
    assert all(
        row["probability_score_source"] == "sigmoid_cv_2_fold_outer_training_only"
        for row in payload["outer_fold_results"]
    )
    assert len(payload["outer_fold_predictions"]) == 80
    assert payload["calibration_summary"][0]["samples"] == 80


def test_nested_cv_same_configuration_reproduces_scientific_results(tmp_path) -> None:
    config = NestedCVConfig(
        features=4,
        outer_folds=2,
        inner_folds=2,
        models=("logistic_regression",),
        calibration_bins=5,
        max_samples=80,
        output_dir=str(tmp_path / "unused"),
    )

    first = run_nested_cv(config, write_output=False)
    second = run_nested_cv(config, write_output=False)

    assert normalize_nested_experiment_for_reproducibility(first) == (
        normalize_nested_experiment_for_reproducibility(second)
    )
