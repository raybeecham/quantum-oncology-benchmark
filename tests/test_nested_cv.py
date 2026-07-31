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


def test_nested_model_specs_keep_preprocessing_inside_locked_pipelines() -> None:
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
    assert len(specs["rbf_svm"].param_grid["model__estimator__C"]) == 3
    assert len(specs["rbf_svm"].param_grid["model__estimator__gamma"]) == 3
    assert specs["random_forest"].param_grid["model__n_estimators"] == [200, 500]
    assert specs["hist_gradient_boosting"].param_grid["model__max_leaf_nodes"] == [15, 31]


def test_nested_cv_writes_complete_fold_level_artifacts(tmp_path) -> None:
    output = tmp_path / "nested"
    config = NestedCVConfig(
        features=4,
        outer_folds=2,
        inner_folds=2,
        models=("logistic_regression", "hist_gradient_boosting"),
        max_samples=80,
        output_dir=str(output),
    )

    payload = run_nested_cv(config)

    assert payload["schema_version"] == "nested-cv-1.0"
    assert payload["methodology"]["primary_endpoint_locked"] is True
    assert payload["methodology"]["outer_test_usage"] == (
        "single_final_evaluation_after_inner_selection"
    )
    assert len(payload["outer_fold_results"]) == 4
    assert len(payload["summary"]) == 2
    assert len(payload["pairwise_comparisons"]) == 2
    assert len(payload["pairwise_summary"]) == 1
    assert len(payload["inner_search_results"]) == 14

    for row in payload["outer_fold_results"]:
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

    selected_candidates = [row for row in payload["inner_search_results"] if row["selected"]]
    assert len(selected_candidates) == config.outer_folds * len(config.models)

    assert (output / "nested_experiment.json").exists()
    assert (output / "outer_fold_results.csv").exists()
    assert (output / "outer_fold_predictions.csv").exists()
    assert (output / "inner_search_results.csv").exists()
    assert (output / "nested_summary.csv").exists()
    assert (output / "nested_pairwise_comparisons.csv").exists()
    assert (output / "NESTED_CV_REPORT.md").exists()

    report = (output / "NESTED_CV_REPORT.md").read_text(encoding="utf-8")
    assert "## Locked Evaluation Protocol" in report
    assert "single_final_evaluation_after_inner_selection" in report
    assert "No pooled p-value" in report


def test_nested_cv_same_configuration_reproduces_scientific_results(tmp_path) -> None:
    config = NestedCVConfig(
        features=4,
        outer_folds=2,
        inner_folds=2,
        models=("logistic_regression",),
        max_samples=80,
        output_dir=str(tmp_path / "unused"),
    )

    first = run_nested_cv(config, write_output=False)
    second = run_nested_cv(config, write_output=False)

    assert normalize_nested_experiment_for_reproducibility(first) == (
        normalize_nested_experiment_for_reproducibility(second)
    )
