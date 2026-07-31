"""Classical nested cross-validation with locked evaluation boundaries."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .calibration import build_calibration_diagnostics
from .config import NestedCVConfig
from .data import DatasetBundle, load_dataset
from .metrics import evaluate_binary_classifier
from .nested_reporting import write_nested_artifacts
from .reporting import environment_metadata, utc_now
from .statistical import (
    attach_bootstrap_intervals,
    exact_mcnemar_comparison,
    summarize_pairwise_comparisons,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]

_METRIC_NAMES = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "sensitivity",
    "specificity",
    "f1",
    "roc_auc",
    "average_precision",
    "brier_score",
    "log_loss",
)
_CONFIDENCE_LEVEL = 0.95
_BOOTSTRAP_RESAMPLES = 5000
_SIGNIFICANCE_LEVEL = 0.05


@dataclass(frozen=True, slots=True)
class NestedModelSpec:
    """One leakage-resistant model pipeline and its bounded search grid."""

    estimator: Any
    param_grid: dict[str, list[Any]]


def _preprocessing_steps(feature_count: int) -> list[tuple[str, Any]]:
    return [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("selector", SelectKBest(score_func=f_classif, k=feature_count)),
    ]


def _model_pipeline(feature_count: int, model: Any) -> Pipeline:
    return Pipeline([*_preprocessing_steps(feature_count), ("model", model)])


def _profile_grids(search_profile: str) -> dict[str, dict[str, list[Any]]]:
    if search_profile == "reference-v1":
        svm_c = [0.1, 1.0, 10.0]
        boosting_leaves = [15, 31]
    elif search_profile == "sensitivity-v1":
        svm_c = [1.0, 10.0, 100.0]
        boosting_leaves = [7, 15, 31]
    else:
        raise ValueError("unsupported nested search profile")

    return {
        "logistic_regression": {"model__C": [0.1, 1.0, 10.0]},
        "rbf_svm": {
            "model__C": svm_c,
            "model__gamma": ["scale", 0.01, 0.1],
        },
        "random_forest": {
            "model__n_estimators": [200, 500],
            "model__max_depth": [None, 5, 10],
        },
        "hist_gradient_boosting": {
            "model__learning_rate": [0.05, 0.1],
            "model__max_leaf_nodes": boosting_leaves,
        },
    }


def build_nested_model_specs(
    *,
    seed: int,
    feature_count: int,
    models: tuple[str, ...],
    calibration_folds: int,
    search_profile: str = "reference-v1",
) -> dict[str, NestedModelSpec]:
    """Return deterministic classical pipelines and versioned parameter grids."""
    del calibration_folds
    grids = _profile_grids(search_profile)
    specs: dict[str, NestedModelSpec] = {}

    if "logistic_regression" in models:
        specs["logistic_regression"] = NestedModelSpec(
            estimator=_model_pipeline(
                feature_count,
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=seed,
                ),
            ),
            param_grid=grids["logistic_regression"],
        )

    if "rbf_svm" in models:
        specs["rbf_svm"] = NestedModelSpec(
            estimator=_model_pipeline(
                feature_count,
                SVC(
                    kernel="rbf",
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
            param_grid=grids["rbf_svm"],
        )

    if "random_forest" in models:
        specs["random_forest"] = NestedModelSpec(
            estimator=_model_pipeline(
                feature_count,
                RandomForestClassifier(
                    class_weight="balanced_subsample",
                    min_samples_leaf=2,
                    n_jobs=1,
                    random_state=seed,
                ),
            ),
            param_grid=grids["random_forest"],
        )

    if "hist_gradient_boosting" in models:
        specs["hist_gradient_boosting"] = NestedModelSpec(
            estimator=_model_pipeline(
                feature_count,
                HistGradientBoostingClassifier(
                    max_iter=250,
                    l2_regularization=1.0,
                    random_state=seed,
                ),
            ),
            param_grid=grids["hist_gradient_boosting"],
        )

    return {name: specs[name] for name in models}


def _index_hash(indices: IntArray) -> str:
    canonical = ",".join(str(int(index)) for index in indices)
    return sha256(canonical.encode("utf-8")).hexdigest()


def _sample_index_hash(dataset_fingerprint: str, index: int) -> str:
    return sha256(f"{dataset_fingerprint}:{index}".encode()).hexdigest()


def _binary_class_counts(target: IntArray) -> dict[str, int]:
    return {
        "0": int(np.count_nonzero(target == 0)),
        "1": int(np.count_nonzero(target == 1)),
    }


def _positive_scores(estimator: Any, features: Any) -> FloatArray:
    probabilities = estimator.predict_proba(features)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("binary classifier must return two probability columns")
    return np.asarray(probabilities[:, 1], dtype=float)


def _fit_probability_estimator(
    *,
    model_name: str,
    best_estimator: Any,
    x_train: Any,
    y_train: IntArray,
    calibration_folds: int,
) -> tuple[Any, str]:
    """Fit probability estimation without changing the locked classifier prediction."""
    if model_name != "rbf_svm":
        return best_estimator, "native_predict_proba"

    calibration_cv = max(2, min(3, calibration_folds))
    calibrated = CalibratedClassifierCV(
        estimator=clone(best_estimator),
        method="sigmoid",
        cv=calibration_cv,
        ensemble=False,
        n_jobs=1,
    )
    calibrated.fit(x_train, y_train)
    return calibrated, f"sigmoid_cv_{calibration_cv}_fold_outer_training_only"


def _json_safe_parameter(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _safe_params(params: dict[str, Any]) -> dict[str, Any]:
    return {str(key): _json_safe_parameter(value) for key, value in sorted(params.items())}


def _selected_features(estimator: Any, columns: Any) -> list[str]:
    selector = estimator.named_steps["selector"]
    support = selector.get_support()
    return [str(name) for name in columns[support]]


def _inner_search_rows(
    search: Any,
    *,
    outer_fold: int,
    model: str,
    search_profile: str,
) -> list[dict[str, Any]]:
    results = search.cv_results_
    rows: list[dict[str, Any]] = []
    for candidate_index, params in enumerate(results["params"]):
        rows.append(
            {
                "outer_fold": outer_fold,
                "search_profile": search_profile,
                "model": model,
                "candidate_index": candidate_index,
                "selected": candidate_index == int(search.best_index_),
                "rank_test_score": int(results["rank_test_score"][candidate_index]),
                "mean_test_score": float(results["mean_test_score"][candidate_index]),
                "std_test_score": float(results["std_test_score"][candidate_index]),
                "params": _safe_params(dict(params)),
            }
        )
    return rows


def _aggregate_outer_results(
    outer_rows: list[dict[str, Any]],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in outer_rows:
        grouped[str(row["model"])].append(row)

    summary: list[dict[str, Any]] = []
    for model, rows in grouped.items():
        aggregate: dict[str, Any] = {
            "model": model,
            "model_family": "classical",
            "outer_folds": len(rows),
        }
        for metric in _METRIC_NAMES:
            values = np.asarray([float(row[metric]) for row in rows], dtype=float)
            aggregate[f"{metric}_mean"] = float(np.mean(values))
            aggregate[f"{metric}_std"] = float(np.std(values, ddof=0))
        fit_values = np.asarray([float(row["fit_seconds"]) for row in rows], dtype=float)
        aggregate["fit_seconds_mean"] = float(np.mean(fit_values))
        aggregate["fit_seconds_std"] = float(np.std(fit_values, ddof=0))
        summary.append(aggregate)

    summary.sort(key=lambda row: float(row["balanced_accuracy_mean"]), reverse=True)
    attach_bootstrap_intervals(
        summary,
        outer_rows,
        confidence_level=_CONFIDENCE_LEVEL,
        n_resamples=_BOOTSTRAP_RESAMPLES,
        seed=seed,
    )
    return summary


def _attach_calibration_summary(
    summary: list[dict[str, Any]],
    calibration_summary: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_model = {str(row["model"]): row for row in calibration_summary}
    for row in summary:
        diagnostics = by_model[str(row["model"])]
        for key in (
            "prevalence",
            "mean_predicted_probability",
            "calibration_in_the_large",
            "expected_calibration_error",
            "maximum_calibration_error",
            "pooled_out_of_fold_brier_score",
            "pooled_out_of_fold_log_loss",
            "false_positive_count",
            "false_negative_count",
        ):
            row[key] = diagnostics[key]
    return summary


def _nested_pairwise_rows(
    y_true: IntArray,
    predictions: dict[str, IntArray],
    *,
    outer_fold: int,
    seed: int,
    test_index_hash: str,
    test_samples: int,
    search_profile: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_a, model_b in combinations(sorted(predictions), 2):
        comparison = exact_mcnemar_comparison(
            y_true,
            predictions[model_a],
            predictions[model_b],
            model_a=model_a,
            model_b=model_b,
            repeat=outer_fold,
            seed=seed,
            test_index_hash=test_index_hash,
            test_samples=test_samples,
            alpha=_SIGNIFICANCE_LEVEL,
        )
        comparison["outer_fold"] = comparison.pop("repeat")
        comparison["search_profile"] = search_profile
        rows.append(comparison)
    return rows


def _nested_pairwise_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    base = summarize_pairwise_comparisons(rows)
    summaries: list[dict[str, Any]] = []
    for row in base:
        summaries.append(
            {
                "model_a": row["model_a"],
                "model_b": row["model_b"],
                "outer_folds": row["repeats"],
                "significant_outer_folds": row["significant_repeats"],
                "model_a_statistically_favored_outer_folds": row[
                    "model_a_statistically_favored_repeats"
                ],
                "model_b_statistically_favored_outer_folds": row[
                    "model_b_statistically_favored_repeats"
                ],
                "model_a_more_correct_outer_folds": row["model_a_more_correct_repeats"],
                "model_b_more_correct_outer_folds": row["model_b_more_correct_repeats"],
                "equal_correctness_outer_folds": row["equal_correctness_repeats"],
                "minimum_exact_p_value": row["minimum_exact_p_value"],
                "pooled_p_value": None,
                "aggregation_note": (
                    "No pooled p-value is reported across outer folds. Descriptive direction "
                    "counts do not imply statistical significance."
                ),
            }
        )
    return summaries


def _evidence_statement(
    summary: list[dict[str, Any]],
    *,
    search_profile: str,
) -> dict[str, Any]:
    if not summary:
        raise ValueError("nested summary must contain at least one model")
    best = summary[0]
    runner_up = summary[1] if len(summary) > 1 else None
    profile_note = (
        "This is the locked classical reference profile."
        if search_profile == "reference-v1"
        else (
            "This sensitivity profile tests grid-boundary assumptions and does not replace "
            "the locked reference profile."
        )
    )
    return {
        "status": "classical_nested_cv",
        "primary_metric": "balanced_accuracy",
        "primary_endpoint_locked": True,
        "search_profile": search_profile,
        "best_model": str(best["model"]),
        "runner_up_model": None if runner_up is None else str(runner_up["model"]),
        "statement": (
            f"Among the evaluated classical models, {best['model']} had the highest mean "
            "outer-fold balanced accuracy. Hyperparameters were selected only within outer "
            f"training partitions. {profile_note} Fold-level intervals and calibration "
            "diagnostics are descriptive and do not replace external validation."
        ),
        "claim_boundary": "No clinical utility or quantum advantage is claimed.",
    }


def _dataset_payload(dataset: DatasetBundle) -> dict[str, Any]:
    return {
        "name": dataset.name,
        "source": dataset.source,
        "positive_class": dataset.positive_class,
        "samples_used": len(dataset.target),
        "feature_count_available": int(dataset.features.shape[1]),
        "fingerprint": dataset.fingerprint,
        "metadata": dataset.metadata,
    }


def run_nested_cv(
    config: NestedCVConfig,
    *,
    write_output: bool = True,
) -> dict[str, Any]:
    """Run classical nested cross-validation with untouched outer test folds."""
    config.validate()
    dataset = load_dataset(
        config.dataset,
        csv_path=config.csv_path,
        target_column=config.target_column,
        positive_label=config.positive_label,
    ).subset(config.max_samples, config.seed)

    target = dataset.target.to_numpy(dtype=int)
    minimum_class_count = min(_binary_class_counts(target).values())
    if minimum_class_count < config.outer_folds:
        raise ValueError("each class must contain at least outer_folds samples")

    feature_count = min(config.features, int(dataset.features.shape[1]))
    outer_cv = StratifiedKFold(
        n_splits=config.outer_folds,
        shuffle=True,
        random_state=config.seed,
    )

    outer_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    inner_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []

    for outer_fold, (train_indices_raw, test_indices_raw) in enumerate(
        outer_cv.split(dataset.features, target)
    ):
        train_indices = np.asarray(train_indices_raw, dtype=int)
        test_indices = np.asarray(test_indices_raw, dtype=int)
        fold_seed = config.seed + outer_fold
        inner_seed = config.seed + 10_000 + outer_fold

        x_train = dataset.features.iloc[train_indices]
        x_test = dataset.features.iloc[test_indices]
        y_train = target[train_indices]
        y_test = target[test_indices]
        if min(_binary_class_counts(y_train).values()) < config.inner_folds:
            raise ValueError("each outer training class must support the configured inner_folds")

        train_hash = _index_hash(train_indices)
        test_hash = _index_hash(test_indices)
        inner_cv = StratifiedKFold(
            n_splits=config.inner_folds,
            shuffle=True,
            random_state=inner_seed,
        )
        specs = build_nested_model_specs(
            seed=fold_seed,
            feature_count=feature_count,
            models=config.models,
            calibration_folds=config.inner_folds,
            search_profile=config.search_profile,
        )
        fold_predictions: dict[str, IntArray] = {}

        for model_name, spec in specs.items():
            search = GridSearchCV(
                estimator=spec.estimator,
                param_grid=spec.param_grid,
                scoring=config.primary_metric,
                cv=inner_cv,
                refit=True,
                n_jobs=1,
                error_score="raise",
                return_train_score=False,
            )
            started = time.perf_counter()
            search.fit(x_train, y_train)
            prediction = np.asarray(search.predict(x_test), dtype=int)
            probability_estimator, probability_score_source = _fit_probability_estimator(
                model_name=model_name,
                best_estimator=search.best_estimator_,
                x_train=x_train,
                y_train=y_train,
                calibration_folds=config.inner_folds,
            )
            score = _positive_scores(probability_estimator, x_test)
            fit_seconds = time.perf_counter() - started
            metrics = evaluate_binary_classifier(y_test, prediction, score)
            fold_predictions[model_name] = prediction

            best_index = int(search.best_index_)
            selected_features = _selected_features(
                search.best_estimator_,
                dataset.features.columns,
            )
            outer_rows.append(
                {
                    "outer_fold": outer_fold,
                    "search_profile": config.search_profile,
                    "seed": fold_seed,
                    "inner_seed": inner_seed,
                    "model": model_name,
                    "model_family": "classical",
                    "primary_metric": config.primary_metric,
                    "train_index_hash": train_hash,
                    "test_index_hash": test_hash,
                    "train_samples": len(train_indices),
                    "test_samples": len(test_indices),
                    "training_class_counts": _binary_class_counts(y_train),
                    "test_class_counts": _binary_class_counts(y_test),
                    "selected_features": selected_features,
                    "best_params": _safe_params(dict(search.best_params_)),
                    "inner_best_mean": float(search.best_score_),
                    "inner_best_std": float(search.cv_results_["std_test_score"][best_index]),
                    "candidate_count": len(search.cv_results_["params"]),
                    "probability_score_source": probability_score_source,
                    "fit_seconds": float(fit_seconds),
                    **metrics,
                }
            )
            inner_rows.extend(
                _inner_search_rows(
                    search,
                    outer_fold=outer_fold,
                    model=model_name,
                    search_profile=config.search_profile,
                )
            )

            for fold_sample_order, dataset_index in enumerate(test_indices):
                prediction_rows.append(
                    {
                        "outer_fold": outer_fold,
                        "search_profile": config.search_profile,
                        "model": model_name,
                        "fold_sample_order": fold_sample_order,
                        "sample_index_hash": _sample_index_hash(
                            dataset.fingerprint,
                            int(dataset_index),
                        ),
                        "y_true": int(y_test[fold_sample_order]),
                        "y_pred": int(prediction[fold_sample_order]),
                        "y_score": float(score[fold_sample_order]),
                    }
                )

        pairwise_rows.extend(
            _nested_pairwise_rows(
                y_test,
                fold_predictions,
                outer_fold=outer_fold,
                seed=fold_seed,
                test_index_hash=test_hash,
                test_samples=len(test_indices),
                search_profile=config.search_profile,
            )
        )

    calibration = build_calibration_diagnostics(
        prediction_rows,
        bins=config.calibration_bins,
    )
    summary = _aggregate_outer_results(outer_rows, seed=config.seed)
    summary = _attach_calibration_summary(summary, calibration["summary"])
    pairwise_summary = _nested_pairwise_summary(pairwise_rows)
    search_spaces = build_nested_model_specs(
        seed=config.seed,
        feature_count=feature_count,
        models=config.models,
        calibration_folds=config.inner_folds,
        search_profile=config.search_profile,
    )
    payload: dict[str, Any] = {
        "schema_version": "nested-cv-1.1",
        "project_version": "0.1.0",
        "evaluation_mode": "classical_nested_cross_validation",
        "generated_at": utc_now(),
        "research_use_only": True,
        "quantum_advantage_claimed": False,
        "config": config.to_dict(),
        "dataset": _dataset_payload(dataset),
        "methodology": {
            "primary_metric": config.primary_metric,
            "primary_endpoint_locked": True,
            "search_profile": config.search_profile,
            "reference_profile_immutable": True,
            "outer_splitter": "StratifiedKFold(shuffle=True)",
            "inner_splitter": "StratifiedKFold(shuffle=True)",
            "outer_folds": config.outer_folds,
            "inner_folds": config.inner_folds,
            "preprocessing_scope": "inside_pipeline_within_inner_search",
            "outer_test_usage": "single_final_evaluation_after_inner_selection",
            "grid_search_refit": True,
            "grid_search_n_jobs": 1,
            "classification_prediction_source": "selected_inner_search_pipeline",
            "svm_probability_calibration": "sigmoid_cv_on_outer_training_after_selection",
            "svm_calibration_changes_class_predictions": False,
            "model_search_spaces": {
                model: {key: list(values) for key, values in spec.param_grid.items()}
                for model, spec in search_spaces.items()
            },
            "confidence_intervals": {
                "confidence_level": _CONFIDENCE_LEVEL,
                "bootstrap_resamples": _BOOTSTRAP_RESAMPLES,
                "resampling_unit": "outer_fold_metric",
                "limitation": (
                    "Outer-fold intervals are descriptive and do not replace external validation."
                ),
            },
            "paired_tests": {
                "method": "exact_mcnemar_binomial",
                "alpha": _SIGNIFICANCE_LEVEL,
                "unit": "shared_outer_test_fold",
                "pooled_outer_fold_p_value_reported": False,
            },
            "calibration_diagnostics": {
                "scope": "pooled_out_of_fold_predictions",
                "binning_strategy": "uniform_probability_width",
                "configured_bins": config.calibration_bins,
                "each_sample_predicted_once_per_model": True,
                "interpretation": "descriptive_not_clinical_validation",
            },
        },
        "outer_fold_results": outer_rows,
        "outer_fold_predictions": prediction_rows,
        "inner_search_results": inner_rows,
        "summary": summary,
        "pairwise_comparisons": pairwise_rows,
        "pairwise_summary": pairwise_summary,
        "calibration_summary": calibration["summary"],
        "calibration_bins": calibration["reliability_bins"],
        "probability_distribution": calibration["probability_distribution"],
        "classification_errors": calibration["classification_errors"],
        "evidence_statement": _evidence_statement(
            summary,
            search_profile=config.search_profile,
        ),
        "environment": environment_metadata(),
    }
    if write_output:
        artifacts = write_nested_artifacts(payload, config.output_dir)
        payload["artifacts"] = {name: str(path) for name, path in artifacts.items()}
    return payload
