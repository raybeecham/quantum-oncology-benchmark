"""Benchmark orchestration."""

from __future__ import annotations

import time
from collections import defaultdict
from hashlib import sha256
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config import ExperimentConfig
from .data import load_dataset
from .metrics import evaluate_binary_classifier
from .models.classical import build_classical_models
from .models.quantum_kernel import QuantumKernelClassifier
from .preprocessing import prepare_split
from .reporting import environment_metadata, utc_now, write_artifacts
from .statistical import (
    attach_bootstrap_intervals,
    build_evidence_statement,
    compare_repeat_predictions,
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
    "elapsed_seconds",
)

_CONFIDENCE_LEVEL = 0.95
_BOOTSTRAP_RESAMPLES = 5000
_SIGNIFICANCE_LEVEL = 0.05


def _positive_scores(model: Any, x_test: FloatArray) -> FloatArray:
    probabilities = model.predict_proba(x_test)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("binary classifier must return two probability columns")
    return np.asarray(probabilities[:, 1], dtype=float)


def _index_hash(indices: IntArray) -> str:
    """Hash partition membership without exposing raw row indices."""
    canonical = ",".join(str(int(index)) for index in indices)
    return sha256(canonical.encode("utf-8")).hexdigest()


def _binary_class_counts(target: IntArray) -> dict[str, int]:
    return {
        "0": int(np.count_nonzero(target == 0)),
        "1": int(np.count_nonzero(target == 1)),
    }


def run_benchmark(config: ExperimentConfig, *, write_output: bool = True) -> dict[str, Any]:
    """Run classical and optional quantum benchmarks with shared data splits."""
    config.validate()
    dataset = load_dataset(
        config.dataset,
        csv_path=config.csv_path,
        target_column=config.target_column,
        positive_label=config.positive_label,
    )

    effective_max_samples = config.max_samples
    if config.model_set in {"quantum", "all"} and effective_max_samples is None:
        effective_max_samples = 160
    dataset = dataset.subset(effective_max_samples, config.seed)

    run_rows: list[dict[str, Any]] = []
    split_provenance: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    selected_features: tuple[str, ...] = ()
    quantum_resources: dict[str, Any] | None = None

    for repeat_index in range(config.repeats):
        repeat_seed = config.seed + repeat_index
        split = prepare_split(
            dataset,
            feature_count=config.features,
            test_size=config.test_size,
            seed=repeat_seed,
        )
        if not selected_features:
            selected_features = split.selected_features

        split_provenance.append(
            {
                "repeat": repeat_index,
                "seed": repeat_seed,
                "selected_features": list(split.selected_features),
                "train_index_hash": _index_hash(split.train_indices),
                "test_index_hash": _index_hash(split.test_indices),
                "train_samples": len(split.y_train),
                "test_samples": len(split.y_test),
                "training_class_counts": _binary_class_counts(split.y_train),
                "test_class_counts": _binary_class_counts(split.y_test),
                "preprocessing_fit_scope": "training_partition_only",
            }
        )

        repeat_predictions: dict[str, IntArray] = {}
        if config.model_set in {"classical", "all"}:
            for model_name, model in build_classical_models(repeat_seed).items():
                started = time.perf_counter()
                model.fit(split.x_train_classical, split.y_train)
                prediction = np.asarray(model.predict(split.x_test_classical), dtype=int)
                score = _positive_scores(model, split.x_test_classical)
                elapsed = time.perf_counter() - started
                metrics = evaluate_binary_classifier(split.y_test, prediction, score)
                repeat_predictions[model_name] = prediction
                run_rows.append(
                    {
                        "repeat": repeat_index,
                        "seed": repeat_seed,
                        "model": model_name,
                        "model_family": "classical",
                        "elapsed_seconds": float(elapsed),
                        **metrics,
                    }
                )

        if config.model_set in {"quantum", "all"}:
            quantum_model = QuantumKernelClassifier(
                reps=config.quantum_reps,
                shots=config.quantum_shots,
                seed=repeat_seed,
            )
            started = time.perf_counter()
            quantum_model.fit(split.x_train_quantum, split.y_train)
            prediction, score = quantum_model.predict_with_scores(split.x_test_quantum)
            elapsed = time.perf_counter() - started
            metrics = evaluate_binary_classifier(split.y_test, prediction, score)
            repeat_predictions["quantum_fidelity_svm"] = prediction
            run_rows.append(
                {
                    "repeat": repeat_index,
                    "seed": repeat_seed,
                    "model": "quantum_fidelity_svm",
                    "model_family": "quantum_kernel",
                    "elapsed_seconds": float(elapsed),
                    **metrics,
                }
            )
            if quantum_resources is None:
                quantum_resources = quantum_model.resource_summary()

        pairwise_rows.extend(
            compare_repeat_predictions(
                split.y_test,
                repeat_predictions,
                repeat=repeat_index,
                seed=repeat_seed,
                alpha=_SIGNIFICANCE_LEVEL,
            )
        )

    summary = _aggregate_runs(run_rows)
    summary = attach_bootstrap_intervals(
        summary,
        run_rows,
        confidence_level=_CONFIDENCE_LEVEL,
        n_resamples=_BOOTSTRAP_RESAMPLES,
        seed=config.seed,
    )
    pairwise_summary = summarize_pairwise_comparisons(pairwise_rows)
    evidence_statement = build_evidence_statement(summary, pairwise_summary)
    payload: dict[str, Any] = {
        "schema_version": "1.2",
        "project_version": "0.1.0",
        "generated_at": utc_now(),
        "research_use_only": True,
        "quantum_advantage_claimed": False,
        "config": {**config.to_dict(), "max_samples": effective_max_samples},
        "dataset": {
            "name": dataset.name,
            "source": dataset.source,
            "positive_class": dataset.positive_class,
            "samples_used": len(dataset.target),
            "feature_count_available": int(dataset.features.shape[1]),
            "fingerprint": dataset.fingerprint,
            "metadata": dataset.metadata,
        },
        "selected_features": list(selected_features),
        "split_provenance": split_provenance,
        "runs": run_rows,
        "summary": summary,
        "statistical_analysis": {
            "confidence_intervals": {
                "confidence_level": _CONFIDENCE_LEVEL,
                "bootstrap_resamples": _BOOTSTRAP_RESAMPLES,
                "resampling_unit": "repeat_level_metric",
                "default_method": "BCa",
                "limitation": (
                    "Repeat-level intervals are descriptive and do not replace external validation."
                ),
            },
            "paired_tests": {
                "method": "exact_mcnemar_binomial",
                "alpha": _SIGNIFICANCE_LEVEL,
                "unit": "shared_test_partition_within_repeat",
                "pooled_repeated_holdout_p_value_reported": False,
            },
            "pairwise_comparisons": pairwise_rows,
            "pairwise_summary": pairwise_summary,
        },
        "evidence_statement": evidence_statement,
        "quantum_resources": quantum_resources,
        "environment": environment_metadata(),
    }
    if write_output:
        artifacts = write_artifacts(payload, config.output_dir)
        payload["artifacts"] = {name: str(path) for name, path in artifacts.items()}
    return payload


def _aggregate_runs(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped[str(row["model"])].append(row)

    summary: list[dict[str, Any]] = []
    for model_name, rows in grouped.items():
        aggregate: dict[str, Any] = {
            "model": model_name,
            "model_family": rows[0]["model_family"],
            "runs": len(rows),
        }
        for metric in _METRIC_NAMES:
            values = np.asarray([float(row[metric]) for row in rows], dtype=float)
            aggregate[f"{metric}_mean"] = float(np.mean(values))
            aggregate[f"{metric}_std"] = float(np.std(values, ddof=0))
        summary.append(aggregate)

    summary.sort(key=lambda row: float(row["balanced_accuracy_mean"]), reverse=True)
    return summary
