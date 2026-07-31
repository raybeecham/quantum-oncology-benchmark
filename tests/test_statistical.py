from __future__ import annotations

import numpy as np

from quantum_oncology_benchmark.statistical import (
    bootstrap_mean_interval,
    build_evidence_statement,
    exact_mcnemar_comparison,
    summarize_pairwise_comparisons,
)


def test_bootstrap_mean_interval_is_reproducible() -> None:
    values = np.array([0.81, 0.84, 0.86, 0.88, 0.90], dtype=float)
    first = bootstrap_mean_interval(values, n_resamples=1000, seed=7)
    second = bootstrap_mean_interval(values, n_resamples=1000, seed=7)

    assert first == second
    assert first["available"] is True
    assert float(first["lower"]) <= float(values.mean()) <= float(first["upper"])


def test_bootstrap_interval_requires_multiple_repeat_values() -> None:
    interval = bootstrap_mean_interval(np.array([0.9]), n_resamples=100, seed=1)

    assert interval["available"] is False
    assert interval["lower"] is None
    assert interval["method"] == "not_estimated"


def test_exact_mcnemar_uses_only_discordant_pairs() -> None:
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    prediction_a = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    prediction_b = np.array([0, 0, 0, 1, 1, 0, 0, 0])

    result = exact_mcnemar_comparison(
        y_true,
        prediction_a,
        prediction_b,
        model_a="a",
        model_b="b",
        repeat=0,
        seed=42,
        test_index_hash="a" * 64,
        test_samples=8,
    )

    assert result["test_index_hash"] == "a" * 64
    assert result["test_samples"] == 8
    assert result["model_a_correct_model_b_wrong"] == 4
    assert result["model_a_wrong_model_b_correct"] == 0
    assert result["discordant_pairs"] == 4
    assert result["descriptive_direction"] == "a"
    assert result["exact_p_value"] == 0.125
    assert result["significant"] is False
    assert result["favored_model"] is None


def test_pairwise_summary_separates_direction_from_significance() -> None:
    rows = [
        {
            "model_a": "a",
            "model_b": "b",
            "significant": True,
            "favored_model": "a",
            "descriptive_direction": "a",
            "exact_p_value": 0.01,
        },
        {
            "model_a": "a",
            "model_b": "b",
            "significant": False,
            "favored_model": None,
            "descriptive_direction": "b",
            "exact_p_value": 0.5,
        },
        {
            "model_a": "a",
            "model_b": "b",
            "significant": False,
            "favored_model": None,
            "descriptive_direction": "no_direction",
            "exact_p_value": 1.0,
        },
    ]

    summary = summarize_pairwise_comparisons(rows)[0]

    assert summary["pooled_p_value"] is None
    assert summary["significant_repeats"] == 1
    assert summary["model_a_statistically_favored_repeats"] == 1
    assert summary["model_b_statistically_favored_repeats"] == 0
    assert summary["model_a_more_correct_repeats"] == 1
    assert summary["model_b_more_correct_repeats"] == 1
    assert summary["equal_correctness_repeats"] == 1
    assert "Descriptive direction counts" in summary["aggregation_note"]


def test_evidence_statement_refuses_quantum_advantage_claim() -> None:
    summary = [
        {
            "model": "quantum_fidelity_svm",
            "model_family": "quantum_kernel",
            "balanced_accuracy_mean": 0.91,
        },
        {
            "model": "logistic_regression",
            "model_family": "classical",
            "balanced_accuracy_mean": 0.90,
        },
    ]

    evidence = build_evidence_statement(summary, [])

    assert evidence["status"] == "exploratory_quantum_signal"
    assert "does not establish quantum advantage" in evidence["statement"]
    assert "No clinical utility or quantum advantage" in evidence["claim_boundary"]
