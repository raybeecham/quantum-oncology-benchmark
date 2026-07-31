"""Statistical uncertainty and paired classifier comparison utilities."""

from __future__ import annotations

import warnings
from collections import defaultdict
from hashlib import sha256
from itertools import combinations
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import binomtest, bootstrap

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]

_BOOTSTRAP_METRICS = (
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


def _derived_seed(seed: int, *parts: str) -> int:
    payload = ":".join([str(seed), *parts]).encode("utf-8")
    return int.from_bytes(sha256(payload).digest()[:4], "big")


def bootstrap_mean_interval(
    values: FloatArray,
    *,
    confidence_level: float = 0.95,
    n_resamples: int = 5000,
    seed: int = 42,
) -> dict[str, Any]:
    """Return a deterministic BCa interval for a repeat-level mean.

    Repeated holdouts are the resampling unit. When fewer than two repeat-level
    observations are available, an interval is not estimated.
    """
    sample = np.asarray(values, dtype=float)
    if sample.ndim != 1:
        raise ValueError("bootstrap values must be one-dimensional")
    if len(sample) < 2:
        return {
            "available": False,
            "lower": None,
            "upper": None,
            "standard_error": None,
            "method": "not_estimated",
            "reason": "at least two repeat-level observations are required",
        }

    rng = np.random.default_rng(seed)
    method = "BCa"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = bootstrap(
            (sample,),
            np.mean,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            method=method,
            rng=rng,
        )

    lower = float(result.confidence_interval.low)
    upper = float(result.confidence_interval.high)
    standard_error = float(result.standard_error)
    if not np.isfinite([lower, upper, standard_error]).all():
        method = "percentile_fallback"
        rng = np.random.default_rng(seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = bootstrap(
                (sample,),
                np.mean,
                confidence_level=confidence_level,
                n_resamples=n_resamples,
                method="percentile",
                rng=rng,
            )
        lower = float(result.confidence_interval.low)
        upper = float(result.confidence_interval.high)
        standard_error = float(result.standard_error)

    return {
        "available": True,
        "lower": lower,
        "upper": upper,
        "standard_error": standard_error,
        "method": method,
        "reason": None,
    }


def attach_bootstrap_intervals(
    summary: list[dict[str, Any]],
    run_rows: list[dict[str, Any]],
    *,
    confidence_level: float = 0.95,
    n_resamples: int = 5000,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Attach repeat-level bootstrap intervals to aggregate model rows."""
    rows_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        rows_by_model[str(row["model"])].append(row)

    for aggregate in summary:
        model = str(aggregate["model"])
        model_rows = rows_by_model[model]
        for metric in _BOOTSTRAP_METRICS:
            values = np.asarray([float(row[metric]) for row in model_rows], dtype=float)
            interval = bootstrap_mean_interval(
                values,
                confidence_level=confidence_level,
                n_resamples=n_resamples,
                seed=_derived_seed(seed, model, metric),
            )
            aggregate[f"{metric}_ci_low"] = interval["lower"]
            aggregate[f"{metric}_ci_high"] = interval["upper"]
            aggregate[f"{metric}_ci_method"] = interval["method"]
    return summary


def exact_mcnemar_comparison(
    y_true: IntArray,
    prediction_a: IntArray,
    prediction_b: IntArray,
    *,
    model_a: str,
    model_b: str,
    repeat: int,
    seed: int,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Compare two classifiers on one shared test partition using exact McNemar."""
    truth = np.asarray(y_true, dtype=int)
    first = np.asarray(prediction_a, dtype=int)
    second = np.asarray(prediction_b, dtype=int)
    if truth.shape != first.shape or truth.shape != second.shape:
        raise ValueError("paired McNemar inputs must have identical shapes")

    first_correct = first == truth
    second_correct = second == truth
    a_correct_b_wrong = int(np.count_nonzero(first_correct & ~second_correct))
    a_wrong_b_correct = int(np.count_nonzero(~first_correct & second_correct))
    discordant = a_correct_b_wrong + a_wrong_b_correct
    p_value = (
        1.0
        if discordant == 0
        else float(
            binomtest(
                min(a_correct_b_wrong, a_wrong_b_correct),
                n=discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )
    )
    direction = "no_direction"
    if a_correct_b_wrong > a_wrong_b_correct:
        direction = model_a
    elif a_wrong_b_correct > a_correct_b_wrong:
        direction = model_b

    return {
        "repeat": repeat,
        "seed": seed,
        "model_a": model_a,
        "model_b": model_b,
        "model_a_correct_model_b_wrong": a_correct_b_wrong,
        "model_a_wrong_model_b_correct": a_wrong_b_correct,
        "discordant_pairs": discordant,
        "exact_p_value": p_value,
        "alpha": alpha,
        "significant": bool(p_value < alpha),
        "favored_model": direction if p_value < alpha else None,
        "method": "exact_mcnemar_binomial",
    }


def compare_repeat_predictions(
    y_true: IntArray,
    predictions: dict[str, IntArray],
    *,
    repeat: int,
    seed: int,
    alpha: float = 0.05,
) -> list[dict[str, Any]]:
    """Create all pairwise exact McNemar comparisons for one repeat."""
    rows: list[dict[str, Any]] = []
    for model_a, model_b in combinations(sorted(predictions), 2):
        rows.append(
            exact_mcnemar_comparison(
                y_true,
                predictions[model_a],
                predictions[model_b],
                model_a=model_a,
                model_b=model_b,
                repeat=repeat,
                seed=seed,
                alpha=alpha,
            )
        )
    return rows


def summarize_pairwise_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize per-repeat tests without pooling dependent repeated holdouts."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model_a"]), str(row["model_b"]))].append(row)

    summaries: list[dict[str, Any]] = []
    for (model_a, model_b), pair_rows in sorted(grouped.items()):
        summaries.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "repeats": len(pair_rows),
                "significant_repeats": sum(bool(row["significant"]) for row in pair_rows),
                "model_a_favored_repeats": sum(
                    row["favored_model"] == model_a for row in pair_rows
                ),
                "model_b_favored_repeats": sum(
                    row["favored_model"] == model_b for row in pair_rows
                ),
                "minimum_exact_p_value": min(float(row["exact_p_value"]) for row in pair_rows),
                "pooled_p_value": None,
                "aggregation_note": (
                    "No pooled p-value is reported because repeated holdouts can reuse observations."
                ),
            }
        )
    return summaries


def build_evidence_statement(
    summary: list[dict[str, Any]],
    pairwise_summary: list[dict[str, Any]],
) -> dict[str, Any]:
    """Generate a bounded, deterministic interpretation statement."""
    if not summary:
        raise ValueError("summary must contain at least one model")
    ranked = sorted(summary, key=lambda row: float(row["balanced_accuracy_mean"]), reverse=True)
    best = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    best_classical = next(
        (row for row in ranked if str(row["model_family"]) == "classical"),
        None,
    )
    best_quantum = next(
        (row for row in ranked if str(row["model_family"]) == "quantum_kernel"),
        None,
    )

    if best_quantum is None:
        statement = (
            f"Among the evaluated models, {best['model']} had the highest mean balanced "
            "accuracy. This classical-only experiment does not evaluate quantum advantage, "
            "and the repeat-level intervals are descriptive rather than external validation."
        )
        status = "classical_only"
    elif best_classical is None:
        statement = (
            f"{best_quantum['model']} was evaluated without a classical comparator. The result "
            "cannot support a quantum-advantage claim."
        )
        status = "quantum_without_classical_comparator"
    else:
        quantum_mean = float(best_quantum["balanced_accuracy_mean"])
        classical_mean = float(best_classical["balanced_accuracy_mean"])
        if quantum_mean <= classical_mean:
            statement = (
                f"The strongest quantum model ({best_quantum['model']}) did not outperform the "
                f"strongest classical baseline ({best_classical['model']}) in mean balanced "
                "accuracy. No quantum advantage was demonstrated under the configured conditions."
            )
            status = "no_quantum_advantage"
        else:
            statement = (
                f"The strongest quantum model ({best_quantum['model']}) had a higher mean balanced "
                f"accuracy than {best_classical['model']}, but repeated-holdout comparisons are not "
                "independent external validation. The result is an exploratory signal and does not "
                "establish quantum advantage."
            )
            status = "exploratory_quantum_signal"

    return {
        "status": status,
        "primary_metric": "balanced_accuracy",
        "best_model": str(best["model"]),
        "runner_up_model": None if runner_up is None else str(runner_up["model"]),
        "best_classical_model": None if best_classical is None else str(best_classical["model"]),
        "best_quantum_model": None if best_quantum is None else str(best_quantum["model"]),
        "statement": statement,
        "pairwise_summary_available": bool(pairwise_summary),
        "claim_boundary": "No clinical utility or quantum advantage is claimed.",
    }
