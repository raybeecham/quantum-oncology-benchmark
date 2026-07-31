"""Calibration and error diagnostics for out-of-fold predictions."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.metrics import log_loss


def _bin_index(score: float, bins: int) -> int:
    if not 0.0 <= score <= 1.0:
        raise ValueError("probability scores must be between 0 and 1")
    return min(int(score * bins), bins - 1)


def build_calibration_diagnostics(
    prediction_rows: list[dict[str, Any]],
    *,
    bins: int = 10,
) -> dict[str, list[dict[str, Any]]]:
    """Build deterministic diagnostics from one out-of-fold prediction per sample."""
    if bins < 2:
        raise ValueError("calibration bins must be at least 2")
    if not prediction_rows:
        raise ValueError("prediction_rows must not be empty")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        grouped[str(row["model"])].append(row)

    summary_rows: list[dict[str, Any]] = []
    reliability_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []

    for model, rows in sorted(grouped.items()):
        truth = np.asarray([int(row["y_true"]) for row in rows], dtype=int)
        prediction = np.asarray([int(row["y_pred"]) for row in rows], dtype=int)
        score = np.asarray([float(row["y_score"]) for row in rows], dtype=float)
        if not np.isin(truth, [0, 1]).all() or not np.isin(prediction, [0, 1]).all():
            raise ValueError("calibration diagnostics require binary truth and predictions")
        if not np.isfinite(score).all() or np.any((score < 0.0) | (score > 1.0)):
            raise ValueError("calibration diagnostics require finite probabilities in [0, 1]")

        total = len(rows)
        prevalence = float(np.mean(truth))
        mean_probability = float(np.mean(score))
        expected_calibration_error = 0.0
        maximum_calibration_error = 0.0

        for bin_number in range(bins):
            lower = bin_number / bins
            upper = (bin_number + 1) / bins
            indices = np.asarray([_bin_index(value, bins) == bin_number for value in score])
            count = int(np.count_nonzero(indices))
            if count == 0:
                continue
            bin_scores = score[indices]
            bin_truth = truth[indices]
            mean_predicted = float(np.mean(bin_scores))
            observed_rate = float(np.mean(bin_truth))
            gap = abs(mean_predicted - observed_rate)
            expected_calibration_error += (count / total) * gap
            maximum_calibration_error = max(maximum_calibration_error, gap)
            reliability_rows.append(
                {
                    "model": model,
                    "bin": bin_number,
                    "bin_lower": lower,
                    "bin_upper": upper,
                    "right_closed": bin_number == bins - 1,
                    "samples": count,
                    "positive_samples": int(np.count_nonzero(bin_truth == 1)),
                    "negative_samples": int(np.count_nonzero(bin_truth == 0)),
                    "mean_predicted_probability": mean_predicted,
                    "observed_positive_rate": observed_rate,
                    "absolute_calibration_gap": gap,
                    "fraction_of_model_predictions": count / total,
                }
            )

        false_positive_count = int(np.count_nonzero((truth == 0) & (prediction == 1)))
        false_negative_count = int(np.count_nonzero((truth == 1) & (prediction == 0)))
        brier_score = float(np.mean(np.square(score - truth)))
        summary_rows.append(
            {
                "model": model,
                "samples": total,
                "positive_samples": int(np.count_nonzero(truth == 1)),
                "negative_samples": int(np.count_nonzero(truth == 0)),
                "prevalence": prevalence,
                "mean_predicted_probability": mean_probability,
                "calibration_in_the_large": mean_probability - prevalence,
                "expected_calibration_error": expected_calibration_error,
                "maximum_calibration_error": maximum_calibration_error,
                "pooled_out_of_fold_brier_score": brier_score,
                "pooled_out_of_fold_log_loss": float(log_loss(truth, score, labels=[0, 1])),
                "false_positive_count": false_positive_count,
                "false_negative_count": false_negative_count,
                "binning_strategy": "uniform_probability_width",
                "configured_bins": bins,
            }
        )

        for true_class in (0, 1):
            class_mask = truth == true_class
            class_total = int(np.count_nonzero(class_mask))
            for bin_number in range(bins):
                indices = class_mask & np.asarray(
                    [_bin_index(value, bins) == bin_number for value in score]
                )
                count = int(np.count_nonzero(indices))
                distribution_rows.append(
                    {
                        "model": model,
                        "true_class": true_class,
                        "bin": bin_number,
                        "bin_lower": bin_number / bins,
                        "bin_upper": (bin_number + 1) / bins,
                        "right_closed": bin_number == bins - 1,
                        "samples": count,
                        "fraction_within_true_class": 0.0
                        if class_total == 0
                        else count / class_total,
                        "fraction_of_model_predictions": count / total,
                    }
                )

        for row in rows:
            true_value = int(row["y_true"])
            predicted_value = int(row["y_pred"])
            if true_value == predicted_value:
                continue
            positive_probability = float(row["y_score"])
            error_rows.append(
                {
                    "outer_fold": int(row["outer_fold"]),
                    "search_profile": str(row["search_profile"]),
                    "model": model,
                    "sample_index_hash": str(row["sample_index_hash"]),
                    "y_true": true_value,
                    "y_pred": predicted_value,
                    "positive_class_probability": positive_probability,
                    "predicted_class_confidence": positive_probability
                    if predicted_value == 1
                    else 1.0 - positive_probability,
                    "error_type": "false_negative" if true_value == 1 else "false_positive",
                }
            )

    return {
        "summary": summary_rows,
        "reliability_bins": reliability_rows,
        "probability_distribution": distribution_rows,
        "classification_errors": error_rows,
    }
