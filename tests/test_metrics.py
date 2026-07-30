from __future__ import annotations

import numpy as np

from quantum_oncology_benchmark.metrics import evaluate_binary_classifier


def test_specificity_and_sensitivity_are_calculated_for_positive_class_one() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_score = np.array([0.1, 0.6, 0.7, 0.9])

    metrics = evaluate_binary_classifier(y_true, y_pred, y_score)

    assert metrics["specificity"] == 0.5
    assert metrics["sensitivity"] == 1.0
    assert metrics["true_negative"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 0
    assert metrics["true_positive"] == 2
