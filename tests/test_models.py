from __future__ import annotations

import warnings

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC

from quantum_oncology_benchmark.models.classical import build_classical_models


def test_classical_svm_uses_training_only_calibration() -> None:
    model = build_classical_models(seed=42)["rbf_svm"]

    assert isinstance(model, CalibratedClassifierCV)
    assert model.cv == 5
    assert model.ensemble is False
    assert isinstance(model.estimator, SVC)
    assert model.estimator.kernel == "rbf"

    x_train = np.array(
        [[-2.0], [-1.5], [-1.0], [-0.5], [0.5], [1.0], [1.5], [2.0], [2.5], [3.0]]
    )
    y_train = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(np.array([[-0.75], [2.25]]))

    assert probabilities.shape == (2, 2)
