"""Strong classical comparison models."""

from __future__ import annotations

from typing import Any

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def build_classical_models(seed: int) -> dict[str, Any]:
    """Return diverse, reasonably strong classical baselines."""
    rbf_svm = SVC(
        kernel="rbf",
        class_weight="balanced",
        random_state=seed,
    )
    return {
        "logistic_regression": LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="liblinear",
            random_state=seed,
        ),
        "rbf_svm": CalibratedClassifierCV(
            estimator=rbf_svm,
            method="sigmoid",
            cv=5,
            ensemble=False,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=seed,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=250,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=seed,
        ),
    }
