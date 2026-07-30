"""Leakage-resistant preprocessing for classical and quantum models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .data import DatasetBundle


@dataclass(frozen=True, slots=True)
class PreparedSplit:
    """Preprocessed train/test arrays sharing the same selected features."""

    x_train_classical: np.ndarray
    x_test_classical: np.ndarray
    x_train_quantum: np.ndarray
    x_test_quantum: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    selected_features: tuple[str, ...]
    train_indices: np.ndarray
    test_indices: np.ndarray


def prepare_split(
    dataset: DatasetBundle,
    *,
    feature_count: int,
    test_size: float,
    seed: int,
) -> PreparedSplit:
    """Split first, then fit all preprocessing on the training partition only."""
    indices = np.arange(len(dataset.target))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=dataset.target.to_numpy(),
    )

    x_train = dataset.features.iloc[train_indices]
    x_test = dataset.features.iloc[test_indices]
    y_train = dataset.target.iloc[train_indices].to_numpy(dtype=int)
    y_test = dataset.target.iloc[test_indices].to_numpy(dtype=int)

    k = min(feature_count, dataset.features.shape[1])
    classical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("selector", SelectKBest(score_func=f_classif, k=k)),
        ]
    )
    x_train_classical = classical_pipeline.fit_transform(x_train, y_train)
    x_test_classical = classical_pipeline.transform(x_test)

    selector = classical_pipeline.named_steps["selector"]
    support = selector.get_support()
    selected_features = tuple(str(name) for name in dataset.features.columns[support])

    quantum_scaler = MinMaxScaler(feature_range=(0.0, float(np.pi)), clip=True)
    x_train_quantum = quantum_scaler.fit_transform(x_train_classical)
    x_test_quantum = quantum_scaler.transform(x_test_classical)

    return PreparedSplit(
        x_train_classical=np.asarray(x_train_classical, dtype=float),
        x_test_classical=np.asarray(x_test_classical, dtype=float),
        x_train_quantum=np.asarray(x_train_quantum, dtype=float),
        x_test_quantum=np.asarray(x_test_quantum, dtype=float),
        y_train=y_train,
        y_test=y_test,
        selected_features=selected_features,
        train_indices=np.asarray(train_indices, dtype=int),
        test_indices=np.asarray(test_indices, dtype=int),
    )
