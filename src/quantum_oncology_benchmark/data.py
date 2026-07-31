"""Dataset loading, validation, and provenance."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


@dataclass(frozen=True, slots=True)
class DatasetBundle:
    """A validated binary classification dataset."""

    name: str
    features: pd.DataFrame
    target: pd.Series
    positive_class: str
    source: str
    metadata: dict[str, Any]

    @property
    def fingerprint(self) -> str:
        """Return a stable SHA-256 fingerprint of feature and target values."""
        feature_hash = pd.util.hash_pandas_object(self.features, index=True).to_numpy()
        target_hash = pd.util.hash_pandas_object(self.target, index=True).to_numpy()
        digest = sha256()
        digest.update(feature_hash.tobytes())
        digest.update(target_hash.tobytes())
        return digest.hexdigest()

    def subset(self, max_samples: int | None, seed: int) -> DatasetBundle:
        """Return a reproducible stratified subset."""
        if max_samples is None or max_samples >= len(self.target):
            return self
        indices = np.arange(len(self.target))
        selected, _ = train_test_split(
            indices,
            train_size=max_samples,
            random_state=seed,
            stratify=self.target,
        )
        selected = np.sort(selected)
        return DatasetBundle(
            name=self.name,
            features=self.features.iloc[selected].reset_index(drop=True),
            target=self.target.iloc[selected].reset_index(drop=True),
            positive_class=self.positive_class,
            source=self.source,
            metadata={**self.metadata, "subset_size": max_samples, "subset_seed": seed},
        )


def load_builtin_breast_cancer() -> DatasetBundle:
    """Load the public Wisconsin Diagnostic Breast Cancer demonstration dataset.

    Scikit-learn encodes malignant as 0 and benign as 1. This project remaps
    malignant to the positive class (1) so sensitivity refers to malignant-case
    detection.
    """
    raw = load_breast_cancer(as_frame=True)
    features = raw.data.copy()
    target = (raw.target == 0).astype(int).rename("malignant")
    return DatasetBundle(
        name="Wisconsin Diagnostic Breast Cancer",
        features=features,
        target=target,
        positive_class="malignant",
        source="scikit-learn packaged copy of the UCI WDBC dataset",
        metadata={
            "samples": int(features.shape[0]),
            "features": int(features.shape[1]),
            "original_target_mapping": {"0": "malignant", "1": "benign"},
            "benchmark_target_mapping": {"0": "benign", "1": "malignant"},
            "clinical_use": False,
        },
    )


def load_csv_dataset(
    path: str | Path,
    target_column: str,
    positive_label: str | int | float | None = None,
) -> DatasetBundle:
    """Load a user-supplied CSV containing numeric features and a binary target."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    frame = pd.read_csv(csv_path)
    if target_column not in frame.columns:
        raise ValueError(f"target column '{target_column}' was not found")

    raw_target = frame.pop(target_column)
    if raw_target.isna().any():
        raise ValueError("target column contains missing values")
    unique = list(pd.unique(raw_target))
    if len(unique) != 2:
        raise ValueError(f"target must contain exactly two classes, found {len(unique)}")

    if positive_label is None:
        normalized = set(unique)
        if normalized == {0, 1} or normalized == {0.0, 1.0}:
            target = raw_target.astype(int)
            positive_name = "1"
        else:
            raise ValueError(
                "positive_label is required when the target is not already encoded as 0 and 1"
            )
    else:
        matched = raw_target == positive_label
        if not bool(matched.any()):
            positive_as_text = str(positive_label)
            matched = raw_target.astype(str) == positive_as_text
        if not bool(matched.any()):
            raise ValueError(f"positive_label '{positive_label}' was not found in target values")
        target = matched.astype(int)
        positive_name = str(positive_label)

    numeric = frame.apply(pd.to_numeric, errors="coerce")
    all_missing_columns = [column for column in numeric.columns if numeric[column].isna().all()]
    if all_missing_columns:
        raise ValueError(f"non-numeric or empty feature columns: {all_missing_columns}")
    if numeric.shape[1] < 1:
        raise ValueError("at least one feature column is required")

    return DatasetBundle(
        name=csv_path.stem,
        features=numeric,
        target=target.rename(target_column),
        positive_class=positive_name,
        source=str(csv_path.resolve()),
        metadata={
            "samples": int(numeric.shape[0]),
            "features": int(numeric.shape[1]),
            "clinical_use": False,
            "user_supplied": True,
        },
    )


def load_dataset(
    dataset: str,
    csv_path: str | None = None,
    target_column: str | None = None,
    positive_label: str | int | float | None = None,
) -> DatasetBundle:
    """Load a supported dataset by identifier."""
    if dataset == "breast-cancer":
        return load_builtin_breast_cancer()
    if dataset == "csv":
        if csv_path is None or target_column is None:
            raise ValueError("csv_path and target_column are required for CSV datasets")
        return load_csv_dataset(csv_path, target_column, positive_label)
    raise ValueError(f"unsupported dataset: {dataset}")
