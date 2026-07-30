from __future__ import annotations

import pandas as pd
import pytest

from quantum_oncology_benchmark.data import (
    load_builtin_breast_cancer,
    load_csv_dataset,
)


def test_builtin_dataset_maps_malignant_to_positive_class() -> None:
    dataset = load_builtin_breast_cancer()
    assert dataset.target.name == "malignant"
    assert set(dataset.target.unique()) == {0, 1}
    assert dataset.positive_class == "malignant"
    assert int(dataset.target.sum()) == 212
    assert dataset.features.shape == (569, 30)
    assert len(dataset.fingerprint) == 64


def test_subset_is_reproducible_and_stratified() -> None:
    dataset = load_builtin_breast_cancer()
    first = dataset.subset(100, seed=7)
    second = dataset.subset(100, seed=7)
    assert first.fingerprint == second.fingerprint
    assert len(first.target) == 100
    assert 0.30 < first.target.mean() < 0.45


def test_csv_requires_positive_label_for_text_targets(tmp_path) -> None:
    path = tmp_path / "cohort.csv"
    pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0, 4.0],
            "feature_b": [4.0, 3.0, 2.0, 1.0],
            "diagnosis": ["benign", "malignant", "benign", "malignant"],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="positive_label"):
        load_csv_dataset(path, "diagnosis")

    dataset = load_csv_dataset(path, "diagnosis", "malignant")
    assert dataset.target.tolist() == [0, 1, 0, 1]
    assert dataset.positive_class == "malignant"
