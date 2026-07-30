from __future__ import annotations

import numpy as np

from quantum_oncology_benchmark.data import load_builtin_breast_cancer
from quantum_oncology_benchmark.preprocessing import prepare_split


def test_prepare_split_is_reproducible_and_bounded() -> None:
    dataset = load_builtin_breast_cancer().subset(120, seed=42)
    first = prepare_split(dataset, feature_count=4, test_size=0.25, seed=42)
    second = prepare_split(dataset, feature_count=4, test_size=0.25, seed=42)

    assert first.selected_features == second.selected_features
    assert np.array_equal(first.train_indices, second.train_indices)
    assert np.array_equal(first.test_indices, second.test_indices)
    assert first.x_train_classical.shape[1] == 4
    assert first.x_train_quantum.shape[1] == 4
    assert float(first.x_train_quantum.min()) >= 0.0
    assert float(first.x_train_quantum.max()) <= float(np.pi) + 1e-12
    assert not set(first.train_indices).intersection(set(first.test_indices))
