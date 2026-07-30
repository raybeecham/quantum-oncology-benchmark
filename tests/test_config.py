from __future__ import annotations

import pytest

from quantum_oncology_benchmark.config import ExperimentConfig


def test_config_rejects_invalid_quantum_shots() -> None:
    with pytest.raises(ValueError, match="quantum_shots"):
        ExperimentConfig(quantum_shots=0).validate()


def test_csv_config_requires_path_and_target() -> None:
    with pytest.raises(ValueError, match="csv_path"):
        ExperimentConfig(dataset="csv").validate()
