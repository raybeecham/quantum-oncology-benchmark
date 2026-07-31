from __future__ import annotations

import pytest

from quantum_oncology_benchmark.config import ExperimentConfig, NestedCVConfig


def test_config_rejects_invalid_quantum_shots() -> None:
    with pytest.raises(ValueError, match="quantum_shots"):
        ExperimentConfig(quantum_shots=0).validate()


def test_csv_config_requires_path_and_target() -> None:
    with pytest.raises(ValueError, match="csv_path"):
        ExperimentConfig(dataset="csv").validate()


def test_nested_config_locks_primary_metric() -> None:
    with pytest.raises(ValueError, match="primary_metric"):
        NestedCVConfig(primary_metric="accuracy").validate()


def test_nested_config_rejects_invalid_folds_and_models() -> None:
    with pytest.raises(ValueError, match="outer_folds"):
        NestedCVConfig(outer_folds=1).validate()
    with pytest.raises(ValueError, match="unsupported nested-cv models"):
        NestedCVConfig(models=("unsupported",)).validate()


def test_nested_config_rejects_unknown_profile_and_bins() -> None:
    with pytest.raises(ValueError, match="search_profile"):
        NestedCVConfig(search_profile="unversioned").validate()
    with pytest.raises(ValueError, match="calibration_bins"):
        NestedCVConfig(calibration_bins=1).validate()
