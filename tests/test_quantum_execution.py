from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from quantum_oncology_benchmark.quantum_execution import (
    QuantumKernelSpecification,
    QuantumProtocolConfig,
    QuantumResourceBudget,
    QiskitStatevectorKernelBackend,
    ResourceBudgetExceeded,
    built_in_quantum_backends,
    validate_resource_request,
)


def test_quantum_protocol_configuration_loads_and_locks_claim_fields() -> None:
    config = QuantumProtocolConfig.from_yaml("configs/quantum-protocol-statevector.yaml")

    assert config.specification.protocol_version == "quantum-protocol-v1"
    assert config.specification.primary_metric == "balanced_accuracy"
    assert config.specification.classical_comparator == "logistic_regression"
    assert config.specification.execution_mode == "exact_statevector"
    assert config.budget.hardware_execution_allowed is False
    assert config.to_dict()["budget"]["max_features"] == 4


def test_quantum_protocol_rejects_unlocked_endpoint_and_comparator() -> None:
    with pytest.raises(ValueError, match="primary_metric"):
        QuantumKernelSpecification(primary_metric="accuracy").validate()
    with pytest.raises(ValueError, match="classical_comparator"):
        QuantumKernelSpecification(classical_comparator="rbf_svm").validate()


def test_resource_budget_is_enforced_before_backend_execution() -> None:
    specification = QuantumKernelSpecification(feature_map_repetitions=2)
    budget = QuantumResourceBudget(
        max_features=2,
        max_train_samples=4,
        max_test_samples=2,
        max_feature_map_repetitions=2,
        max_kernel_entries=20,
    )
    x_train = np.zeros((5, 2), dtype=float)
    x_test = np.zeros((1, 2), dtype=float)

    with pytest.raises(ResourceBudgetExceeded, match="train samples"):
        validate_resource_request(specification, budget, x_train, x_test)


def test_resource_budget_accounts_for_finite_shots() -> None:
    specification = QuantumKernelSpecification(shots=100)
    budget = QuantumResourceBudget(
        max_features=2,
        max_train_samples=4,
        max_test_samples=2,
        max_feature_map_repetitions=2,
        max_kernel_entries=20,
        max_shots_per_kernel_entry=200,
        max_total_shots=1_000,
    )
    x_train = np.zeros((3, 2), dtype=float)
    x_test = np.zeros((2, 2), dtype=float)

    with pytest.raises(ResourceBudgetExceeded, match="total shots"):
        validate_resource_request(specification, budget, x_train, x_test)


def test_resource_request_returns_declared_accounting() -> None:
    specification = QuantumKernelSpecification(feature_map_repetitions=1)
    budget = QuantumResourceBudget(
        max_features=2,
        max_train_samples=4,
        max_test_samples=2,
        max_feature_map_repetitions=1,
        max_kernel_entries=20,
    )
    x_train = np.zeros((3, 2), dtype=float)
    x_test = np.zeros((2, 2), dtype=float)

    accounting = validate_resource_request(specification, budget, x_train, x_test)

    assert accounting == {
        "feature_count": 2,
        "train_samples": 3,
        "test_samples": 2,
        "kernel_entries": 15,
        "estimated_total_shots": 0,
    }


def test_statevector_backend_capabilities_do_not_claim_hardware_support() -> None:
    backend = QiskitStatevectorKernelBackend()
    capabilities = backend.capabilities()

    assert capabilities.backend_id == "qiskit-fidelity-statevector"
    assert capabilities.physical_qpu is False
    assert capabilities.execution_modes == ("exact_statevector", "shot_statevector")
    assert capabilities.supports_calibration_snapshots is False
    assert capabilities.supports_cost_records is False
    assert built_in_quantum_backends()[backend.backend_id] is not None


def test_protocol_configuration_rejects_budget_mismatch() -> None:
    config = QuantumProtocolConfig(
        specification=QuantumKernelSpecification(feature_map_repetitions=2),
        budget=replace(QuantumResourceBudget(), max_feature_map_repetitions=1),
    )

    with pytest.raises(ValueError, match="feature-map repetitions"):
        config.validate()
