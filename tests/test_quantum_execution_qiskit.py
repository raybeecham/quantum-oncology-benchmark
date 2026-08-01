from __future__ import annotations

import numpy as np
import pytest

from quantum_oncology_benchmark.quantum_execution import (
    QiskitStatevectorKernelBackend,
    QuantumKernelSpecification,
    QuantumResourceBudget,
    validate_result,
)

pytestmark = pytest.mark.quantum


def test_qiskit_statevector_backend_produces_valid_bounded_kernels() -> None:
    x_train = np.asarray(
        [
            [0.1, 0.2],
            [0.4, 0.3],
            [0.7, 0.8],
        ],
        dtype=float,
    )
    x_test = np.asarray(
        [
            [0.2, 0.1],
            [0.8, 0.7],
        ],
        dtype=float,
    )
    specification = QuantumKernelSpecification(
        profile_name="statevector-adapter-test",
        feature_map_repetitions=1,
        shots=None,
        seed=7,
    )
    budget = QuantumResourceBudget(
        max_features=2,
        max_train_samples=3,
        max_test_samples=2,
        max_feature_map_repetitions=1,
        max_kernel_entries=15,
    )

    result = QiskitStatevectorKernelBackend().evaluate(
        specification,
        budget,
        x_train,
        x_test,
    )

    validate_result(result)
    assert result.train_kernel.shape == (3, 3)
    assert result.test_kernel.shape == (2, 3)
    assert result.resources.execution_mode == "exact_statevector"
    assert result.resources.physical_qpu is False
    assert result.resources.classical_simulation is True
    assert result.resources.logical_qubits == 2
    assert result.resources.kernel_entries == 15
    assert result.resources.estimated_total_shots == 0
    assert result.resources.circuit_depth > 0
    assert result.resources.circuit_size > 0
    assert len(result.resources.record_fingerprint) == 64
