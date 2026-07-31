from __future__ import annotations

import sys
import warnings
from types import ModuleType, SimpleNamespace

import numpy as np

from quantum_oncology_benchmark.models.quantum_kernel import QuantumKernelClassifier


class _FakeFeatureMap:
    num_qubits = 2
    num_parameters = 2

    def count_ops(self) -> dict[str, int]:
        return {"h": 2, "rzz": 1}

    def depth(self) -> int:
        return 3

    def size(self) -> int:
        return 5


class _FakeKernel:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs

    def evaluate(self, x_vec: np.ndarray, y_vec: np.ndarray | None = None) -> np.ndarray:
        left = np.asarray(x_vec, dtype=float)
        right = left if y_vec is None else np.asarray(y_vec, dtype=float)
        squared_distance = np.sum((left[:, None, :] - right[None, :, :]) ** 2, axis=2)
        return np.exp(-squared_distance)


def test_quantum_classifier_wiring_with_fake_qiskit(monkeypatch) -> None:
    qiskit = ModuleType("qiskit")
    qiskit_circuit = ModuleType("qiskit.circuit")
    qiskit_library = ModuleType("qiskit.circuit.library")
    qiskit_library.zz_feature_map = lambda **kwargs: _FakeFeatureMap()  # type: ignore[attr-defined]

    qml = ModuleType("qiskit_machine_learning")
    qml_kernels = ModuleType("qiskit_machine_learning.kernels")
    qml_kernels.FidelityStatevectorKernel = _FakeKernel  # type: ignore[attr-defined]
    qml_utils = ModuleType("qiskit_machine_learning.utils")
    qml_utils.algorithm_globals = SimpleNamespace(random_seed=None)  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "qiskit", qiskit)
    monkeypatch.setitem(sys.modules, "qiskit.circuit", qiskit_circuit)
    monkeypatch.setitem(sys.modules, "qiskit.circuit.library", qiskit_library)
    monkeypatch.setitem(sys.modules, "qiskit_machine_learning", qml)
    monkeypatch.setitem(sys.modules, "qiskit_machine_learning.kernels", qml_kernels)
    monkeypatch.setitem(sys.modules, "qiskit_machine_learning.utils", qml_utils)

    x_train = np.array([[0.0, 0.0], [0.2, 0.1], [2.8, 3.0], [3.0, 2.9]])
    y_train = np.array([0, 0, 1, 1])
    x_test = np.array([[0.1, 0.1], [2.9, 2.9]])

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        model = QuantumKernelClassifier(reps=1, seed=7).fit(x_train, y_train)
        predictions, scores = model.predict_with_scores(x_test)
    resources = model.resource_summary()

    assert predictions.tolist() == [0, 1]
    assert scores.shape == (2,)
    assert np.all((scores >= 0.0) & (scores <= 1.0))
    assert resources["qubits"] == 2
    assert resources["two_qubit_operations"] == 1
    assert resources["physical_qpu"] is False
    assert resources["calibration_folds"] == 2
    assert resources["probability_calibration"] == "training_only_stratified_oof_sigmoid"
