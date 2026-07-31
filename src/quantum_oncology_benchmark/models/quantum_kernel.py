"""Quantum-kernel model implemented with Qiskit Machine Learning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.svm import SVC

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]


class QuantumDependencyError(RuntimeError):
    """Raised when optional quantum dependencies are unavailable."""


def quantum_dependencies_available() -> bool:
    """Return whether Qiskit quantum-kernel dependencies can be imported."""
    try:
        import qiskit  # noqa: F401
        import qiskit_machine_learning  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass(slots=True)
class QuantumKernelClassifier:
    """SVC operating on an exact or shot-sampled quantum fidelity kernel.

    The default implementation is a classical statevector simulation. It does
    not represent execution on a physical quantum processor.
    """

    reps: int = 2
    shots: int | None = None
    seed: int = 42
    _kernel: Any | None = field(init=False, default=None, repr=False)
    _classifier: SVC | None = field(init=False, default=None, repr=False)
    _feature_map: Any | None = field(init=False, default=None, repr=False)
    _x_train: FloatArray | None = field(init=False, default=None, repr=False)

    def fit(self, x_train: FloatArray, y_train: IntArray) -> QuantumKernelClassifier:
        """Fit the precomputed-kernel support-vector classifier."""
        if not quantum_dependencies_available():
            raise QuantumDependencyError(
                "Quantum dependencies are not installed. Run: pip install -e '.[quantum]'"
            )

        from qiskit.circuit.library import zz_feature_map
        from qiskit_machine_learning.kernels import FidelityStatevectorKernel
        from qiskit_machine_learning.utils import algorithm_globals

        algorithm_globals.random_seed = self.seed
        num_features = int(x_train.shape[1])
        feature_map = zz_feature_map(
            feature_dimension=num_features,
            reps=self.reps,
            entanglement="linear",
        )
        kernel = FidelityStatevectorKernel(
            feature_map=feature_map,
            shots=self.shots,
            auto_clear_cache=False,
            enforce_psd=True,
        )
        train_kernel = kernel.evaluate(x_train)
        classifier = SVC(
            kernel="precomputed",
            class_weight="balanced",
            probability=True,
            random_state=self.seed,
        )
        classifier.fit(train_kernel, y_train)

        self._kernel = kernel
        self._classifier = classifier
        self._feature_map = feature_map
        self._x_train = np.asarray(x_train, dtype=float)
        return self

    def _test_kernel(self, x_test: FloatArray) -> FloatArray:
        if self._kernel is None or self._x_train is None:
            raise RuntimeError("model has not been fitted")
        return np.asarray(self._kernel.evaluate(x_test, self._x_train), dtype=float)

    def predict(self, x_test: FloatArray) -> IntArray:
        """Predict binary labels."""
        if self._classifier is None:
            raise RuntimeError("model has not been fitted")
        return np.asarray(self._classifier.predict(self._test_kernel(x_test)), dtype=int)

    def predict_proba(self, x_test: FloatArray) -> FloatArray:
        """Return calibrated class probabilities from SVC Platt scaling."""
        if self._classifier is None:
            raise RuntimeError("model has not been fitted")
        return np.asarray(self._classifier.predict_proba(self._test_kernel(x_test)), dtype=float)

    def predict_with_scores(self, x_test: FloatArray) -> tuple[IntArray, FloatArray]:
        """Calculate one test kernel and return predictions plus positive-class scores."""
        if self._classifier is None:
            raise RuntimeError("model has not been fitted")
        test_kernel = self._test_kernel(x_test)
        predictions = np.asarray(self._classifier.predict(test_kernel), dtype=int)
        probabilities = np.asarray(self._classifier.predict_proba(test_kernel), dtype=float)
        return predictions, probabilities[:, 1]

    def resource_summary(self) -> dict[str, Any]:
        """Return transparent circuit and execution metadata."""
        if self._feature_map is None:
            raise RuntimeError("model has not been fitted")
        operations = {str(name): int(count) for name, count in self._feature_map.count_ops().items()}
        two_qubit_operations = sum(
            count
            for name, count in operations.items()
            if name.lower() in {"cx", "cz", "ecr", "rzz", "swap"}
        )
        return {
            "execution_mode": "statevector_simulation",
            "physical_qpu": False,
            "qubits": int(self._feature_map.num_qubits),
            "parameters": int(self._feature_map.num_parameters),
            "feature_map_repetitions": int(self.reps),
            "shots": self.shots,
            "circuit_depth": int(self._feature_map.depth()),
            "circuit_size": int(self._feature_map.size()),
            "operations": operations,
            "two_qubit_operations": int(two_qubit_operations),
        }
