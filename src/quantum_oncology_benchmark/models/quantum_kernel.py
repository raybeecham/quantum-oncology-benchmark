"""Quantum-kernel model implemented with Qiskit Machine Learning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
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
    not represent execution on a physical quantum processor. Probability scores
    are fitted through training-only, out-of-fold sigmoid calibration.
    """

    reps: int = 2
    shots: int | None = None
    seed: int = 42
    _kernel: Any | None = field(init=False, default=None, repr=False)
    _classifier: SVC | None = field(init=False, default=None, repr=False)
    _calibrator: LogisticRegression | None = field(init=False, default=None, repr=False)
    _calibration_folds: int | None = field(init=False, default=None, repr=False)
    _feature_map: Any | None = field(init=False, default=None, repr=False)
    _x_train: FloatArray | None = field(init=False, default=None, repr=False)

    def _new_classifier(self) -> SVC:
        return SVC(
            kernel="precomputed",
            class_weight="balanced",
            random_state=self.seed,
        )

    def _fit_training_only_calibration(
        self,
        train_kernel: FloatArray,
        y_train: IntArray,
    ) -> tuple[SVC, LogisticRegression, int]:
        """Fit out-of-fold sigmoid calibration without using the test partition."""
        class_counts = np.bincount(y_train, minlength=2)
        minimum_class_count = int(class_counts.min())
        folds = min(5, minimum_class_count)
        if folds < 2:
            raise ValueError("quantum calibration requires at least two samples per class")

        out_of_fold_scores = np.empty(len(y_train), dtype=float)
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.seed)
        for fit_indices, calibration_indices in splitter.split(train_kernel, y_train):
            fold_classifier = self._new_classifier()
            fold_classifier.fit(
                train_kernel[np.ix_(fit_indices, fit_indices)],
                y_train[fit_indices],
            )
            calibration_kernel = train_kernel[np.ix_(calibration_indices, fit_indices)]
            out_of_fold_scores[calibration_indices] = np.asarray(
                fold_classifier.decision_function(calibration_kernel),
                dtype=float,
            )

        calibrator = LogisticRegression(solver="lbfgs", random_state=self.seed)
        calibrator.fit(out_of_fold_scores.reshape(-1, 1), y_train)

        classifier = self._new_classifier()
        classifier.fit(train_kernel, y_train)
        return classifier, calibrator, folds

    def fit(self, x_train: FloatArray, y_train: IntArray) -> QuantumKernelClassifier:
        """Fit the precomputed-kernel classifier and training-only calibrator."""
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
        train_kernel = np.asarray(kernel.evaluate(x_train), dtype=float)
        classifier, calibrator, calibration_folds = self._fit_training_only_calibration(
            train_kernel,
            y_train,
        )

        self._kernel = kernel
        self._classifier = classifier
        self._calibrator = calibrator
        self._calibration_folds = calibration_folds
        self._feature_map = feature_map
        self._x_train = np.asarray(x_train, dtype=float)
        return self

    def _test_kernel(self, x_test: FloatArray) -> FloatArray:
        if self._kernel is None or self._x_train is None:
            raise RuntimeError("model has not been fitted")
        return np.asarray(self._kernel.evaluate(x_test, self._x_train), dtype=float)

    def _calibrated_probabilities(self, decision_scores: FloatArray) -> FloatArray:
        if self._calibrator is None:
            raise RuntimeError("model has not been fitted")
        probabilities = np.asarray(
            self._calibrator.predict_proba(decision_scores.reshape(-1, 1)),
            dtype=float,
        )
        positive_indices = np.flatnonzero(self._calibrator.classes_ == 1)
        if len(positive_indices) != 1:
            raise RuntimeError("calibrator does not contain positive class 1")
        positive = probabilities[:, int(positive_indices[0])]
        return np.column_stack([1.0 - positive, positive])

    def predict(self, x_test: FloatArray) -> IntArray:
        """Predict binary labels."""
        if self._classifier is None:
            raise RuntimeError("model has not been fitted")
        return np.asarray(self._classifier.predict(self._test_kernel(x_test)), dtype=int)

    def predict_proba(self, x_test: FloatArray) -> FloatArray:
        """Return training-only calibrated class probabilities."""
        if self._classifier is None:
            raise RuntimeError("model has not been fitted")
        decision_scores = np.asarray(
            self._classifier.decision_function(self._test_kernel(x_test)),
            dtype=float,
        )
        return self._calibrated_probabilities(decision_scores)

    def predict_with_scores(self, x_test: FloatArray) -> tuple[IntArray, FloatArray]:
        """Calculate one test kernel and return predictions plus positive-class scores."""
        if self._classifier is None:
            raise RuntimeError("model has not been fitted")
        test_kernel = self._test_kernel(x_test)
        predictions = np.asarray(self._classifier.predict(test_kernel), dtype=int)
        decision_scores = np.asarray(self._classifier.decision_function(test_kernel), dtype=float)
        probabilities = self._calibrated_probabilities(decision_scores)
        return predictions, probabilities[:, 1]

    def resource_summary(self) -> dict[str, Any]:
        """Return transparent circuit and execution metadata."""
        if self._feature_map is None or self._calibration_folds is None:
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
            "probability_calibration": "training_only_stratified_oof_sigmoid",
            "calibration_folds": self._calibration_folds,
        }
