"""Backend-neutral resource contract for quantum-kernel execution."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
import yaml
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
ExecutionMode = Literal["exact_statevector", "shot_statevector", "physical_qpu"]


class QuantumProtocolError(RuntimeError):
    """Base error for the resource-bounded quantum protocol."""


class ResourceBudgetExceeded(QuantumProtocolError):
    """Raised when a quantum execution request exceeds its declared budget."""


@dataclass(frozen=True, slots=True)
class QuantumResourceBudget:
    """Hard resource ceilings applied before a backend may execute."""

    max_features: int = 4
    max_train_samples: int = 120
    max_test_samples: int = 40
    max_feature_map_repetitions: int = 2
    max_kernel_entries: int = 20_000
    max_shots_per_kernel_entry: int = 4_096
    max_total_shots: int = 20_000_000
    max_circuits_per_job: int = 300
    hardware_execution_allowed: bool = False

    def validate(self) -> None:
        """Validate resource ceilings."""
        integer_fields = {
            "max_features": self.max_features,
            "max_train_samples": self.max_train_samples,
            "max_test_samples": self.max_test_samples,
            "max_feature_map_repetitions": self.max_feature_map_repetitions,
            "max_kernel_entries": self.max_kernel_entries,
            "max_shots_per_kernel_entry": self.max_shots_per_kernel_entry,
            "max_total_shots": self.max_total_shots,
            "max_circuits_per_job": self.max_circuits_per_job,
        }
        for name, value in integer_fields.items():
            if value < 1:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class QuantumKernelSpecification:
    """Versioned quantum-kernel model and execution specification."""

    protocol_version: str = "quantum-protocol-v1"
    profile_name: str = "statevector-smoke-v1"
    backend_id: str = "qiskit-fidelity-statevector"
    feature_map: str = "zz"
    feature_map_repetitions: int = 2
    entanglement: str = "linear"
    shots: int | None = None
    seed: int = 42
    enforce_psd: bool = True
    primary_metric: str = "balanced_accuracy"
    classical_comparator: str = "logistic_regression"

    def validate(self) -> None:
        """Validate the locked model and evidence fields."""
        if self.protocol_version != "quantum-protocol-v1":
            raise ValueError("protocol_version must be 'quantum-protocol-v1'")
        if not self.profile_name:
            raise ValueError("profile_name is required")
        if self.backend_id != "qiskit-fidelity-statevector":
            raise ValueError("only the qiskit fidelity statevector backend is enabled in this slice")
        if self.feature_map != "zz":
            raise ValueError("feature_map is locked to 'zz' in quantum-protocol-v1")
        if self.feature_map_repetitions < 1:
            raise ValueError("feature_map_repetitions must be positive")
        if self.entanglement not in {"linear", "full"}:
            raise ValueError("entanglement must be 'linear' or 'full'")
        if self.shots is not None and self.shots < 1:
            raise ValueError("shots must be positive when provided")
        if self.primary_metric != "balanced_accuracy":
            raise ValueError("primary_metric is locked to 'balanced_accuracy'")
        if self.classical_comparator != "logistic_regression":
            raise ValueError("classical_comparator is locked to 'logistic_regression'")

    @property
    def execution_mode(self) -> ExecutionMode:
        """Return the simulator mode implied by the shot configuration."""
        return "exact_statevector" if self.shots is None else "shot_statevector"


@dataclass(frozen=True, slots=True)
class QuantumProtocolConfig:
    """Serializable quantum protocol profile."""

    specification: QuantumKernelSpecification = field(
        default_factory=QuantumKernelSpecification
    )
    budget: QuantumResourceBudget = field(default_factory=QuantumResourceBudget)

    def validate(self) -> None:
        """Validate the protocol profile and budget together."""
        self.specification.validate()
        self.budget.validate()
        if self.specification.feature_map_repetitions > self.budget.max_feature_map_repetitions:
            raise ValueError("feature-map repetitions exceed the configured resource budget")
        if (
            self.specification.shots is not None
            and self.specification.shots > self.budget.max_shots_per_kernel_entry
        ):
            raise ValueError("shots exceed the per-entry resource budget")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "specification": asdict(self.specification),
            "budget": asdict(self.budget),
        }

    @classmethod
    def from_yaml(cls, path: str | Path) -> QuantumProtocolConfig:
        """Load a protocol profile from YAML."""
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise ValueError("quantum protocol configuration root must be a mapping")
        raw_specification = payload.get("specification", {})
        raw_budget = payload.get("budget", {})
        if not isinstance(raw_specification, dict) or not isinstance(raw_budget, dict):
            raise ValueError("specification and budget must be mappings")
        config = cls(
            specification=QuantumKernelSpecification(**raw_specification),
            budget=QuantumResourceBudget(**raw_budget),
        )
        config.validate()
        return config


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    """Backend capabilities exposed without credentials or execution."""

    backend_id: str
    provider: str
    execution_modes: tuple[ExecutionMode, ...]
    physical_qpu: bool
    supports_exact: bool
    supports_finite_shots: bool
    supports_transpilation_records: bool
    supports_calibration_snapshots: bool
    supports_cost_records: bool


@dataclass(frozen=True, slots=True)
class QuantumResourceRecord:
    """Auditable resource and execution metadata for one kernel request."""

    protocol_version: str
    profile_name: str
    backend_id: str
    provider: str
    backend_version: str | None
    execution_mode: ExecutionMode
    physical_qpu: bool
    classical_simulation: bool
    feature_map: str
    feature_map_repetitions: int
    entanglement: str
    logical_qubits: int
    physical_qubits: tuple[int, ...] | None
    circuit_depth: int
    circuit_size: int
    two_qubit_gate_count: int
    train_samples: int
    test_samples: int
    feature_count: int
    kernel_entries: int
    shots_per_kernel_entry: int | None
    estimated_total_shots: int
    max_circuits_per_job: int
    seed: int
    transpiled: bool
    transpiled_depth: int | None
    backend_calibration_snapshot: dict[str, Any] | None
    job_ids: tuple[str, ...]
    queue_seconds: float
    execution_seconds: float
    failed_jobs: int
    retries: int
    cost: float | None
    cost_unit: str | None
    status: str
    record_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable resource record."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class QuantumKernelResult:
    """Backend-neutral quantum-kernel matrices and resource metadata."""

    train_kernel: FloatArray
    test_kernel: FloatArray
    resources: QuantumResourceRecord


@runtime_checkable
class QuantumKernelBackend(Protocol):
    """Backend-neutral interface required by quantum-protocol-v1."""

    backend_id: str

    def capabilities(self) -> BackendCapabilities:
        """Return backend capabilities without executing a workload."""
        ...

    def evaluate(
        self,
        specification: QuantumKernelSpecification,
        budget: QuantumResourceBudget,
        x_train: FloatArray,
        x_test: FloatArray,
    ) -> QuantumKernelResult:
        """Evaluate train and test quantum-kernel matrices within a hard budget."""
        ...


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _kernel_entries(train_samples: int, test_samples: int) -> int:
    return train_samples * train_samples + test_samples * train_samples


def validate_resource_request(
    specification: QuantumKernelSpecification,
    budget: QuantumResourceBudget,
    x_train: FloatArray,
    x_test: FloatArray,
) -> dict[str, int]:
    """Reject a request before any backend work occurs when it exceeds its budget."""
    specification.validate()
    budget.validate()
    train = np.asarray(x_train, dtype=float)
    test = np.asarray(x_test, dtype=float)
    if train.ndim != 2 or test.ndim != 2:
        raise ValueError("quantum kernel inputs must be two-dimensional")
    if train.shape[1] != test.shape[1]:
        raise ValueError("train and test feature counts must match")
    if train.shape[0] < 2 or test.shape[0] < 1:
        raise ValueError("quantum kernel inputs require at least two train and one test sample")
    if not np.isfinite(train).all() or not np.isfinite(test).all():
        raise ValueError("quantum kernel inputs must contain only finite values")

    feature_count = int(train.shape[1])
    train_samples = int(train.shape[0])
    test_samples = int(test.shape[0])
    entries = _kernel_entries(train_samples, test_samples)
    shots = specification.shots or 0
    total_shots = entries * shots

    violations: list[str] = []
    if feature_count > budget.max_features:
        violations.append(f"features {feature_count} > {budget.max_features}")
    if train_samples > budget.max_train_samples:
        violations.append(f"train samples {train_samples} > {budget.max_train_samples}")
    if test_samples > budget.max_test_samples:
        violations.append(f"test samples {test_samples} > {budget.max_test_samples}")
    if specification.feature_map_repetitions > budget.max_feature_map_repetitions:
        violations.append(
            "feature-map repetitions "
            f"{specification.feature_map_repetitions} > {budget.max_feature_map_repetitions}"
        )
    if entries > budget.max_kernel_entries:
        violations.append(f"kernel entries {entries} > {budget.max_kernel_entries}")
    if specification.shots is not None:
        if specification.shots > budget.max_shots_per_kernel_entry:
            violations.append(
                f"shots per entry {specification.shots} > {budget.max_shots_per_kernel_entry}"
            )
        if total_shots > budget.max_total_shots:
            violations.append(f"total shots {total_shots} > {budget.max_total_shots}")
    if specification.execution_mode == "physical_qpu" and not budget.hardware_execution_allowed:
        violations.append("physical QPU execution is not allowed by this budget")
    if violations:
        raise ResourceBudgetExceeded("; ".join(violations))

    return {
        "feature_count": feature_count,
        "train_samples": train_samples,
        "test_samples": test_samples,
        "kernel_entries": entries,
        "estimated_total_shots": total_shots,
    }


def _resource_fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class QiskitStatevectorKernelBackend:
    """Qiskit statevector adapter implementing the backend-neutral kernel contract."""

    backend_id = "qiskit-fidelity-statevector"

    def capabilities(self) -> BackendCapabilities:
        """Return the intentionally limited simulator capabilities."""
        return BackendCapabilities(
            backend_id=self.backend_id,
            provider="local-qiskit",
            execution_modes=("exact_statevector", "shot_statevector"),
            physical_qpu=False,
            supports_exact=True,
            supports_finite_shots=True,
            supports_transpilation_records=False,
            supports_calibration_snapshots=False,
            supports_cost_records=False,
        )

    def evaluate(
        self,
        specification: QuantumKernelSpecification,
        budget: QuantumResourceBudget,
        x_train: FloatArray,
        x_test: FloatArray,
    ) -> QuantumKernelResult:
        """Evaluate a bounded fidelity statevector kernel with lazy Qiskit imports."""
        if specification.backend_id != self.backend_id:
            raise ValueError(f"backend mismatch: {specification.backend_id}")
        request = validate_resource_request(specification, budget, x_train, x_test)
        train = np.asarray(x_train, dtype=float)
        test = np.asarray(x_test, dtype=float)

        try:
            circuit_library = import_module("qiskit.circuit.library")
            kernels = import_module("qiskit_machine_learning.kernels")
            utilities = import_module("qiskit_machine_learning.utils")
        except ModuleNotFoundError as exc:
            raise QuantumProtocolError(
                "Qiskit dependencies are required; install with pip install -e '.[quantum]'"
            ) from exc

        algorithm_globals = utilities.algorithm_globals
        algorithm_globals.random_seed = specification.seed
        zz_feature_map = circuit_library.zz_feature_map
        kernel_type = kernels.FidelityStatevectorKernel
        feature_map = zz_feature_map(
            feature_dimension=request["feature_count"],
            reps=specification.feature_map_repetitions,
            entanglement=specification.entanglement,
        )
        kernel = kernel_type(
            feature_map=feature_map,
            shots=specification.shots,
            enforce_psd=specification.enforce_psd,
        )

        started = time.perf_counter()
        train_kernel = np.asarray(kernel.evaluate(train), dtype=float)
        test_kernel = np.asarray(kernel.evaluate(test, train), dtype=float)
        execution_seconds = time.perf_counter() - started
        if train_kernel.shape != (request["train_samples"], request["train_samples"]):
            raise QuantumProtocolError("unexpected training kernel shape")
        if test_kernel.shape != (request["test_samples"], request["train_samples"]):
            raise QuantumProtocolError("unexpected test kernel shape")
        if not np.isfinite(train_kernel).all() or not np.isfinite(test_kernel).all():
            raise QuantumProtocolError("backend returned non-finite kernel values")

        decomposed = feature_map.decompose()
        two_qubit_gate_count = sum(
            1 for instruction in decomposed.data if len(instruction.qubits) == 2
        )
        fingerprint_payload = {
            "protocol_version": specification.protocol_version,
            "profile_name": specification.profile_name,
            "backend_id": self.backend_id,
            "execution_mode": specification.execution_mode,
            "feature_map": specification.feature_map,
            "feature_map_repetitions": specification.feature_map_repetitions,
            "entanglement": specification.entanglement,
            "logical_qubits": request["feature_count"],
            "circuit_depth": int(decomposed.depth()),
            "circuit_size": int(decomposed.size()),
            "two_qubit_gate_count": two_qubit_gate_count,
            "train_samples": request["train_samples"],
            "test_samples": request["test_samples"],
            "kernel_entries": request["kernel_entries"],
            "shots": specification.shots,
            "seed": specification.seed,
        }
        resources = QuantumResourceRecord(
            protocol_version=specification.protocol_version,
            profile_name=specification.profile_name,
            backend_id=self.backend_id,
            provider="local-qiskit",
            backend_version=_package_version("qiskit-machine-learning"),
            execution_mode=specification.execution_mode,
            physical_qpu=False,
            classical_simulation=True,
            feature_map=specification.feature_map,
            feature_map_repetitions=specification.feature_map_repetitions,
            entanglement=specification.entanglement,
            logical_qubits=request["feature_count"],
            physical_qubits=None,
            circuit_depth=int(decomposed.depth()),
            circuit_size=int(decomposed.size()),
            two_qubit_gate_count=two_qubit_gate_count,
            train_samples=request["train_samples"],
            test_samples=request["test_samples"],
            feature_count=request["feature_count"],
            kernel_entries=request["kernel_entries"],
            shots_per_kernel_entry=specification.shots,
            estimated_total_shots=request["estimated_total_shots"],
            max_circuits_per_job=budget.max_circuits_per_job,
            seed=specification.seed,
            transpiled=False,
            transpiled_depth=None,
            backend_calibration_snapshot=None,
            job_ids=(),
            queue_seconds=0.0,
            execution_seconds=float(execution_seconds),
            failed_jobs=0,
            retries=0,
            cost=None,
            cost_unit=None,
            status="completed",
            record_fingerprint=_resource_fingerprint(fingerprint_payload),
        )
        return QuantumKernelResult(
            train_kernel=train_kernel,
            test_kernel=test_kernel,
            resources=resources,
        )


def built_in_quantum_backends() -> dict[str, QuantumKernelBackend]:
    """Return built-in adapters without credentials or remote connections."""
    backend = QiskitStatevectorKernelBackend()
    return {backend.backend_id: backend}


def validate_kernel_matrix(matrix: FloatArray, *, symmetric: bool) -> None:
    """Validate backend output before it is consumed by a classifier."""
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise QuantumProtocolError("kernel matrix must be finite and two-dimensional")
    if np.any(array < -1e-12) or np.any(array > 1.0 + 1e-12):
        raise QuantumProtocolError("fidelity kernel values must remain within [0, 1]")
    if symmetric:
        if array.shape[0] != array.shape[1]:
            raise QuantumProtocolError("training kernel must be square")
        if not np.allclose(array, array.T, rtol=0.0, atol=1e-10):
            raise QuantumProtocolError("training kernel must be symmetric")
        diagonal = np.diag(array)
        if not np.allclose(diagonal, np.ones_like(diagonal), rtol=0.0, atol=1e-8):
            raise QuantumProtocolError("training fidelity diagonal must be one")


def validate_result(result: QuantumKernelResult) -> None:
    """Validate a backend-neutral result and its immutable resource record."""
    validate_kernel_matrix(result.train_kernel, symmetric=True)
    validate_kernel_matrix(result.test_kernel, symmetric=False)
    if result.test_kernel.shape[1] != result.train_kernel.shape[0]:
        raise QuantumProtocolError("test kernel width must match training sample count")
    if len(result.resources.record_fingerprint) != 64:
        raise QuantumProtocolError("resource record fingerprint must be SHA-256")
    if result.resources.physical_qpu and result.resources.classical_simulation:
        raise QuantumProtocolError("physical_qpu and classical_simulation cannot both be true")
    if not math.isfinite(result.resources.execution_seconds):
        raise QuantumProtocolError("execution_seconds must be finite")
