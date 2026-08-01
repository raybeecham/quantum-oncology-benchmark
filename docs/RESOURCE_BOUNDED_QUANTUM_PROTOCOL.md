# Resource-Bounded Quantum Protocol v1

## Purpose

`quantum-protocol-v1` defines the minimum evidence, resource, and backend contract required before a quantum-kernel model can be compared with the frozen classical protocol.

The protocol begins with local statevector simulation. It does not connect to a physical quantum computer, accept provider credentials, or claim quantum advantage.

## Scientific comparison rule

The historical 569-sample classical result is a protocol reference, not a score that can be compared directly with a smaller quantum experiment.

Every quantum experiment must rerun the frozen classical comparator on the exact same:

- dataset subset,
- outer partitions,
- training and test samples,
- feature count,
- selected features,
- preprocessing scope,
- positive-class definition,
- primary endpoint.

The primary frozen comparator is logistic regression. RBF SVM remains the primary nonlinear classical comparator. A quantum model evaluated on a reduced cohort must be compared only with classical controls rerun on that reduced cohort.

## Initial implementation slice

The first enabled backend is:

```text
qiskit-fidelity-statevector
```

It supports:

- exact statevector fidelity,
- optional finite-shot statevector sampling,
- ZZ feature maps,
- deterministic seeds,
- train and test kernel matrices,
- pre-execution resource-budget enforcement,
- circuit and execution resource records.

It does not support:

- physical-QPU execution,
- provider sessions,
- backend calibration snapshots,
- transpiled physical-qubit records,
- queue or job identifiers,
- monetary cost records.

Those fields exist in the backend-neutral record and remain explicitly empty for local simulation.

## Staged evaluation plan

### Q0: contract smoke

Purpose: verify the execution interface, kernel matrices, resource accounting, and artifact schema.

Recommended limits:

- 2 to 4 features,
- no more than 120 training samples,
- no more than 40 test samples,
- one or two feature-map repetitions,
- exact statevector simulation,
- no physical hardware.

Q0 results are engineering evidence only.

### Q1: matched simulator benchmark

Purpose: compare a bounded quantum-kernel SVM with rerun classical controls on identical partitions.

Requirements:

- predeclared cohort cap,
- locked balanced-accuracy endpoint,
- frozen logistic-regression comparator,
- identical outer partitions,
- prediction-level evidence,
- exact paired comparisons,
- resource records for every quantum fold,
- no quantum-advantage claim.

### Q2: finite-shot simulator sensitivity

Purpose: quantify sensitivity to sampling noise before hardware use.

Requirements:

- identical Q1 partitions and features,
- predeclared shot counts,
- deterministic simulator seeds,
- total-shot budget,
- exact-versus-shot paired prediction comparison,
- repeated shot-noise runs where justified.

### Q3: physical-QPU pilot

Purpose: test execution feasibility and measure hardware-induced change under a small, predeclared budget.

A Q3 run must not begin until the provider adapter supplies all required hardware metadata.

## Backend-neutral interface

A backend implements `QuantumKernelBackend` and must expose:

```python
capabilities() -> BackendCapabilities

evaluate(
    specification,
    budget,
    x_train,
    x_test,
) -> QuantumKernelResult
```

The caller does not receive provider-specific objects. It receives:

- a training kernel matrix,
- a test kernel matrix,
- an immutable `QuantumResourceRecord`.

This keeps model evaluation separate from provider authentication, job submission, and hardware-specific APIs.

## Resource budget

The budget is enforced before backend execution.

Current ceilings include:

- maximum features and logical qubits,
- maximum training samples,
- maximum test samples,
- maximum feature-map repetitions,
- maximum kernel-matrix entries,
- maximum shots per kernel entry,
- maximum estimated total shots,
- maximum circuits per job,
- explicit physical-hardware permission.

A request exceeding any ceiling raises `ResourceBudgetExceeded`. The backend must not partially execute an over-budget request.

## Kernel-entry accounting

For `n_train` training samples and `n_test` test samples, the conservative logical kernel-entry count is:

```text
n_train × n_train + n_test × n_train
```

This count is used for budget enforcement across simulator and future hardware adapters. A provider may optimize duplicate or symmetric calculations internally, but the declared comparison budget remains based on the full logical matrices.

## Resource record

Every completed request records:

- protocol and profile version,
- backend and provider identity,
- backend package version,
- simulator or physical-QPU status,
- feature map, repetitions, and entanglement,
- logical and physical qubits,
- circuit depth and size,
- two-qubit gate count,
- sample and feature counts,
- kernel-entry count,
- shots and estimated total shots,
- seed,
- transpilation status,
- backend calibration snapshot,
- job identifiers,
- queue and execution time,
- failed jobs and retries,
- cost and cost unit,
- completion status,
- SHA-256 resource-record fingerprint.

Simulator adapters must not fabricate hardware metadata. Unsupported fields remain `null`, empty, or false as appropriate.

## Kernel validation

Before a kernel reaches a classifier, the framework verifies:

- finite two-dimensional values,
- fidelity values within numerical tolerance of `[0, 1]`,
- square and symmetric training kernels,
- unit training diagonal,
- test-kernel width matching the training sample count,
- valid resource-record fingerprint.

## Future provider adapters

A physical-QPU adapter must additionally capture:

- provider and backend name,
- backend calibration timestamp,
- basis gates and coupling map,
- logical-to-physical qubit layout,
- transpilation optimization level,
- transpiled depth,
- two-qubit gate counts after transpilation,
- shots and mitigation configuration,
- session and job identifiers,
- queue, execution, and wall-clock time,
- failed jobs and retries,
- monetary or credit cost.

Provider credentials must remain outside configuration files, artifacts, logs, issues, and pull requests.

## Claim boundary

A quantum score higher than a classical score on one simulator, one QPU, or one small dataset is not quantum advantage.

Any future positive signal must remain exploratory until it is reproduced under matched budgets, repeated across execution conditions, and validated on independent data.
