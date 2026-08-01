# Changelog

All notable changes to this project will be documented here.

## Unreleased

### Added

- Repeat-level BCa bootstrap confidence intervals for model metrics.
- Exact paired McNemar comparisons within shared test partitions.
- Pairwise comparison CSV output and bounded evidence statements.
- Statistical evaluation methods documentation and regression tests.
- Self-contained pairwise test-partition hashes, sample counts, and descriptive direction fields.
- Separate descriptive direction counts and statistically favored counts in pairwise summaries.
- A separate `qob nested-cv` mode for classical nested cross-validation.
- Locked five-outer-fold, three-inner-fold reference configuration with balanced accuracy as the primary endpoint.
- Fold-level predictions, inner-search results, selected parameters, selected features, and partition provenance.
- A seven-file nested cross-validation artifact package and methodology report.
- Nested cross-validation reproducibility normalization and regression tests.
- Versioned `reference-v1` and `sensitivity-v1` classical nested search profiles.
- Out-of-fold calibration summary, reliability-bin, probability-distribution, and hashed classification-error artifacts.
- Calibration-in-the-large, uniform-bin ECE and MCE, pooled Brier score and log loss, and false-positive/false-negative diagnostics.
- Calibration diagnostics methodology documentation and regression tests.
- A standalone `qob compare-profiles` command for paired comparison of completed nested-CV profiles.
- Cross-profile prediction, parameter-boundary, protocol-freeze JSON, CSV, and Markdown artifacts.
- Compatibility checks for dataset, endpoint, folds, models, sample hashes, truth labels, and outer-fold assignments.
- A documented resource and provenance boundary for future simulator and physical-QPU profile comparisons.
- Formal `quantum-protocol-v1` specification with a locked balanced-accuracy endpoint and logistic-regression comparator.
- Backend-neutral `QuantumKernelBackend` execution interface and immutable kernel-result contract.
- Hard pre-execution budgets for features, samples, feature-map repetitions, kernel entries, shots, total shots, and circuits per job.
- Local Qiskit fidelity-statevector adapter supporting exact and finite-shot simulation modes.
- Simulator resource records for logical qubits, circuit depth and size, two-qubit gates, execution timing, and SHA-256 fingerprints.
- Quantum-kernel validation for fidelity bounds, symmetry, unit diagonal, finite values, and matrix shapes.
- Staged Q0 through Q3 resource-bounded quantum evaluation protocol documentation and regression tests.

### Changed

- Reproducibility normalization now canonicalizes finite floats to 12 decimal places for equality checks while preserving the original stored artifacts.
- Experiment artifact schema advanced to `1.3` for the expanded pairwise statistical contract.
- Nested cross-validation artifact schema advanced to `nested-cv-1.1` for search-profile identity and calibration diagnostics.

## 0.1.0, 2026-07-30

### Added

- Reproducible classical and quantum-kernel benchmark runner.
- Built-in Wisconsin Diagnostic Breast Cancer demonstration dataset.
- Binary numeric CSV adapter.
- Training-only imputation, scaling, and feature selection.
- Four classical baselines and one optional Qiskit fidelity-kernel SVM.
- Repeated holdouts and oncology-oriented performance metrics.
- Dataset fingerprint, environment provenance, and circuit resource records.
- JSON, CSV, Markdown, and Streamlit reporting.
- Public GDC metadata manifest generator with query receipts.
- Research protocol, model card, data governance, security guidance, CI, and tests.
