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

### Changed

- Reproducibility normalization now canonicalizes finite floats to 12 decimal places for equality checks while preserving the original stored artifacts.
- Experiment artifact schema advanced to `1.3` for the expanded pairwise statistical contract.

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
