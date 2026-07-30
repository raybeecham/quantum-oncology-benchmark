# Architecture

## Design goals

- Reproducible by default.
- Strong classical controls.
- No hidden data preprocessing.
- Quantum execution mode made explicit.
- Machine-readable evidence artifacts.
- No requirement for protected data.

## Components

### Configuration

`ExperimentConfig` is an immutable dataclass loaded from YAML or CLI arguments. Validation occurs before data loading or model execution.

### Data layer

`DatasetBundle` contains features, target, source, positive-class definition, metadata, and a stable fingerprint.

Current adapters:

- Built-in Wisconsin Diagnostic Breast Cancer dataset.
- User-supplied numeric binary CSV.

### Preprocessing

The system creates a stratified train/test split before fitting:

1. Median imputation.
2. Standard scaling.
3. Training-only univariate feature selection.
4. Quantum-only scaling to `[0, π]`.

Classical and quantum models use the same selected features and partition indices.

### Model layer

Classical estimators use the scikit-learn interface.

The quantum estimator:

1. Creates a Qiskit `zz_feature_map` with one qubit per selected feature.
2. Calculates a fidelity kernel with `FidelityStatevectorKernel`.
3. Trains an SVC using the precomputed training kernel.
4. Evaluates test samples against the training samples.
5. Records circuit resources and that execution was simulated.

### Evaluation

All models produce positive-class probabilities and binary predictions. Metrics are calculated with class `1` as the clinically positive class.

### Reporting

Artifacts include:

- Complete experiment JSON.
- Aggregate CSV.
- Per-run CSV.
- Markdown report.
- Optional Streamlit visualization.

## Trust boundaries

- CSV files are untrusted input.
- YAML configuration is untrusted input.
- No dynamic code execution is used for configuration.
- The dashboard reads local generated artifacts and should not be exposed publicly without authentication and deployment review.
- Future hardware credentials must be loaded from environment or approved secret storage, never configuration files.
