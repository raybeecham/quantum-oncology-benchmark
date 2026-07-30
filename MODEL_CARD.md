# Model Card

## System name

Quantum Oncology Benchmark, version 0.1.0.

## Intended use

- Reproducible research and education.
- Comparing classical and quantum-kernel classifiers under shared controls.
- Demonstrating evidence, provenance, and resource-reporting practices.
- Generating hypotheses for further investigation.

## Out-of-scope use

- Diagnosis, screening, triage, prognosis, or treatment selection.
- Patient-specific recommendations.
- Autonomous clinical decisions.
- Use with unidentified regulatory requirements.
- Claims of quantum, biomedical, or clinical advantage based on built-in examples.

## Built-in task

Binary classification using the Wisconsin Diagnostic Breast Cancer dataset packaged by scikit-learn.

The original dataset encodes malignant as `0` and benign as `1`. This project remaps malignant to positive class `1` so that sensitivity measures malignant-case detection.

## Model families

### Classical

- Regularized logistic regression.
- RBF support-vector classifier.
- Random forest.
- Histogram gradient boosting.

### Quantum

- Qiskit fidelity statevector kernel.
- Precomputed-kernel support-vector classifier.

The quantum model is simulated unless a future hardware adapter explicitly records otherwise.

## Inputs

Numeric tabular features. Version 0.1.0 does not directly process medical images, sequences, free-text notes, or molecular structures.

## Outputs

- Binary predictions.
- Positive-class scores.
- Performance metrics.
- Dataset and environment provenance.
- Quantum circuit/resource metadata.

## Known limitations

- Small demonstration dataset.
- No nested hyperparameter optimization.
- No grouped subject splitting.
- No fairness or subgroup module.
- No external clinical validation.
- No physical-QPU adapter.
- SVC probability calibration uses internal Platt scaling and should not be treated as clinical calibration.
- Univariate feature selection may miss interacting biomarkers.
- Runtime comparisons between model families are implementation-specific.

## Ethical and safety considerations

Cancer labels and health data can produce serious harms if models are misused. Do not infer clinical utility from benchmark metrics. Do not publish identifiable or sensitive patient data. Evaluate population shift, subgroup performance, consent, lawful use, and clinical context before translational work.

## Interpretation rule

A quantum model that scores higher than a classical model on one run has generated a result, not established an advantage.
