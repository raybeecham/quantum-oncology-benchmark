# Research Protocol

## 1. Purpose

This protocol defines the minimum controls for comparing classical and quantum machine-learning methods in the Quantum Oncology Benchmark.

The project is exploratory and research-only. It is not intended to diagnose disease, select treatment, estimate an individual patient's prognosis, or support clinical operations.

## 2. Primary research question

Under matched data, preprocessing, and evaluation conditions, does a quantum-kernel classifier exhibit a reproducible difference from strong classical baselines on a defined binary cancer-classification task?

A difference may involve predictive performance, sample efficiency, computational cost, robustness, calibration, or scaling behavior. A difference is not automatically an advantage.

## 3. Null and alternative hypotheses

**Null hypothesis:** The quantum model does not provide a practically meaningful and reproducible improvement over the best classical comparator under the defined protocol.

**Alternative hypothesis:** The quantum model provides a practically meaningful and reproducible improvement on one or more predeclared outcomes without unacceptable degradation in other required outcomes.

The null hypothesis is the default interpretation unless evidence is sufficient to reject it.

## 4. Required comparators

At minimum, each quantum experiment must include:

- Regularized logistic regression.
- A nonlinear classical kernel method.
- A tree ensemble.
- A boosting method.

Hyperparameter budgets should be comparable and documented. An untuned classical baseline must not be used to support a quantum-superiority claim.

## 5. Data controls

- The unit of analysis must be declared.
- Multiple samples from one subject must not cross train/test boundaries.
- Train/test splitting must occur before imputation, normalization, feature selection, dimensionality reduction, or resampling.
- The positive class must be explicit.
- Class prevalence must be reported.
- Missingness and exclusion criteria must be documented.
- Public, controlled-access, simulated, and synthetic data must be clearly distinguished.

Version 0.1.0 supports independent rows but does not yet implement subject-grouped splitting. A cohort with repeated subject identifiers must be processed outside the current runner or deferred until grouped splitting is implemented.

## 6. Evaluation design

### Demonstration tier

Used for software verification and examples:

- Public educational dataset.
- One or more stratified holdouts.
- No biomedical or quantum-advantage claim.

### Research tier

Required before publishing comparative conclusions:

- Predeclared primary and secondary endpoints.
- Repeated nested cross-validation or a locked external test set.
- Hyperparameter selection inside the training process.
- Confidence intervals.
- Sensitivity analyses across seeds and feature counts.
- Calibration analysis.
- Subgroup analysis only when statistically and ethically justified.

### Validation tier

Required before translational or clinical discussion:

- Independent external cohort.
- Locked model and analysis plan.
- Site, platform, and demographic-shift analysis.
- Domain-expert review.
- Regulatory, privacy, and ethics review where applicable.

## 7. Metrics

Primary metrics should be declared before running the experiment. Recommended outcomes include:

- Balanced accuracy.
- Sensitivity for malignant or otherwise clinically positive cases.
- Specificity.
- ROC AUC.
- Average precision under class imbalance.
- Calibration error or proper scoring rules.
- False-negative count.

Runtime and quantum resources are not substitutes for predictive or clinical usefulness.

## 8. Quantum resource reporting

Every quantum result must report:

- Simulator or physical hardware.
- Backend and backend snapshot when applicable.
- Logical and physical qubit counts when applicable.
- Feature-map construction.
- Circuit depth and operation counts.
- Two-qubit operation counts.
- Shot count.
- Error-mitigation method.
- Queue, execution, and total wall-clock time when available.
- Failed jobs and reruns.

Statevector simulation must never be presented as physical-QPU execution.

## 9. Advantage language

The following statements require increasing levels of evidence:

1. **Quantum model evaluated:** A quantum algorithm was run.
2. **Quantum model competitive:** Performance falls within a predefined practical margin of a classical comparator.
3. **Quantum model superior on this benchmark:** Predeclared statistical and practical criteria are satisfied on a locked benchmark.
4. **Quantum advantage:** A well-defined computational task is solved better than the best relevant classical approach under defensible resource accounting and independent verification.
5. **Clinical advantage:** Patient- or workflow-relevant benefit is demonstrated through appropriate clinical validation.

Version 0.1.0 supports only the first statement.

## 10. Reproducibility package

A reportable experiment must retain:

- Configuration file.
- Code commit.
- Dataset fingerprint or immutable manifest.
- Environment and dependency versions.
- Random seeds.
- Per-run predictions or sufficient derived artifacts, subject to data policy.
- Aggregate metrics.
- Circuit/resource metadata.
- Known deviations from protocol.

## 11. Stop conditions

An experiment should stop or be downgraded to exploratory status when:

- Data leakage is discovered.
- Class labels are ambiguous.
- The test set influences feature or model selection.
- A classical comparator fails basic quality checks.
- Results cannot be reproduced.
- Protected data handling is uncertain.
- The quantum execution mode cannot be independently verified.

## 12. Statistical evaluation layer

The benchmark reports repeat-level BCa bootstrap confidence intervals for model metric means and exact McNemar tests for paired predictions within each shared test partition.

- The repeat-level metric is the bootstrap resampling unit.
- At least two repeat values are required to estimate an interval.
- Paired tests use only predictions generated on the same test partition.
- Repeated-holdout p-values are not pooled because observations can recur across test partitions.
- Statistical outputs are exploratory benchmark evidence and do not establish external validity, clinical utility, or quantum advantage.

See `docs/STATISTICAL_EVALUATION.md` for implementation details and deferred methods.

## 13. Classical nested cross-validation layer

The `qob nested-cv` mode provides a stronger classical model-selection baseline while preserving the repeated-holdout mode as a separate evaluation design.

- Balanced accuracy is the locked primary endpoint.
- The reference protocol uses five stratified outer folds and three stratified inner folds.
- Median imputation, standardization, feature selection, calibration, and estimator fitting remain inside the model pipeline.
- Bounded hyperparameter selection uses only the outer training partition.
- The selected pipeline is refit on the complete outer training partition.
- The outer test fold is evaluated once and never influences preprocessing or model selection.
- Each sample receives one outer-fold prediction per evaluated model.
- Exact McNemar tests are calculated only within one shared outer test fold.
- Outer-fold p-values are not pooled.
- Fold-level bootstrap intervals remain descriptive and do not represent external validation.

The first implementation is classical-only. Quantum nested cross-validation requires a separate, resource-bounded protocol.

See `docs/NESTED_CROSS_VALIDATION.md` for search spaces, artifacts, reproducibility controls, and limitations.
