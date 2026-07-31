# Classical Nested Cross-Validation Protocol

## Purpose

The `qob nested-cv` mode estimates the performance of a bounded classical model-selection procedure while keeping each outer test fold outside preprocessing, feature selection, hyperparameter selection, calibration, and refitting.

It is a separate evaluation mode. It does not replace the repeated-holdout benchmark or change the meaning of existing artifacts.

## Locked defaults

The reference configuration is `configs/nested-classical.yaml`:

- Search profile: `reference-v1`.
- Primary endpoint: balanced accuracy.
- Primary endpoint status: locked.
- Outer splitter: five-fold stratified cross-validation with shuffling.
- Inner splitter: three-fold stratified cross-validation with shuffling.
- Feature count: eight.
- Models: logistic regression, RBF SVM with separately calibrated probability scores, random forest, and histogram gradient boosting.
- Calibration bins: ten uniform probability-width bins.
- Parallelism: one job for deterministic execution and bounded resource use.

The CLI permits a smaller model subset for development and targeted studies, but the reference configuration includes all four classical comparators.

## Versioned search profiles

### `reference-v1`

This profile preserves the completed classical reference baseline:

- Logistic regression `C`: `0.1`, `1.0`, `10.0`.
- RBF SVM `C`: `0.1`, `1.0`, `10.0`.
- RBF SVM `gamma`: `scale`, `0.01`, `0.1`.
- Random forest `n_estimators`: `200`, `500`.
- Random forest `max_depth`: unrestricted, `5`, `10`.
- Histogram gradient boosting `learning_rate`: `0.05`, `0.1`.
- Histogram gradient boosting `max_leaf_nodes`: `15`, `31`.

### `sensitivity-v1`

This profile tests only the two boundary-selection findings identified during review of the full reference run:

- RBF SVM `C`: `1.0`, `10.0`, `100.0`.
- Histogram gradient boosting `max_leaf_nodes`: `7`, `15`, `31`.

All other grids, models, seeds, folds, feature count, preprocessing steps, calibration method, endpoint, and artifact calculations remain the same as `reference-v1`.

A sensitivity result must not silently replace the reference result. Search-profile identity is recorded in configuration, methodology, outer-fold results, predictions, inner-search results, pairwise comparisons, error rows, and evidence statements.

## Evaluation sequence

For each outer fold:

1. Split the raw dataset into an outer training partition and an untouched outer test partition.
2. Construct a deterministic inner stratified splitter from the experiment seed and outer-fold number.
3. For each model, run bounded `GridSearchCV` only on the raw outer training partition.
4. Keep median imputation, standardization, univariate feature selection, and the estimator inside one scikit-learn pipeline.
5. Select parameters using inner-fold balanced accuracy.
6. Refit the selected pipeline on the complete outer training partition.
7. For the RBF SVM only, clone the selected pipeline and fit sigmoid calibration by cross-validation using only the complete outer training partition.
8. Generate class predictions from the selected classifier pipeline and probability scores from the training-only calibrated SVM or the model's native `predict_proba` implementation.
9. Evaluate once on the outer test partition.
10. Store fold-level metrics, predictions, selected features, parameters, probability-score source, search profile, partition hashes, and inner-search results.
11. After all folds complete, calculate pooled out-of-fold calibration and classification-error diagnostics.

The outer test data is never passed to `fit`, `GridSearchCV`, preprocessing, feature selection, calibration, or parameter selection.

## Model controls

### Logistic regression

Fixed controls include class balancing, the `liblinear` solver, and a 5,000-iteration limit.

### RBF SVM with training-only score calibration

The primary classification endpoint is tuned and evaluated from the selected SVM pipeline without using the deprecated internal probability mode. After inner selection, a clone of the complete selected pipeline, including imputation, scaling, and feature selection, is wrapped by `CalibratedClassifierCV` and fitted only on the outer training partition.

The calibrated clone supplies probability scores for ROC AUC, average precision, Brier score, log loss, and calibration diagnostics. It does not replace the selected classifier's class predictions used for balanced accuracy, sensitivity, specificity, F1, or paired McNemar comparisons.

### Random forest

Fixed controls include balanced subsampling, a minimum leaf size of two, deterministic seeds, and one worker.

### Histogram gradient boosting

Fixed controls include 250 boosting iterations and L2 regularization of `1.0`.

The grids are intentionally limited. They support a fair, reproducible comparison rather than an unlimited optimization contest.

## Artifact package

Every completed `nested-cv-1.1` run writes:

- `nested_experiment.json`: complete structured experiment record.
- `outer_fold_results.csv`: selected configuration, provenance, probability-score source, and final metrics for each model and outer fold.
- `outer_fold_predictions.csv`: search profile, hashed sample identifier, truth, prediction, and positive-class score.
- `inner_search_results.csv`: every bounded candidate score and selected-candidate flag.
- `nested_summary.csv`: aggregate outer-fold metrics, descriptive intervals, and pooled calibration fields.
- `nested_pairwise_comparisons.csv`: exact McNemar comparisons within each shared outer test fold.
- `calibration_summary.csv`: pooled out-of-fold calibration and error summary by model.
- `calibration_bins.csv`: occupied reliability-bin coordinates and supporting counts.
- `probability_distribution.csv`: every configured probability bin by model and true class.
- `classification_errors.csv`: hashed out-of-fold false-positive and false-negative rows.
- `NESTED_CV_REPORT.md`: human-readable method, results, calibration summary, evidence statement, and limitations.

Raw row indices are not published. Partition membership and sample identity are represented with deterministic SHA-256 hashes.

See [Out-of-Fold Calibration Diagnostics](CALIBRATION_DIAGNOSTICS.md) for the diagnostic definitions and interpretation limits.

## Statistical interpretation

- Bootstrap intervals resample outer-fold metric values and are descriptive.
- Outer folds share training observations and are not independent studies.
- Exact McNemar tests compare paired predictions only within one shared outer test fold.
- P-values are not pooled across outer folds.
- A descriptive direction count does not imply statistical significance.
- Confidence intervals from different models must not be treated as a paired significance test.
- Calibration diagnostics pool one untouched outer-fold prediction per sample and model.
- Expected calibration error and maximum calibration error depend on the configured bins.

Nested cross-validation estimates performance of the configured model-selection procedure on the available dataset. It does not establish external validity, clinical utility, or quantum advantage.

## Reproducibility

`normalize_nested_experiment_for_reproducibility` removes timestamps, output locations, artifact paths, and measured fit times. It canonicalizes finite floating-point values to 12 decimal places while retaining:

- dataset fingerprint,
- search profile,
- model and fold seeds,
- outer partition hashes,
- selected features,
- selected parameters,
- inner candidate scores,
- outer predictions and probability scores,
- probability-score source,
- pairwise comparisons,
- calibration bins and summaries,
- probability distributions,
- hashed classification errors,
- evidence statements,
- environment provenance.

Two normalized payloads from the same code, configuration, dataset, and environment should compare equal.

## Commands

Reference profile:

```bash
python -m pip install -e '.[dev]'
qob nested-cv --config configs/nested-classical.yaml
```

Sensitivity profile:

```bash
qob nested-cv --config configs/nested-classical-sensitivity.yaml
```

## Deferred scope

The current nested-CV implementation does not include:

- quantum-kernel model selection,
- repeated outer cross-validation,
- grouped or subject-aware splitting,
- external-cohort validation,
- rendered calibration or reliability figures,
- decision-curve analysis,
- subgroup analysis,
- DeLong ROC comparison,
- multiplicity-adjusted cross-dataset inference.
