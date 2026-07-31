# Classical Nested Cross-Validation Protocol

## Purpose

The `qob nested-cv` mode estimates the performance of a bounded classical model-selection procedure while keeping each outer test fold outside preprocessing, feature selection, hyperparameter selection, calibration, and refitting.

It is a separate evaluation mode. It does not replace the repeated-holdout benchmark or change the meaning of existing artifacts.

## Locked defaults

The reference configuration is `configs/nested-classical.yaml`:

- Primary endpoint: balanced accuracy.
- Primary endpoint status: locked.
- Outer splitter: five-fold stratified cross-validation with shuffling.
- Inner splitter: three-fold stratified cross-validation with shuffling.
- Feature count: eight.
- Models: logistic regression, RBF SVM with separately calibrated probability scores, random forest, and histogram gradient boosting.
- Parallelism: one job for deterministic execution and bounded resource use.

The CLI permits a smaller model subset for development and targeted studies, but the reference configuration includes all four classical comparators.

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
10. Store fold-level metrics, predictions, selected features, parameters, probability-score source, partition hashes, and inner-search results.

The outer test data is never passed to `fit`, `GridSearchCV`, preprocessing, feature selection, calibration, or parameter selection.

## Bounded search spaces

### Logistic regression

- `C`: `0.1`, `1.0`, `10.0`

Fixed controls include class balancing, the `liblinear` solver, and a 5,000-iteration limit.

### RBF SVM with training-only score calibration

- `C`: `0.1`, `1.0`, `10.0`
- `gamma`: `scale`, `0.01`, `0.1`

The primary classification endpoint is tuned and evaluated from the selected SVM pipeline without using the deprecated internal probability mode. After inner selection, a clone of the complete selected pipeline, including imputation, scaling, and feature selection, is wrapped by `CalibratedClassifierCV` and fitted only on the outer training partition. The calibrated clone supplies probability scores for ROC AUC, average precision, Brier score, and log loss. It does not replace the selected classifier's class predictions used for balanced accuracy, sensitivity, specificity, F1, or paired McNemar comparisons.

### Random forest

- `n_estimators`: `200`, `500`
- `max_depth`: unrestricted, `5`, `10`

Fixed controls include balanced subsampling, a minimum leaf size of two, deterministic seeds, and one worker.

### Histogram gradient boosting

- `learning_rate`: `0.05`, `0.1`
- `max_leaf_nodes`: `15`, `31`

Fixed controls include 250 boosting iterations and L2 regularization of `1.0`.

These grids are intentionally limited. They support a fair, reproducible comparison rather than an unlimited optimization contest.

## Artifact package

Every completed run writes:

- `nested_experiment.json`: complete structured experiment record.
- `outer_fold_results.csv`: selected configuration, provenance, probability-score source, and final metrics for each model and outer fold.
- `outer_fold_predictions.csv`: hashed sample identifier, truth, prediction, and positive-class score.
- `inner_search_results.csv`: every bounded candidate score and selected-candidate flag.
- `nested_summary.csv`: aggregate outer-fold metrics and descriptive intervals.
- `nested_pairwise_comparisons.csv`: exact McNemar comparisons within each shared outer test fold.
- `NESTED_CV_REPORT.md`: human-readable method, results, evidence statement, and limitations.

Raw row indices are not published. Partition membership and sample identity are represented with deterministic SHA-256 hashes.

## Statistical interpretation

- Bootstrap intervals resample outer-fold metric values and are descriptive.
- Outer folds share training observations and are not independent studies.
- Exact McNemar tests compare paired predictions only within one shared outer test fold.
- P-values are not pooled across outer folds.
- A descriptive direction count does not imply statistical significance.
- Confidence intervals from different models must not be treated as a paired significance test.

Nested cross-validation estimates performance of the configured model-selection procedure on the available dataset. It does not establish external validity, clinical utility, or quantum advantage.

## Reproducibility

`normalize_nested_experiment_for_reproducibility` removes timestamps, output locations, artifact paths, and measured fit times. It canonicalizes finite floating-point values to 12 decimal places while retaining:

- dataset fingerprint,
- model and fold seeds,
- outer partition hashes,
- selected features,
- selected parameters,
- inner candidate scores,
- outer predictions and probability scores,
- probability-score source,
- pairwise comparisons,
- evidence statements,
- environment provenance.

Two normalized payloads from the same code, configuration, dataset, and environment should compare equal.

## Reference command

```bash
python -m pip install -e '.[dev]'
qob nested-cv --config configs/nested-classical.yaml
```

## Deferred scope

The first nested-CV implementation does not include:

- quantum-kernel model selection,
- repeated outer cross-validation,
- grouped or subject-aware splitting,
- external-cohort validation,
- decision-curve analysis,
- subgroup analysis,
- DeLong ROC comparison,
- multiplicity-adjusted cross-dataset inference.
