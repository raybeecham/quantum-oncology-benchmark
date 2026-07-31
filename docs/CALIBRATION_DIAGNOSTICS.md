# Out-of-Fold Calibration Diagnostics

## Purpose

The nested cross-validation workflow produces one untouched outer-fold prediction for every sample and evaluated model. Version `nested-cv-1.1` uses those predictions to generate descriptive calibration and classification-error diagnostics.

These diagnostics evaluate the probability outputs of the configured model-selection procedure on the built-in benchmark dataset. They are not evidence of clinical reliability, transportability, prospective performance, or patient benefit.

## Prediction scope

Calibration calculations use `outer_fold_predictions` after all outer folds complete.

For each model:

- every included sample contributes exactly one prediction,
- predictions were generated only when that sample was in an outer test fold,
- preprocessing, feature selection, model selection, fitting, and SVM probability calibration used only the corresponding outer training partition,
- sample identity is represented by a deterministic SHA-256 hash rather than a raw row index.

The pooled prediction set is suitable for descriptive reliability analysis because no sample's score was produced by a model trained on that sample. The outer folds still share training observations, so the pooled results are not independent external replications.

## Binning contract

The default configuration uses ten uniform probability-width bins:

- `[0.0, 0.1)`
- `[0.1, 0.2)`
- continuing through
- `[0.9, 1.0]`

Empty reliability bins are omitted from `calibration_bins.csv`. Probability-distribution rows retain every configured bin for each model and true class so distributions can be compared without reconstructing missing bins.

The bin count is recorded in configuration and methodology. Changing the bin count changes expected calibration error and maximum calibration error and therefore defines a distinct experiment configuration.

## Reported diagnostics

### Calibration-in-the-large

The mean predicted positive-class probability minus observed positive-class prevalence.

- A value near zero indicates that the overall mean predicted risk matches prevalence.
- A positive value indicates higher mean predicted probability than observed prevalence.
- A negative value indicates lower mean predicted probability than observed prevalence.

This does not measure local reliability across the probability range.

### Expected calibration error

The sample-weighted mean absolute difference between each occupied bin's mean predicted probability and observed positive rate.

Expected calibration error depends on bin definitions and should not be compared across experiments with different bin counts or binning strategies without qualification.

### Maximum calibration error

The largest absolute calibration gap among occupied bins.

This measure can be unstable when a bin contains few samples. The supporting bin sample counts must be reviewed before interpretation.

### Brier score and log loss

The workflow calculates pooled out-of-fold Brier score and log loss directly from the complete prediction set.

- Brier score measures squared probability error.
- Log loss penalizes confident incorrect probabilities more strongly.

These pooled diagnostics complement, but do not replace, the mean and standard deviation of fold-level metrics already reported in `nested_summary.csv`.

### Error counts

False-positive and false-negative counts are reported for each model. `classification_errors.csv` records each error with:

- outer fold,
- search profile,
- model,
- hashed sample identifier,
- true class,
- predicted class,
- positive-class probability,
- confidence in the predicted class,
- error type.

No direct patient or raw dataset identifier is written.

## Artifact contract

### `calibration_summary.csv`

One row per model containing:

- sample and class counts,
- prevalence,
- mean predicted probability,
- calibration-in-the-large,
- expected calibration error,
- maximum calibration error,
- pooled out-of-fold Brier score,
- pooled out-of-fold log loss,
- false-positive and false-negative counts,
- binning metadata.

### `calibration_bins.csv`

One row per occupied model and probability bin containing the reliability-curve coordinates and supporting sample counts.

### `probability_distribution.csv`

One row per model, true class, and configured probability bin. This supports score-distribution plots and class-separation review.

### `classification_errors.csv`

One row per out-of-fold classification error with hashed sample identity and probability context.

## Search-profile relationship

Calibration diagnostics are generated for both versioned search profiles:

- `reference-v1`: the locked classical reference grid.
- `sensitivity-v1`: a bounded grid-boundary sensitivity analysis.

Sensitivity-profile calibration results must not silently replace the reference-profile result. The profile is carried in configuration, prediction rows, error rows, and evidence statements.

## Interpretation boundaries

- Better calibration on the built-in dataset is not clinical validation.
- Expected calibration error is bin-dependent.
- Sparse bins can make maximum calibration error unstable.
- Models may have similar discrimination but different calibration.
- Threshold-specific false-negative and false-positive counts depend on the model's default decision threshold.
- External cohorts, prevalence shift analysis, and prospective evaluation are required before translational claims.
