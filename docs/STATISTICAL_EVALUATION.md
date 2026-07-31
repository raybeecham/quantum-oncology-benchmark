# Statistical Evaluation Method

## Scope

Version 0.1.x adds a bounded statistical layer for the existing repeated-holdout benchmark. It is intended to quantify uncertainty and paired disagreement without converting exploratory benchmark results into clinical or quantum-advantage claims.

## Bootstrap confidence intervals

For each model and metric, the benchmark computes a 95% bias-corrected and accelerated (BCa) bootstrap interval for the mean across repeat-level metric values.

- Resampling unit: one repeated-holdout result.
- Default resamples: 5,000.
- Randomness: deterministically derived from the experiment seed, model, and metric.
- Fallback: percentile interval when BCa is degenerate.
- Minimum: at least two repeat-level values. Otherwise, the interval is recorded as unavailable.

These intervals describe variability across configured repeated holdouts. They do not represent external-cohort uncertainty, prospective validation, or subject-level bootstrap inference.

## Paired model comparison

Models evaluated within a repeat share the same test partition. Every model pair is compared with an exact McNemar test using only discordant classifications:

- Model A correct, Model B wrong.
- Model A wrong, Model B correct.

The exact two-sided p-value is calculated as a binomial test with null probability 0.5.

A separate result is recorded for every repeat. The benchmark does not pool repeated-holdout p-values because the same observations may appear in more than one test partition, violating the independence assumption of a naive pooled test.

## Evidence statement

Every experiment produces a deterministic evidence statement that:

- identifies the strongest model by mean balanced accuracy,
- distinguishes classical-only, quantum-only, and hybrid comparisons,
- refuses to claim clinical utility,
- refuses to claim quantum advantage from a simulator or an isolated benchmark result,
- describes a higher quantum mean as exploratory unless externally replicated.

## Deferred methods

The following remain outside this first statistical slice:

- DeLong tests for correlated ROC AUCs,
- nested cross-validation,
- multiplicity correction across many comparisons,
- subject-level clustered bootstrap procedures,
- calibration and decision-curve figures,
- external-cohort validation.
