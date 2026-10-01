# Evolution reference review, October 1, 2026

Executed the declared 128-tumor Latin-hypercube reference cohort, seed 42,
comparing continuous with burden-adaptive treatment over the base 365-day horizon.

| Adaptive result versus continuous | Scenarios |
|---|---:|
| Delayed resistant dominance | 128/128 |
| Lower cumulative dose | 128/128 |
| Lower tumor-burden AUC | 1/128 |
| Lower final burden | 12/128 |
| Delayed configured burden threshold | 9/128 |

Median paired differences: +190.5 horizon-capped resistance-control days,
-181 dose-days, +541,797 final burden units and +221,755,348 burden-unit days of AUC.
The median horizon-capped burden-threshold timing difference was zero.

This shows competing objectives, not a universally better policy. Resistant *fraction*
dominance is sensitive to the sensitive population denominator; it is not itself absolute
resistant burden. Future optimization should report both. Capped timing differences do
not establish control beyond the 365-day horizon.

The separate sensitivity study ran five acquired-transition settings and nine threshold
pairs, with two policies per scenario (28 outcome rows). All 14 adaptive scenarios used
less dose but had higher AUC than their corresponding continuous references. This is a
single-base-tumor sensitivity result, not cohort-wide robustness.

Cohort fingerprint: `f8b8521557c951c624e4814587b2ef6ca6090c0abb1fe882222c245455e31672`.
Sensitivity fingerprint: `a8283b5d1dcae8fc356271d52b479fb566ccf14366ff8ae1876615f3a38d9ee7`.
Complete machine-readable evidence is under `evidence/evolution-v1/`.

## Next experiment

Define and version a normalized schedule objective before tuning optimizers. Include
burden AUC, absolute resistant burden and drug exposure, with declared weights and
weight sensitivity; retain component outcomes. Separate search tumors from held-out
virtual tumors. Start with bounded exhaustive search, then equal-budget random search
and simulated annealing, recording simulator evaluations and runtime.

A simulator objective is not automatically quadratic. The attached QCi notebooks accept
explicit polynomial coefficients and distinguish integer-level variables from nonnegative
sum-constrained continuous variables. A Dirac comparison requires an explicit encoding
or a held-out-validated polynomial surrogate, plus decoding and simulator re-evaluation.
Do not identify a continuous schedule allocation with a binary schedule without a decoder.
No QCi job or quantum computation was used here.

These designed scenarios are not patients and do not demonstrate therapeutic benefit.
The model is still an illustrative two-clone abstraction. Reversible tolerance, stochastic
mutation, spatial effects, toxicity and pharmacokinetics remain absent. Model freeze is
pending objective design and numerical-resolution review.

Validation: 64 core tests passed (2 optional quantum tests deselected); Ruff and strict mypy passed. Full 128-tumor replay matched the cohort fingerprint. Two existing cohort tests emitted constant-input Spearman warnings for undefined correlations.
