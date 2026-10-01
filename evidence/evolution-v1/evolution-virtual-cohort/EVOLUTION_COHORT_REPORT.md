# Virtual Tumor Cohort and Parameter Sensitivity Report

> **Research-use-only warning:** These are deterministic virtual tumors generated from
> declared parameter ranges. They are not patients, clinical predictions, or treatment guidance.

## Cohort Contract

- Protocol: `evolution-cohort-v1`
- Profile: `two-clone-virtual-cohort-v1`
- Virtual tumors: **128**
- Sampling: `latin_hypercube` with seed `42`
- Reference strategy: `continuous`
- Candidate strategy: `burden_adaptive`
- Treatment-policy settings are fixed; only declared biological parameters vary.
- Event times not reached within the horizon are capped at the simulation horizon for paired summaries.

## Paired Robustness Summary

- Candidate delayed resistant dominance in **100.0%** of virtual tumors.
- Candidate delayed the configured burden threshold in **7.0%** of virtual tumors.
- Candidate reduced tumor-burden AUC in **0.8%** of virtual tumors.
- Candidate reduced cumulative dose in **100.0%** of virtual tumors.
- Median resistance-control difference: **+190.5 days**.
- Median configured burden-threshold difference: **+0.0 days**.

## Strategy Robustness

| Strategy | Dominance reached | Burden threshold reached | Final burden median | Resistance control median | Progression control median | Dose median |
|---|---:|---:|---:|---:|---:|---:|
| burden_adaptive | 63.3% | 29.7% | 872,600 | 288.5 | 365.0 | 184.0 |
| continuous | 100.0% | 25.8% | 242,656 | 71.0 | 365.0 | 365.0 |

## Sensitivity Interpretation

Spearman coefficients describe monotonic associations within this designed parameter sample.
They are not causal effects, calibrated biological importance scores, or inferential evidence from patients.
Nominal p-values are provided for auditability and are not corrected for multiple comparisons.

## Boundaries

- Every virtual tumor remains a deterministic two-clone abstraction.
- Parameter ranges are scenario bounds, not a fitted population distribution.
- The study does not include toxicity, pharmacokinetics, immune effects, spatial structure, or stochastic evolution.
- A strategy can appear robust under these ranges and still fail under omitted biology or different ranges.
- No quantum algorithm is used in this cohort study.
