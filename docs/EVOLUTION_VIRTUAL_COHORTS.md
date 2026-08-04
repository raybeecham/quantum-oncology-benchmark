# Virtual Tumor Cohorts and Parameter Sensitivity

## Purpose

`evolution-cohort-v1` evaluates whether a treatment-policy result remains stable when the biological assumptions of the deterministic two-clone model are varied. It creates a designed cohort of **virtual tumors**, runs the same treatment policies against every virtual tumor, and summarizes matched outcome differences.

A virtual tumor is a parameter scenario. It is not a patient, a fitted digital twin, or a draw from a clinically estimated population distribution.

## Command

```bash
qob evolve-cohort --config configs/evolution-virtual-cohort.yaml
```

An output directory can be overridden without changing the protocol:

```bash
qob evolve-cohort \
  --config configs/evolution-virtual-cohort.yaml \
  --output reports/evolution-virtual-cohort-run-2
```

## Study design

The reference profile uses deterministic Latin-hypercube sampling with a recorded seed. Each sampled biological parameter occupies the declared range broadly, reducing the clustering that can occur with independent random draws.

For each virtual tumor:

1. The initial total burden is held equal to the base evolution profile.
2. The sampled initial resistant fraction divides that burden into sensitive and resistant populations.
3. Sampled growth, treatment-response, and competition assumptions replace the corresponding base values.
4. The continuous-treatment reference policy is simulated.
5. The burden-adaptive candidate policy is simulated with the same virtual tumor.
6. Outcomes are compared as a matched pair.

Treatment thresholds and scheduling rules are held fixed. This separates **biological robustness analysis** from later **policy optimization**.

## Reference parameter ranges

The repository profile varies:

- initial resistant fraction,
- sensitive growth rate,
- resistant growth rate,
- sensitive drug-kill rate,
- resistant drug-kill rate,
- resistant effect on sensitive cells,
- sensitive effect on resistant cells.

The ranges are scenario bounds chosen for computational exploration. They are not confidence intervals, prevalence estimates, or fitted distributions from patients.

The one-way sensitive-to-resistant transition rate remains fixed in the initial cohort. Acquired-transition uncertainty should be introduced as a separate versioned cohort profile so pre-existing selection and acquired transition are not silently mixed.

## Paired outcomes

Each virtual tumor produces one row for each policy and one paired comparison row. Paired outcomes include:

- difference in resistance-control days,
- difference in configured burden-threshold control days,
- tumor-burden area-under-the-curve difference,
- final burden difference,
- cumulative dose-days difference.

Positive timing differences favor the candidate policy. Negative burden and dose differences favor the candidate policy.

### Horizon-capped event times

A dominance or burden-threshold event may not occur before the simulation ends. For paired descriptive summaries, an event not reached is assigned the configured simulation horizon.

This is a bounded comparison convention, not survival analysis. The report must not describe a horizon-capped value as an observed event time or claim control beyond the simulation horizon.

## Strategy robustness summaries

For each policy, the cohort reports:

- fraction reaching resistant dominance,
- fraction reaching the configured burden threshold,
- 5th, 25th, 50th, 75th, and 95th percentiles for final burden,
- the same quantiles for tumor-burden AUC,
- horizon-capped resistance-control days,
- horizon-capped burden-threshold control days,
- cumulative dose-days.

The paired summary reports the fraction of virtual tumors in which the candidate policy:

- delays resistant dominance,
- delays the configured burden threshold,
- lowers tumor-burden AUC,
- lowers final burden,
- lowers cumulative dose.

These outcomes are deliberately kept separate. A policy can lower dose while increasing burden, or delay resistance while failing another objective.

## Parameter sensitivity

The cohort calculates Spearman rank correlations between every sampled biological parameter and every paired outcome delta.

These coefficients are **descriptive global sensitivity associations** within the designed sample. They can identify parameters that deserve closer one-at-a-time analysis or narrower calibration work, but they are not:

- causal effects,
- patient-level predictors,
- calibrated biological importance scores,
- proof that one mechanism dominates real tumors.

Nominal p-values are retained for auditability. They are not corrected for multiple comparisons and should not be treated as confirmatory evidence.

## Artifact package

Each run writes:

- `evolution_cohort_experiment.json`: complete cohort configuration, base model, sampling contract, virtual tumors, outcomes, paired comparisons, summaries, sensitivity results, environment, and fingerprint.
- `virtual_tumors.csv`: one sampled parameter row per virtual tumor.
- `virtual_tumor_outcomes.csv`: one strategy outcome row per virtual tumor.
- `paired_strategy_comparisons.csv`: matched continuous-versus-adaptive deltas.
- `strategy_robustness_summary.csv`: cohort quantiles and event fractions by strategy.
- `parameter_sensitivity.csv`: descriptive Spearman associations.
- `EVOLUTION_COHORT_REPORT.md`: bounded human-readable summary.

Full time-resolved trajectories are not retained for every virtual tumor in this protocol. This keeps the cohort artifact bounded and avoids duplicating large intermediate arrays. Individual scenarios can be replayed later from their parameter rows and the recorded base profile.

## Reproducibility

The cohort fingerprint includes:

- cohort protocol and sampling configuration, excluding output paths,
- base evolution configuration, excluding output paths,
- every sampled virtual-tumor parameter row,
- every paired strategy comparison.

The generated timestamp, measured runtime, and artifact locations are operational metadata and are not scientific inputs.

## Interpretation boundaries

The cohort remains a deterministic two-clone abstraction. It does not include:

- stochastic mutation, extinction, or clonal drift,
- drug-specific pharmacokinetics or toxicity,
- immune or stromal dynamics,
- spatial structure or drug penetration,
- metastatic compartments,
- reversible drug-tolerant states,
- clinical calibration or patient outcomes.

Robustness within a declared parameter box is not universal robustness. Results may change when the ranges, omitted biology, policy rules, or model family change.

No quantum algorithm is used in `evolution-cohort-v1`. Quantum or hybrid optimization remains deferred until the classical simulator, uncertainty model, and policy objective are stable.
