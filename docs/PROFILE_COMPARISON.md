# Cross-Profile Comparison and Classical Protocol Freeze

## Purpose

The `qob compare-profiles` command compares two completed nested cross-validation artifact directories. It does not train models. It verifies that the profiles are directly comparable, joins their out-of-fold predictions, quantifies prediction changes, and produces a bounded recommendation for freezing the classical protocol before future quantum evaluation.

## Command

```bash
qob compare-profiles \
  --reference reports/nested-classical \
  --candidate reports/nested-classical-sensitivity \
  --output reports/classical-protocol-freeze
```

## Compatibility checks

The command requires both profiles to share:

- dataset fingerprint,
- dataset adapter and sample limit,
- feature count,
- seed,
- outer and inner fold counts,
- model list,
- locked primary endpoint,
- hashed sample identities,
- truth labels,
- outer-fold assignments.

A mismatch stops the comparison rather than silently aligning incompatible experiments.

## Paired evidence

Each sample contributes one out-of-fold prediction per model and profile. For each model, the comparison records:

- both profiles correct,
- both profiles wrong,
- reference only correct,
- candidate only correct,
- changed class predictions,
- balanced-accuracy difference,
- descriptive direction,
- exact two-sided McNemar p-value.

Because every sample appears once in each complete out-of-fold prediction set, the paired test is calculated across the complete matched sample set for that model. This differs from pooling dependent repeated-holdout tests.

## Parameter boundary evidence

`parameter_boundary_summary.csv` records how often each selected parameter value appeared across outer folds in each profile. The comparison also records which model search spaces changed.

The decision rules are intentionally bounded:

- unchanged search spaces remain paired controls,
- a degrading expansion retains the historical reference grid,
- an improving bounded expansion may be retained for future protocol versions,
- the historical reference result is never rewritten,
- no change is described as statistical superiority unless supported by the paired test and independent validation boundaries.

## Artifact package

- `profile_comparison.json`: complete comparison, compatibility contract, changed grids, paired results, and freeze recommendation.
- `cross_profile_summary.csv`: one paired summary row per model.
- `cross_profile_predictions.csv`: hashed sample-level joined predictions and scores.
- `parameter_boundary_summary.csv`: selected parameter frequency by profile and model.
- `PROTOCOL_FREEZE_REPORT.md`: human-readable evidence and recommendation.

## Classical protocol freeze

The comparison selects a primary classical comparator based on stability across both profiles rather than the best score from one profile. Model-specific grid recommendations remain explicit and versioned.

The freeze is methodological, not clinical. It establishes the comparator protocol to be used when a future quantum model is introduced.

## Future quantum-computer boundary

A future quantum profile should reuse the same evidence structure where technically feasible and must preserve:

- the locked primary endpoint,
- identical outer partitions,
- one prediction record per sample,
- classical comparator identities and budgets,
- feature and preprocessing provenance,
- paired prediction evidence,
- resource and execution records,
- conservative evidence statements.

A physical-QPU run must additionally record:

- provider and backend identity,
- backend calibration snapshot and timestamp,
- logical and physical qubit mapping,
- circuit depth and two-qubit gate counts after transpilation,
- shots and mitigation configuration,
- queue time and execution time,
- failed-job and retry records,
- monetary or credit cost,
- simulator and hardware distinction.

A higher hardware score on one small dataset is not quantum advantage. Any such result remains exploratory until reproduced under matched budgets and independently validated.
