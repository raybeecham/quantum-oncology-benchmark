# Evolution sensitivity v1

Run `qob evolve-sensitivity --config configs/evolution-sensitivity.yaml`.
Use `--output PATH` to override the destination.

The acquired-transition sweep varies only the continuous one-way sensitive-to-resistant
rate. It is not a mechanistic mutation model or a treatment-induced-only mechanism.
The policy sweep varies stop/restart thresholds in a Cartesian grid while every biological
parameter, including the transition rate, remains fixed at the base profile. All grid pairs
must satisfy 0 < stop < restart; invalid inputs fail before simulation or output.

Rates and thresholds are illustrative scenario choices, not fitted clinical values.
The protocol runs continuous and burden-adaptive policies for every scenario, records
raw summaries and paired burden-AUC and dose deltas, and does not pick a winning policy.

Artifacts are `sensitivity_outcomes.csv` and `evolution_sensitivity_experiment.json`.
The JSON records the resolved biological base configuration, sweep inputs, results,
environment and SHA-256 fingerprint. Output locations and timestamps do not enter the
fingerprint. Event times remain sampled on the base model's policy time grid.

These are local scenario sweeps, not an acquired-resistance population cohort or a
policy robustness test across heterogeneous tumors. A later matched-cohort extension
must reuse identical virtual tumors across policy settings.
