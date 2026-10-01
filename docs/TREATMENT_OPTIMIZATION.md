# Treatment schedule optimization protocol v1

This is computational oncology methods research using an illustrative deterministic
competitive two-clone model. No drug, patient, toxicity model or calibrated biological
population is represented. No result is treatment guidance.

## Run

```bash
pip install -e ".[dev]"
qob optimize-treatment --config configs/treatment-optimization-smoke.yaml
qob optimize-treatment --config configs/treatment-optimization-reference.yaml
qob optimize-treatment --config configs/treatment-optimization-robust.yaml
```

The reference experiments use the existing 365-day model. Equal-duration treatment
intervals have binary intensities. The ODE never integrates across a schedule switch.
Recorded points are no farther apart than the base time step, and include every switching
boundary. AUC uses trapezoidal integration over those recorded points. The schedule is
open loop; adaptive feedback policies are separate descriptive controls.

The exact profile uses 12 intervals, one search scenario and 16 disjoint evaluation
scenarios. The robust profile uses eight intervals, eight search scenarios and 32 disjoint
evaluation scenarios. A cohort profile supplies biological ranges; the optimization base
profile supplies the resolved biological model. The original cohort profile's base path
is not used as a second, implicit model. A single Latin-hypercube design is partitioned
before search. These are disjoint designed scenarios, not independent patient samples.

The smoke profile has no cohort and reuses the same scenario for evaluation. Its artifacts
explicitly label this as same-reference-scenario evaluation, not held-out evidence.

## Objective

For tumor i, minimize

\[
L_i(x) = w_B \frac{\operatorname{AUC}(S_i+R_i)}{B_i(0)T}
       + w_R \frac{R_i(T)}{B_i(0)}
       + w_D \frac{\int_0^T u_x(t)\,dt}{T}.
\]

Across search tumors, minimize mean(L) + lambda * population_std(L). The initial
weights are (1, 1, 0.25). The robust profile sets lambda to 0.25; the exact-reference
profile sets it to zero. Values can exceed one and are not clipped.

These dimensionless weights and lambda are declared research preferences, not empirical
estimates of clinical utility. Dose is cumulative exposure, not toxicity. Absolute resistant
burden is deliberately used rather than resistant fraction. Every endpoint is retained so
a favorable combined score cannot hide a worsening component. This first objective does
not directly penalize transient progression events; maximum burden and progression
threshold times remain reported diagnostics.

## Hard constraints and domain

- At most the declared number of intervals can be treated.
- No treated run can exceed the declared consecutive-interval cap.
- Every variable must be exactly 0 or 1.

There is no minimum treatment requirement, holiday duration, pharmacokinetic carryover
or clinical scheduling rule. The all-off schedule is feasible. Continuous treatment and
adaptive feedback can violate exposure/run caps or differ from the binary grid, and are
reported as descriptive controls rather than eligible competitors for the same optimum.

## Search and accounting

Exact enumeration is capped at 12 intervals. Its best objective is a numerical optimum
on this finite feasible domain and these search tumors. It is not an analytic or clinical
optimum, and it does not establish the optimum on the evaluation tumors.

Random search samples the feasible domain uniformly without replacement for <=12
intervals. Larger profiles use uniform binary proposals and reject infeasible ones.
Simulated annealing starts at all-off and proposes one-bit flips. It rejects infeasible
proposals before calling the simulator, uses geometric cooling based on unique schedule
evaluations, and may reuse cached values. Temperature settings and seeds are fixed.

Each run has a private cache. Oracle labels are not supplied to random or annealing.
Their budgets cap unique feasible schedule evaluations, not wall time. The artifact records
unique evaluations, requests, cache hits, tumor simulations, proposals, simulator time,
wall time and stopping reason. A proposal cap bounds revisiting/stalled searches. A method
that stops early is not described as using a full matched budget.

A successful oracle hit is a loss within 1e-8 of the numerical oracle, not a distinct
biological finding. Repeats are search seeds, not independent biological replications.
Winners are fixed before evaluation. Evaluation tumors never tune schedules, objective
weights, temperature settings or method selection.

## Evidence package

The JSON contains configurations, resolved model/ranges, scenario partition, resource
records, all search traces, winner outcomes and control policies. CSVs provide search
results/traces, evaluation outcomes/summaries, reference-policy outcomes and cohort
parameters. The Markdown report supports a direct review of objective gaps and schedules.

The fingerprint excludes timestamps, paths, environment and measured runtime. Scientific
floats are rounded to ten decimal places for fingerprinting; complete stored values retain
full precision. Environment and source commit are retained separately.

## Offline Dirac formulation

```bash
qob compile-dirac-treatment \
  --experiment reports/treatment-optimization-reference/treatment_optimization_experiment.json \
  --output reports/dirac-treatment-reference
```

The treatment simulator is nonlinear in schedule bits. Its objective is not automatically
a QUBO. This command uses previously acquired exact-oracle labels to fit a ridge-regularized
linear/pairwise surrogate. A fixed 70/30 schedule partition separates fit and validation.
Validation reports error, rank correlation, design rank, the surrogate winner's true regret,
and original-simulator replay on search and evaluation scenarios.

This is a representation experiment using precomputed exhaustive labels. Label acquisition
cost is reported; it is not a budget-matched comparison with the classical searches.

The integer polynomial includes one native integer slack variable s in 0..budget:

\[
\widehat J(x) + P(\sum_t x_t+s-budget)^2
+ P\sum_{\text{forbidden runs}} \prod_{t\in\text{run}} x_t.
\]

With P = 1 + sum(abs(surrogate coefficients)), every infeasible integer assignment is
above every feasible assignment in exact arithmetic. This encodes both constraints;
feasible assignments have no penalty. Consecutive caps through four require degree at
most five. The constant offset is retained outside the uploaded polynomial. Coefficients
are divided by their maximum absolute value; the scale and constant restore energy units.
The exact-arithmetic bound does not guarantee an analog machine's resolution or sampling.

The file and job template follow the supplied Dirac-3 beginner notebook's integer example:
`sample-hamiltonian-integer`, binary levels for schedule variables and multiple levels for
slack, one-based polynomial indices with zero padding. No `qci-client`, token, network
submission or hardware execution is used. Current cloud acceptance, variable/degree capacity,
precision, device allocation and Dirac-3S compatibility remain unverified. The offline
compiled ground state is exhaustively verified on the bounded domain, including constraints.

Decoder validation rejects noninteger, missing and infeasible samples. It does not silently
round fractional schedules or repair violations. A future real job must record queue/device
usage, raw counts, energies, constraint failures, decoding work and re-evaluation costs.
A validated surrogate and a compiled payload do not demonstrate hardware performance.

## Review frozen winners

Run `python scripts/review_treatment.py --experiment PATH --output PATH --plots` from the repository root. Optional plotting dependencies install with `pip install -e ".[plots]"`. The review halves the numerical time step and varies resistance/exposure weights only among already frozen winners on search tumors. It does not reoptimize under new weights or validate the complete oracle ranking at finer resolution.

## Import sample results

`qob import-dirac-treatment --compiled PATH --results PATH --output PATH`
accepts a completed response or its `results` mapping, with `solutions`, positive integer
`counts`, and optional finite `energies`. It retains raw responses, rejects invalid schedules
without repair, re-evaluates every unique valid schedule on both scenario partitions, and
reports count-weighted feasibility and observed oracle-hit fractions. Cached simulator
work is recorded. Compiled evidence integrity and current scenario definitions are checked.

Imported files have unverified external provenance by default; importing a file does not
verify hardware execution. Local validation fixtures require `--origin local_test_samples`.
Reported machine energy is retained separately from locally restored polynomial energy.
These sample frequencies are descriptive; they are not an independent success-probability
estimate or a budget-matched hardware comparison.
