# Treatment Optimization Reference Review, October 1, 2026

## Completed experiments

| Profile | Search / evaluation tumors | Binary intervals | Feasible schedules in oracle | Budget per random/annealing run |
|---|---:|---:|---:|---:|
| Exact reference | 1 / 16 | 12 | 3,519 | 128 |
| Cohort robust | 8 / 32 | 8 | 208 | 64 |

The oracle scores the entire feasible grid on the search scenarios. There are three
repetitions per budgeted method. Every final random and annealing run consumed its full
unique-schedule budget. The oracle's much larger effort is recorded separately.

## Numerical search findings

| Profile | Oracle objective | Random median gap | Annealing median gap | Surrogate-selected true regret |
|---|---:|---:|---:|---:|
| reference | 3.599226 | 0.004883 | 0.003551 | 0.098355 |
| robust | 1.473827 | 0.005034 | 0.014529 | 0.005034 |

One of three cohort random-search repetitions recovered the numerical oracle. The final
annealing runs did not recover it. No final budgeted reference run recovered its oracle.
These three search seeds do not establish statistical superiority or a success probability.
The best schedule on search scenarios was not consistently the lowest-loss schedule on
evaluation scenarios. Held-out outcomes describe robustness of frozen schedules; they do
not identify a new evaluation-selected winner.

An initial single-chain annealing pilot repeatedly revisited cached states and reached its
proposal cap before consuming budget. The implementation added a fixed novelty-restart rule
and reran both experiments. This was an implementation-baseline correction based on search
resource records, not temperature or objective tuning using evaluation outcomes. Only the
final protocol results are included in this evidence package.

## Surrogate and native integer encoding

The quadratic fit used disjoint 70/30 schedule-label partitions from the completed oracle.
The reference validation rank correlation was 0.8668; the cohort value was 0.9904.
Normalized validation RMSE (divided by the observed validation objective range) was about
0.0325 and 0.0261, respectively. Those range-normalized values can hide errors near the
optimum. In particular, the reference surrogate's chosen schedule had true regret 0.09836.
Average prediction quality does not guarantee a useful optimum ranking.

A native integer slack variable encodes the exposure cap, and polynomial forbidden-run
terms encode consecutive limits. The reference payload has 13 variables and degree five;
the cohort payload has nine variables and degree four. All binary assignments with their
minimizing slack were checked locally, and the encoded ground states were feasible.
Small-domain regression tests independently enumerate every slack assignment and verify
penalty energies. Analog precision, cloud acceptance and hardware performance are untested.

The compiled files are dry-run artifacts. Exhaustive label-acquisition costs remain charged
to the representation study; this is not a fair end-to-end hardware-versus-classical contest.
The sample-import evidence uses explicitly labeled local fixtures containing both valid and
invalid samples. It tests count-weighted rejection and simulator re-evaluation, with no
hardware samples, no fabricated device usage and no hardware provenance claim.

## Resolution and weight diagnostics

For frozen winners, halving the numerical time step changed the combined objective by at
most 0.00003678 in the reference experiment and 0.00001383 in the cohort experiment. This
is a winner-replay check, not a verification of the entire oracle ranking at finer resolution.
Weight diagnostics cover resistance weights 0.5, 1 and 2 and exposure weights 0.1, 0.25 and
0.5 among frozen search winners only. They do not establish optima under alternative weights.

## Evidence and limits

Artifacts are under `evidence/treatment-optimization-v1/`, including complete search traces,
scenarios, outcomes, resources, surrogate predictions, native polynomial/job templates,
resolution/preference CSVs, plots, local-import fixtures and source-content hashes.

All 87 tests passed, including the optional Qiskit simulator tests, after installing the
optional quantum dependencies. Ruff and strict mypy passed. The two pre-existing
constant-input Spearman warnings remain confined to cohort tests. Separate complete smoke replays matched scientific
content and fingerprints. Numerical, constraint, objective, cohort-partition, cache/budget,
surrogate-integrity and sample-decoding contracts have regression coverage.

These are designed two-clone scenarios, not patients. Exposure is not toxicity. Weights are
methodological preferences, and pre-existing scenario bounds are not a fitted population.
Stochastic evolution, reversible tolerance, spatial biology and multiple drugs are deferred.
The objective does not directly penalize transient progression, and that diagnostic must be
reviewed alongside the combined score. No therapeutic benefit or quantum advantage is claimed.

The next defensible hardware experiment requires a locked acquisition/validation protocol
for the surrogate, stronger search replications, provider capacity/precision confirmation,
verified raw job provenance and full label/encoding/decoding/re-evaluation cost accounting.
