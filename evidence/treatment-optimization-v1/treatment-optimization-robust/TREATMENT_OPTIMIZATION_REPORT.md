# Classical Treatment Schedule Benchmark

Deterministic scenario research, without patient calibration or therapeutic-benefit claims.

Protocol: `treatment-optimization-v1`. Profile: `binary-schedule-robust-cohort-v1`.
Binary intervals: 8; exposure cap: 6; consecutive cap: 3.
Search tumors: 8; evaluation tumors: 32.
Evaluation design: `disjoint_virtual_tumor_scenarios`.

Objective: mean normalized loss + risk_aversion * population standard deviation of loss.
Loss combines burden AUC / (initial burden * horizon), terminal absolute resistant cells / initial burden, and dose-days / horizon.
Weights are declared research preferences; dose is an exposure proxy, not a toxicity model.

| Method | Repeat | Schedule | Search objective | Evaluation objective | Unique evaluations | Gap to oracle | Stop |
|---|---:|---|---:|---:|---:|---:|---|
| exact | 0 | `11101101` | 1.47383 | 3.21323 | 208 | 0 | complete_feasible_domain |
| random | 0 | `11101011` | 1.47886 | 3.19461 | 64 | 0.00503399 | budget |
| random | 1 | `11101101` | 1.47383 | 3.21323 | 64 | 0 | budget |
| random | 2 | `11011101` | 1.48836 | 3.21436 | 64 | 0.0145294 | budget |
| annealing | 0 | `11011101` | 1.48836 | 3.21436 | 64 | 0.0145294 | budget |
| annealing | 1 | `11011101` | 1.48836 | 3.21436 | 64 | 0.0145294 | budget |
| annealing | 2 | `11101011` | 1.47886 | 3.19461 | 64 | 0.00503399 | budget |

The exact oracle is exhaustive only on the declared feasible binary grid and search tumors, with numerical ODE scoring. It is not a continuous-control or clinical optimum.
Oracle effort is reported separately. Random and annealing have identical maximum unique-evaluation budgets, private caches and declared seeds. Annealing may stop early at its proposal cap.
Evaluation scenarios were never used for optimizer selection, temperature tuning or objective-weight choice. Repeated search seeds are computational repetitions, not independent biological samples.
Reference policy results are descriptive controls. Policies that violate the exposure limits or differ from the schedule grid are not eligible competitors in the constrained-search optimum.
No quantum computation was used. A Dirac adapter requires explicit polynomial formulation or validated surrogate and simulator re-evaluation of decoded schedules.

Scientific fingerprint: `dcfc9347fc5ed91feda5fdf09f06a6abb744bc3ca0522f6ea6c712f8cb24f77e`.
