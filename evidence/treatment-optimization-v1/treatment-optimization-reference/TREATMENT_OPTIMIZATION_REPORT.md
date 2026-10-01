# Classical Treatment Schedule Benchmark

Deterministic scenario research, without patient calibration or therapeutic-benefit claims.

Protocol: `treatment-optimization-v1`. Profile: `binary-schedule-exact-reference-v1`.
Binary intervals: 12; exposure cap: 9; consecutive cap: 4.
Search tumors: 1; evaluation tumors: 16.
Evaluation design: `disjoint_virtual_tumor_scenarios`.

Objective: mean normalized loss + risk_aversion * population standard deviation of loss.
Loss combines burden AUC / (initial burden * horizon), terminal absolute resistant cells / initial burden, and dose-days / horizon.
Weights are declared research preferences; dose is an exposure proxy, not a toxicity model.

| Method | Repeat | Schedule | Search objective | Evaluation objective | Unique evaluations | Gap to oracle | Stop |
|---|---:|---|---:|---:|---:|---:|---|
| exact | 0 | `010111101111` | 3.59923 | 1.80016 | 3519 | 0 | complete_feasible_domain |
| random | 0 | `110111001111` | 3.61458 | 1.68491 | 128 | 0.0153572 | budget |
| random | 1 | `011110110111` | 3.60411 | 1.75457 | 128 | 0.00488287 | budget |
| random | 2 | `011101101111` | 3.60278 | 1.75728 | 128 | 0.00355091 | budget |
| annealing | 0 | `011101101111` | 3.60278 | 1.75728 | 128 | 0.00355091 | budget |
| annealing | 1 | `011101111011` | 3.60419 | 1.76264 | 128 | 0.00495954 | budget |
| annealing | 2 | `011011101111` | 3.60144 | 1.7678 | 128 | 0.00221251 | budget |

The exact oracle is exhaustive only on the declared feasible binary grid and search tumors, with numerical ODE scoring. It is not a continuous-control or clinical optimum.
Oracle effort is reported separately. Random and annealing have identical maximum unique-evaluation budgets, private caches and declared seeds. Annealing may stop early at its proposal cap.
Evaluation scenarios were never used for optimizer selection, temperature tuning or objective-weight choice. Repeated search seeds are computational repetitions, not independent biological samples.
Reference policy results are descriptive controls. Policies that violate the exposure limits or differ from the schedule grid are not eligible competitors in the constrained-search optimum.
No quantum computation was used. A Dirac adapter requires explicit polynomial formulation or validated surrogate and simulator re-evaluation of decoded schedules.

Scientific fingerprint: `0a573849e88a6f44950e82bcc1b6d7685ebfd2de8991a46c2baf7437be14f52b`.
