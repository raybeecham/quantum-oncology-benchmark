# Roadmap

## Current focus

The benchmark foundation, statistical evaluation layer, reproducibility contract, classical nested cross-validation, grid-boundary sensitivity analysis, out-of-fold calibration diagnostics, paired classical protocol freeze, resource-bounded quantum execution foundation, and deterministic two-clone evolution simulator are complete.

The current computational milestone is `evolution-cohort-v1`: test treatment-policy robustness across deterministic virtual tumors, separate biological uncertainty from policy optimization, and identify which declared assumptions most strongly track paired outcome changes. This track remains separate from quantum execution until the classical simulator, uncertainty model, and policy objective are stable.

## Version 0.1, benchmark foundation

- [x] Public demonstration dataset.
- [x] Binary CSV adapter.
- [x] Leakage-resistant preprocessing.
- [x] Four classical baselines.
- [x] Fidelity statevector quantum kernel.
- [x] Repeated holdout evaluation.
- [x] JSON, CSV, Markdown, and dashboard outputs.
- [x] Dataset fingerprint and environment provenance.
- [x] CI and optional quantum smoke tests.

## Version 0.2, stronger statistical controls

- [x] Repeat-level bootstrap confidence intervals.
- [x] Exact paired statistical comparisons within shared test partitions.
- [x] Pairwise comparison artifacts and conservative evidence statements.
- [x] Machine-precision-aware reproducibility normalization.
- [x] Self-contained partition hashes and descriptive direction counts.
- [x] Classical nested cross-validation.
- [x] Locked balanced-accuracy endpoint.
- [x] Locked classical hyperparameter search spaces.
- [x] Outer-fold prediction artifacts.
- [x] Inner-search candidate artifacts.
- [x] End-to-end nested-CV CLI smoke testing.
- [x] Full classical nested-CV reference baseline and review.
- [x] Versioned classical reference and grid-boundary sensitivity profiles.
- [x] Pooled out-of-fold calibration summary and reliability-bin artifacts.
- [x] Probability-distribution and hashed classification-error artifacts.
- [x] Full `sensitivity-v1` execution and review.
- [x] Cross-profile paired comparison and parameter-boundary artifacts.
- [x] Execute the cross-profile comparison and approve the classical protocol freeze.
- [ ] Calibration and reliability figures generated from the diagnostic CSV contract.
- [ ] Repeated outer cross-validation.
- [ ] Subject-grouped and site-grouped splitting.
- [ ] Missingness and class-imbalance reports.

## Version 0.3, GDC cohort tooling

- [x] Public GDC project and file manifest client.
- [x] Reproducible query receipt.
- [x] Explicit no-download metadata boundary.
- [ ] TCGA LUAD/LUSC cohort recipe.
- [ ] Case and sample deduplication rules.
- [ ] Biospecimen and assay metadata validation.
- [ ] Download boundary enforcement for public versus controlled data.
- [ ] Cohort-definition artifact and data dictionary.
- [ ] Subject independence and site-effect validation.

## Version 0.4, quantum and hardware evaluation

- [x] Formal `quantum-protocol-v1` resource and evidence specification.
- [x] Backend-neutral quantum-kernel execution interface.
- [x] Pre-execution limits for features, samples, kernel entries, repetitions, shots, and total shots.
- [x] Local Qiskit fidelity-statevector adapter.
- [x] Simulator resource records with circuit depth, size, two-qubit gates, timing, and fingerprints.
- [x] Exact and finite-shot simulator capability declaration.
- [ ] Q0 command and artifact package for contract-smoke execution.
- [ ] Q1 matched simulator benchmark with rerun frozen classical controls.
- [ ] Matched classical and quantum tuning budgets.
- [ ] Q2 finite-shot sensitivity analysis.
- [ ] Backend-neutral remote job lifecycle interface.
- [ ] Physical-QPU provider adapter.
- [ ] Backend property and calibration snapshot.
- [ ] Transpilation and physical-qubit accounting.
- [ ] Shot, queue, execution, retry, and cost records.
- [ ] Noise and mitigation experiments.
- [ ] Simulator-versus-hardware comparison.

## Version 0.5, evolutionary tumor dynamics

- [x] Versioned deterministic two-clone protocol and YAML profile.
- [x] Competitive sensitive and resistant population dynamics.
- [x] No-treatment, continuous, fixed-intermittent, and burden-adaptive policies.
- [x] Piecewise SciPy integration with explicit treatment schedules.
- [x] Burden, resistance, diversity, progression, dose, and cycle metrics.
- [x] Trajectory, schedule, summary, event, JSON, and Markdown artifacts.
- [x] Deterministic replay normalization and CI smoke profile.
- [x] Deterministic Latin-hypercube parameter-sensitivity analysis.
- [x] Virtual-tumor parameter cohorts with matched policy comparisons.
- [x] Strategy robustness quantiles and horizon-capped event summaries.
- [x] Descriptive Spearman sensitivity artifacts and cohort fingerprinting.
- [ ] Execute and review the full 128-tumor reference cohort.
- [ ] One-way acquired-resistance sensitivity profile.
- [ ] Policy-threshold sensitivity separated from biological uncertainty.
- [ ] Reversible drug-tolerant state model.
- [ ] Stochastic branching or Gillespie model.
- [ ] Multi-clone and multi-drug dynamics.
- [ ] Spatial or agent-based tumor ecosystem.
- [ ] Classical treatment-policy optimization benchmark.
- [ ] Matched hybrid or quantum optimization benchmark.

## Version 0.6, broader oncology workloads

- [ ] Drug-response classification.
- [ ] Survival-analysis research adapter.
- [ ] Radiomics and compact embedding inputs.
- [ ] Multi-omics late-fusion baseline.
- [ ] Molecular and active-space chemistry benchmark design.

## Version 1.0, independently reproducible benchmark

- [ ] External cohort.
- [ ] Independent reproducer protocol.
- [ ] Versioned benchmark registry.
- [ ] Published model and data cards.
- [ ] Stable output schemas.
- [ ] Formal evidence grading for competitive, superior, and advantage claims.
