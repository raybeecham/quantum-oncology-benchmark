# Roadmap

## Current focus

The benchmark foundation, statistical evaluation layer, reproducibility contract, classical nested cross-validation engine, full classical reference baseline, grid-boundary sensitivity run, and out-of-fold calibration diagnostics are complete.

The immediate milestone is to produce and review the paired cross-profile comparison, freeze the classical comparator protocol, and preserve the resulting evidence package before designing a matched, resource-bounded quantum nested-CV protocol.

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
- [x] Cross-profile paired comparison and parameter-boundary artifact implementation.
- [ ] Execute the cross-profile comparison and approve the classical protocol freeze.
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

- [ ] Resource-bounded quantum nested-CV protocol.
- [ ] Matched classical and quantum tuning budgets.
- [ ] Backend-neutral execution interface.
- [ ] IBM Runtime adapter.
- [ ] Backend property snapshot.
- [ ] Transpilation and physical-qubit accounting.
- [ ] Shot, queue, execution, and cost records.
- [ ] Noise and mitigation experiments.
- [ ] Simulator-versus-hardware comparison.

## Version 0.5, broader oncology workloads

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
