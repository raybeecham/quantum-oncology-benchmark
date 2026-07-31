<div align="center">

# Quantum Oncology Benchmark

### Reproducible evaluation of classical and quantum machine learning for oncology research

[![CI](https://github.com/raybeecham/quantum-oncology-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/raybeecham/quantum-oncology-benchmark/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%E2%80%93%203.13-3776AB.svg)](pyproject.toml)
[![Qiskit](https://img.shields.io/badge/quantum-Qiskit-6929C4.svg)](https://qiskit.org/)
[![Research Use Only](https://img.shields.io/badge/use-research%20only-red.svg)](MODEL_CARD.md)

</div>

<p align="center">
  <img
    src="docs/assets/cancer-cell-dark.png"
    alt="Quantum Oncology Benchmark Cancer Cell Anatomy"
    width="100%"
  />
</p>

> [!IMPORTANT]
> **Research use only.** This project is not a medical device, diagnostic system, treatment recommendation, or clinical decision-support product. It does not claim that quantum computing improves cancer outcomes or that any benchmark result demonstrates quantum advantage.

## Overview

The **Quantum Oncology Benchmark** is an open, reproducible framework for evaluating classical machine-learning models and quantum-kernel methods on binary cancer-classification tasks.

The project is organized around a narrow, falsifiable question:

> Under a shared and auditable evaluation protocol, when does a quantum model perform differently from strong classical baselines, and what computational resources are required?

The benchmark treats negative and inconclusive results as useful evidence. It is designed to make data leakage, weak classical controls, hidden preprocessing, selective reporting, uncontrolled tuning, irreproducible experiments, and unsupported quantum-advantage claims harder to introduce.

## Current project status

| Milestone | Status | Current capability |
|---|---|---|
| Reproducible benchmark foundation | Complete | Classical and quantum-kernel runners, fingerprints, configuration capture, provenance, reports, dashboard, and CI |
| Leakage and calibration hardening | Complete | Training-only preprocessing, selected-feature provenance, partition hashes, and training-only probability calibration |
| Statistical evaluation foundation | Complete | Bootstrap intervals, exact paired McNemar comparisons, pairwise artifacts, and bounded evidence statements |
| Reproducibility contract | Complete | Scientific equality checks excluding timestamps, paths, timing, and machine-precision noise |
| Classical nested cross-validation | Complete | Locked balanced-accuracy endpoint, bounded inner search, untouched outer folds, predictions, and search artifacts |
| Full classical reference execution | Complete | Five outer folds, three inner folds, eight selected features, and all four classical comparators executed and reviewed |
| Classical grid-boundary sensitivity | Implemented | Versioned `sensitivity-v1` profile expands only the identified SVM and boosting boundaries |
| Out-of-fold calibration diagnostics | Implemented | Reliability bins, ECE, MCE, calibration-in-the-large, Brier score, log loss, probability distributions, and hashed error rows |
| Full sensitivity execution | Next | Run and compare `sensitivity-v1` against the immutable `reference-v1` result |
| Quantum nested cross-validation | Planned | Design a resource-bounded protocol after the classical sensitivity conclusions are settled |
| External oncology cohorts | Planned | Add documented cohorts, grouped splitting, external validation, and independent replication |

The current benchmark supports **exploratory and methodological evidence**. It does not support clinical utility, external validity, statistical superiority across independent cohorts, or quantum-advantage claims.

## Project at a glance

<table>
<tr>
<td width="50%" valign="top">

### Classical evaluation

- Logistic regression
- RBF support-vector machine
- Random forest
- Histogram gradient boosting
- Repeated stratified holdouts
- Classical nested cross-validation
- Locked balanced-accuracy endpoint
- Versioned bounded search profiles

</td>
<td width="50%" valign="top">

### Quantum baseline

- Qiskit fidelity statevector kernel
- Precomputed-kernel SVM
- Exact or shot-sampled simulation
- Training-only score calibration
- Circuit resource accounting
- Explicit physical-QPU status
- No automatic advantage claim
- Quantum nested CV deferred

</td>
</tr>
<tr>
<td width="50%" valign="top">

### Statistical and calibration evidence

- BCa bootstrap confidence intervals
- Exact paired McNemar tests
- Shared-partition comparison hashes
- Descriptive direction counts
- Calibration-in-the-large
- Uniform-bin ECE and MCE
- Pooled out-of-fold Brier score and log loss
- False-positive and false-negative artifacts

</td>
<td width="50%" valign="top">

### Data and provenance

- Built-in public breast-cancer dataset
- Binary numeric CSV adapter
- Public GDC metadata manifests
- Dataset and partition fingerprints
- Selected-feature provenance
- Fold-level prediction artifacts
- Hashed sample-level error records
- Environment and commit metadata

</td>
</tr>
</table>

## Why this project exists

Quantum machine learning in oncology is promising, but early-stage results are easy to overstate. Small datasets, leakage, weak classical controls, simulator-only experiments, unrestricted optimization, and selective reporting can make an experimental model appear more useful than it is.

This repository establishes a defensible baseline by requiring:

1. Explicit positive classes and predeclared evaluation endpoints.
2. Identical evaluation partitions for directly compared models.
3. Preprocessing, feature selection, and calibration fitted only on training data.
4. Strong classical comparators with documented tuning budgets.
5. Repeated or nested evaluation with prediction-level evidence.
6. Transparent uncertainty, paired disagreement, calibration, and error metrics.
7. Machine-readable provenance and deterministic configuration.
8. Explicit interpretation boundaries and refusal to infer advantage from one score.

A higher score on one split, one simulator run, or one small dataset is not quantum advantage.

## Evaluation modes

### Repeated-holdout benchmark

```bash
qob benchmark --config configs/classical.yaml
```

Use this mode for software verification, exploratory comparisons, quantum-kernel experiments, and sensitivity analysis across deterministic holdout seeds.

It supports classical-only, quantum-only, or combined model sets with shared partitions, training-only preprocessing and calibration, repeat-level intervals, exact paired comparisons within each partition, and bounded evidence statements.

Repeated holdouts may reuse observations across test partitions. The benchmark therefore does not pool repeat-level p-values.

### Classical nested cross-validation

Reference profile:

```bash
qob nested-cv --config configs/nested-classical.yaml
```

Grid-boundary sensitivity profile:

```bash
qob nested-cv --config configs/nested-classical-sensitivity.yaml
```

Both profiles use:

- five stratified outer folds,
- three stratified inner folds,
- balanced accuracy as the locked primary endpoint,
- eight selected features,
- four classical model families,
- deterministic seeds and single-worker searches,
- training-only preprocessing, feature selection, and SVM calibration,
- one final evaluation of each untouched outer test fold.

The two profiles differ only where the completed reference run selected a grid boundary:

| Model | `reference-v1` | `sensitivity-v1` |
|---|---|---|
| RBF SVM `C` | `0.1, 1, 10` | `1, 10, 100` |
| Histogram boosting `max_leaf_nodes` | `15, 31` | `7, 15, 31` |

Logistic regression, random forest, SVM gamma, folds, seeds, feature count, endpoint, preprocessing, calibration, and artifact calculations remain unchanged.

For the RBF SVM, class predictions come from the selected pipeline. Probability scores come from a separately cross-validated clone fitted only on the outer training partition. Calibration does not replace predictions used for balanced accuracy or McNemar comparisons.

See:

- [Classical Nested Cross-Validation Protocol](docs/NESTED_CROSS_VALIDATION.md)
- [Out-of-Fold Calibration Diagnostics](docs/CALIBRATION_DIAGNOSTICS.md)
- [Statistical Evaluation](docs/STATISTICAL_EVALUATION.md)

## Current capabilities

- Wisconsin Diagnostic Breast Cancer public demonstration dataset.
- Malignant disease explicitly mapped to positive class `1`.
- Binary numeric CSV adapter for externally prepared cohorts and features.
- Public GDC metadata manifest generator with reproducible query receipts.
- Four classical baselines and one optional Qiskit fidelity-kernel baseline.
- Repeated stratified holdouts and classical nested cross-validation.
- Versioned `reference-v1` and `sensitivity-v1` nested search profiles.
- Per-split or per-fold features, hashes, class counts, seeds, parameters, and predictions.
- Balanced accuracy, sensitivity, specificity, precision, F1, ROC AUC, average precision, Brier score, and log loss.
- Repeat-level or outer-fold descriptive bootstrap intervals.
- Exact McNemar comparisons within shared test partitions.
- Pooled out-of-fold calibration and classification-error diagnostics.
- Deterministic reproducibility comparison with operational fields removed.
- JSON, CSV, Markdown, and Streamlit reporting.
- CI across Python 3.10 through 3.13, strict mypy, nested-CV CLI smoke testing, and Qiskit smoke testing.

## Architecture

```text
Public demo dataset, validated CSV, or GDC-derived cohort metadata
                              │
                              ▼
                  Cohort validation and fingerprinting
                              │
               ┌──────────────┴──────────────┐
               │                             │
               ▼                             ▼
     Repeated-holdout mode          Classical nested-CV mode
               │                             │
    Shared deterministic split       Untouched outer test fold
               │                             │
    Training-only preprocessing      Inner search on outer train
               │                             │
       ┌───────┴────────┐            Selected pipeline + profile
       ▼                ▼                     │
 Classical models   Quantum kernel            ▼
       │                │             One outer-fold evaluation
       └───────┬────────┘                     │
               └──────────────┬───────────────┘
                              ▼
     Metrics, intervals, paired tests, calibration, and provenance
                              │
                              ▼
       JSON, CSV, Markdown, prediction, error, and dashboard artifacts
```

Detailed design and governance documentation:

- [Architecture](docs/ARCHITECTURE.md)
- [Research Protocol](RESEARCH_PROTOCOL.md)
- [Statistical Evaluation](docs/STATISTICAL_EVALUATION.md)
- [Nested Cross-Validation](docs/NESTED_CROSS_VALIDATION.md)
- [Calibration Diagnostics](docs/CALIBRATION_DIAGNOSTICS.md)
- [Model Card](MODEL_CARD.md)
- [Data Sources](docs/DATA_SOURCES.md)
- [Data Governance](DATA_GOVERNANCE.md)

## Quick start

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux, macOS, or WSL:

```bash
source .venv/bin/activate
```

### 2. Install the core benchmark

```bash
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

### 3. Run the repeated-holdout classical benchmark

```bash
qob benchmark --config configs/classical.yaml
```

### 4. Run the classical nested-CV reference profile

```bash
qob nested-cv --config configs/nested-classical.yaml
```

### 5. Run the classical sensitivity profile

```bash
qob nested-cv --config configs/nested-classical-sensitivity.yaml
```

### 6. Install and run the quantum benchmark

```bash
pip install -e ".[dev,quantum]"
qob doctor
qob benchmark --config configs/baseline.yaml
```

The default quantum model uses classical simulation. Reports explicitly record whether a physical QPU was used.

### 7. Launch the dashboard

```bash
pip install -e ".[dashboard]"
streamlit run app.py
```

## Common workflows

### Run a reduced nested-CV smoke experiment

```bash
qob nested-cv --config configs/nested-smoke.yaml
```

The smoke configuration uses two outer folds, two inner folds, logistic regression, RBF SVM, five calibration bins, and a bounded sample. It validates the command and artifact contract; it is not substantive scientific evidence.

### Select a nested-CV profile and model subset

```bash
qob nested-cv \
  --search-profile sensitivity-v1 \
  --calibration-bins 10 \
  --model logistic_regression \
  --model rbf_svm \
  --outer-folds 5 \
  --inner-folds 3 \
  --features 8 \
  --output reports/nested-logistic-svm-sensitivity
```

### Compare classical and quantum models on a bounded sample

```bash
qob benchmark \
  --model-set all \
  --features 4 \
  --max-samples 160 \
  --repeats 3 \
  --output reports/all-models
```

### Validate a prepared CSV

```bash
qob validate-csv cohort.csv \
  --target-column diagnosis \
  --positive-label malignant
```

### Create a public GDC manifest

```bash
qob gdc-manifest \
  --project TCGA-LUAD \
  --project TCGA-LUSC \
  --data-type "Gene Expression Quantification" \
  --workflow-type "STAR - Counts" \
  --sample-type "Primary Tumor" \
  --output manifests/tcga-lung-star-counts.csv
```

This creates metadata and a `.query.json` receipt. It does not download genomic files.

## Output contracts

### Repeated-holdout schema `1.3`

| Artifact | Purpose |
|---|---|
| `experiment.json` | Configuration, dataset fingerprint, environment, split provenance, statistical analysis, and quantum-resource record |
| `summary.csv` | Aggregate metrics and confidence intervals across repeated splits |
| `runs.csv` | Per-model, per-repeat metrics and confusion counts |
| `pairwise_comparisons.csv` | Exact paired disagreement counts, partition hashes, direction, and significance by repeat |
| `REPORT.md` | Human-readable method, results, evidence statement, and limitations |

### Nested-CV schema `nested-cv-1.1`

| Artifact | Purpose |
|---|---|
| `nested_experiment.json` | Complete configuration, search profile, methodology, provenance, results, diagnostics, and environment |
| `outer_fold_results.csv` | Profile, parameters, features, hashes, probability source, and final metrics by model and fold |
| `outer_fold_predictions.csv` | Profile, hashed sample identifier, truth, prediction, and positive-class probability |
| `inner_search_results.csv` | Profile, every candidate, inner score, rank, and selected flag |
| `nested_summary.csv` | Aggregate outer-fold metrics, intervals, and pooled calibration fields |
| `nested_pairwise_comparisons.csv` | Exact paired disagreement results within each shared outer test fold |
| `calibration_summary.csv` | Calibration-in-the-large, ECE, MCE, pooled Brier score and log loss, and error counts |
| `calibration_bins.csv` | Occupied reliability-bin coordinates and supporting sample counts |
| `probability_distribution.csv` | Every configured probability bin by model and true class |
| `classification_errors.csv` | Hashed out-of-fold false-positive and false-negative rows with probability context |
| `NESTED_CV_REPORT.md` | Human-readable protocol, results, calibration summary, evidence statement, and limitations |

All experiment payloads retain:

```json
{
  "quantum_advantage_claimed": false
}
```

## Reproducibility contract

The benchmark retains fingerprints, configurations, search profiles, seeds, selected features and parameters, partition hashes, metrics, intervals, paired comparisons, prediction evidence, calibration diagnostics, hashed error rows, and environment provenance.

Scientific replay normalization removes only fields expected to vary operationally:

- generation timestamps,
- output directories,
- artifact paths,
- measured execution times.

Finite floating-point values are canonicalized to 12 decimal places during comparison only. Stored JSON and CSV precision is unchanged.

## Data strategy and boundaries

The built-in dataset is for software verification and methodological demonstration, not biomedical conclusions.

Externally prepared cohorts should contain:

- one row per independent subject or specimen,
- numeric model features only,
- a binary target with a declared positive class,
- no direct identifiers or free text,
- a cohort-definition record and data dictionary,
- documented authorization and an appropriate ethical or legal basis.

The CSV adapter validates shape and label encoding. It does not certify de-identification, validity, representativeness, row independence, or lawful use.

Subject-grouped and site-grouped splitting are not yet implemented. Repeated subjects or site effects require an external preparation and validation process until those controls are added.

The project does not yet:

- execute through a production physical-QPU adapter,
- perform quantum nested cross-validation,
- download controlled-access patient or genomic data,
- enforce subject-grouped or site-grouped splitting,
- validate a clinical biomarker,
- provide prospective or external-cohort validation,
- establish quantum or clinical advantage,
- recommend treatment, diagnosis, or patient-specific action.

## Scientific interpretation

Nested cross-validation improves internal model-selection discipline. Out-of-fold calibration diagnostics improve probability-quality visibility. Neither replaces external validation or independent replication.

A quantum result should be considered credible only when classical and quantum models use identical evaluation partitions, preprocessing and calibration remain training-only, tuning budgets are documented and matched, prediction evidence and uncertainty are retained, resources are disclosed, external data confirm the finding, and an independent researcher can reproduce it.

Until stronger evidence exists, results should be described as exploratory benchmark evidence.

## Development

```bash
pip install -e ".[dev,quantum]"
ruff check .
mypy src
pytest
```

GitHub Actions runs:

- Ruff across Python 3.10 through 3.13.
- Strict mypy against the Python 3.10 target.
- Core tests across Python 3.10 through 3.13.
- A real nested-CV command and artifact smoke run on Python 3.12.
- A Qiskit test and quantum benchmark smoke run on Python 3.12.

## Roadmap from the current baseline

| Priority | Objective | Current state |
|---|---|---|
| 1 | Run and review the full `sensitivity-v1` profile | Next evidence milestone |
| 2 | Generate calibration and reliability figures from the diagnostic CSV contract | Planned presentation layer |
| 3 | Add subject-grouped and site-grouped split controls | Planned methodology control |
| 4 | Add documented TCGA cohort recipes and data-quality artifacts | Planned cohort milestone |
| 5 | Design a matched, resource-bounded quantum nested-CV protocol | Planned after classical sensitivity review |
| 6 | Add external-cohort and independent replication workflows | Required before translational claims |
| 7 | Add physical-QPU adapters with backend snapshots and cost records | Future hardware milestone |
| 8 | Expand into drug-response, multi-omics, radiomics, and molecular workloads | Longer-term research scope |

See [ROADMAP.md](ROADMAP.md) for the broader version plan.

## Governance and safety

Read before proposing substantive changes:

- [Contributing](CONTRIBUTING.md)
- [Governance](GOVERNANCE.md)
- [Security](SECURITY.md)
- [Data Governance](DATA_GOVERNANCE.md)
- [Research Protocol](RESEARCH_PROTOCOL.md)

Do not submit protected health information, controlled-access data, credentials, signed URLs, patient screenshots, or identifying clinical content in code, issues, pull requests, discussions, logs, or test artifacts.

## Citation

When citing results, record the exact repository commit, evaluation mode, search profile, configuration, dataset fingerprint, seeds, partition hashes, dependency environment, and simulator or hardware execution mode.

## Contributing

Contributions are welcome when they preserve the scientific and safety boundaries. Methodological changes should include protocol updates, tests, migration notes, and an evidence-based rationale.

## License

Licensed under the [Apache License 2.0](LICENSE).
