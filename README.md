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
    src="docs/assets/cancer-cell-dark.webp"
    alt="Detailed dark-mode cancer cell anatomy infographic"
    width="100%"
  />
</p>

> [!IMPORTANT]
> **Research use only.** This project is not a medical device, diagnostic system, treatment recommendation, or clinical decision-support product. It does not claim that quantum computing improves cancer outcomes or that any benchmark result demonstrates quantum advantage.

## Overview

The **Quantum Oncology Benchmark** is an open, reproducible framework for comparing classical machine-learning models with quantum-kernel methods on cancer-classification tasks.

The project is designed around a narrow, falsifiable question:

> Under a shared and auditable evaluation protocol, when does a quantum model perform differently from strong classical baselines, and what computational resources are required?

The benchmark treats negative and inconclusive results as useful evidence. It is built to make leakage, weak baselines, hidden preprocessing, selective reporting, and unsupported quantum-advantage claims harder to introduce.

## Project at a glance

<table>
<tr>
<td width="50%" valign="top">

### Classical baselines

- Logistic regression
- RBF support-vector machine
- Random forest
- Histogram gradient boosting
- Shared train/test partitions
- Shared selected features

</td>
<td width="50%" valign="top">

### Quantum baseline

- Qiskit fidelity statevector kernel
- Precomputed-kernel SVM
- Exact or shot-sampled simulation
- Circuit resource accounting
- Explicit physical-QPU status
- No automatic advantage claim

</td>
</tr>
<tr>
<td width="50%" valign="top">

### Data and provenance

- Built-in public breast-cancer dataset
- Binary numeric CSV adapter
- Public GDC metadata manifests
- Dataset fingerprints
- Environment metadata
- Configuration and seed capture

</td>
<td width="50%" valign="top">

### Reporting and governance

- JSON, CSV, and Markdown artifacts
- Streamlit results dashboard
- Research protocol
- Model card
- Data-governance controls
- Security and contribution guidance

</td>
</tr>
</table>

## Why this project exists

Quantum machine learning in oncology is promising, but early-stage results are easy to overstate. Small datasets, data leakage, weak classical controls, simulator-only experiments, and selective reporting can make an experimental model appear more useful than it is.

This repository establishes a defensible baseline by requiring:

1. Identical evaluation partitions for classical and quantum models.
2. Preprocessing fitted only on training data.
3. Strong classical comparators.
4. Repeated evaluation across seeds or splits.
5. Transparent resource and execution metadata.
6. Machine-readable outputs and reproducible configuration.
7. Explicit limitations and bounded interpretation.

A higher score on one small split is not quantum advantage.

## Current capabilities

- Built-in public demonstration dataset: Wisconsin Diagnostic Breast Cancer.
- Malignant disease explicitly mapped to positive class `1`.
- CSV adapter for binary tabular cohorts, biomarkers, or externally prepared genomic features.
- Public GDC metadata manifest generator for reproducible TCGA cohort discovery.
- Train/test splitting before imputation, scaling, or feature selection.
- Shared selected features and shared splits across all model families.
- Repeated stratified holdouts.
- Oncology-oriented metrics:
  - sensitivity
  - specificity
  - precision
  - F1
  - ROC AUC
  - average precision
  - Brier score
  - log loss
- Dataset fingerprinting, random seeds, environment metadata, and circuit statistics.
- Streamlit dashboard for reviewing completed experiments.

## Architecture

```text
Public demo dataset, validated CSV, or GDC-derived cohort metadata
                              │
                              ▼
                  Cohort validation and fingerprinting
                              │
                              ▼
                  Stratified train/test split first
                              │
                              ▼
          Training-only imputation, scaling, and feature selection
                              │
                 ┌────────────┴────────────┐
                 ▼                         ▼
        Classical feature space      Quantum angle encoding
                 │                         │
       Classical model suite       Fidelity kernel + SVM
                 └────────────┬────────────┘
                              ▼
           Shared metrics, uncertainty, and resource accounting
                              │
                              ▼
             JSON, CSV, Markdown, and dashboard artifacts
```

Detailed design documentation:

- [Architecture](docs/ARCHITECTURE.md)
- [Research Protocol](RESEARCH_PROTOCOL.md)
- [Model Card](MODEL_CARD.md)
- [Data Sources](docs/DATA_SOURCES.md)
- [Data Governance](DATA_GOVERNANCE.md)

## Quick start

### 1. Create a virtual environment

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

### 3. Run the classical benchmark

```bash
qob benchmark --config configs/classical.yaml
```

Artifacts are written to `reports/classical/`.

### 4. Install and run the quantum benchmark

```bash
pip install -e ".[dev,quantum]"
qob doctor
qob benchmark --config configs/baseline.yaml
```

The default quantum model uses classical statevector simulation. Reports explicitly record that no physical QPU was used.

### 5. Launch the dashboard

```bash
pip install -e ".[dashboard]"
streamlit run app.py
```

## Common workflows

### Compare all models on a bounded sample

```bash
qob benchmark \
  --model-set all \
  --features 4 \
  --max-samples 160 \
  --repeats 3 \
  --output reports/all-models
```

### Run the exact statevector quantum kernel

```bash
qob benchmark --model-set quantum
```

Omit `--quantum-shots` for exact statevector fidelity. Set it to a positive integer to simulate finite-shot sampling.

### Validate a prepared CSV

```bash
qob validate-csv cohort.csv \
  --target-column diagnosis \
  --positive-label malignant
```

### Run a CSV experiment

```bash
qob benchmark \
  --dataset csv \
  --csv-path cohort.csv \
  --target-column diagnosis \
  --positive-label malignant \
  --model-set all \
  --features 4
```

### Create a public GDC manifest for TCGA lung projects

```bash
qob gdc-manifest \
  --project TCGA-LUAD \
  --project TCGA-LUSC \
  --data-type "Gene Expression Quantification" \
  --workflow-type "STAR - Counts" \
  --sample-type "Primary Tumor" \
  --output manifests/tcga-lung-star-counts.csv
```

This command creates metadata and a `.query.json` receipt. It does not download genomic files.

## Output contract

Each benchmark run produces:

| Artifact | Purpose |
|---|---|
| `experiment.json` | Full configuration, provenance, dataset fingerprint, results, and quantum-resource record |
| `summary.csv` | Aggregate model metrics across repeated splits |
| `runs.csv` | Per-model, per-repeat metrics |
| `REPORT.md` | Human-readable research summary, execution context, and limitations |

The JSON payload includes the field:

```json
{
  "quantum_advantage_claimed": false
}
```

This is intentional. The framework records comparative evidence; it does not convert a small benchmark difference into an advantage claim.

## Data strategy

The built-in dataset is intended for software verification and demonstration, not biomedical conclusions.

For more serious work, prepare a documented cohort outside the public repository and provide a binary tabular CSV containing:

- One row per independent subject or specimen.
- Numeric model features only.
- A binary target with a declared positive class.
- No names, medical record numbers, dates of birth, free text, or direct identifiers.
- A cohort-definition record and data dictionary.
- Documented authorization and an appropriate ethical or legal basis for use.

The CSV adapter validates shape and label encoding. It does not certify de-identification, scientific validity, representativeness, or lawful use.

## What the project does not do yet

- Execute on a physical quantum processor.
- Download controlled-access patient or genomic data.
- Validate a clinical biomarker.
- Perform prospective clinical evaluation.
- Provide external-cohort validation.
- Establish statistical superiority or quantum advantage.
- Recommend treatment, diagnosis, or patient-specific action.

These boundaries are deliberate.

## Scientific interpretation

A quantum result should be considered credible only when:

1. Quantum and classical models use identical evaluation partitions.
2. Preprocessing is fitted only on training data.
3. Classical baselines receive reasonable tuning.
4. Results are repeated across seeds or folds.
5. Confidence intervals and uncertainty are reported.
6. Circuit resources and execution mode are disclosed.
7. External or independently held-out data confirm the finding.
8. Another researcher can reproduce the result from the repository.

Until those conditions are met, results should be described as exploratory benchmark evidence.

## Development

```bash
pip install -e ".[dev,quantum]"
ruff check .
mypy src
pytest
```

The GitHub Actions workflow runs:

- Ruff across Python 3.10, 3.11, 3.12, and 3.13.
- Strict mypy against the configured Python 3.10 target.
- Core tests across Python 3.10 through 3.13.
- A dedicated Qiskit quantum smoke test on Python 3.12.

## Roadmap

| Phase | Objective |
|---|---|
| 1 | Reproducible tabular benchmark foundation |
| 2 | Public GDC manifest and cohort-building adapters |
| 3 | Nested cross-validation and bootstrap confidence intervals |
| 4 | Physical-QPU execution adapters with backend snapshots |
| 5 | External-cohort and independent replication workflows |
| 6 | Drug-response and molecular-chemistry benchmark modules |

See [ROADMAP.md](ROADMAP.md) for the detailed plan.

## Governance and safety

Read these documents before proposing substantive changes:

- [Contributing](CONTRIBUTING.md)
- [Governance](GOVERNANCE.md)
- [Security](SECURITY.md)
- [Data Governance](DATA_GOVERNANCE.md)
- [Research Protocol](RESEARCH_PROTOCOL.md)

Do not submit protected health information, controlled-access data, credentials, signed URLs, patient screenshots, or identifying clinical content in code, issues, pull requests, discussions, logs, or test artifacts.

## Citation

The repository includes a [`CITATION.cff`](CITATION.cff) file. When citing results, record the exact:

- repository commit
- benchmark configuration
- dataset or manifest fingerprint
- random seeds
- dependency environment
- simulator or hardware execution mode

## Contributing

Contributions are welcome when they preserve the project’s scientific and safety boundaries. Methodological changes should include protocol updates, tests, migration notes, and a clear evidence-based rationale.

## License

Licensed under the [Apache License 2.0](LICENSE).
