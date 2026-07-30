# Quantum Oncology Benchmark

[![CI](https://github.com/raybeecham/quantum-oncology-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/raybeecham/quantum-oncology-benchmark/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Research Use Only](https://img.shields.io/badge/use-research%20only-red.svg)](MODEL_CARD.md)

A reproducible research framework for comparing classical machine-learning models with quantum-kernel methods on cancer-classification tasks.

> **Research use only:** This project is not a medical device, diagnostic system, treatment recommendation, or clinical decision-support product. It does not claim that quantum computing improves cancer outcomes or that any benchmark result demonstrates quantum advantage.

## Why this project exists

Quantum machine learning in oncology is promising but easy to overstate. Small datasets, weak baselines, data leakage, simulator-only experiments, and selective reporting can make an experimental model appear more useful than it is.

This repository is designed to answer a narrower and more defensible question:

> Under a shared, auditable evaluation protocol, when does a quantum model perform differently from strong classical baselines, and what computational resources are required?

The framework treats a negative result as useful evidence.

## Current capabilities

- Built-in public demonstration dataset: Wisconsin Diagnostic Breast Cancer.
- CSV adapter for binary tabular cohorts, including externally prepared genomic or biomarker data.
- Public GDC metadata manifest generator for reproducible TCGA cohort discovery.
- Malignant disease is explicitly mapped to positive class `1` in the built-in benchmark.
- Train/test splitting occurs before imputation, scaling, or feature selection.
- Shared selected features and shared splits across classical and quantum models.
- Classical baselines:
  - Logistic regression
  - RBF support-vector machine
  - Random forest
  - Histogram gradient boosting
- Quantum baseline:
  - Qiskit fidelity statevector kernel
  - Precomputed-kernel support-vector machine
  - Exact or shot-sampled simulation
- Oncology-oriented metrics:
  - Sensitivity
  - Specificity
  - Precision
  - F1
  - ROC AUC
  - Average precision
  - Calibration losses
- Repeated stratified holdouts.
- Dataset fingerprinting, environment metadata, random seeds, circuit statistics, and machine-readable reports.
- Streamlit dashboard for completed experiments.

## What it does not do yet

- It does not execute on a physical quantum processor.
- It does not download controlled-access patient data.
- It does not validate a clinical biomarker.
- It does not perform prospective or external-cohort validation.
- It does not establish statistical superiority or quantum advantage.

Those boundaries are intentional.

## Architecture

```text
Public demo dataset or validated CSV
                 │
                 ▼
       Stratified sample selection
                 │
                 ▼
        Train/test split first
                 │
                 ▼
 Training-only imputation, scaling,
 and univariate feature selection
                 │
          ┌──────┴────────┐
          ▼               ▼
 Classical features   Quantum angle encoding
          │               │
 Classical models     Fidelity kernel + SVM
          └──────┬────────┘
                 ▼
 Shared metrics and resource accounting
                 │
                 ▼
 JSON, CSV, Markdown, and dashboard artifacts
```

See [Architecture](docs/ARCHITECTURE.md) and [Research Protocol](RESEARCH_PROTOCOL.md).

## Quick start

### 1. Create an environment

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

The quantum model uses a classical statevector simulation. The report explicitly records that no physical QPU was used.

### 5. Launch the dashboard

```bash
pip install -e ".[dashboard]"
streamlit run app.py
```

## Command examples

Run all models on a 160-sample stratified subset:

```bash
qob benchmark \
  --model-set all \
  --features 4 \
  --max-samples 160 \
  --repeats 3 \
  --output reports/all-models
```

Run an exact statevector kernel:

```bash
qob benchmark --model-set quantum
```

Omit `--quantum-shots` for exact statevector fidelity. Set it to a positive number to simulate finite-shot sampling.

Validate a prepared CSV:

```bash
qob validate-csv cohort.csv \
  --target-column diagnosis \
  --positive-label malignant
```

Run a CSV experiment:

```bash
qob benchmark \
  --dataset csv \
  --csv-path cohort.csv \
  --target-column diagnosis \
  --positive-label malignant \
  --model-set all \
  --features 4
```

Create a public GDC metadata manifest for TCGA lung cancer projects:

```bash
qob gdc-manifest \
  --project TCGA-LUAD \
  --project TCGA-LUSC \
  --data-type "Gene Expression Quantification" \
  --workflow-type "STAR - Counts" \
  --sample-type "Primary Tumor" \
  --output manifests/tcga-lung-star-counts.csv
```

This command writes file and sample metadata plus a `.query.json` receipt. It does not download genomic files.

## Output contract

Each run writes:

| Artifact | Purpose |
|---|---|
| `experiment.json` | Full configuration, provenance, dataset fingerprint, results, and quantum resource record |
| `summary.csv` | Aggregate model metrics across repeated splits |
| `runs.csv` | Per-model, per-repeat metrics |
| `REPORT.md` | Human-readable research summary and limitations |

The JSON payload includes `quantum_advantage_claimed: false` by design.

## Dataset strategy

The built-in dataset is deliberately small and convenient. It is useful for software verification, not for biomedical claims.

For more serious work, researchers should prepare a documented cohort outside this repository and provide a binary tabular CSV containing:

- One row per independent subject or specimen.
- Numeric model features only.
- A binary target with a declared positive class.
- No names, medical record numbers, dates of birth, free text, or other direct identifiers.
- A cohort-definition record and data dictionary.
- A lawful and ethical basis for use.

The planned GDC/TCGA integration will separate public metadata retrieval from controlled-access data workflows. See [Data Sources](docs/DATA_SOURCES.md) and [Data Governance](DATA_GOVERNANCE.md).

## Scientific interpretation

A quantum result should be considered credible only when all of the following are satisfied:

1. The quantum and classical models use identical evaluation partitions.
2. Preprocessing is fitted only on training data.
3. Classical baselines receive reasonable tuning.
4. Results are repeated across seeds or folds.
5. Confidence intervals and uncertainty are reported.
6. Circuit resources and execution mode are disclosed.
7. External or independently held-out data confirm the finding.
8. Another researcher can reproduce the result from the repository.

A higher score on one small split is not quantum advantage.

## Development

```bash
pip install -e ".[dev,quantum]"
ruff check .
mypy src
pytest
```

## Roadmap

- Phase 1: reproducible tabular benchmark foundation.
- Phase 2: public GDC manifest and cohort-building adapters.
- Phase 3: nested cross-validation and bootstrap confidence intervals.
- Phase 4: physical-QPU execution adapters with backend snapshots.
- Phase 5: external-cohort and independent replication workflows.
- Phase 6: drug-response and molecular-chemistry benchmark modules.

See [ROADMAP.md](ROADMAP.md).

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md), [GOVERNANCE.md](GOVERNANCE.md), and [SECURITY.md](SECURITY.md). Do not submit protected health information in code, datasets, issues, discussions, screenshots, or logs.

## License

Apache License 2.0. See [LICENSE](LICENSE).
