# Contributing

## Before opening a change

- Read `RESEARCH_PROTOCOL.md` and `DATA_GOVERNANCE.md`.
- Do not use real patient data in tests or examples.
- Open an issue before changing the benchmark methodology or interpretation language.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev,quantum]"
```

## Required checks

```bash
ruff check .
mypy src
pytest
```

## Pull request expectations

A pull request should state:

- What changed.
- Why the change is needed.
- Whether methodology or output schemas changed.
- What tests were run.
- Whether data, privacy, clinical, or vendor considerations apply.
- What limitations remain.

## Methodological changes

Changes to splitting, preprocessing, metrics, model selection, or advantage criteria must include:

- A protocol update.
- Tests preventing regression.
- A migration note for existing results.
- A rationale grounded in appropriate primary literature or official documentation.

## Commit hygiene

Do not commit generated virtual environments, credentials, protected data, large raw datasets, or local benchmark reports containing restricted information.
