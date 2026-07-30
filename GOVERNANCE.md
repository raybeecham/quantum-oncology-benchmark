# Governance

## Project purpose

The project develops transparent, reproducible benchmarks for evaluating quantum methods in oncology-related research tasks.

## Maintainer responsibilities

Maintainers are responsible for:

- Protecting the research-use-only boundary.
- Rejecting unsupported clinical or quantum-advantage claims.
- Requiring tests and provenance for substantive changes.
- Preventing sensitive data from entering the repository.
- Documenting breaking methodological changes.
- Encouraging independent replication and negative-result reporting.

## Decision principles

1. Scientific defensibility over promotional value.
2. Strong classical controls before quantum interpretation.
3. Reproducibility over isolated peak performance.
4. Explicit uncertainty over confident speculation.
5. Public, non-sensitive examples by default.
6. Modular adapters rather than hidden vendor dependencies.

## Change categories

### Code implementation

Changes that preserve the protocol may be merged after tests and review.

### Methodological change

Changes to splitting, feature selection, metrics, model comparison, or advantage language require an update to the research protocol and a documented rationale.

### Data-source change

New sources require license/terms review, provenance documentation, and privacy assessment.

### Clinical or translational claim

Clinical claims are not accepted without appropriate domain, statistical, ethical, and regulatory evidence.

## Releases

- Patch: implementation fixes that do not change intended methodology.
- Minor: backward-compatible capabilities or documented methodological additions.
- Major: incompatible output schemas, protocol changes, or interpretation changes.

## Conflicts of interest

Contributors should disclose material vendor, funding, or institutional interests relevant to a benchmark or conclusion.
