# GDC Integration Plan

## Current capability

Version 0.1.0 can query the public NCI Genomic Data Commons `files` endpoint and produce a local CSV manifest plus a JSON query receipt.

The default lung-cancer query targets:

- Projects: `TCGA-LUAD` and `TCGA-LUSC`.
- Data type: `Gene Expression Quantification`.
- Workflow: `STAR - Counts`.
- Sample type: `Primary Tumor`.
- Access: `open`.

The manifest includes file, project, case, and sample identifiers. The command does not download data files.

## Why manifest-first

A manifest-first workflow separates cohort definition from data acquisition. It creates an auditable record of:

- Which projects were queried.
- Which workflow and data type were selected.
- Whether access was open or controlled.
- Which sample types were included.
- What the GDC returned at a point in time.

This is preferable to embedding an opaque download script in a model-training pipeline.

## Planned cohort pipeline

1. Generate and review a GDC manifest.
2. Lock the manifest and query receipt.
3. Check case and sample uniqueness.
4. Define one record per subject or implement subject-grouped splitting.
5. Download only authorized files in an approved environment.
6. Build a documented expression matrix.
7. Apply gene filtering and transformation inside the research protocol.
8. Create a non-sensitive benchmark artifact or local CSV.
9. Run matched classical and quantum experiments.
10. Retain the cohort recipe without committing protected data.

## Proposed LUAD versus LUSC controls

- Primary tumor samples only for the initial subtype task.
- One sample per case unless grouping is implemented.
- Explicit handling of duplicate aliquots.
- Predeclared normalization and gene-filtering method.
- Feature selection inside each training fold.
- No test-set-informed gene selection.
- External validation before biological interpretation.

## Controlled-access boundary

The client can describe controlled-access files when explicitly requested, but it does not authenticate, request tokens, or download those files. Controlled-access workflows require separate authorization, data-use compliance, and secure infrastructure.
