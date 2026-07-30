# Data Sources

## Built-in demonstration dataset

The repository uses scikit-learn's packaged Wisconsin Diagnostic Breast Cancer dataset for deterministic examples and CI.

Characteristics:

- 569 samples.
- 30 numeric features.
- Binary malignant/benign target.
- Suitable for software demonstration.
- Not sufficient for a contemporary clinical or genomic claim.

## CSV cohorts

The CSV adapter supports prepared binary tabular datasets. It is deliberately generic so the benchmark is not coupled to one institution or assay.

Required properties:

- One target column.
- Exactly two target classes.
- At least one numeric feature.
- A declared positive label unless already encoded as `0` and `1`.

## GDC and TCGA plan

The National Cancer Institute Genomic Data Commons provides project, case, sample, file, and annotation search endpoints. The planned adapter will create reproducible public manifests rather than silently downloading arbitrary data.

Initial target recipe:

- TCGA-LUAD versus TCGA-LUSC.
- Public clinical and biospecimen metadata.
- Explicit assay and workflow filtering.
- One independent record per subject or a grouped-split strategy.
- Compact, predeclared feature sets suitable for small-qubit experiments.

Controlled-access data will not be included in this public repository.

## Synthetic data

Synthetic data may be used for pipeline testing but must be labeled as synthetic. It must not be used to imply biological validity.
