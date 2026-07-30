# Data Governance

## Core rule

Do not commit, upload, paste, log, screenshot, or discuss protected health information or directly identifying patient information in this repository.

## Permitted by default

- Public educational datasets with documented licenses or terms.
- Synthetic datasets that do not reproduce real individuals.
- Aggregated metrics that cannot reasonably identify a person.
- Public metadata retrieved under applicable source terms.

## Requires project-specific approval

- Controlled-access genomic data.
- Limited datasets.
- Coded or pseudonymized patient data.
- Institutionally held clinical data.
- Data governed by a data-use agreement, IRB protocol, contract, or sponsor restriction.

## Prohibited in the public repository

- Names, initials, email addresses, phone numbers, or street addresses.
- Medical record, insurance, account, device, or certificate identifiers.
- Full dates tied to individuals.
- Facial images or other identifying biometrics.
- Free-text notes that may contain identifiers.
- Raw controlled-access files.
- Access tokens, credentials, signed URLs, or cloud storage secrets.

## CSV adapter responsibility

The CSV adapter validates shape and target encoding. It does not certify that a dataset is de-identified, lawfully obtained, unbiased, representative, or appropriate for research.

Before using a CSV, document:

- Data owner and source.
- Authorization and applicable terms.
- Cohort definition.
- Unit of analysis.
- Label definition.
- Exclusions.
- Missing-data policy.
- Subject-level grouping requirements.
- Retention and deletion requirements.

## GDC and TCGA direction

Future integration will use public GDC endpoints to create manifests and retrieve public metadata. Controlled-access data will remain outside the repository and require the user's authorized environment and applicable approvals.

## Incident response

If sensitive data is committed:

1. Stop sharing or cloning the repository.
2. Revoke exposed credentials immediately.
3. Notify the data owner and security/privacy contacts.
4. Remove the data from Git history using an approved history-rewrite process.
5. Treat forks, caches, CI artifacts, and logs as potentially affected.
6. Document the incident and corrective actions.
