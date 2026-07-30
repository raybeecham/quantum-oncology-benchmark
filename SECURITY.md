# Security Policy

## Supported versions

The latest minor release is supported during the pre-1.0 development period.

## Reporting a vulnerability

Do not open a public issue for vulnerabilities involving credentials, code execution, dependency compromise, private data exposure, or sensitive-data leakage.

Contact the maintainer privately through the security-reporting mechanism configured for the GitHub repository.

## Sensitive data

Do not include patient information, controlled genomic data, access tokens, API keys, cloud credentials, signed URLs, or private logs in a vulnerability report.

## Scope

Security concerns may include:

- Unsafe deserialization.
- Path traversal or arbitrary file writes.
- Dependency or workflow compromise.
- Credential leakage.
- Dashboard exposure.
- Accidental publication of sensitive data.
- Untrusted CSV or configuration handling.

The project is research software and should run in an isolated environment when processing untrusted files.
