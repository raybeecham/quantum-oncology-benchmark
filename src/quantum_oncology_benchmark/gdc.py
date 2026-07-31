"""Minimal public-metadata client for the NCI Genomic Data Commons API."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

_GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
_PROJECT_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
_DEFAULT_FIELDS = (
    "file_id",
    "file_name",
    "md5sum",
    "file_size",
    "access",
    "data_category",
    "data_type",
    "data_format",
    "analysis.workflow_type",
    "cases.case_id",
    "cases.submitter_id",
    "cases.project.project_id",
    "cases.samples.sample_id",
    "cases.samples.submitter_id",
    "cases.samples.sample_type",
    "cases.samples.tissue_type",
)


class GDCClientError(RuntimeError):
    """Raised when a GDC query fails or returns an unexpected payload."""


@dataclass(frozen=True, slots=True)
class GDCManifestQuery:
    """Declarative file-manifest query for public GDC metadata."""

    projects: tuple[str, ...]
    data_type: str = "Gene Expression Quantification"
    workflow_type: str = "STAR - Counts"
    access: str = "open"
    sample_types: tuple[str, ...] = ("Primary Tumor",)
    size: int = 10_000

    def validate(self) -> None:
        """Validate query fields before creating a remote request."""
        if not self.projects:
            raise ValueError("at least one GDC project is required")
        invalid = [project for project in self.projects if not _PROJECT_PATTERN.fullmatch(project)]
        if invalid:
            raise ValueError(f"invalid GDC project identifiers: {invalid}")
        if self.access not in {"open", "controlled"}:
            raise ValueError("access must be 'open' or 'controlled'")
        if not 1 <= self.size <= 10_000:
            raise ValueError("size must be between 1 and 10000")
        if not self.data_type.strip() or not self.workflow_type.strip():
            raise ValueError("data_type and workflow_type must not be empty")

    def payload(self) -> dict[str, Any]:
        """Build a GDC files-endpoint POST payload."""
        self.validate()
        filters: list[dict[str, Any]] = [
            {
                "op": "in",
                "content": {
                    "field": "cases.project.project_id",
                    "value": list(self.projects),
                },
            },
            {
                "op": "=",
                "content": {"field": "files.data_type", "value": self.data_type},
            },
            {
                "op": "=",
                "content": {
                    "field": "files.analysis.workflow_type",
                    "value": self.workflow_type,
                },
            },
            {
                "op": "=",
                "content": {"field": "files.access", "value": self.access},
            },
        ]
        if self.sample_types:
            filters.append(
                {
                    "op": "in",
                    "content": {
                        "field": "cases.samples.sample_type",
                        "value": list(self.sample_types),
                    },
                }
            )
        return {
            "filters": {"op": "and", "content": filters},
            "format": "JSON",
            "fields": ",".join(_DEFAULT_FIELDS),
            "size": str(self.size),
        }


def fetch_manifest_metadata(
    query: GDCManifestQuery,
    *,
    timeout_seconds: float = 60.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Query GDC file metadata and return a flattened manifest plus receipt."""
    payload = query.payload()
    request = Request(
        _GDC_FILES_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "quantum-oncology-benchmark/0.1.0",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise GDCClientError(f"GDC returned HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise GDCClientError(f"unable to reach the GDC API: {exc.reason}") from exc

    try:
        decoded = json.loads(raw.decode("utf-8"))
        hits = decoded["data"]["hits"]
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise GDCClientError("GDC returned an unexpected response format") from exc
    if not isinstance(hits, list):
        raise GDCClientError("GDC response data.hits was not a list")

    rows: list[dict[str, Any]] = []
    for hit in hits:
        if not isinstance(hit, dict):
            continue
        cases = hit.get("cases") or [None]
        for case in cases:
            case_mapping = case if isinstance(case, dict) else {}
            samples = case_mapping.get("samples") or [None]
            project = case_mapping.get("project")
            project_mapping = project if isinstance(project, dict) else {}
            for sample in samples:
                sample_mapping = sample if isinstance(sample, dict) else {}
                analysis = hit.get("analysis")
                analysis_mapping = analysis if isinstance(analysis, dict) else {}
                rows.append(
                    {
                        "file_id": hit.get("file_id"),
                        "file_name": hit.get("file_name"),
                        "md5sum": hit.get("md5sum"),
                        "file_size": hit.get("file_size"),
                        "access": hit.get("access"),
                        "data_category": hit.get("data_category"),
                        "data_type": hit.get("data_type"),
                        "data_format": hit.get("data_format"),
                        "workflow_type": analysis_mapping.get("workflow_type"),
                        "project_id": project_mapping.get("project_id"),
                        "case_id": case_mapping.get("case_id"),
                        "case_submitter_id": case_mapping.get("submitter_id"),
                        "sample_id": sample_mapping.get("sample_id"),
                        "sample_submitter_id": sample_mapping.get("submitter_id"),
                        "sample_type": sample_mapping.get("sample_type"),
                        "tissue_type": sample_mapping.get("tissue_type"),
                    }
                )

    manifest = pd.DataFrame(rows)
    if not manifest.empty:
        manifest = manifest.drop_duplicates().sort_values(
            by=["project_id", "case_submitter_id", "sample_submitter_id", "file_name"],
            na_position="last",
        )
        manifest = manifest.reset_index(drop=True)

    receipt = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "endpoint": _GDC_FILES_ENDPOINT,
        "query": asdict(query),
        "request_payload": payload,
        "response_pagination": decoded.get("data", {}).get("pagination", {}),
        "rows_written": len(manifest),
        "downloads_performed": False,
        "controlled_data_downloaded": False,
    }
    return manifest, receipt


def write_manifest_artifacts(
    manifest: pd.DataFrame,
    receipt: dict[str, Any],
    output_path: str | Path,
) -> tuple[Path, Path]:
    """Write a CSV manifest and adjacent JSON query receipt."""
    manifest_path = Path(output_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    receipt_path = manifest_path.with_suffix(".query.json")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path, receipt_path
