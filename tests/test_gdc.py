from __future__ import annotations

import json
from typing import Any

from quantum_oncology_benchmark.gdc import (
    GDCManifestQuery,
    fetch_manifest_metadata,
    write_manifest_artifacts,
)


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._payload


def test_gdc_query_builds_restricted_public_metadata_filter() -> None:
    query = GDCManifestQuery(projects=("TCGA-LUAD", "TCGA-LUSC"))
    payload = query.payload()
    filters = payload["filters"]["content"]

    assert payload["format"] == "JSON"
    assert any(item["content"]["field"] == "files.access" for item in filters)
    assert any(item["content"]["value"] == "STAR - Counts" for item in filters)
    assert "cases.project.project_id" in json.dumps(payload)


def test_gdc_response_is_flattened_and_receipted(monkeypatch, tmp_path) -> None:
    response = {
        "data": {
            "hits": [
                {
                    "file_id": "file-1",
                    "file_name": "sample.tsv",
                    "md5sum": "abc123",
                    "file_size": 1234,
                    "access": "open",
                    "data_category": "Transcriptome Profiling",
                    "data_type": "Gene Expression Quantification",
                    "data_format": "TSV",
                    "analysis": {"workflow_type": "STAR - Counts"},
                    "cases": [
                        {
                            "case_id": "case-1",
                            "submitter_id": "TCGA-00-0001",
                            "project": {"project_id": "TCGA-LUAD"},
                            "samples": [
                                {
                                    "sample_id": "sample-1",
                                    "submitter_id": "TCGA-00-0001-01A",
                                    "sample_type": "Primary Tumor",
                                    "tissue_type": "Tumor",
                                }
                            ],
                        }
                    ],
                }
            ],
            "pagination": {"count": 1, "total": 1},
        }
    }

    def fake_urlopen(*args: object, **kwargs: object) -> _FakeResponse:
        return _FakeResponse(response)

    monkeypatch.setattr("quantum_oncology_benchmark.gdc.urlopen", fake_urlopen)
    manifest, receipt = fetch_manifest_metadata(
        GDCManifestQuery(projects=("TCGA-LUAD",), size=100)
    )

    assert manifest.loc[0, "project_id"] == "TCGA-LUAD"
    assert manifest.loc[0, "sample_type"] == "Primary Tumor"
    assert receipt["downloads_performed"] is False
    assert receipt["rows_written"] == 1

    manifest_path, receipt_path = write_manifest_artifacts(
        manifest, receipt, tmp_path / "manifest.csv"
    )
    assert manifest_path.exists()
    assert receipt_path.exists()
