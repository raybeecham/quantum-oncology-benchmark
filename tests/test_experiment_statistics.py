from __future__ import annotations

from quantum_oncology_benchmark.config import ExperimentConfig
from quantum_oncology_benchmark.experiment import run_benchmark


def test_benchmark_emits_statistical_artifacts(tmp_path) -> None:
    output = tmp_path / "statistical"
    config = ExperimentConfig(
        model_set="classical",
        features=4,
        repeats=2,
        max_samples=120,
        output_dir=str(output),
    )

    payload = run_benchmark(config)

    assert payload["schema_version"] == "1.2"
    assert payload["evidence_statement"]["status"] == "classical_only"
    assert payload["statistical_analysis"]["confidence_intervals"]["confidence_level"] == 0.95
    assert payload["statistical_analysis"]["paired_tests"][
        "pooled_repeated_holdout_p_value_reported"
    ] is False
    assert len(payload["statistical_analysis"]["pairwise_comparisons"]) == 12
    assert len(payload["statistical_analysis"]["pairwise_summary"]) == 6
    assert all(row["balanced_accuracy_ci_low"] is not None for row in payload["summary"])
    assert (output / "pairwise_comparisons.csv").exists()
    report = (output / "REPORT.md").read_text(encoding="utf-8")
    assert "## Statistical Evaluation" in report
    assert "## Evidence Statement" in report
