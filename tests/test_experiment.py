from __future__ import annotations

import json

from quantum_oncology_benchmark.config import ExperimentConfig
from quantum_oncology_benchmark.experiment import run_benchmark


def test_classical_benchmark_writes_complete_artifacts(tmp_path) -> None:
    output = tmp_path / "report"
    config = ExperimentConfig(
        model_set="classical",
        features=4,
        repeats=1,
        max_samples=120,
        output_dir=str(output),
    )

    payload = run_benchmark(config)

    assert payload["quantum_advantage_claimed"] is False
    assert len(payload["summary"]) == 4
    assert len(payload["selected_features"]) == 4
    assert (output / "experiment.json").exists()
    assert (output / "summary.csv").exists()
    assert (output / "runs.csv").exists()
    assert (output / "REPORT.md").exists()

    saved = json.loads((output / "experiment.json").read_text(encoding="utf-8"))
    assert saved["dataset"]["positive_class"] == "malignant"
    assert saved["research_use_only"] is True
