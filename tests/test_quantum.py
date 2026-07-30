from __future__ import annotations

import pytest

from quantum_oncology_benchmark.config import ExperimentConfig
from quantum_oncology_benchmark.experiment import run_benchmark
from quantum_oncology_benchmark.models.quantum_kernel import quantum_dependencies_available


@pytest.mark.quantum
@pytest.mark.skipif(not quantum_dependencies_available(), reason="quantum extras not installed")
def test_quantum_kernel_smoke_run(tmp_path) -> None:
    config = ExperimentConfig(
        model_set="quantum",
        features=2,
        repeats=1,
        max_samples=40,
        quantum_reps=1,
        output_dir=str(tmp_path / "quantum"),
    )
    payload = run_benchmark(config)
    assert len(payload["summary"]) == 1
    assert payload["summary"][0]["model"] == "quantum_fidelity_svm"
    assert payload["quantum_resources"]["physical_qpu"] is False
    assert payload["quantum_resources"]["qubits"] == 2
