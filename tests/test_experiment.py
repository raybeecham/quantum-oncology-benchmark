from __future__ import annotations

import json

from quantum_oncology_benchmark.config import ExperimentConfig
from quantum_oncology_benchmark.experiment import run_benchmark
from quantum_oncology_benchmark.reporting import normalize_experiment_for_reproducibility


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

    assert payload["schema_version"] == "1.3"
    assert payload["quantum_advantage_claimed"] is False
    assert len(payload["summary"]) == 4
    assert len(payload["selected_features"]) == 4
    assert len(payload["split_provenance"]) == 1
    assert (output / "experiment.json").exists()
    assert (output / "summary.csv").exists()
    assert (output / "runs.csv").exists()
    assert (output / "pairwise_comparisons.csv").exists()
    assert (output / "REPORT.md").exists()

    saved = json.loads((output / "experiment.json").read_text(encoding="utf-8"))
    assert saved["dataset"]["positive_class"] == "malignant"
    assert saved["research_use_only"] is True


def test_same_configuration_produces_same_scientific_results(tmp_path) -> None:
    config = ExperimentConfig(
        model_set="classical",
        features=4,
        repeats=1,
        max_samples=80,
        output_dir=str(tmp_path / "unused"),
    )

    first = run_benchmark(config, write_output=False)
    second = run_benchmark(config, write_output=False)

    assert normalize_experiment_for_reproducibility(first) == (
        normalize_experiment_for_reproducibility(second)
    )


def test_runtime_fields_are_excluded_from_reproducibility_comparison() -> None:
    first = {
        "generated_at": "2026-01-01T00:00:00+00:00",
        "artifacts": {"json": "one/experiment.json"},
        "config": {"seed": 42, "output_dir": "one"},
        "runs": [{"model": "example", "accuracy": 0.9, "elapsed_seconds": 1.0}],
        "summary": [
            {
                "model": "example",
                "accuracy_mean": 0.9,
                "elapsed_seconds_mean": 1.0,
                "elapsed_seconds_std": 0.1,
            }
        ],
    }
    second = {
        "generated_at": "2026-01-02T00:00:00+00:00",
        "artifacts": {"json": "two/experiment.json"},
        "config": {"seed": 42, "output_dir": "two"},
        "runs": [{"model": "example", "accuracy": 0.9, "elapsed_seconds": 9.0}],
        "summary": [
            {
                "model": "example",
                "accuracy_mean": 0.9,
                "elapsed_seconds_mean": 9.0,
                "elapsed_seconds_std": 0.9,
            }
        ],
    }

    assert normalize_experiment_for_reproducibility(first) == (
        normalize_experiment_for_reproducibility(second)
    )


def test_reproducibility_normalization_ignores_machine_precision_noise() -> None:
    first = {
        "summary": [
            {
                "model": "random_forest",
                "log_loss_mean": 0.203032398166005,
            }
        ]
    }
    second = {
        "summary": [
            {
                "model": "random_forest",
                "log_loss_mean": 0.203032398166006,
            }
        ]
    }

    assert normalize_experiment_for_reproducibility(first) == (
        normalize_experiment_for_reproducibility(second)
    )


def test_reproducibility_normalization_preserves_meaningful_differences() -> None:
    first = {"metric": 0.203032398166}
    second = {"metric": 0.203032399166}

    assert normalize_experiment_for_reproducibility(first) != (
        normalize_experiment_for_reproducibility(second)
    )


def test_each_split_records_provenance(tmp_path) -> None:
    config = ExperimentConfig(
        model_set="classical",
        features=4,
        repeats=2,
        max_samples=100,
        output_dir=str(tmp_path / "unused"),
    )

    payload = run_benchmark(config, write_output=False)

    provenance = payload["split_provenance"]
    assert len(provenance) == 2
    assert [entry["seed"] for entry in provenance] == [42, 43]
    for entry in provenance:
        assert len(entry["selected_features"]) == 4
        assert len(entry["train_index_hash"]) == 64
        assert len(entry["test_index_hash"]) == 64
        assert entry["preprocessing_fit_scope"] == "training_partition_only"
        assert sum(entry["training_class_counts"].values()) == entry["train_samples"]
        assert sum(entry["test_class_counts"].values()) == entry["test_samples"]
        assert entry["train_index_hash"] != entry["test_index_hash"]
