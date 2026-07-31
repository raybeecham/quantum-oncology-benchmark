from __future__ import annotations

from typing import Any

from quantum_oncology_benchmark.cli import main
from quantum_oncology_benchmark.config import NestedCVConfig


def test_doctor_returns_success(capsys) -> None:
    assert main(["doctor"]) == 0
    output = capsys.readouterr().out
    assert "Core dependencies" in output
    assert "Quantum dependencies" in output


def test_nested_cv_cli_builds_locked_configuration(monkeypatch, capsys, tmp_path) -> None:
    captured: dict[str, Any] = {}

    def fake_run_nested_cv(config: NestedCVConfig) -> dict[str, Any]:
        captured["config"] = config
        return {
            "summary": [
                {
                    "model": "logistic_regression",
                    "balanced_accuracy_mean": 0.9,
                }
            ]
        }

    monkeypatch.setattr("quantum_oncology_benchmark.cli.run_nested_cv", fake_run_nested_cv)
    output = tmp_path / "nested"
    result = main(
        [
            "nested-cv",
            "--outer-folds",
            "2",
            "--inner-folds",
            "2",
            "--search-profile",
            "sensitivity-v1",
            "--calibration-bins",
            "8",
            "--model",
            "logistic_regression",
            "--max-samples",
            "80",
            "--output",
            str(output),
        ]
    )

    assert result == 0
    config = captured["config"]
    assert isinstance(config, NestedCVConfig)
    assert config.primary_metric == "balanced_accuracy"
    assert config.search_profile == "sensitivity-v1"
    assert config.calibration_bins == 8
    assert config.models == ("logistic_regression",)
    assert config.outer_folds == 2
    assert config.inner_folds == 2
    stdout = capsys.readouterr().out
    assert "Nested cross-validation complete" in stdout
    assert "Search profile: sensitivity-v1" in stdout
