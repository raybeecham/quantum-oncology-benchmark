from __future__ import annotations

from dataclasses import replace

import pytest

from quantum_oncology_benchmark.evolution import (
    EvolutionConfig,
    normalize_evolution_for_reproducibility,
    run_evolution_simulation,
)


def _summary_by_strategy(payload: dict[str, object]) -> dict[str, dict[str, object]]:
    rows = payload["strategy_summary"]
    assert isinstance(rows, list)
    return {str(row["strategy"]): row for row in rows if isinstance(row, dict)}


def test_evolution_profile_loads_with_research_boundaries() -> None:
    config = EvolutionConfig.from_yaml("configs/evolution-two-clone.yaml")

    assert config.protocol_version == "evolution-protocol-v1"
    assert config.profile_name == "two-clone-selection-v1"
    assert config.initial_total_burden == 1_000_000.0
    assert config.strategies == (
        "no_treatment",
        "continuous",
        "fixed_intermittent",
        "burden_adaptive",
    )


def test_evolution_profile_rejects_invalid_adaptive_thresholds() -> None:
    config = EvolutionConfig.from_yaml("configs/evolution-smoke.yaml")

    with pytest.raises(ValueError, match="adaptive_stop_fraction"):
        replace(
            config,
            adaptive_stop_fraction=1.0,
            adaptive_restart_fraction=0.5,
        ).validate()


def test_treatment_selection_increases_resistant_fraction() -> None:
    config = replace(
        EvolutionConfig.from_yaml("configs/evolution-two-clone.yaml"),
        horizon_days=120.0,
        strategies=("no_treatment", "continuous"),
    )

    payload = run_evolution_simulation(config, write_output=False)
    summary = _summary_by_strategy(payload)

    untreated = summary["no_treatment"]
    continuous = summary["continuous"]
    assert float(continuous["final_resistant_fraction"]) > float(
        untreated["final_resistant_fraction"]
    )
    assert float(untreated["cumulative_dose_days"]) == 0.0
    assert float(continuous["cumulative_dose_days"]) == 120.0


def test_adaptive_policy_delays_resistant_dominance_under_reference_assumptions() -> None:
    config = replace(
        EvolutionConfig.from_yaml("configs/evolution-two-clone.yaml"),
        strategies=("continuous", "burden_adaptive"),
    )

    payload = run_evolution_simulation(config, write_output=False)
    summary = _summary_by_strategy(payload)

    continuous_day = summary["continuous"]["resistant_dominance_day"]
    adaptive_day = summary["burden_adaptive"]["resistant_dominance_day"]
    assert continuous_day is not None
    assert adaptive_day is not None
    assert float(adaptive_day) > float(continuous_day)
    assert float(summary["burden_adaptive"]["cumulative_dose_days"]) < float(
        summary["continuous"]["cumulative_dose_days"]
    )


def test_evolution_outputs_are_nonnegative_and_complete() -> None:
    config = EvolutionConfig.from_yaml("configs/evolution-smoke.yaml")

    payload = run_evolution_simulation(config, write_output=False)
    trajectories = payload["population_trajectories"]
    schedules = payload["treatment_schedule"]
    assert isinstance(trajectories, list)
    assert isinstance(schedules, list)
    expected_trajectory_rows = len(config.strategies) * (int(config.horizon_days) + 1)
    expected_schedule_rows = len(config.strategies) * int(config.horizon_days)
    assert len(trajectories) == expected_trajectory_rows
    assert len(schedules) == expected_schedule_rows
    for row in trajectories:
        assert float(row["sensitive_cells"]) >= 0.0
        assert float(row["resistant_cells"]) >= 0.0
        assert float(row["total_burden"]) == pytest.approx(
            float(row["sensitive_cells"]) + float(row["resistant_cells"])
        )
        assert 0.0 <= float(row["resistant_fraction"]) <= 1.0
    for row in schedules:
        assert float(row["treatment_intensity"]) in {0.0, 1.0}


def test_evolution_writes_artifact_package(tmp_path) -> None:
    config = replace(
        EvolutionConfig.from_yaml("configs/evolution-smoke.yaml"),
        output_dir=str(tmp_path / "evolution"),
    )

    payload = run_evolution_simulation(config)

    assert payload["schema_version"] == "evolution-1.0"
    assert payload["clinical_decision_support"] is False
    assert payload["quantum_algorithm_used"] is False
    output = tmp_path / "evolution"
    assert (output / "evolution_experiment.json").exists()
    assert (output / "population_trajectories.csv").exists()
    assert (output / "treatment_schedule.csv").exists()
    assert (output / "strategy_summary.csv").exists()
    assert (output / "evolutionary_events.csv").exists()
    report = (output / "EVOLUTION_REPORT.md").read_text(encoding="utf-8")
    assert "## Strategy Results" in report
    assert "not patient-specific" in report
    assert "No quantum algorithm" in report


def test_evolution_reproduces_scientific_payload() -> None:
    config = EvolutionConfig.from_yaml("configs/evolution-smoke.yaml")

    first = run_evolution_simulation(config, write_output=False)
    second = run_evolution_simulation(config, write_output=False)

    assert normalize_evolution_for_reproducibility(first) == (
        normalize_evolution_for_reproducibility(second)
    )
