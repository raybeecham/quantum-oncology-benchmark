from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from quantum_oncology_benchmark.evolution_cohort import (
    generate_virtual_tumors,
    run_evolution_cohort,
)
from quantum_oncology_benchmark.evolution_cohort_config import (
    EvolutionCohortConfig,
    ParameterRange,
)


def test_cohort_profile_loads_with_resolved_base_profile() -> None:
    config = EvolutionCohortConfig.from_yaml("configs/evolution-cohort-smoke.yaml")

    assert config.protocol_version == "evolution-cohort-v1"
    assert config.virtual_tumors == 6
    assert config.reference_strategy == "continuous"
    assert config.candidate_strategy == "burden_adaptive"
    assert Path(config.base_profile).name == "evolution-smoke.yaml"
    assert len(config.parameter_ranges) == 3


def test_latin_hypercube_sampling_is_deterministic_and_bounded() -> None:
    config = EvolutionCohortConfig.from_yaml("configs/evolution-cohort-smoke.yaml")

    first = generate_virtual_tumors(config)
    second = generate_virtual_tumors(config)

    assert first == second
    assert len(first) == config.virtual_tumors
    assert len({row["parameter_fingerprint"] for row in first}) == config.virtual_tumors
    for row in first:
        assert 0.005 <= float(row["initial_resistant_fraction"]) <= 0.05
        assert 0.010 <= float(row["resistant_growth_rate_per_day"]) <= 0.020
        assert 0.060 <= float(row["sensitive_drug_kill_rate_per_day"]) <= 0.100


def test_cohort_run_produces_matched_outcomes_and_sensitivity_rows() -> None:
    config = EvolutionCohortConfig.from_yaml("configs/evolution-cohort-smoke.yaml")

    payload = run_evolution_cohort(config, write_output=False)

    assert payload["schema_version"] == "evolution-cohort-1.0"
    assert payload["quantum_algorithm_used"] is False
    assert len(payload["virtual_tumors"]) == 6
    assert len(payload["virtual_tumor_outcomes"]) == 12
    assert len(payload["paired_strategy_comparisons"]) == 6
    assert len(payload["strategy_robustness_summary"]) == 2
    assert len(payload["parameter_sensitivity"]) == 15
    assert len(payload["cohort_fingerprint"]) == 64
    assert payload["paired_robustness_summary"]["virtual_tumors"] == 6


def test_cohort_artifacts_are_complete(tmp_path: Path) -> None:
    config = EvolutionCohortConfig.from_yaml("configs/evolution-cohort-smoke.yaml")
    config = replace(config, output_dir=str(tmp_path / "cohort"))

    payload = run_evolution_cohort(config)

    expected = {
        "evolution_cohort_experiment.json",
        "virtual_tumors.csv",
        "virtual_tumor_outcomes.csv",
        "paired_strategy_comparisons.csv",
        "strategy_robustness_summary.csv",
        "parameter_sensitivity.csv",
        "EVOLUTION_COHORT_REPORT.md",
    }
    assert {path.name for path in (tmp_path / "cohort").iterdir()} == expected
    assert set(payload["artifacts"]) == {
        "experiment",
        "virtual_tumors",
        "outcomes",
        "paired",
        "strategy_summary",
        "sensitivity",
        "report",
    }


def test_parameter_range_rejects_invalid_bounds_and_log_zero() -> None:
    with pytest.raises(ValueError, match="maximum must exceed minimum"):
        ParameterRange(
            name="sensitive_growth_rate_per_day",
            minimum=0.03,
            maximum=0.02,
        ).validate()

    with pytest.raises(ValueError, match="positive minimum"):
        ParameterRange(
            name="sensitive_to_resistant_rate_per_day",
            minimum=0.0,
            maximum=0.001,
            scale="log10",
        ).validate()


def test_cohort_rejects_identical_comparison_strategies() -> None:
    config = EvolutionCohortConfig.from_yaml("configs/evolution-cohort-smoke.yaml")
    invalid = replace(config, candidate_strategy="continuous")

    with pytest.raises(ValueError, match="must differ"):
        invalid.validate()
