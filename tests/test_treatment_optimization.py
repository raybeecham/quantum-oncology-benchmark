"""Independent numerical and research-contract checks for schedule optimization."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from quantum_oncology_benchmark.cli import main
from quantum_oncology_benchmark.evolution_config import CloneParameters, EvolutionConfig
from quantum_oncology_benchmark.evolution_model import simulate_strategy, summarize_strategy
from quantum_oncology_benchmark.treatment_config import (
    ObjectiveWeights,
    TreatmentOptimizationConfig,
)
from quantum_oncology_benchmark.treatment_optimization import (
    build_optimization_cohorts,
    normalize_treatment_experiment,
    run_treatment_optimization,
)
from quantum_oncology_benchmark.treatment_search import (
    ScheduleEvaluator,
    aggregate_loss,
    exact_search,
    feasible_schedules,
    objective_components,
    random_search,
    schedule_feasible,
    simulated_annealing,
)


def smoke() -> TreatmentOptimizationConfig:
    return TreatmentOptimizationConfig.from_yaml("configs/treatment-optimization-smoke.yaml")


def test_explicit_all_on_and_off_match_existing_policies() -> None:
    model = EvolutionConfig.from_yaml("configs/evolution-smoke.yaml")
    for policy, bits in (("continuous", (1, 1, 1)), ("no_treatment", (0, 0, 0))):
        expected, _, _ = simulate_strategy(policy, model)
        actual, dosing, _ = simulate_strategy("continuous", model, explicit_schedule=bits)
        assert [r["total_burden"] for r in actual] == pytest.approx(
            [r["total_burden"] for r in expected], rel=1e-9)
        summary, _ = summarize_strategy("scheduled", actual, dosing, model)
        assert summary["cumulative_dose_days"] == model.horizon_days * bits[0]


def test_fractional_boundaries_respect_analytic_drug_kill() -> None:
    # With zero growth and no transition, S(t)=S(0)*exp(-kill*cumulative dose).
    model = replace(EvolutionConfig(), horizon_days=10.0, time_step_days=2.0,
                    sensitive=CloneParameters(0.0, 0.2), resistant=CloneParameters(0.0, 0.0))
    trajectory, dosing, _ = simulate_strategy("continuous", model, explicit_schedule=(1, 0, 1))
    assert [r["interval_end_days"] for r in dosing if r["interval_end_days"] in (10 / 3, 20 / 3)] == pytest.approx([10 / 3, 20 / 3])
    dose = sum(r["duration_days"] * r["treatment_intensity"] for r in dosing)
    assert dose == pytest.approx(20 / 3)
    assert trajectory[-1]["sensitive_cells"] == pytest.approx(
        model.sensitive_initial * np.exp(-0.2 * 20 / 3), rel=2e-8)
    assert trajectory[-1]["resistant_cells"] == pytest.approx(model.resistant_initial)
    assert all(r["duration_days"] > 0 for r in dosing)
    assert trajectory[-1]["time_days"] == pytest.approx(model.horizon_days)


def test_objective_normalization_and_absolute_resistance() -> None:
    model = replace(EvolutionConfig(), horizon_days=10.0)
    weights = ObjectiveWeights(2.0, 3.0, 4.0)
    values = objective_components({"tumor_burden_auc": 5e6,
                                   "final_resistant_cells": 100_000,
                                   "cumulative_dose_days": 2.0}, model, weights)
    assert values["loss"] == pytest.approx(2 * 0.5 + 3 * 0.1 + 4 * 0.2)
    robust = aggregate_loss([{"loss": 1.0}, {"loss": 3.0}], 0.5)
    assert robust["objective"] == 2.5  # population std = 1; not sample std or variance.


def test_schedule_constraints_enforce_domain() -> None:
    cfg = smoke()
    assert schedule_feasible((1, 1, 0, 1), cfg)
    assert not schedule_feasible((1, 1, 1, 0), cfg)
    assert not schedule_feasible((1, 1, 1, 1), cfg)
    assert not schedule_feasible((0, 0), cfg)
    assert not schedule_feasible((True, 0, 0, 0), cfg)


def test_exact_oracle_and_budget_accounting(monkeypatch: pytest.MonkeyPatch) -> None:
    # Synthetic objective with a known optimum; no tumor model implementation is mirrored.
    target = (1, 0, 1, 0)
    def score(bits: tuple[int, ...], *_: object) -> dict[str, float]:
        return {"loss": float(sum(a != b for a, b in zip(bits, target, strict=True)))}
    monkeypatch.setattr("quantum_oncology_benchmark.treatment_search.score_schedule", score)
    cfg = smoke()
    models = [EvolutionConfig()]
    oracle = exact_search(ScheduleEvaluator(models, cfg))
    assert oracle["schedule"] == "1010"
    assert oracle["objective"] == 0.0
    assert oracle["unique_schedule_evaluations"] == len(feasible_schedules(cfg))
    for search in (random_search, simulated_annealing):
        evaluator = ScheduleEvaluator(models, cfg)
        result = search(evaluator, 42)
        assert result["unique_schedule_evaluations"] == cfg.evaluation_budget
        assert result["tumor_simulations"] == result["unique_schedule_evaluations"]
        assert result["objective_requests"] == result["cache_hits"] + result["unique_schedule_evaluations"]
        assert result["objective"] >= oracle["objective"]
        assert all(schedule_feasible(bits, cfg) for bits in evaluator.cache)


def test_disjoint_cohort_and_replay_contract(tmp_path: Path) -> None:
    cfg = replace(smoke(), cohort_profile="configs/evolution-virtual-cohort.yaml",
                  search_tumors=2, evaluation_tumors=2, repeats=1, evaluation_budget=4)
    search, evaluation, tumors, design = build_optimization_cohorts(cfg)
    assert len(search) == len(evaluation) == 2
    assert {m.profile_name for m in search}.isdisjoint({m.profile_name for m in evaluation})
    assert {r["partition"] for r in tumors} == {"search", "evaluation"}
    assert design["evaluation_kind"] == "disjoint_virtual_tumor_scenarios"
    first = run_treatment_optimization(replace(cfg, output_dir=str(tmp_path / "a")))
    second = run_treatment_optimization(replace(cfg, output_dir=str(tmp_path / "b")))
    assert first["fingerprint"] == second["fingerprint"]
    assert normalize_treatment_experiment(first) == normalize_treatment_experiment(second)
    for run in first["scientific"]["search_results"]:
        assert len(run["tumor_outcomes"]) == 2
        assert run["absolute_optimality_gap"] >= 0
    assert (tmp_path / "a/TREATMENT_OPTIMIZATION_REPORT.md").exists()
    assert (tmp_path / "a/evaluation_outcomes.csv").exists()


def test_unavailable_oracle_is_not_reported_as_zero_gap() -> None:
    payload = run_treatment_optimization(replace(smoke(), exact_search=False, repeats=1), write_output=False)
    assert all(r["absolute_optimality_gap"] is None and r["oracle_hit"] is None
               for r in payload["scientific"]["search_results"])


@pytest.mark.parametrize("field,value", [("intervals", 13), ("risk_aversion", float("nan")),
                                         ("search_tumors", True), ("evaluation_budget", 0)])
def test_invalid_profiles_rejected(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        replace(smoke(), **{field: value}).validate()


def test_cli_smoke(tmp_path: Path) -> None:
    assert main(["optimize-treatment", "--config", "configs/treatment-optimization-smoke.yaml",
                 "--output", str(tmp_path)]) == 0
    assert (tmp_path / "search_results.csv").exists()
