"""End-to-end reproducible classical treatment-scheduling benchmark."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from .evolution_cohort import build_virtual_config, generate_virtual_tumors
from .evolution_cohort_config import EvolutionCohortConfig
from .evolution_config import EvolutionConfig
from .evolution_model import simulate_strategy, summarize_strategy
from .reporting import environment_metadata, utc_now
from .treatment_config import TreatmentOptimizationConfig
from .treatment_search import (
    SEARCH_METHODS,
    ScheduleEvaluator,
    exact_search,
    objective_components,
)

Row = dict[str, Any]


def build_optimization_cohorts(
    config: TreatmentOptimizationConfig,
) -> tuple[list[EvolutionConfig], list[EvolutionConfig], list[Row], Row]:
    """Partition a declared scenario design before any search or evaluation."""
    base = EvolutionConfig.from_yaml(config.base_profile)
    if config.cohort_profile is None:
        clean = base.to_dict()
        clean.pop("output_dir", None)
        return [base], [base], [], {"evaluation_kind": "same_reference_scenario",
                                   "base_config": clean, "cohort_definition": None}
    cohort = EvolutionCohortConfig.from_yaml(config.cohort_profile)
    cohort = replace(cohort, virtual_tumors=config.search_tumors + config.evaluation_tumors,
                     seed=config.cohort_seed)
    cohort.validate()
    tumors = generate_virtual_tumors(cohort)
    models: list[EvolutionConfig] = []
    for index, tumor in enumerate(tumors):
        tumor["partition"] = "search" if index < config.search_tumors else "evaluation"
        models.append(build_virtual_config(base, cohort, tumor))
    clean = base.to_dict()
    clean.pop("output_dir", None)
    design = cohort.to_dict()
    design.pop("output_dir", None)
    design.pop("base_profile", None)
    return models[:config.search_tumors], models[config.search_tumors:], tumors, {
        "evaluation_kind": "disjoint_virtual_tumor_scenarios", "base_config": clean,
        "cohort_definition": design,
        "split_rule": "first search_tumors rows search, remaining rows evaluation, fixed before optimization",
        "population_interpretation": "one designed LHS sample, not independent patient observations",
    }


def _policy_outcomes(model: EvolutionConfig, config: TreatmentOptimizationConfig) -> list[Row]:
    rows: list[Row] = []
    for strategy in model.strategies:
        trajectory, schedule, _ = simulate_strategy(strategy, model)
        summary, _ = summarize_strategy(strategy, trajectory, schedule, model)
        max_on = 0.0
        on_days = 0.0
        for row in schedule:
            on_days = on_days + float(row["duration_days"]) if row["treatment_on"] else 0.0
            max_on = max(max_on, on_days)
        interval_days = model.horizon_days / config.intervals
        compatible = (
            summary["cumulative_dose_days"] <= config.max_treated_intervals * interval_days + 1e-8
            and max_on <= config.max_consecutive_treated * interval_days + 1e-8
        )
        rows.append({**summary, **objective_components(summary, model, config.weights),
                     "within_exposure_and_consecutive_limits": compatible,
                     "schedule_grid_membership": "not_asserted_for_feedback_or_fixed_policies"})
    return rows


def normalize_treatment_experiment(payload: Row) -> Row:
    """Scientific content only, excluding paths, timings and environment."""
    scientific = deepcopy(payload["scientific"])
    config = scientific["config"]
    for key in ("output_dir", "base_profile", "cohort_profile"):
        config.pop(key, None)
    for run in scientific["search_results"]:
        run.pop("wall_seconds", None)
        run.pop("simulator_seconds", None)
    # Keep full stored precision; equality contract rounds dimensionless objectives/endpoints.
    def canonical(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: canonical(v) for k, v in value.items()}
        if isinstance(value, list):
            return [canonical(v) for v in value]
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("non-finite scientific artifact")
            return round(value, 10)
        return value
    return canonical(scientific)  # type: ignore[no-any-return]


def render_treatment_report(payload: Row) -> str:
    scientific = payload["scientific"]
    config = scientific["config"]
    lines = ["# Classical Treatment Schedule Benchmark", "",
             "Deterministic scenario research, without patient calibration or therapeutic-benefit claims.", "",
             f"Protocol: `{config['protocol_version']}`. Profile: `{config['profile_name']}`.",
             f"Binary intervals: {config['intervals']}; exposure cap: {config['max_treated_intervals']}; consecutive cap: {config['max_consecutive_treated']}.",
             f"Search tumors: {config['search_tumors']}; evaluation tumors: {config['evaluation_tumors']}.",
             f"Evaluation design: `{scientific['design']['evaluation_kind']}`.", "",
             "Objective: mean normalized loss + risk_aversion * population standard deviation of loss.",
             "Loss combines burden AUC / (initial burden * horizon), terminal absolute resistant cells / initial burden, and dose-days / horizon.",
             "Weights are declared research preferences; dose is an exposure proxy, not a toxicity model.", "",
             "| Method | Repeat | Schedule | Search objective | Evaluation objective | Unique evaluations | Gap to oracle | Stop |",
             "|---|---:|---|---:|---:|---:|---:|---|"]
    evaluation = {(r["method"], r["repeat"]): r for r in scientific["evaluation_summary"]}
    for run in scientific["search_results"]:
        value = evaluation[(run["method"], run["repeat"])]["objective"]
        gap = run["absolute_optimality_gap"]
        gap_text = "unavailable" if gap is None else f"{gap:.6g}"
        lines.append(f"| {run['method']} | {run['repeat']} | `{run['schedule']}` | {run['objective']:.6g} | {value:.6g} | {run['unique_schedule_evaluations']} | {gap_text} | {run['stop_reason']} |")
    lines.extend(["", "The exact oracle is exhaustive only on the declared feasible binary grid and search tumors, with numerical ODE scoring. It is not a continuous-control or clinical optimum.",
                  "Oracle effort is reported separately. Random and annealing have identical maximum unique-evaluation budgets, private caches and declared seeds. Annealing may stop early at its proposal cap.",
                  "Evaluation scenarios were never used for optimizer selection, temperature tuning or objective-weight choice. Repeated search seeds are computational repetitions, not independent biological samples.",
                  "Reference policy results are descriptive controls. Policies that violate the exposure limits or differ from the schedule grid are not eligible competitors in the constrained-search optimum.",
                  "No quantum computation was used. A Dirac adapter requires explicit polynomial formulation or validated surrogate and simulator re-evaluation of decoded schedules.", "",
                  f"Scientific fingerprint: `{payload['fingerprint']}`.", ""])
    return "\n".join(lines)


def run_treatment_optimization(
    config: TreatmentOptimizationConfig, *, write_output: bool = True
) -> Row:
    """Search schedules and evaluate the frozen winners on separate scenarios."""
    config.validate()
    search_models, evaluation_models, tumors, design = build_optimization_cohorts(config)
    runs: list[Row] = []
    if config.exact_search:
        oracle = exact_search(ScheduleEvaluator(search_models, config))
        oracle.update({"repeat": 0, "seed": None})
        runs.append(oracle)
        optimum: float | None = float(oracle["objective"])
    else:
        optimum = None
    # No oracle values or cached schedules are passed to budgeted methods.
    for search in SEARCH_METHODS.values():
        for repeat in range(config.repeats):
            seed = config.seed + repeat
            run = search(ScheduleEvaluator(search_models, config), seed)
            run.update({"repeat": repeat, "seed": seed})
            runs.append(run)
    evaluation_rows: list[Row] = []
    evaluation_summary: list[Row] = []
    evaluator = ScheduleEvaluator(evaluation_models, config)
    for run in runs:
        gap = None if optimum is None else max(0.0, float(run["objective"]) - optimum)
        run["absolute_optimality_gap"] = gap
        run["oracle_hit"] = None if gap is None else gap <= 1e-8
        schedule = tuple(int(x) for x in run["schedule"])
        evaluated = evaluator(schedule)
        evaluation_summary.append({"method": run["method"], "repeat": run["repeat"],
                                   "schedule": run["schedule"],
                                   **{k: v for k, v in evaluated.items() if k != "tumor_outcomes"}})
        for index, outcome in enumerate(evaluated["tumor_outcomes"]):
            evaluation_rows.append({"method": run["method"], "repeat": run["repeat"],
                                    "schedule": run["schedule"], "evaluation_tumor_index": index,
                                    **outcome})
    controls: list[Row] = []
    for partition, models in (("search", search_models), ("evaluation", evaluation_models)):
        for index, model in enumerate(models):
            controls.extend({"partition": partition, "tumor_index": index, **row}
                            for row in _policy_outcomes(model, config))
    scientific = {"config": config.to_dict(), "design": design, "virtual_tumors": tumors,
                  "search_results": runs, "evaluation_outcomes": evaluation_rows,
                  "evaluation_summary": evaluation_summary, "reference_policy_outcomes": controls,
                  "numerical_contract": {"ode_rtol": 1e-8, "ode_atol": 1e-6,
                                         "integration_policy": "piecewise_at_schedule_boundaries_and_base_time_step",
                                         "auc": "trapezoidal_on_recorded_time_grid", "oracle_hit_tolerance": 1e-8},
                  "search_contract": {"annealing_restart_after_proposals_without_novelty": 20 * config.intervals,
                                      "annealing_restart_selection": "uniform unseen feasible schedule on small domains",
                                      "cache_scope": "private_per_solver_run", "budget_unit": "unique_feasible_schedule"}}
    payload: Row = {"schema_version": "treatment-optimization-1.0", "generated_at": utc_now(),
                    "research_use_only": True, "clinical_decision_support": False,
                    "quantum_algorithm_used": False, "environment": environment_metadata(),
                    "scientific": scientific,
                    "evaluation_resources": {"unique_schedules": len(evaluator.cache),
                                             "tumor_simulations": len(evaluator.cache) * len(evaluation_models),
                                             "simulator_seconds": evaluator.simulator_seconds}}
    payload["fingerprint"] = hashlib.sha256(json.dumps(
        normalize_treatment_experiment(payload), sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    if write_output:
        destination = Path(config.output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "treatment_optimization_experiment.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
        pd.DataFrame([{k: v for k, v in run.items() if k not in {"trace", "tumor_outcomes"}} for run in runs]).to_csv(destination / "search_results.csv", index=False)
        pd.DataFrame([{ "method": run["method"], "repeat": run["repeat"], **row }
                      for run in runs for row in run["trace"]]).to_csv(destination / "search_trace.csv", index=False)
        pd.DataFrame(evaluation_summary).to_csv(destination / "evaluation_summary.csv", index=False)
        pd.DataFrame(evaluation_rows).to_csv(destination / "evaluation_outcomes.csv", index=False)
        pd.DataFrame(controls).to_csv(destination / "reference_policy_outcomes.csv", index=False)
        if tumors:
            pd.DataFrame(tumors).to_csv(destination / "virtual_tumors.csv", index=False)
        (destination / "TREATMENT_OPTIMIZATION_REPORT.md").write_text(render_treatment_report(payload), encoding="utf-8")
    return payload
