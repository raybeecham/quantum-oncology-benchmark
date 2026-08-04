"""Public orchestration API for evolutionary tumor simulations."""

from __future__ import annotations

from typing import Any

from .evolution_config import (
    CloneParameters,
    CompetitionParameters,
    EvolutionConfig,
    TreatmentStrategy,
)
from .evolution_model import simulate_strategy, summarize_strategy
from .evolution_reporting import (
    fingerprint_evolution_payload,
    normalize_evolution_for_reproducibility,
    render_evolution_report,
    write_evolution_artifacts,
)
from .reporting import environment_metadata, utc_now


def run_evolution_simulation(
    config: EvolutionConfig,
    *,
    write_output: bool = True,
) -> dict[str, Any]:
    """Run all configured two-clone treatment strategies."""
    config.validate()
    if write_output and not config.output_dir:
        raise ValueError("output_dir is required when write_output=True")
    trajectories: list[dict[str, Any]] = []
    schedules: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []

    for strategy in config.strategies:
        trajectory, schedule, policy_events = simulate_strategy(strategy, config)
        summary, milestone_events = summarize_strategy(
            strategy,
            trajectory,
            schedule,
            config,
        )
        trajectories.extend(trajectory)
        schedules.extend(schedule)
        summaries.append(summary)
        events.extend([*policy_events, *milestone_events])

    summaries.sort(key=lambda row: float(row["tumor_burden_auc"]))
    fingerprint_config = config.to_dict()
    fingerprint_config.pop("output_dir", None)
    fingerprint_payload = {
        "config": fingerprint_config,
        "strategy_summary": summaries,
        "population_trajectories": trajectories,
        "treatment_schedule": schedules,
    }
    payload: dict[str, Any] = {
        "schema_version": "evolution-1.0",
        "generated_at": utc_now(),
        "research_use_only": True,
        "clinical_decision_support": False,
        "treatment_recommendation": False,
        "quantum_algorithm_used": False,
        "config": config.to_dict(),
        "model": {
            "family": "deterministic_competitive_two_clone_ode",
            "solver": "scipy_solve_ivp_piecewise_policy",
            "state_variables": ["sensitive_cells", "resistant_cells"],
            "selection_mechanism": "differential drug kill with ecological competition",
            "transition": "optional one-way sensitive_to_resistant",
            "parameter_source": "illustrative_repository_defaults",
        },
        "population_trajectories": trajectories,
        "treatment_schedule": schedules,
        "strategy_summary": summaries,
        "evolutionary_events": sorted(
            events,
            key=lambda row: (
                str(row["strategy"]),
                float(row["time_days"]),
                str(row["event_type"]),
            ),
        ),
        "simulation_fingerprint": fingerprint_evolution_payload(fingerprint_payload),
        "environment": environment_metadata(),
        "claim_boundary": (
            "Assumption-driven research simulation only; no patient-specific prediction, clinical "
            "utility, treatment recommendation, or quantum advantage is claimed."
        ),
    }
    if write_output:
        artifacts = write_evolution_artifacts(payload, config.output_dir)
        payload["artifacts"] = {name: str(path) for name, path in artifacts.items()}
    return payload


__all__ = [
    "CloneParameters",
    "CompetitionParameters",
    "EvolutionConfig",
    "TreatmentStrategy",
    "normalize_evolution_for_reproducibility",
    "render_evolution_report",
    "run_evolution_simulation",
    "write_evolution_artifacts",
]
