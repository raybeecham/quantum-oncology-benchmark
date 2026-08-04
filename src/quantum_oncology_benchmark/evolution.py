"""Deterministic two-clone tumor evolution and treatment strategy simulation."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from .reporting import environment_metadata, utc_now

FloatArray = NDArray[np.float64]
TreatmentStrategy = Literal[
    "no_treatment",
    "continuous",
    "fixed_intermittent",
    "burden_adaptive",
]

_SUPPORTED_STRATEGIES: tuple[TreatmentStrategy, ...] = (
    "no_treatment",
    "continuous",
    "fixed_intermittent",
    "burden_adaptive",
)


@dataclass(frozen=True, slots=True)
class CloneParameters:
    """Growth and treatment-response assumptions for one tumor clone."""

    growth_rate_per_day: float
    drug_kill_rate_per_day: float

    def validate(self, label: str) -> None:
        """Validate one clone parameter block."""
        if self.growth_rate_per_day < 0:
            raise ValueError(f"{label}.growth_rate_per_day must be nonnegative")
        if self.drug_kill_rate_per_day < 0:
            raise ValueError(f"{label}.drug_kill_rate_per_day must be nonnegative")


@dataclass(frozen=True, slots=True)
class CompetitionParameters:
    """Pairwise competition coefficients in the two-clone system."""

    resistant_effect_on_sensitive: float = 1.0
    sensitive_effect_on_resistant: float = 1.0

    def validate(self) -> None:
        """Validate competition coefficients."""
        if self.resistant_effect_on_sensitive < 0:
            raise ValueError("resistant_effect_on_sensitive must be nonnegative")
        if self.sensitive_effect_on_resistant < 0:
            raise ValueError("sensitive_effect_on_resistant must be nonnegative")


@dataclass(frozen=True, slots=True)
class EvolutionConfig:
    """Configuration for one assumption-driven evolutionary simulation."""

    protocol_version: str = "evolution-protocol-v1"
    profile_name: str = "two-clone-selection-v1"
    sensitive_initial: float = 990_000.0
    resistant_initial: float = 10_000.0
    carrying_capacity: float = 10_000_000.0
    sensitive: CloneParameters = CloneParameters(0.03, 0.08)
    resistant: CloneParameters = CloneParameters(0.015, 0.0)
    competition: CompetitionParameters = CompetitionParameters()
    sensitive_to_resistant_rate_per_day: float = 0.0
    horizon_days: float = 365.0
    time_step_days: float = 1.0
    strategies: tuple[TreatmentStrategy, ...] = _SUPPORTED_STRATEGIES
    intermittent_on_days: float = 14.0
    intermittent_off_days: float = 14.0
    adaptive_stop_fraction: float = 0.50
    adaptive_restart_fraction: float = 1.00
    resistant_dominance_fraction: float = 0.50
    progression_burden_multiple: float = 1.20
    output_dir: str = "reports/evolution-two-clone"

    def validate(self) -> None:
        """Validate biological assumptions, policies, and output settings."""
        if self.protocol_version != "evolution-protocol-v1":
            raise ValueError("protocol_version must be 'evolution-protocol-v1'")
        if not self.profile_name:
            raise ValueError("profile_name is required")
        if self.sensitive_initial <= 0 or self.resistant_initial < 0:
            raise ValueError("initial clone populations must be nonnegative with sensitive > 0")
        if self.carrying_capacity <= 0:
            raise ValueError("carrying_capacity must be positive")
        if self.sensitive_initial + self.resistant_initial > self.carrying_capacity:
            raise ValueError("initial total burden must not exceed carrying_capacity")
        self.sensitive.validate("sensitive")
        self.resistant.validate("resistant")
        self.competition.validate()
        if self.sensitive_to_resistant_rate_per_day < 0:
            raise ValueError("sensitive_to_resistant_rate_per_day must be nonnegative")
        if self.horizon_days <= 0:
            raise ValueError("horizon_days must be positive")
        if self.time_step_days <= 0 or self.time_step_days > self.horizon_days:
            raise ValueError("time_step_days must be positive and no greater than horizon_days")
        if not self.strategies:
            raise ValueError("at least one treatment strategy is required")
        unsupported = sorted(set(self.strategies) - set(_SUPPORTED_STRATEGIES))
        if unsupported:
            raise ValueError(f"unsupported treatment strategies: {unsupported}")
        if len(set(self.strategies)) != len(self.strategies):
            raise ValueError("strategies must not contain duplicates")
        if self.intermittent_on_days <= 0 or self.intermittent_off_days <= 0:
            raise ValueError("intermittent on/off durations must be positive")
        if not 0 < self.adaptive_stop_fraction < self.adaptive_restart_fraction:
            raise ValueError(
                "adaptive_stop_fraction must be positive and below adaptive_restart_fraction"
            )
        if not 0 < self.resistant_dominance_fraction < 1:
            raise ValueError("resistant_dominance_fraction must be between 0 and 1")
        if self.progression_burden_multiple <= 1:
            raise ValueError("progression_burden_multiple must be greater than 1")
        if not self.output_dir:
            raise ValueError("output_dir is required")

    @property
    def initial_total_burden(self) -> float:
        """Return the initial total tumor burden."""
        return self.sensitive_initial + self.resistant_initial

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable configuration mapping."""
        payload = asdict(self)
        payload["strategies"] = list(self.strategies)
        return payload

    @classmethod
    def from_yaml(cls, path: str | Path) -> EvolutionConfig:
        """Load a versioned evolution profile from YAML."""
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise ValueError("evolution configuration root must be a mapping")
        raw_sensitive = payload.pop("sensitive", {})
        raw_resistant = payload.pop("resistant", {})
        raw_competition = payload.pop("competition", {})
        if not isinstance(raw_sensitive, dict) or not isinstance(raw_resistant, dict):
            raise ValueError("sensitive and resistant parameter blocks must be mappings")
        if not isinstance(raw_competition, dict):
            raise ValueError("competition parameter block must be a mapping")
        if isinstance(payload.get("strategies"), list):
            payload["strategies"] = tuple(str(item) for item in payload["strategies"])
        config = cls(
            sensitive=CloneParameters(**raw_sensitive),
            resistant=CloneParameters(**raw_resistant),
            competition=CompetitionParameters(**raw_competition),
            **payload,
        )
        config.validate()
        return config


def _shannon_diversity(sensitive: float, resistant: float) -> float:
    total = sensitive + resistant
    if total <= 0:
        return 0.0
    proportions = (sensitive / total, resistant / total)
    return -sum(value * math.log(value) for value in proportions if value > 0)


def _derivatives(
    _time: float,
    populations: FloatArray,
    *,
    treatment_intensity: float,
    config: EvolutionConfig,
) -> list[float]:
    sensitive = max(float(populations[0]), 0.0)
    resistant = max(float(populations[1]), 0.0)
    capacity = config.carrying_capacity
    mutation = config.sensitive_to_resistant_rate_per_day * sensitive

    sensitive_growth = config.sensitive.growth_rate_per_day * sensitive * (
        1.0
        - (
            sensitive
            + config.competition.resistant_effect_on_sensitive * resistant
        )
        / capacity
    )
    resistant_growth = config.resistant.growth_rate_per_day * resistant * (
        1.0
        - (
            resistant
            + config.competition.sensitive_effect_on_resistant * sensitive
        )
        / capacity
    )
    sensitive_treatment_loss = (
        config.sensitive.drug_kill_rate_per_day * treatment_intensity * sensitive
    )
    resistant_treatment_loss = (
        config.resistant.drug_kill_rate_per_day * treatment_intensity * resistant
    )
    return [
        sensitive_growth - sensitive_treatment_loss - mutation,
        resistant_growth - resistant_treatment_loss + mutation,
    ]


def _policy_intensity(
    strategy: TreatmentStrategy,
    time_days: float,
    total_burden: float,
    adaptive_on: bool,
    config: EvolutionConfig,
) -> tuple[float, bool, str | None]:
    if strategy == "no_treatment":
        return 0.0, False, None
    if strategy == "continuous":
        return 1.0, True, None
    if strategy == "fixed_intermittent":
        cycle_days = config.intermittent_on_days + config.intermittent_off_days
        position = time_days % cycle_days
        is_on = position < config.intermittent_on_days
        return (1.0 if is_on else 0.0), is_on, None

    stop_threshold = config.adaptive_stop_fraction * config.initial_total_burden
    restart_threshold = config.adaptive_restart_fraction * config.initial_total_burden
    event: str | None = None
    next_state = adaptive_on
    if adaptive_on and total_burden <= stop_threshold:
        next_state = False
        event = "adaptive_treatment_stopped"
    elif not adaptive_on and total_burden >= restart_threshold:
        next_state = True
        event = "adaptive_treatment_restarted"
    return (1.0 if next_state else 0.0), next_state, event


def _first_crossing(
    rows: list[dict[str, Any]],
    predicate: Any,
) -> float | None:
    for row in rows:
        if bool(predicate(row)):
            return float(row["time_days"])
    return None


def _simulate_strategy(
    strategy: TreatmentStrategy,
    config: EvolutionConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    trajectory: list[dict[str, Any]] = []
    schedule: list[dict[str, Any]] = []
    policy_events: list[dict[str, Any]] = []

    time_days = 0.0
    sensitive = config.sensitive_initial
    resistant = config.resistant_initial
    adaptive_on = strategy in {"continuous", "fixed_intermittent", "burden_adaptive"}
    previous_on: bool | None = None
    treatment_cycle = 0

    def append_trajectory_row() -> None:
        total = sensitive + resistant
        resistant_fraction = 0.0 if total <= 0 else resistant / total
        trajectory.append(
            {
                "strategy": strategy,
                "time_days": time_days,
                "sensitive_cells": sensitive,
                "resistant_cells": resistant,
                "total_burden": total,
                "sensitive_fraction": 0.0 if total <= 0 else sensitive / total,
                "resistant_fraction": resistant_fraction,
                "shannon_diversity": _shannon_diversity(sensitive, resistant),
            }
        )

    append_trajectory_row()
    while time_days < config.horizon_days - 1e-12:
        total = sensitive + resistant
        intensity, adaptive_on, adaptive_event = _policy_intensity(
            strategy,
            time_days,
            total,
            adaptive_on,
            config,
        )
        is_on = intensity > 0
        if previous_on is None:
            if is_on:
                treatment_cycle = 1
                policy_events.append(
                    {
                        "strategy": strategy,
                        "time_days": time_days,
                        "event_type": "treatment_started",
                        "total_burden": total,
                        "resistant_fraction": 0.0 if total <= 0 else resistant / total,
                    }
                )
        elif is_on != previous_on:
            event_type = "treatment_restarted" if is_on else "treatment_stopped"
            if is_on:
                treatment_cycle += 1
            policy_events.append(
                {
                    "strategy": strategy,
                    "time_days": time_days,
                    "event_type": adaptive_event or event_type,
                    "total_burden": total,
                    "resistant_fraction": 0.0 if total <= 0 else resistant / total,
                }
            )
        previous_on = is_on

        duration = min(config.time_step_days, config.horizon_days - time_days)
        schedule.append(
            {
                "strategy": strategy,
                "interval_start_days": time_days,
                "interval_end_days": time_days + duration,
                "duration_days": duration,
                "treatment_intensity": intensity,
                "treatment_on": is_on,
                "treatment_cycle": treatment_cycle,
            }
        )
        solution = solve_ivp(
            lambda current_time, populations: _derivatives(
                current_time,
                cast(FloatArray, populations),
                treatment_intensity=intensity,
                config=config,
            ),
            (time_days, time_days + duration),
            np.asarray([sensitive, resistant], dtype=float),
            t_eval=[time_days + duration],
            rtol=1e-8,
            atol=1e-6,
        )
        if not solution.success:
            raise RuntimeError(f"evolution solver failed for {strategy}: {solution.message}")
        sensitive = max(float(solution.y[0, -1]), 0.0)
        resistant = max(float(solution.y[1, -1]), 0.0)
        time_days += duration
        append_trajectory_row()

    return trajectory, schedule, policy_events


def _summarize_strategy(
    strategy: TreatmentStrategy,
    trajectory: list[dict[str, Any]],
    schedule: list[dict[str, Any]],
    config: EvolutionConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    times = np.asarray([float(row["time_days"]) for row in trajectory], dtype=float)
    burden = np.asarray([float(row["total_burden"]) for row in trajectory], dtype=float)
    resistant_fraction = np.asarray(
        [float(row["resistant_fraction"]) for row in trajectory], dtype=float
    )
    diversity = np.asarray(
        [float(row["shannon_diversity"]) for row in trajectory], dtype=float
    )
    initial = config.initial_total_burden
    progression_threshold = config.progression_burden_multiple * initial
    dominance_day = _first_crossing(
        trajectory,
        lambda row: float(row["resistant_fraction"])
        >= config.resistant_dominance_fraction,
    )
    progression_day = _first_crossing(
        trajectory[1:],
        lambda row: float(row["total_burden"]) >= progression_threshold,
    )
    nadir_index = int(np.argmin(burden))
    dose_days = sum(
        float(row["treatment_intensity"]) * float(row["duration_days"])
        for row in schedule
    )
    treatment_cycles = max(
        (int(row["treatment_cycle"]) for row in schedule),
        default=0,
    )
    below_initial = (burden <= initial).astype(float)
    summary = {
        "strategy": strategy,
        "initial_total_burden": initial,
        "final_sensitive_cells": float(trajectory[-1]["sensitive_cells"]),
        "final_resistant_cells": float(trajectory[-1]["resistant_cells"]),
        "final_total_burden": float(burden[-1]),
        "minimum_total_burden": float(burden[nadir_index]),
        "nadir_day": float(times[nadir_index]),
        "maximum_total_burden": float(np.max(burden)),
        "final_resistant_fraction": float(resistant_fraction[-1]),
        "maximum_resistant_fraction": float(np.max(resistant_fraction)),
        "resistant_dominance_day": dominance_day,
        "progression_threshold": progression_threshold,
        "time_to_progression_days": progression_day,
        "tumor_burden_auc": float(np.trapezoid(burden, times)),
        "mean_shannon_diversity": float(np.trapezoid(diversity, times) / times[-1]),
        "control_time_below_initial_days": float(np.trapezoid(below_initial, times)),
        "cumulative_dose_days": dose_days,
        "treatment_fraction": dose_days / config.horizon_days,
        "treatment_cycles": treatment_cycles,
    }
    events: list[dict[str, Any]] = [
        {
            "strategy": strategy,
            "time_days": float(times[nadir_index]),
            "event_type": "tumor_burden_nadir",
            "total_burden": float(burden[nadir_index]),
            "resistant_fraction": float(resistant_fraction[nadir_index]),
        }
    ]
    if dominance_day is not None:
        row = next(
            item for item in trajectory if float(item["time_days"]) == dominance_day
        )
        events.append(
            {
                "strategy": strategy,
                "time_days": dominance_day,
                "event_type": "resistant_dominance_threshold_crossed",
                "total_burden": float(row["total_burden"]),
                "resistant_fraction": float(row["resistant_fraction"]),
            }
        )
    if progression_day is not None:
        row = next(
            item for item in trajectory if float(item["time_days"]) == progression_day
        )
        events.append(
            {
                "strategy": strategy,
                "time_days": progression_day,
                "event_type": "progression_threshold_crossed",
                "total_burden": float(row["total_burden"]),
                "resistant_fraction": float(row["resistant_fraction"]),
            }
        )
    return summary, events


def _fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def render_evolution_report(payload: dict[str, Any]) -> str:
    """Render a bounded human-readable evolutionary simulation report."""
    config = payload["config"]
    lines = [
        "# Tumor Evolution and Treatment Strategy Simulation Report",
        "",
        "> **Research-use-only warning:** This deterministic two-clone model is an assumption-driven",
        "> educational research simulator. It is not patient-specific, clinically calibrated, or a",
        "> treatment recommendation or clinical decision-support system.",
        "",
        "## Experiment",
        "",
        f"- Protocol: `{config['protocol_version']}`",
        f"- Profile: `{config['profile_name']}`",
        f"- Horizon: **{float(config['horizon_days']):.1f} days**",
        f"- Time step: **{float(config['time_step_days']):.2f} days**",
        f"- Initial sensitive population: **{float(config['sensitive_initial']):,.0f}**",
        f"- Initial resistant population: **{float(config['resistant_initial']):,.0f}**",
        f"- Carrying capacity: **{float(config['carrying_capacity']):,.0f}**",
        "",
        "## Strategy Results",
        "",
        "| Strategy | Final burden | Nadir (day) | Final resistant fraction | Resistant dominance day | Progression day | Dose-days | Cycles |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["strategy_summary"]:
        dominance = (
            "not reached"
            if row["resistant_dominance_day"] is None
            else f"{float(row['resistant_dominance_day']):.1f}"
        )
        progression = (
            "not reached"
            if row["time_to_progression_days"] is None
            else f"{float(row['time_to_progression_days']):.1f}"
        )
        lines.append(
            f"| {row['strategy']} | {float(row['final_total_burden']):,.0f} | "
            f"{float(row['minimum_total_burden']):,.0f} ({float(row['nadir_day']):.1f}) | "
            f"{float(row['final_resistant_fraction']):.3f} | {dominance} | "
            f"{progression} | {float(row['cumulative_dose_days']):.1f} | "
            f"{int(row['treatment_cycles'])} |"
        )

    lines.extend(
        [
            "",
            "## Model Equations",
            "",
            "The simulator uses a deterministic competitive two-clone system with treatment selection",
            "and an optional one-way sensitive-to-resistant transition:",
            "",
            "```text",
            "dS/dt = rS*S*(1 - (S + alpha_SR*R)/K) - killS*u(t)*S - mu*S",
            "dR/dt = rR*R*(1 - (R + alpha_RS*S)/K) - killR*u(t)*R + mu*S",
            "```",
            "",
            "Treatment intensity `u(t)` is defined by each policy. The burden-adaptive policy stops",
            "and restarts treatment using configured fractions of the initial total burden.",
            "",
            "## Interpretation Boundaries",
            "",
            "- The two populations are abstractions, not measured patient clones.",
            "- Parameter values are illustrative and are not inferred from a clinical cohort.",
            "- Deterministic trajectories represent model expectations, not stochastic outcomes.",
            "- Resistance may be genetic, epigenetic, microenvironmental, or pharmacologic in reality;",
            "  this first protocol compresses those mechanisms into one resistant population.",
            "- Strategy comparisons are conditional on the declared assumptions and do not establish",
            "  treatment efficacy, safety, or patient benefit.",
            "- No quantum algorithm is used in `evolution-protocol-v1`.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "python -m pip install -e '.[dev]'",
            f"qob evolve --config configs/evolution-two-clone.yaml --output {config['output_dir']}",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def write_evolution_artifacts(
    payload: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write machine-readable and human-readable evolution artifacts."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "evolution_experiment.json"
    trajectory_path = destination / "population_trajectories.csv"
    schedule_path = destination / "treatment_schedule.csv"
    summary_path = destination / "strategy_summary.csv"
    events_path = destination / "evolutionary_events.csv"
    report_path = destination / "EVOLUTION_REPORT.md"

    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    pd.DataFrame(payload["population_trajectories"]).to_csv(trajectory_path, index=False)
    pd.DataFrame(payload["treatment_schedule"]).to_csv(schedule_path, index=False)
    pd.DataFrame(payload["strategy_summary"]).to_csv(summary_path, index=False)
    pd.DataFrame(payload["evolutionary_events"]).to_csv(events_path, index=False)
    report_path.write_text(render_evolution_report(payload), encoding="utf-8")
    return {
        "experiment_json": json_path,
        "population_trajectories_csv": trajectory_path,
        "treatment_schedule_csv": schedule_path,
        "strategy_summary_csv": summary_path,
        "evolutionary_events_csv": events_path,
        "report": report_path,
    }


def normalize_evolution_for_reproducibility(payload: dict[str, Any]) -> dict[str, Any]:
    """Remove expected operational variance from an evolution payload copy."""
    normalized = deepcopy(payload)
    normalized.pop("generated_at", None)
    normalized.pop("artifacts", None)
    config = normalized.get("config")
    if isinstance(config, dict):
        config.pop("output_dir", None)
    environment = normalized.get("environment")
    if isinstance(environment, dict):
        environment.pop("platform", None)
        environment.pop("git_commit", None)
    return cast(dict[str, Any], normalized)


def run_evolution_simulation(
    config: EvolutionConfig,
    *,
    write_output: bool = True,
) -> dict[str, Any]:
    """Run all configured two-clone treatment strategies."""
    config.validate()
    trajectories: list[dict[str, Any]] = []
    schedules: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []

    for strategy in config.strategies:
        trajectory, schedule, policy_events = _simulate_strategy(strategy, config)
        summary, milestone_events = _summarize_strategy(
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
    fingerprint_payload = {
        "config": config.to_dict(),
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
            key=lambda row: (str(row["strategy"]), float(row["time_days"]), str(row["event_type"])),
        ),
        "simulation_fingerprint": _fingerprint(fingerprint_payload),
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
