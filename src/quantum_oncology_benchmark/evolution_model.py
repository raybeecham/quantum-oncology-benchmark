"""Numerical engine for deterministic two-clone tumor evolution."""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from .evolution_config import EvolutionConfig, TreatmentStrategy

FloatArray = NDArray[np.float64]
Row = dict[str, Any]


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
    transition = config.sensitive_to_resistant_rate_per_day * sensitive

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
        sensitive_growth - sensitive_treatment_loss - transition,
        resistant_growth - resistant_treatment_loss + transition,
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
    rows: list[Row],
    predicate: Callable[[Row], bool],
) -> float | None:
    for row in rows:
        if predicate(row):
            return float(row["time_days"])
    return None


def simulate_strategy(
    strategy: TreatmentStrategy,
    config: EvolutionConfig,
) -> tuple[list[Row], list[Row], list[Row]]:
    """Simulate one treatment strategy over the configured time horizon."""
    trajectory: list[Row] = []
    schedule: list[Row] = []
    policy_events: list[Row] = []

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
        right_hand_side = partial(
            _derivatives,
            treatment_intensity=intensity,
            config=config,
        )
        solution = solve_ivp(
            right_hand_side,
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


def summarize_strategy(
    strategy: TreatmentStrategy,
    trajectory: list[Row],
    schedule: list[Row],
    config: EvolutionConfig,
) -> tuple[Row, list[Row]]:
    """Summarize one simulated policy and emit milestone events."""
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
    summary: Row = {
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
    events: list[Row] = [
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
