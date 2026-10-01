"""Simulator-scored binary schedule searches with explicit resource accounting."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import product
from typing import Any

import numpy as np

from .evolution_config import EvolutionConfig
from .evolution_model import simulate_strategy, summarize_strategy
from .treatment_config import ObjectiveWeights, TreatmentOptimizationConfig

Schedule = tuple[int, ...]
Row = dict[str, Any]


def schedule_feasible(schedule: Schedule, config: TreatmentOptimizationConfig) -> bool:
    """Hard exposure and consecutive-treatment limits, not soft penalties."""
    if len(schedule) != config.intervals or any(type(x) is not int or x not in (0, 1) for x in schedule):
        return False
    if sum(schedule) > config.max_treated_intervals:
        return False
    run = 0
    for value in schedule:
        run = run + 1 if value else 0
        if run > config.max_consecutive_treated:
            return False
    return True


def objective_components(summary: Row, model: EvolutionConfig, weights: ObjectiveWeights) -> Row:
    """Normalize by each tumor's initial burden and model horizon, retaining raw endpoints."""
    weights.validate()
    burden_auc = float(summary["tumor_burden_auc"]) / (model.initial_total_burden * model.horizon_days)
    terminal_resistant = float(summary["final_resistant_cells"]) / model.initial_total_burden
    dose = float(summary["cumulative_dose_days"]) / model.horizon_days
    loss = weights.burden_auc * burden_auc + weights.terminal_resistant * terminal_resistant + weights.dose * dose
    if not all(math.isfinite(v) for v in (burden_auc, terminal_resistant, dose, loss)):
        raise RuntimeError("non-finite treatment objective")
    return {"normalized_burden_auc": burden_auc, "normalized_terminal_resistant": terminal_resistant,
            "normalized_dose": dose, "loss": loss}


def score_schedule(schedule: Schedule, model: EvolutionConfig, weights: ObjectiveWeights) -> Row:
    """Evaluate a complete schedule using the existing two-clone ODE engine."""
    trajectory, dosing, _ = simulate_strategy("continuous", model, explicit_schedule=schedule)
    summary, _ = summarize_strategy("scheduled", trajectory, dosing, model)
    return {**summary, **objective_components(summary, model, weights)}


def aggregate_loss(rows: list[Row], risk_aversion: float) -> Row:
    """Mean plus a declared population-standard-deviation risk penalty."""
    losses = np.asarray([row["loss"] for row in rows], dtype=float)
    mean = float(np.mean(losses))
    std = float(np.std(losses, ddof=0))
    return {"objective": mean + risk_aversion * std, "mean_loss": mean, "std_loss": std,
            "worst_loss": float(np.max(losses)), "tumors": len(rows)}


@dataclass
class ScheduleEvaluator:
    """Private cache per solver run; oracle and other solvers do not share objective values."""

    models: list[EvolutionConfig]
    config: TreatmentOptimizationConfig
    cache: dict[Schedule, Row] = field(default_factory=dict)
    trace: list[Row] = field(default_factory=list)
    requests: int = 0
    simulator_seconds: float = 0.0

    def __call__(self, schedule: Schedule) -> Row:
        if not schedule_feasible(schedule, self.config):
            raise ValueError("infeasible treatment schedule")
        self.requests += 1
        if schedule not in self.cache:
            started = time.perf_counter()
            rows = [score_schedule(schedule, model, self.config.weights) for model in self.models]
            self.simulator_seconds += time.perf_counter() - started
            value = {**aggregate_loss(rows, self.config.risk_aversion), "tumor_outcomes": rows}
            self.cache[schedule] = value
            best = min(float(v["objective"]) for v in self.cache.values())
            self.trace.append({"unique_evaluation": len(self.cache), "schedule": "".join(map(str, schedule)),
                               "objective": value["objective"], "best_objective": best})
        return self.cache[schedule]


def feasible_schedules(config: TreatmentOptimizationConfig) -> list[Schedule]:
    """Enumerate the small exact-reference domain only."""
    if config.intervals > 12:
        raise ValueError("enumeration is capped at 12 intervals")
    return [x for x in product((0, 1), repeat=config.intervals) if schedule_feasible(x, config)]


def _finish(method: str, evaluator: ScheduleEvaluator, started: float, proposals: int, stop: str) -> Row:
    winner = min(evaluator.cache, key=lambda x: (float(evaluator.cache[x]["objective"]), x))
    return {"method": method, "schedule": "".join(map(str, winner)), **evaluator.cache[winner],
            "unique_schedule_evaluations": len(evaluator.cache), "objective_requests": evaluator.requests,
            "cache_hits": evaluator.requests - len(evaluator.cache),
            "tumor_simulations": len(evaluator.cache) * len(evaluator.models),
            "proposals": proposals, "stop_reason": stop,
            "wall_seconds": time.perf_counter() - started, "simulator_seconds": evaluator.simulator_seconds,
            "trace": evaluator.trace}


def exact_search(evaluator: ScheduleEvaluator) -> Row:
    """Exhaust every feasible schedule, giving a numerical grid oracle."""
    started = time.perf_counter()
    schedules = feasible_schedules(evaluator.config)
    for schedule in schedules:
        evaluator(schedule)
    return _finish("exact", evaluator, started, len(schedules), "complete_feasible_domain")


def random_search(evaluator: ScheduleEvaluator, seed: int) -> Row:
    """Uniform feasible sampling without replacement on the bounded domain."""
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    config = evaluator.config
    proposals = 0
    if config.intervals <= 12:
        schedules = feasible_schedules(config)
        for index in rng.permutation(len(schedules))[:config.evaluation_budget]:
            evaluator(schedules[int(index)])
            proposals += 1
        stop = "budget" if len(evaluator.cache) == config.evaluation_budget else "domain_exhausted"
    else:
        limit = config.evaluation_budget * 100
        while len(evaluator.cache) < config.evaluation_budget and proposals < limit:
            schedule = tuple(int(x) for x in rng.integers(0, 2, config.intervals))
            proposals += 1
            if schedule_feasible(schedule, config):
                evaluator(schedule)
        if not evaluator.cache:
            evaluator((0,) * config.intervals)
        stop = "budget" if len(evaluator.cache) == config.evaluation_budget else "proposal_limit"
    return _finish("random", evaluator, started, proposals, stop)


def simulated_annealing(evaluator: ScheduleEvaluator, seed: int) -> Row:
    """Single-bit proposals; infeasible proposals are rejected without simulation."""
    started = time.perf_counter()
    config = evaluator.config
    rng = np.random.default_rng(seed)
    current = (0,) * config.intervals
    current_loss = float(evaluator(current)["objective"])
    proposals = 0
    restarts = 0
    since_novel = 0
    domain = feasible_schedules(config) if config.intervals <= 12 else None
    domain_size = len(domain) if domain is not None else None
    target = min(config.evaluation_budget, domain_size) if domain_size is not None else config.evaluation_budget
    limit = config.evaluation_budget * 100
    while len(evaluator.cache) < target and proposals < limit:
        proposals += 1
        since_novel += 1
        if since_novel >= 20 * config.intervals:
            # Restart after a declared novelty plateau; neither oracle nor evaluation
            # outcomes inform restart selection. Every novel restart consumes budget.
            if domain is not None:
                unseen = [bits for bits in domain if bits not in evaluator.cache]
                current = unseen[int(rng.integers(len(unseen)))]
                current_loss = float(evaluator(current)["objective"])
                restarts += 1
                since_novel = 0
                continue
            candidate_restart = tuple(int(x) for x in rng.integers(0, 2, config.intervals))
            if schedule_feasible(candidate_restart, config) and candidate_restart not in evaluator.cache:
                current = candidate_restart
                current_loss = float(evaluator(current)["objective"])
                restarts += 1
                since_novel = 0
                continue
        index = int(rng.integers(config.intervals))
        candidate = (*current[:index], 1 - current[index], *current[index + 1:])
        if not schedule_feasible(candidate, config):
            continue
        fraction = min(len(evaluator.cache) / max(target - 1, 1), 1.0)
        temperature = config.annealing_start_temperature * (
            config.annealing_end_temperature / config.annealing_start_temperature
        ) ** fraction
        is_novel = candidate not in evaluator.cache
        candidate_loss = float(evaluator(candidate)["objective"])
        if is_novel:
            since_novel = 0
        delta = candidate_loss - current_loss
        if delta <= 0 or rng.random() < math.exp(-delta / temperature):
            current, current_loss = candidate, candidate_loss
    stop = "budget" if len(evaluator.cache) == config.evaluation_budget else (
        "domain_exhausted" if domain_size == len(evaluator.cache) else "proposal_limit")
    result = _finish("annealing", evaluator, started, proposals, stop)
    result["restarts"] = restarts
    return result


SEARCH_METHODS: dict[str, Callable[[ScheduleEvaluator, int], Row]] = {
    "random": random_search, "annealing": simulated_annealing,
}
