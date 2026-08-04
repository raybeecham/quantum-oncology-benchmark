"""Reporting and reproducibility helpers for evolutionary simulations."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd


def fingerprint_evolution_payload(payload: dict[str, Any]) -> str:
    """Return a stable SHA-256 fingerprint for scientific simulation content."""
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
    normalized: dict[str, Any] = deepcopy(payload)
    normalized.pop("generated_at", None)
    normalized.pop("artifacts", None)
    config = normalized.get("config")
    if isinstance(config, dict):
        config.pop("output_dir", None)
    environment = normalized.get("environment")
    if isinstance(environment, dict):
        environment.pop("platform", None)
        environment.pop("git_commit", None)
    return normalized
