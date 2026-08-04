"""Deterministic virtual-tumor cohorts and parameter-sensitivity analysis."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .evolution import run_evolution_simulation
from .evolution_cohort_config import EvolutionCohortConfig, ParameterRange
from .evolution_config import (
    CloneParameters,
    CompetitionParameters,
    EvolutionConfig,
    TreatmentStrategy,
)
from .reporting import environment_metadata, utc_now

Row = dict[str, Any]


def _fingerprint(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _sample_value(specification: ParameterRange, unit_value: float) -> float:
    if specification.scale == "linear":
        return specification.minimum + unit_value * (
            specification.maximum - specification.minimum
        )
    lower = math.log10(specification.minimum)
    upper = math.log10(specification.maximum)
    return float(10 ** (lower + unit_value * (upper - lower)))


def generate_virtual_tumors(config: EvolutionCohortConfig) -> list[Row]:
    """Generate a deterministic Latin-hypercube parameter cohort."""
    config.validate()
    rng = np.random.default_rng(config.seed)
    sample_count = config.virtual_tumors
    dimensions = len(config.parameter_ranges)
    unit_samples = np.empty((sample_count, dimensions), dtype=float)
    for column in range(dimensions):
        strata = (np.arange(sample_count, dtype=float) + rng.random(sample_count)) / sample_count
        rng.shuffle(strata)
        unit_samples[:, column] = strata

    rows: list[Row] = []
    for index in range(sample_count):
        parameters = {
            specification.name: _sample_value(
                specification,
                float(unit_samples[index, column]),
            )
            for column, specification in enumerate(config.parameter_ranges)
        }
        tumor_id = f"virtual-tumor-{index + 1:04d}"
        rows.append(
            {
                "virtual_tumor_id": tumor_id,
                **parameters,
                "parameter_fingerprint": _fingerprint(parameters),
            }
        )
    return rows


def _build_virtual_config(
    base: EvolutionConfig,
    cohort: EvolutionCohortConfig,
    row: Row,
) -> EvolutionConfig:
    total_initial = base.initial_total_burden
    resistant_fraction = float(
        row.get(
            "initial_resistant_fraction",
            base.resistant_initial / total_initial,
        )
    )
    resistant_initial = total_initial * resistant_fraction
    sensitive_initial = total_initial - resistant_initial

    sensitive = CloneParameters(
        growth_rate_per_day=float(
            row.get(
                "sensitive_growth_rate_per_day",
                base.sensitive.growth_rate_per_day,
            )
        ),
        drug_kill_rate_per_day=float(
            row.get(
                "sensitive_drug_kill_rate_per_day",
                base.sensitive.drug_kill_rate_per_day,
            )
        ),
    )
    resistant = CloneParameters(
        growth_rate_per_day=float(
            row.get(
                "resistant_growth_rate_per_day",
                base.resistant.growth_rate_per_day,
            )
        ),
        drug_kill_rate_per_day=float(
            row.get(
                "resistant_drug_kill_rate_per_day",
                base.resistant.drug_kill_rate_per_day,
            )
        ),
    )
    competition = CompetitionParameters(
        resistant_effect_on_sensitive=float(
            row.get(
                "resistant_effect_on_sensitive",
                base.competition.resistant_effect_on_sensitive,
            )
        ),
        sensitive_effect_on_resistant=float(
            row.get(
                "sensitive_effect_on_resistant",
                base.competition.sensitive_effect_on_resistant,
            )
        ),
    )
    strategies = cast(
        tuple[TreatmentStrategy, ...],
        (cohort.reference_strategy, cohort.candidate_strategy),
    )
    virtual_config = replace(
        base,
        profile_name=f"{base.profile_name}-{row['virtual_tumor_id']}",
        sensitive_initial=sensitive_initial,
        resistant_initial=resistant_initial,
        sensitive=sensitive,
        resistant=resistant,
        competition=competition,
        sensitive_to_resistant_rate_per_day=float(
            row.get(
                "sensitive_to_resistant_rate_per_day",
                base.sensitive_to_resistant_rate_per_day,
            )
        ),
        strategies=strategies,
        output_dir="",
    )
    virtual_config.validate()
    return virtual_config


def _capped_event_day(value: Any, horizon_days: float) -> float:
    return horizon_days if value is None else min(float(value), horizon_days)


def _outcome_rows(
    tumor_id: str,
    payload: dict[str, Any],
    horizon_days: float,
) -> list[Row]:
    rows: list[Row] = []
    for summary in payload["strategy_summary"]:
        row = dict(summary)
        row["virtual_tumor_id"] = tumor_id
        row["resistance_control_days"] = _capped_event_day(
            summary["resistant_dominance_day"],
            horizon_days,
        )
        row["progression_control_days"] = _capped_event_day(
            summary["time_to_progression_days"],
            horizon_days,
        )
        row["resistant_dominance_reached"] = summary["resistant_dominance_day"] is not None
        row["progression_threshold_reached"] = summary["time_to_progression_days"] is not None
        rows.append(row)
    return rows


def _direction(delta: float, *, candidate_positive: bool) -> str:
    if abs(delta) <= 1e-12:
        return "tie"
    candidate_favored = delta > 0 if candidate_positive else delta < 0
    return "candidate" if candidate_favored else "reference"


def _paired_row(
    tumor_id: str,
    outcomes: list[Row],
    config: EvolutionCohortConfig,
) -> Row:
    by_strategy = {str(row["strategy"]): row for row in outcomes}
    reference = by_strategy[config.reference_strategy]
    candidate = by_strategy[config.candidate_strategy]
    resistance_delta = float(candidate["resistance_control_days"]) - float(
        reference["resistance_control_days"]
    )
    progression_delta = float(candidate["progression_control_days"]) - float(
        reference["progression_control_days"]
    )
    auc_delta = float(candidate["tumor_burden_auc"]) - float(
        reference["tumor_burden_auc"]
    )
    final_delta = float(candidate["final_total_burden"]) - float(
        reference["final_total_burden"]
    )
    dose_delta = float(candidate["cumulative_dose_days"]) - float(
        reference["cumulative_dose_days"]
    )
    return {
        "virtual_tumor_id": tumor_id,
        "reference_strategy": config.reference_strategy,
        "candidate_strategy": config.candidate_strategy,
        "resistance_control_delta_days": resistance_delta,
        "resistance_control_direction": _direction(
            resistance_delta,
            candidate_positive=True,
        ),
        "progression_control_delta_days": progression_delta,
        "progression_control_direction": _direction(
            progression_delta,
            candidate_positive=True,
        ),
        "tumor_burden_auc_delta": auc_delta,
        "tumor_burden_auc_direction": _direction(auc_delta, candidate_positive=False),
        "final_total_burden_delta": final_delta,
        "final_total_burden_direction": _direction(final_delta, candidate_positive=False),
        "cumulative_dose_days_delta": dose_delta,
        "cumulative_dose_direction": _direction(dose_delta, candidate_positive=False),
        "candidate_delayed_resistant_dominance": resistance_delta > 1e-12,
        "candidate_delayed_progression": progression_delta > 1e-12,
        "candidate_lower_burden_auc": auc_delta < -1e-12,
        "candidate_lower_final_burden": final_delta < -1e-12,
        "candidate_lower_cumulative_dose": dose_delta < -1e-12,
    }


def _quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "q05": float(np.quantile(array, 0.05)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.50)),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
    }


def summarize_strategy_robustness(outcomes: list[Row]) -> list[Row]:
    """Aggregate strategy outcomes across the complete virtual cohort."""
    frame = pd.DataFrame(outcomes)
    rows: list[Row] = []
    metrics = (
        "final_total_burden",
        "tumor_burden_auc",
        "resistance_control_days",
        "progression_control_days",
        "cumulative_dose_days",
    )
    for strategy, group in frame.groupby("strategy", sort=True):
        row: Row = {
            "strategy": str(strategy),
            "virtual_tumors": len(group),
            "resistant_dominance_reached_fraction": float(
                group["resistant_dominance_reached"].mean()
            ),
            "progression_threshold_reached_fraction": float(
                group["progression_threshold_reached"].mean()
            ),
        }
        for metric in metrics:
            quantiles = _quantiles([float(value) for value in group[metric].tolist()])
            for label, value in quantiles.items():
                row[f"{metric}_{label}"] = value
        rows.append(row)
    return rows


def summarize_paired_robustness(pairs: list[Row]) -> Row:
    """Summarize how often the candidate policy changes matched outcomes."""
    frame = pd.DataFrame(pairs)
    count = len(frame)
    return {
        "virtual_tumors": count,
        "candidate_delayed_resistant_dominance_fraction": float(
            frame["candidate_delayed_resistant_dominance"].mean()
        ),
        "candidate_delayed_progression_fraction": float(
            frame["candidate_delayed_progression"].mean()
        ),
        "candidate_lower_burden_auc_fraction": float(
            frame["candidate_lower_burden_auc"].mean()
        ),
        "candidate_lower_final_burden_fraction": float(
            frame["candidate_lower_final_burden"].mean()
        ),
        "candidate_lower_cumulative_dose_fraction": float(
            frame["candidate_lower_cumulative_dose"].mean()
        ),
        "resistance_control_delta_days_median": float(
            frame["resistance_control_delta_days"].median()
        ),
        "progression_control_delta_days_median": float(
            frame["progression_control_delta_days"].median()
        ),
        "tumor_burden_auc_delta_median": float(frame["tumor_burden_auc_delta"].median()),
        "final_total_burden_delta_median": float(
            frame["final_total_burden_delta"].median()
        ),
        "cumulative_dose_days_delta_median": float(
            frame["cumulative_dose_days_delta"].median()
        ),
    }


def parameter_sensitivity_rows(
    tumors: list[Row],
    pairs: list[Row],
    config: EvolutionCohortConfig,
) -> list[Row]:
    """Calculate descriptive Spearman associations for sampled assumptions."""
    tumor_frame = pd.DataFrame(tumors)
    pair_frame = pd.DataFrame(pairs)
    merged = tumor_frame.merge(pair_frame, on="virtual_tumor_id", validate="one_to_one")
    outcomes = (
        "resistance_control_delta_days",
        "progression_control_delta_days",
        "tumor_burden_auc_delta",
        "final_total_burden_delta",
        "cumulative_dose_days_delta",
    )
    rows: list[Row] = []
    for specification in config.parameter_ranges:
        for outcome in outcomes:
            result = spearmanr(
                merged[specification.name].to_numpy(dtype=float),
                merged[outcome].to_numpy(dtype=float),
            )
            rho = float(result.statistic)
            p_value = float(result.pvalue)
            rows.append(
                {
                    "parameter": specification.name,
                    "parameter_scale": specification.scale,
                    "outcome": outcome,
                    "spearman_rho": None if not math.isfinite(rho) else rho,
                    "absolute_spearman_rho": None if not math.isfinite(rho) else abs(rho),
                    "nominal_p_value": None if not math.isfinite(p_value) else p_value,
                    "virtual_tumors": len(merged),
                    "interpretation": "descriptive_global_sensitivity_association",
                }
            )
    rows.sort(
        key=lambda row: (
            str(row["outcome"]),
            -(
                float(row["absolute_spearman_rho"])
                if row["absolute_spearman_rho"] is not None
                else -1.0
            ),
            str(row["parameter"]),
        )
    )
    return rows


def render_cohort_report(payload: dict[str, Any]) -> str:
    """Render a bounded human-readable virtual-cohort report."""
    config = payload["config"]
    paired = payload["paired_robustness_summary"]
    lines = [
        "# Virtual Tumor Cohort and Parameter Sensitivity Report",
        "",
        "> **Research-use-only warning:** These are deterministic virtual tumors generated from",
        "> declared parameter ranges. They are not patients, clinical predictions, or treatment guidance.",
        "",
        "## Cohort Contract",
        "",
        f"- Protocol: `{config['protocol_version']}`",
        f"- Profile: `{config['profile_name']}`",
        f"- Virtual tumors: **{config['virtual_tumors']}**",
        f"- Sampling: `{config['sampling_method']}` with seed `{config['seed']}`",
        f"- Reference strategy: `{config['reference_strategy']}`",
        f"- Candidate strategy: `{config['candidate_strategy']}`",
        "- Treatment-policy settings are fixed; only declared biological parameters vary.",
        "- Event times not reached within the horizon are capped at the simulation horizon for paired summaries.",
        "",
        "## Paired Robustness Summary",
        "",
        f"- Candidate delayed resistant dominance in **{paired['candidate_delayed_resistant_dominance_fraction']:.1%}** of virtual tumors.",
        f"- Candidate delayed the configured burden threshold in **{paired['candidate_delayed_progression_fraction']:.1%}** of virtual tumors.",
        f"- Candidate reduced tumor-burden AUC in **{paired['candidate_lower_burden_auc_fraction']:.1%}** of virtual tumors.",
        f"- Candidate reduced cumulative dose in **{paired['candidate_lower_cumulative_dose_fraction']:.1%}** of virtual tumors.",
        f"- Median resistance-control difference: **{paired['resistance_control_delta_days_median']:+.1f} days**.",
        f"- Median configured burden-threshold difference: **{paired['progression_control_delta_days_median']:+.1f} days**.",
        "",
        "## Strategy Robustness",
        "",
        "| Strategy | Dominance reached | Burden threshold reached | Final burden median | Resistance control median | Progression control median | Dose median |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["strategy_robustness_summary"]:
        lines.append(
            f"| {row['strategy']} | {row['resistant_dominance_reached_fraction']:.1%} | "
            f"{row['progression_threshold_reached_fraction']:.1%} | "
            f"{row['final_total_burden_median']:,.0f} | "
            f"{row['resistance_control_days_median']:.1f} | "
            f"{row['progression_control_days_median']:.1f} | "
            f"{row['cumulative_dose_days_median']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Sensitivity Interpretation",
            "",
            "Spearman coefficients describe monotonic associations within this designed parameter sample.",
            "They are not causal effects, calibrated biological importance scores, or inferential evidence from patients.",
            "Nominal p-values are provided for auditability and are not corrected for multiple comparisons.",
            "",
            "## Boundaries",
            "",
            "- Every virtual tumor remains a deterministic two-clone abstraction.",
            "- Parameter ranges are scenario bounds, not a fitted population distribution.",
            "- The study does not include toxicity, pharmacokinetics, immune effects, spatial structure, or stochastic evolution.",
            "- A strategy can appear robust under these ranges and still fail under omitted biology or different ranges.",
            "- No quantum algorithm is used in this cohort study.",
            "",
        ]
    )
    return "\n".join(lines)


def write_cohort_artifacts(payload: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    """Write the complete cohort evidence package."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "experiment": destination / "evolution_cohort_experiment.json",
        "virtual_tumors": destination / "virtual_tumors.csv",
        "outcomes": destination / "virtual_tumor_outcomes.csv",
        "paired": destination / "paired_strategy_comparisons.csv",
        "strategy_summary": destination / "strategy_robustness_summary.csv",
        "sensitivity": destination / "parameter_sensitivity.csv",
        "report": destination / "EVOLUTION_COHORT_REPORT.md",
    }
    paths["experiment"].write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    pd.DataFrame(payload["virtual_tumors"]).to_csv(paths["virtual_tumors"], index=False)
    pd.DataFrame(payload["virtual_tumor_outcomes"]).to_csv(paths["outcomes"], index=False)
    pd.DataFrame(payload["paired_strategy_comparisons"]).to_csv(paths["paired"], index=False)
    pd.DataFrame(payload["strategy_robustness_summary"]).to_csv(
        paths["strategy_summary"],
        index=False,
    )
    pd.DataFrame(payload["parameter_sensitivity"]).to_csv(paths["sensitivity"], index=False)
    paths["report"].write_text(render_cohort_report(payload), encoding="utf-8")
    return paths


def run_evolution_cohort(
    config: EvolutionCohortConfig,
    *,
    write_output: bool = True,
) -> dict[str, Any]:
    """Run a matched deterministic virtual-tumor cohort study."""
    config.validate()
    base = EvolutionConfig.from_yaml(config.base_profile)
    if config.reference_strategy not in base.strategies:
        raise ValueError("reference_strategy is not present in the base evolution profile")
    if config.candidate_strategy not in base.strategies:
        raise ValueError("candidate_strategy is not present in the base evolution profile")

    tumors = generate_virtual_tumors(config)
    outcomes: list[Row] = []
    pairs: list[Row] = []
    for tumor in tumors:
        virtual_config = _build_virtual_config(base, config, tumor)
        simulation = run_evolution_simulation(virtual_config, write_output=False)
        tumor_outcomes = _outcome_rows(
            str(tumor["virtual_tumor_id"]),
            simulation,
            virtual_config.horizon_days,
        )
        outcomes.extend(tumor_outcomes)
        pairs.append(
            _paired_row(
                str(tumor["virtual_tumor_id"]),
                tumor_outcomes,
                config,
            )
        )

    strategy_summary = summarize_strategy_robustness(outcomes)
    paired_summary = summarize_paired_robustness(pairs)
    sensitivity = parameter_sensitivity_rows(tumors, pairs, config)
    fingerprint_config = config.to_dict()
    fingerprint_config.pop("output_dir", None)
    cohort_fingerprint = _fingerprint(
        {
            "config": fingerprint_config,
            "base_config": {
                key: value
                for key, value in base.to_dict().items()
                if key != "output_dir"
            },
            "virtual_tumors": tumors,
            "paired_strategy_comparisons": pairs,
        }
    )
    payload: dict[str, Any] = {
        "schema_version": "evolution-cohort-1.0",
        "generated_at": utc_now(),
        "research_use_only": True,
        "clinical_decision_support": False,
        "treatment_recommendation": False,
        "quantum_algorithm_used": False,
        "config": config.to_dict(),
        "base_evolution_config": base.to_dict(),
        "sampling_contract": {
            "method": "deterministic_latin_hypercube",
            "seed": config.seed,
            "policy_parameters_varied": False,
            "full_trajectories_retained": False,
            "event_time_censoring": "not_reached_values_capped_at_simulation_horizon_for_paired_summaries",
        },
        "virtual_tumors": tumors,
        "virtual_tumor_outcomes": outcomes,
        "paired_strategy_comparisons": pairs,
        "strategy_robustness_summary": strategy_summary,
        "paired_robustness_summary": paired_summary,
        "parameter_sensitivity": sensitivity,
        "cohort_fingerprint": cohort_fingerprint,
        "environment": environment_metadata(),
        "claim_boundary": (
            "Deterministic scenario cohort only; no patient distribution, clinical benefit, "
            "treatment recommendation, causal parameter importance, or quantum advantage is claimed."
        ),
    }
    if write_output:
        artifacts = write_cohort_artifacts(payload, config.output_dir)
        payload["artifacts"] = {name: str(path) for name, path in artifacts.items()}
    return payload


__all__ = [
    "EvolutionCohortConfig",
    "ParameterRange",
    "generate_virtual_tumors",
    "parameter_sensitivity_rows",
    "render_cohort_report",
    "run_evolution_cohort",
    "summarize_paired_robustness",
    "summarize_strategy_robustness",
    "write_cohort_artifacts",
]
