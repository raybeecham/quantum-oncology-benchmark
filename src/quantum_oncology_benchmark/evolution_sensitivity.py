"""One-at-a-time sensitivity, separate from biological cohort sampling."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .evolution import run_evolution_simulation
from .evolution_config import EvolutionConfig
from .reporting import environment_metadata, utc_now


def run_evolution_sensitivity(
    profile: str | Path, *, output_dir: str | Path | None = None
) -> dict[str, Any]:
    """Run declared scenario sweeps without fitting or selecting a policy."""
    path = Path(profile)
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    allowed = {"protocol_version", "base_profile", "transition_rates_per_day",
               "adaptive_stop_fractions", "adaptive_restart_fractions", "output_dir"}
    if not isinstance(spec, dict) or set(spec) != allowed:
        raise ValueError("sensitivity profile must contain exactly the documented fields")
    if spec["protocol_version"] != "evolution-sensitivity-v1":
        raise ValueError("unsupported sensitivity protocol")
    values: dict[str, list[float]] = {}
    for name in ("transition_rates_per_day", "adaptive_stop_fractions",
                 "adaptive_restart_fractions"):
        raw = spec[name]
        if not isinstance(raw, list) or not raw:
            raise ValueError(f"{name} must be a nonempty list")
        if any(isinstance(v, bool) or not isinstance(v, (float, int)) for v in raw):
            raise ValueError(f"{name} must contain finite numbers")
        values[name] = [float(v) for v in raw]
        if any(not math.isfinite(v) for v in values[name]):
            raise ValueError(f"{name} must contain finite numbers")
        if len(set(values[name])) != len(raw):
            raise ValueError(f"{name} must contain unique values")
    if any(v < 0 for v in values["transition_rates_per_day"]):
        raise ValueError("transition rates must be nonnegative")
    base_path = Path(spec["base_profile"])
    if not base_path.is_absolute():
        base_path = path.parent / base_path
    base = EvolutionConfig.from_yaml(base_path)
    scenarios: list[tuple[str, EvolutionConfig]] = []
    for rate in values["transition_rates_per_day"]:
        scenarios.append(("acquired_transition", replace(
            base, sensitive_to_resistant_rate_per_day=rate)))
    for stop, restart in itertools.product(values["adaptive_stop_fractions"],
                                           values["adaptive_restart_fractions"]):
        scenario = replace(base, adaptive_stop_fraction=stop, adaptive_restart_fraction=restart)
        scenarios.append(("policy_threshold", scenario))
    if len(scenarios) > 100:
        raise ValueError("sensitivity study is limited to 100 scenarios")
    # Validate every scenario before any simulation or output is produced.
    for _, scenario in scenarios:
        scenario.validate()
    rows: list[dict[str, Any]] = []
    for index, (sweep, scenario) in enumerate(scenarios):
        scenario = replace(scenario, strategies=("continuous", "burden_adaptive"))
        result = run_evolution_simulation(scenario, write_output=False)
        reference = next(r for r in result["strategy_summary"] if r["strategy"] == "continuous")
        for summary in result["strategy_summary"]:
            rows.append({
                "scenario_id": f"scenario-{index + 1:03d}", "sweep": sweep,
                "transition_rate_per_day": scenario.sensitive_to_resistant_rate_per_day,
                "adaptive_stop_fraction": scenario.adaptive_stop_fraction,
                "adaptive_restart_fraction": scenario.adaptive_restart_fraction,
                **summary,
                "burden_auc_delta_vs_continuous": summary["tumor_burden_auc"] - reference["tumor_burden_auc"],
                "dose_days_delta_vs_continuous": summary["cumulative_dose_days"] - reference["cumulative_dose_days"],
            })
    base_inputs = base.to_dict()
    base_inputs.pop("output_dir", None)
    inputs = {k: v for k, v in spec.items() if k not in {"output_dir", "base_profile"}}
    scientific = {"inputs": inputs, "base_config": base_inputs, "outcomes": rows}
    fingerprint = hashlib.sha256(json.dumps(scientific, sort_keys=True).encode()).hexdigest()
    payload = {"schema_version": "evolution-sensitivity-1.0", "generated_at": utc_now(),
               "research_use_only": True, "quantum_algorithm_used": False,
               "scientific_inputs_and_results": scientific, "fingerprint": fingerprint,
               "environment": environment_metadata(),
               "claim_boundary": "Designed deterministic scenarios; no patient calibration or therapeutic benefit.",
               "design": "Transition sweep fixes policy; threshold sweep fixes all biology at base values."}
    destination = Path(output_dir if output_dir is not None else spec["output_dir"])
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "evolution_sensitivity_experiment.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    pd.DataFrame(rows).to_csv(destination / "sensitivity_outcomes.csv", index=False)
    return payload
