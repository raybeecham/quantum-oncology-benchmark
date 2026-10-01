"""Numerical resolution and preference sensitivity for already frozen schedule winners.

Run from the repository root. Optional plots require matplotlib, not the core engine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

from quantum_oncology_benchmark.treatment_config import (
    ObjectiveWeights,
    TreatmentOptimizationConfig,
)
from quantum_oncology_benchmark.treatment_optimization import build_optimization_cohorts
from quantum_oncology_benchmark.treatment_search import aggregate_loss, score_schedule


def review(experiment: Path, output: Path, plots: bool = False) -> dict[str, Any]:
    source = json.loads(experiment.read_text())
    raw = dict(source["scientific"]["config"])
    weights = raw.pop("weights")
    config = TreatmentOptimizationConfig(weights=ObjectiveWeights(**weights), **raw)
    search_models, evaluation_models, _, _ = build_optimization_cohorts(config)
    runs = source["scientific"]["search_results"]
    unique = {r["schedule"]: r for r in runs}
    resolution: list[dict[str, Any]] = []
    preferences: list[dict[str, Any]] = []
    # Frozen winners only. No schedule is reoptimized or selected using these diagnostics.
    for bits, run in unique.items():
        schedule = tuple(int(x) for x in bits)
        for partition, models in (("search", search_models), ("evaluation", evaluation_models)):
            coarse = [score_schedule(schedule, m, config.weights) for m in models]
            fine = [score_schedule(schedule, replace(m, time_step_days=m.time_step_days / 2), config.weights)
                    for m in models]
            coarse_loss = float(aggregate_loss(coarse, config.risk_aversion)["objective"])
            fine_loss = float(aggregate_loss(fine, config.risk_aversion)["objective"])
            resolution.append({"schedule": bits, "partition": partition,
                               "base_objective": coarse_loss, "half_step_objective": fine_loss,
                               "objective_delta": fine_loss - coarse_loss,
                               "max_absolute_terminal_burden_delta": max(abs(a["final_total_burden"] - b["final_total_burden"]) for a, b in zip(coarse, fine, strict=True)),
                               "max_absolute_auc_delta": max(abs(a["tumor_burden_auc"] - b["tumor_burden_auc"]) for a, b in zip(coarse, fine, strict=True))})
        for resistant_weight, dose_weight in product((0.5, 1.0, 2.0), (0.1, 0.25, 0.5)):
            outcomes = [{"loss": row["normalized_burden_auc"] + resistant_weight * row["normalized_terminal_resistant"] + dose_weight * row["normalized_dose"]}
                        for row in run["tumor_outcomes"]]
            preferences.append({"schedule": bits, "burden_weight": 1.0,
                                "resistant_weight": resistant_weight, "dose_weight": dose_weight,
                                **aggregate_loss(outcomes, config.risk_aversion)})
    report: dict[str, Any] = {"schema_version": "treatment-review-1.0",
                              "source_fingerprint": source["fingerprint"],
                              "resolution_checks": resolution, "preference_checks": preferences,
                              "scope": "Frozen winners only; no reoptimization, evaluation-based selection, or verification of the entire oracle ranking at finer resolution."}
    report["fingerprint"] = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    (output / "treatment_review.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    pd.DataFrame(resolution).to_csv(output / "numerical_resolution.csv", index=False)
    pd.DataFrame(preferences).to_csv(output / "weight_sensitivity.csv", index=False)
    if plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        summaries = pd.DataFrame(source["scientific"]["evaluation_summary"])
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        names = [f"{r['method']} {r['repeat']}" for r in runs]
        axes[0].barh(names, [r["objective"] for r in runs], color="#7c3aed")
        axes[0].set_xlabel("Search objective (lower is better)")
        axes[1].barh(names, summaries["objective"].tolist(), color="#0891b2")
        axes[1].set_xlabel("Evaluation objective (lower is better)")
        for axis in axes:
            axis.invert_yaxis()
            axis.grid(axis="x", alpha=0.25)
            axis.set_axisbelow(True)
        fig.suptitle(config.profile_name)
        fig.tight_layout()
        fig.savefig(output / "search_and_evaluation.png", dpi=180)
        plt.close(fig)
        fig, axis = plt.subplots(figsize=(10, 4))
        for run in runs:
            if run["method"] != "exact":
                axis.plot([r["unique_evaluation"] for r in run["trace"]],
                          [r["best_objective"] for r in run["trace"]],
                          label=f"{run['method']} {run['repeat']}")
        if config.exact_search:
            oracle = next(r for r in runs if r["method"] == "exact")
            axis.axhline(oracle["objective"], color="black", linestyle="--", label="search oracle")
        axis.set_xlabel("Unique feasible schedule evaluations")
        axis.set_ylabel("Best search objective (lower is better)")
        axis.grid(alpha=0.2)
        axis.legend(ncol=2)
        fig.tight_layout()
        fig.savefig(output / "search_convergence.png", dpi=180)
        plt.close(fig)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    result = review(args.experiment, args.output, args.plots)
    print(f"Treatment review complete: {result['fingerprint']}")
