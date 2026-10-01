"""Offline surrogate validation and constrained Dirac integer-polynomial compilation."""

from __future__ import annotations

import hashlib
import json
import math
from itertools import combinations, product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from .treatment_config import ObjectiveWeights, TreatmentOptimizationConfig
from .treatment_optimization import build_optimization_cohorts, normalize_treatment_experiment
from .treatment_search import aggregate_loss, feasible_schedules, schedule_feasible, score_schedule

Row = dict[str, Any]
FloatArray = NDArray[np.float64]


def quadratic_features(schedules: list[tuple[int, ...]]) -> FloatArray:
    """Binary linear/pairwise monomials in a fixed, recorded order."""
    matrix = np.asarray(schedules, dtype=float)
    columns = [matrix[:, i] for i in range(matrix.shape[1])]
    columns.extend(matrix[:, i] * matrix[:, j] for i, j in combinations(range(matrix.shape[1]), 2))
    return np.column_stack(columns)


def compile_integer_polynomial(
    linear: list[float], pairwise: list[float], offset: float,
    config: TreatmentOptimizationConfig,
) -> Row:
    """Enforce binary exposure/run limits using integer slack and polynomial penalties."""
    count = config.intervals
    if len(linear) != count or len(pairwise) != count * (count - 1) // 2:
        raise ValueError("coefficient dimensions do not match schedule")
    if not all(math.isfinite(v) for v in [*linear, *pairwise, offset]):
        raise ValueError("coefficients must be finite")
    if config.max_consecutive_treated > 4:
        raise ValueError("offline compiler supports consecutive limits up to four (degree <= 5)")
    # Any binary surrogate value differs by at most the sum of absolute coefficients.
    # Each violated integer constraint contributes at least one. This penalty therefore
    # puts every infeasible assignment above every feasible assignment in exact arithmetic.
    penalty = sum(abs(v) for v in [*linear, *pairwise]) + 1.0
    slack = count + 1  # native integer in 0..max_treated_intervals
    budget = config.max_treated_intervals
    terms: dict[tuple[int, ...], float] = {}
    def add(indices: tuple[int, ...], value: float) -> None:
        key = tuple(sorted(indices))
        terms[key] = terms.get(key, 0.0) + value
    for i, value in enumerate(linear, start=1):
        add((i,), value + penalty * (1 - 2 * budget))
        add((i, slack), 2 * penalty)
    for (i, j), value in zip(combinations(range(1, count + 1), 2), pairwise, strict=True):
        add((i, j), value + 2 * penalty)
    add((slack,), -2 * penalty * budget)
    add((slack, slack), penalty)
    run_length = config.max_consecutive_treated + 1
    for start in range(1, count - run_length + 2):
        add(tuple(range(start, start + run_length)), penalty)
    terms = {idx: val for idx, val in terms.items() if val != 0.0}
    degree = max(len(idx) for idx in terms)
    scale = max(abs(value) for value in terms.values())
    data = [{"idx": [0] * (degree - len(idx)) + list(idx), "val": val / scale}
            for idx, val in sorted(terms.items())]
    return {"polynomial_file": {"file_name": "qob_treatment_integer_surrogate",
                                 "file_config": {"polynomial": {
                                     "num_variables": count + 1, "min_degree": 1,
                                     "max_degree": degree, "data": data}}},
            "job_template": {"job_type": "sample-hamiltonian-integer",
                             "polynomial_file_id": "REQUIRES_UPLOAD_FILE_ID",
                             "job_params": {"device_type": "dirac-3", "num_samples": 10,
                                            "relaxation_schedule": 1,
                                            "num_levels": [2] * count + [budget + 1]}},
            "coefficient_scale": scale, "energy_offset": offset + penalty * budget ** 2,
            "penalty": penalty, "schedule_variables": count, "slack_variable_index": slack,
            "constraint_contract": "P*(sum(x)+slack-budget)^2 + P*sum(forbidden consecutive products)",
            "source_interface": "user-supplied Dirac-3 Developer Beginner Guide notebook, integer example",
            "hardware_compatibility_verified": False, "hardware_submission_enabled": False}


def polynomial_energy(values: tuple[int, ...], compiled: Row) -> float:
    """Restore energy units and omitted constant for local verification."""
    terms = compiled["polynomial_file"]["file_config"]["polynomial"]["data"]
    energy = 0.0
    for term in terms:
        monomial = math.prod(values[index - 1] for index in term["idx"] if index > 0)
        energy += float(term["val"]) * monomial
    return energy * float(compiled["coefficient_scale"]) + float(compiled["energy_offset"])


def validate_integer_sample(values: list[float], config: TreatmentOptimizationConfig) -> tuple[int, ...]:
    """Strict decoding, without rounding or silently repairing infeasible hardware samples."""
    if len(values) != config.intervals + 1:
        raise ValueError("integer sample must include every schedule variable and slack")
    if any(isinstance(v, bool) or not math.isfinite(v) or abs(v - round(v)) > 1e-8 for v in values):
        raise ValueError("sample must contain finite integer values")
    integers = tuple(round(v) for v in values)
    schedule, slack = integers[:-1], integers[-1]
    if not schedule_feasible(schedule, config) or not 0 <= slack <= config.max_treated_intervals:
        raise ValueError("infeasible integer sample")
    if sum(schedule) + slack != config.max_treated_intervals:
        raise ValueError("integer slack does not satisfy exposure constraint")
    return schedule


def compile_dirac_treatment(
    experiment: str | Path, output_dir: str | Path, *, seed: int = 31415,
    training_fraction: float = 0.7, ridge_alpha: float = 1e-6,
) -> Row:
    """Fit a quadratic to prior oracle labels, validate separately, and export a dry-run payload."""
    source = json.loads(Path(experiment).read_text(encoding="utf-8"))
    if source.get("schema_version") != "treatment-optimization-1.0":
        raise ValueError("expected a treatment optimization experiment")
    if not 0.1 <= training_fraction <= 0.9 or not math.isfinite(ridge_alpha) or ridge_alpha <= 0:
        raise ValueError("invalid surrogate training configuration")
    source_hash = hashlib.sha256(json.dumps(normalize_treatment_experiment(source), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if source_hash != source["fingerprint"]:
        raise ValueError("source experiment fingerprint does not match scientific content")
    raw = dict(source["scientific"]["config"])
    weights = raw.pop("weights")
    config = TreatmentOptimizationConfig(weights=ObjectiveWeights(**weights), **raw)
    config.validate()
    if config.intervals > 12:
        raise ValueError("offline surrogate domain verification is capped at 12 intervals")
    oracle = next((r for r in source["scientific"]["search_results"] if r["method"] == "exact"), None)
    if oracle is None or oracle["stop_reason"] != "complete_feasible_domain":
        raise ValueError("compiler requires a completed exact oracle label table")
    labels = oracle["trace"]
    schedules = [tuple(int(x) for x in r["schedule"]) for r in labels]
    if len(set(schedules)) != len(schedules) or not all(schedule_feasible(x, config) for x in schedules):
        raise ValueError("oracle label schedules are duplicate or infeasible")
    if set(schedules) != set(feasible_schedules(config)):
        raise ValueError("oracle labels do not cover the complete feasible domain")
    targets = np.asarray([r["objective"] for r in labels], dtype=float)
    if not np.isfinite(targets).all():
        raise ValueError("nonfinite oracle labels")
    features = quadratic_features(schedules)
    feature_count = features.shape[1]
    training_count = int(len(labels) * training_fraction)
    if training_count < feature_count + 2 or len(labels) - training_count < 4:
        raise ValueError("insufficient schedules for a held-out quadratic surrogate fit")
    permutation = np.random.default_rng(seed).permutation(len(labels))
    training, validation = permutation[:training_count], permutation[training_count:]
    regressor = Ridge(alpha=ridge_alpha)
    regressor.fit(features[training], targets[training])
    predictions = regressor.predict(features)
    errors = predictions[validation] - targets[validation]
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    rank_result = spearmanr(targets[validation], predictions[validation])
    rho = float(rank_result.statistic)
    value_range = float(np.ptp(targets[validation]))
    metrics = {"validation_rmse": rmse, "validation_mae": float(np.mean(np.abs(errors))),
               "validation_normalized_rmse": None if value_range <= 0 else rmse / value_range,
               "validation_spearman_rho": rho if math.isfinite(rho) else None,
               "training_schedules": len(training), "validation_schedules": len(validation),
               "linear_and_pairwise_features": feature_count,
               "training_design_rank": int(np.linalg.matrix_rank(np.column_stack([np.ones(len(training)), features[training]]))),
               "validation_design": "disjoint schedule labels, same search tumor objective"}
    coefficients = regressor.coef_.tolist()
    compiled = compile_integer_polynomial(coefficients[:config.intervals], coefficients[config.intervals:],
                                          float(regressor.intercept_), config)
    # Verify minimum over all binary assignments and their analytically minimizing integer slack.
    candidates: list[tuple[float, tuple[int, ...]]] = []
    for schedule in product((0, 1), repeat=config.intervals):
        slack = max(0, config.max_treated_intervals - sum(schedule))
        values = (*schedule, slack)
        candidates.append((polynomial_energy(values, compiled), values))
    energy, values = min(candidates, key=lambda row: (row[0], row[1]))
    winner = validate_integer_sample(list(values), config)
    index = schedules.index(winner)
    if abs(energy - float(predictions[index])) > 1e-8:
        raise ValueError("encoded energy does not reproduce the surrogate value")
    true_regret = float(targets[index] - np.min(targets))
    metrics.update({"selected_schedule": "".join(map(str, winner)),
                    "selected_schedule_partition": "training" if index in training else "validation",
                    "selected_predicted_objective": float(predictions[index]),
                    "selected_true_objective": float(targets[index]), "selected_true_regret": true_regret,
                    "encoded_ground_energy": energy,
                    "encoded_ground_feasible": True})
    search_models, evaluation_models, _, design = build_optimization_cohorts(config)
    search_replay = aggregate_loss([score_schedule(winner, m, config.weights) for m in search_models], config.risk_aversion)
    if abs(float(search_replay["objective"]) - float(targets[index])) > 1e-8:
        raise ValueError("source label does not match current simulator replay")
    heldout = [score_schedule(winner, m, config.weights) for m in evaluation_models]
    label_rows = [{"schedule": "".join(map(str, x)), "partition": "training" if i in training else "validation",
                   "true_objective": float(targets[i]), "predicted_objective": float(predictions[i])}
                  for i, x in enumerate(schedules)]
    report: Row = {"schema_version": "dirac-treatment-surrogate-1.0", "quantum_algorithm_used": False,
                   "hardware_submission_enabled": False, "source_fingerprint": source["fingerprint"],
                   "config": config.to_dict(), "simulation_design": design, "fit": {"seed": seed, "training_fraction": training_fraction,
                                                         "ridge_alpha": ridge_alpha, "coefficients": coefficients,
                                                         "intercept": float(regressor.intercept_)},
                   "validation": metrics, "compiled": compiled, "label_predictions": label_rows,
                   "simulator_replay": search_replay, "evaluation_summary": aggregate_loss(heldout, config.risk_aversion),
                   "evaluation_outcomes": heldout,
                   "cost_boundary": {"label_source": "precomputed exhaustive oracle, not a budget-matched solver comparison",
                                     "source_label_tumor_simulations": oracle["tumor_simulations"],
                                     "additional_replay_tumor_simulations": len(search_models) + len(evaluation_models)},
                   "claim_boundary": "Approximate surrogate, offline integer encoding; no hardware execution, clinical claim or quantum advantage."}
    report["fingerprint"] = compiled_surrogate_fingerprint(report)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "dirac_surrogate_experiment.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    (destination / "dirac_polynomial_file.json").write_text(json.dumps(compiled["polynomial_file"], indent=2), encoding="utf-8")
    (destination / "dirac_job_template.json").write_text(json.dumps(compiled["job_template"], indent=2), encoding="utf-8")
    pd.DataFrame(label_rows).to_csv(destination / "surrogate_schedule_predictions.csv", index=False)
    return report


def compiled_surrogate_fingerprint(report: Row) -> str:
    scientific = {k: v for k, v in report.items() if k != "fingerprint"}
    scientific["config"] = {k: v for k, v in report["config"].items()
                            if k not in {"base_profile", "cohort_profile", "output_dir"}}
    return hashlib.sha256(json.dumps(scientific, sort_keys=True, allow_nan=False).encode()).hexdigest()


def import_dirac_treatment_samples(
    compiled_experiment: str | Path, results_file: str | Path, output_dir: str | Path,
    *, origin: str = "unverified_external_samples",
) -> Row:
    """Validate supplied samples and benchmark decoded schedules, without submitting jobs."""
    if origin not in {"unverified_external_samples", "local_test_samples"}:
        raise ValueError("sample origin must be external-unverified or local-test")
    compiled_source = json.loads(Path(compiled_experiment).read_text(encoding="utf-8"))
    if compiled_source.get("schema_version") != "dirac-treatment-surrogate-1.0":
        raise ValueError("expected a compiled treatment surrogate experiment")
    if compiled_surrogate_fingerprint(compiled_source) != compiled_source["fingerprint"]:
        raise ValueError("compiled surrogate fingerprint does not match content")
    raw = dict(compiled_source["config"])
    weights = raw.pop("weights")
    config = TreatmentOptimizationConfig(weights=ObjectiveWeights(**weights), **raw)
    config.validate()
    result = json.loads(Path(results_file).read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError("sample result root must be a mapping")
    if "status" in result and result["status"] != "COMPLETED":
        raise ValueError("only completed result files can be imported")
    samples = result.get("results", result)
    if not isinstance(samples, dict):
        raise ValueError("results must be a mapping")
    solutions, counts = samples.get("solutions"), samples.get("counts")
    if not isinstance(solutions, list) or not isinstance(counts, list) or not solutions:
        raise ValueError("solutions and counts must be nonempty lists")
    if len(solutions) != len(counts) or len(solutions) > 10000:
        raise ValueError("solution and count dimensions are invalid")
    if any(type(count) is not int or count <= 0 for count in counts) or sum(counts) > 100000:
        raise ValueError("counts must be positive integers with total <= 100000")
    energies = samples.get("energies")
    if energies is not None and (not isinstance(energies, list) or len(energies) != len(solutions)
                                or any(type(v) not in (int, float) or not math.isfinite(v) for v in energies)):
        raise ValueError("energies must be a finite numeric list matching solutions")
    search_models, evaluation_models, _, design = build_optimization_cohorts(config)
    if design != compiled_source["simulation_design"]:
        raise ValueError("current simulator scenario definition differs from compiled evidence")
    source_optimum = (compiled_source["validation"]["selected_true_objective"]
                      - compiled_source["validation"]["selected_true_regret"])
    rows: list[Row] = []
    cache: dict[tuple[int, ...], Row] = {}
    for index, (values, count) in enumerate(zip(solutions, counts, strict=True)):
        row: Row = {"sample_index": index, "count": count, "raw_solution": values,
                    "reported_energy": None if energies is None else energies[index]}
        try:
            if not isinstance(values, list) or any(type(v) not in (int, float) for v in values):
                raise ValueError("solution must be a numeric list")
            schedule = validate_integer_sample(values, config)
        except ValueError as exc:
            row.update({"valid": False, "rejection_reason": str(exc)})
        else:
            if schedule not in cache:
                search_outcomes = [score_schedule(schedule, m, config.weights) for m in search_models]
                evaluation_outcomes = [score_schedule(schedule, m, config.weights) for m in evaluation_models]
                cache[schedule] = {"search": aggregate_loss(search_outcomes, config.risk_aversion),
                                   "evaluation": aggregate_loss(evaluation_outcomes, config.risk_aversion)}
            score = cache[schedule]
            gap = max(0.0, score["search"]["objective"] - source_optimum)
            row.update({"valid": True, "schedule": "".join(map(str, schedule)),
                        "restored_polynomial_energy": polynomial_energy(tuple(round(v) for v in values), compiled_source["compiled"]),
                        "search_objective": score["search"]["objective"],
                        "evaluation_objective": score["evaluation"]["objective"],
                        "absolute_oracle_gap": gap, "oracle_hit": gap <= 1e-8})
        rows.append(row)
    total = sum(counts)
    valid_count = sum(r["count"] for r in rows if r["valid"])
    oracle_count = sum(r["count"] for r in rows if r.get("oracle_hit", False))
    report = {"schema_version": "dirac-treatment-sample-import-1.0", "execution_origin": origin,
              "verified_hardware_execution": False, "hardware_submission_performed": False,
              "compiled_fingerprint": compiled_source["fingerprint"], "samples": rows,
              "summary": {"sample_count": total, "valid_sample_count": valid_count,
                          "valid_fraction": valid_count / total,
                          "observed_oracle_hit_fraction_all_samples": oracle_count / total,
                          "observed_oracle_hit_fraction_valid_samples": None if valid_count == 0 else oracle_count / valid_count,
                          "unique_valid_schedules": len(cache),
                          "simulator_evaluations": len(cache) * (len(search_models) + len(evaluation_models))},
              "raw_result": result,
              "claim_boundary": "Imported counts describe this supplied sample file, not verified hardware provenance or an independent estimate of success probability. No repair, clinical or quantum-advantage claim."}
    report["fingerprint"] = hashlib.sha256(json.dumps(report, sort_keys=True, allow_nan=False).encode()).hexdigest()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "dirac_sample_import.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    pd.DataFrame(rows).to_csv(destination / "decoded_sample_outcomes.csv", index=False)
    return report
