"""Paired comparison of completed nested cross-validation profiles."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, cast

import pandas as pd
from scipy.stats import binomtest

from .reporting import utc_now


def _load_payload(directory: str | Path) -> dict[str, Any]:
    path = Path(directory) / "nested_experiment.json"
    if not path.exists():
        raise FileNotFoundError(f"nested experiment not found: {path}")
    payload = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    if not str(payload.get("schema_version", "")).startswith("nested-cv-"):
        raise ValueError(f"unsupported nested experiment schema in {path}")
    return payload


def _summary_by_model(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["model"]): row for row in payload["summary"]}


def _predictions_by_model(
    payload: dict[str, Any],
) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in payload["outer_fold_predictions"]:
        model = str(row["model"])
        sample_hash = str(row["sample_index_hash"])
        if sample_hash in grouped[model]:
            raise ValueError(f"duplicate out-of-fold prediction for {model}: {sample_hash}")
        grouped[model][sample_hash] = row
    return dict(grouped)


def _validate_compatibility(reference: dict[str, Any], candidate: dict[str, Any]) -> None:
    if reference["dataset"]["fingerprint"] != candidate["dataset"]["fingerprint"]:
        raise ValueError("profile datasets do not share the same fingerprint")

    required_equal = (
        "dataset",
        "features",
        "seed",
        "outer_folds",
        "inner_folds",
        "models",
        "primary_metric",
        "max_samples",
    )
    for field in required_equal:
        if reference["config"].get(field) != candidate["config"].get(field):
            raise ValueError(f"profile configuration mismatch: {field}")

    reference_predictions = _predictions_by_model(reference)
    candidate_predictions = _predictions_by_model(candidate)
    if set(reference_predictions) != set(candidate_predictions):
        raise ValueError("profiles do not contain the same models")

    for model in reference_predictions:
        reference_rows = reference_predictions[model]
        candidate_rows = candidate_predictions[model]
        if set(reference_rows) != set(candidate_rows):
            raise ValueError(f"profile sample hashes differ for model {model}")
        for sample_hash in reference_rows:
            left = reference_rows[sample_hash]
            right = candidate_rows[sample_hash]
            if int(left["y_true"]) != int(right["y_true"]):
                raise ValueError(f"profile truth labels differ for model {model}")
            if int(left["outer_fold"]) != int(right["outer_fold"]):
                raise ValueError(f"profile outer-fold assignment differs for model {model}")


def _exact_p_value(reference_only_correct: int, candidate_only_correct: int) -> float:
    discordant = reference_only_correct + candidate_only_correct
    if discordant == 0:
        return 1.0
    return float(
        binomtest(
            min(reference_only_correct, candidate_only_correct),
            n=discordant,
            p=0.5,
            alternative="two-sided",
        ).pvalue
    )


def _profile_prediction_rows(
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reference_predictions = _predictions_by_model(reference)
    candidate_predictions = _predictions_by_model(candidate)
    reference_summary = _summary_by_model(reference)
    candidate_summary = _summary_by_model(candidate)

    joined_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for model in sorted(reference_predictions):
        reference_only_correct = 0
        candidate_only_correct = 0
        both_correct = 0
        both_wrong = 0
        changed_predictions = 0

        for sample_hash in sorted(reference_predictions[model]):
            left = reference_predictions[model][sample_hash]
            right = candidate_predictions[model][sample_hash]
            truth = int(left["y_true"])
            reference_prediction = int(left["y_pred"])
            candidate_prediction = int(right["y_pred"])
            reference_correct = reference_prediction == truth
            candidate_correct = candidate_prediction == truth
            if reference_correct and candidate_correct:
                outcome = "both_correct"
                both_correct += 1
            elif reference_correct:
                outcome = "reference_only_correct"
                reference_only_correct += 1
            elif candidate_correct:
                outcome = "candidate_only_correct"
                candidate_only_correct += 1
            else:
                outcome = "both_wrong"
                both_wrong += 1
            if reference_prediction != candidate_prediction:
                changed_predictions += 1

            joined_rows.append(
                {
                    "model": model,
                    "outer_fold": int(left["outer_fold"]),
                    "sample_index_hash": sample_hash,
                    "y_true": truth,
                    "reference_y_pred": reference_prediction,
                    "candidate_y_pred": candidate_prediction,
                    "reference_y_score": float(left["y_score"]),
                    "candidate_y_score": float(right["y_score"]),
                    "paired_outcome": outcome,
                }
            )

        reference_metric = float(reference_summary[model]["balanced_accuracy_mean"])
        candidate_metric = float(candidate_summary[model]["balanced_accuracy_mean"])
        p_value = _exact_p_value(reference_only_correct, candidate_only_correct)
        direction = "no_direction"
        if reference_only_correct > candidate_only_correct:
            direction = "reference"
        elif candidate_only_correct > reference_only_correct:
            direction = "candidate"
        summary_rows.append(
            {
                "model": model,
                "samples": len(reference_predictions[model]),
                "reference_balanced_accuracy": reference_metric,
                "candidate_balanced_accuracy": candidate_metric,
                "balanced_accuracy_delta": candidate_metric - reference_metric,
                "both_correct": both_correct,
                "both_wrong": both_wrong,
                "reference_only_correct": reference_only_correct,
                "candidate_only_correct": candidate_only_correct,
                "discordant_predictions": reference_only_correct + candidate_only_correct,
                "changed_predictions": changed_predictions,
                "descriptive_direction": direction,
                "exact_mcnemar_p_value": p_value,
                "significant_at_0_05": bool(p_value < 0.05),
            }
        )
    return joined_rows, summary_rows


def _parameter_rows(payload: dict[str, Any], profile_role: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in payload["outer_fold_results"]:
        model = str(row["model"])
        params = row["best_params"]
        for parameter, value in params.items():
            grouped[(model, str(parameter))][json.dumps(value, sort_keys=True)] += 1
    for (model, parameter), counts in sorted(grouped.items()):
        for value, selected_folds in sorted(counts.items()):
            rows.append(
                {
                    "profile_role": profile_role,
                    "search_profile": payload["config"]["search_profile"],
                    "model": model,
                    "parameter": parameter,
                    "value": json.loads(value),
                    "selected_outer_folds": selected_folds,
                    "outer_folds": int(payload["config"]["outer_folds"]),
                }
            )
    return rows


def _search_spaces(payload: dict[str, Any]) -> dict[str, dict[str, list[Any]]]:
    return {
        str(model): {str(name): list(values) for name, values in grid.items()}
        for model, grid in payload["methodology"]["model_search_spaces"].items()
    }


def _changed_search_models(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    left = _search_spaces(reference)
    right = _search_spaces(candidate)
    changes: dict[str, dict[str, Any]] = {}
    for model in sorted(left):
        if left[model] != right[model]:
            changes[model] = {"reference": left[model], "candidate": right[model]}
    return changes


def _freeze_recommendations(
    comparison_rows: list[dict[str, Any]],
    changed_models: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    recommendations: list[dict[str, Any]] = []
    for row in comparison_rows:
        model = str(row["model"])
        delta = float(row["balanced_accuracy_delta"])
        changed = model in changed_models
        if not changed:
            decision = "retain_reference_control"
            rationale = "The model search space did not change and serves as a paired control."
        elif delta < 0:
            decision = "retain_reference_grid"
            rationale = (
                "The expanded candidate profile reduced mean outer-fold balanced accuracy; "
                "the added boundary values are not retained in the frozen reference grid."
            )
        elif delta > 0:
            decision = "retain_candidate_expansion_for_future_protocol"
            rationale = (
                "The bounded candidate expansion improved mean outer-fold balanced accuracy. "
                "The historical reference result remains immutable and no superiority claim is made."
            )
        else:
            decision = "retain_reference_grid"
            rationale = "The expanded search produced no change in the primary endpoint."
        recommendations.append(
            {
                "model": model,
                "search_space_changed": changed,
                "decision": decision,
                "balanced_accuracy_delta": delta,
                "exact_mcnemar_p_value": float(row["exact_mcnemar_p_value"]),
                "rationale": rationale,
            }
        )
    return recommendations


def _primary_comparator(summary_rows: list[dict[str, Any]]) -> str:
    ranked = sorted(
        summary_rows,
        key=lambda row: (
            min(
                float(row["reference_balanced_accuracy"]),
                float(row["candidate_balanced_accuracy"]),
            ),
            -abs(float(row["balanced_accuracy_delta"])),
        ),
        reverse=True,
    )
    return str(ranked[0]["model"])


def render_profile_comparison_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Nested Cross-Validation Profile Comparison Report",
        "",
        "> **Research-use-only warning:** This comparison does not establish clinical utility,",
        "> external validity, statistical superiority across independent cohorts, or quantum advantage.",
        "",
        "## Comparison Contract",
        "",
        f"- Reference profile: `{payload['reference_profile']}`",
        f"- Candidate profile: `{payload['candidate_profile']}`",
        f"- Dataset fingerprint: `{payload['dataset_fingerprint']}`",
        "- Outer partitions, sample hashes, truth labels, models, folds, seeds, and endpoint were verified identical.",
        "- Each sample contributes one paired out-of-fold prediction per model and profile.",
        "- Exact McNemar tests compare profile predictions for the same model on the same samples.",
        "",
        "## Paired Results",
        "",
        "| Model | Reference BA | Candidate BA | Delta | Ref only correct | Candidate only correct | Changed predictions | Exact p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["comparison_summary"]:
        lines.append(
            f"| {row['model']} | {float(row['reference_balanced_accuracy']):.6f} | "
            f"{float(row['candidate_balanced_accuracy']):.6f} | "
            f"{float(row['balanced_accuracy_delta']):+.6f} | "
            f"{row['reference_only_correct']} | {row['candidate_only_correct']} | "
            f"{row['changed_predictions']} | {float(row['exact_mcnemar_p_value']):.4f} |"
        )

    lines.extend(
        [
            "",
            "## Protocol Freeze Recommendation",
            "",
            f"**Primary classical comparator:** `{payload['protocol_freeze']['primary_classical_comparator']}`",
            "",
        ]
    )
    for row in payload["protocol_freeze"]["model_recommendations"]:
        lines.append(f"- **{row['model']}**, `{row['decision']}`: {row['rationale']}")

    lines.extend(
        [
            "",
            "## Future Quantum Boundary",
            "",
            "The classical protocol is frozen before introducing a resource-bounded quantum profile. ",
            "A future simulator or physical-QPU comparison must preserve the declared endpoint, outer partitions, ",
            "prediction-level evidence, resource records, and interpretation boundaries. Hardware execution must also ",
            "record backend identity, calibration snapshot, transpilation, shots, queue time, execution time, and cost.",
            "",
            "**Claim boundary:** No clinical utility or quantum advantage is claimed.",
            "",
        ]
    )
    return "\n".join(lines)


def compare_nested_profiles(
    reference_dir: str | Path,
    candidate_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Compare two completed nested-CV profiles and write protocol-freeze artifacts."""
    reference = _load_payload(reference_dir)
    candidate = _load_payload(candidate_dir)
    _validate_compatibility(reference, candidate)

    prediction_rows, comparison_rows = _profile_prediction_rows(reference, candidate)
    changed_models = _changed_search_models(reference, candidate)
    recommendations = _freeze_recommendations(comparison_rows, changed_models)
    payload: dict[str, Any] = {
        "schema_version": "profile-comparison-1.0",
        "generated_at": utc_now(),
        "research_use_only": True,
        "quantum_advantage_claimed": False,
        "reference_profile": reference["config"]["search_profile"],
        "candidate_profile": candidate["config"]["search_profile"],
        "reference_source": str(Path(reference_dir)),
        "candidate_source": str(Path(candidate_dir)),
        "dataset_fingerprint": reference["dataset"]["fingerprint"],
        "comparison_contract": {
            "same_dataset": True,
            "same_outer_partitions": True,
            "same_sample_hashes": True,
            "same_truth_labels": True,
            "same_models": True,
            "same_primary_endpoint": True,
            "paired_unit": "one_out_of_fold_prediction_per_sample_model_and_profile",
            "paired_test": "exact_mcnemar_binomial",
        },
        "changed_search_spaces": changed_models,
        "comparison_summary": comparison_rows,
        "paired_predictions": prediction_rows,
        "parameter_selection_summary": [
            *_parameter_rows(reference, "reference"),
            *_parameter_rows(candidate, "candidate"),
        ],
        "protocol_freeze": {
            "status": "classical_protocol_ready_for_freeze",
            "primary_classical_comparator": _primary_comparator(comparison_rows),
            "model_recommendations": recommendations,
            "quantum_profile_status": "deferred_resource_bounded_future_work",
            "claim_boundary": "No clinical utility or quantum advantage is claimed.",
        },
    }

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "profile_comparison.json"
    summary_path = destination / "cross_profile_summary.csv"
    predictions_path = destination / "cross_profile_predictions.csv"
    parameters_path = destination / "parameter_boundary_summary.csv"
    report_path = destination / "PROTOCOL_FREEZE_REPORT.md"

    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    pd.DataFrame(comparison_rows).to_csv(summary_path, index=False)
    pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)
    pd.DataFrame(payload["parameter_selection_summary"]).to_csv(parameters_path, index=False)
    report_path.write_text(render_profile_comparison_report(payload), encoding="utf-8")
    payload["artifacts"] = {
        "profile_comparison_json": str(json_path),
        "cross_profile_summary_csv": str(summary_path),
        "cross_profile_predictions_csv": str(predictions_path),
        "parameter_boundary_summary_csv": str(parameters_path),
        "report": str(report_path),
    }
    return payload
