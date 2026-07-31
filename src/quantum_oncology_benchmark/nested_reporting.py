"""Artifacts and reports for classical nested cross-validation."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pandas as pd

_REPRODUCIBILITY_FLOAT_DECIMALS = 12


def _clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _canonicalize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_canonicalize(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return round(value, _REPRODUCIBILITY_FLOAT_DECIMALS)
    return value


def normalize_nested_experiment_for_reproducibility(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Remove operational nested-CV variance for scientific equality checks."""
    normalized: dict[str, Any] = deepcopy(payload)
    normalized.pop("generated_at", None)
    normalized.pop("artifacts", None)

    config = normalized.get("config")
    if isinstance(config, dict):
        config.pop("output_dir", None)

    outer_rows = normalized.get("outer_fold_results")
    if isinstance(outer_rows, list):
        for row in outer_rows:
            if isinstance(row, dict):
                row.pop("fit_seconds", None)

    summary = normalized.get("summary")
    if isinstance(summary, list):
        for row in summary:
            if isinstance(row, dict):
                row.pop("fit_seconds_mean", None)
                row.pop("fit_seconds_std", None)

    return cast(dict[str, Any], _canonicalize(normalized))


def _json_column(value: Any) -> str:
    return json.dumps(_clean_json(value), sort_keys=True, separators=(",", ":"))


def _flatten_rows(
    rows: list[dict[str, Any]],
    structured_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for row in rows:
        output = dict(row)
        for field in structured_fields:
            if field in output:
                output[field] = _json_column(output[field])
        flattened.append(output)
    return flattened


def _format_interval(row: dict[str, Any], metric: str) -> str:
    mean = float(row[f"{metric}_mean"])
    low = row.get(f"{metric}_ci_low")
    high = row.get(f"{metric}_ci_high")
    if low is None or high is None:
        return f"{mean:.3f} (CI not estimated)"
    return f"{mean:.3f} [{float(low):.3f}, {float(high):.3f}]"


def _reproduction_config(search_profile: str) -> str:
    if search_profile == "sensitivity-v1":
        return "configs/nested-classical-sensitivity.yaml"
    return "configs/nested-classical.yaml"


def render_nested_cv_report(payload: dict[str, Any]) -> str:
    """Render a bounded Markdown report for nested cross-validation."""
    config = payload["config"]
    dataset = payload["dataset"]
    methodology = payload["methodology"]
    summary = payload["summary"]
    pairwise = payload.get("pairwise_summary", [])
    calibration = payload.get("calibration_summary", [])
    evidence = payload["evidence_statement"]

    lines = [
        "# Quantum Oncology Benchmark Nested Cross-Validation Report",
        "",
        "> **Research-use-only warning:** This software and its outputs are not a medical device,",
        "> diagnostic system, treatment recommendation, or evidence of quantum advantage.",
        "",
        "## Experiment",
        "",
        f"- Generated: `{payload['generated_at']}`",
        f"- Evaluation mode: `{payload['evaluation_mode']}`",
        f"- Search profile: **{config['search_profile']}**",
        f"- Dataset: **{dataset['name']}**",
        f"- Dataset fingerprint: `{dataset['fingerprint']}`",
        f"- Samples used: **{dataset['samples_used']}**",
        f"- Positive class: **{dataset['positive_class']}**",
        f"- Outer folds: **{config['outer_folds']}**",
        f"- Inner folds: **{config['inner_folds']}**",
        f"- Selected features: **{config['features']}**",
        f"- Models: **{', '.join(config['models'])}**",
        "",
        "## Locked Evaluation Protocol",
        "",
        f"- Primary endpoint: **{methodology['primary_metric']}**",
        f"- Primary endpoint locked: **{methodology['primary_endpoint_locked']}**",
        f"- Search profile: `{methodology['search_profile']}`",
        f"- Preprocessing scope: `{methodology['preprocessing_scope']}`",
        f"- Outer test usage: `{methodology['outer_test_usage']}`",
        "- Imputation, scaling, feature selection, and model fitting occur inside each search pipeline.",
        "- The outer test fold is evaluated once after inner-loop model selection and refitting.",
        "- No pooled p-value is reported across outer folds.",
        "",
        "## Aggregate Outer-Fold Results",
        "",
        "Intervals are bootstrap confidence intervals over outer-fold metric values.",
        "",
        "| Model | Balanced accuracy (95% CI) | Sensitivity | Specificity | F1 | ROC AUC (95% CI) | Search + refit time (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['model']} | {_format_interval(row, 'balanced_accuracy')} | "
            f"{float(row['sensitivity_mean']):.3f} ± {float(row['sensitivity_std']):.3f} | "
            f"{float(row['specificity_mean']):.3f} ± {float(row['specificity_std']):.3f} | "
            f"{float(row['f1_mean']):.3f} ± {float(row['f1_std']):.3f} | "
            f"{_format_interval(row, 'roc_auc')} | "
            f"{float(row['fit_seconds_mean']):.3f} |"
        )

    lines.extend(
        [
            "",
            "## Outer-Fold Model Selection",
            "",
            "| Fold | Model | Inner best balanced accuracy | Outer balanced accuracy | Candidates | Selected parameters |",
            "|---:|---|---:|---:|---:|---|",
        ]
    )
    for row in payload["outer_fold_results"]:
        lines.append(
            f"| {row['outer_fold']} | {row['model']} | "
            f"{float(row['inner_best_mean']):.3f} ± {float(row['inner_best_std']):.3f} | "
            f"{float(row['balanced_accuracy']):.3f} | {row['candidate_count']} | "
            f"`{_json_column(row['best_params'])}` |"
        )

    if calibration:
        lines.extend(
            [
                "",
                "## Out-of-Fold Calibration Diagnostics",
                "",
                "Diagnostics pool one untouched outer-fold prediction per sample and model.",
                "Reliability curve points are written to `calibration_bins.csv`.",
                "",
                "| Model | ECE | MCE | Calibration-in-the-large | Brier score | Log loss | False negatives | False positives |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in calibration:
            lines.append(
                f"| {row['model']} | "
                f"{float(row['expected_calibration_error']):.4f} | "
                f"{float(row['maximum_calibration_error']):.4f} | "
                f"{float(row['calibration_in_the_large']):.4f} | "
                f"{float(row['pooled_out_of_fold_brier_score']):.4f} | "
                f"{float(row['pooled_out_of_fold_log_loss']):.4f} | "
                f"{row['false_negative_count']} | {row['false_positive_count']} |"
            )
        lines.extend(
            [
                "",
                "Expected calibration error and maximum calibration error depend on the configured uniform probability bins. They are descriptive diagnostics, not clinical validation.",
            ]
        )

    if pairwise:
        lines.extend(
            [
                "",
                "## Pairwise Outer-Fold Comparison Summary",
                "",
                "| Model A | Model B | Significant folds | A more correct | B more correct | Equal | Minimum p |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in pairwise:
            lines.append(
                f"| {row['model_a']} | {row['model_b']} | "
                f"{row['significant_outer_folds']}/{row['outer_folds']} | "
                f"{row['model_a_more_correct_outer_folds']} | "
                f"{row['model_b_more_correct_outer_folds']} | "
                f"{row['equal_correctness_outer_folds']} | "
                f"{float(row['minimum_exact_p_value']):.4f} |"
            )

    lines.extend(
        [
            "",
            "## Evidence Statement",
            "",
            str(evidence["statement"]),
            "",
            f"**Claim boundary:** {evidence['claim_boundary']}",
            "",
            "## Interpretation Boundaries",
            "",
            "- Nested cross-validation estimates performance of the model-selection procedure on this dataset.",
            "- Outer folds share training observations and are not independent external replications.",
            "- Fold-level confidence intervals are descriptive, especially with five outer folds.",
            "- Calibration bins and error rows summarize out-of-fold predictions but do not establish clinical reliability.",
            "- The built-in dataset is an educational benchmark, not a prospective clinical cohort.",
            "- This classical-only mode does not evaluate quantum advantage.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "python -m pip install -e '.[dev]'",
            f"qob nested-cv --config {_reproduction_config(config['search_profile'])}",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def write_nested_artifacts(
    payload: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write the complete nested cross-validation artifact package."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    experiment_path = destination / "nested_experiment.json"
    experiment_path.write_text(
        json.dumps(_clean_json(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    outer_path = destination / "outer_fold_results.csv"
    pd.DataFrame(
        _flatten_rows(
            payload["outer_fold_results"],
            (
                "training_class_counts",
                "test_class_counts",
                "selected_features",
                "best_params",
            ),
        )
    ).to_csv(outer_path, index=False)

    predictions_path = destination / "outer_fold_predictions.csv"
    pd.DataFrame(payload["outer_fold_predictions"]).to_csv(predictions_path, index=False)

    inner_path = destination / "inner_search_results.csv"
    pd.DataFrame(_flatten_rows(payload["inner_search_results"], ("params",))).to_csv(
        inner_path,
        index=False,
    )

    summary_path = destination / "nested_summary.csv"
    pd.DataFrame(payload["summary"]).to_csv(summary_path, index=False)

    pairwise_path = destination / "nested_pairwise_comparisons.csv"
    pd.DataFrame(payload["pairwise_comparisons"]).to_csv(pairwise_path, index=False)

    calibration_summary_path = destination / "calibration_summary.csv"
    pd.DataFrame(payload["calibration_summary"]).to_csv(
        calibration_summary_path,
        index=False,
    )

    calibration_bins_path = destination / "calibration_bins.csv"
    pd.DataFrame(payload["calibration_bins"]).to_csv(calibration_bins_path, index=False)

    distribution_path = destination / "probability_distribution.csv"
    pd.DataFrame(payload["probability_distribution"]).to_csv(
        distribution_path,
        index=False,
    )

    errors_path = destination / "classification_errors.csv"
    pd.DataFrame(payload["classification_errors"]).to_csv(errors_path, index=False)

    report_path = destination / "NESTED_CV_REPORT.md"
    report_path.write_text(render_nested_cv_report(payload), encoding="utf-8")

    return {
        "experiment_json": experiment_path,
        "outer_fold_results_csv": outer_path,
        "outer_fold_predictions_csv": predictions_path,
        "inner_search_results_csv": inner_path,
        "nested_summary_csv": summary_path,
        "nested_pairwise_comparisons_csv": pairwise_path,
        "calibration_summary_csv": calibration_summary_path,
        "calibration_bins_csv": calibration_bins_path,
        "probability_distribution_csv": distribution_path,
        "classification_errors_csv": errors_path,
        "report": report_path,
    }
