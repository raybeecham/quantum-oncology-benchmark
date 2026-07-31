"""Artifact generation for benchmark results."""

from __future__ import annotations

import json
import math
import platform
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, cast

import pandas as pd


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def package_version(name: str) -> str | None:
    """Return an installed package version when available."""
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def git_commit() -> str | None:
    """Return the current Git commit when executed in a checkout."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_metadata() -> dict[str, Any]:
    """Capture software provenance for reproducibility."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            "numpy": package_version("numpy"),
            "pandas": package_version("pandas"),
            "scikit-learn": package_version("scikit-learn"),
            "scipy": package_version("scipy"),
            "qiskit": package_version("qiskit"),
            "qiskit-machine-learning": package_version("qiskit-machine-learning"),
        },
        "git_commit": git_commit(),
    }


def _clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _canonicalize_reproducible_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _canonicalize_reproducible_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_canonicalize_reproducible_value(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return round(value, 15)
    return value


def normalize_experiment_for_reproducibility(payload: dict[str, Any]) -> dict[str, Any]:
    """Remove operational variance and canonicalize negligible float differences."""
    normalized: dict[str, Any] = deepcopy(payload)
    normalized.pop("generated_at", None)
    normalized.pop("artifacts", None)

    config = normalized.get("config")
    if isinstance(config, dict):
        config.pop("output_dir", None)

    runs = normalized.get("runs")
    if isinstance(runs, list):
        for row in runs:
            if isinstance(row, dict):
                row.pop("elapsed_seconds", None)

    summary = normalized.get("summary")
    if isinstance(summary, list):
        for row in summary:
            if isinstance(row, dict):
                row.pop("elapsed_seconds_mean", None)
                row.pop("elapsed_seconds_std", None)

    return cast(dict[str, Any], _canonicalize_reproducible_value(normalized))


def write_artifacts(payload: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    """Write machine-readable and human-readable benchmark artifacts."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    cleaned = _clean_json(payload)
    json_path = destination / "experiment.json"
    json_path.write_text(json.dumps(cleaned, indent=2, sort_keys=True), encoding="utf-8")

    summary = pd.DataFrame(payload["summary"])
    csv_path = destination / "summary.csv"
    summary.to_csv(csv_path, index=False)

    runs = pd.DataFrame(payload["runs"])
    runs_path = destination / "runs.csv"
    runs.to_csv(runs_path, index=False)

    pairwise_rows = payload.get("statistical_analysis", {}).get("pairwise_comparisons", [])
    pairwise_path = destination / "pairwise_comparisons.csv"
    pd.DataFrame(pairwise_rows).to_csv(pairwise_path, index=False)

    report_path = destination / "REPORT.md"
    report_path.write_text(render_markdown_report(payload), encoding="utf-8")

    return {
        "json": json_path,
        "summary_csv": csv_path,
        "runs_csv": runs_path,
        "pairwise_csv": pairwise_path,
        "report": report_path,
    }


def _format_interval(row: dict[str, Any], metric: str) -> str:
    mean = float(row[f"{metric}_mean"])
    low = row.get(f"{metric}_ci_low")
    high = row.get(f"{metric}_ci_high")
    if low is None or high is None:
        return f"{mean:.3f} (CI not estimated)"
    return f"{mean:.3f} [{float(low):.3f}, {float(high):.3f}]"


def render_markdown_report(payload: dict[str, Any]) -> str:
    """Render a concise research report from benchmark results."""
    dataset = payload["dataset"]
    config = payload["config"]
    summary = payload["summary"]
    selected = payload.get("selected_features", [])
    statistical = payload.get("statistical_analysis", {})
    confidence = statistical.get("confidence_intervals", {})
    pairwise_summary = statistical.get("pairwise_summary", [])
    evidence = payload.get("evidence_statement", {})

    lines = [
        "# Quantum Oncology Benchmark Report",
        "",
        "> **Research-use-only warning:** This software and its outputs are not a medical device,",
        "> diagnostic system, treatment recommendation, or evidence of quantum advantage.",
        "",
        "## Experiment",
        "",
        f"- Generated: `{payload['generated_at']}`",
        f"- Dataset: **{dataset['name']}**",
        f"- Dataset fingerprint: `{dataset['fingerprint']}`",
        f"- Samples used: **{dataset['samples_used']}**",
        f"- Positive class: **{dataset['positive_class']}**",
        f"- Repeated holdouts: **{config['repeats']}**",
        f"- Selected features per split: **{config['features']}**",
        "",
        "## Aggregate Results",
        "",
        "Intervals are bootstrap confidence intervals over repeat-level metric means.",
        "",
        "| Model | Balanced accuracy (95% CI) | Sensitivity | Specificity | F1 | ROC AUC (95% CI) | Time (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['model']} | {_format_interval(row, 'balanced_accuracy')} | "
            f"{float(row['sensitivity_mean']):.3f} ± {float(row['sensitivity_std']):.3f} | "
            f"{float(row['specificity_mean']):.3f} ± {float(row['specificity_std']):.3f} | "
            f"{float(row['f1_mean']):.3f} ± {float(row['f1_std']):.3f} | "
            f"{_format_interval(row, 'roc_auc')} | "
            f"{float(row['elapsed_seconds_mean']):.3f} |"
        )

    lines.extend(
        [
            "",
            "## Statistical Evaluation",
            "",
            f"- Confidence level: **{float(confidence.get('confidence_level', 0.95)):.0%}**",
            f"- Bootstrap resamples: **{confidence.get('bootstrap_resamples', 'not recorded')}**",
            f"- Resampling unit: `{confidence.get('resampling_unit', 'not recorded')}`",
            "- Paired classifier test: exact McNemar test within each shared test partition.",
            "- No pooled repeated-holdout p-value is reported because observations may be reused.",
            "- These intervals and tests are descriptive benchmark evidence, not external validation.",
        ]
    )

    if pairwise_summary:
        lines.extend(
            [
                "",
                "### Pairwise Comparison Summary",
                "",
                "| Model A | Model B | Significant repeats | A favored | B favored | Minimum p |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in pairwise_summary:
            lines.append(
                f"| {row['model_a']} | {row['model_b']} | "
                f"{row['significant_repeats']}/{row['repeats']} | "
                f"{row['model_a_favored_repeats']} | {row['model_b_favored_repeats']} | "
                f"{float(row['minimum_exact_p_value']):.4f} |"
            )

    lines.extend(
        [
            "",
            "## Evidence Statement",
            "",
            str(evidence.get("statement", "No evidence statement was generated.")),
            "",
            f"**Claim boundary:** {evidence.get('claim_boundary', 'No claim boundary recorded.')}",
            "",
            "## Feature Selection",
            "",
            "Feature selection was fitted only on each training partition. The first split selected:",
            "",
        ]
    )
    lines.extend([f"- `{feature}`" for feature in selected])

    quantum_resources = payload.get("quantum_resources")
    if quantum_resources:
        lines.extend(
            [
                "",
                "## Quantum Resource Record",
                "",
                f"- Execution mode: `{quantum_resources['execution_mode']}`",
                f"- Physical QPU used: **{quantum_resources['physical_qpu']}**",
                f"- Logical qubits: **{quantum_resources['qubits']}**",
                f"- Circuit depth: **{quantum_resources['circuit_depth']}**",
                f"- Feature-map repetitions: **{quantum_resources['feature_map_repetitions']}**",
                f"- Shots: **{quantum_resources['shots']}**",
            ]
        )

    install_target = ".[quantum]" if quantum_resources else ".[dev]"
    command_parts = [
        "qob benchmark",
        f"--dataset {config['dataset']}",
        f"--model-set {config['model_set']}",
        f"--features {config['features']}",
        f"--test-size {config['test_size']}",
        f"--seed {config['seed']}",
        f"--repeats {config['repeats']}",
    ]
    if config.get("max_samples") is not None:
        command_parts.append(f"--max-samples {config['max_samples']}")
    if quantum_resources:
        command_parts.append(f"--quantum-reps {config['quantum_reps']}")
        if config.get("quantum_shots") is not None:
            command_parts.append(f"--quantum-shots {config['quantum_shots']}")
    command_parts.append(f"--output {config['output_dir']}")

    lines.extend(
        [
            "",
            "## Interpretation Boundaries",
            "",
            "- A quantum model appearing in this table does not establish quantum advantage.",
            "- The built-in dataset is a small educational benchmark, not a prospective clinical cohort.",
            "- Hyperparameter search is intentionally limited in version 0.1.0.",
            "- Any scientific claim requires external cohorts and independent replication.",
            "",
            "## Reproduction",
            "",
            "```bash",
            f"python -m pip install -e '{install_target}'",
            " \\\n  ".join(command_parts),
            "```",
            "",
        ]
    )
    return "\n".join(lines)
