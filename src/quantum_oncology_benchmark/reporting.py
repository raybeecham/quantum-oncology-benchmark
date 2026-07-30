"""Artifact generation for benchmark results."""

from __future__ import annotations

import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

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

    report_path = destination / "REPORT.md"
    report_path.write_text(render_markdown_report(payload), encoding="utf-8")

    return {
        "json": json_path,
        "summary_csv": csv_path,
        "runs_csv": runs_path,
        "report": report_path,
    }


def render_markdown_report(payload: dict[str, Any]) -> str:
    """Render a concise research report from benchmark results."""
    dataset = payload["dataset"]
    config = payload["config"]
    summary = payload["summary"]
    selected = payload.get("selected_features", [])

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
        "| Model | Balanced accuracy | Sensitivity | Specificity | F1 | ROC AUC | Time (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {model} | {balanced_accuracy_mean:.3f} ± {balanced_accuracy_std:.3f} | "
            "{sensitivity_mean:.3f} ± {sensitivity_std:.3f} | "
            "{specificity_mean:.3f} ± {specificity_std:.3f} | "
            "{f1_mean:.3f} ± {f1_std:.3f} | "
            "{roc_auc_mean:.3f} ± {roc_auc_std:.3f} | {elapsed_seconds_mean:.3f} |".format(**row)
        )

    lines.extend(
        [
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
            "- Any scientific claim requires repeated validation, external cohorts, uncertainty analysis,",
            "  and independent replication.",
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
