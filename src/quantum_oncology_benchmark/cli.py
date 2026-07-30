"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from collections.abc import Sequence

from .config import ExperimentConfig
from .data import load_csv_dataset
from .experiment import run_benchmark
from .gdc import GDCManifestQuery, fetch_manifest_metadata, write_manifest_artifacts
from .models.quantum_kernel import quantum_dependencies_available


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qob",
        description="Reproducible classical and quantum oncology benchmarks.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    benchmark = subparsers.add_parser("benchmark", help="run a benchmark")
    benchmark.add_argument("--config", type=Path, help="YAML configuration file")
    benchmark.add_argument("--dataset", choices=["breast-cancer", "csv"])
    benchmark.add_argument("--csv-path")
    benchmark.add_argument("--target-column")
    benchmark.add_argument("--positive-label")
    benchmark.add_argument("--model-set", choices=["classical", "quantum", "all"])
    benchmark.add_argument("--features", type=int)
    benchmark.add_argument("--test-size", type=float)
    benchmark.add_argument("--seed", type=int)
    benchmark.add_argument("--repeats", type=int)
    benchmark.add_argument("--max-samples", type=int)
    benchmark.add_argument("--quantum-reps", type=int)
    benchmark.add_argument("--quantum-shots", type=int)
    benchmark.add_argument("--output", dest="output_dir")

    doctor = subparsers.add_parser("doctor", help="check optional capabilities")
    doctor.add_argument("--json", action="store_true", dest="as_json")

    validate = subparsers.add_parser("validate-csv", help="validate a candidate CSV dataset")
    validate.add_argument("path", type=Path)
    validate.add_argument("--target-column", required=True)
    validate.add_argument("--positive-label")

    gdc = subparsers.add_parser(
        "gdc-manifest",
        help="query public GDC metadata and write a reproducible file manifest",
    )
    gdc.add_argument("--project", action="append", required=True, dest="projects")
    gdc.add_argument("--data-type", default="Gene Expression Quantification")
    gdc.add_argument("--workflow-type", default="STAR - Counts")
    gdc.add_argument("--access", choices=["open", "controlled"], default="open")
    gdc.add_argument("--sample-type", action="append", dest="sample_types")
    gdc.add_argument("--size", type=int, default=10_000)
    gdc.add_argument("--output", type=Path, required=True)

    return parser


def _benchmark_config(args: argparse.Namespace) -> ExperimentConfig:
    config = ExperimentConfig.from_yaml(args.config) if args.config else ExperimentConfig()
    replacements = {
        "dataset": args.dataset,
        "csv_path": args.csv_path,
        "target_column": args.target_column,
        "positive_label": args.positive_label,
        "model_set": args.model_set,
        "features": args.features,
        "test_size": args.test_size,
        "seed": args.seed,
        "repeats": args.repeats,
        "max_samples": args.max_samples,
        "quantum_reps": args.quantum_reps,
        "quantum_shots": args.quantum_shots,
        "output_dir": args.output_dir,
    }
    specified = {key: value for key, value in replacements.items() if value is not None}
    config = replace(config, **specified)
    config.validate()
    return config


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = _parser().parse_args(argv)
    try:
        if args.command == "doctor":
            status = {
                "core": True,
                "quantum": quantum_dependencies_available(),
                "quantum_install": "pip install -e '.[quantum]'",
                "research_use_only": True,
            }
            if args.as_json:
                print(json.dumps(status, indent=2))
            else:
                print(f"Core dependencies: {'available' if status['core'] else 'missing'}")
                print(f"Quantum dependencies: {'available' if status['quantum'] else 'missing'}")
                if not status["quantum"]:
                    print(f"Install with: {status['quantum_install']}")
            return 0

        if args.command == "validate-csv":
            dataset = load_csv_dataset(args.path, args.target_column, args.positive_label)
            print(
                json.dumps(
                    {
                        "valid": True,
                        "name": dataset.name,
                        "samples": len(dataset.target),
                        "features": dataset.features.shape[1],
                        "positive_class": dataset.positive_class,
                        "fingerprint": dataset.fingerprint,
                    },
                    indent=2,
                )
            )
            return 0

        if args.command == "gdc-manifest":
            query = GDCManifestQuery(
                projects=tuple(args.projects),
                data_type=args.data_type,
                workflow_type=args.workflow_type,
                access=args.access,
                sample_types=tuple(args.sample_types or ["Primary Tumor"]),
                size=args.size,
            )
            manifest, receipt = fetch_manifest_metadata(query)
            manifest_path, receipt_path = write_manifest_artifacts(manifest, receipt, args.output)
            print(f"Manifest rows: {len(manifest)}")
            print(f"Manifest: {manifest_path}")
            print(f"Query receipt: {receipt_path}")
            print("No data files were downloaded.")
            return 0

        if args.command == "benchmark":
            config = _benchmark_config(args)
            payload = run_benchmark(config)
            print("Benchmark complete.")
            print(f"Output directory: {config.output_dir}")
            for row in payload["summary"]:
                print(
                    f"- {row['model']}: balanced accuracy "
                    f"{row['balanced_accuracy_mean']:.3f}"
                )
            return 0
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 1
