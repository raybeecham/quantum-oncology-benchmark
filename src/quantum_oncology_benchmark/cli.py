"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from .config import ExperimentConfig, NestedCVConfig
from .data import load_csv_dataset
from .experiment import run_benchmark
from .gdc import GDCManifestQuery, fetch_manifest_metadata, write_manifest_artifacts
from .models.quantum_kernel import quantum_dependencies_available
from .nested_cv import run_nested_cv
from .profile_comparison import compare_nested_profiles


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

    nested = subparsers.add_parser(
        "nested-cv",
        help="run classical nested cross-validation with a locked primary endpoint",
    )
    nested.add_argument("--config", type=Path, help="YAML configuration file")
    nested.add_argument("--dataset", choices=["breast-cancer", "csv"])
    nested.add_argument("--csv-path")
    nested.add_argument("--target-column")
    nested.add_argument("--positive-label")
    nested.add_argument("--features", type=int)
    nested.add_argument("--seed", type=int)
    nested.add_argument("--outer-folds", type=int)
    nested.add_argument("--inner-folds", type=int)
    nested.add_argument(
        "--search-profile",
        choices=["reference-v1", "sensitivity-v1"],
        help="versioned classical hyperparameter search profile",
    )
    nested.add_argument("--calibration-bins", type=int)
    nested.add_argument(
        "--model",
        action="append",
        dest="models",
        choices=[
            "logistic_regression",
            "rbf_svm",
            "random_forest",
            "hist_gradient_boosting",
        ],
        help="classical model to include; repeat to select multiple models",
    )
    nested.add_argument("--max-samples", type=int)
    nested.add_argument("--output", dest="output_dir")

    compare_profiles = subparsers.add_parser(
        "compare-profiles",
        help="compare two completed nested-CV profiles and produce a protocol-freeze report",
    )
    compare_profiles.add_argument("--reference", type=Path, required=True)
    compare_profiles.add_argument("--candidate", type=Path, required=True)
    compare_profiles.add_argument("--output", type=Path, required=True)

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


def _nested_cv_config(args: argparse.Namespace) -> NestedCVConfig:
    config = NestedCVConfig.from_yaml(args.config) if args.config else NestedCVConfig()
    if args.dataset is not None:
        config = replace(config, dataset=str(args.dataset))
    if args.csv_path is not None:
        config = replace(config, csv_path=str(args.csv_path))
    if args.target_column is not None:
        config = replace(config, target_column=str(args.target_column))
    if args.positive_label is not None:
        config = replace(config, positive_label=args.positive_label)
    if args.features is not None:
        config = replace(config, features=int(args.features))
    if args.seed is not None:
        config = replace(config, seed=int(args.seed))
    if args.outer_folds is not None:
        config = replace(config, outer_folds=int(args.outer_folds))
    if args.inner_folds is not None:
        config = replace(config, inner_folds=int(args.inner_folds))
    if args.search_profile is not None:
        config = replace(config, search_profile=str(args.search_profile))
    if args.calibration_bins is not None:
        config = replace(config, calibration_bins=int(args.calibration_bins))
    if args.models is not None:
        config = replace(config, models=tuple(str(model) for model in args.models))
    if args.max_samples is not None:
        config = replace(config, max_samples=int(args.max_samples))
    if args.output_dir is not None:
        config = replace(config, output_dir=str(args.output_dir))
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

        if args.command == "nested-cv":
            nested_config = _nested_cv_config(args)
            payload = run_nested_cv(nested_config)
            print("Nested cross-validation complete.")
            print(f"Search profile: {nested_config.search_profile}")
            print(f"Output directory: {nested_config.output_dir}")
            for row in payload["summary"]:
                print(
                    f"- {row['model']}: outer-fold balanced accuracy "
                    f"{row['balanced_accuracy_mean']:.3f}"
                )
            return 0

        if args.command == "compare-profiles":
            payload = compare_nested_profiles(args.reference, args.candidate, args.output)
            print("Profile comparison complete.")
            print(f"Reference profile: {payload['reference_profile']}")
            print(f"Candidate profile: {payload['candidate_profile']}")
            print(f"Output directory: {args.output}")
            print(
                "Primary classical comparator: "
                f"{payload['protocol_freeze']['primary_classical_comparator']}"
            )
            return 0
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 1
