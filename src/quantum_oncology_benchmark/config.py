"""Experiment configuration models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

_CLASSICAL_MODEL_NAMES = (
    "logistic_regression",
    "rbf_svm",
    "random_forest",
    "hist_gradient_boosting",
)


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Configuration for one benchmark execution."""

    dataset: str = "breast-cancer"
    csv_path: str | None = None
    target_column: str | None = None
    positive_label: str | int | float | None = None
    model_set: str = "classical"
    features: int = 4
    test_size: float = 0.25
    seed: int = 42
    repeats: int = 1
    max_samples: int | None = None
    quantum_reps: int = 2
    quantum_shots: int | None = None
    output_dir: str = "reports/latest"

    def validate(self) -> None:
        """Validate configuration values."""
        if self.dataset not in {"breast-cancer", "csv"}:
            raise ValueError("dataset must be 'breast-cancer' or 'csv'")
        if self.dataset == "csv" and not self.csv_path:
            raise ValueError("csv_path is required when dataset='csv'")
        if self.dataset == "csv" and not self.target_column:
            raise ValueError("target_column is required when dataset='csv'")
        if self.model_set not in {"classical", "quantum", "all"}:
            raise ValueError("model_set must be 'classical', 'quantum', or 'all'")
        if self.features < 1:
            raise ValueError("features must be at least 1")
        if not 0.05 <= self.test_size <= 0.5:
            raise ValueError("test_size must be between 0.05 and 0.5")
        if self.repeats < 1:
            raise ValueError("repeats must be at least 1")
        if self.max_samples is not None and self.max_samples < 20:
            raise ValueError("max_samples must be at least 20 when provided")
        if self.quantum_reps < 1:
            raise ValueError("quantum_reps must be at least 1")
        if self.quantum_shots is not None and self.quantum_shots < 1:
            raise ValueError("quantum_shots must be positive when provided")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mapping."""
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """Load a configuration from YAML."""
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise ValueError("configuration root must be a mapping")
        config = cls(**payload)
        config.validate()
        return config


@dataclass(frozen=True, slots=True)
class NestedCVConfig:
    """Configuration for classical nested cross-validation."""

    dataset: str = "breast-cancer"
    csv_path: str | None = None
    target_column: str | None = None
    positive_label: str | int | float | None = None
    features: int = 8
    seed: int = 42
    outer_folds: int = 5
    inner_folds: int = 3
    models: tuple[str, ...] = _CLASSICAL_MODEL_NAMES
    primary_metric: str = "balanced_accuracy"
    max_samples: int | None = None
    output_dir: str = "reports/nested-classical"

    def validate(self) -> None:
        """Validate nested cross-validation configuration values."""
        if self.dataset not in {"breast-cancer", "csv"}:
            raise ValueError("dataset must be 'breast-cancer' or 'csv'")
        if self.dataset == "csv" and not self.csv_path:
            raise ValueError("csv_path is required when dataset='csv'")
        if self.dataset == "csv" and not self.target_column:
            raise ValueError("target_column is required when dataset='csv'")
        if self.features < 1:
            raise ValueError("features must be at least 1")
        if self.outer_folds < 2:
            raise ValueError("outer_folds must be at least 2")
        if self.inner_folds < 2:
            raise ValueError("inner_folds must be at least 2")
        if not self.models:
            raise ValueError("models must contain at least one classical model")
        unsupported = sorted(set(self.models) - set(_CLASSICAL_MODEL_NAMES))
        if unsupported:
            raise ValueError(f"unsupported nested-cv models: {unsupported}")
        if len(set(self.models)) != len(self.models):
            raise ValueError("models must not contain duplicates")
        if self.primary_metric != "balanced_accuracy":
            raise ValueError("primary_metric is locked to 'balanced_accuracy'")
        if self.max_samples is not None and self.max_samples < 40:
            raise ValueError("max_samples must be at least 40 when provided")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mapping."""
        payload = asdict(self)
        payload["models"] = list(self.models)
        return payload

    @classmethod
    def from_yaml(cls, path: str | Path) -> NestedCVConfig:
        """Load a nested cross-validation configuration from YAML."""
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise ValueError("configuration root must be a mapping")
        if isinstance(payload.get("models"), list):
            payload["models"] = tuple(str(item) for item in payload["models"])
        config = cls(**payload)
        config.validate()
        return config
