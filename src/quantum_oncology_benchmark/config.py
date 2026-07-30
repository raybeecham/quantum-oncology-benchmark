"""Experiment configuration models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


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
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        """Load a configuration from YAML."""
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise ValueError("configuration root must be a mapping")
        config = cls(**payload)
        config.validate()
        return config
