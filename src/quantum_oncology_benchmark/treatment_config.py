"""Versioned computational treatment-scheduling protocol, without clinical calibration."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class ObjectiveWeights:
    """Weights for dimensionless outcome components; these encode research preferences."""

    burden_auc: float = 1.0
    terminal_resistant: float = 1.0
    dose: float = 0.25

    def validate(self) -> None:
        values = (self.burden_auc, self.terminal_resistant, self.dose)
        if any(not math.isfinite(v) or v < 0 for v in values) or sum(values) <= 0:
            raise ValueError("objective weights must be finite, nonnegative and not all zero")


@dataclass(frozen=True, slots=True)
class TreatmentOptimizationConfig:
    """Bounded binary schedule search with a separate evaluation cohort."""

    protocol_version: str = "treatment-optimization-v1"
    profile_name: str = "binary-schedule-reference-v1"
    base_profile: str = "configs/evolution-two-clone.yaml"
    cohort_profile: str | None = None
    intervals: int = 12
    max_treated_intervals: int = 9
    max_consecutive_treated: int = 4
    search_tumors: int = 1
    evaluation_tumors: int = 1
    cohort_seed: int = 1729
    repeats: int = 3
    seed: int = 42
    evaluation_budget: int = 128
    exact_search: bool = True
    risk_aversion: float = 0.0
    weights: ObjectiveWeights = ObjectiveWeights()
    annealing_start_temperature: float = 0.25
    annealing_end_temperature: float = 0.001
    output_dir: str = "reports/treatment-optimization"

    def validate(self) -> None:
        if self.protocol_version != "treatment-optimization-v1" or not self.profile_name:
            raise ValueError("unsupported optimization protocol or missing profile name")
        integer_bounds = {
            "intervals": (2, 20), "search_tumors": (1, 128),
            "evaluation_tumors": (1, 128), "repeats": (1, 20),
            "evaluation_budget": (1, 100_000), "max_treated_intervals": (1, self.intervals),
            "max_consecutive_treated": (1, self.intervals),
        }
        for name, (lower, upper) in integer_bounds.items():
            value = getattr(self, name)
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"{name} must be an integer between {lower} and {upper}")
        for name in ("seed", "cohort_seed"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if type(self.exact_search) is not bool:
            raise ValueError("exact_search must be boolean")
        if self.exact_search and self.intervals > 12:
            raise ValueError("exact search is capped at 12 intervals")
        if not self.cohort_profile and (self.search_tumors != 1 or self.evaluation_tumors != 1):
            raise ValueError("multiple tumors require a cohort_profile")
        if not math.isfinite(self.risk_aversion) or self.risk_aversion < 0:
            raise ValueError("risk_aversion must be finite and nonnegative")
        temperatures = (self.annealing_start_temperature, self.annealing_end_temperature)
        if any(not math.isfinite(v) or v <= 0 for v in temperatures):
            raise ValueError("annealing temperatures must be finite and positive")
        if temperatures[0] < temperatures[1]:
            raise ValueError("annealing start temperature must be >= end temperature")
        self.weights.validate()
        if not self.base_profile or not self.output_dir:
            raise ValueError("base_profile and output_dir are required")
        # Conservative cap counts uncached simulator requests, including the exact oracle.
        requests = self.repeats * 2 * self.evaluation_budget
        if self.exact_search:
            requests += 2 ** self.intervals
        if requests * self.search_tumors > 100_000:
            raise ValueError("profile exceeds 100000 search tumor-schedule evaluations")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str | Path) -> TreatmentOptimizationConfig:
        config_path = Path(path)
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("optimization configuration root must be a mapping")
        unknown = set(raw) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown optimization fields: {sorted(unknown)}")
        payload = dict(raw)
        weights = payload.pop("weights", {})
        if not isinstance(weights, dict):
            raise ValueError("weights must be a mapping")
        for name in ("base_profile", "cohort_profile"):
            if payload.get(name):
                target = Path(payload[name])
                if not target.is_absolute():
                    target = config_path.parent / target
                payload[name] = str(target)
        config = cls(weights=ObjectiveWeights(**weights), **payload)
        config.validate()
        return config
