"""Configuration for deterministic virtual-tumor cohort studies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

ParameterScale = Literal["linear", "log10"]
SamplingMethod = Literal["latin_hypercube"]

_SUPPORTED_PARAMETERS = {
    "initial_resistant_fraction",
    "sensitive_growth_rate_per_day",
    "resistant_growth_rate_per_day",
    "sensitive_drug_kill_rate_per_day",
    "resistant_drug_kill_rate_per_day",
    "resistant_effect_on_sensitive",
    "sensitive_effect_on_resistant",
    "sensitive_to_resistant_rate_per_day",
}


@dataclass(frozen=True, slots=True)
class ParameterRange:
    """One bounded biological parameter sampled for a virtual cohort."""

    name: str
    minimum: float
    maximum: float
    scale: ParameterScale = "linear"

    def validate(self) -> None:
        """Validate a declared parameter range."""
        if self.name not in _SUPPORTED_PARAMETERS:
            raise ValueError(f"unsupported cohort parameter: {self.name}")
        if self.maximum <= self.minimum:
            raise ValueError(f"{self.name}: maximum must exceed minimum")
        if self.scale not in {"linear", "log10"}:
            raise ValueError(f"{self.name}: scale must be 'linear' or 'log10'")
        if self.scale == "log10" and self.minimum <= 0:
            raise ValueError(f"{self.name}: log10 ranges require a positive minimum")
        if self.name == "initial_resistant_fraction" and not (
            0 < self.minimum < self.maximum < 1
        ):
            raise ValueError("initial_resistant_fraction must remain between 0 and 1")
        if self.name != "initial_resistant_fraction" and self.minimum < 0:
            raise ValueError(f"{self.name}: parameter ranges must be nonnegative")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable range."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvolutionCohortConfig:
    """Versioned configuration for a deterministic virtual-tumor cohort."""

    protocol_version: str = "evolution-cohort-v1"
    profile_name: str = "two-clone-virtual-cohort-v1"
    base_profile: str = "configs/evolution-two-clone.yaml"
    virtual_tumors: int = 128
    seed: int = 42
    sampling_method: SamplingMethod = "latin_hypercube"
    reference_strategy: str = "continuous"
    candidate_strategy: str = "burden_adaptive"
    parameter_ranges: tuple[ParameterRange, ...] = ()
    output_dir: str = "reports/evolution-virtual-cohort"

    def validate(self) -> None:
        """Validate cohort size, strategy comparison, and parameter ranges."""
        if self.protocol_version != "evolution-cohort-v1":
            raise ValueError("protocol_version must be 'evolution-cohort-v1'")
        if not self.profile_name:
            raise ValueError("profile_name is required")
        if not self.base_profile:
            raise ValueError("base_profile is required")
        if not 4 <= self.virtual_tumors <= 10_000:
            raise ValueError("virtual_tumors must be between 4 and 10000")
        if self.sampling_method != "latin_hypercube":
            raise ValueError("sampling_method must be 'latin_hypercube'")
        if self.reference_strategy == self.candidate_strategy:
            raise ValueError("reference_strategy and candidate_strategy must differ")
        if not self.parameter_ranges:
            raise ValueError("parameter_ranges must contain at least one parameter")
        names = [item.name for item in self.parameter_ranges]
        if len(names) != len(set(names)):
            raise ValueError("parameter_ranges must not contain duplicate names")
        for item in self.parameter_ranges:
            item.validate()
        if not self.output_dir:
            raise ValueError("output_dir is required")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable cohort configuration."""
        return {
            "protocol_version": self.protocol_version,
            "profile_name": self.profile_name,
            "base_profile": self.base_profile,
            "virtual_tumors": self.virtual_tumors,
            "seed": self.seed,
            "sampling_method": self.sampling_method,
            "reference_strategy": self.reference_strategy,
            "candidate_strategy": self.candidate_strategy,
            "parameter_ranges": [item.to_dict() for item in self.parameter_ranges],
            "output_dir": self.output_dir,
        }

    @classmethod
    def from_yaml(cls, path: str | Path) -> EvolutionCohortConfig:
        """Load a cohort profile and resolve its base profile path."""
        config_path = Path(path)
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError("evolution cohort configuration root must be a mapping")
        payload: dict[str, Any] = {str(key): value for key, value in loaded.items()}
        raw_ranges = payload.pop("parameter_ranges", {})
        if not isinstance(raw_ranges, dict):
            raise ValueError("parameter_ranges must be a mapping")
        ranges: list[ParameterRange] = []
        for name, raw in raw_ranges.items():
            if not isinstance(raw, dict):
                raise ValueError(f"parameter range {name} must be a mapping")
            ranges.append(ParameterRange(name=str(name), **raw))
        base_profile = Path(str(payload.get("base_profile", "evolution-two-clone.yaml")))
        if not base_profile.is_absolute() and not base_profile.exists():
            base_profile = config_path.parent / base_profile
        payload["base_profile"] = str(base_profile)
        config = cls(parameter_ranges=tuple(ranges), **payload)
        config.validate()
        return config
