"""Configuration models for the evolutionary treatment simulator."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

TreatmentStrategy = Literal[
    "no_treatment",
    "continuous",
    "fixed_intermittent",
    "burden_adaptive",
]

SUPPORTED_STRATEGIES: tuple[TreatmentStrategy, ...] = (
    "no_treatment",
    "continuous",
    "fixed_intermittent",
    "burden_adaptive",
)


@dataclass(frozen=True, slots=True)
class CloneParameters:
    """Growth and treatment-response assumptions for one tumor clone."""

    growth_rate_per_day: float
    drug_kill_rate_per_day: float

    def validate(self, label: str) -> None:
        """Validate one clone parameter block."""
        if self.growth_rate_per_day < 0:
            raise ValueError(f"{label}.growth_rate_per_day must be nonnegative")
        if self.drug_kill_rate_per_day < 0:
            raise ValueError(f"{label}.drug_kill_rate_per_day must be nonnegative")


@dataclass(frozen=True, slots=True)
class CompetitionParameters:
    """Pairwise competition coefficients in the two-clone system."""

    resistant_effect_on_sensitive: float = 1.0
    sensitive_effect_on_resistant: float = 1.0

    def validate(self) -> None:
        """Validate competition coefficients."""
        if self.resistant_effect_on_sensitive < 0:
            raise ValueError("resistant_effect_on_sensitive must be nonnegative")
        if self.sensitive_effect_on_resistant < 0:
            raise ValueError("sensitive_effect_on_resistant must be nonnegative")


@dataclass(frozen=True, slots=True)
class EvolutionConfig:
    """Configuration for one assumption-driven evolutionary simulation."""

    protocol_version: str = "evolution-protocol-v1"
    profile_name: str = "two-clone-selection-v1"
    sensitive_initial: float = 990_000.0
    resistant_initial: float = 10_000.0
    carrying_capacity: float = 10_000_000.0
    sensitive: CloneParameters = CloneParameters(0.03, 0.08)
    resistant: CloneParameters = CloneParameters(0.015, 0.0)
    competition: CompetitionParameters = CompetitionParameters()
    sensitive_to_resistant_rate_per_day: float = 0.0
    horizon_days: float = 365.0
    time_step_days: float = 1.0
    strategies: tuple[TreatmentStrategy, ...] = SUPPORTED_STRATEGIES
    intermittent_on_days: float = 14.0
    intermittent_off_days: float = 14.0
    adaptive_stop_fraction: float = 0.50
    adaptive_restart_fraction: float = 1.00
    resistant_dominance_fraction: float = 0.50
    progression_burden_multiple: float = 1.20
    output_dir: str = "reports/evolution-two-clone"

    def validate(self) -> None:
        """Validate biological assumptions and treatment policies."""
        if self.protocol_version != "evolution-protocol-v1":
            raise ValueError("protocol_version must be 'evolution-protocol-v1'")
        if not self.profile_name:
            raise ValueError("profile_name is required")
        if self.sensitive_initial <= 0 or self.resistant_initial < 0:
            raise ValueError("initial clone populations must be nonnegative with sensitive > 0")
        if self.carrying_capacity <= 0:
            raise ValueError("carrying_capacity must be positive")
        if self.sensitive_initial + self.resistant_initial > self.carrying_capacity:
            raise ValueError("initial total burden must not exceed carrying_capacity")
        self.sensitive.validate("sensitive")
        self.resistant.validate("resistant")
        self.competition.validate()
        if self.sensitive_to_resistant_rate_per_day < 0:
            raise ValueError("sensitive_to_resistant_rate_per_day must be nonnegative")
        if self.horizon_days <= 0:
            raise ValueError("horizon_days must be positive")
        if self.time_step_days <= 0 or self.time_step_days > self.horizon_days:
            raise ValueError("time_step_days must be positive and no greater than horizon_days")
        if not self.strategies:
            raise ValueError("at least one treatment strategy is required")
        unsupported = sorted(set(self.strategies) - set(SUPPORTED_STRATEGIES))
        if unsupported:
            raise ValueError(f"unsupported treatment strategies: {unsupported}")
        if len(set(self.strategies)) != len(self.strategies):
            raise ValueError("strategies must not contain duplicates")
        if self.intermittent_on_days <= 0 or self.intermittent_off_days <= 0:
            raise ValueError("intermittent on/off durations must be positive")
        if not 0 < self.adaptive_stop_fraction < self.adaptive_restart_fraction:
            raise ValueError(
                "adaptive_stop_fraction must be positive and below adaptive_restart_fraction"
            )
        if not 0 < self.resistant_dominance_fraction < 1:
            raise ValueError("resistant_dominance_fraction must be between 0 and 1")
        if self.progression_burden_multiple <= 1:
            raise ValueError("progression_burden_multiple must be greater than 1")

    @property
    def initial_total_burden(self) -> float:
        """Return the initial total tumor burden."""
        return self.sensitive_initial + self.resistant_initial

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable configuration mapping."""
        payload = asdict(self)
        payload["strategies"] = list(self.strategies)
        return payload

    @classmethod
    def from_yaml(cls, path: str | Path) -> EvolutionConfig:
        """Load a versioned evolution profile from YAML."""
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError("evolution configuration root must be a mapping")
        payload: dict[str, Any] = {str(key): value for key, value in loaded.items()}
        raw_sensitive = payload.pop("sensitive", {})
        raw_resistant = payload.pop("resistant", {})
        raw_competition = payload.pop("competition", {})
        if not isinstance(raw_sensitive, dict) or not isinstance(raw_resistant, dict):
            raise ValueError("sensitive and resistant parameter blocks must be mappings")
        if not isinstance(raw_competition, dict):
            raise ValueError("competition parameter block must be a mapping")
        if isinstance(payload.get("strategies"), list):
            payload["strategies"] = tuple(str(item) for item in payload["strategies"])
        config = cls(
            sensitive=CloneParameters(**raw_sensitive),
            resistant=CloneParameters(**raw_resistant),
            competition=CompetitionParameters(**raw_competition),
            **payload,
        )
        config.validate()
        return config
