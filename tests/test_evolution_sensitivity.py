"""Contracts for independent biological and policy sensitivity sweeps."""

import json
from pathlib import Path

import pytest
import yaml

from quantum_oncology_benchmark.evolution_sensitivity import run_evolution_sensitivity


def profile(tmp_path: Path) -> Path:
    base = {
        "horizon_days": 10.0, "time_step_days": 1.0,
        "sensitive": {"growth_rate_per_day": 0.03, "drug_kill_rate_per_day": 0.08},
        "resistant": {"growth_rate_per_day": 0.015, "drug_kill_rate_per_day": 0.0},
    }
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(base))
    spec = {
        "protocol_version": "evolution-sensitivity-v1", "base_profile": "base.yaml",
        "transition_rates_per_day": [0.0, 0.001], "adaptive_stop_fractions": [0.25, 0.5],
        "adaptive_restart_fractions": [1.0], "output_dir": str(tmp_path / "out"),
    }
    path = tmp_path / "sweep.yaml"
    path.write_text(yaml.safe_dump(spec))
    return path


def test_sweeps_separate_inputs_and_replay(tmp_path: Path) -> None:
    path = profile(tmp_path)
    first = run_evolution_sensitivity(path)
    second = run_evolution_sensitivity(path, output_dir=tmp_path / "replay")
    assert first["fingerprint"] == second["fingerprint"]
    rows = first["scientific_inputs_and_results"]["outcomes"]
    assert len(rows) == 8
    transition = [r for r in rows if r["sweep"] == "acquired_transition"]
    policy = [r for r in rows if r["sweep"] == "policy_threshold"]
    assert {r["adaptive_stop_fraction"] for r in transition} == {0.5}
    assert {r["transition_rate_per_day"] for r in policy} == {0.0}
    for row in rows:
        if row["strategy"] == "continuous":
            assert row["burden_auc_delta_vs_continuous"] == 0
            assert row["dose_days_delta_vs_continuous"] == 0
    stored = json.loads((tmp_path / "out/evolution_sensitivity_experiment.json").read_text())
    assert stored["fingerprint"] == first["fingerprint"]


@pytest.mark.parametrize("values", [[float("nan")], [-0.1], [0.0, 0.0], [True]])
def test_invalid_rates_fail_before_output(tmp_path: Path, values: list[float]) -> None:
    path = profile(tmp_path)
    spec = yaml.safe_load(path.read_text())
    spec["transition_rates_per_day"] = values
    path.write_text(yaml.safe_dump(spec))
    with pytest.raises(ValueError):
        run_evolution_sensitivity(path)
    assert not (tmp_path / "out").exists()


def test_invalid_threshold_pair_rejected(tmp_path: Path) -> None:
    path = profile(tmp_path)
    spec = yaml.safe_load(path.read_text())
    spec["adaptive_restart_fractions"] = [0.4]
    path.write_text(yaml.safe_dump(spec))
    with pytest.raises(ValueError, match="below"):
        run_evolution_sensitivity(path)
    assert not (tmp_path / "out").exists()
