"""Verify native integer penalties and strict decoding with exhaustive small domains."""

from dataclasses import replace
from itertools import product

import pytest

from quantum_oncology_benchmark.treatment_config import TreatmentOptimizationConfig
from quantum_oncology_benchmark.treatment_dirac import (
    compile_integer_polynomial,
    polynomial_energy,
    validate_integer_sample,
)
from quantum_oncology_benchmark.treatment_search import schedule_feasible


def config() -> TreatmentOptimizationConfig:
    return TreatmentOptimizationConfig.from_yaml("configs/treatment-optimization-smoke.yaml")


def test_native_penalty_energy_and_ground_state() -> None:
    cfg = config()
    linear = [-1.0, -2.0, 0.5, -3.0]
    pairwise = [0.2, -0.4, 0.6, 0.1, -0.8, 0.2]
    compiled = compile_integer_polynomial(linear, pairwise, 1.23, cfg)
    best_feasible = float("inf")
    best_infeasible = float("inf")
    for bits in product((0, 1), repeat=cfg.intervals):
        quadratic = sum(v * b for v, b in zip(linear, bits, strict=True)) + 1.23
        pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
        quadratic += sum(v * bits[i] * bits[j] for v, (i, j) in zip(pairwise, pairs, strict=True))
        for slack in range(cfg.max_treated_intervals + 1):
            values = (*bits, slack)
            expected_violation = (sum(bits) + slack - cfg.max_treated_intervals) ** 2
            expected_violation += bits[0] * bits[1] * bits[2] + bits[1] * bits[2] * bits[3]
            energy = polynomial_energy(values, compiled)
            assert energy == pytest.approx(quadratic + compiled["penalty"] * expected_violation)
            if schedule_feasible(bits, cfg) and sum(bits) + slack == cfg.max_treated_intervals:
                best_feasible = min(best_feasible, energy)
                assert validate_integer_sample(list(values), cfg) == bits
            else:
                best_infeasible = min(best_infeasible, energy)
                with pytest.raises(ValueError):
                    validate_integer_sample(list(values), cfg)
    assert best_feasible < best_infeasible
    assert compiled["hardware_submission_enabled"] is False
    polynomial = compiled["polynomial_file"]["file_config"]["polynomial"]
    assert polynomial["num_variables"] == 5
    assert polynomial["max_degree"] == 3
    assert all(len(term["idx"]) == 3 for term in polynomial["data"])
    assert compiled["job_template"]["job_params"]["num_levels"] == [2, 2, 2, 2, 4]


@pytest.mark.parametrize("sample", [[0, 0], [1.4, 0, 0, 0, 2],
                                     [float("nan"), 0, 0, 0, 3], [True, 0, 0, 0, 2]])
def test_decoder_rejects_incomplete_noninteger_and_nonfinite(sample: list[float]) -> None:
    with pytest.raises(ValueError):
        validate_integer_sample(sample, config())


def test_degree_overflow_rejected() -> None:
    cfg = replace(config(), intervals=6, max_consecutive_treated=5)
    with pytest.raises(ValueError, match="degree"):
        compile_integer_polynomial([0.0] * 6, [0.0] * 15, 0.0, cfg)


def test_end_to_end_surrogate_and_source_integrity(tmp_path) -> None:
    import json

    from quantum_oncology_benchmark.treatment_dirac import compile_dirac_treatment
    from quantum_oncology_benchmark.treatment_optimization import run_treatment_optimization

    cfg = replace(config(), intervals=6, max_treated_intervals=4,
                  max_consecutive_treated=2, repeats=1, evaluation_budget=4,
                  output_dir=str(tmp_path / "source"))
    run_treatment_optimization(cfg)
    experiment = tmp_path / "source/treatment_optimization_experiment.json"
    result = compile_dirac_treatment(experiment, tmp_path / "compiled")
    assert result["validation"]["training_schedules"] > 21
    assert result["validation"]["validation_schedules"] >= 4
    assert result["validation"]["encoded_ground_feasible"]
    assert result["validation"]["selected_true_regret"] >= 0
    assert (tmp_path / "compiled/dirac_polynomial_file.json").exists()
    tampered = json.loads(experiment.read_text())
    tampered["scientific"]["config"]["weights"]["dose"] = 99.0
    experiment.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="fingerprint"):
        compile_dirac_treatment(experiment, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_imported_counts_validation_and_origin(tmp_path) -> None:
    import json

    from quantum_oncology_benchmark.treatment_dirac import (
        compile_dirac_treatment,
        import_dirac_treatment_samples,
    )
    from quantum_oncology_benchmark.treatment_optimization import run_treatment_optimization

    cfg = replace(config(), intervals=6, max_treated_intervals=4, repeats=1,
                  evaluation_budget=4, output_dir=str(tmp_path / "source"))
    run_treatment_optimization(cfg)
    compiled = compile_dirac_treatment(tmp_path / "source/treatment_optimization_experiment.json",
                                      tmp_path / "compiled")
    winner = [int(x) for x in compiled["validation"]["selected_schedule"]]
    winner.append(cfg.max_treated_intervals - sum(winner))
    samples = {"results": {"solutions": [winner, [1.4] * 7, [1] * 7], "counts": [3, 2, 1]}}
    sample_path = tmp_path / "samples.json"
    sample_path.write_text(json.dumps(samples))
    result = import_dirac_treatment_samples(tmp_path / "compiled/dirac_surrogate_experiment.json",
                                           sample_path, tmp_path / "import", origin="local_test_samples")
    assert result["summary"]["sample_count"] == 6
    assert result["summary"]["valid_sample_count"] == 3
    assert result["summary"]["valid_fraction"] == 0.5
    assert result["verified_hardware_execution"] is False
    assert sum(r["count"] for r in result["samples"] if not r["valid"]) == 3
    samples["results"]["counts"] = [1]
    sample_path.write_text(json.dumps(samples))
    with pytest.raises(ValueError, match="dimensions"):
        import_dirac_treatment_samples(tmp_path / "compiled/dirac_surrogate_experiment.json",
                                      sample_path, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_compiled_fingerprint_covers_objective_configuration() -> None:
    from quantum_oncology_benchmark.treatment_dirac import compiled_surrogate_fingerprint

    report = {"config": config().to_dict(), "fit": {"coefficient": 1.0}}
    before = compiled_surrogate_fingerprint(report)
    report["config"]["weights"]["dose"] = 2.0
    assert compiled_surrogate_fingerprint(report) != before
