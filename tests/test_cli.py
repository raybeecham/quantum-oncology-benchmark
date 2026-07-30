from __future__ import annotations

from quantum_oncology_benchmark.cli import main


def test_doctor_returns_success(capsys) -> None:
    assert main(["doctor"]) == 0
    output = capsys.readouterr().out
    assert "Core dependencies" in output
    assert "Quantum dependencies" in output
