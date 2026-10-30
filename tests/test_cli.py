"""Exercise the installed CLI and its complete offline reporting workflow."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest


def run_cli(directory: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    """Run the installed entry point with a noninteractive plotting backend."""
    executable = shutil.which("calibrated-lab")
    assert executable is not None, "Install the package before running CLI tests"
    return subprocess.run(
        [executable, *arguments],
        cwd=directory,
        env={**os.environ, "MPLBACKEND": "Agg"},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


@pytest.mark.parametrize("command", [[], ["sample-data"], ["run"]])
def test_cli_help(tmp_path: Path, command: list[str]) -> None:
    result = run_cli(tmp_path, *command, "--help")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Usage:" in result.stdout


def test_cli_generates_data_and_complete_report(tmp_path: Path) -> None:
    working_dir = tmp_path / "working directory"
    working_dir.mkdir()
    generated = run_cli(
        working_dir, "sample-data", "--output-dir", "input data", "--rows", "400", "--seed", "42"
    )
    assert generated.returncode == 0, generated.stdout + generated.stderr
    assert len(pd.read_csv(working_dir / "input data/classification.csv")) == 400

    completed = run_cli(
        working_dir,
        "run",
        "--input",
        "input data/classification.csv",
        "--regression-input",
        "input data/regression.csv",
        "--output-dir",
        "output report",
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    output = working_dir / "output report"
    report = (output / "calibration_report.md").read_text(encoding="utf-8")
    assert "## Calibration metrics" in report
    assert "## Split-conformal classification" in report
    assert "## Split-conformal regression" in report
    assert "## Decision-threshold analysis" in report

    metrics = pd.read_csv(output / "tables/calibration_metrics.csv")
    assert set(metrics["model"]) == {"uncalibrated", "platt", "isotonic", "temperature"}
    assert metrics[["ece", "brier_score", "log_loss"]].notna().all().all()
    for name in ["conformal_classification_summary.csv", "conformal_regression_summary.csv"]:
        summary = pd.read_csv(output / "tables" / name)
        assert summary["coverage"].between(0, 1).all()
    assert not pd.read_csv(output / "tables/threshold_metrics.csv").empty
    assert not pd.read_csv(output / "scored_classification_test.csv").empty
    scored_regression = pd.read_csv(output / "scored_regression_test.csv")
    assert not scored_regression.empty
    assert (scored_regression["interval_lower"] <= scored_regression["interval_upper"]).all()

    for name in [
        "reliability_diagram.png",
        "probability_histogram.png",
        "conformal_regression_intervals.png",
        "threshold_tradeoff.png",
    ]:
        assert (output / "figures" / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
