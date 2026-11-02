"""Protect existing reports when a later workflow stage fails."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from calibrated_ml_lab import pipeline


@pytest.mark.parametrize(
    "frame,message",
    [
        (pd.DataFrame({"feature": [1, 2]}), "target column 'y' not found"),
        (pd.DataFrame({"feature": [1, 2], "y": [1, np.nan]}), "finite values"),
        (pd.DataFrame({"feature": [1, 2], "y": [1, np.inf]}), "finite values"),
        (pd.DataFrame({"feature": [1, 2], "y": ["1", "bad"]}), "real numeric values"),
        (pd.DataFrame({"feature": ["a", "b"], "y": [1, 2]}), "numeric feature column"),
        (pd.DataFrame({"feature": [], "y": []}), "nonempty"),
    ],
)
def test_regression_rejects_invalid_data_before_fitting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frame: pd.DataFrame,
    message: str,
) -> None:
    def unexpected_fit(*args, **kwargs):
        pytest.fail("invalid input reached model fitting")

    monkeypatch.setattr(pipeline, "build_regression_pipeline", unexpected_fit)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match=message):
        pipeline.run_regression_conformal(frame, "y", output)
    assert not output.exists()


@pytest.mark.parametrize("failure_stage", ["classification", "regression", "report", None])
@pytest.mark.parametrize("existing_output", [False, True])
def test_workflow_only_publishes_after_all_stages_succeed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str | None,
    existing_output: bool,
) -> None:
    csv = tmp_path / "data.csv"
    csv.write_text("feature,label,y\n1,0,2\n", encoding="utf-8")
    output = tmp_path / "output"
    reference_directory = tmp_path / "reference"
    reference_directory.mkdir()
    if existing_output:
        (output / "tables").mkdir(parents=True)
        for name in [
            "calibration_report.md",
            "tables/metrics.csv",
            "notes.txt",
            "tables/previous_version.csv",
        ]:
            (output / name).write_text("old content", encoding="utf-8")
        if os.name == "posix":
            output.chmod(0o750)
            (output / "tables").chmod(0o750)
            (output / "tables/metrics.csv").chmod(0o640)
    existing_modes = {
        path: path.stat().st_mode for path in [output, *output.rglob("*")] if path.exists()
    }
    before = {
        path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()
    }
    temporary_paths = []

    def finish_stage(stage: str, path: Path) -> None:
        temporary_paths.append(path)
        if failure_stage == stage:
            raise ValueError(f"{stage} failed")

    def classification(frame, target, output_dir):
        (output_dir / "tables").mkdir()
        (output_dir / "tables/metrics.csv").write_text("new metrics", encoding="utf-8")
        finish_stage("classification", output_dir)
        return dict.fromkeys(
            [
                "metrics",
                "conformal_classification",
                "threshold_metrics",
                "selection_metrics",
                "classification_splits",
            ],
            pd.DataFrame(),
        )

    def regression(frame, target, output_dir):
        (output_dir / "scored_regression_test.csv").write_text("new scores", encoding="utf-8")
        finish_stage("regression", output_dir)
        return pd.DataFrame()

    def report(path, **kwargs):
        path.write_text("new report", encoding="utf-8")
        finish_stage("report", path.parent)

    monkeypatch.setattr(pipeline, "run_classification_calibration", classification)
    monkeypatch.setattr(pipeline, "run_regression_conformal", regression)
    monkeypatch.setattr(pipeline, "write_report", report)
    if failure_stage:
        with pytest.raises(ValueError, match=f"{failure_stage} failed"):
            pipeline.run_workflow(csv, csv, output)
        after = {
            path.relative_to(output): path.read_bytes()
            for path in output.rglob("*")
            if path.is_file()
        }
        assert after == before
        assert output.exists() == existing_output
    else:
        pipeline.run_workflow(csv, csv, output)
        assert (output / "calibration_report.md").read_text(encoding="utf-8") == "new report"
        assert (output / "tables/metrics.csv").read_text(encoding="utf-8") == "new metrics"
        assert (output / "scored_regression_test.csv").read_text(encoding="utf-8") == "new scores"
        if existing_output:
            assert (output / "notes.txt").read_text(encoding="utf-8") == "old content"
            assert (output / "tables/previous_version.csv").read_text(
                encoding="utf-8"
            ) == "old content"
        else:
            assert output.stat().st_mode == reference_directory.stat().st_mode
    assert all(path.stat().st_mode == mode for path, mode in existing_modes.items())
    assert temporary_paths
    assert all(not path.exists() for path in temporary_paths)


def test_workflow_rejects_file_as_output_directory(tmp_path: Path) -> None:
    output = tmp_path / "existing.txt"
    output.write_text("keep me", encoding="utf-8")
    with pytest.raises(NotADirectoryError, match="output directory"):
        pipeline.run_workflow(tmp_path / "missing.csv", tmp_path / "missing.csv", output)
    assert output.read_text(encoding="utf-8") == "keep me"
