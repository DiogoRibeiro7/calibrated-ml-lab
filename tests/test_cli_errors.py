"""Check actionable command errors without running expensive models for each case."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from calibrated_ml_lab import cli
from calibrated_ml_lab.sample_data import generate_classification_data

runner = CliRunner()


@pytest.mark.parametrize("option,value", [("--rows", "0"), ("--rows", "-1"), ("--seed", "-1")])
def test_sample_data_rejects_invalid_ranges(tmp_path: Path, option: str, value: str) -> None:
    output = tmp_path / "output"
    result = runner.invoke(cli.app, ["sample-data", "--output-dir", str(output), option, value])
    assert result.exit_code == 2
    assert option in result.stderr
    assert "Traceback" not in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("command", ["sample-data", "run"])
def test_output_must_be_a_directory(tmp_path: Path, command: str) -> None:
    output = tmp_path / "existing.txt"
    output.write_text("keep me", encoding="utf-8")
    arguments = [command, "--output-dir", str(output)]
    if command == "run":
        arguments += ["--input", str(output), "--regression-input", str(output)]
    result = runner.invoke(cli.app, arguments)
    assert result.exit_code == 2
    assert "is a file" in result.stderr
    assert output.read_text(encoding="utf-8") == "keep me"


@pytest.mark.parametrize("option", ["--input", "--regression-input"])
def test_input_must_be_a_file(tmp_path: Path, option: str) -> None:
    csv = tmp_path / "data.csv"
    csv.write_text("feature,label,y\n1,0,2\n", encoding="utf-8")
    arguments = ["run", "--input", str(csv), "--regression-input", str(csv)]
    arguments[arguments.index(option) + 1] = str(tmp_path)
    result = runner.invoke(cli.app, arguments)
    assert result.exit_code == 2
    assert "is a directory" in result.stderr


@pytest.mark.parametrize(
    "contents,message",
    [
        ("", "No columns to parse"),
        ('feature,label\n1,"unterminated', "Error tokenizing data"),
        ("feature,label\n1,3\n", "binary labels"),
    ],
)
def test_run_reports_invalid_csv(tmp_path: Path, contents: str, message: str) -> None:
    csv = tmp_path / "data.csv"
    csv.write_text(contents, encoding="utf-8")
    output = tmp_path / "output"
    result = runner.invoke(
        cli.app,
        [
            "run",
            "--input",
            str(csv),
            "--regression-input",
            str(csv),
            "--output-dir",
            str(output),
        ],
    )
    assert result.exit_code == 1
    assert message in result.stderr
    assert "Traceback" not in result.stderr
    assert "Wrote" not in result.stdout
    assert not output.exists()


def test_run_reports_missing_regression_target_literally(tmp_path: Path) -> None:
    classification = tmp_path / "classification.csv"
    generate_classification_data(400).to_csv(classification, index=False)
    regression = tmp_path / "regression.csv"
    regression.write_text("feature,wrong_target\n1,2\n", encoding="utf-8")
    output = tmp_path / "output"
    result = runner.invoke(
        cli.app,
        [
            "run",
            "--input",
            str(classification),
            "--regression-input",
            str(regression),
            "--regression-target",
            "[bold]y[/bold]",
            "--output-dir",
            str(output),
        ],
    )
    assert result.exit_code == 1
    assert "regression target column '[bold]y[/bold]' not found" in result.stderr
    assert "Traceback" not in result.stderr
    assert "Wrote" not in result.stdout
    assert not output.exists()


@pytest.mark.parametrize(
    "command,function",
    [
        ("sample-data", "write_sample_data"),
        ("run", "run_workflow"),
    ],
)
@pytest.mark.parametrize(
    "error", [PermissionError("permission denied [red]"), ValueError("bad data")]
)
def test_expected_errors_use_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    function: str,
    error: Exception,
) -> None:
    def fail(**kwargs):
        raise error

    monkeypatch.setattr(cli, function, fail)
    arguments = [command]
    if command == "run":
        csv = tmp_path / "data.csv"
        csv.touch()
        arguments += ["--input", str(csv), "--regression-input", str(csv)]
    result = runner.invoke(cli.app, arguments)
    assert result.exit_code == 1
    assert f"Error: {error}" in result.stderr
    assert "Traceback" not in result.stderr
    assert not result.stdout


def test_unexpected_programming_error_is_not_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(**kwargs):
        raise RuntimeError("unexpected failure")

    monkeypatch.setattr(cli, "write_sample_data", fail)
    result = runner.invoke(cli.app, ["sample-data"])
    assert isinstance(result.exception, RuntimeError)
