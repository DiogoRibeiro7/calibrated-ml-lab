"""Command-line interface for calibrated-ml-lab."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from calibrated_ml_lab.pipeline import run_workflow
from calibrated_ml_lab.sample_data import write_sample_data

app = typer.Typer(help="Calibration and uncertainty estimation workbench.")
console = Console()


@app.command("sample-data")
def sample_data(
    output_dir: Path = typer.Option(Path("data/raw"), help="Directory for generated CSV files."),
    rows: int = typer.Option(5_000, help="Number of classification rows to generate."),
    seed: int = typer.Option(42, help="Random seed."),
) -> None:
    """Generate synthetic classification and regression datasets."""

    classification_path, regression_path = write_sample_data(output_dir=output_dir, rows=rows, seed=seed)
    console.print(f"Wrote classification data to [bold]{classification_path}[/bold]")
    console.print(f"Wrote regression data to [bold]{regression_path}[/bold]")


@app.command("run")
def run(
    input: Path = typer.Option(..., help="Classification CSV path."),
    regression_input: Path = typer.Option(..., help="Regression CSV path."),
    output_dir: Path = typer.Option(Path("reports/sample_run"), help="Output directory."),
    target: str = typer.Option("label", help="Classification target column."),
    regression_target: str = typer.Option("y", help="Regression target column."),
) -> None:
    """Run the full calibration and uncertainty workflow."""

    run_workflow(
        classification_path=input,
        regression_path=regression_input,
        output_dir=output_dir,
        target=target,
        regression_target=regression_target,
    )
    console.print(f"Wrote report to [bold]{output_dir / 'calibration_report.md'}[/bold]")


if __name__ == "__main__":
    app()
