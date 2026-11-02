"""Command-line interface for calibrated-ml-lab."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from calibrated_ml_lab.pipeline import run_workflow
from calibrated_ml_lab.sample_data import write_sample_data

app = typer.Typer(help="Calibration and uncertainty estimation workbench.")
console = Console()
error_console = Console(stderr=True)


@app.command("sample-data")
def sample_data(
    output_dir: Annotated[
        Path, typer.Option(file_okay=False, help="Directory for generated CSV files.")
    ] = Path("data/raw"),
    rows: Annotated[
        int, typer.Option(min=1, help="Number of classification rows to generate.")
    ] = 5_000,
    seed: Annotated[int, typer.Option(min=0, help="Random seed.")] = 42,
) -> None:
    """Generate synthetic classification and regression datasets."""

    try:
        classification_path, regression_path = write_sample_data(
            output_dir=output_dir, rows=rows, seed=seed
        )
    except (ValueError, OSError) as error:
        error_console.print(f"Error: {error}", markup=False)
        raise typer.Exit(code=1) from error
    console.print(f"Wrote classification data to {classification_path}", markup=False)
    console.print(f"Wrote regression data to {regression_path}", markup=False)


@app.command("run")
def run(
    input: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, readable=True, help="Classification CSV path."),
    ],
    regression_input: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, readable=True, help="Regression CSV path."),
    ],
    output_dir: Annotated[Path, typer.Option(file_okay=False, help="Output directory.")] = Path(
        "reports/sample_run"
    ),
    target: Annotated[str, typer.Option(help="Classification target column.")] = "label",
    regression_target: Annotated[str, typer.Option(help="Regression target column.")] = "y",
) -> None:
    """Run the full calibration and uncertainty workflow."""

    try:
        run_workflow(
            classification_path=input,
            regression_path=regression_input,
            output_dir=output_dir,
            target=target,
            regression_target=regression_target,
        )
    except (ValueError, OSError) as error:
        error_console.print(f"Error: {error}", markup=False)
        raise typer.Exit(code=1) from error
    console.print(f"Wrote report to {output_dir / 'calibration_report.md'}", markup=False)


if __name__ == "__main__":
    app()
