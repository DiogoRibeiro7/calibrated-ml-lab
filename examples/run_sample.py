"""Run a complete sample workflow."""

from pathlib import Path

from calibrated_ml_lab.pipeline import run_workflow
from calibrated_ml_lab.sample_data import write_sample_data

raw_dir = Path("data/raw")
output_dir = Path("reports/sample_run")
classification_path, regression_path = write_sample_data(raw_dir, rows=5_000)
run_workflow(classification_path, regression_path, output_dir)
print(f"Report written to {output_dir / 'calibration_report.md'}")
