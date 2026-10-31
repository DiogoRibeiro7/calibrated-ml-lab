"""Markdown reporting utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _markdown_table(frame: pd.DataFrame, max_rows: int = 12) -> str:
    """Convert a dataframe to markdown safely."""

    if frame.empty:
        return "_No rows._"
    shown = frame.head(max_rows).copy()
    return shown.to_markdown(index=False)


def write_report(
    output_path: Path,
    metrics: pd.DataFrame,
    conformal_classification: pd.DataFrame,
    conformal_regression: pd.DataFrame,
    threshold_metrics: pd.DataFrame,
) -> None:
    """Write a concise calibration and uncertainty report."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_ece = metrics.sort_values("ece", ascending=True).iloc[0]
    best_log_loss = metrics.sort_values("log_loss", ascending=True).iloc[0]
    best_threshold = threshold_metrics.sort_values("expected_cost", ascending=True).iloc[0]

    text = f"""
# Calibration and uncertainty report

## Executive summary

The workflow compared uncalibrated model probabilities with Platt scaling,
isotonic regression, and temperature scaling. Calibration was evaluated separately
from discrimination.

- Best ECE: **{best_ece['model']}** with ECE **{best_ece['ece']:.4f}**.
- Best log loss: **{best_log_loss['model']}** with log loss **{best_log_loss['log_loss']:.4f}**.
- Lowest expected-cost threshold for the selected calibrated model:
  **{best_threshold['threshold']:.2f}**.

## Calibration metrics

{_markdown_table(metrics)}

## Split-conformal classification

{_markdown_table(conformal_classification)}

## Split-conformal regression

{_markdown_table(conformal_regression)}

## Decision-threshold analysis

{_markdown_table(threshold_metrics.sort_values('expected_cost').head(10))}

## Figures

- `figures/reliability_diagram.png`
- `figures/probability_histogram.png`
- `figures/conformal_regression_intervals.png`
- `figures/threshold_tradeoff.png`

## Interpretation notes

Calibration answers whether predicted probabilities match observed frequencies.
It does not prove causal validity, fairness, clinical safety, or robustness under
distribution shift. Conformal coverage is empirical in this workflow and relies
on the calibration/test split being representative of the deployment setting.
"""
    output_path.write_text(text.strip() + "\n", encoding="utf-8")
