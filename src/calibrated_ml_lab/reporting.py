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
    selection_metrics: pd.DataFrame,
    classification_splits: pd.DataFrame,
) -> None:
    """Write a concise calibration and uncertainty report."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected = selection_metrics.loc[selection_metrics["selected"]].iloc[0]
    selected_test = metrics.loc[metrics["model"] == selected["model"]].iloc[0]

    text = f"""
# Calibration and uncertainty report

## Executive summary

The workflow compared uncalibrated model probabilities with Platt scaling,
isotonic regression, and temperature scaling. Calibration was evaluated separately
from discrimination.

- Selected method: **{selected['model']}**, chosen by lowest validation ECE
  (**{selected['ece']:.4f}**) before final test evaluation.
- Selected method's final test ECE: **{selected_test['ece']:.4f}**.
- Selected method's final test log loss: **{selected_test['log_loss']:.4f}**.

## Classification data splits

Training, calibrator fitting, method selection, conformal calibration, and final
testing use disjoint rows. The selected method and conformal threshold are frozen
before final test evaluation. No calibrator is refitted after method selection.

{_markdown_table(classification_splits)}

## Calibration-method selection

These metrics use the selection (validation) split. Exact ECE ties use candidate
order: uncalibrated, Platt, isotonic, then temperature scaling.

{_markdown_table(selection_metrics)}

## Calibration metrics

These are final test metrics. The selected flag records the validation choice;
test rankings do not change that choice. ECE, MCE, Brier score, and log loss
describe probability quality; ROC AUC describes discrimination.

{_markdown_table(metrics)}

## Split-conformal classification

{_markdown_table(conformal_classification)}

## Split-conformal regression

{_markdown_table(conformal_regression)}

## Decision-threshold analysis

This table describes the selected method on the final test set. Its lowest-cost
threshold is exploratory, not a validated deployment recommendation. Choosing a
deployment threshold requires separate validation data and a new test evaluation.

{_markdown_table(threshold_metrics.sort_values('expected_cost').head(10))}

## Figures

- `figures/reliability_diagram.png`
- `figures/probability_histogram.png`
- `figures/conformal_regression_intervals.png`
- `figures/threshold_tradeoff.png`

## Interpretation notes

Calibration answers whether predicted probabilities match observed frequencies.
It does not prove causal validity, fairness, clinical safety, or robustness under
distribution shift. Reported coverage is empirical and can differ from the nominal
90% target. Conformal validity requires exchangeable conformal-calibration and
future examples; random row splits do not address time dependence, repeated
entities, or distribution shift.
"""
    output_path.write_text(text.strip() + "\n", encoding="utf-8")
