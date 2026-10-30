"""Calibration and probabilistic classification metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

EPS = 1e-12


def _validate_binary_inputs(
    y_true: np.ndarray, y_prob: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and coerce binary labels and probabilities."""

    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    if y.ndim != 1 or p.ndim != 1:
        raise ValueError("y_true and y_prob must be one-dimensional")
    if len(y) != len(p):
        raise ValueError("y_true and y_prob must have the same length")
    if len(y) == 0:
        raise ValueError("inputs must not be empty")
    if not np.all((p >= 0) & (p <= 1)):
        raise ValueError("probabilities must be between 0 and 1")
    return y, np.clip(p, EPS, 1 - EPS)


def calibration_bins(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Compute calibration-bin summary statistics.

    Bins are equal-width over [0, 1]. Empty bins are preserved with zero counts.
    """

    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    y, p = _validate_binary_inputs(y_true, y_prob)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_id = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)

    rows: list[dict[str, float]] = []
    total = len(y)
    for i in range(n_bins):
        mask = bin_id == i
        count = int(mask.sum())
        if count == 0:
            mean_pred = np.nan
            frac_pos = np.nan
            abs_gap = 0.0
        else:
            mean_pred = float(p[mask].mean())
            frac_pos = float(y[mask].mean())
            abs_gap = abs(mean_pred - frac_pos)
        rows.append(
            {
                "bin": i,
                "bin_lower": float(edges[i]),
                "bin_upper": float(edges[i + 1]),
                "count": count,
                "weight": count / total,
                "mean_predicted_probability": mean_pred,
                "observed_event_rate": frac_pos,
                "absolute_gap": abs_gap,
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute expected calibration error with equal-width bins."""

    bins = calibration_bins(y_true, y_prob, n_bins=n_bins)
    return float((bins["weight"] * bins["absolute_gap"]).sum())


def maximum_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute maximum calibration error over non-empty bins."""

    bins = calibration_bins(y_true, y_prob, n_bins=n_bins)
    non_empty = bins[bins["count"] > 0]
    if non_empty.empty:
        return 0.0
    return float(non_empty["absolute_gap"].max())


def probabilistic_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> dict[str, float]:
    """Compute discrimination and calibration metrics for binary probabilities."""

    y, p = _validate_binary_inputs(y_true, y_prob)
    metrics = {
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ece": expected_calibration_error(y, p, n_bins=n_bins),
        "mce": maximum_calibration_error(y, p, n_bins=n_bins),
    }
    if len(np.unique(y)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y, p))
    else:
        metrics["roc_auc"] = float("nan")
    return metrics
