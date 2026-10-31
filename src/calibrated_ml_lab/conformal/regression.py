"""Split conformal intervals for regression."""

from __future__ import annotations

import math

import numpy as np


def conformal_residual_quantile(
    y_calibration: np.ndarray, y_pred_calibration: np.ndarray, alpha: float = 0.1
) -> float:
    """Compute split-conformal absolute-residual quantile."""

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    y = np.asarray(y_calibration, dtype=float)
    pred = np.asarray(y_pred_calibration, dtype=float)
    if len(y) != len(pred) or len(y) == 0:
        raise ValueError("calibration arrays must have the same non-zero length")
    residuals = np.abs(y - pred)
    n = len(residuals)
    q_level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(residuals, q_level, method="higher"))


def conformal_intervals(y_pred: np.ndarray, qhat: float) -> tuple[np.ndarray, np.ndarray]:
    """Return lower and upper split-conformal intervals."""

    pred = np.asarray(y_pred, dtype=float)
    if qhat < 0:
        raise ValueError("qhat must be non-negative")
    return pred - qhat, pred + qhat


def regression_interval_metrics(
    y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> dict[str, float]:
    """Compute coverage and width diagnostics for prediction intervals."""

    y = np.asarray(y_true, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    if not (len(y) == len(lo) == len(hi)):
        raise ValueError("all arrays must have the same length")
    covered = (y >= lo) & (y <= hi)
    width = hi - lo
    return {
        "coverage": float(np.mean(covered)),
        "average_width": float(np.mean(width)),
        "median_width": float(np.median(width)),
        "min_width": float(np.min(width)),
        "max_width": float(np.max(width)),
    }
