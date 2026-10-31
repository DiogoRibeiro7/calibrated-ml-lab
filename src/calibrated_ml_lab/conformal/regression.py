"""Split conformal intervals for regression."""

from __future__ import annotations

import numpy as np

from calibrated_ml_lab._validation import as_vector
from calibrated_ml_lab.conformal._utils import conformal_quantile, nonnegative_threshold


def conformal_residual_quantile(
    y_calibration: np.ndarray, y_pred_calibration: np.ndarray, alpha: float = 0.1
) -> float:
    """Compute the absolute-residual threshold, possibly infinite for small samples."""

    y = as_vector(y_calibration, "y_calibration")
    pred = as_vector(y_pred_calibration, "y_pred_calibration")
    if len(y) != len(pred):
        raise ValueError("calibration arrays must have the same length")
    with np.errstate(over="ignore"):
        residuals = np.abs(y - pred)
    return conformal_quantile(residuals, alpha)


def conformal_intervals(y_pred: np.ndarray, qhat: float) -> tuple[np.ndarray, np.ndarray]:
    """Return intervals, using (-infinity, +infinity) when qhat is infinite."""

    pred = as_vector(y_pred, "y_pred")
    qhat = nonnegative_threshold(qhat, "qhat")
    with np.errstate(over="ignore"):
        return pred - qhat, pred + qhat


def regression_interval_metrics(
    y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> dict[str, float]:
    """Compute coverage and width diagnostics for prediction intervals."""

    y = as_vector(y_true, "y_true")
    lo = as_vector(lower, "lower", allow_infinite=True)
    hi = as_vector(upper, "upper", allow_infinite=True)
    if not (len(y) == len(lo) == len(hi)):
        raise ValueError("all arrays must have the same length")
    if np.isposinf(lo).any() or np.isneginf(hi).any() or (lo > hi).any():
        raise ValueError("interval bounds must be ordered and contain finite points")
    covered = (y >= lo) & (y <= hi)
    with np.errstate(over="ignore"):
        width = hi - lo
    return {
        "coverage": float(np.mean(covered)),
        "average_width": float(np.mean(width)),
        "median_width": float(np.median(width)),
        "min_width": float(np.min(width)),
        "max_width": float(np.max(width)),
    }
