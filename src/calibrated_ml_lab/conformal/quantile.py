"""Split conformal calibration of quantile-regression intervals, including shrinkage."""

from __future__ import annotations

import numpy as np

from calibrated_ml_lab._validation import as_vector
from calibrated_ml_lab.conformal._utils import _real_scalar, conformal_quantile


def _base_bounds(lower: np.ndarray, upper: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Require aligned finite ordered base intervals before calibration or correction."""
    lo, hi = as_vector(lower, "lower"), as_vector(upper, "upper")
    if len(lo) != len(hi):
        raise ValueError("lower and upper must have the same length")
    if (lo > hi).any():
        raise ValueError("base interval bounds must be ordered before conformal calibration")
    return lo, hi


def conformalized_quantile_threshold(
    y_calibration: np.ndarray,
    lower_calibration: np.ndarray,
    upper_calibration: np.ndarray,
    alpha: float = 0.1,
) -> float:
    """Return the exact conformal order statistic of max(lower - y, y - upper).

    Scores and the threshold may be negative, allowing overwide intervals to shrink.
    If the required rank exceeds the calibration count, return positive infinity.
    Freeze the quantile models and bound-ordering rule before using fresh calibration
    rows, and use the same rule on exchangeable future observations.
    """
    lo, hi = _base_bounds(lower_calibration, upper_calibration)
    y = as_vector(y_calibration, "y_calibration")
    if len(y) != len(lo):
        raise ValueError("calibration arrays must have the same length")
    with np.errstate(over="ignore"):
        scores = np.maximum(lo - y, y - hi)
    return conformal_quantile(scores, alpha)


def conformalized_quantile_intervals(
    lower: np.ndarray, upper: np.ndarray, qhat: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return (lower - qhat, upper + qhat), preserving empty sets after shrinkage.

    Base bounds must be finite and ordered. qhat may be finite with either sign or
    positive infinity, which returns the full real line. A corrected lower bound
    exceeding its upper bound denotes an empty set: do not sort these endpoints.
    """
    lo, hi = _base_bounds(lower, upper)
    threshold = _real_scalar(qhat, "qhat", allow_infinite=True)
    if np.isneginf(threshold):
        raise ValueError("qhat must be finite or positive infinity")
    with np.errstate(over="ignore"):
        return lo - threshold, hi + threshold


def conformalized_quantile_metrics(
    y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> dict[str, float]:
    """Summarize CQR coverage and widths, retaining empty sets in every denominator.

    Reversed corrected bounds are empty sets with zero coverage and zero width.
    Equal finite bounds are singleton sets and can cover a matching response.
    Unbounded intervals are accepted; NaN and singleton infinities are rejected.
    """
    y = as_vector(y_true, "y_true")
    lo = as_vector(lower, "lower", allow_infinite=True)
    hi = as_vector(upper, "upper", allow_infinite=True)
    if not (len(y) == len(lo) == len(hi)):
        raise ValueError("all arrays must have the same length")
    empty = lo > hi
    if ((~empty) & (np.isposinf(lo) | np.isneginf(hi))).any():
        raise ValueError("nonempty intervals must contain finite points")
    covered = (~empty) & (y >= lo) & (y <= hi)
    with np.errstate(over="ignore"):
        widths = np.where(empty, 0.0, hi - lo)
    max_width = float(np.max(widths))
    average_width = max_width
    if 0 < max_width < np.inf:
        # Scaling avoids overflow of the sum and preserves uniformly tiny widths.
        average_width = float(np.mean(widths / max_width) * max_width)
    middle = len(widths) // 2
    ordered = np.partition(widths, [max(0, middle - 1), middle])
    median_width = float(ordered[middle])
    if len(widths) % 2 == 0 and np.isfinite(median_width):
        left = ordered[middle - 1]
        median_width = float(left + (median_width - left) / 2)
    return {
        "coverage": float(np.mean(covered)),
        "average_width": average_width,
        "median_width": median_width,
        "min_width": float(np.min(widths)),
        "max_width": max_width,
        "empty_set_rate": float(np.mean(empty)),
    }
