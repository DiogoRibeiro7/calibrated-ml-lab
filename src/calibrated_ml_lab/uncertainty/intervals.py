"""Empirical prediction-interval diagnostics across nominal levels and feature bins."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from calibrated_ml_lab._validation import as_vector
from calibrated_ml_lab.conformal.quantile import conformalized_quantile_metrics


def _nominal_coverage(value: float) -> float:
    """Require a real scalar coverage level strictly between zero and one."""
    raw = np.asarray(value)
    if raw.ndim != 0:
        raise ValueError("nominal coverage must be a real scalar")
    level = float(as_vector(raw.reshape(1), "nominal coverage")[0])
    if not 0 < level < 1:
        raise ValueError("nominal coverage must be strictly between zero and one")
    return level


def interval_calibration_table(
    y_true: np.ndarray,
    intervals: Mapping[float, tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    """Summarize supplied intervals at each nominal coverage, sorted by nominal level.

    Each mapping entry contains (lower, upper) arrays for the same evaluation rows.
    Levels must be distinct real scalars strictly between zero and one. Reversed
    bounds denote empty sets; they count as uncovered with zero width. Unbounded
    intervals remain included. This function does not fit or select any intervals.
    """
    y = as_vector(y_true, "y_true")
    if not isinstance(intervals, Mapping) or not intervals:
        raise ValueError("intervals must be a nonempty mapping from nominal coverage to bounds")
    rows = []
    levels: set[float] = set()
    for nominal, bounds in intervals.items():
        level = _nominal_coverage(nominal)
        if level in levels:
            raise ValueError("nominal coverage levels must remain distinct as float values")
        levels.add(level)
        if not isinstance(bounds, tuple | list) or len(bounds) != 2:
            raise ValueError("each nominal coverage must have a (lower, upper) pair")
        lo, hi = (
            as_vector(bounds[0], "lower", allow_infinite=True),
            as_vector(bounds[1], "upper", allow_infinite=True),
        )
        summary = conformalized_quantile_metrics(y, lo, hi)
        covered_count = int(np.count_nonzero((y >= lo) & (y <= hi)))
        rows.append(
            {
                "nominal_coverage": level,
                "count": len(y),
                "covered_count": covered_count,
                **summary,
                "coverage_gap": summary["coverage"] - level,
            }
        )
    return pd.DataFrame(rows).sort_values("nominal_coverage", ignore_index=True)


def interval_coverage_by_feature(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    feature_values: np.ndarray,
    bin_edges: np.ndarray,
    *,
    nominal_coverage: float,
) -> pd.DataFrame:
    """Report empirical interval coverage and widths within fixed numeric feature bins.

    Supply aligned finite responses and feature values. Edges must be strictly
    increasing, with optional infinite outer endpoints, and contain every feature
    value. Bins are [left, right), except the final bin includes its right edge.
    Empty bins retain zero counts and NaN statistics; empty prediction sets count
    as uncovered with zero width. Alignment is positional, including Pandas inputs.
    Freeze bins before evaluation. These diagnostics do not guarantee conditional
    coverage, fit thresholds, or choose bins from evaluation outcomes.
    """
    y = as_vector(y_true, "y_true")
    lo = as_vector(lower, "lower", allow_infinite=True)
    hi = as_vector(upper, "upper", allow_infinite=True)
    feature = as_vector(feature_values, "feature_values")
    if not (len(y) == len(lo) == len(hi) == len(feature)):
        raise ValueError("responses, bounds, and feature_values must have the same length")
    edges = as_vector(bin_edges, "bin_edges", allow_infinite=True)
    if len(edges) < 2 or (edges[1:] <= edges[:-1]).any():
        raise ValueError("bin_edges must contain at least two strictly increasing values")
    if (feature < edges[0]).any() or (feature > edges[-1]).any():
        raise ValueError("bin_edges must contain every feature value; extend the outer edges")
    level = _nominal_coverage(nominal_coverage)
    # Validate all prediction sets before partitioning and retain the shared metric schema.
    overall = conformalized_quantile_metrics(y, lo, hi)
    assignments = np.searchsorted(edges[1:-1], feature, side="right")
    covered = (y >= lo) & (y <= hi)
    rows = []
    for index in range(len(edges) - 1):
        selected = assignments == index
        count = int(np.count_nonzero(selected))
        summary = (
            conformalized_quantile_metrics(y[selected], lo[selected], hi[selected])
            if count
            else {name: float("nan") for name in overall}
        )
        rows.append(
            {
                "bin_index": index,
                "bin_lower": edges[index],
                "bin_upper": edges[index + 1],
                "right_closed": index == len(edges) - 2,
                "count": count,
                "fraction": count / len(y),
                "covered_count": int(np.count_nonzero(covered[selected])),
                "nominal_coverage": level,
                **summary,
                "coverage_gap": summary["coverage"] - level,
            }
        )
    return pd.DataFrame(rows)
