"""Empirical calibration diagnostics across nominal prediction-interval levels."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from calibrated_ml_lab._validation import as_vector
from calibrated_ml_lab.conformal.quantile import conformalized_quantile_metrics


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
        raw = np.asarray(nominal)
        if raw.ndim != 0:
            raise ValueError("nominal coverage must be a real scalar")
        level = float(as_vector(raw.reshape(1), "nominal coverage")[0])
        if not 0 < level < 1:
            raise ValueError("nominal coverage must be strictly between zero and one")
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
