"""Split-conformal prediction sets for multiclass probability matrices."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from calibrated_ml_lab._validation import (
    binary_labels,
    class_indices,
    multiclass_inputs,
    probability_matrix,
)
from calibrated_ml_lab.conformal._utils import conformal_quantile, nonnegative_threshold


def multiclass_conformal_threshold(
    y_calibration: np.ndarray,
    probability_calibration: np.ndarray,
    alpha: float = 0.1,
) -> float:
    """Return the finite-sample threshold of scores 1 - p(true class).

    The model and any probability calibrator must be fixed before using these rows.
    Use fresh, exchangeable conformal-calibration and test observations. If the
    required score rank exceeds the calibration row count, return positive infinity.
    """
    y, p = multiclass_inputs(y_calibration, probability_calibration)
    scores = 1.0 - p[np.arange(len(y)), y]
    return conformal_quantile(scores, alpha)


def multiclass_prediction_sets(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    """Return a boolean (rows, classes) membership matrix, including score ties.

    Include a class when 1 - p(class) <= threshold, using the same arithmetic as
    threshold fitting. Empty sets are retained; positive infinity includes every
    class. Preserve the class-column mapping used for conformal calibration.
    """
    p = probability_matrix(probabilities, "probabilities")
    threshold = nonnegative_threshold(threshold, "threshold")
    return np.asarray((1.0 - p) <= threshold, dtype=bool)


def _set_inputs(y_true: np.ndarray, prediction_sets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate aligned class indices and a nonempty set-membership matrix."""
    raw = np.asarray(prediction_sets)
    if raw.ndim != 2 or raw.shape[0] == 0 or raw.shape[1] < 2:
        raise ValueError("prediction_sets must be a nonempty matrix with at least 2 class columns")
    membership = binary_labels(raw.reshape(-1), "prediction_sets").reshape(raw.shape)
    y = class_indices(y_true, membership.shape[1])
    if len(y) != len(membership):
        raise ValueError("y_true and prediction_sets must have the same number of rows")
    return y, membership


def _coverage_summary(covered: np.ndarray, sizes: np.ndarray) -> dict[str, float]:
    """Summarize validated observations, leaving empty-group statistics undefined."""
    if len(covered) == 0:
        return {
            "coverage": float("nan"),
            "average_set_size": float("nan"),
            "singleton_rate": float("nan"),
            "empty_set_rate": float("nan"),
        }
    return {
        "coverage": float(covered.mean()),
        "average_set_size": float(sizes.mean()),
        "singleton_rate": float((sizes == 1).mean()),
        "empty_set_rate": float((sizes == 0).mean()),
    }


def multiclass_set_coverage(y_true: np.ndarray, prediction_sets: np.ndarray) -> dict[str, float]:
    """Summarize empirical coverage, average size, singleton rate, and empty-set rate.

    Membership must be a nonempty matrix of booleans or numeric 0/1 values, with
    at least two class columns. Empirical coverage need not equal nominal coverage,
    and a marginal coverage guarantee does not imply class-conditional coverage.
    """
    y, membership = _set_inputs(y_true, prediction_sets)
    covered = membership[np.arange(len(y)), y]
    return _coverage_summary(covered, membership.sum(axis=1))


def multiclass_coverage_by_class(y_true: np.ndarray, prediction_sets: np.ndarray) -> pd.DataFrame:
    """Report coverage by true class, including absent classes with zero counts and NaN rates.

    Every class column appears in index order. Each row includes count, covered_count,
    coverage, average_set_size, singleton_rate, and empty_set_rate. These are
    descriptive diagnostics, not class-conditional coverage guarantees.
    """
    y, membership = _set_inputs(y_true, prediction_sets)
    covered = membership[np.arange(len(y)), y]
    sizes = membership.sum(axis=1)
    rows = []
    for class_index in range(membership.shape[1]):
        selected = y == class_index
        rows.append(
            {
                "class_index": class_index,
                "count": int(selected.sum()),
                "covered_count": int(covered[selected].sum()),
                **_coverage_summary(covered[selected], sizes[selected]),
            }
        )
    return pd.DataFrame(rows)


def multiclass_coverage_by_slice(
    y_true: np.ndarray,
    prediction_sets: np.ndarray,
    slices: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    """Report empirical coverage for named, positionally aligned binary masks.

    Preserve mapping iteration order. Masks may overlap or omit observations; their
    counts must not be pooled as if they partitioned the data. All-false masks retain
    zero counts and NaN statistics. An empty mapping returns an empty table with
    the usual columns. Names must be nonblank strings, and masks must be nonempty
    vectors of booleans or numeric 0/1 values with one entry per evaluation row.
    """
    y, membership = _set_inputs(y_true, prediction_sets)
    if not isinstance(slices, Mapping):
        raise ValueError("slices must be a mapping from names to binary masks")
    covered = membership[np.arange(len(y)), y]
    sizes = membership.sum(axis=1)
    rows = []
    for name, mask in slices.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("slice names must be nonblank strings")
        selected = binary_labels(mask, f"slice '{name}'").astype(bool)
        if len(selected) != len(y):
            raise ValueError(f"slice '{name}' must have one mask entry per evaluation row")
        rows.append(
            {
                "slice": name,
                "count": int(selected.sum()),
                "covered_count": int(covered[selected].sum()),
                **_coverage_summary(covered[selected], sizes[selected]),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "slice",
            "count",
            "covered_count",
            "coverage",
            "average_set_size",
            "singleton_rate",
            "empty_set_rate",
        ],
    )
