"""Split-conformal prediction sets for multiclass probability matrices."""

from __future__ import annotations

import numpy as np

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


def multiclass_set_coverage(y_true: np.ndarray, prediction_sets: np.ndarray) -> dict[str, float]:
    """Summarize empirical coverage, average size, singleton rate, and empty-set rate.

    Membership must be a nonempty matrix of booleans or numeric 0/1 values, with
    at least two class columns. Empirical coverage need not equal nominal coverage,
    and a marginal coverage guarantee does not imply class-conditional coverage.
    """
    raw = np.asarray(prediction_sets)
    if raw.ndim != 2 or raw.shape[0] == 0 or raw.shape[1] < 2:
        raise ValueError("prediction_sets must be a nonempty matrix with at least 2 class columns")
    membership = binary_labels(raw.reshape(-1), "prediction_sets").reshape(raw.shape)
    y = class_indices(y_true, membership.shape[1])
    if len(y) != len(membership):
        raise ValueError("y_true and prediction_sets must have the same number of rows")
    sizes = membership.sum(axis=1)
    return {
        "coverage": float(membership[np.arange(len(y)), y].mean()),
        "average_set_size": float(sizes.mean()),
        "singleton_rate": float((sizes == 1).mean()),
        "empty_set_rate": float((sizes == 0).mean()),
    }
