"""Split conformal prediction for binary classification."""

from __future__ import annotations

import numpy as np
import pandas as pd

from calibrated_ml_lab.conformal._utils import (
    as_vector,
    binary_labels,
    conformal_quantile,
    nonnegative_threshold,
)
from calibrated_ml_lab.conformal._utils import (
    probabilities as validate_probabilities,
)


def conformal_classification_threshold(
    y_calibration: np.ndarray, probability_calibration: np.ndarray, alpha: float = 0.1
) -> float:
    """Compute a split-conformal score threshold, possibly infinite for small samples."""

    y = binary_labels(y_calibration, "y_calibration")
    p = validate_probabilities(probability_calibration)
    if len(y) != len(p):
        raise ValueError("calibration arrays must have the same length")
    scores = np.where(y == 1, 1 - p, p)
    return conformal_quantile(scores, alpha)


def conformal_prediction_sets(probabilities: np.ndarray, threshold: float) -> pd.DataFrame:
    """Return binary prediction sets; an infinite threshold includes both labels."""

    p = validate_probabilities(probabilities)
    threshold = nonnegative_threshold(threshold, "threshold")
    include_0 = p <= threshold
    include_1 = (1 - p) <= threshold
    set_size = include_0.astype(int) + include_1.astype(int)
    return pd.DataFrame({"include_0": include_0, "include_1": include_1, "set_size": set_size})


def classification_set_coverage(
    y_true: np.ndarray, prediction_sets: pd.DataFrame
) -> dict[str, float]:
    """Compute empirical coverage and average set size."""

    y = binary_labels(y_true, "y_true")
    if len(y) != len(prediction_sets):
        raise ValueError("y_true and prediction_sets must have the same length")
    required = {"include_0", "include_1", "set_size"}
    if not required.issubset(prediction_sets.columns):
        raise ValueError("prediction_sets must contain include_0, include_1, and set_size")
    include_0 = binary_labels(prediction_sets["include_0"].to_numpy(), "include_0")
    include_1 = binary_labels(prediction_sets["include_1"].to_numpy(), "include_1")
    size = as_vector(prediction_sets["set_size"].to_numpy(), "set_size")
    if not np.array_equal(size, include_0 + include_1):
        raise ValueError("set_size must equal include_0 + include_1")
    covered = np.where(y == 1, include_1, include_0)
    return {
        "coverage": float(np.mean(covered)),
        "average_set_size": float(np.mean(size)),
        "singleton_rate": float(np.mean(size == 1)),
        "empty_set_rate": float(np.mean(size == 0)),
    }
