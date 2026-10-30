"""Split conformal prediction for binary classification."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def conformal_classification_threshold(
    y_calibration: np.ndarray, probability_calibration: np.ndarray, alpha: float = 0.1
) -> float:
    """Compute split-conformal threshold using nonconformity score 1 - p_true."""

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    y = np.asarray(y_calibration, dtype=int)
    p = np.asarray(probability_calibration, dtype=float)
    if len(y) != len(p) or len(y) == 0:
        raise ValueError("calibration arrays must have the same non-zero length")
    p_true = np.where(y == 1, p, 1 - p)
    scores = 1 - p_true
    n = len(scores)
    q_level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, q_level, method="higher"))


def conformal_prediction_sets(probabilities: np.ndarray, threshold: float) -> pd.DataFrame:
    """Return binary conformal prediction sets for each probability."""

    p = np.asarray(probabilities, dtype=float)
    if not np.all((p >= 0) & (p <= 1)):
        raise ValueError("probabilities must be in [0, 1]")
    include_0 = (1 - (1 - p)) <= threshold  # same as p <= threshold
    include_1 = (1 - p) <= threshold
    set_size = include_0.astype(int) + include_1.astype(int)
    return pd.DataFrame({"include_0": include_0, "include_1": include_1, "set_size": set_size})


def classification_set_coverage(
    y_true: np.ndarray, prediction_sets: pd.DataFrame
) -> dict[str, float]:
    """Compute empirical coverage and average set size."""

    y = np.asarray(y_true, dtype=int)
    if len(y) != len(prediction_sets):
        raise ValueError("y_true and prediction_sets must have the same length")
    covered = np.where(
        y == 1, prediction_sets["include_1"].to_numpy(), prediction_sets["include_0"].to_numpy()
    )
    return {
        "coverage": float(np.mean(covered)),
        "average_set_size": float(prediction_sets["set_size"].mean()),
        "singleton_rate": float(np.mean(prediction_sets["set_size"] == 1)),
        "empty_set_rate": float(np.mean(prediction_sets["set_size"] == 0)),
    }
