"""Shared validation and finite-sample quantiles for conformal prediction."""

from __future__ import annotations

import math

import numpy as np

from calibrated_ml_lab._validation import as_vector


def nonnegative_threshold(value: float, name: str) -> float:
    """Accept a nonnegative scalar threshold, including the +infinity fallback."""
    if np.ndim(value) != 0 or np.iscomplexobj(value):
        raise ValueError(f"{name} must be a nonnegative real scalar")
    threshold = float(value)
    if np.isnan(threshold) or threshold < 0:
        raise ValueError(f"{name} must be non-negative and not NaN")
    return threshold


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Return the ceil((n+1)*(1-alpha))-th score, or infinity when its rank exceeds n."""
    if np.ndim(alpha) != 0 or not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    scores = as_vector(scores, "nonconformity scores")
    count = len(scores) + 1
    if alpha < 1 / count:
        return float("inf")
    # Equivalent to ceil(count * (1 - alpha)), with less subtraction cancellation.
    # The guard above establishes a finite rank; clamp multiplication roundoff at n.
    rank = min(len(scores), count - math.floor(count * alpha))
    # np.quantile(..., method="higher") uses a different index convention.
    return float(np.partition(scores, rank - 1)[rank - 1])
