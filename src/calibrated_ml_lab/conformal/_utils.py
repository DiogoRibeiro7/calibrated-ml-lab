"""Shared validation and finite-sample quantiles for conformal prediction."""

from __future__ import annotations

import math

import numpy as np

from calibrated_ml_lab._validation import as_vector


def _real_scalar(value: float, name: str, *, allow_infinite: bool = False) -> float:
    """Validate a numeric scalar without coercing strings or one-element vectors."""
    array = np.asarray(value)
    if array.ndim != 0:
        raise ValueError(f"{name} must be a real scalar")
    return float(as_vector(array.reshape(1), name, allow_infinite=allow_infinite)[0])


def nonnegative_threshold(value: float, name: str) -> float:
    """Accept a nonnegative scalar threshold, including the +infinity fallback."""
    threshold = _real_scalar(value, name, allow_infinite=True)
    if threshold < 0:
        raise ValueError(f"{name} must be non-negative and not NaN")
    return threshold


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Return the ceil((n+1)*(1-alpha))-th score, or infinity when its rank exceeds n."""
    alpha = _real_scalar(alpha, "alpha")
    if not 0 < alpha < 1:
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
