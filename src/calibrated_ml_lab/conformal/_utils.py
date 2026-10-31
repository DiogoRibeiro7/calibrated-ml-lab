"""Shared validation and finite-sample quantiles for conformal prediction."""

from __future__ import annotations

import math

import numpy as np


def as_vector(values: np.ndarray, name: str, *, allow_infinite: bool = False) -> np.ndarray:
    """Require a nonempty real vector, allowing infinite interval bounds explicitly."""
    if np.iscomplexobj(values):
        raise ValueError(f"{name} must be real-valued")
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    if np.isnan(array).any() or (not allow_infinite and not np.isfinite(array).all()):
        raise ValueError(
            f"{name} must contain {'no NaN values' if allow_infinite else 'finite values'}"
        )
    return array


def binary_labels(values: np.ndarray, name: str) -> np.ndarray:
    """Validate binary labels without truncating fractional values to integers."""
    labels = as_vector(values, name)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError(f"{name} must contain only binary labels 0 and 1")
    return labels


def probabilities(values: np.ndarray) -> np.ndarray:
    """Validate probabilities while preserving exact zero and one endpoints."""
    values = as_vector(values, "probabilities")
    if not ((values >= 0) & (values <= 1)).all():
        raise ValueError("probabilities must be in [0, 1]")
    return values


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
