"""Shared array validation for probability evaluation and conformal prediction."""

from __future__ import annotations

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


def probabilities(values: np.ndarray, name: str = "probabilities") -> np.ndarray:
    """Validate probabilities while preserving exact zero and one endpoints."""
    values = as_vector(values, name)
    if not ((values >= 0) & (values <= 1)).all():
        raise ValueError(f"{name} must be in [0, 1]")
    return values


def binary_inputs(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate aligned binary labels and probabilities without clipping their values."""
    y = binary_labels(y_true, "y_true")
    p = probabilities(y_prob)
    if len(y) != len(p):
        raise ValueError("y_true and y_prob must have the same length")
    return y, p
