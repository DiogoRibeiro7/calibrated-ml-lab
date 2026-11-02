"""Shared array validation for probability evaluation and conformal prediction."""

from __future__ import annotations

from numbers import Number

import numpy as np


def as_vector(values: np.ndarray, name: str, *, allow_infinite: bool = False) -> np.ndarray:
    """Require a nonempty real vector, allowing infinite interval bounds explicitly."""
    if np.iscomplexobj(values):
        raise ValueError(f"{name} must be real-valued")
    raw = np.asarray(values)
    if raw.dtype.kind not in "biuf" and (
        raw.dtype.kind != "O"
        or not all(
            isinstance(value, Number | np.bool_)
            and not isinstance(value, complex | np.complexfloating | np.timedelta64)
            for value in raw.flat
        )
    ):
        raise ValueError(f"{name} must contain real numeric values")
    array = np.asarray(raw, dtype=float)
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


def class_indices(values: np.ndarray, n_classes: int, name: str = "y_true") -> np.ndarray:
    """Validate a vector of integer-valued class-column indices."""
    y = as_vector(values, name)
    if not ((y >= 0) & (y < n_classes) & (y == np.floor(y))).all():
        raise ValueError(f"{name} must contain integer class indices in [0, n_classes)")
    return y.astype(int)


def probability_matrix(values: np.ndarray, name: str = "y_prob") -> np.ndarray:
    """Validate multiclass probability rows without clipping or renormalizing them."""
    raw = np.asarray(values)
    if raw.ndim != 2 or raw.shape[0] == 0 or raw.shape[1] < 2:
        raise ValueError(f"{name} must be a nonempty matrix with at least 2 class columns")
    p = as_vector(raw.reshape(-1), name).reshape(raw.shape)
    if not ((p >= 0) & (p <= 1)).all():
        raise ValueError(f"{name} must contain probabilities in [0, 1]")
    if not np.isclose(p.sum(axis=1), 1.0, rtol=0.0, atol=1e-6).all():
        raise ValueError(f"each {name} row must sum to 1 within an absolute tolerance of 1e-6")
    return np.asarray(p, dtype=float)


def multiclass_inputs(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate aligned class indices and multiclass probability rows."""
    p = probability_matrix(y_prob)
    y = class_indices(y_true, p.shape[1])
    if len(y) != len(p):
        raise ValueError("y_true and y_prob must have the same number of rows")
    return y, p
