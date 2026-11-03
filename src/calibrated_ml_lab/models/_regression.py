"""Shared dense-input and estimator helpers for regression interval wrappers."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, clone

from calibrated_ml_lab._validation import as_vector


def _features(features: np.ndarray) -> np.ndarray:
    """Require a finite dense real matrix, preserving positional feature columns."""
    raw = np.asarray(features)
    if raw.ndim != 2 or 0 in raw.shape:
        raise ValueError("features must be a nonempty matrix with at least one column")
    return as_vector(raw.reshape(-1), "features").reshape(raw.shape)


def _predict(model: BaseEstimator, features: np.ndarray) -> np.ndarray:
    """Validate finite scalar predictions without exposing input arrays to mutation."""
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            predictions = as_vector(model.predict(features.copy()), "predictions")
        if len(predictions) != len(features):
            raise ValueError("predictions must have one value per row")
    except (ValueError, FloatingPointError) as error:
        raise RuntimeError("regressor returned invalid predictions") from error
    return predictions


def _clone_regressor(template: BaseEstimator, seed: int) -> BaseEstimator:
    """Clone an unfitted regressor and fix all exposed estimator random seeds."""
    candidate = clone(template)
    if candidate is template:
        raise ValueError("estimator must clone to a distinct, unfitted regressor")
    seeds = {
        name: seed
        for name in candidate.get_params(deep=True)
        if name == "random_state" or name.endswith("__random_state")
    }
    if seeds:
        candidate.set_params(**seeds)
    return candidate
