"""Temperature scaling for matrices of multiclass logits."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax

from calibrated_ml_lab._validation import as_vector, logit_matrix

_LOG_TEMPERATURE_BOUNDS = (-3.0, 3.0)


def _center_logits(logits: np.ndarray) -> np.ndarray:
    """Validate a real matrix and remove row offsets before temperature scaling."""
    values = logit_matrix(logits)
    with np.errstate(over="ignore"):
        centered = values - values.max(axis=1, keepdims=True)
    minimum_temperature = np.exp(_LOG_TEMPERATURE_BOUNDS[0])
    if (
        not np.isfinite(centered).all()
        or centered.min() < -np.finfo(float).max * minimum_temperature
    ):
        raise ValueError("logit differences are too large to scale safely")
    return np.asarray(centered, dtype=float)


class MulticlassTemperatureScaler:
    """Fit one positive temperature for ``softmax(logits / temperature_)``.

    Inputs are finite real matrices, with one column per class. Labels are integer
    column indices from zero. Fit on held-out calibration data, then transform new
    logits with the same class columns in the same order. Missing classes in the
    calibration labels are allowed but cannot establish calibration for those classes.

    The fitted temperature is constrained to [exp(-3), exp(3)]. Identity scaling and
    the endpoints are considered alongside the numerical optimum. Fitting minimizes
    calibration negative log likelihood; improvement on unseen data is not guaranteed.
    """

    def __init__(self) -> None:
        self.temperature_: float | None = None
        self.n_classes_: int | None = None

    def fit(self, logits: np.ndarray, y_true: np.ndarray) -> MulticlassTemperatureScaler:
        """Fit a temperature, preserving a previous successful fit if this call fails."""
        centered = _center_logits(logits)
        y = as_vector(y_true, "y_true")
        n_rows, n_classes = centered.shape
        if len(y) != n_rows:
            raise ValueError("logits and y_true must have the same number of rows")
        if not ((y >= 0) & (y < n_classes) & (y == np.floor(y))).all():
            raise ValueError("y_true must contain integer class indices in [0, n_classes)")
        indices = y.astype(int)
        row_indices = np.arange(n_rows)
        # A fixed positive scale keeps the optimizer's arithmetic finite for very
        # large logit gaps without changing the minimizing temperature.
        loss_scale = max(1.0, float(np.max(-centered)))

        def objective(log_temperature: float) -> float:
            scaled = centered / np.exp(log_temperature)
            losses = logsumexp(scaled, axis=1) - scaled[row_indices, indices]
            return float(np.mean(losses / loss_scale))

        result = minimize_scalar(objective, bounds=_LOG_TEMPERATURE_BOUNDS, method="bounded")
        if (
            not result.success
            or not np.isfinite(result.x)
            or not np.isfinite(result.fun)
            or not _LOG_TEMPERATURE_BOUNDS[0] <= result.x <= _LOG_TEMPERATURE_BOUNDS[1]
        ):
            raise RuntimeError("multiclass temperature optimisation failed")
        candidates = [0.0, float(result.x), *_LOG_TEMPERATURE_BOUNDS]
        log_temperature = min(candidates, key=objective)
        temperature = float(np.exp(log_temperature))
        self.temperature_ = temperature
        self.n_classes_ = n_classes
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Return one normalized class-probability row per input row."""
        if self.temperature_ is None:
            raise RuntimeError("MulticlassTemperatureScaler must be fitted before transform")
        centered = _center_logits(logits)
        if centered.shape[1] != self.n_classes_:
            raise ValueError("logits must have the same number of class columns as the fitted data")
        return np.asarray(softmax(centered / self.temperature_, axis=1), dtype=float)
