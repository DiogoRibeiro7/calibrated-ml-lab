"""Post-hoc probability calibration methods."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from calibrated_ml_lab._validation import binary_inputs
from calibrated_ml_lab._validation import probabilities as validate_probabilities

EPS = 1e-12


def _clip_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Validate probabilities and clip them for finite logit calculations."""

    p = validate_probabilities(probabilities)
    return np.clip(p, EPS, 1 - EPS)


class PlattScaler:
    """Binary Platt scaling fitted on logit probabilities."""

    def __init__(self) -> None:
        self.model: LogisticRegression | None = None

    def fit(self, probabilities: np.ndarray, y_true: np.ndarray) -> PlattScaler:
        """Fit on binary data with both classes, preserving the previous fit on failure."""

        y, p = binary_inputs(y_true, probabilities)
        if len(np.unique(y)) < 2:
            raise ValueError("Platt scaling requires both classes")
        logits = logit(_clip_probabilities(p)).reshape(-1, 1)
        fitted = LogisticRegression(solver="lbfgs")
        fitted.fit(logits, y)
        self.model = fitted
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities."""

        if self.model is None:
            raise RuntimeError("PlattScaler must be fitted before transform")
        p = _clip_probabilities(probabilities)
        return np.asarray(self.model.predict_proba(logit(p).reshape(-1, 1))[:, 1], dtype=float)


class IsotonicCalibrator:
    """Monotonic isotonic probability calibrator."""

    def __init__(self) -> None:
        self.model: IsotonicRegression | None = None

    def fit(self, probabilities: np.ndarray, y_true: np.ndarray) -> IsotonicCalibrator:
        """Fit on original probabilities, preserving the previous fit on failure."""

        y, p = binary_inputs(y_true, probabilities)
        fitted = IsotonicRegression(out_of_bounds="clip")
        fitted.fit(p, y)
        self.model = fitted
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities."""

        if self.model is None:
            raise RuntimeError("IsotonicCalibrator must be fitted before transform")
        p = validate_probabilities(probabilities)
        return np.asarray(self.model.predict(p), dtype=float)


class TemperatureScaler:
    """Binary temperature scaling on logits.

    A temperature greater than 1 softens overconfident probabilities. A temperature below
    1 sharpens probabilities. The parameter is fitted by minimising negative log loss.
    """

    def __init__(self) -> None:
        self.temperature_: float | None = None

    def fit(self, probabilities: np.ndarray, y_true: np.ndarray) -> TemperatureScaler:
        """Fit on binary data, preserving the previous temperature on failure."""

        y, p = binary_inputs(y_true, probabilities)
        logits = logit(_clip_probabilities(p))

        def objective(log_temperature: float) -> float:
            temperature = float(np.exp(log_temperature))
            calibrated = expit(logits / temperature)
            return float(log_loss(y, calibrated, labels=[0, 1]))

        result = minimize_scalar(objective, bounds=(-3.0, 3.0), method="bounded")
        if not result.success:
            raise RuntimeError("temperature optimisation failed")
        self.temperature_ = float(np.exp(result.x))
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        """Return temperature-scaled probabilities."""

        if self.temperature_ is None:
            raise RuntimeError("TemperatureScaler must be fitted before transform")
        p = _clip_probabilities(probabilities)
        return np.asarray(expit(logit(p) / self.temperature_), dtype=float)
