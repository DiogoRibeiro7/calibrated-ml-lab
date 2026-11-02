"""Feature-dependent prediction intervals from two quantile regression models."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from calibrated_ml_lab._validation import as_vector


def _scalar(value: float, name: str) -> float:
    """Require a finite real scalar without accepting booleans or numeric strings."""
    raw = np.asarray(value)
    if raw.ndim != 0 or isinstance(raw.item(), bool | np.bool_):
        raise ValueError(f"{name} must be a finite real scalar")
    return float(as_vector(raw.reshape(1), name)[0])


def _integer(value: int, name: str, minimum: int = 1) -> int:
    """Require an integer hyperparameter with an explicit lower bound."""
    if (
        isinstance(value, bool | np.bool_)
        or not isinstance(value, int | np.integer)
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _features(features: np.ndarray) -> np.ndarray:
    """Validate a dense numeric matrix compatible with the trees' float32 inputs."""
    raw = np.asarray(features)
    if raw.ndim != 2 or 0 in raw.shape:
        raise ValueError("features must be a nonempty matrix with at least one column")
    values = as_vector(raw.reshape(-1), "features").reshape(raw.shape)
    if (np.abs(values) > np.finfo(np.float32).max).any():
        raise ValueError("features must be representable as finite float32 values")
    return values


def _predict(model: GradientBoostingRegressor, features: np.ndarray) -> np.ndarray:
    """Reject malformed or nonfinite model output before returning interval endpoints."""
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            predictions = as_vector(model.predict(features), "quantile predictions")
        if len(predictions) != len(features):
            raise ValueError("quantile predictions must have one value per row")
    except (ValueError, FloatingPointError) as error:
        raise RuntimeError("quantile model returned invalid predictions") from error
    return predictions


class QuantileIntervalRegressor:
    """Fit two gradient-boosted quantiles for a nominal central prediction interval.

    Fit the alpha/2 and 1-alpha/2 conditional quantiles with pinball loss. The default
    alpha=0.1 targets 90% coverage, without a finite-sample coverage guarantee.
    Quantiles are fitted independently and may cross. ``predict_quantiles`` retains
    their original identities; ``predict_interval`` orders each pair of endpoints.

    Features are finite dense numeric matrices with consistent positional columns;
    targets are finite real vectors. Fit preprocessing on training rows separately.
    Tune model settings on development data and evaluate coverage on untouched test
    rows. This wrapper exposes fit/prediction methods, not the full sklearn estimator API.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        *,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        max_depth: int = 3,
        min_samples_leaf: int = 9,
        random_state: int = 42,
    ) -> None:
        self.alpha = alpha
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.lower_model_: GradientBoostingRegressor | None = None
        self.upper_model_: GradientBoostingRegressor | None = None
        self.n_features_in_: int | None = None
        self.quantiles_: tuple[float, float] | None = None
        self.alpha_: float | None = None
        self._validate_options()

    def _validate_options(self) -> tuple[float, float, float]:
        """Validate both construction and refits after option changes."""
        alpha = _scalar(self.alpha, "alpha")
        lower, upper = alpha / 2, 1 - alpha / 2
        if not 0 < lower < upper < 1:
            raise ValueError("alpha must yield distinct quantiles strictly between 0 and 1")
        if _scalar(self.learning_rate, "learning_rate") <= 0:
            raise ValueError("learning_rate must be positive")
        for name in ("n_estimators", "max_depth", "min_samples_leaf"):
            _integer(getattr(self, name), name)
        if _integer(self.random_state, "random_state", 0) > 2**32 - 1:
            raise ValueError("random_state must be at most 2**32 - 1")
        return alpha, lower, upper

    def fit(self, features: np.ndarray, y_true: np.ndarray) -> QuantileIntervalRegressor:
        """Fit both quantile models, keeping any previous successful fit on failure."""
        alpha, lower, upper = self._validate_options()
        values = _features(features)
        targets = as_vector(y_true, "y_true")
        if len(values) != len(targets):
            raise ValueError("features and y_true must have the same number of rows")
        models = []
        for quantile in (lower, upper):
            model = GradientBoostingRegressor(
                loss="quantile",
                alpha=quantile,
                n_estimators=int(self.n_estimators),
                learning_rate=float(self.learning_rate),
                max_depth=int(self.max_depth),
                min_samples_leaf=int(self.min_samples_leaf),
                random_state=int(self.random_state),
            )
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    model.fit(values, targets)
            except FloatingPointError as error:
                raise RuntimeError("quantile regression fitting failed numerically") from error
            _predict(model, values)
            models.append(model)
        self.lower_model_, self.upper_model_ = models
        self.n_features_in_ = values.shape[1]
        self.quantiles_, self.alpha_ = (lower, upper), alpha
        return self

    def predict_quantiles(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return raw lower-quantile and upper-quantile predictions, retaining crossings."""
        if self.lower_model_ is None or self.upper_model_ is None:
            raise RuntimeError("QuantileIntervalRegressor must be fitted before prediction")
        values = _features(features)
        if values.shape[1] != self.n_features_in_:
            raise ValueError("features must have the same number of columns as the fitted data")
        return _predict(self.lower_model_, values), _predict(self.upper_model_, values)

    def predict_interval(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ordered bounds by swapping any crossed pair, without a coverage correction."""
        lower, upper = self.predict_quantiles(features)
        return np.minimum(lower, upper), np.maximum(lower, upper)
