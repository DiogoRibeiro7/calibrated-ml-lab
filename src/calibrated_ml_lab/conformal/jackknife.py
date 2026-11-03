"""Leave-one-out jackknife-plus prediction intervals with finite-sample ranks."""

from __future__ import annotations

import math

import numpy as np
from sklearn.base import BaseEstimator, is_regressor
from sklearn.linear_model import LinearRegression

from calibrated_ml_lab._validation import as_vector
from calibrated_ml_lab.conformal._utils import _real_scalar
from calibrated_ml_lab.models._regression import _clone_regressor, _features, _predict


def _alpha(value: float) -> float:
    """Require a nonvacuous jackknife-plus error level."""
    alpha = _real_scalar(value, "alpha")
    if not 0 < alpha < 0.5:
        raise ValueError("alpha must be strictly between zero and 0.5")
    return alpha


def _lower_rank(n: int, alpha: float) -> int:
    """Compute floor((n + 1) * alpha), respecting represented k/(n + 1) boundaries."""
    count = n + 1
    rank = math.floor(count * alpha)
    # Multiplication may round a neighboring float onto an integer boundary.
    if alpha < rank / count:
        rank -= 1
    elif alpha >= (rank + 1) / count:
        rank += 1
    return rank


def jackknife_plus_intervals(
    loo_predictions: np.ndarray, loo_residuals: np.ndarray, alpha: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate paired leave-one-out predictions and absolute held-out residuals.

    Predictions have shape (n_training_rows, n_query_rows); residual i must come
    from the same model as prediction row i. Inputs must be finite, with n >= 2
    and nonnegative residuals. Endpoints use the floor((n+1)*alpha)-th lower
    candidate and ceil((n+1)*(1-alpha))-th upper candidate, without interpolation.
    A missing rank returns the full real line. This uses the conventional nominal
    target 1-alpha; the general marginal coverage bound is only 1-2*alpha under
    exchangeability and a learner invariant to training-row order.
    """
    level = _alpha(alpha)
    raw = np.asarray(loo_predictions)
    if raw.ndim != 2 or raw.shape[0] < 2 or raw.shape[1] == 0:
        raise ValueError("loo_predictions must have shape (n >= 2, n_query_rows >= 1)")
    predictions = as_vector(raw.reshape(-1), "loo_predictions").reshape(raw.shape)
    residuals = as_vector(loo_residuals, "loo_residuals")
    if len(residuals) != len(predictions) or (residuals < 0).any():
        raise ValueError("loo_residuals must be nonnegative with one value per prediction row")
    rank = _lower_rank(len(residuals), level)
    if rank == 0:
        return np.full(raw.shape[1], -np.inf), np.full(raw.shape[1], np.inf)
    # Nonnegative finite radii can overflow only toward conservative infinite bounds.
    with np.errstate(over="ignore"):
        lower_candidates = predictions - residuals[:, None]
        upper_candidates = predictions + residuals[:, None]
    lower = np.partition(lower_candidates, rank - 1, axis=0)[rank - 1]
    upper_index = len(residuals) - rank
    upper = np.partition(upper_candidates, upper_index, axis=0)[upper_index]
    return lower, upper


class JackknifePlusRegressor:
    """Fit one model per omitted training row and a full-data model for point predictions.

    The default learner is ordinary linear regression. Nominal coverage is 1-alpha;
    the distribution-free lower bound is 1-2*alpha, requiring exchangeable rows and
    a training-order-invariant learner. A fixed seed alone does not ensure that
    symmetry. Use separate development data for tuning and untouched test labels
    for evaluation. This wrapper is not a full scikit-learn estimator.
    """

    def __init__(
        self,
        estimator: BaseEstimator | None = None,
        *,
        alpha: float = 0.1,
        random_state: int = 42,
    ) -> None:
        self.estimator = estimator
        self.alpha = alpha
        self.random_state = random_state
        self.estimator_: BaseEstimator | None = None
        self.loo_estimators_: tuple[BaseEstimator, ...] = ()
        self.loo_residuals_: np.ndarray | None = None
        self.n_features_in_: int | None = None
        self.alpha_: float | None = None
        self._validate_options()

    def _validate_options(self) -> tuple[float, int]:
        """Validate construction and refits after option changes."""
        alpha = _alpha(self.alpha)
        seed = self.random_state
        if (
            isinstance(seed, bool | np.bool_)
            or not isinstance(seed, int | np.integer)
            or not 0 <= seed <= 2**32 - 1
        ):
            raise ValueError("random_state must be an integer from zero to 2**32 - 1")
        if self.estimator is not None and (
            not isinstance(self.estimator, BaseEstimator) or not is_regressor(self.estimator)
        ):
            raise ValueError("estimator must be a scikit-learn regressor or None")
        return alpha, int(seed)

    def fit(self, features: np.ndarray, y_true: np.ndarray) -> JackknifePlusRegressor:
        """Fit n+1 cloned regressors, preserving previous fitted state if any fit fails.

        Preprocessing in a supplied pipeline is refitted inside every omission.
        All exposed estimator random_state parameters receive the wrapper seed.
        Input arrays and the supplied estimator are preserved. At least two finite
        training rows are required, and the learner must support n-1 rows.
        """
        alpha, seed = self._validate_options()
        values, targets = _features(features), as_vector(y_true, "y_true")
        if len(values) != len(targets):
            raise ValueError("features and y_true must have the same number of rows")
        if len(targets) < 2:
            raise ValueError("at least two training rows are required")
        template = LinearRegression() if self.estimator is None else self.estimator
        model = _clone_regressor(template, seed)
        models, residuals = [], np.empty(len(targets))
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                model.fit(values.copy(), targets.copy())
                _predict(model, values)
                for index in range(len(targets)):
                    selected = np.arange(len(targets)) != index
                    candidate = _clone_regressor(template, seed)
                    candidate.fit(values[selected], targets[selected])
                    prediction = _predict(candidate, values[index : index + 1])[0]
                    residuals[index] = abs(targets[index] - prediction)
                    models.append(candidate)
        except FloatingPointError as error:
            raise RuntimeError("jackknife-plus fitting failed numerically") from error
        self.estimator_, self.loo_estimators_ = model, tuple(models)
        self.loo_residuals_, self.n_features_in_, self.alpha_ = residuals, values.shape[1], alpha
        return self

    def _prediction_features(self, features: np.ndarray) -> np.ndarray:
        """Require a successful fit and aligned feature columns."""
        if self.estimator_ is None:
            raise RuntimeError("JackknifePlusRegressor must be fitted before prediction")
        values = _features(features)
        if values.shape[1] != self.n_features_in_:
            raise ValueError("features must have the same number of columns as the fitted data")
        return values

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return predictions from the full-data model, separate from interval aggregation."""
        values = self._prediction_features(features)
        return _predict(self.estimator_, values)

    def predict_interval(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return exact jackknife-plus endpoints using each paired leave-one-out model."""
        values = self._prediction_features(features)
        assert self.loo_residuals_ is not None and self.alpha_ is not None
        predictions = np.stack([_predict(model, values) for model in self.loo_estimators_])
        return jackknife_plus_intervals(predictions, self.loo_residuals_, self.alpha_)
