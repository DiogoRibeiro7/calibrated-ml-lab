"""Fixed-design residual bootstrap prediction intervals for regression."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, clone, is_regressor
from sklearn.linear_model import LinearRegression

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
        raise RuntimeError("bootstrap estimator returned invalid predictions") from error
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


class ResidualBootstrapRegressor:
    """Refit regressors on resampled residuals and include future-observation noise.

    Fit a base regressor, center its training residuals, and refit ``n_bootstrap``
    clones on fitted values plus resampled residuals at the original features.
    Prediction intervals take percentile endpoints of each clone's prediction plus
    a separate residual draw. The default estimator is ordinary linear regression.

    This assumes a suitable mean model with IID additive errors independent of the
    features. It offers no finite-sample or conditional coverage guarantee. Pooled
    fitted residuals can understate noise, especially with small or overfit samples.
    This wrapper exposes fit/prediction methods, not the full sklearn estimator API.
    """

    def __init__(
        self,
        estimator: BaseEstimator | None = None,
        *,
        alpha: float = 0.1,
        n_bootstrap: int = 200,
        random_state: int = 42,
    ) -> None:
        self.estimator = estimator
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        self.estimator_: BaseEstimator | None = None
        self.bootstrap_estimators_: tuple[BaseEstimator, ...] = ()
        self.residuals_: np.ndarray | None = None
        self.noise_draws_: np.ndarray | None = None
        self.n_features_in_: int | None = None
        self.alpha_: float | None = None
        self.quantiles_: tuple[float, float] | None = None
        self._validate_options()

    def _validate_options(self) -> tuple[float, tuple[float, float], int, int]:
        """Validate construction and refits, including changes to public options."""
        raw = np.asarray(self.alpha)
        if raw.ndim != 0:
            raise ValueError("alpha must be a real scalar")
        alpha = float(as_vector(raw.reshape(1), "alpha")[0])
        quantiles = (alpha / 2, 1 - alpha / 2)
        if not 0 < quantiles[0] < quantiles[1] < 1:
            raise ValueError("alpha must yield distinct quantiles strictly between 0 and 1")
        for name, minimum in (("n_bootstrap", 2), ("random_state", 0)):
            value = getattr(self, name)
            if (
                isinstance(value, bool | np.bool_)
                or not isinstance(value, int | np.integer)
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.random_state > 2**32 - 1:
            raise ValueError("random_state must be at most 2**32 - 1")
        if self.estimator is not None and (
            not isinstance(self.estimator, BaseEstimator) or not is_regressor(self.estimator)
        ):
            raise ValueError("estimator must be a scikit-learn regressor or None")
        return alpha, quantiles, int(self.n_bootstrap), int(self.random_state)

    def fit(self, features: np.ndarray, y_true: np.ndarray) -> ResidualBootstrapRegressor:
        """Fit on training rows only; a failed refit preserves the previous fitted state.

        All exposed ``random_state`` parameters, including nested pipeline options,
        are set to the wrapper seed on each clone. Inputs and the supplied estimator
        are not modified. At least two training rows are required. Prediction noise
        draws are stored at fit time, so prediction does not advance a random stream.
        """
        alpha, quantiles, repetitions, seed = self._validate_options()
        values = _features(features)
        targets = as_vector(y_true, "y_true")
        if len(values) != len(targets):
            raise ValueError("features and y_true must have the same number of rows")
        if len(targets) < 2:
            raise ValueError("at least two training rows are required")
        template = LinearRegression() if self.estimator is None else self.estimator
        rng = np.random.default_rng(seed)
        model = _clone_regressor(template, seed)
        models = []
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                model.fit(values.copy(), targets.copy())
                fitted = _predict(model, values)
                residuals = targets - fitted
                scale = float(np.max(np.abs(residuals)))
                # Scale the mean to avoid overflowing a sum of finite residuals.
                if scale:
                    residuals = residuals - np.mean(residuals / scale) * scale
                for _ in range(repetitions):
                    pseudo_targets = fitted + rng.choice(residuals, size=len(targets))
                    candidate = _clone_regressor(template, seed)
                    candidate.fit(values.copy(), pseudo_targets.copy())
                    _predict(candidate, values)
                    models.append(candidate)
                # Separate draws include new-observation noise as well as refit variation.
                noise = rng.choice(residuals, size=repetitions)
        except FloatingPointError as error:
            raise RuntimeError("residual bootstrap fitting failed numerically") from error
        self.estimator_, self.bootstrap_estimators_ = model, tuple(models)
        self.residuals_, self.noise_draws_ = residuals, noise
        self.n_features_in_ = values.shape[1]
        self.alpha_, self.quantiles_ = alpha, quantiles
        return self

    def _prediction_features(self, features: np.ndarray) -> np.ndarray:
        """Require a successful fit and a compatible feature matrix."""
        if self.estimator_ is None:
            raise RuntimeError("ResidualBootstrapRegressor must be fitted before prediction")
        values = _features(features)
        if values.shape[1] != self.n_features_in_:
            raise ValueError("features must have the same number of columns as the fitted data")
        return values

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return point predictions from the base regressor fitted on all training rows."""
        values = self._prediction_features(features)
        return _predict(self.estimator_, values)

    def predict_interval(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return pointwise percentile prediction intervals with future residual noise.

        Each replicate reuses its stored noise draw across query rows. This makes
        results repeatable and independent of query ordering or batching; it does
        not represent independent joint draws or a simultaneous prediction band.
        """
        values = self._prediction_features(features)
        assert self.noise_draws_ is not None and self.quantiles_ is not None
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                draws = np.stack([_predict(model, values) for model in self.bootstrap_estimators_])
                draws += self.noise_draws_[:, None]
                endpoints = np.quantile(draws, self.quantiles_, axis=0, method="linear")
                lower = as_vector(endpoints[0], "lower")
                upper = as_vector(endpoints[1], "upper")
        except (ValueError, FloatingPointError) as error:
            raise RuntimeError(
                "residual bootstrap interval calculation failed numerically"
            ) from error
        return lower, upper
