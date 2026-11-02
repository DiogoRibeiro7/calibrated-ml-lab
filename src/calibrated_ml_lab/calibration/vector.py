"""Regularized diagonal affine calibration of multiclass logits."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

from calibrated_ml_lab._validation import as_vector, class_indices, logit_matrix


def _center_affine(logits: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Apply the affine map before removing row offsets for stable softmax arithmetic."""
    scaled = logits * weights + bias
    scaled -= scaled.max(axis=1, keepdims=True)
    if not np.isfinite(scaled).all():
        raise FloatingPointError("nonfinite scaled logits")
    return np.asarray(scaled, dtype=float)


def _objective(
    parameters: np.ndarray, logits: np.ndarray, labels: np.ndarray, l2: float
) -> tuple[float, np.ndarray]:
    """Return mean negative log likelihood plus identity shrinkage and its gradient."""
    n_rows, n_classes = logits.shape
    weights, bias = parameters[:n_classes], parameters[n_classes:]
    scaled = _center_affine(logits, weights, bias)
    losses = logsumexp(scaled, axis=1) - scaled[np.arange(n_rows), labels]
    errors = softmax(scaled, axis=1)
    errors[np.arange(n_rows), labels] -= 1
    errors /= n_rows
    delta = np.concatenate((weights - 1, bias))
    loss = float(np.sum(losses / n_rows))
    gradient = np.concatenate(((errors * logits).sum(axis=0), errors.sum(axis=0)))
    if l2:
        loss += float(0.5 * l2 * np.dot(delta, delta))
        gradient += l2 * delta
    if not np.isfinite(loss) or not np.isfinite(gradient).all():
        raise FloatingPointError("nonfinite vector-scaling objective")
    return loss, gradient


class MulticlassVectorScaler:
    """Fit ``softmax(logits * weights_ + bias_)`` on held-out calibration rows.

    Weights are unconstrained real numbers, one per class; predictions can change.
    Minimize mean negative log likelihood plus ``l2 / 2`` times the sum of squared
    deviations of weights from one and biases from zero. The default l2 is 1e-3;
    zero disables regularization. Use independent data to choose l2 or a method.

    Preserve the model's logit convention and class-column order at transform time.
    Row offsets are removed only after the affine map, since unequal weights make
    vector scaling sensitive to offsets in the original logits. Missing calibration
    classes are allowed but cannot establish calibration for those classes.
    """

    def __init__(self, l2: float = 1e-3, max_iter: int = 1000) -> None:
        self.l2 = l2
        self.max_iter = max_iter
        self.weights_: np.ndarray | None = None
        self.bias_: np.ndarray | None = None
        self.n_classes_: int | None = None
        self._validate_options()

    def _validate_options(self) -> tuple[float, int]:
        """Validate options on construction and on refits after attribute changes."""
        raw = np.asarray(self.l2)
        if raw.ndim != 0 or isinstance(raw.item(), bool | np.bool_):
            raise ValueError("l2 must be a finite nonnegative scalar")
        l2 = float(as_vector(raw.reshape(1), "l2")[0])
        if l2 < 0:
            raise ValueError("l2 must be a finite nonnegative scalar")
        if (
            isinstance(self.max_iter, bool | np.bool_)
            or not isinstance(self.max_iter, int | np.integer)
            or self.max_iter <= 0
        ):
            raise ValueError("max_iter must be a positive integer")
        return l2, int(self.max_iter)

    def fit(self, logits: np.ndarray, y_true: np.ndarray) -> MulticlassVectorScaler:
        """Fit deterministically from identity; failed refits retain previous parameters."""
        l2, max_iter = self._validate_options()
        values = logit_matrix(logits)
        n_rows, n_classes = values.shape
        labels = class_indices(y_true, n_classes)
        if len(labels) != n_rows:
            raise ValueError("logits and y_true must have the same number of rows")
        initial = np.concatenate((np.ones(n_classes), np.zeros(n_classes)))
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                initial_loss, _ = _objective(initial, values, labels, l2)
        except FloatingPointError as error:
            raise ValueError("logits are too large for vector scaling arithmetic") from error
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                result = minimize(
                    _objective,
                    initial,
                    args=(values, labels, l2),
                    method="L-BFGS-B",
                    jac=True,
                    options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-8},
                )
                if (
                    not result.success
                    or np.shape(result.x) != initial.shape
                    or not np.isfinite(result.x).all()
                    or not np.isfinite(result.fun)
                ):
                    raise RuntimeError("multiclass vector optimisation failed")
                # Recompute the candidate objective; never replace identity with a worse fit.
                candidate_loss, _ = _objective(result.x, values, labels, l2)
        except FloatingPointError as error:
            raise RuntimeError("multiclass vector optimisation failed numerically") from error
        parameters = result.x if candidate_loss < initial_loss else initial
        weights = np.array(parameters[:n_classes], dtype=float, copy=True)
        bias = np.array(parameters[n_classes:], dtype=float, copy=True)
        self.weights_, self.bias_, self.n_classes_ = weights, bias, n_classes
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Return normalized class probabilities, rejecting unsafe affine arithmetic."""
        if self.weights_ is None or self.bias_ is None:
            raise RuntimeError("MulticlassVectorScaler must be fitted before transform")
        values = logit_matrix(logits)
        if values.shape[1] != self.n_classes_:
            raise ValueError("logits must have the same number of class columns as the fitted data")
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                scaled = _center_affine(values, self.weights_, self.bias_)
                return np.asarray(softmax(scaled, axis=1), dtype=float)
        except FloatingPointError as error:
            raise ValueError("logits are too large for vector scaling arithmetic") from error
