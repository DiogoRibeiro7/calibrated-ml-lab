"""Paired row-bootstrap intervals for fixed-prediction calibration diagnostics."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from calibrated_ml_lab._validation import as_vector, binary_inputs, multiclass_inputs
from calibrated_ml_lab.calibration.metrics import probabilistic_metrics
from calibrated_ml_lab.calibration.multiclass_metrics import multiclass_calibration_metrics


def _bootstrap_summary(
    y: np.ndarray,
    p: np.ndarray,
    evaluate: Callable[[np.ndarray, np.ndarray, int], dict[str, float]],
    *,
    n_bins: int,
    n_bootstrap: int,
    confidence_level: float,
    random_state: int,
) -> pd.DataFrame:
    """Resample aligned rows once per replicate, using every draw for every metric."""
    if len(y) < 2:
        raise ValueError("at least two evaluation rows are required")
    raw = np.asarray(confidence_level)
    if raw.ndim != 0 or isinstance(raw.item(), bool | np.bool_):
        raise ValueError("confidence_level must be a real scalar strictly between 0 and 1")
    level = float(as_vector(raw.reshape(1), "confidence_level")[0])
    tail = (1 - level) / 2
    quantiles = (tail, 1 - tail)
    if not 0 < level < 1 or not 0 < quantiles[0] < quantiles[1] < 1:
        raise ValueError("confidence_level must yield distinct quantiles strictly between 0 and 1")
    for name, value, minimum in (
        ("n_bins", n_bins, 1),
        ("n_bootstrap", n_bootstrap, 2),
        ("random_state", random_state, 0),
    ):
        if (
            isinstance(value, bool | np.bool_)
            or not isinstance(value, int | np.integer)
            or value < minimum
        ):
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if random_state > 2**32 - 1:
        raise ValueError("random_state must be at most 2**32 - 1")

    estimates = evaluate(y, p, n_bins)
    names = list(estimates)
    draws = np.empty((n_bootstrap, len(names)))
    rng = np.random.default_rng(int(random_state))
    for replicate in range(n_bootstrap):
        indices = rng.integers(0, len(y), size=len(y))
        metrics = evaluate(y[indices], p[indices], n_bins)
        draws[replicate] = [metrics[name] for name in names]
    lower, upper = np.quantile(draws, quantiles, axis=0, method="linear")
    return pd.DataFrame(
        {
            "metric": names,
            "estimate": [estimates[name] for name in names],
            "lower": lower,
            "upper": upper,
            "standard_error": draws.std(axis=0, ddof=1),
            "confidence_level": level,
            "n_samples": len(y),
            "n_bootstrap": n_bootstrap,
            "n_bins": n_bins,
            "method": "percentile",
        }
    )


def _binary_metrics(y: np.ndarray, p: np.ndarray, n_bins: int) -> dict[str, float]:
    """Keep probability-quality and calibration metrics, excluding discrimination."""
    metrics = probabilistic_metrics(y, p, n_bins=n_bins)
    del metrics["roc_auc"]
    return metrics


def bootstrap_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return percentile bootstrap intervals for binary Brier score, log loss, ECE, and MCE.

    Labels and probabilities are resampled together with replacement, preserving
    the original sample size. Predictions, bin edges, and fitted models stay fixed.
    At least two rows and two replicates are required; single-class resamples are
    retained. Input validation and endpoint clipping match ``probabilistic_metrics``.
    ROC AUC is excluded because it measures discrimination.

    Each metric's row contains its original estimate, lower/upper percentiles,
    bootstrap standard error (sample SD, ddof=1), confidence level, sample and
    replicate counts, bin count, and method. Quantiles use linear interpolation.
    Integer seeds in [0, 2**32 - 1] reproduce draws without changing global RNG state.

    These are marginal, approximate intervals for IID evaluation rows conditional
    on the fitted predictor, not prediction intervals or guarantees of calibration.
    Binned ECE/MCE can be biased and nonsmooth; percentile intervals can perform
    poorly near zero, with sparse bins, or with small samples. Degenerate resamples
    may yield zero-width intervals without establishing population certainty.
    """
    y, p = binary_inputs(y_true, y_prob)
    return _bootstrap_summary(
        y,
        p,
        _binary_metrics,
        n_bins=n_bins,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        random_state=random_state,
    )


def bootstrap_multiclass_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return percentile intervals for confidence, top-label, and macro classwise ECE.

    Use the paired resampling, options, output columns, and limitations of
    ``bootstrap_calibration_metrics``. Labels are integer class-column indices;
    probability matrices retain every class column even if a resample lacks that
    class. Two-column matrices use multiclass confidence/top-label semantics,
    distinct from the binary positive-class probability API. No models are refit.
    """
    y, p = multiclass_inputs(y_true, y_prob)
    return _bootstrap_summary(
        y,
        p,
        multiclass_calibration_metrics,
        n_bins=n_bins,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        random_state=random_state,
    )
