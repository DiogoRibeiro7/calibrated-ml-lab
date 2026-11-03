"""Moment decomposition for equally weighted regression predictive mixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd

from calibrated_ml_lab._validation import as_vector


def _member_matrix(values: np.ndarray, name: str) -> np.ndarray:
    """Require a nonempty finite matrix with members on rows and queries on columns."""
    raw = np.asarray(values)
    if raw.ndim != 2 or 0 in raw.shape:
        raise ValueError(f"{name} must have shape (n_members >= 1, n_queries >= 1)")
    return as_vector(raw.reshape(-1), name).reshape(raw.shape)


def ensemble_regression_summary(
    member_means: np.ndarray, member_variances: np.ndarray
) -> pd.DataFrame:
    """Decompose an equally weighted mixture's predictive variance for each query.

    Both inputs must be finite matrices of shape (n_members, n_queries), with
    nonnegative predictive variances. Alignment is positional, including Pandas
    inputs. Return the mean prediction, mean member variance, population variance
    of member means (denominator n_members), their sum, and total standard deviation.
    Variances are required: disagreement alone does not determine predictive noise.
    This reports supplied mixture moments, not calibration or coverage guarantees.
    """
    means = _member_matrix(member_means, "member_means")
    variances = _member_matrix(member_variances, "member_variances")
    if means.shape != variances.shape:
        raise ValueError("member_means and member_variances must have the same shape")
    if (variances < 0).any():
        raise ValueError("member_variances must be nonnegative")

    minimum, maximum = means.min(axis=0), means.max(axis=0)
    with np.errstate(over="ignore"):
        span = maximum - minimum
        # A nearby center retains small differences on top of a large shared offset.
        center = np.where(np.isfinite(span), minimum + span / 2, minimum / 2 + maximum / 2)
    offsets = means - center
    scale = np.max(np.abs(offsets), axis=0)
    scaled_offsets = np.divide(offsets, scale, out=np.zeros_like(offsets), where=scale > 0)
    mean_offset = scaled_offsets.mean(axis=0)
    with np.errstate(over="ignore"):
        prediction = center + mean_offset * scale
    # A finite-input mixture mean lies in the convex hull, even at float limits.
    prediction = np.clip(prediction, minimum, maximum)
    between_std = np.sqrt(np.mean((scaled_offsets - mean_offset) ** 2, axis=0)) * scale

    variance_scale = variances.max(axis=0)
    scaled_variances = np.divide(
        variances, variance_scale, out=np.zeros_like(variances), where=variance_scale > 0
    )
    within = scaled_variances.mean(axis=0) * variance_scale
    with np.errstate(over="ignore"):
        between = between_std * between_std
        total = within + between
    # Standard deviation can remain representable even when its square overflows.
    total_std = np.hypot(np.sqrt(within), between_std)
    return pd.DataFrame(
        {
            "mean_prediction": prediction,
            "within_model_variance": within,
            "between_model_variance": between,
            "total_variance": total,
            "total_std": total_std,
        }
    )
