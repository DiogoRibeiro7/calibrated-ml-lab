"""Descriptive reference-versus-production comparisons on labeled evaluation rows."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from calibrated_ml_lab._validation import binary_inputs, multiclass_inputs
from calibrated_ml_lab.calibration.metrics import probabilistic_metrics
from calibrated_ml_lab.calibration.multiclass_metrics import multiclass_calibration_metrics


def _period_inputs(
    y: np.ndarray,
    p: np.ndarray,
    period: str,
    validate: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Identify which period has malformed or unavailable evaluation data."""
    try:
        return validate(y, p)
    except ValueError as error:
        raise ValueError(f"{period} period: {error}") from error


def _comparison_table(
    reference: dict[str, float],
    production: dict[str, float],
    categories: dict[str, str],
    reference_count: int,
    production_count: int,
    n_bins: int,
) -> pd.DataFrame:
    """Keep undefined metrics visible and compute signed changes in original metric units."""
    return pd.DataFrame(
        [
            {
                "metric": name,
                "category": categories[name],
                "reference": value,
                "production": production[name],
                "change": production[name] - value,
                "reference_count": reference_count,
                "production_count": production_count,
                "n_bins": n_bins,
            }
            for name, value in reference.items()
        ]
    )


def compare_binary_calibration(
    reference_y: np.ndarray,
    reference_prob: np.ndarray,
    production_y: np.ndarray,
    production_prob: np.ndarray,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compare binary calibration, probability quality, ranking, and sample summaries.

    Each labeled period is validated separately and may have a different number
    of rows. Labels and probabilities align by position within a period, not by
    Pandas index; there is no pairing between periods. Both use the same positive
    integer bin count and fixed equal-width edges over [0, 1].

    Rows report Brier score, log loss, ECE, MCE, ROC AUC, positive-label frequency,
    and mean predicted probability. Categories distinguish probabilistic quality,
    calibration, discrimination, and sample summaries. ``change`` is production
    minus reference, with both sample counts and the bin count retained. Single-
    class ROC AUC and its change remain NaN. Inputs and predictions are unchanged.

    Supply predictions from a fixed model/calibrator and labels measured on the
    same basis in both periods. No fitting, date selection, missing-label removal,
    significance testing, or alerting is performed. Changes may reflect sampling
    variation or population composition and do not by themselves establish drift.
    """
    ref_y, ref_p = _period_inputs(reference_y, reference_prob, "reference", binary_inputs)
    prod_y, prod_p = _period_inputs(production_y, production_prob, "production", binary_inputs)
    reference = probabilistic_metrics(ref_y, ref_p, n_bins=n_bins)
    production = probabilistic_metrics(prod_y, prod_p, n_bins=n_bins)
    reference.update(positive_rate=float(ref_y.mean()), mean_probability=float(ref_p.mean()))
    production.update(positive_rate=float(prod_y.mean()), mean_probability=float(prod_p.mean()))
    categories = {
        "brier_score": "probabilistic_quality",
        "log_loss": "probabilistic_quality",
        "ece": "calibration",
        "mce": "calibration",
        "roc_auc": "discrimination",
        "positive_rate": "sample_summary",
        "mean_probability": "sample_summary",
    }
    return _comparison_table(reference, production, categories, len(ref_y), len(prod_y), n_bins)


def _multiclass_summary(y: np.ndarray, p: np.ndarray, n_bins: int) -> dict[str, float]:
    """Retain confidence diagnostics and every class's observed label frequency."""
    metrics = multiclass_calibration_metrics(y, p, n_bins=n_bins)
    metrics["accuracy"] = float(np.mean(p.argmax(axis=1) == y))
    metrics["mean_confidence"] = float(p.max(axis=1).mean())
    for class_index in range(p.shape[1]):
        metrics[f"class_{class_index}_rate"] = float(np.mean(y == class_index))
    return metrics


def compare_multiclass_calibration(
    reference_y: np.ndarray,
    reference_prob: np.ndarray,
    production_y: np.ndarray,
    production_prob: np.ndarray,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compare confidence, top-label, and macro classwise ECE between labeled periods.

    Use the output columns, positional alignment, fixed-bin convention, and
    descriptive interpretation of ``compare_binary_calibration``. Matrices must
    have the same number of class columns, in the same semantic order. Semantic
    class mappings cannot be inferred or verified from numeric arrays.

    Additional rows contain accuracy, mean maximum confidence, and one label
    frequency per class (``class_0_rate``, etc.). Classes absent from the observed
    labels retain their columns and zero frequency. Accuracy is predictive performance,
    not calibration; exact probability ties choose the first class column.
    """
    ref_y, ref_p = _period_inputs(reference_y, reference_prob, "reference", multiclass_inputs)
    prod_y, prod_p = _period_inputs(production_y, production_prob, "production", multiclass_inputs)
    if ref_p.shape[1] != prod_p.shape[1]:
        raise ValueError("reference and production must have the same number of class columns")
    reference = _multiclass_summary(ref_y, ref_p, n_bins)
    production = _multiclass_summary(prod_y, prod_p, n_bins)
    categories = {
        "confidence_ece": "calibration",
        "top_label_ece": "calibration",
        "classwise_ece": "calibration",
        "accuracy": "predictive_performance",
        "mean_confidence": "sample_summary",
        **{f"class_{i}_rate": "sample_summary" for i in range(ref_p.shape[1])},
    }
    return _comparison_table(reference, production, categories, len(ref_y), len(prod_y), n_bins)
