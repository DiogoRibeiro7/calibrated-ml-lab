"""Confidence, top-label, and one-versus-rest calibration diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from calibrated_ml_lab._validation import multiclass_inputs
from calibrated_ml_lab.calibration.metrics import calibration_bins


def confidence_calibration_bins(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Bin maximum confidence against prediction correctness, pooling all classes.

    Exact probability ties select the first class column. Bins are equal-width,
    left-inclusive, with confidence 1 in the last bin. Empty bins remain visible.
    Weights are fractions of all input rows.
    """
    y, p = multiclass_inputs(y_true, y_prob)
    predictions = p.argmax(axis=1)
    bins = calibration_bins((predictions == y).astype(int), p.max(axis=1), n_bins=n_bins)
    return bins.rename(
        columns={
            "mean_predicted_probability": "mean_confidence",
            "observed_event_rate": "accuracy",
        }
    )


def top_label_calibration_bins(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Bin confidence and correctness separately for each predicted class.

    Weights use the total input row count, so summing weight * absolute_gap gives
    top-label ECE. Classes never predicted retain empty bins with zero weight and
    NaN means. Exact ties select the first class column, as in NumPy argmax.
    """
    y, p = multiclass_inputs(y_true, y_prob)
    predictions = p.argmax(axis=1)
    confidence = p.max(axis=1)
    correct = (predictions == y).astype(int)
    template = calibration_bins(correct, confidence, n_bins=n_bins)
    frames = []
    for class_index in range(p.shape[1]):
        selected = predictions == class_index
        if selected.any():
            bins = calibration_bins(correct[selected], confidence[selected], n_bins=n_bins)
        else:
            bins = template.copy()
            bins["count"] = 0
            bins["absolute_gap"] = 0.0
            bins[["mean_predicted_probability", "observed_event_rate"]] = np.nan
        bins["weight"] = bins["count"] / len(y)
        bins.insert(0, "predicted_class", class_index)
        frames.append(bins)
    return pd.concat(frames, ignore_index=True).rename(
        columns={
            "mean_predicted_probability": "mean_confidence",
            "observed_event_rate": "accuracy",
        }
    )


def classwise_calibration_errors(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Report one-versus-rest ECE and MCE using every row for each class.

    Include all probability columns, even classes absent from the labels or never
    predicted. positive_count exposes missing labels; count is the total row count.
    """
    y, p = multiclass_inputs(y_true, y_prob)
    rows = []
    for class_index in range(p.shape[1]):
        events = (y == class_index).astype(int)
        bins = calibration_bins(events, p[:, class_index], n_bins=n_bins)
        rows.append(
            {
                "class_index": class_index,
                "count": len(y),
                "positive_count": int(events.sum()),
                "ece": float((bins["weight"] * bins["absolute_gap"]).sum()),
                "mce": float(bins["absolute_gap"].max()),
            }
        )
    return pd.DataFrame(rows)


def multiclass_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict[str, float]:
    """Summarize pooled confidence ECE, conditional top-label ECE, and macro classwise ECE.

    The classwise mean gives every probability column equal weight. These binned
    diagnostics depend on the sample and bin count; they do not establish full
    multiclass calibration, fairness, or robustness under distribution shift.
    """
    confidence = confidence_calibration_bins(y_true, y_prob, n_bins=n_bins)
    top_label = top_label_calibration_bins(y_true, y_prob, n_bins=n_bins)
    classwise = classwise_calibration_errors(y_true, y_prob, n_bins=n_bins)
    return {
        "confidence_ece": float((confidence["weight"] * confidence["absolute_gap"]).sum()),
        "top_label_ece": float((top_label["weight"] * top_label["absolute_gap"]).sum()),
        "classwise_ece": float(classwise["ece"].mean()),
    }
