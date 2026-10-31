"""Decision threshold analysis for calibrated probabilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from calibrated_ml_lab._validation import binary_inputs, probabilities


def _validate_cost(value: float, name: str) -> float:
    """Require a finite nonnegative scalar decision cost."""
    if np.ndim(value) != 0 or np.iscomplexobj(value):
        raise ValueError(f"{name} must be a finite nonnegative scalar")
    cost = float(value)
    if not np.isfinite(cost) or cost < 0:
        raise ValueError(f"{name} must be a finite nonnegative scalar")
    return cost


def threshold_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
    cost_false_positive: float = 1.0,
    cost_false_negative: float = 5.0,
) -> pd.DataFrame:
    """Evaluate p >= threshold decisions with finite nonnegative error costs.

    Thresholds must be a nonempty vector in [0, 1]; order and duplicates are preserved.
    """

    y, p = binary_inputs(y_true, y_prob)
    cost_false_positive = _validate_cost(cost_false_positive, "cost_false_positive")
    cost_false_negative = _validate_cost(cost_false_negative, "cost_false_negative")
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)
    thresholds = probabilities(thresholds, name="thresholds")

    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        pred = (p >= threshold).astype(int)
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tp = int(((pred == 1) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        # Normalize before multiplying to avoid overflowing a finite average cost.
        cost = cost_false_positive * (fp / len(y)) + cost_false_negative * (fn / len(y))
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision_score(y, pred, zero_division=0)),
                "recall": float(recall_score(y, pred, zero_division=0)),
                "f1": float(f1_score(y, pred, zero_division=0)),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "expected_cost": float(cost),
                "alert_rate": float(pred.mean()),
            }
        )
    return pd.DataFrame(rows)
