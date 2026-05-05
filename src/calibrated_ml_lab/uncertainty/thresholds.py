"""Decision threshold analysis for calibrated probabilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def threshold_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
    cost_false_positive: float = 1.0,
    cost_false_negative: float = 5.0,
) -> pd.DataFrame:
    """Evaluate classification metrics and expected cost over thresholds."""

    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    if len(y) != len(p):
        raise ValueError("y_true and y_prob must have the same length")
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)

    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        pred = (p >= threshold).astype(int)
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tp = int(((pred == 1) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        cost = cost_false_positive * fp + cost_false_negative * fn
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
                "expected_cost": float(cost / len(y)),
                "alert_rate": float(pred.mean()),
            }
        )
    return pd.DataFrame(rows)
