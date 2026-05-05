"""Visualization functions for calibration and uncertainty reports."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from calibrated_ml_lab.calibration.metrics import calibration_bins


def plot_reliability_diagram(
    y_true: np.ndarray, probability_columns: dict[str, np.ndarray], output_path: Path, n_bins: int = 10
) -> None:
    """Plot reliability curves for one or more probability estimates."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    for name, probabilities in probability_columns.items():
        bins = calibration_bins(y_true, probabilities, n_bins=n_bins)
        non_empty = bins[bins["count"] > 0]
        ax.plot(
            non_empty["mean_predicted_probability"],
            non_empty["observed_event_rate"],
            marker="o",
            label=name,
        )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed event rate")
    ax.set_title("Reliability diagram")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_probability_histogram(probability_columns: dict[str, np.ndarray], output_path: Path) -> None:
    """Plot probability distributions."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, probabilities in probability_columns.items():
        ax.hist(probabilities, bins=30, alpha=0.35, density=True, label=name)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Density")
    ax.set_title("Predicted probability distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_threshold_tradeoff(threshold_table: pd.DataFrame, output_path: Path) -> None:
    """Plot threshold metrics."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for column in ["precision", "recall", "f1", "alert_rate"]:
        ax.plot(threshold_table["threshold"], threshold_table[column], marker="o", label=column)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric value")
    ax.set_title("Decision-threshold trade-off")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_conformal_regression_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    output_path: Path,
    max_points: int = 120,
) -> None:
    """Plot a sample of conformal regression intervals."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    y = np.asarray(y_true)
    pred = np.asarray(y_pred)
    lo = np.asarray(lower)
    hi = np.asarray(upper)
    order = np.argsort(pred)[: min(max_points, len(pred))]
    x = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(x, pred[order], yerr=[pred[order] - lo[order], hi[order] - pred[order]], fmt="o", alpha=0.6)
    ax.scatter(x, y[order], s=12, label="Observed")
    ax.set_xlabel("Ordered sample index")
    ax.set_ylabel("Target")
    ax.set_title("Split-conformal regression intervals")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
