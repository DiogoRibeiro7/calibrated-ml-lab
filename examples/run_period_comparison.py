"""Compare a fixed predictor across independent synthetic reference and production periods."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from calibrated_ml_lab.calibration.metrics import calibration_bins  # noqa: E402
from calibrated_ml_lab.calibration.multiclass_metrics import (
    confidence_calibration_bins,  # noqa: E402
)
from calibrated_ml_lab.monitoring.calibration import (  # noqa: E402
    compare_binary_calibration,
    compare_multiclass_calibration,
)


def _sample(
    rng: np.random.Generator,
    count: int,
    weights: np.ndarray,
    bias: np.ndarray,
    shift: tuple[float, float] = (0, 0),
) -> tuple[np.ndarray, np.ndarray]:
    """Generate an independent labeled sample from a specified softmax mechanism."""
    features = rng.normal(size=(count, 2)) + shift
    probabilities = softmax(features @ weights.T + bias, axis=1)
    labels = np.array([rng.choice(len(bias), p=row) for row in probabilities])
    return features, labels


def _plot_periods(bins: pd.DataFrame, path: Path) -> None:
    """Plot each task's reliability curve, leaving gaps where bins have no observations."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    try:
        for axis, task in zip(axes, ["binary", "multiclass"], strict=True):
            axis.plot([0, 1], [0, 1], "--", color="0.5", label="Equal predicted and observed rates")
            for period in ["reference", "production"]:
                rows = bins[(bins["task"] == task) & (bins["period"] == period)]
                axis.plot(
                    rows["mean_predicted_probability"],
                    rows["observed_event_rate"],
                    marker="o",
                    label=f"{period.capitalize()} (n={rows['count'].sum()})",
                )
            axis.set(
                xlabel="Mean positive-class probability" if task == "binary" else "Mean confidence",
                ylabel="Observed positive rate" if task == "binary" else "Accuracy",
                title="Binary calibration"
                if task == "binary"
                else "Multiclass confidence calibration",
                xlim=(0, 1),
                ylim=(0, 1),
            )
            axis.legend(fontsize=8)
            axis.grid(alpha=0.2)
        fig.suptitle("Fixed predictor evaluated in two labeled periods")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
    finally:
        plt.close(fig)


def main() -> None:
    """Fit only on training rows and export descriptive metrics, bins, predictions, and a PNG."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(".artifacts/period_comparison"))
    output_dir = parser.parse_args().output_dir
    rng = np.random.default_rng(42)
    definitions = {
        "binary": (
            np.array([[0, 0], [1.2, -0.8]]),
            np.array([0, 0.3]),
            np.array([[0, 0], [0.35, 0.15]]),
            np.array([0, -0.8]),
        ),
        "multiclass": (
            np.array([[1.2, -0.4], [-0.5, 1.0], [-0.4, -0.6]]),
            np.array([0.2, 0, -0.2]),
            np.array([[0.6, -0.2], [-0.25, 0.5], [-0.2, -0.3]]),
            np.array([0.8, -0.5, 0.1]),
        ),
    }
    comparisons, bin_tables, prediction_paths = [], [], {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for task, (weights, bias, production_weights, production_bias) in definitions.items():
        train_x, train_y = _sample(rng, 2000, weights, bias)
        reference_x, reference_y = _sample(rng, 1200, weights, bias)
        production_x, production_y = _sample(
            rng, 800, production_weights, production_bias, shift=(0.8, -0.3)
        )
        model = LogisticRegression(max_iter=500).fit(train_x, train_y)
        reference_p, production_p = (
            model.predict_proba(reference_x),
            model.predict_proba(production_x),
        )
        if task == "binary":
            comparison = compare_binary_calibration(
                reference_y, reference_p[:, 1], production_y, production_p[:, 1], n_bins=10
            )
        else:
            comparison = compare_multiclass_calibration(
                reference_y, reference_p, production_y, production_p, n_bins=10
            )
        comparisons.append(comparison.assign(task=task))
        predictions = []
        for period, y, p in [
            ("reference", reference_y, reference_p),
            ("production", production_y, production_p),
        ]:
            if task == "binary":
                bins = calibration_bins(y, p[:, 1], n_bins=10)
            else:
                bins = confidence_calibration_bins(y, p, n_bins=10).rename(
                    columns={
                        "mean_confidence": "mean_predicted_probability",
                        "accuracy": "observed_event_rate",
                    }
                )
            bin_tables.append(bins.assign(task=task, period=period))
            rows = pd.DataFrame(p, columns=[f"probability_{i}" for i in range(p.shape[1])])
            predictions.append(rows.assign(y_true=y, period=period))
        prediction_path = output_dir / f"{task}_period_predictions.csv"
        pd.concat(predictions, ignore_index=True).to_csv(prediction_path, index=False)
        prediction_paths[task] = str(prediction_path)
    comparison_path = output_dir / "period_comparison.csv"
    bins_path = output_dir / "period_bins.csv"
    plot_path = output_dir / "period_comparison.png"
    pd.concat(comparisons, ignore_index=True).to_csv(comparison_path, index=False)
    bins = pd.concat(bin_tables, ignore_index=True)
    bins.to_csv(bins_path, index=False)
    _plot_periods(bins, plot_path)
    print(
        json.dumps(
            {
                "seed": 42,
                "train_rows_per_task": 2000,
                "reference_rows_per_task": 1200,
                "production_rows_per_task": 800,
                "n_bins": 10,
                "comparison_path": str(comparison_path),
                "bins_path": str(bins_path),
                "plot_path": str(plot_path),
                "prediction_paths": prediction_paths,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
