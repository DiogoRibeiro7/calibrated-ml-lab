"""Export the variance decomposition of a small ensemble of linear predictive models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.linear_model import LinearRegression

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from calibrated_ml_lab.uncertainty.ensemble import ensemble_regression_summary  # noqa: E402


def main() -> None:
    """Fit training-only components, summarize untouched test rows, and plot a query grid."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(".artifacts/ensemble_uncertainty"))
    output_dir = parser.parse_args().output_dir
    rng = np.random.default_rng(42)
    train_x = rng.uniform(-2, 2, size=(80, 1))
    train_y = 1 + 2 * train_x[:, 0] + rng.normal(scale=0.8, size=len(train_x))
    test_x = rng.uniform(-2, 2, size=(1000, 1))
    test_y = 1 + 2 * test_x[:, 0] + rng.normal(scale=0.8, size=len(test_x))
    grid = np.linspace(-5, 5, 401)[:, None]
    query = np.concatenate([test_x, grid])
    means, variances = [], []
    for _ in range(40):
        selected = rng.integers(0, len(train_x), size=len(train_x))
        x, y = train_x[selected], train_y[selected]
        model = LinearRegression().fit(x, y)
        residuals = y - model.predict(x)
        # Two fitted coefficients: intercept and slope. This example assumes constant noise.
        noise_variance = np.sum(residuals**2) / (len(y) - 2)
        means.append(model.predict(query))
        variances.append(np.full(len(query), noise_variance))
    summary = ensemble_regression_summary(np.stack(means), np.stack(variances))
    test_table = summary.iloc[: len(test_x)].copy().assign(x=test_x[:, 0], y_true=test_y)
    grid_table = summary.iloc[len(test_x) :].copy().assign(x=grid[:, 0])
    output_dir.mkdir(parents=True, exist_ok=True)
    test_path = output_dir / "ensemble_test_predictions.csv"
    grid_path = output_dir / "ensemble_uncertainty.csv"
    plot_path = output_dir / "ensemble_uncertainty.png"
    test_table.to_csv(test_path, index=False)
    grid_table.to_csv(grid_path, index=False)
    fig, axis = plt.subplots(figsize=(7, 4.5))
    try:
        axis.stackplot(
            grid[:, 0],
            grid_table["within_model_variance"],
            grid_table["between_model_variance"],
            labels=["Within-model variance", "Between-model variance"],
            colors=["#4477AA", "#EEAA33"],
            alpha=0.85,
        )
        axis.axvline(-2, color="0.3", linestyle="--", label="Training range boundaries")
        axis.axvline(2, color="0.3", linestyle="--")
        axis.set(xlabel="Input x", ylabel="Predictive variance", xlim=(-5, 5), ylim=(0, None))
        axis.set_title("Variance of an equally weighted regression ensemble")
        axis.legend(loc="upper center", fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=160)
    finally:
        plt.close(fig)
    print(
        json.dumps(
            {
                "seed": 42,
                "train_rows": len(train_x),
                "test_rows": len(test_x),
                "members": len(means),
                "test_mean_squared_error": float(
                    np.mean((test_y - test_table["mean_prediction"]) ** 2)
                ),
                "test_mean_total_variance": float(test_table["total_variance"].mean()),
                "test_table_path": str(test_path),
                "grid_table_path": str(grid_path),
                "plot_path": str(plot_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
