"""Compare epistemic and aleatoric variance in a known-noise Bayesian linear model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import norm

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

TRAIN_SIZES = (20, 80, 320)
NOISE_STDS = (0.4, 1.2)
PRIOR_VARIANCE = 25.0
NOMINAL_COVERAGE = 0.9


def _predictive_components(
    train_x: np.ndarray,
    train_y: np.ndarray,
    query_x: np.ndarray,
    *,
    noise_std: float,
    prior_variance: float = PRIOR_VARIANCE,
) -> pd.DataFrame:
    """Compute exact moments for this example's finite, univariate arrays and positive scales.

    The intercept and slope have independent N(0, prior_variance) priors.
    Observation noise is known, Gaussian, and constant across inputs. These are
    conditional model uncertainties, not estimates of arbitrary real-world sources.
    """
    design = np.column_stack([np.ones(len(train_x)), train_x])
    query = np.column_stack([np.ones(len(query_x)), query_x])
    noise_variance = noise_std**2
    precision = np.eye(2) / prior_variance + design.T @ design / noise_variance
    covariance = np.linalg.solve(precision, np.eye(2))
    coefficients = np.linalg.solve(precision, design.T @ train_y / noise_variance)
    epistemic = np.einsum("ij,jk,ik->i", query, covariance, query)
    return pd.DataFrame(
        {
            "x": query_x,
            "mean_prediction": query @ coefficients,
            "epistemic_variance": epistemic,
            "aleatoric_variance": noise_variance,
            "total_variance": epistemic + noise_variance,
        }
    )


def _run_experiment(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Use nested training samples and separate test rows for six preset scenarios."""
    rng = np.random.default_rng(seed)
    train_x = rng.uniform(-2, 2, size=max(TRAIN_SIZES))
    train_noise = rng.normal(size=len(train_x))
    test_x = rng.uniform(-2, 2, size=1000)
    test_noise = rng.normal(size=len(test_x))
    grid = np.linspace(-5, 5, 401)
    query = np.concatenate([grid, test_x])
    quantile = norm.ppf((1 + NOMINAL_COVERAGE) / 2)
    grid_tables, test_tables, summaries = [], [], []
    for noise_std in NOISE_STDS:
        # Reuse standard-normal draws across noise levels to isolate the comparison.
        train_y = 1 + 2 * train_x + noise_std * train_noise
        test_y = 1 + 2 * test_x + noise_std * test_noise
        for train_rows in TRAIN_SIZES:
            table = _predictive_components(
                train_x[:train_rows], train_y[:train_rows], query, noise_std=noise_std
            ).assign(train_rows=train_rows, noise_std=noise_std)
            half_width = quantile * np.sqrt(table["total_variance"])
            table["lower"] = table["mean_prediction"] - half_width
            table["upper"] = table["mean_prediction"] + half_width
            grid_tables.append(table.iloc[: len(grid)].copy())
            test = table.iloc[len(grid) :].copy().assign(y_true=test_y)
            test["covered"] = (test["lower"] <= test_y) & (test_y <= test["upper"])
            test_tables.append(test)
            # Test responses enter only these diagnostics; they never affect the fit.
            summaries.append(
                {
                    "train_rows": train_rows,
                    "noise_std": noise_std,
                    "test_rows": len(test),
                    "nominal_coverage": NOMINAL_COVERAGE,
                    "test_mean_squared_error": float(
                        np.mean((test_y - test["mean_prediction"]) ** 2)
                    ),
                    "test_mean_epistemic_variance": test["epistemic_variance"].mean(),
                    "test_mean_aleatoric_variance": test["aleatoric_variance"].mean(),
                    "test_mean_total_variance": test["total_variance"].mean(),
                    "test_coverage": test["covered"].mean(),
                    "test_mean_interval_width": (test["upper"] - test["lower"]).mean(),
                }
            )
    return (
        pd.DataFrame(summaries),
        pd.concat(grid_tables, ignore_index=True),
        pd.concat(test_tables, ignore_index=True),
    )


def _plot_components(summary: pd.DataFrame, grid: pd.DataFrame, path: Path) -> None:
    """Show training-size comparisons and extrapolation, with explicit variance units."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    try:
        for noise_std in NOISE_STDS:
            rows = summary[summary["noise_std"] == noise_std]
            for axis, source in zip(axes[0], ["epistemic", "aleatoric"], strict=True):
                axis.plot(
                    rows["train_rows"],
                    rows[f"test_mean_{source}_variance"],
                    marker="o",
                    label=f"Noise SD = {noise_std}",
                )
        for axis, source in zip(axes[0], ["epistemic", "aleatoric"], strict=True):
            axis.set(
                xlabel="Training rows (nested samples)",
                ylabel="Mean variance on fixed test inputs",
                title=f"{source.capitalize()} uncertainty",
                ylim=(0, None),
            )
            axis.set_xticks(TRAIN_SIZES)
            axis.legend(fontsize=8)
        for axis, noise_std in zip(axes[1], NOISE_STDS, strict=True):
            rows = grid[grid["noise_std"] == noise_std]
            axis.axvspan(-2, 2, color="0.9", label="Training input domain")
            for train_rows in TRAIN_SIZES:
                scenario = rows[rows["train_rows"] == train_rows]
                axis.plot(
                    scenario["x"],
                    scenario["epistemic_variance"],
                    label=f"Epistemic, n = {train_rows}",
                )
            axis.axhline(noise_std**2, color="0.2", linestyle="--", label="Known aleatoric")
            axis.set(
                xlabel="Input x",
                ylabel="Variance (log scale)",
                title=f"Extrapolation, noise SD = {noise_std}",
                xlim=(-5, 5),
                yscale="log",
            )
            axis.legend(fontsize=8)
        # Use the same scale for the two noise scenarios in the feature panels.
        limits = [axis.get_ylim() for axis in axes[1]]
        for axis in axes[1]:
            axis.set_ylim(min(low for low, _ in limits), max(high for _, high in limits))
        fig.suptitle("Uncertainty sources under a known-noise Bayesian linear model")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
    finally:
        plt.close(fig)


def main() -> None:
    """Export reproducible tables and a plot without network access or a GUI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(".artifacts/uncertainty_sources"))
    output_dir = parser.parse_args().output_dir
    summary, grid, test = _run_experiment(seed=42)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "uncertainty_sources_summary.csv"
    grid_path = output_dir / "uncertainty_sources_grid.csv"
    test_path = output_dir / "uncertainty_sources_test.csv"
    plot_path = output_dir / "uncertainty_sources.png"
    summary.to_csv(summary_path, index=False)
    grid.to_csv(grid_path, index=False)
    test.to_csv(test_path, index=False)
    _plot_components(summary, grid, plot_path)
    print(
        json.dumps(
            {
                "seed": 42,
                "prior_variance": PRIOR_VARIANCE,
                "known_noise_stds": NOISE_STDS,
                "train_sizes": TRAIN_SIZES,
                "nominal_coverage": NOMINAL_COVERAGE,
                "summary_path": str(summary_path),
                "grid_path": str(grid_path),
                "test_path": str(test_path),
                "plot_path": str(plot_path),
                "scenarios": summary.to_dict(orient="records"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
