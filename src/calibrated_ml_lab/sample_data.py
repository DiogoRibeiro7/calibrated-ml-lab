"""Synthetic datasets for calibration and uncertainty experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit


def generate_classification_data(rows: int = 5_000, seed: int = 42) -> pd.DataFrame:
    """Generate a binary classification dataset with intentional miscalibration risk.

    The latent probability is non-linear and includes group effects. This makes it useful for
    testing calibration and subgroup reliability.
    """

    if rows <= 0:
        raise ValueError("rows must be positive")

    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(48, 14, rows), 18, 90)
    income = np.exp(rng.normal(10.3, 0.45, rows))
    usage_30d = rng.gamma(3.0, 8.0, rows)
    prior_events = rng.poisson(1.1, rows)
    risk_signal = rng.normal(0, 1, rows)
    region = rng.choice(["North", "South", "East", "West"], rows, p=[0.25, 0.25, 0.25, 0.25])
    segment = rng.choice(["A", "B", "C"], rows, p=[0.45, 0.35, 0.20])

    region_effect = pd.Series(region).map({"North": 0.15, "South": -0.05, "East": 0.25, "West": -0.15}).to_numpy()
    segment_effect = pd.Series(segment).map({"A": -0.10, "B": 0.05, "C": 0.35}).to_numpy()

    linear = (
        -3.2
        + 0.028 * (age - 45)
        - 0.000018 * (income - income.mean())
        + 0.025 * usage_30d
        + 0.42 * prior_events
        + 0.85 * risk_signal
        + 0.28 * np.sin(usage_30d / 10)
        + region_effect
        + segment_effect
    )
    probability = expit(linear)
    label = rng.binomial(1, probability)

    return pd.DataFrame(
        {
            "age": age.round(2),
            "income": income.round(2),
            "usage_30d": usage_30d.round(2),
            "prior_events": prior_events,
            "risk_signal": risk_signal.round(4),
            "region": region,
            "segment": segment,
            "true_probability": probability.round(6),
            "label": label,
        }
    )


def generate_regression_data(rows: int = 2_000, seed: int = 123) -> pd.DataFrame:
    """Generate a heteroskedastic regression dataset for conformal intervals."""

    if rows <= 0:
        raise ValueError("rows must be positive")

    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, rows)
    x2 = rng.uniform(-2, 2, rows)
    x3 = rng.gamma(2, 1, rows)
    noise_scale = 0.35 + 0.25 * np.abs(x2) + 0.1 * x3
    y = 1.2 + 2.0 * x1 - 1.4 * x2 + 0.8 * np.sin(x3) + rng.normal(0, noise_scale)

    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "noise_scale": noise_scale, "y": y})


def write_sample_data(output_dir: Path, rows: int = 5_000, seed: int = 42) -> tuple[Path, Path]:
    """Write classification and regression sample datasets to disk."""

    output_dir.mkdir(parents=True, exist_ok=True)
    classification = generate_classification_data(rows=rows, seed=seed)
    regression = generate_regression_data(rows=max(500, rows // 2), seed=seed + 1)

    classification_path = output_dir / "classification.csv"
    regression_path = output_dir / "regression.csv"
    classification.to_csv(classification_path, index=False)
    regression.to_csv(regression_path, index=False)
    return classification_path, regression_path
