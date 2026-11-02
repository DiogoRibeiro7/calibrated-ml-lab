"""Evaluate quantile regression intervals on independent heteroskedastic data."""

from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import mean_pinball_loss

from calibrated_ml_lab.conformal.regression import regression_interval_metrics
from calibrated_ml_lab.models.quantile import QuantileIntervalRegressor


def main() -> None:
    """Print held-out coverage, widths, crossing rate, and raw-quantile pinball losses."""
    rng = np.random.default_rng(42)
    features = rng.uniform(-2, 2, size=(3000, 1))
    x = features[:, 0]
    noise_scale = 0.2 + 0.5 * (x + 2)
    targets = np.sin(2 * x) + 0.7 * x + rng.normal(size=len(x)) * noise_scale
    train_x, test_x = features[:2000], features[2000:]
    train_y, test_y = targets[:2000], targets[2000:]
    # Fix all settings before evaluating the independent test rows.
    model = QuantileIntervalRegressor(alpha=0.1).fit(train_x, train_y)
    raw_lower, raw_upper = model.predict_quantiles(test_x)
    lower, upper = model.predict_interval(test_x)
    low_noise = test_x[:, 0] < 0
    print(
        json.dumps(
            {
                "seed": 42,
                "train_rows": len(train_y),
                "test_rows": len(test_y),
                "nominal_coverage": 1 - model.alpha_,
                "quantile_levels": model.quantiles_,
                "test_crossing_rate": float(np.mean(raw_lower > raw_upper)),
                "test_lower_pinball_loss": mean_pinball_loss(
                    test_y, raw_lower, alpha=model.quantiles_[0]
                ),
                "test_upper_pinball_loss": mean_pinball_loss(
                    test_y, raw_upper, alpha=model.quantiles_[1]
                ),
                "test_intervals": regression_interval_metrics(test_y, lower, upper),
                "test_by_noise_region": {
                    name: {
                        "count": int(mask.sum()),
                        **regression_interval_metrics(test_y[mask], lower[mask], upper[mask]),
                    }
                    for name, mask in {"lower_noise": low_noise, "higher_noise": ~low_noise}.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
