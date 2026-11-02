"""Compare quantile and conformalized intervals on separate train/calibration/test rows."""

from __future__ import annotations

import json

import numpy as np

from calibrated_ml_lab.conformal.quantile import (
    conformalized_quantile_intervals,
    conformalized_quantile_metrics,
    conformalized_quantile_threshold,
)
from calibrated_ml_lab.models.quantile import QuantileIntervalRegressor


def main() -> None:
    """Print held-out coverage and widths after calibrating on fresh observations."""
    rng = np.random.default_rng(42)
    features = rng.uniform(-2, 2, size=(4000, 1))
    x = features[:, 0]
    noise_scale = 0.2 + 0.5 * (x + 2)
    targets = np.sin(2 * x) + 0.7 * x + rng.normal(size=len(x)) * noise_scale
    train_x, calibration_x, test_x = features[:2000], features[2000:3000], features[3000:]
    train_y, calibration_y, test_y = targets[:2000], targets[2000:3000], targets[3000:]
    alpha = 0.1
    model = QuantileIntervalRegressor(alpha=alpha).fit(train_x, train_y)
    # Use the same fixed bound-ordering rule for calibration and test observations.
    calibration_lower, calibration_upper = model.predict_interval(calibration_x)
    qhat = conformalized_quantile_threshold(
        calibration_y, calibration_lower, calibration_upper, alpha
    )
    base_lower, base_upper = model.predict_interval(test_x)
    lower, upper = conformalized_quantile_intervals(base_lower, base_upper, qhat)
    low_noise = test_x[:, 0] < 0
    print(
        json.dumps(
            {
                "seed": 42,
                "train_rows": len(train_y),
                "calibration_rows": len(calibration_y),
                "test_rows": len(test_y),
                "alpha": alpha,
                "nominal_coverage": 1 - alpha,
                "conformal_threshold": qhat,
                "test_base_intervals": conformalized_quantile_metrics(
                    test_y, base_lower, base_upper
                ),
                "test_conformal_intervals": conformalized_quantile_metrics(test_y, lower, upper),
                "test_by_noise_region": {
                    name: {
                        "count": int(mask.sum()),
                        "base": conformalized_quantile_metrics(
                            test_y[mask], base_lower[mask], base_upper[mask]
                        ),
                        "conformal": conformalized_quantile_metrics(
                            test_y[mask], lower[mask], upper[mask]
                        ),
                    }
                    for name, mask in {"lower_noise": low_noise, "higher_noise": ~low_noise}.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
