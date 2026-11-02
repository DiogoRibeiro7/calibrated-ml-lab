"""Export empirical interval coverage by feature bins fixed before evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from calibrated_ml_lab.conformal.quantile import (
    conformalized_quantile_intervals,
    conformalized_quantile_metrics,
    conformalized_quantile_threshold,
)
from calibrated_ml_lab.models.quantile import QuantileIntervalRegressor
from calibrated_ml_lab.uncertainty.intervals import interval_coverage_by_feature


def main() -> None:
    """Fit and calibrate on separate rows, then export fixed-bin test diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(".artifacts/feature_coverage"))
    output_dir = parser.parse_args().output_dir
    alpha = 0.1
    # Freeze the partition independently of test outcomes; infinite tails retain every row.
    edges = np.array([-np.inf, -1, 0, 1, np.inf])
    rng = np.random.default_rng(42)
    features = rng.uniform(-2, 2, size=(4000, 1))
    x = features[:, 0]
    targets = np.sin(2 * x) + 0.7 * x + rng.normal(size=len(x)) * (0.2 + 0.5 * (x + 2))
    train_x, calibration_x, test_x = features[:2000], features[2000:3000], features[3000:]
    train_y, calibration_y, test_y = targets[:2000], targets[2000:3000], targets[3000:]
    model = QuantileIntervalRegressor(alpha=alpha).fit(train_x, train_y)
    calibration_lower, calibration_upper = model.predict_interval(calibration_x)
    threshold = conformalized_quantile_threshold(
        calibration_y, calibration_lower, calibration_upper, alpha
    )
    base = model.predict_interval(test_x)
    predictions = {
        "Quantile": base,
        "Conformalized quantile": conformalized_quantile_intervals(*base, threshold),
    }
    # Test labels enter only final evaluation, after models, threshold, and bins are fixed.
    tables = [
        interval_coverage_by_feature(
            test_y, *bounds, test_x[:, 0], edges, nominal_coverage=1 - alpha
        ).assign(method=name, feature="x")
        for name, bounds in predictions.items()
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "feature_coverage.csv"
    pd.concat(tables, ignore_index=True).to_csv(table_path, index=False)
    print(
        json.dumps(
            {
                "seed": 42,
                "train_rows": len(train_y),
                "calibration_rows": len(calibration_y),
                "test_rows": len(test_y),
                "nominal_coverage": 1 - alpha,
                "overall": {
                    name: conformalized_quantile_metrics(test_y, *bounds)
                    for name, bounds in predictions.items()
                },
                "table_path": str(table_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
