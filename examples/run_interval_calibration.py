"""Export calibration curves and tables for quantile and conformalized intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from calibrated_ml_lab.conformal.quantile import (  # noqa: E402
    conformalized_quantile_intervals,
    conformalized_quantile_threshold,
)
from calibrated_ml_lab.models.quantile import QuantileIntervalRegressor  # noqa: E402
from calibrated_ml_lab.uncertainty.intervals import interval_calibration_table  # noqa: E402
from calibrated_ml_lab.visualization.plots import plot_interval_calibration  # noqa: E402


def main() -> None:
    """Fit preset levels on training rows, calibrate independently, then export test diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(".artifacts/interval_calibration"))
    output_dir = parser.parse_args().output_dir
    rng = np.random.default_rng(42)
    features = rng.uniform(-2, 2, size=(4000, 1))
    x = features[:, 0]
    targets = np.sin(2 * x) + 0.7 * x + rng.normal(size=len(x)) * (0.2 + 0.5 * (x + 2))
    train_x, calibration_x, test_x = features[:2000], features[2000:3000], features[3000:]
    train_y, calibration_y, test_y = targets[:2000], targets[2000:3000], targets[3000:]
    levels = [0.5, 0.7, 0.8, 0.9, 0.95]
    predictions: dict[str, dict[float, tuple[np.ndarray, np.ndarray]]] = {
        "Quantile": {},
        "Conformalized quantile": {},
    }
    for level in levels:
        alpha = 1 - level
        model = QuantileIntervalRegressor(alpha=alpha).fit(train_x, train_y)
        calibration_lower, calibration_upper = model.predict_interval(calibration_x)
        threshold = conformalized_quantile_threshold(
            calibration_y, calibration_lower, calibration_upper, alpha
        )
        base = model.predict_interval(test_x)
        predictions["Quantile"][level] = base
        predictions["Conformalized quantile"][level] = conformalized_quantile_intervals(
            *base, threshold
        )
    # Test labels are used only for the fixed methods' final evaluation, never fitting or tuning.
    tables = [
        interval_calibration_table(test_y, intervals).assign(method=name)
        for name, intervals in predictions.items()
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path, plot_path = (
        output_dir / "interval_calibration.csv",
        output_dir / "interval_calibration.png",
    )
    pd.concat(tables, ignore_index=True).to_csv(table_path, index=False)
    plot_interval_calibration(test_y, predictions, plot_path)
    print(
        json.dumps(
            {
                "seed": 42,
                "train_rows": len(train_y),
                "calibration_rows": len(calibration_y),
                "test_rows": len(test_y),
                "nominal_levels": levels,
                "table_path": str(table_path),
                "plot_path": str(plot_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
