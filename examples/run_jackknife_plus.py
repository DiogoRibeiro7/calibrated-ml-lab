"""Evaluate jackknife-plus intervals and distinguish nominal targets from coverage bounds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from calibrated_ml_lab.conformal.jackknife import JackknifePlusRegressor
from calibrated_ml_lab.conformal.quantile import conformalized_quantile_metrics
from calibrated_ml_lab.uncertainty.intervals import interval_coverage_by_feature


def main() -> None:
    """Fit preset error levels on training rows and report independent test coverage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(".artifacts/jackknife_plus"))
    output_dir = parser.parse_args().output_dir
    levels = [0.1, 0.05]
    edges = np.array([-np.inf, -1, 0, 1, np.inf])
    rng = np.random.default_rng(42)
    features = rng.uniform(-2, 2, size=(2200, 1))
    x = features[:, 0]
    targets = 1 + 2 * x + rng.normal(size=len(x)) * (0.3 + 0.4 * (x + 2))
    train_x, test_x, train_y, test_y = features[:200], features[200:], targets[:200], targets[200:]
    tables, evaluations = [], []
    for alpha in levels:
        model = JackknifePlusRegressor(alpha=alpha).fit(train_x, train_y)
        lower, upper = model.predict_interval(test_x)
        evaluations.append(
            {
                "alpha": alpha,
                "nominal_coverage": 1 - alpha,
                "general_coverage_bound": 1 - 2 * alpha,
                "test_intervals": conformalized_quantile_metrics(test_y, lower, upper),
            }
        )
        tables.append(
            interval_coverage_by_feature(
                test_y, lower, upper, test_x[:, 0], edges, nominal_coverage=1 - alpha
            ).assign(alpha=alpha, general_coverage_bound=1 - 2 * alpha)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "jackknife_plus_coverage.csv"
    pd.concat(tables, ignore_index=True).to_csv(table_path, index=False)
    print(
        json.dumps(
            {
                "seed": 42,
                "train_rows": len(train_y),
                "test_rows": len(test_y),
                "fits_per_level": len(train_y) + 1,
                "evaluations": evaluations,
                "table_path": str(table_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
