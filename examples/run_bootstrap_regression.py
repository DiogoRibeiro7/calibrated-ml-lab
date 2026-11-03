"""Compare residual-bootstrap coverage under constant and feature-dependent noise."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from calibrated_ml_lab.conformal.quantile import conformalized_quantile_metrics
from calibrated_ml_lab.models.bootstrap import ResidualBootstrapRegressor
from calibrated_ml_lab.uncertainty.intervals import interval_coverage_by_feature


def main() -> None:
    """Evaluate fixed bootstrap settings and feature bins on untouched test responses."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(".artifacts/bootstrap_regression"))
    output_dir = parser.parse_args().output_dir
    rng = np.random.default_rng(42)
    features = rng.uniform(-2, 2, size=(3000, 1))
    x = features[:, 0]
    noise = rng.normal(size=len(x))
    alpha, repetitions = 0.1, 500
    edges = np.array([-np.inf, -1, 0, 1, np.inf])
    train_x, test_x = features[:1000], features[1000:]
    tables, summaries = [], {}
    for name, scale in {
        "constant_noise": np.full(len(x), 1.2),
        "feature_dependent_noise": 0.25 + 0.5 * (x + 2),
    }.items():
        targets = 1 + 2 * x + noise * scale
        train_y, test_y = targets[:1000], targets[1000:]
        model = ResidualBootstrapRegressor(alpha=alpha, n_bootstrap=repetitions).fit(
            train_x, train_y
        )
        lower, upper = model.predict_interval(test_x)
        summaries[name] = conformalized_quantile_metrics(test_y, lower, upper)
        tables.append(
            interval_coverage_by_feature(
                test_y, lower, upper, test_x[:, 0], edges, nominal_coverage=1 - alpha
            ).assign(scenario=name)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "bootstrap_coverage.csv"
    pd.concat(tables, ignore_index=True).to_csv(table_path, index=False)
    print(
        json.dumps(
            {
                "seed": 42,
                "train_rows": len(train_x),
                "test_rows": len(test_x),
                "nominal_coverage": 1 - alpha,
                "n_bootstrap": repetitions,
                "scenarios": summaries,
                "table_path": str(table_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
