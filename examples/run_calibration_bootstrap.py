"""Export bootstrap intervals for fixed binary and multiclass test predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.special import expit, softmax
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from calibrated_ml_lab.calibration.bootstrap import (
    bootstrap_calibration_metrics,
    bootstrap_multiclass_calibration_metrics,
)
from calibrated_ml_lab.calibration.multiclass import MulticlassTemperatureScaler
from calibrated_ml_lab.calibration.scaling import TemperatureScaler


def main() -> None:
    """Fit on separate training/calibration rows, freeze predictions, and resample test rows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(".artifacts/calibration_bootstrap"))
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    args = parser.parse_args()
    if args.n_bootstrap < 2:
        parser.error("--n-bootstrap must be at least 2")
    tables, prediction_paths = [], {}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for task, n_classes in [("binary", 2), ("multiclass", 3)]:
        features, labels = make_classification(
            n_samples=2400,
            n_features=12,
            n_informative=8,
            n_redundant=0,
            n_classes=n_classes,
            n_clusters_per_class=1,
            random_state=42,
        )
        train_x, held_x, train_y, held_y = train_test_split(
            features, labels, test_size=0.5, random_state=42, stratify=labels
        )
        calibration_x, test_x, calibration_y, test_y = train_test_split(
            held_x, held_y, test_size=0.5, random_state=43
        )
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
        model.fit(train_x, train_y)
        # Deliberately amplify confidence before fitting temperature on calibration rows.
        calibration_logits = 3 * model.decision_function(calibration_x)
        test_logits = 3 * model.decision_function(test_x)
        if task == "binary":
            scaler = TemperatureScaler().fit(expit(calibration_logits), calibration_y)
            raw = expit(test_logits)
            calibrated = scaler.transform(raw)
            evaluate = bootstrap_calibration_metrics
            predictions = pd.DataFrame(
                {"y_true": test_y, "raw_probability": raw, "calibrated_probability": calibrated}
            )
        else:
            multiclass_scaler = MulticlassTemperatureScaler().fit(calibration_logits, calibration_y)
            raw = softmax(test_logits, axis=1)
            calibrated = multiclass_scaler.transform(test_logits)
            evaluate = bootstrap_multiclass_calibration_metrics
            predictions = pd.DataFrame({"y_true": test_y})
            for name, probabilities in [("raw", raw), ("calibrated", calibrated)]:
                for class_index in range(n_classes):
                    predictions[f"{name}_probability_{class_index}"] = probabilities[:, class_index]
        for name, probabilities in [("raw", raw), ("calibrated", calibrated)]:
            # Identical seeds pair the resamples across methods, but each interval is marginal.
            table = evaluate(
                test_y,
                probabilities,
                n_bins=10,
                n_bootstrap=args.n_bootstrap,
                confidence_level=0.95,
                random_state=42,
            )
            tables.append(table.assign(task=task, predictor=name))
        prediction_path = args.output_dir / f"{task}_test_predictions.csv"
        predictions.to_csv(prediction_path, index=False)
        prediction_paths[task] = str(prediction_path)
    table_path = args.output_dir / "calibration_bootstrap_intervals.csv"
    pd.concat(tables, ignore_index=True).to_csv(table_path, index=False)
    print(
        json.dumps(
            {
                "seed": 42,
                "train_rows_per_task": 1200,
                "calibration_rows_per_task": 600,
                "test_rows_per_task": 600,
                "n_bootstrap": args.n_bootstrap,
                "n_bins": 10,
                "confidence_level": 0.95,
                "method": "percentile",
                "table_path": str(table_path),
                "prediction_paths": prediction_paths,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
