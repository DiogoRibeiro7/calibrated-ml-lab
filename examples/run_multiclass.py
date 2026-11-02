"""Evaluate multiclass temperature scaling on a separate synthetic test set."""

from __future__ import annotations

import json

import numpy as np
from scipy.special import softmax
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from calibrated_ml_lab.calibration.multiclass import MulticlassTemperatureScaler
from calibrated_ml_lab.calibration.multiclass_metrics import (
    classwise_calibration_errors,
    multiclass_calibration_metrics,
)


def main() -> None:
    """Print held-out metrics after fitting the model and calibrator on separate rows."""
    features, labels = make_classification(
        n_samples=2400,
        n_features=12,
        n_informative=8,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42,
    )
    train_x, held_out_x, train_y, held_out_y = train_test_split(
        features,
        labels,
        test_size=0.4,
        stratify=labels,
        random_state=42,
    )
    calibration_x, test_x, calibration_y, test_y = train_test_split(
        held_out_x,
        held_out_y,
        test_size=0.5,
        stratify=held_out_y,
        random_state=43,
    )
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
    model.fit(train_x, train_y)
    # Deliberately amplify confidence to illustrate a known source of miscalibration.
    calibration_logits = 3.0 * model.decision_function(calibration_x)
    scaler = MulticlassTemperatureScaler().fit(calibration_logits, calibration_y)
    test_logits = 3.0 * model.decision_function(test_x)
    raw = softmax(test_logits, axis=1)
    calibrated = scaler.transform(test_logits)
    print(
        json.dumps(
            {
                "seed": 42,
                "train_rows": len(train_y),
                "calibration_rows": len(calibration_y),
                "test_rows": len(test_y),
                "temperature": scaler.temperature_,
                "test_log_loss_before": log_loss(test_y, raw, labels=np.arange(3)),
                "test_log_loss_after": log_loss(test_y, calibrated, labels=np.arange(3)),
                "test_accuracy_before": accuracy_score(test_y, raw.argmax(axis=1)),
                "test_accuracy_after": accuracy_score(test_y, calibrated.argmax(axis=1)),
                "calibration_bins": 10,
                "test_calibration_before": multiclass_calibration_metrics(test_y, raw),
                "test_calibration_after": multiclass_calibration_metrics(test_y, calibrated),
                "test_classwise_before": classwise_calibration_errors(test_y, raw).to_dict(
                    "records"
                ),
                "test_classwise_after": classwise_calibration_errors(test_y, calibrated).to_dict(
                    "records"
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
