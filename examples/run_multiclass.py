"""Evaluate multiclass calibration and conformal sets on held-out synthetic data."""

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
from calibrated_ml_lab.conformal.multiclass import (
    multiclass_conformal_threshold,
    multiclass_prediction_sets,
    multiclass_set_coverage,
)


def split_multiclass_data(
    features: np.ndarray,
    labels: np.ndarray,
    seed: int = 42,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Allocate roughly 60/20/10/10% to disjoint model, temperature, conformal, and test rows."""
    development_x, held_out_x, development_y, held_out_y = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=seed,
    )
    train_x, calibration_x, train_y, calibration_y = train_test_split(
        development_x,
        development_y,
        test_size=0.25,
        stratify=development_y,
        random_state=seed,
    )
    conformal_x, test_x, conformal_y, test_y = train_test_split(
        held_out_x,
        held_out_y,
        test_size=0.5,
        random_state=seed + 1,
    )
    return {
        "train": (train_x, train_y),
        "calibration": (calibration_x, calibration_y),
        "conformal": (conformal_x, conformal_y),
        "test": (test_x, test_y),
    }


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
    splits = split_multiclass_data(features, labels)
    train_x, train_y = splits["train"]
    calibration_x, calibration_y = splits["calibration"]
    conformal_x, conformal_y = splits["conformal"]
    test_x, test_y = splits["test"]
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
    model.fit(train_x, train_y)
    # Deliberately amplify confidence to illustrate a known source of miscalibration.
    calibration_logits = 3.0 * model.decision_function(calibration_x)
    scaler = MulticlassTemperatureScaler().fit(calibration_logits, calibration_y)
    alpha = 0.1
    conformal_probabilities = scaler.transform(3.0 * model.decision_function(conformal_x))
    threshold = multiclass_conformal_threshold(conformal_y, conformal_probabilities, alpha)
    test_logits = 3.0 * model.decision_function(test_x)
    raw = softmax(test_logits, axis=1)
    calibrated = scaler.transform(test_logits)
    prediction_sets = multiclass_prediction_sets(calibrated, threshold)
    print(
        json.dumps(
            {
                "seed": 42,
                "train_rows": len(train_y),
                "calibration_rows": len(calibration_y),
                "conformal_rows": len(conformal_y),
                "test_rows": len(test_y),
                "temperature": scaler.temperature_,
                "conformal_alpha": alpha,
                "conformal_threshold": threshold,
                "test_conformal": multiclass_set_coverage(test_y, prediction_sets),
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
