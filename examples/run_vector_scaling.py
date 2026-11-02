"""Compare fixed calibration methods on independent synthetic logit/label pairs."""

from __future__ import annotations

import json

import numpy as np
from scipy.special import softmax
from sklearn.metrics import accuracy_score, log_loss

from calibrated_ml_lab.calibration.multiclass import MulticlassTemperatureScaler
from calibrated_ml_lab.calibration.multiclass_metrics import multiclass_calibration_metrics
from calibrated_ml_lab.calibration.vector import MulticlassVectorScaler


def main() -> None:
    """Print held-out metrics for a fixed class-specific distortion, without selecting a winner."""
    rng = np.random.default_rng(42)
    true_logits = rng.normal(size=(3000, 3))
    true_probabilities = softmax(true_logits, axis=1)
    labels = (rng.random((3000, 1)) > true_probabilities.cumsum(axis=1)[:, :-1]).sum(axis=1)
    # This frozen generator stands in for model outputs; it learns nothing from test rows.
    logits = true_logits * [0.6, 1.8, 3.0] + [1.0, -0.5, 0.3]
    calibration_logits, test_logits = logits[:2000], logits[2000:]
    calibration_labels, test_labels = labels[:2000], labels[2000:]
    temperature = MulticlassTemperatureScaler().fit(calibration_logits, calibration_labels)
    # Fix the method and regularization before examining test outcomes.
    vector = MulticlassVectorScaler(l2=1e-3).fit(calibration_logits, calibration_labels)
    probabilities = {
        "uncalibrated": softmax(test_logits, axis=1),
        "temperature": temperature.transform(test_logits),
        "vector": vector.transform(test_logits),
    }
    metrics = {
        name: {
            "log_loss": log_loss(test_labels, values, labels=np.arange(3)),
            "accuracy": accuracy_score(test_labels, values.argmax(axis=1)),
            **multiclass_calibration_metrics(test_labels, values),
        }
        for name, values in probabilities.items()
    }
    print(
        json.dumps(
            {
                "seed": 42,
                "calibration_rows": len(calibration_labels),
                "test_rows": len(test_labels),
                "temperature": temperature.temperature_,
                "vector_l2": vector.l2,
                "vector_weights": vector.weights_.tolist(),
                "vector_bias": vector.bias_.tolist(),
                "test_metrics": metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
