import numpy as np

from calibrated_ml_lab.conformal.classification import (
    classification_set_coverage,
    conformal_classification_threshold,
    conformal_prediction_sets,
)
from calibrated_ml_lab.conformal.regression import (
    conformal_intervals,
    conformal_residual_quantile,
    regression_interval_metrics,
)


def test_conformal_classification_outputs_sets():
    y = np.array([0, 1, 1, 0, 1, 0])
    p = np.array([0.1, 0.9, 0.7, 0.2, 0.6, 0.3])
    threshold = conformal_classification_threshold(y, p, alpha=0.2)
    sets = conformal_prediction_sets(p, threshold)
    summary = classification_set_coverage(y, sets)
    assert 0 <= summary["coverage"] <= 1
    assert sets["set_size"].min() >= 0


def test_conformal_regression_intervals_cover_some_points():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.array([1.1, 1.9, 3.1, 3.8])
    qhat = conformal_residual_quantile(y, pred, alpha=0.25)
    lower, upper = conformal_intervals(pred, qhat)
    summary = regression_interval_metrics(y, lower, upper)
    assert summary["coverage"] >= 0.75
    assert summary["average_width"] > 0
