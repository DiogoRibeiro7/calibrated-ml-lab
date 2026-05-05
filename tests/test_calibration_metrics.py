import numpy as np

from calibrated_ml_lab.calibration.metrics import (
    calibration_bins,
    expected_calibration_error,
    maximum_calibration_error,
    probabilistic_metrics,
)


def test_calibration_bins_preserve_requested_number_of_bins():
    y = np.array([0, 1, 1, 0])
    p = np.array([0.1, 0.6, 0.8, 0.2])
    bins = calibration_bins(y, p, n_bins=5)
    assert len(bins) == 5
    assert bins["count"].sum() == 4


def test_ece_and_mce_are_non_negative():
    y = np.array([0, 1, 1, 0, 1])
    p = np.array([0.05, 0.9, 0.7, 0.2, 0.6])
    assert expected_calibration_error(y, p) >= 0
    assert maximum_calibration_error(y, p) >= 0


def test_probabilistic_metrics_include_core_metrics():
    y = np.array([0, 1, 1, 0, 1, 0])
    p = np.array([0.05, 0.9, 0.8, 0.2, 0.7, 0.3])
    metrics = probabilistic_metrics(y, p)
    assert {"brier_score", "log_loss", "ece", "mce", "roc_auc"}.issubset(metrics)
