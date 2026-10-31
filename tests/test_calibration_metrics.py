import numpy as np
import pytest

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


def test_perfect_endpoints_remain_exact_and_empty_bins_remain_visible() -> None:
    y, p = np.array([0, 1]), np.array([0.0, 1.0])
    metrics = probabilistic_metrics(y, p, n_bins=4)
    assert metrics["brier_score"] == metrics["ece"] == metrics["mce"] == 0.0
    assert metrics["roc_auc"] == 1.0
    assert 0 <= metrics["log_loss"] < 1e-10
    bins = calibration_bins(y, p, n_bins=4)
    assert bins["count"].tolist() == [1, 0, 0, 1]
    assert bins.loc[[0, 3], "mean_predicted_probability"].tolist() == [0.0, 1.0]
    assert bins.loc[[1, 2], ["weight", "absolute_gap"]].eq(0).all().all()
    assert (
        bins.loc[[1, 2], ["mean_predicted_probability", "observed_event_rate"]].isna().all().all()
    )


@pytest.mark.parametrize("probabilities", [[0.0, 1e-15], [1 - 1e-15, 1.0]])
def test_auc_preserves_distinct_scores_near_endpoints(probabilities: list[float]) -> None:
    p = np.array(probabilities)
    before = p.copy()
    assert probabilistic_metrics(np.array([0, 1]), p)["roc_auc"] == 1.0
    np.testing.assert_array_equal(p, before)


def test_metrics_match_hand_calculated_bin_boundaries() -> None:
    y, p = np.array([0, 1, 1, 0]), np.array([0.0, 0.25, 0.5, 1.0])
    bins = calibration_bins(y, p, n_bins=4)
    assert bins["count"].tolist() == [1, 1, 1, 1]
    metrics = probabilistic_metrics(y, p, n_bins=4)
    assert metrics["ece"] == 0.5625
    assert metrics["mce"] == 1.0
    assert metrics["brier_score"] == 0.453125
    assert metrics["roc_auc"] == 0.5
    assert np.isfinite(metrics["log_loss"])


@pytest.mark.parametrize("label", [0, 1])
def test_single_class_metrics_keep_auc_undefined(label: int) -> None:
    metrics = probabilistic_metrics(np.full(3, label), np.full(3, float(label)))
    assert np.isnan(metrics["roc_auc"])
    assert metrics["brier_score"] == metrics["ece"] == metrics["mce"] == 0.0
    assert np.isfinite(metrics["log_loss"])


@pytest.mark.parametrize("n_bins", [0, -1, 2.5, np.nan, np.inf, True])
def test_bin_count_requires_a_positive_integer(n_bins: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        calibration_bins(np.array([0, 1]), np.array([0.2, 0.8]), n_bins=n_bins)


def test_numpy_integer_bin_count_is_supported() -> None:
    assert len(calibration_bins(np.array([0, 1]), np.array([0.0, 1.0]), n_bins=np.int64(3))) == 3
