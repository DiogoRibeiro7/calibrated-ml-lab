import numpy as np
import pandas as pd
import pytest

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


@pytest.mark.parametrize("alpha,expected", [(0.5, 0.5), (0.8, 0.125), (0.2, 0.75)])
def test_exact_conformal_order_statistic(alpha: float, expected: float) -> None:
    scores = np.array([0.75, 0.125, 0.5, 0.25])
    original = scores.copy()
    assert conformal_classification_threshold(np.zeros(4), scores, alpha) == expected
    assert conformal_residual_quantile(scores, np.zeros(4), alpha) == expected
    np.testing.assert_array_equal(scores, original)


def test_insufficient_calibration_returns_full_prediction_space() -> None:
    q_class = conformal_classification_threshold(np.array([0, 1]), np.array([0.2, 0.8]))
    q_reg = conformal_residual_quantile(np.array([1.0, 2.0]), np.zeros(2))
    assert q_class == q_reg == np.inf
    sets = conformal_prediction_sets(np.array([0.0, 0.5, 1.0]), q_class)
    assert sets["set_size"].eq(2).all()
    assert classification_set_coverage(np.array([0, 1, 1]), sets)["coverage"] == 1.0
    lower, upper = conformal_intervals(np.array([0.0, 100.0]), q_reg)
    np.testing.assert_array_equal(lower, [-np.inf, -np.inf])
    np.testing.assert_array_equal(upper, [np.inf, np.inf])
    summary = regression_interval_metrics(np.array([-1e6, 1e6]), lower, upper)
    assert summary == {
        "coverage": 1.0,
        "average_width": np.inf,
        "median_width": np.inf,
        "min_width": np.inf,
        "max_width": np.inf,
    }


def test_exact_rank_handles_ties_and_zero_scores() -> None:
    assert conformal_residual_quantile(np.array([3.0, 1.0, 1.0, 0.0]), np.zeros(4), 0.5) == 1.0
    assert conformal_classification_threshold(np.array([1]), np.array([1.0]), 0.5) == 0.0
    assert conformal_residual_quantile(np.array([3.0]), np.array([3.0]), 0.5) == 0.0


@pytest.mark.parametrize("size", [1, 2, 9, 48])
def test_alpha_at_and_just_below_minimum_finite_threshold(size: int) -> None:
    scores = np.arange(1.0, size + 1.0)
    alpha = 1 / (size + 1)
    assert conformal_residual_quantile(scores, np.zeros(size), alpha=alpha) == size
    assert (
        conformal_residual_quantile(scores, np.zeros(size), alpha=np.nextafter(alpha, 0.0))
        == np.inf
    )


def test_rank_avoids_subtraction_roundoff_at_integer_boundary() -> None:
    # Directly evaluating 100 * (1 - 0.7) gives 30.000000000000004.
    scores = np.arange(1.0, 100.0)
    assert conformal_residual_quantile(scores, np.zeros(99), alpha=0.7) == 30.0


def test_prediction_set_boundaries_use_the_same_scores_as_calibration() -> None:
    tiny = 1e-20
    threshold = conformal_classification_threshold(np.array([0]), np.array([tiny]), alpha=0.5)
    assert threshold == tiny
    sets = conformal_prediction_sets(np.array([tiny, np.nextafter(tiny, np.inf)]), threshold)
    assert sets["include_0"].tolist() == [True, False]
    sets = conformal_prediction_sets(np.array([0.0, 0.5, 1.0]), 0.0)
    assert sets["set_size"].tolist() == [1, 0, 1]


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, np.nan, np.inf])
def test_invalid_alpha_is_rejected(alpha: float) -> None:
    with pytest.raises(ValueError):
        conformal_classification_threshold(np.array([0]), np.array([0.5]), alpha)
    with pytest.raises(ValueError):
        conformal_residual_quantile(np.array([0.0]), np.array([0.0]), alpha)


@pytest.mark.parametrize("labels", [[0.5], [2], [-1], [np.nan], [np.inf]])
def test_invalid_class_labels_are_rejected(labels: list[float]) -> None:
    with pytest.raises(ValueError):
        conformal_classification_threshold(np.array(labels), np.array([0.5]))
    with pytest.raises(ValueError):
        classification_set_coverage(
            np.array(labels), conformal_prediction_sets(np.array([0.5]), 0.5)
        )


@pytest.mark.parametrize("probabilities", [[-0.1], [1.1], [np.nan], [np.inf]])
def test_invalid_probabilities_are_rejected(probabilities: list[float]) -> None:
    with pytest.raises(ValueError):
        conformal_classification_threshold(np.array([0]), np.array(probabilities))
    with pytest.raises(ValueError):
        conformal_prediction_sets(np.array(probabilities), 0.5)


@pytest.mark.parametrize("values", [np.array([]), np.array(0.5), np.array([[0.5]])])
def test_empty_and_nonvector_inputs_are_rejected(values: np.ndarray) -> None:
    with pytest.raises(ValueError):
        conformal_classification_threshold(values, values)
    with pytest.raises(ValueError):
        conformal_residual_quantile(values, values)
    with pytest.raises(ValueError):
        conformal_prediction_sets(values, 0.5)
    with pytest.raises(ValueError):
        conformal_intervals(values, 1.0)
    with pytest.raises(ValueError):
        regression_interval_metrics(values, values, values)
    with pytest.raises(ValueError):
        classification_set_coverage(
            values, pd.DataFrame(columns=["include_0", "include_1", "set_size"])
        )


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_regression_targets_and_predictions_must_be_finite(value: float) -> None:
    invalid, valid = np.array([value]), np.array([0.0])
    for labels, predictions in [(invalid, valid), (valid, invalid)]:
        with pytest.raises(ValueError):
            conformal_residual_quantile(labels, predictions)
    with pytest.raises(ValueError):
        conformal_intervals(invalid, 1.0)
    with pytest.raises(ValueError):
        regression_interval_metrics(invalid, valid, valid)


@pytest.mark.parametrize("threshold", [-1.0, -np.inf, np.nan])
def test_invalid_thresholds_are_rejected(threshold: float) -> None:
    with pytest.raises(ValueError):
        conformal_prediction_sets(np.array([0.5]), threshold)
    with pytest.raises(ValueError):
        conformal_intervals(np.array([0.0]), threshold)


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(ValueError):
        conformal_classification_threshold(np.array([0, 1]), np.array([0.5]))
    with pytest.raises(ValueError):
        conformal_residual_quantile(np.zeros(2), np.zeros(1))
    with pytest.raises(ValueError):
        regression_interval_metrics(np.zeros(2), np.zeros(1), np.zeros(2))
    with pytest.raises(ValueError):
        classification_set_coverage(
            np.array([0, 1]), conformal_prediction_sets(np.array([0.5]), 0.5)
        )


@pytest.mark.parametrize(
    "lower,upper", [(2.0, 1.0), (np.nan, 1.0), (0.0, np.nan), (np.inf, np.inf), (-np.inf, -np.inf)]
)
def test_invalid_interval_bounds_are_rejected(lower: float, upper: float) -> None:
    with pytest.raises(ValueError):
        regression_interval_metrics(np.array([1.0]), np.array([lower]), np.array([upper]))


@pytest.mark.parametrize("column,value", [("include_0", 2), ("include_1", np.nan), ("set_size", 0)])
def test_malformed_prediction_sets_are_rejected(column: str, value: float) -> None:
    sets = pd.DataFrame({"include_0": [1.0], "include_1": [0.0], "set_size": [1.0]})
    sets.loc[0, column] = value
    with pytest.raises(ValueError):
        classification_set_coverage(np.array([0]), sets)
    with pytest.raises(ValueError):
        classification_set_coverage(np.array([0]), sets.drop(columns=[column]))


def test_regression_residual_overflow_is_rejected() -> None:
    largest = np.finfo(float).max
    with pytest.raises(ValueError, match="finite"):
        conformal_residual_quantile(np.array([largest]), np.array([-largest]))


@pytest.mark.parametrize("alpha,expected", [(0.1, 1.0), (0.2, 0.8), (0.5, 0.6), (0.8, 0.2)])
def test_finite_sample_marginal_coverage_over_all_test_ranks(alpha: float, expected: float) -> None:
    # Enumerate every possible held-out rank, avoiding a flaky Monte Carlo check.
    values = np.arange(1.0, 6.0)
    covered, class_covered = [], []
    for index, held_out in enumerate(values):
        calibration = np.delete(values, index)
        threshold = conformal_residual_quantile(calibration, np.zeros(4), alpha=alpha)
        covered.append(held_out <= threshold)
        class_threshold = conformal_classification_threshold(np.zeros(4), calibration / 8, alpha)
        sets = conformal_prediction_sets(np.array([held_out / 8]), class_threshold)
        class_covered.append(sets["include_0"].iloc[0])
    assert np.mean(covered) == np.mean(class_covered) == expected
