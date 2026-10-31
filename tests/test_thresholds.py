"""Verify threshold boundaries, decision costs, and invalid input handling."""

import numpy as np
import pytest

from calibrated_ml_lab.uncertainty.thresholds import threshold_metrics


def test_threshold_metrics_match_hand_calculated_confusion_counts() -> None:
    table = threshold_metrics(
        np.array([0, 1, 0, 1]),
        np.array([0.0, 0.5, 0.5, 1.0]),
        thresholds=np.array([0.0, 0.5, 1.0]),
        cost_false_positive=2,
        cost_false_negative=7,
    )
    assert table[["tp", "fp", "tn", "fn"]].values.tolist() == [
        [2, 2, 0, 0],
        [2, 1, 1, 0],
        [1, 0, 2, 1],
    ]
    np.testing.assert_allclose(table["precision"], [0.5, 2 / 3, 1.0])
    np.testing.assert_allclose(table["recall"], [1.0, 1.0, 0.5])
    np.testing.assert_allclose(table["f1"], [2 / 3, 0.8, 2 / 3])
    np.testing.assert_allclose(table["expected_cost"], [1.0, 0.5, 1.75])
    np.testing.assert_allclose(table["alert_rate"], [1.0, 0.75, 0.25])


@pytest.mark.parametrize("thresholds", [[], [[0.5]], 0.5, [np.nan], [np.inf], [-0.1], [1.1]])
def test_invalid_thresholds_are_rejected(thresholds: object) -> None:
    with pytest.raises(ValueError, match="thresholds"):
        threshold_metrics(np.array([0, 1]), np.array([0.2, 0.8]), thresholds=np.asarray(thresholds))


@pytest.mark.parametrize("cost_name", ["cost_false_positive", "cost_false_negative"])
@pytest.mark.parametrize("cost", [-1.0, np.nan, np.inf, np.array([1.0])])
def test_costs_must_be_finite_nonnegative_scalars(cost_name: str, cost: object) -> None:
    with pytest.raises(ValueError, match=cost_name):
        threshold_metrics(np.array([0, 1]), np.array([0.2, 0.8]), **{cost_name: cost})


def test_large_finite_costs_do_not_overflow_before_averaging() -> None:
    table = threshold_metrics(
        np.zeros(4), np.ones(4), thresholds=np.array([0.5]), cost_false_positive=1e308
    )
    assert table["expected_cost"].item() == 1e308


@pytest.mark.parametrize("label", [0, 1])
def test_single_class_inputs_and_zero_costs_are_supported(label: int) -> None:
    table = threshold_metrics(
        np.full(3, label),
        np.full(3, float(label)),
        thresholds=np.array([0.5]),
        cost_false_positive=0,
        cost_false_negative=0,
    )
    assert table["expected_cost"].item() == 0.0
    assert table["alert_rate"].item() == label
    assert table["tp"].item() + table["tn"].item() == 3
    assert table["precision"].item() == table["recall"].item() == table["f1"].item() == label


def test_threshold_order_and_duplicates_are_preserved() -> None:
    thresholds = np.array([1.0, 0.0, 0.5, 0.5])
    table = threshold_metrics(np.array([0, 1]), np.array([0.0, 1.0]), thresholds)
    assert table["threshold"].tolist() == thresholds.tolist()
    assert table.iloc[2].equals(table.iloc[3])
    np.testing.assert_array_equal(thresholds, [1.0, 0.0, 0.5, 0.5])
