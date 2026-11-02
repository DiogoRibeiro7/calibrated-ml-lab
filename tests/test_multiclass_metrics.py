"""Check multiclass diagnostics against hand calculations and calibration counterexamples."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from calibrated_ml_lab.calibration.multiclass_metrics import (
    classwise_calibration_errors,
    confidence_calibration_bins,
    multiclass_calibration_metrics,
    top_label_calibration_bins,
)

DIAGNOSTICS = [
    confidence_calibration_bins,
    top_label_calibration_bins,
    classwise_calibration_errors,
    multiclass_calibration_metrics,
]


@pytest.fixture
def sample() -> tuple[np.ndarray, np.ndarray]:
    return np.array([0, 0, 2, 1]), np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.2, 0.6],
            [0.6, 0.3, 0.1],
        ]
    )


def test_diagnostics_match_hand_calculations(sample) -> None:
    y, p = sample
    metrics = multiclass_calibration_metrics(y, p, n_bins=2)
    assert metrics == pytest.approx(
        {"confidence_ece": 0.2, "top_label_ece": 0.4, "classwise_ece": 0.25}
    )
    confidence = confidence_calibration_bins(y, p, n_bins=2)
    assert confidence["count"].tolist() == [0, 4]
    assert confidence.loc[1, "mean_confidence"] == pytest.approx(0.7)
    assert confidence.loc[1, "accuracy"] == 0.5
    top = top_label_calibration_bins(y, p, n_bins=2)
    assert len(top) == 6
    assert top.groupby("predicted_class")["count"].sum().tolist() == [2, 1, 1]
    np.testing.assert_allclose(top.groupby("predicted_class")["weight"].sum(), [0.5, 0.25, 0.25])
    assert top["weight"].sum() == 1
    per_class = classwise_calibration_errors(y, p, n_bins=2)
    assert per_class["class_index"].tolist() == [0, 1, 2]
    assert per_class["count"].tolist() == [4, 4, 4]
    assert per_class["positive_count"].tolist() == [2, 1, 1]
    np.testing.assert_allclose(per_class["ece"], [0.275, 0.3, 0.175])
    np.testing.assert_allclose(per_class["mce"], [0.35, 0.8, 0.4])


def test_pooled_confidence_can_hide_top_label_miscalibration() -> None:
    p = np.array([[0.75, 0.25]] * 2 + [[0.25, 0.75]] * 2)
    y = np.array([0, 0, 0, 1])
    metrics = multiclass_calibration_metrics(y, p, n_bins=4)
    assert metrics["confidence_ece"] == 0
    assert metrics["top_label_ece"] == 0.25
    assert metrics["classwise_ece"] == 0.25


def test_absent_and_unpredicted_classes_stay_visible_in_macro_average() -> None:
    y = np.zeros(3)
    p = np.tile([0.6, 0.3, 0.1], (3, 1))
    per_class = classwise_calibration_errors(y, p, n_bins=5)
    assert per_class["positive_count"].tolist() == [3, 0, 0]
    np.testing.assert_allclose(per_class["ece"], [0.4, 0.3, 0.1])
    assert multiclass_calibration_metrics(y, p, n_bins=5)["classwise_ece"] == pytest.approx(0.8 / 3)
    top = top_label_calibration_bins(y, p, n_bins=5)
    assert len(top) == 15
    empty = top[top["count"] == 0]
    assert empty[["weight", "absolute_gap"]].eq(0).all().all()
    assert empty[["mean_confidence", "accuracy"]].isna().all().all()
    assert top.groupby("predicted_class")["count"].sum().tolist() == [3, 0, 0]


def test_perfect_endpoints_produce_zero_error_including_missing_class() -> None:
    y, p = np.array([0, 1]), np.eye(3)[:2]
    assert set(multiclass_calibration_metrics(y, p).values()) == {0.0}
    assert classwise_calibration_errors(y, p)[["ece", "mce"]].eq(0).all().all()
    confidence = confidence_calibration_bins(y, p, n_bins=4)
    assert confidence["count"].tolist() == [0, 0, 0, 2]
    assert confidence.loc[3, "mean_confidence"] == 1


def test_certain_wrong_predictions_use_the_full_error_range() -> None:
    metrics = multiclass_calibration_metrics(np.array([1, 2, 0]), np.eye(3), n_bins=2)
    assert metrics == pytest.approx(
        {"confidence_ece": 1.0, "top_label_ece": 1.0, "classwise_ece": 2 / 3}
    )


def test_exact_ties_choose_the_first_class_column() -> None:
    y = np.array([0, 2, 1])
    p = np.array([[0.5, 0.5, 0], [0, 0.5, 0.5], [1 / 3] * 3])
    top = top_label_calibration_bins(y, p, n_bins=1)
    assert top["count"].tolist() == [2, 1, 0]
    np.testing.assert_allclose(top["accuracy"], [0.5, 0.0, np.nan], equal_nan=True)


def test_bin_boundaries_are_left_inclusive() -> None:
    p = np.array([[0.5, 0.5, 0], [0.75, 0.25, 0], [1, 0, 0]])
    bins = confidence_calibration_bins(np.array([0, 1, 0]), p, n_bins=4)
    assert bins["count"].tolist() == [0, 0, 1, 2]
    np.testing.assert_allclose(bins["bin_lower"], [0.0, 0.25, 0.5, 0.75])


def test_class_permutation_and_row_order_preserve_summary_without_ties(sample) -> None:
    y, p = sample
    expected = multiclass_calibration_metrics(y, p, n_bins=2)
    permutation = np.array([2, 0, 1])
    mapped_labels = np.argsort(permutation)[y]
    assert multiclass_calibration_metrics(
        mapped_labels, p[:, permutation], n_bins=2
    ) == pytest.approx(expected)
    assert multiclass_calibration_metrics(y[::-1], p[::-1], n_bins=2) == pytest.approx(expected)
    original = classwise_calibration_errors(y, p, n_bins=2)
    permuted = classwise_calibration_errors(mapped_labels, p[:, permutation], n_bins=2)
    np.testing.assert_allclose(permuted["ece"], original["ece"].to_numpy()[permutation])


@pytest.mark.parametrize("diagnostic", DIAGNOSTICS)
def test_inputs_are_not_mutated_and_calls_are_reproducible(diagnostic, sample) -> None:
    y, p = sample
    original_y, original_p = y.copy(), p.copy()
    first, second = diagnostic(y, p), diagnostic(y, p)
    if isinstance(first, pd.DataFrame):
        pd.testing.assert_frame_equal(first, second)
    else:
        assert first == second
    np.testing.assert_array_equal(y, original_y)
    np.testing.assert_array_equal(p, original_p)


@pytest.mark.parametrize(
    "p",
    [
        [],
        [0.2, 0.8],
        0.5,
        np.empty((0, 3)),
        np.empty((2, 0)),
        [[1]],
        [[[0.2, 0.8]]],
        [[0.2, np.nan]],
        [[0.2, np.inf]],
        [[0.2 + 1j, 0.8]],
        [["0.2", "0.8"]],
        np.array([[0.2, "0.8"]], dtype=object),
        [[-1e-9, 1.000000001]],
        [[0.2, 0.2]],
        [[0, 0]],
        [[0.5, 0.500002]],
    ],
)
@pytest.mark.parametrize("diagnostic", DIAGNOSTICS)
def test_invalid_probability_matrices_are_rejected(diagnostic, p) -> None:
    with pytest.raises(ValueError):
        diagnostic(np.array([0]), np.asarray(p))


@pytest.mark.parametrize(
    "y",
    [
        [],
        0,
        [[0], [1]],
        [0],
        [0, 1, 2],
        [0, 0.5],
        [-1, 1],
        [0, 3],
        [0, np.nan],
        [0, np.inf],
        [0, 1j],
        ["0", "1"],
        np.array([0, "1"], dtype=object),
    ],
)
def test_invalid_class_indices_are_rejected(y) -> None:
    with pytest.raises(ValueError):
        multiclass_calibration_metrics(np.asarray(y), np.array([[0.2, 0.5, 0.3], [0.4, 0.1, 0.5]]))


@pytest.mark.parametrize("n_bins", [0, -1, 2.5, np.nan, np.inf, True])
@pytest.mark.parametrize("diagnostic", DIAGNOSTICS)
def test_invalid_bin_counts_are_rejected(diagnostic, n_bins, sample) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        diagnostic(*sample, n_bins=n_bins)


def test_numpy_bin_count_and_numeric_object_arrays_are_supported() -> None:
    y = np.array([0, 1], dtype=object)
    p = np.array([[1, 0], [0, 1]], dtype=object)
    assert set(multiclass_calibration_metrics(y, p, n_bins=np.int64(3)).values()) == {0.0}


def test_small_row_sum_roundoff_is_accepted_without_renormalization() -> None:
    p = np.array([[0.5, 0.5000005]])
    bins = confidence_calibration_bins(np.array([1]), p, n_bins=1)
    assert bins.loc[0, "mean_confidence"] == p[0, 1]
    float32_p = np.array([[0.1, 0.2, 0.7]], dtype=np.float32)
    assert len(top_label_calibration_bins(np.array([2]), float32_p, n_bins=1)) == 3
