"""Check empirical coverage summaries for classes, sparse groups, and overlapping slices."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from calibrated_ml_lab.conformal.multiclass import (
    multiclass_coverage_by_class,
    multiclass_coverage_by_slice,
    multiclass_set_coverage,
)

STATISTICS = ["coverage", "average_set_size", "singleton_rate", "empty_set_rate"]


@pytest.fixture
def sample() -> tuple[np.ndarray, np.ndarray]:
    y = np.array([0, 0, 1, 1, 2, 2])
    membership = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 0],
            [1, 1, 1, 1],
            [0, 1, 1, 0],
        ]
    )
    return y, membership


def test_class_summaries_match_hand_calculations_and_keep_absent_classes(sample) -> None:
    frame = multiclass_coverage_by_class(*sample)
    assert frame["class_index"].tolist() == [0, 1, 2, 3]
    assert frame["count"].tolist() == [2, 2, 2, 0]
    assert frame["covered_count"].tolist() == [1, 1, 2, 0]
    np.testing.assert_allclose(frame.loc[:2, "coverage"], [0.5, 0.5, 1.0])
    np.testing.assert_allclose(frame.loc[:2, "average_set_size"], [0.5, 1.5, 3.0])
    np.testing.assert_allclose(frame.loc[:2, "singleton_rate"], [0.5, 0.5, 0.0])
    np.testing.assert_allclose(frame.loc[:2, "empty_set_rate"], [0.5, 0.0, 0.0])
    assert frame.loc[3, STATISTICS].isna().all()


def test_good_marginal_coverage_can_hide_an_uncovered_true_class() -> None:
    y = np.array([0] * 9 + [1])
    membership = np.tile([1, 0, 0], (10, 1))
    assert multiclass_set_coverage(y, membership)["coverage"] == 0.9
    groups = multiclass_coverage_by_class(y, membership)
    assert groups["count"].tolist() == [9, 1, 0]
    np.testing.assert_array_equal(groups.loc[:1, "coverage"], [1.0, 0.0])
    assert groups.loc[2, STATISTICS].isna().all()


def test_overlapping_slices_are_counted_independently_in_mapping_order(sample) -> None:
    y, membership = sample
    slices = {"all": np.ones(6), "class_two": y == 2, "none": np.zeros(6)}
    frame = multiclass_coverage_by_slice(y, membership, slices)
    assert frame["slice"].tolist() == ["all", "class_two", "none"]
    assert frame["count"].tolist() == [6, 2, 0]
    assert frame["covered_count"].tolist() == [4, 2, 0]
    assert frame.loc[0, STATISTICS].to_dict() == pytest.approx(
        multiclass_set_coverage(y, membership)
    )
    assert frame.loc[1, STATISTICS].to_dict() == {
        "coverage": 1.0,
        "average_set_size": 3.0,
        "singleton_rate": 0.0,
        "empty_set_rate": 0.0,
    }
    assert frame.loc[2, STATISTICS].isna().all()
    assert frame["count"].sum() > len(y)


def test_slice_partition_reconstructs_overall_statistics(sample) -> None:
    y, membership = sample
    slices = {"first": np.array([1, 1, 1, 0, 0, 0]), "last": np.array([0, 0, 0, 1, 1, 1])}
    frame = multiclass_coverage_by_slice(y, membership, slices)
    np.testing.assert_allclose(frame["coverage"], [2 / 3, 2 / 3])
    np.testing.assert_allclose(frame["average_set_size"], [1, 7 / 3])
    np.testing.assert_allclose(frame["singleton_rate"], [1 / 3, 1 / 3])
    np.testing.assert_allclose(frame["empty_set_rate"], [1 / 3, 0])
    assert frame["count"].sum() == len(y)
    overall = multiclass_set_coverage(y, membership)
    for statistic in STATISTICS:
        assert np.average(frame[statistic], weights=frame["count"]) == pytest.approx(
            overall[statistic]
        )
    assert frame["covered_count"].sum() / len(y) == overall["coverage"]


def test_slices_may_omit_rows_and_an_empty_mapping_keeps_the_table_schema(sample) -> None:
    y, membership = sample
    partial = multiclass_coverage_by_slice(y, membership, {"single": [1, 0, 0, 0, 0, 0]})
    assert partial["count"].tolist() == [1]
    assert partial["coverage"].tolist() == [1.0]
    empty = multiclass_coverage_by_slice(y, membership, {})
    assert empty.empty
    assert empty.columns.tolist() == ["slice", "count", "covered_count", *STATISTICS]


def test_one_row_groups_and_full_or_empty_prediction_sets() -> None:
    y = np.array([0, 1])
    membership = np.array([[1, 1], [0, 0]])
    frame = multiclass_coverage_by_class(y, membership)
    assert frame["count"].tolist() == [1, 1]
    assert frame["coverage"].tolist() == [1.0, 0.0]
    assert frame["average_set_size"].tolist() == [2.0, 0.0]
    assert frame["singleton_rate"].tolist() == [0.0, 0.0]
    assert frame["empty_set_rate"].tolist() == [0.0, 1.0]


def test_class_column_permutation_and_row_order_preserve_results(sample) -> None:
    y, membership = sample
    expected = multiclass_coverage_by_class(y, membership)
    pd.testing.assert_frame_equal(multiclass_coverage_by_class(y[::-1], membership[::-1]), expected)
    permutation = np.array([3, 1, 0, 2])
    actual = multiclass_coverage_by_class(np.argsort(permutation)[y], membership[:, permutation])
    columns = ["count", "covered_count", *STATISTICS]
    pd.testing.assert_frame_equal(
        actual[columns], expected.iloc[permutation][columns].reset_index(drop=True)
    )


def test_masks_are_positional_and_inputs_are_not_modified(sample) -> None:
    y, membership = sample
    mask = np.array([True, False, True, False, True, False])
    original_y, original_membership, original_mask = y.copy(), membership.copy(), mask.copy()
    expected = multiclass_coverage_by_slice(y, membership, {"selected": mask})
    series = pd.Series(mask, index=np.arange(100, 106)[::-1])
    actual = multiclass_coverage_by_slice(y, membership, {"selected": series})
    pd.testing.assert_frame_equal(actual, expected)
    permuted = multiclass_coverage_by_slice(y[::-1], membership[::-1], {"selected": mask[::-1]})
    pd.testing.assert_frame_equal(permuted, expected)
    np.testing.assert_array_equal(y, original_y)
    np.testing.assert_array_equal(membership, original_membership)
    np.testing.assert_array_equal(mask, original_mask)


@pytest.mark.parametrize(
    "mask",
    [
        [],
        True,
        [[1, 0]],
        [1],
        [1, 0, 1],
        [0.5, 0],
        [-1, 1],
        [2, 0],
        [np.nan, 0],
        [np.inf, 0],
        [1j, 0],
        ["1", "0"],
        np.array([True, "0"], dtype=object),
    ],
)
def test_malformed_slice_masks_are_rejected(mask) -> None:
    with pytest.raises(ValueError):
        multiclass_coverage_by_slice(np.array([0, 1]), np.eye(2), {"bad": mask})


@pytest.mark.parametrize(
    "slices",
    [None, [], np.array([True, False]), {1: [1, 0]}, {None: [1, 0]}, {"": [1, 0]}, {" \t": [1, 0]}],
)
def test_invalid_slice_mappings_and_names_are_rejected(slices) -> None:
    with pytest.raises(ValueError):
        multiclass_coverage_by_slice(np.array([0, 1]), np.eye(2), slices)


@pytest.mark.parametrize(
    "y,membership",
    [
        ([0, 9], [[1, 0], [0, 1]]),
        ([0, 0.5], [[1, 0], [0, 1]]),
        ([0, 1], [[1, 0], [0, 2]]),
        ([0, 1], [[1, 0], [0, np.nan]]),
        ([0], [[1, 0], [0, 1]]),
        ([], np.empty((0, 2))),
        ([0], [[1]]),
        ([0, 1], [True, False]),
        ([0, 1], [["1", "0"], ["0", "1"]]),
    ],
)
def test_both_grouped_apis_validate_all_rows_before_filtering(y, membership) -> None:
    with pytest.raises(ValueError):
        multiclass_coverage_by_class(np.asarray(y), np.asarray(membership))
    with pytest.raises(ValueError):
        multiclass_coverage_by_slice(np.asarray(y), np.asarray(membership), {"first": [1, 0]})
    with pytest.raises(ValueError):
        multiclass_coverage_by_slice(np.asarray(y), np.asarray(membership), {})


def test_numeric_object_membership_and_masks_remain_supported() -> None:
    frame = multiclass_coverage_by_slice(
        np.array([0, 1], dtype=object),
        np.array([[True, 0], [0, 1]], dtype=object),
        {"selected": np.array([1, False], dtype=object)},
    )
    assert frame["covered_count"].tolist() == [1]
    assert frame["coverage"].tolist() == [1.0]
