"""Check multiclass set membership, finite-sample ranks, and evaluation separation."""

import runpy
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

from calibrated_ml_lab.conformal.classification import (
    conformal_classification_threshold,
    conformal_prediction_sets,
)
from calibrated_ml_lab.conformal.multiclass import (
    multiclass_conformal_threshold,
    multiclass_prediction_sets,
    multiclass_set_coverage,
)
from calibrated_ml_lab.conformal.regression import conformal_residual_quantile


def probabilities_from_scores(scores: np.ndarray) -> np.ndarray:
    """Construct three-class rows whose class-zero score is the supplied value."""
    return np.column_stack([1 - scores, scores / 2, scores / 2])


@pytest.mark.parametrize("alpha,expected", [(0.5, 0.5), (0.8, 0.125), (0.2, 0.75)])
def test_threshold_uses_the_exact_ranked_true_class_score(alpha: float, expected: float) -> None:
    p = probabilities_from_scores(np.array([0.75, 0.125, 0.5, 0.25]))
    assert multiclass_conformal_threshold(np.zeros(4), p, alpha) == expected


def test_membership_includes_boundary_ties_and_preserves_empty_sets() -> None:
    p = np.array([[0.3, 0.4, 0.3]])
    qhat = multiclass_conformal_threshold(np.array([0]), p, alpha=0.5)
    # Reconstructing a probability cutoff as 1 - qhat would exclude 0.3 by roundoff.
    assert qhat == 0.7
    np.testing.assert_array_equal(multiclass_prediction_sets(p, qhat), [[True, True, True]])
    rows = np.array([[0.75, 0.125, 0.125], [0.5, 0.25, 0.25], [0.25, 0.5, 0.25]])
    sets = multiclass_prediction_sets(rows, 0.25)
    assert sets.dtype == bool and sets.shape == rows.shape
    np.testing.assert_array_equal(
        sets, [[True, False, False], [False, False, False], [False, False, False]]
    )


def test_small_calibration_sample_returns_every_class() -> None:
    p = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])
    threshold = multiclass_conformal_threshold(np.array([0, 1]), p)
    assert threshold == np.inf
    sets = multiclass_prediction_sets(p, threshold)
    assert sets.all()
    assert multiclass_set_coverage(np.array([2, 2]), sets) == {
        "coverage": 1.0,
        "average_set_size": 3.0,
        "singleton_rate": 0.0,
        "empty_set_rate": 0.0,
    }


def test_zero_threshold_preserves_exact_probability_endpoints() -> None:
    p = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.5, 0.5, 0.0]])
    sets = multiclass_prediction_sets(p, 0.0)
    np.testing.assert_array_equal(
        sets, [[True, False, False], [False, False, True], [False, False, False]]
    )


def test_coverage_summary_matches_hand_calculation() -> None:
    membership = np.array([[1, 0, 0], [0, 1, 1], [0, 0, 0], [1, 1, 1]])
    summary = multiclass_set_coverage(np.array([0, 0, 2, 2]), membership)
    assert summary == {
        "coverage": 0.5,
        "average_set_size": 1.5,
        "singleton_rate": 0.25,
        "empty_set_rate": 0.25,
    }


@pytest.mark.parametrize("alpha,expected", [(0.1, 1.0), (0.2, 0.8), (0.5, 0.6), (0.8, 0.2)])
def test_marginal_coverage_over_every_possible_held_out_rank(alpha: float, expected: float) -> None:
    scores = np.arange(1, 6) / 8
    p = probabilities_from_scores(scores)
    covered = []
    for index in range(len(p)):
        calibration = np.delete(p, index, axis=0)
        threshold = multiclass_conformal_threshold(np.zeros(4), calibration, alpha)
        sets = multiclass_prediction_sets(p[index : index + 1], threshold)
        covered.append(multiclass_set_coverage(np.array([0]), sets)["coverage"])
    assert np.mean(covered) == expected


@pytest.mark.parametrize("size", [1, 2, 9, 48])
def test_alpha_at_and_below_the_finite_rank_boundary(size: int) -> None:
    p = np.tile([0.25, 0.5, 0.25], (size, 1))
    alpha = 1 / (size + 1)
    assert multiclass_conformal_threshold(np.zeros(size), p, alpha) == 0.75
    assert multiclass_conformal_threshold(np.zeros(size), p, np.nextafter(alpha, 0)) == np.inf


def test_class_and_row_permutations_preserve_sets_and_coverage() -> None:
    p = np.array([[0.75, 0.125, 0.125], [0.25, 0.5, 0.25], [0.25, 0.25, 0.5], [0.5, 0.25, 0.25]])
    y = np.array([0, 1, 2, 1])
    original_p, original_y = p.copy(), y.copy()
    threshold = multiclass_conformal_threshold(y, p, alpha=0.5)
    sets = multiclass_prediction_sets(p, threshold)
    permutation = np.array([2, 0, 1])
    remapped_y = np.argsort(permutation)[y]
    assert multiclass_conformal_threshold(remapped_y, p[:, permutation], 0.5) == threshold
    assert multiclass_conformal_threshold(y[::-1], p[::-1], 0.5) == threshold
    permuted_sets = multiclass_prediction_sets(p[:, permutation], threshold)
    np.testing.assert_array_equal(permuted_sets, sets[:, permutation])
    assert multiclass_set_coverage(remapped_y, permuted_sets) == multiclass_set_coverage(y, sets)
    np.testing.assert_array_equal(p, original_p)
    np.testing.assert_array_equal(y, original_y)


def test_two_column_case_agrees_with_binary_api_on_exact_probabilities() -> None:
    positive = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    y = np.array([0, 0, 1, 0, 1])
    p = np.column_stack([1 - positive, positive])
    threshold = multiclass_conformal_threshold(y, p, alpha=0.25)
    assert threshold == conformal_classification_threshold(y, positive, alpha=0.25)
    binary = conformal_prediction_sets(positive, threshold)
    np.testing.assert_array_equal(
        multiclass_prediction_sets(p, threshold), binary[["include_0", "include_1"]]
    )


def test_sets_grow_monotonically_with_the_threshold() -> None:
    p = probabilities_from_scores(np.array([0.0, 0.25, 0.5, 0.75, 1.0]))
    previous = np.zeros_like(p, dtype=bool)
    for threshold in [0.0, 0.25, 0.5, 0.75, 1.0, np.inf]:
        current = multiclass_prediction_sets(p, threshold)
        assert (current | ~previous).all()
        previous = current
    assert previous.all()


@pytest.mark.parametrize(
    "p",
    [
        [],
        [0.2, 0.8],
        0.5,
        np.empty((0, 3)),
        [[1]],
        [[[0.2, 0.8]]],
        [[0.2, np.nan]],
        [[0.2, np.inf]],
        [[0.2 + 1j, 0.8]],
        [["0.2", "0.8"]],
        [[-0.1, 1.1]],
        [[0.2, 0.2]],
        [[0, 0]],
        [[0.5, 0.500002]],
    ],
)
def test_invalid_probability_matrices_are_rejected(p) -> None:
    with pytest.raises(ValueError):
        multiclass_conformal_threshold(np.array([0]), np.asarray(p))
    with pytest.raises(ValueError):
        multiclass_prediction_sets(np.asarray(p), 0.5)


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
        multiclass_conformal_threshold(np.asarray(y), np.array([[0.2, 0.5, 0.3], [0.4, 0.1, 0.5]]))
    with pytest.raises(ValueError):
        multiclass_set_coverage(np.asarray(y), np.ones((2, 3), dtype=bool))


@pytest.mark.parametrize(
    "membership",
    [
        [],
        [True, False],
        np.empty((0, 3)),
        [[True]],
        [[0.2, 1]],
        [[2, 0]],
        [[-1, 1]],
        [[np.nan, 1]],
        [[np.inf, 0]],
        [[1j, 0]],
        [["0", "1"]],
        np.array([[0, "1"]], dtype=object),
    ],
)
def test_malformed_membership_is_rejected(membership) -> None:
    with pytest.raises(ValueError):
        multiclass_set_coverage(np.array([0]), np.asarray(membership))


@pytest.mark.parametrize("alpha", [0, 1, -0.1, np.nan, np.inf, -np.inf, 0.1j, "0.1", None, [0.1]])
def test_invalid_alpha_is_rejected(alpha) -> None:
    with pytest.raises(ValueError):
        multiclass_conformal_threshold(np.array([0]), np.array([[0.2, 0.5, 0.3]]), alpha)
    with pytest.raises(ValueError):
        conformal_classification_threshold(np.array([0]), np.array([0.5]), alpha)
    with pytest.raises(ValueError):
        conformal_residual_quantile(np.array([0.0]), np.array([0.0]), alpha)


@pytest.mark.parametrize("threshold", [-1, -np.inf, np.nan, 0.1j, "0.1", None, [0.1]])
def test_invalid_threshold_is_rejected(threshold) -> None:
    with pytest.raises(ValueError):
        multiclass_prediction_sets(np.array([[0.2, 0.5, 0.3]]), threshold)


def test_example_splits_are_disjoint_and_held_out_labels_cannot_change_fitting_rows() -> None:
    example = Path(__file__).resolve().parents[1] / "examples/run_multiclass.py"
    split_data = runpy.run_path(str(example))["split_multiclass_data"]
    features, labels = np.arange(1000).reshape(-1, 1), np.arange(1000) % 3
    splits = split_data(features, labels)
    row_sets = {name: set(x[:, 0]) for name, (x, _) in splits.items()}
    assert [len(rows) for rows in row_sets.values()] == [600, 200, 100, 100]
    assert set.union(*row_sets.values()) == set(range(1000))
    assert all(left.isdisjoint(right) for left, right in combinations(row_sets.values(), 2))
    repeated = split_data(features, labels)
    for name in splits:
        np.testing.assert_array_equal(splits[name][0], repeated[name][0])
        np.testing.assert_array_equal(splits[name][1], repeated[name][1])
    changed_labels = labels.copy()
    held_out_rows = list(row_sets["conformal"] | row_sets["test"])
    changed_labels[held_out_rows] = (changed_labels[held_out_rows] + 1) % 3
    changed = split_data(features, changed_labels)
    for name in ("train", "calibration"):
        np.testing.assert_array_equal(splits[name][0], changed[name][0])
        np.testing.assert_array_equal(splits[name][1], changed[name][1])
    for name in ("conformal", "test"):
        np.testing.assert_array_equal(splits[name][0], changed[name][0])
