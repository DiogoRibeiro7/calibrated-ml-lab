"""Check regression mixture moments, numerical limits, and offline exports."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from calibrated_ml_lab.uncertainty.ensemble import ensemble_regression_summary


@pytest.fixture
def members() -> tuple[np.ndarray, np.ndarray]:
    return np.array([[0.0, 10], [2.0, 14]]), np.array([[1.0, 4], [9.0, 0]])


def test_hand_calculated_decomposition_matches_an_explicit_discrete_mixture(members) -> None:
    table = ensemble_regression_summary(*members)
    assert table.columns.tolist() == [
        "mean_prediction",
        "within_model_variance",
        "between_model_variance",
        "total_variance",
        "total_std",
    ]
    np.testing.assert_allclose(table["mean_prediction"], [1, 12])
    np.testing.assert_allclose(table["within_model_variance"], [5, 2])
    np.testing.assert_allclose(table["between_model_variance"], [1, 4])
    np.testing.assert_allclose(table["total_variance"], [6, 6])
    np.testing.assert_allclose(table["total_std"], np.sqrt([6, 6]))
    # Each component puts half its mass at mean +/- sqrt(variance).
    mixture_points = np.array([[-1, 8], [1, 12], [-1, 14], [5, 14]])
    np.testing.assert_allclose(table["mean_prediction"], mixture_points.mean(axis=0))
    np.testing.assert_allclose(table["total_variance"], mixture_points.var(axis=0))


def test_member_replication_preserves_mixture_variance_instead_of_shrinking_it(members) -> None:
    expected = ensemble_regression_summary(*members)
    repeated = [np.repeat(values, 3, axis=0) for values in members]
    pd.testing.assert_frame_equal(ensemble_regression_summary(*repeated), expected)
    table = ensemble_regression_summary(np.array([[0], [2]]), np.zeros((2, 1)))
    assert table["between_model_variance"].item() == 1


def test_member_and_query_permutations_preserve_results(members) -> None:
    means, variances = members
    expected = ensemble_regression_summary(means, variances)
    pd.testing.assert_frame_equal(
        ensemble_regression_summary(means[::-1], variances[::-1]), expected
    )
    pd.testing.assert_frame_equal(
        ensemble_regression_summary(means[:, ::-1], variances[:, ::-1]),
        expected.iloc[::-1].reset_index(drop=True),
    )


def test_one_member_keeps_its_predictive_noise_without_disagreement() -> None:
    table = ensemble_regression_summary(np.array([[2, -3]]), np.array([[4, 9]]))
    np.testing.assert_array_equal(table["mean_prediction"], [2, -3])
    np.testing.assert_array_equal(table["between_model_variance"], [0, 0])
    np.testing.assert_array_equal(table["within_model_variance"], [4, 9])
    np.testing.assert_array_equal(table["total_variance"], [4, 9])
    np.testing.assert_array_equal(table["total_std"], [2, 3])


def test_identical_means_keep_nonzero_within_model_variance() -> None:
    table = ensemble_regression_summary(np.full((3, 1), 2.0), np.array([[1], [4], [7]]))
    assert table["mean_prediction"].item() == 2
    assert table["within_model_variance"].item() == 4
    assert table["between_model_variance"].item() == 0
    assert table["total_std"].item() == 2


def test_identical_deterministic_members_can_report_zero_uncertainty() -> None:
    table = ensemble_regression_summary(np.zeros((2, 3)), np.zeros((2, 3)))
    assert table.eq(0).all().all()


def test_translation_and_rescaling_have_the_expected_effect(members) -> None:
    means, variances = members
    expected = ensemble_regression_summary(*members)
    shifted = ensemble_regression_summary(means + 1e12, variances)
    np.testing.assert_allclose(shifted["mean_prediction"], expected["mean_prediction"] + 1e12)
    np.testing.assert_allclose(shifted.iloc[:, 1:], expected.iloc[:, 1:])
    scaled = ensemble_regression_summary(means * -5, variances * 25)
    np.testing.assert_allclose(scaled["mean_prediction"], expected["mean_prediction"] * -5)
    np.testing.assert_allclose(scaled.iloc[:, 1:4], expected.iloc[:, 1:4] * 25)
    np.testing.assert_allclose(scaled["total_std"], expected["total_std"] * 5)


def test_pandas_alignment_is_positional_and_inputs_are_unchanged(members) -> None:
    means, variances = members
    expected = ensemble_regression_summary(means, variances)
    before_means, before_variances = means.copy(), variances.copy()
    means.setflags(write=False)
    variances.setflags(write=False)
    actual = ensemble_regression_summary(
        pd.DataFrame(means, index=[9, 8], columns=["a", "b"]),
        pd.DataFrame(variances, index=[20, 21], columns=["y", "x"]),
    )
    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(ensemble_regression_summary(means, variances), expected)
    np.testing.assert_array_equal(means, before_means)
    np.testing.assert_array_equal(variances, before_variances)


def test_numeric_objects_and_boolean_values_are_supported() -> None:
    table = ensemble_regression_summary(
        np.array([[False], [1]], dtype=object), np.array([[True], [0]], dtype=object)
    )
    assert table["mean_prediction"].item() == 0.5
    assert table["within_model_variance"].item() == 0.5
    assert table["between_model_variance"].item() == 0.25
    assert table["total_variance"].item() == 0.75


def test_large_common_offset_does_not_destroy_small_disagreement() -> None:
    table = ensemble_regression_summary(np.array([[1e12], [1e12 + 2]]), np.zeros((2, 1)))
    assert table["mean_prediction"].item() == 1e12 + 1
    assert table["between_model_variance"].item() == 1


def test_finite_constant_extremes_avoid_overflowing_the_mean() -> None:
    largest = np.finfo(float).max
    table = ensemble_regression_summary(np.full((4, 1), largest), np.full((4, 1), largest))
    assert table["mean_prediction"].item() == largest
    assert table["within_model_variance"].item() == largest
    assert table["between_model_variance"].item() == 0
    assert table["total_variance"].item() == largest
    assert table["total_std"].item() == np.sqrt(largest)


def test_opposite_extremes_keep_a_finite_standard_deviation_when_variance_overflows() -> None:
    largest = np.finfo(float).max
    table = ensemble_regression_summary(np.array([[-largest], [largest]]), np.zeros((2, 1)))
    assert table["mean_prediction"].item() == 0
    assert np.isposinf(table["between_model_variance"].item())
    assert np.isposinf(table["total_variance"].item())
    assert table["total_std"].item() == largest
    assert not table.isna().any().any()


def test_total_variance_overflow_does_not_force_standard_deviation_to_infinity() -> None:
    variance = np.finfo(float).max * 0.75
    mean = np.sqrt(variance)
    table = ensemble_regression_summary(np.array([[-mean], [mean]]), np.full((2, 1), variance))
    assert np.isposinf(table["total_variance"].item())
    assert np.isfinite(table["between_model_variance"].item())
    assert table["total_std"].item() == pytest.approx(np.sqrt(1.5) * np.sqrt(np.finfo(float).max))


def test_tiny_representable_variances_and_constant_subnormal_inputs_are_retained() -> None:
    table = ensemble_regression_summary(np.array([[-1e-150], [1e-150]]), np.full((2, 1), 1e-300))
    np.testing.assert_allclose(table["between_model_variance"], [1e-300], rtol=1e-14, atol=0)
    np.testing.assert_allclose(table["total_variance"], [2e-300], rtol=1e-14, atol=0)
    tiny = np.nextafter(0.0, 1.0)
    small = ensemble_regression_summary(np.full((2, 1), tiny), np.full((2, 1), tiny))
    assert small["mean_prediction"].item() == tiny and small["total_variance"].item() == tiny


@pytest.mark.parametrize(
    "invalid",
    [
        [],
        0,
        [0, 1],
        np.empty((0, 2)),
        np.empty((2, 0)),
        np.zeros((2, 1, 1)),
        [[0], [np.nan]],
        [[0], [np.inf]],
        [[0], [-np.inf]],
        [[0], [1j]],
        [["0"], ["1"]],
        [[0], [None]],
    ],
)
@pytest.mark.parametrize("argument", ["member_means", "member_variances"])
def test_malformed_component_matrices_are_rejected(invalid, argument) -> None:
    inputs = {"member_means": np.zeros((2, 1)), "member_variances": np.ones((2, 1))}
    inputs[argument] = invalid
    with pytest.raises(ValueError, match=argument):
        ensemble_regression_summary(**inputs)


@pytest.mark.parametrize("variance_shape", [(1, 3), (2, 1), (3, 2)])
def test_shape_mismatches_are_not_broadcast(variance_shape) -> None:
    with pytest.raises(ValueError, match="same shape"):
        ensemble_regression_summary(np.zeros((2, 3)), np.ones(variance_shape))


@pytest.mark.parametrize("variance", [-1, -np.nextafter(0.0, 1.0)])
def test_negative_variances_are_not_clipped(variance) -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        ensemble_regression_summary(np.zeros((2, 1)), np.array([[0], [variance]]))


def test_offline_example_exports_decomposition_and_untouched_test_predictions(
    tmp_path: Path,
) -> None:
    example = Path(__file__).resolve().parents[1] / "examples/run_ensemble_uncertainty.py"
    result = subprocess.run(
        [sys.executable, str(example), "--output-dir", str(tmp_path / "output with spaces")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert (report["seed"], report["train_rows"], report["test_rows"], report["members"]) == (
        42,
        80,
        1000,
        40,
    )
    test_table = pd.read_csv(report["test_table_path"])
    grid_table = pd.read_csv(report["grid_table_path"])
    assert len(test_table) == 1000 and len(grid_table) == 401
    assert grid_table["x"].iloc[0] == -5 and grid_table["x"].iloc[-1] == 5
    assert grid_table["x"].is_monotonic_increasing
    for table in [test_table, grid_table]:
        assert (table["within_model_variance"] > 0).all()
        assert (table["between_model_variance"] >= 0).all()
        np.testing.assert_allclose(
            table["total_variance"],
            table["within_model_variance"] + table["between_model_variance"],
        )
        np.testing.assert_allclose(table["total_std"] ** 2, table["total_variance"])
    mse = np.mean((test_table["y_true"] - test_table["mean_prediction"]) ** 2)
    assert report["test_mean_squared_error"] == pytest.approx(mse)
    assert report["test_mean_total_variance"] == pytest.approx(test_table["total_variance"].mean())
    inside = grid_table[grid_table["x"].abs() < 1]["between_model_variance"].mean()
    outside = grid_table[grid_table["x"].abs() > 4]["between_model_variance"].mean()
    assert outside > inside
    assert Path(report["plot_path"]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
