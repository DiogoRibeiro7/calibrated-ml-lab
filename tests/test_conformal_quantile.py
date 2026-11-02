"""Check exact CQR ranks, signed corrections, empty sets, and held-out evaluation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from calibrated_ml_lab.conformal.quantile import (
    conformalized_quantile_intervals,
    conformalized_quantile_metrics,
    conformalized_quantile_threshold,
)
from calibrated_ml_lab.conformal.regression import (
    conformal_residual_quantile,
    regression_interval_metrics,
)


@pytest.mark.parametrize("alpha,expected", [(0.2, -1), (0.5, -2), (0.8, -4)])
def test_exact_order_statistic_retains_negative_scores(alpha: float, expected: float) -> None:
    y, lower, upper = np.arange(1.0, 5.0), np.zeros(4), np.full(4, 10.0)
    assert conformalized_quantile_threshold(y, lower, upper, alpha) == expected


def test_positive_zero_and_tied_scores_include_boundaries() -> None:
    lower, upper = np.zeros(4), np.full(4, 10.0)
    assert conformalized_quantile_threshold(np.array([-2, 1, 5, 12]), lower, upper, 0.5) == 2
    assert conformalized_quantile_threshold(np.array([0, 10, 5, 5]), lower, upper, 0.5) == 0
    lo, hi = conformalized_quantile_intervals(np.array([0.0]), np.array([10.0]), 2)
    assert lo.item() == -2 and hi.item() == 12
    summary = conformalized_quantile_metrics(
        np.array([-2.0, 12]), np.repeat(lo, 2), np.repeat(hi, 2)
    )
    assert summary["coverage"] == 1 and summary["average_width"] == 14


@pytest.mark.parametrize("alpha", [0.1, 0.25, 0.5, 0.8])
def test_point_base_intervals_recover_absolute_residual_conformal(alpha: float) -> None:
    y, predictions = np.array([1.0, -3, 2, 5]), np.array([0.0, -2, 1.5, 3])
    assert conformalized_quantile_threshold(
        y, predictions, predictions, alpha
    ) == conformal_residual_quantile(y, predictions, alpha)


def test_shrinkage_preserves_empty_and_singleton_sets() -> None:
    lower, upper = conformalized_quantile_intervals(
        np.array([0.0, 0, 0]), np.array([10.0, 4, 2]), -2
    )
    np.testing.assert_array_equal(lower, [2, 2, 2])
    np.testing.assert_array_equal(upper, [8, 2, 0])
    # The third response would be covered if its empty interval were incorrectly sorted.
    summary = conformalized_quantile_metrics(np.array([5.0, 2, 1]), lower, upper)
    assert summary == {
        "coverage": 2 / 3,
        "average_width": 2,
        "median_width": 0,
        "min_width": 0,
        "max_width": 6,
        "empty_set_rate": 1 / 3,
    }
    with pytest.raises(ValueError):
        regression_interval_metrics(np.array([5.0, 2, 1]), lower, upper)


def test_all_empty_sets_remain_in_coverage_and_width_denominators() -> None:
    summary = conformalized_quantile_metrics(
        np.array([1.0, 0]), np.array([2.0, np.inf]), np.array([0.0, -np.inf])
    )
    assert summary == {
        "coverage": 0,
        "average_width": 0,
        "median_width": 0,
        "min_width": 0,
        "max_width": 0,
        "empty_set_rate": 1,
    }


def test_tiny_calibration_returns_full_real_line_and_infinite_widths() -> None:
    qhat = conformalized_quantile_threshold(np.zeros(2), -np.ones(2), np.ones(2))
    assert qhat == np.inf
    lo, hi = conformalized_quantile_intervals(np.zeros(2), np.ones(2), qhat)
    np.testing.assert_array_equal(lo, [-np.inf, -np.inf])
    np.testing.assert_array_equal(hi, [np.inf, np.inf])
    assert conformalized_quantile_metrics(np.array([-1e300, 1e300]), lo, hi) == {
        "coverage": 1,
        "average_width": np.inf,
        "median_width": np.inf,
        "min_width": np.inf,
        "max_width": np.inf,
        "empty_set_rate": 0,
    }


@pytest.mark.parametrize("size", [1, 2, 9, 48])
def test_finite_rank_boundary_and_just_below_it(size: int) -> None:
    alpha = 1 / (size + 1)
    y = np.arange(1.0, size + 1.0)
    lo, hi = np.zeros(size), np.full(size, 2 * size + 1.0)
    assert conformalized_quantile_threshold(y, lo, hi, alpha) == -1
    assert conformalized_quantile_threshold(y, lo, hi, np.nextafter(alpha, 0.0)) == np.inf


@pytest.mark.parametrize("alpha,expected", [(0.1, 1.0), (0.2, 0.8), (0.5, 0.6), (0.8, 0.2)])
def test_marginal_coverage_enumerates_every_held_out_rank(alpha: float, expected: float) -> None:
    # Five fixed observations with distinct scores -4, -2, -1, 2, 1.
    y = np.array([0.0, 0, 1, 4, 3])
    lo, hi = np.array([-4.0, -2, 0, 1, -1]), np.array([4.0, 3, 2, 2, 2])
    covered = []
    for i in range(len(y)):
        calibration = np.arange(len(y)) != i
        qhat = conformalized_quantile_threshold(
            y[calibration], lo[calibration], hi[calibration], alpha
        )
        lower, upper = conformalized_quantile_intervals(lo[i : i + 1], hi[i : i + 1], qhat)
        actual = conformalized_quantile_metrics(y[i : i + 1], lower, upper)["coverage"]
        assert actual == (max(lo[i] - y[i], y[i] - hi[i]) <= qhat)
        covered.append(actual)
    assert np.mean(covered) == expected


def test_metrics_match_ordinary_interval_summary_without_empty_sets() -> None:
    y, lo, hi = np.array([-1.0, 0, 8]), np.array([-2.0, 1, 2]), np.array([2.0, 4, 8])
    assert conformalized_quantile_metrics(y, lo, hi) == {
        **regression_interval_metrics(y, lo, hi),
        "empty_set_rate": 0,
    }


def test_large_finite_width_mean_and_median_do_not_overflow() -> None:
    summary = conformalized_quantile_metrics(np.zeros(2), np.zeros(2), np.full(2, 1e308))
    assert summary["average_width"] == summary["median_width"] == 1e308


def test_uniform_subnormal_widths_are_preserved() -> None:
    smallest = np.nextafter(0.0, 1.0)
    summary = conformalized_quantile_metrics(np.zeros(2), np.zeros(2), np.full(2, smallest))
    assert summary["average_width"] == summary["median_width"] == smallest


def test_finite_bound_overflow_preserves_unbounded_or_empty_set_meaning() -> None:
    largest = np.finfo(float).max
    lo, hi = conformalized_quantile_intervals(np.array([-largest]), np.array([largest]), largest)
    assert lo.item() == -np.inf and hi.item() == np.inf
    lo, hi = conformalized_quantile_intervals(np.array([1e308]), np.array([1.1e308]), -1e308)
    assert lo.item() == np.inf and np.isfinite(hi.item())
    assert conformalized_quantile_metrics(np.array([1e308]), lo, hi)["empty_set_rate"] == 1


def test_nonrepresentable_scores_are_rejected_even_when_rank_would_be_infinite() -> None:
    largest = np.finfo(float).max
    with pytest.raises(ValueError, match="finite"):
        conformalized_quantile_threshold(
            np.array([largest]), np.array([-largest]), np.array([-largest])
        )
    # A negatively overflowing intermediate is harmless if the final maximum is finite.
    assert (
        conformalized_quantile_threshold(
            np.array([largest]), np.array([-largest]), np.array([largest]), 0.5
        )
        == 0
    )


def test_numeric_objects_row_order_and_inputs_are_preserved() -> None:
    y, lo, hi = np.array([1.0, 2, 3, 4]), np.zeros(4), np.full(4, 10.0)
    originals = y.copy(), lo.copy(), hi.copy()
    qhat = conformalized_quantile_threshold(y, lo, hi, 0.5)
    assert conformalized_quantile_threshold(y[::-1], lo[::-1], hi[::-1], 0.5) == qhat
    assert (
        conformalized_quantile_threshold(
            y.astype(object), lo.astype(object), hi.astype(object), 0.5
        )
        == qhat
    )
    corrected = conformalized_quantile_intervals(
        lo.astype(object), hi.astype(object), np.array(qhat)
    )
    conformalized_quantile_metrics(y.astype(object), *corrected)
    for actual, original in zip((y, lo, hi), originals, strict=True):
        np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize(
    "lo,hi",
    [
        ([], []),
        (0, 1),
        ([[0]], [[1]]),
        ([0], [1, 2]),
        ([2], [1]),
        ([np.nan], [1]),
        ([0], [np.nan]),
        ([-np.inf], [1]),
        ([0], [np.inf]),
        ([1j], [2]),
        (["0"], ["1"]),
        (np.array(["0"], dtype=object), [1]),
    ],
)
def test_invalid_base_bounds_are_rejected_for_calibration_and_application(lo, hi) -> None:
    with pytest.raises(ValueError):
        conformalized_quantile_threshold(np.array([0.0]), np.asarray(lo), np.asarray(hi))
    with pytest.raises(ValueError):
        conformalized_quantile_intervals(np.asarray(lo), np.asarray(hi), np.inf)


@pytest.mark.parametrize("y", [[], 0, [[0]], [0, 1], [np.nan], [np.inf], [1j], ["0"]])
def test_invalid_calibration_and_evaluation_targets_are_rejected(y) -> None:
    with pytest.raises(ValueError):
        conformalized_quantile_threshold(np.asarray(y), np.array([0.0]), np.array([1.0]))
    with pytest.raises(ValueError):
        conformalized_quantile_metrics(np.asarray(y), np.array([0.0]), np.array([1.0]))


@pytest.mark.parametrize("alpha", [0, 1, -0.1, np.nan, np.inf, "0.1", [0.1], 1j])
def test_invalid_alpha_is_rejected(alpha) -> None:
    with pytest.raises(ValueError):
        conformalized_quantile_threshold(np.zeros(2), np.zeros(2), np.ones(2), alpha)


@pytest.mark.parametrize("qhat", [np.nan, -np.inf, "1", [1], 1j, np.array("1", dtype=object)])
def test_invalid_signed_threshold_is_rejected(qhat) -> None:
    with pytest.raises(ValueError):
        conformalized_quantile_intervals(np.zeros(2), np.ones(2), qhat)


@pytest.mark.parametrize(
    "lo,hi",
    [
        ([np.inf], [np.inf]),
        ([-np.inf], [-np.inf]),
        ([np.nan], [0]),
        ([0], [np.nan]),
        ([], []),
        ([[0]], [[1]]),
        ([0], [1, 2]),
        (["0"], ["1"]),
    ],
)
def test_invalid_corrected_intervals_are_rejected_by_metrics(lo, hi) -> None:
    with pytest.raises(ValueError):
        conformalized_quantile_metrics(np.zeros(1), np.asarray(lo), np.asarray(hi))


def test_offline_cqr_example_reports_separate_splits_and_grouped_coverage(tmp_path: Path) -> None:
    example = Path(__file__).resolve().parents[1] / "examples/run_conformalized_quantile.py"
    result = subprocess.run(
        [sys.executable, str(example)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert (report["train_rows"], report["calibration_rows"], report["test_rows"]) == (
        2000,
        1000,
        1000,
    )
    assert report["alpha"] == 0.1 and report["nominal_coverage"] == 0.9
    assert np.isfinite(report["conformal_threshold"])
    for kind in ("base", "conformal"):
        overall = report[f"test_{kind}_intervals"]
        assert 0 <= overall["coverage"] <= 1 - overall["empty_set_rate"]
        assert 0 <= overall["min_width"] <= overall["average_width"] <= overall["max_width"]
        groups = report["test_by_noise_region"]
        assert sum(group["count"] for group in groups.values()) == report["test_rows"]
        for metric in ("coverage", "average_width", "empty_set_rate"):
            weighted = (
                sum(group["count"] * group[kind][metric] for group in groups.values())
                / report["test_rows"]
            )
            assert weighted == pytest.approx(overall[metric])
    qhat = report["conformal_threshold"]
    base, corrected = report["test_base_intervals"], report["test_conformal_intervals"]
    if qhat >= 0:
        assert corrected["coverage"] >= base["coverage"]
        assert corrected["average_width"] == pytest.approx(base["average_width"] + 2 * qhat)
    else:
        assert corrected["coverage"] <= base["coverage"]
        assert corrected["average_width"] <= base["average_width"]
