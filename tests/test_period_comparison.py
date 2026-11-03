"""Check period comparisons, distinct diagnostic roles, and reproducible offline reports."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from calibrated_ml_lab.calibration.metrics import calibration_bins
from calibrated_ml_lab.calibration.multiclass_metrics import confidence_calibration_bins
from calibrated_ml_lab.monitoring.calibration import (
    compare_binary_calibration,
    compare_multiclass_calibration,
)


@pytest.fixture(params=["binary", "multiclass"])
def case(request):
    y = np.array([0, 0, 1, 1])
    p = np.array([0.0, 0.4, 0.7, 1.0])
    if request.param == "binary":
        return compare_binary_calibration, y, p
    return compare_multiclass_calibration, y, np.column_stack([1 - p, p, np.zeros(len(p))])


def test_binary_periods_match_hand_calculation_and_distinguish_calibration_from_quality() -> None:
    table = compare_binary_calibration(
        np.array([0, 1]),
        np.array([0.25, 0.75]),
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.6, 0.8, 0.9]),
        n_bins=2,
    )
    assert table.columns.tolist() == [
        "metric",
        "category",
        "reference",
        "production",
        "change",
        "reference_count",
        "production_count",
        "n_bins",
    ]
    assert table["metric"].tolist() == [
        "brier_score",
        "log_loss",
        "ece",
        "mce",
        "roc_auc",
        "positive_rate",
        "mean_probability",
    ]
    np.testing.assert_allclose(table["reference"], [0.0625, -np.log(0.75), 0.25, 0.25, 1, 0.5, 0.5])
    np.testing.assert_allclose(
        table["production"], [0.105, -np.log([0.9, 0.4, 0.8, 0.9]).mean(), 0.1, 0.1, 1, 0.5, 0.6]
    )
    np.testing.assert_allclose(table["change"], table["production"] - table["reference"])
    assert table["reference_count"].eq(2).all() and table["production_count"].eq(4).all()
    assert table["n_bins"].eq(2).all()
    indexed = table.set_index("metric")
    assert indexed.loc["ece", "change"] < 0 < indexed.loc["brier_score", "change"]
    assert indexed.loc["ece", "category"] == "calibration"
    assert (
        indexed.loc["brier_score", "category"]
        == indexed.loc["log_loss", "category"]
        == "probabilistic_quality"
    )
    assert indexed.loc["roc_auc", "category"] == "discrimination"
    assert indexed.loc["positive_rate", "category"] == "sample_summary"


def test_multiclass_periods_retain_missing_classes_and_distinct_calibration_definitions() -> None:
    table = compare_multiclass_calibration(
        np.array([0, 0, 2, 1]),
        np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6], [0.6, 0.3, 0.1]]),
        np.zeros(3),
        np.tile([0.6, 0.3, 0.1], (3, 1)),
        n_bins=2,
    )
    assert table["metric"].tolist() == [
        "confidence_ece",
        "top_label_ece",
        "classwise_ece",
        "accuracy",
        "mean_confidence",
        "class_0_rate",
        "class_1_rate",
        "class_2_rate",
    ]
    np.testing.assert_allclose(table["reference"], [0.2, 0.4, 0.25, 0.5, 0.7, 0.5, 0.25, 0.25])
    np.testing.assert_allclose(table["production"], [0.4, 0.4, 0.8 / 3, 1, 0.6, 1, 0, 0])
    np.testing.assert_allclose(table["change"], table["production"] - table["reference"])
    assert table["reference_count"].eq(4).all() and table["production_count"].eq(3).all()
    assert table.set_index("metric").loc["accuracy", "category"] == "predictive_performance"


def test_multiclass_probability_ties_select_first_class() -> None:
    p = np.full((2, 2), 0.5)
    table = compare_multiclass_calibration(np.zeros(2), p, np.array([0, 1]), p).set_index("metric")
    assert table.loc["accuracy", "reference"] == 1
    assert table.loc["accuracy", "production"] == 0.5
    assert table.loc["confidence_ece", "reference"] == 0.5
    assert table.loc["confidence_ece", "production"] == 0


@pytest.mark.parametrize("single_class_period", ["reference", "production"])
def test_single_class_auc_and_its_change_remain_undefined(single_class_period) -> None:
    y, p = np.array([0, 1]), np.array([0.0, 1])
    inputs = [y, p, y, p]
    position = 0 if single_class_period == "reference" else 2
    inputs[position : position + 2] = [np.zeros(1), np.zeros(1)]
    table = compare_binary_calibration(*inputs).set_index("metric")
    assert np.isnan(table.loc["roc_auc", single_class_period])
    assert np.isnan(table.loc["roc_auc", "change"])
    other = "production" if single_class_period == "reference" else "reference"
    assert table.loc["roc_auc", other] == 1
    assert np.isfinite(table.drop(index="roc_auc").select_dtypes(include="number")).all().all()


def test_identical_periods_have_zero_changes_and_stable_metadata(case) -> None:
    compare, y, p = case
    table = compare(y, p, y, p, n_bins=np.int64(5))
    assert table["change"].eq(0).all()
    assert table["n_bins"].eq(5).all()
    pd.testing.assert_frame_equal(table, compare(y, p, y, p, n_bins=5))


def test_swapping_periods_reverses_changes_and_counts(case) -> None:
    compare, y, p = case
    forward = compare(y, p, y[:3], p[:3])
    backward = compare(y[:3], p[:3], y, p)
    np.testing.assert_allclose(forward["change"], -backward["change"])
    np.testing.assert_allclose(forward["reference"], backward["production"])
    assert forward["reference_count"].equals(backward["production_count"])


def test_periods_are_not_paired_or_pooled_and_row_order_does_not_matter(case) -> None:
    compare, y, p = case
    expected = compare(y, p, y[:3], p[:3], n_bins=3)
    shuffled = compare(y[::-1], p[::-1], y[[2, 0, 1]], p[[2, 0, 1]], n_bins=3)
    pd.testing.assert_frame_equal(expected, shuffled)
    repeated = compare(np.repeat(y, 2), np.repeat(p, 2, axis=0), y[:3], p[:3], n_bins=3)
    assert repeated["reference_count"].eq(8).all()
    pd.testing.assert_frame_equal(
        expected.drop(columns="reference_count"), repeated.drop(columns="reference_count")
    )


def test_read_only_and_pandas_inputs_are_unchanged_and_aligned_by_position(case) -> None:
    compare, y, p = case
    before_y, before_p = y.copy(), p.copy()
    y.setflags(write=False)
    p.setflags(write=False)
    expected = compare(y, p, y[:3], p[:3])
    frame = pd.Series(p, index=[6, 7, 8, 9]) if p.ndim == 1 else pd.DataFrame(p, index=[6, 7, 8, 9])
    actual = compare(pd.Series(y, index=[20, 30, 40, 50], dtype=object), frame, y[:3], p[:3])
    pd.testing.assert_frame_equal(expected, actual)
    np.testing.assert_array_equal(y, before_y)
    np.testing.assert_array_equal(p, before_p)


@pytest.mark.parametrize("period", ["reference", "production"])
@pytest.mark.parametrize("bad_probability", [np.nan, np.inf, -0.1, 1.1, 1j, "0.5"])
def test_invalid_probabilities_identify_the_affected_period(case, period, bad_probability) -> None:
    compare, y, p = case
    bad = p.astype(object)
    bad.flat[0] = bad_probability
    with pytest.raises(ValueError, match=period):
        if period == "reference":
            compare(y, bad, y, p)
        else:
            compare(y, p, y, bad)


@pytest.mark.parametrize("period", ["reference", "production"])
@pytest.mark.parametrize("bad_label", [np.nan, np.inf, -1, 3, 0.5, 1j, "0"])
def test_invalid_labels_identify_the_affected_period(case, period, bad_label) -> None:
    compare, y, p = case
    bad = y.astype(object)
    bad[0] = bad_label
    with pytest.raises(ValueError, match=period):
        if period == "reference":
            compare(bad, p, y, p)
        else:
            compare(y, p, bad, p)


@pytest.mark.parametrize("period", ["reference", "production"])
@pytest.mark.parametrize(
    "malformation", ["empty", "label_matrix", "mismatch", "probability_tensor"]
)
def test_missing_or_malformed_periods_are_not_silently_dropped(case, period, malformation) -> None:
    compare, y, p = case
    bad_y, bad_p = y, p
    if malformation == "empty":
        bad_y, bad_p = y[:0], p[:0]
    elif malformation == "label_matrix":
        bad_y = y[:, None]
    elif malformation == "mismatch":
        bad_y = y[:-1]
    else:
        bad_p = p[..., None, None]
    with pytest.raises(ValueError, match=period):
        if period == "reference":
            compare(bad_y, bad_p, y, p)
        else:
            compare(y, p, bad_y, bad_p)


@pytest.mark.parametrize("n_bins", [0, -1, True, 2.5, np.nan, np.array(2)])
def test_bin_count_must_be_a_positive_integer(case, n_bins) -> None:
    compare, y, p = case
    with pytest.raises(ValueError, match="n_bins"):
        compare(y, p, y, p, n_bins=n_bins)


def test_multiclass_periods_require_the_same_number_of_columns() -> None:
    with pytest.raises(ValueError, match="same number of class columns"):
        compare_multiclass_calibration(np.array([0, 1]), np.eye(2), np.array([0, 1, 2]), np.eye(3))


def test_offline_example_exports_recomputable_comparisons_and_bins(tmp_path) -> None:
    example = Path(__file__).resolve().parents[1] / "examples/run_period_comparison.py"
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
    assert report["seed"] == 42 and report["n_bins"] == 10
    assert report["train_rows_per_task"] == 2000
    assert report["reference_rows_per_task"] == 1200 and report["production_rows_per_task"] == 800
    comparisons = pd.read_csv(report["comparison_path"])
    bins = pd.read_csv(report["bins_path"])
    assert len(comparisons) == 15 and len(bins) == 40
    for task, compare in [
        ("binary", compare_binary_calibration),
        ("multiclass", compare_multiclass_calibration),
    ]:
        predictions = pd.read_csv(report["prediction_paths"][task])
        inputs = []
        for period, count in [("reference", 1200), ("production", 800)]:
            rows = predictions[predictions["period"] == period]
            assert len(rows) == count
            y = rows["y_true"].to_numpy()
            p = rows.filter(like="probability_").to_numpy()
            np.testing.assert_allclose(p.sum(axis=1), 1)
            if task == "binary":
                p = p[:, 1]
                expected_bins = calibration_bins(y, p)
            else:
                expected_bins = confidence_calibration_bins(y, p).rename(
                    columns={
                        "mean_confidence": "mean_predicted_probability",
                        "accuracy": "observed_event_rate",
                    }
                )
            inputs.extend([y, p])
            actual_bins = bins[(bins["task"] == task) & (bins["period"] == period)]
            pd.testing.assert_frame_equal(
                expected_bins, actual_bins.drop(columns=["task", "period"]).reset_index(drop=True)
            )
        actual = (
            comparisons[comparisons["task"] == task].drop(columns="task").reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(compare(*inputs), actual)
    assert Path(report["plot_path"]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
