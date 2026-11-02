"""Check interval calibration tables, plotted data, and exported offline diagnostics."""

from __future__ import annotations

import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from calibrated_ml_lab.uncertainty.intervals import interval_calibration_table
from calibrated_ml_lab.visualization.plots import plot_interval_calibration


@pytest.fixture
def intervals() -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """Unsorted levels with empty, singleton, and unbounded sets and decreasing coverage."""
    return {
        0.9: (np.full(4, -np.inf), np.full(4, np.inf)),
        0.5: (np.array([0.0, 2, 0, 4]), np.array([0.0, 1, 2, 2])),
        0.8: (np.zeros(4), np.zeros(4)),
    }


def test_table_sorts_levels_and_matches_hand_calculations(intervals) -> None:
    table = interval_calibration_table(np.arange(4.0), intervals)
    assert table["nominal_coverage"].tolist() == [0.5, 0.8, 0.9]
    assert table["count"].tolist() == [4, 4, 4]
    assert table["covered_count"].tolist() == [2, 1, 4]
    assert table["coverage"].tolist() == [0.5, 0.25, 1]
    np.testing.assert_allclose(table["coverage_gap"], [0, -0.55, 0.1])
    assert table["empty_set_rate"].tolist() == [0.5, 0, 0]
    assert table["average_width"].tolist() == [0.5, 0, np.inf]
    assert table["median_width"].tolist() == [0, 0, np.inf]
    assert table["min_width"].tolist() == [0, 0, np.inf]
    assert table["max_width"].tolist() == [2, 0, np.inf]


def test_one_observation_and_numeric_objects_are_supported() -> None:
    table = interval_calibration_table(np.array([2], dtype=object), {0.75: ([2], [2])})
    assert table["count"].item() == table["covered_count"].item() == 1
    assert table["coverage"].item() == 1 and table["average_width"].item() == 0
    assert table["empty_set_rate"].item() == 0


def test_rows_are_positional_and_inputs_are_not_modified(intervals) -> None:
    targets = np.arange(4.0)
    saved = {level: (lo.copy(), hi.copy()) for level, (lo, hi) in intervals.items()}
    expected = interval_calibration_table(targets, intervals)
    indexed = pd.Series(targets, index=[40, 30, 20, 10])
    pd.testing.assert_frame_equal(interval_calibration_table(indexed, intervals), expected)
    reversed_rows = {level: (lo[::-1], hi[::-1]) for level, (lo, hi) in intervals.items()}
    pd.testing.assert_frame_equal(
        interval_calibration_table(targets[::-1], reversed_rows), expected
    )
    assert list(intervals) == [0.9, 0.5, 0.8]
    np.testing.assert_array_equal(targets, np.arange(4.0))
    for level, (lo, hi) in intervals.items():
        np.testing.assert_array_equal(lo, saved[level][0])
        np.testing.assert_array_equal(hi, saved[level][1])


@pytest.mark.parametrize(
    "nominal", [0, 1, -0.1, 1.1, np.nan, np.inf, -np.inf, True, False, "0.9", 1j, None]
)
def test_invalid_nominal_levels_are_rejected(nominal) -> None:
    with pytest.raises(ValueError, match="nominal coverage"):
        interval_calibration_table(np.zeros(2), {nominal: (np.zeros(2), np.ones(2))})


def test_distinct_keys_that_round_to_the_same_level_are_rejected() -> None:
    predictions = {Fraction(4, 5): (np.zeros(1), np.ones(1)), 0.8: (np.zeros(1), np.ones(1))}
    assert len(predictions) == 2
    with pytest.raises(ValueError, match="remain distinct"):
        interval_calibration_table(np.zeros(1), predictions)


@pytest.mark.parametrize("predictions", [None, [], {}, np.ones((2, 2))])
def test_empty_and_nonmapping_predictions_are_rejected(predictions) -> None:
    with pytest.raises(ValueError, match="nonempty mapping"):
        interval_calibration_table(np.zeros(1), predictions)


@pytest.mark.parametrize(
    "bounds", [None, [], [np.zeros(2)], [0, 1, 2], "01", {"lower": [0], "upper": [1]}]
)
def test_malformed_bound_pairs_are_rejected(bounds) -> None:
    with pytest.raises(ValueError, match="pair"):
        interval_calibration_table(np.zeros(2), {0.9: bounds})


@pytest.mark.parametrize(
    "lo,hi",
    [
        ([], []),
        ([0], [1]),
        ([[0], [0]], [[1], [1]]),
        ([0, np.nan], [1, 1]),
        ([0, 0], [1, np.nan]),
        ([0, 1j], [1, 1]),
        (["0", "0"], [1, 1]),
        ([np.inf, 0], [np.inf, 1]),
        ([-np.inf, 0], [-np.inf, 1]),
    ],
)
def test_invalid_bounds_and_row_alignment_are_rejected(lo, hi) -> None:
    with pytest.raises(ValueError):
        interval_calibration_table(np.zeros(2), {0.9: (np.asarray(lo), np.asarray(hi))})


@pytest.mark.parametrize(
    "targets", [[], 0, [[0], [1]], [0, np.nan], [0, np.inf], [1j, 1], ["0", "1"]]
)
def test_invalid_targets_are_rejected(targets) -> None:
    with pytest.raises(ValueError):
        interval_calibration_table(np.asarray(targets), {0.9: (np.zeros(2), np.ones(2))})


def test_plot_uses_actual_sorted_rates_including_empty_and_unbounded_sets(
    tmp_path, monkeypatch, intervals
) -> None:
    saved_figures = []
    savefig = Figure.savefig

    def capture(figure, path, **kwargs):
        saved_figures.append(figure)
        return savefig(figure, path, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    before = plt.get_fignums()
    output = tmp_path / "nested" / "calibration.png"
    plot_interval_calibration(
        np.arange(4.0), {"_raw": intervals, "second": {0.75: (np.zeros(4), np.ones(4))}}, output
    )
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert plt.get_fignums() == before
    figure = saved_figures[0]
    axis = figure.axes[0]
    np.testing.assert_array_equal(axis.lines[0].get_xdata(), [0, 1])
    np.testing.assert_array_equal(axis.lines[0].get_ydata(), [0, 1])
    np.testing.assert_array_equal(axis.lines[1].get_xdata(), [0.5, 0.8, 0.9])
    np.testing.assert_array_equal(axis.lines[1].get_ydata(), [0.5, 0.25, 1])
    np.testing.assert_array_equal(axis.lines[2].get_xdata(), [0.75])
    np.testing.assert_array_equal(axis.lines[2].get_ydata(), [0.5])
    assert axis.get_xlabel() == "Nominal coverage" and axis.get_ylabel() == "Empirical coverage"
    assert [text.get_text() for text in axis.get_legend().get_texts()] == [
        "Nominal = empirical",
        "_raw",
        "second",
    ]
    assert figure.texts[0].get_text() == "Evaluation rows: 4. Empty sets count as uncovered."


@pytest.mark.parametrize("methods", [None, [], {}, {1: {}}, {"": {}}, {" \t": {}}, {"valid": {}}])
def test_invalid_plot_methods_do_not_create_outputs(tmp_path, methods) -> None:
    output = tmp_path / "new" / "calibration.png"
    before = plt.get_fignums()
    with pytest.raises(ValueError):
        plot_interval_calibration(np.zeros(1), methods, output)
    assert not output.parent.exists() and plt.get_fignums() == before


def test_later_invalid_method_does_not_overwrite_an_existing_figure(tmp_path) -> None:
    output = tmp_path / "existing.png"
    output.write_bytes(b"existing figure")
    before = plt.get_fignums()
    methods = {"first": {0.9: (np.zeros(1), np.ones(1))}, "last": {0.8: ([np.nan], [1])}}
    with pytest.raises(ValueError):
        plot_interval_calibration(np.zeros(1), methods, output)
    assert output.read_bytes() == b"existing figure" and plt.get_fignums() == before


def test_save_failure_closes_only_the_new_figure(tmp_path, monkeypatch) -> None:
    unrelated = plt.figure()
    before = plt.get_fignums()

    def fail(*args, **kwargs):
        raise OSError("save failed")

    monkeypatch.setattr(Figure, "savefig", fail)
    try:
        with pytest.raises(OSError, match="save failed"):
            plot_interval_calibration(
                np.zeros(1), {"method": {0.9: (np.zeros(1), np.ones(1))}}, tmp_path / "plot.png"
            )
        assert plt.get_fignums() == before
    finally:
        plt.close(unrelated)


def test_single_level_plot_supports_vector_export(tmp_path) -> None:
    output = tmp_path / "one_level.svg"
    plot_interval_calibration(np.zeros(1), {"One level": {0.9: (np.zeros(1), np.ones(1))}}, output)
    assert "<svg" in output.read_text(encoding="utf-8")


def test_offline_example_exports_matching_calibration_table_and_plot(tmp_path: Path) -> None:
    example = Path(__file__).resolve().parents[1] / "examples/run_interval_calibration.py"
    destination = tmp_path / "output with spaces"
    result = subprocess.run(
        [sys.executable, str(example), "--output-dir", str(destination)],
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
    assert report["nominal_levels"] == [0.5, 0.7, 0.8, 0.9, 0.95]
    assert Path(report["plot_path"]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    table = pd.read_csv(report["table_path"])
    assert len(table) == 10 and table["count"].eq(1000).all()
    assert set(table["method"]) == {"Quantile", "Conformalized quantile"}
    for _, group in table.groupby("method"):
        assert group["nominal_coverage"].tolist() == report["nominal_levels"]
    np.testing.assert_allclose(table["coverage"], table["covered_count"] / table["count"])
    np.testing.assert_allclose(
        table["coverage_gap"], table["coverage"] - table["nominal_coverage"], atol=1e-15
    )
    assert ((table["coverage"] >= 0) & (table["coverage"] <= 1 - table["empty_set_rate"])).all()
    assert ((table["min_width"] >= 0) & (table["min_width"] <= table["average_width"])).all()
    assert (table["average_width"] <= table["max_width"]).all()
