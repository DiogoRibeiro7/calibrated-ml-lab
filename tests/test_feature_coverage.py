"""Check exhaustive feature partitions, interval semantics, and offline diagnostics."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from calibrated_ml_lab.conformal.quantile import conformalized_quantile_metrics
from calibrated_ml_lab.uncertainty.intervals import interval_coverage_by_feature

STATISTICS = [
    "coverage",
    "average_width",
    "median_width",
    "min_width",
    "max_width",
    "empty_set_rate",
    "coverage_gap",
]


@pytest.fixture
def sample() -> tuple[np.ndarray, ...]:
    return (
        np.zeros(6),
        np.array([0, 1, -1, 1, -np.inf, -2]),
        np.array([0, 2, 1, 0, np.inf, -1]),
        np.array([-2, -1, 0, 0.5, 1, 2]),
        np.array([-2, -1, 0, 1, 2, 3, 4]),
    )


def test_bin_summaries_match_hand_calculations_and_keep_empty_bins(sample) -> None:
    table = interval_coverage_by_feature(*sample, nominal_coverage=0.9)
    assert table.columns.tolist() == [
        "bin_index",
        "bin_lower",
        "bin_upper",
        "right_closed",
        "count",
        "fraction",
        "covered_count",
        "nominal_coverage",
        *STATISTICS,
    ]
    assert table["bin_index"].tolist() == list(range(6))
    assert table["bin_lower"].tolist() == [-2, -1, 0, 1, 2, 3]
    assert table["bin_upper"].tolist() == [-1, 0, 1, 2, 3, 4]
    assert table["right_closed"].tolist() == [False] * 5 + [True]
    assert table["count"].tolist() == [1, 1, 2, 1, 1, 0]
    assert table["covered_count"].tolist() == [1, 0, 1, 1, 0, 0]
    np.testing.assert_allclose(table["fraction"], [1 / 6, 1 / 6, 2 / 6, 1 / 6, 1 / 6, 0])
    assert table["nominal_coverage"].eq(0.9).all()
    np.testing.assert_allclose(table.loc[:4, "coverage"], [1, 0, 0.5, 1, 0])
    np.testing.assert_allclose(table.loc[:4, "coverage_gap"], [0.1, -0.9, -0.4, 0.1, -0.9])
    for name in ["average_width", "median_width"]:
        np.testing.assert_allclose(table.loc[:4, name], [0, 1, 1, np.inf, 1])
    np.testing.assert_allclose(table.loc[:4, "min_width"], [0, 1, 0, np.inf, 1])
    np.testing.assert_allclose(table.loc[:4, "max_width"], [0, 1, 2, np.inf, 1])
    np.testing.assert_allclose(table.loc[:4, "empty_set_rate"], [0, 0, 0.5, 0, 0])
    assert table.loc[5, STATISTICS].isna().all()


def test_finite_outer_endpoints_and_internal_ties_are_counted_once() -> None:
    features = np.array([-1, 0, 0, 1, 2])
    table = interval_coverage_by_feature(
        np.zeros(5),
        np.zeros(5),
        np.ones(5),
        features,
        np.array([-1, 0, 1, 2]),
        nominal_coverage=0.8,
    )
    assert table["count"].tolist() == table["covered_count"].tolist() == [1, 2, 2]
    assert table["fraction"].sum() == 1 and table["count"].sum() == len(features)


def test_infinite_tails_and_adjacent_float_values_use_exact_boundaries() -> None:
    tiny = np.nextafter(0.0, 1.0)
    features = np.array([-1e300, -tiny, 0, tiny, 1e300])
    table = interval_coverage_by_feature(
        np.zeros(5),
        np.zeros(5),
        np.ones(5),
        features,
        np.array([-np.inf, 0, np.inf]),
        nominal_coverage=0.8,
    )
    assert table["count"].tolist() == [2, 3]


def test_good_overall_coverage_can_hide_an_uncovered_feature_region() -> None:
    y, lo, hi = np.array([0] * 9 + [2]), np.zeros(10), np.ones(10)
    features = np.array([0] * 9 + [1])
    assert conformalized_quantile_metrics(y, lo, hi)["coverage"] == 0.9
    table = interval_coverage_by_feature(
        y,
        lo,
        hi,
        features,
        np.array([-np.inf, 0.5, np.inf]),
        nominal_coverage=0.9,
    )
    assert table["count"].tolist() == [9, 1]
    assert table["coverage"].tolist() == [1, 0]
    np.testing.assert_allclose(table["coverage_gap"], [0.1, -0.9])


def test_partition_reconstructs_overall_counts_and_weighted_statistics() -> None:
    y = np.zeros(5)
    lo, hi = np.array([-1, 1, -2, 2, 0]), np.array([1, 0, 1, 3, 0])
    table = interval_coverage_by_feature(
        y,
        lo,
        hi,
        np.arange(5),
        np.array([-1, 0, 2, 4, 5, 6]),
        nominal_coverage=0.9,
    )
    populated = table[table["count"] > 0]
    overall = conformalized_quantile_metrics(y, lo, hi)
    assert table["count"].sum() == 5
    assert table["covered_count"].sum() / 5 == overall["coverage"]
    for name in ["coverage", "average_width", "empty_set_rate"]:
        assert np.average(populated[name], weights=populated["count"]) == pytest.approx(
            overall[name]
        )
    assert populated["min_width"].min() == overall["min_width"]
    assert populated["max_width"].max() == overall["max_width"]


def test_one_bin_matches_global_summary_and_handles_extreme_values() -> None:
    largest = np.finfo(float).max
    y, lo, hi = np.zeros(2), np.zeros(2), np.full(2, largest)
    table = interval_coverage_by_feature(
        y,
        lo,
        hi,
        np.array([-largest, largest]),
        np.array([-largest, largest]),
        nominal_coverage=0.9,
    )
    assert table["count"].tolist() == [2]
    assert table.loc[0, STATISTICS[:-1]].to_dict() == conformalized_quantile_metrics(y, lo, hi)
    assert table["average_width"].item() == largest


def test_constant_features_and_numeric_objects_keep_unoccupied_bins() -> None:
    table = interval_coverage_by_feature(
        np.array([0], dtype=object),
        np.array([0], dtype=object),
        np.array([0], dtype=object),
        np.array([1], dtype=object),
        np.array([0, 1, 2, 3], dtype=object),
        nominal_coverage=0.7,
    )
    assert table["count"].tolist() == [0, 1, 0]
    assert table["covered_count"].tolist() == [0, 1, 0]
    assert table.loc[[0, 2], STATISTICS].isna().all().all()
    assert table.loc[1, "coverage"] == 1 and table.loc[1, "average_width"] == 0


def test_positionally_aligned_pandas_inputs_and_row_permutation_preserve_results(sample) -> None:
    saved = [values.copy() for values in sample]
    expected = interval_coverage_by_feature(*sample, nominal_coverage=0.9)
    indexed = [
        pd.Series(values, index=np.arange(len(values))[::-1] + i * 10)
        for i, values in enumerate(sample)
    ]
    pd.testing.assert_frame_equal(
        interval_coverage_by_feature(*indexed, nominal_coverage=0.9),
        expected,
    )
    permuted = [values[::-1] for values in sample[:4]] + [sample[4]]
    pd.testing.assert_frame_equal(
        interval_coverage_by_feature(*permuted, nominal_coverage=0.9),
        expected,
    )
    for original, values in zip(saved, sample, strict=True):
        np.testing.assert_array_equal(original, values)


@pytest.mark.parametrize(
    "edges",
    [
        [],
        [0],
        0,
        [[0, 1]],
        [1, 0],
        [0, 0, 1],
        [0, 1, 0.5],
        [0, np.nan, 1],
        [0, np.inf, np.inf],
        [-np.inf, -np.inf, 1],
        [0, np.inf, 2],
        [0, "1"],
        [0, 1j],
        [0, None, 1],
        [0, 2**53, 2**53 + 1],
    ],
)
def test_malformed_or_collapsed_edges_are_rejected(edges) -> None:
    with pytest.raises(ValueError, match="bin_edges"):
        interval_coverage_by_feature(
            np.zeros(2),
            np.zeros(2),
            np.ones(2),
            np.array([0, 1]),
            edges,
            nominal_coverage=0.9,
        )


@pytest.mark.parametrize("features", [[-1, 0], [0, 2]])
def test_out_of_range_rows_are_rejected_instead_of_silently_dropped(features) -> None:
    with pytest.raises(ValueError, match="contain every feature value"):
        interval_coverage_by_feature(
            np.zeros(2),
            np.zeros(2),
            np.ones(2),
            features,
            np.array([0, 1]),
            nominal_coverage=0.9,
        )


@pytest.mark.parametrize(
    "features",
    [
        [],
        0,
        [[0], [1]],
        [0],
        [0, 1, 2],
        [0, np.nan],
        [0, np.inf],
        [-np.inf, 0],
        [0, 1j],
        ["0", "1"],
        np.array([0, None], dtype=object),
    ],
)
def test_invalid_features_and_misaligned_rows_are_rejected(features) -> None:
    with pytest.raises(ValueError, match="feature_values"):
        interval_coverage_by_feature(
            np.zeros(2),
            np.zeros(2),
            np.ones(2),
            features,
            np.array([-np.inf, np.inf]),
            nominal_coverage=0.9,
        )


@pytest.mark.parametrize(
    "y,lo,hi",
    [
        ([], [], []),
        ([0], [0, 0], [1, 1]),
        ([0, 1], [0], [1, 1]),
        ([0, 1], [0, 0], [1]),
        ([[0], [1]], [0, 0], [1, 1]),
        ([0, np.inf], [0, 0], [1, 1]),
        ([0, np.nan], [0, 0], [1, 1]),
        ([0, 1], [0, np.nan], [1, 1]),
        ([0, 1], [0, 0], [1, np.nan]),
        ([0, 1], [0, np.inf], [1, np.inf]),
        ([0, 1], [0, -np.inf], [1, -np.inf]),
        ([0, 1j], [0, 0], [1, 1]),
        ([0, 1], [0, 1j], [1, 1]),
        (["0", "1"], [0, 0], [1, 1]),
        ([0, 1], [0, 0], ["1", "1"]),
    ],
)
def test_invalid_interval_rows_are_rejected(y, lo, hi) -> None:
    with pytest.raises(ValueError):
        interval_coverage_by_feature(
            y,
            lo,
            hi,
            np.array([0, 1]),
            np.array([-np.inf, 0.5, np.inf]),
            nominal_coverage=0.9,
        )


@pytest.mark.parametrize(
    "nominal",
    [
        0,
        1,
        -0.1,
        1.1,
        np.nan,
        np.inf,
        -np.inf,
        True,
        False,
        "0.9",
        1j,
        None,
        [0.9],
    ],
)
def test_invalid_nominal_coverage_is_rejected(nominal) -> None:
    with pytest.raises(ValueError, match="nominal coverage"):
        interval_coverage_by_feature(
            np.zeros(1),
            np.zeros(1),
            np.ones(1),
            np.zeros(1),
            np.array([0, 1]),
            nominal_coverage=nominal,
        )


def test_offline_example_exports_partitioned_diagnostics(tmp_path: Path) -> None:
    example = Path(__file__).resolve().parents[1] / "examples/run_feature_coverage.py"
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
    assert report["seed"] == 42 and report["nominal_coverage"] == 0.9
    assert (report["train_rows"], report["calibration_rows"], report["test_rows"]) == (
        2000,
        1000,
        1000,
    )
    table = pd.read_csv(report["table_path"])
    assert len(table) == 8 and table["feature"].eq("x").all()
    assert set(table["method"]) == set(report["overall"]) == {"Quantile", "Conformalized quantile"}
    for method, group in table.groupby("method", sort=False):
        assert group["bin_lower"].tolist() == [-np.inf, -1, 0, 1]
        assert group["bin_upper"].tolist() == [-1, 0, 1, np.inf]
        assert group["right_closed"].tolist() == [False, False, False, True]
        assert group["count"].sum() == 1000 and group["fraction"].sum() == pytest.approx(1)
        np.testing.assert_allclose(group["coverage"], group["covered_count"] / group["count"])
        np.testing.assert_allclose(group["coverage_gap"], group["coverage"] - 0.9, atol=1e-15)
        for name in ["coverage", "average_width", "empty_set_rate"]:
            assert np.average(group[name], weights=group["count"]) == pytest.approx(
                report["overall"][method][name]
            )
        assert ((group["min_width"] >= 0) & (group["min_width"] <= group["average_width"])).all()
        assert (group["average_width"] <= group["max_width"]).all()
    first, second = (group for _, group in table.groupby("method", sort=False))
    assert first["count"].tolist() == second["count"].tolist()
