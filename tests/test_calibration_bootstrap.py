"""Check paired calibration bootstraps, fixed metric definitions, and offline exports."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from calibrated_ml_lab.calibration import bootstrap as bootstrap_module
from calibrated_ml_lab.calibration.bootstrap import (
    bootstrap_calibration_metrics,
    bootstrap_multiclass_calibration_metrics,
)
from calibrated_ml_lab.calibration.metrics import probabilistic_metrics
from calibrated_ml_lab.calibration.multiclass_metrics import multiclass_calibration_metrics

APIS = [bootstrap_calibration_metrics, bootstrap_multiclass_calibration_metrics]


@pytest.fixture(params=APIS, ids=["binary", "multiclass"])
def case(request):
    y = np.array([0, 0, 1, 1])
    p = np.array([0.0, 0.4, 0.5, 1.0])
    if request.param is bootstrap_multiclass_calibration_metrics:
        p = np.column_stack([1 - p, p, np.zeros(len(p))])
    return request.param, y, p


def test_exhaustive_two_row_bootstrap_matches_hand_calculation(monkeypatch) -> None:
    class ExhaustiveDraws:
        def __init__(self) -> None:
            self.draws = iter([[0, 0], [0, 1], [1, 0], [1, 1]])
            self.calls = 0

        def integers(self, low, high, size):
            assert (low, high, size) == (0, 2, 2)
            self.calls += 1
            return np.array(next(self.draws))

    rng = ExhaustiveDraws()
    monkeypatch.setattr(bootstrap_module.np.random, "default_rng", lambda seed: rng)
    result = bootstrap_calibration_metrics(
        np.array([0, 1]), np.array([0.2, 0.3]), n_bins=1, n_bootstrap=4, confidence_level=0.5
    ).set_index("metric")
    assert rng.calls == 4
    assert result.index.tolist() == ["brier_score", "log_loss", "ece", "mce"]
    assert result.loc["ece", "estimate"] == pytest.approx(0.25)
    assert result.loc["brier_score", "estimate"] == pytest.approx(0.265)
    expected_draws = {
        "brier_score": [0.04, 0.265, 0.265, 0.49],
        "log_loss": [-np.log(0.8), -np.log(0.24) / 2, -np.log(0.24) / 2, -np.log(0.3)],
        "ece": [0.2, 0.25, 0.25, 0.7],
        "mce": [0.2, 0.25, 0.25, 0.7],
    }
    for metric, draws in expected_draws.items():
        np.testing.assert_allclose(
            result.loc[metric, ["lower", "upper"]].astype(float),
            [
                0.25 * draws[0] + 0.75 * draws[1],
                0.75 * draws[2] + 0.25 * draws[3],
            ],
        )
        assert result.loc[metric, "standard_error"] == pytest.approx(np.std(draws, ddof=1))
    # Exhaustive draws contain both one-class samples; none may be discarded or stratified.
    assert result["n_bootstrap"].eq(4).all()


def test_seeded_multiclass_draws_match_existing_metrics_and_keep_missing_class_columns() -> None:
    y = np.array([0, 0, 1])
    p = np.array([[0.7, 0.3, 0], [0.5, 0.5, 0], [0.1, 0.6, 0.3]])
    result = bootstrap_multiclass_calibration_metrics(
        y, p, n_bins=2, n_bootstrap=13, confidence_level=0.8, random_state=7
    ).set_index("metric")
    rng = np.random.default_rng(7)
    draws = []
    for _ in range(13):
        chosen = rng.integers(0, len(y), size=len(y))
        draws.append(multiclass_calibration_metrics(y[chosen], p[chosen], n_bins=2))
    draws = pd.DataFrame(draws)
    estimates = multiclass_calibration_metrics(y, p, n_bins=2)
    assert result.index.tolist() == list(estimates)
    np.testing.assert_allclose(result["estimate"], list(estimates.values()))
    np.testing.assert_allclose(result["lower"], draws.quantile(0.1))
    np.testing.assert_allclose(result["upper"], draws.quantile(0.9))
    np.testing.assert_allclose(result["standard_error"], draws.std(ddof=1))


def test_repeatability_input_immutability_and_no_global_random_state_changes(case) -> None:
    evaluate, y, p = case
    before_y, before_p = y.copy(), p.copy()
    y.setflags(write=False)
    p.setflags(write=False)
    state = np.random.get_state()
    first = evaluate(y, p, n_bins=3, n_bootstrap=19, random_state=8)
    second = evaluate(y, p, n_bins=3, n_bootstrap=19, random_state=8)
    pd.testing.assert_frame_equal(first, second)
    after = np.random.get_state()
    assert state[0] == after[0] and state[2:] == after[2:]
    np.testing.assert_array_equal(state[1], after[1])
    np.testing.assert_array_equal(y, before_y)
    np.testing.assert_array_equal(p, before_p)
    changed = evaluate(y, p, n_bins=3, n_bootstrap=19, random_state=9)
    assert not first["standard_error"].equals(changed["standard_error"])


def test_output_schema_metadata_and_point_estimates(case) -> None:
    evaluate, y, p = case
    result = evaluate(y, p, n_bins=3, n_bootstrap=5, confidence_level=0.8)
    assert result.columns.tolist() == [
        "metric",
        "estimate",
        "lower",
        "upper",
        "standard_error",
        "confidence_level",
        "n_samples",
        "n_bootstrap",
        "n_bins",
        "method",
    ]
    if evaluate is bootstrap_calibration_metrics:
        expected = probabilistic_metrics(y, p, n_bins=3)
        del expected["roc_auc"]
    else:
        expected = multiclass_calibration_metrics(y, p, n_bins=3)
    assert result["metric"].tolist() == list(expected)
    np.testing.assert_allclose(result["estimate"], list(expected.values()))
    assert result["n_samples"].eq(4).all() and result["n_bootstrap"].eq(5).all()
    assert result["n_bins"].eq(3).all() and result["confidence_level"].eq(0.8).all()
    assert result["method"].eq("percentile").all()
    assert np.isfinite(result.select_dtypes(include="number")).all().all()
    assert (result["lower"] <= result["upper"]).all()
    assert result["standard_error"].ge(0).all()


def test_wider_confidence_level_expands_bounds_using_the_same_draws(case) -> None:
    evaluate, y, p = case
    narrow = evaluate(y, p, n_bootstrap=21, confidence_level=0.5)
    wide = evaluate(y, p, n_bootstrap=21, confidence_level=0.95)
    assert (wide["lower"] <= narrow["lower"]).all()
    assert (wide["upper"] >= narrow["upper"]).all()
    np.testing.assert_array_equal(wide["standard_error"], narrow["standard_error"])


@pytest.mark.parametrize("label", [0, 1])
def test_single_class_perfect_endpoints_have_zero_calibration_error(case, label) -> None:
    evaluate, _, p = case
    y = np.full(4, label)
    if p.ndim == 1:
        p = y.astype(float)
    else:
        p = np.eye(3)[y]
    result = evaluate(y, p, n_bins=6, n_bootstrap=5).set_index("metric")
    errors = result.drop(index="log_loss", errors="ignore")
    assert errors[["estimate", "lower", "upper", "standard_error"]].eq(0).all().all()
    if "log_loss" in result.index:
        assert 0 <= result.loc["log_loss", "estimate"] < 1e-10
        assert np.isfinite(result.loc["log_loss", "upper"])


def test_pandas_inputs_are_paired_by_position_and_numeric_objects_are_supported(case) -> None:
    evaluate, y, p = case
    expected = evaluate(y, p, n_bootstrap=5)
    probabilities = pd.Series(p, index=[5, 6, 7, 8]) if p.ndim == 1 else pd.DataFrame(p)
    actual = evaluate(
        pd.Series(y, index=[20, 30, 40, 50], dtype=object), probabilities, n_bootstrap=5
    )
    pd.testing.assert_frame_equal(expected, actual)
    pd.testing.assert_frame_equal(
        expected, evaluate(y.astype(object), p.astype(object), n_bootstrap=5)
    )


@pytest.mark.parametrize(
    "option,value",
    [
        ("n_bins", 0),
        ("n_bins", 1.5),
        ("n_bins", True),
        ("n_bootstrap", 0),
        ("n_bootstrap", 1),
        ("n_bootstrap", -2),
        ("n_bootstrap", 2.5),
        ("n_bootstrap", True),
        ("n_bootstrap", np.array(3)),
        ("random_state", -1),
        ("random_state", 2**32),
        ("random_state", 1.5),
        ("random_state", True),
        ("random_state", None),
        ("confidence_level", 0),
        ("confidence_level", 1),
        ("confidence_level", -0.1),
        ("confidence_level", 1.1),
        ("confidence_level", True),
        ("confidence_level", np.nan),
        ("confidence_level", np.inf),
        ("confidence_level", "0.9"),
        ("confidence_level", 0.9 + 1j),
        ("confidence_level", [0.9]),
        ("confidence_level", 1e-300),
        ("confidence_level", np.nextafter(1.0, 0.0)),
    ],
)
def test_invalid_options_raise_clear_errors(case, option, value) -> None:
    evaluate, y, p = case
    options = {"n_bootstrap": 2, option: value}
    with pytest.raises(ValueError, match=option):
        evaluate(y, p, **options)


def test_numpy_scalar_options_and_seed_endpoints_are_supported(case) -> None:
    evaluate, y, p = case
    for seed in [np.int64(0), np.int64(2**32 - 1)]:
        result = evaluate(
            y,
            p,
            n_bins=np.int64(2),
            n_bootstrap=np.int64(2),
            confidence_level=np.float64(0.9),
            random_state=seed,
        )
        assert result["n_bootstrap"].eq(2).all()


@pytest.mark.parametrize("rows", [0, 1])
def test_empty_and_single_row_samples_are_rejected(case, rows) -> None:
    evaluate, y, p = case
    with pytest.raises(ValueError):
        evaluate(y[:rows], p[:rows], n_bootstrap=2)


@pytest.mark.parametrize("value", [np.nan, np.inf, 1j, "0.5", -0.2, 1.1])
def test_invalid_probability_values_are_rejected(case, value) -> None:
    evaluate, y, p = case
    p = p.astype(object)
    p.flat[0] = value
    with pytest.raises(ValueError):
        evaluate(y, p, n_bootstrap=2)


@pytest.mark.parametrize("value", [np.nan, np.inf, 1j, "0", -1, 0.5, 3])
def test_invalid_labels_are_rejected(case, value) -> None:
    evaluate, y, p = case
    y = y.astype(object)
    y[0] = value
    with pytest.raises(ValueError):
        evaluate(y, p, n_bootstrap=2)


def test_mismatched_lengths_and_label_matrix_are_rejected(case) -> None:
    evaluate, y, p = case
    with pytest.raises(ValueError):
        evaluate(y[:-1], p, n_bootstrap=2)
    with pytest.raises(ValueError):
        evaluate(y[:, None], p, n_bootstrap=2)


def test_multiclass_requires_a_probability_matrix_with_unit_row_sums() -> None:
    for p in [np.array([0.2, 0.8]), np.ones((2, 1)), np.ones((2, 2)), np.ones((2, 2, 1))]:
        with pytest.raises(ValueError):
            bootstrap_multiclass_calibration_metrics(np.array([0, 1]), p, n_bootstrap=2)
    with pytest.raises(ValueError):
        bootstrap_calibration_metrics(np.array([0, 1]), np.eye(2), n_bootstrap=2)


def test_offline_example_exports_frozen_test_predictions_and_reproducible_intervals(
    tmp_path,
) -> None:
    example = Path(__file__).resolve().parents[1] / "examples/run_calibration_bootstrap.py"
    result = subprocess.run(
        [
            sys.executable,
            str(example),
            "--output-dir",
            str(tmp_path / "output with spaces"),
            "--n-bootstrap",
            "7",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["seed"] == 42 and report["n_bootstrap"] == 7
    assert (
        report["train_rows_per_task"],
        report["calibration_rows_per_task"],
        report["test_rows_per_task"],
    ) == (1200, 600, 600)
    table = pd.read_csv(report["table_path"])
    assert len(table) == 14
    for task, evaluate in zip(["binary", "multiclass"], APIS, strict=True):
        predictions = pd.read_csv(report["prediction_paths"][task])
        assert len(predictions) == 600
        for predictor in ["raw", "calibrated"]:
            p = (
                predictions[f"{predictor}_probability"].to_numpy()
                if task == "binary"
                else predictions[
                    [f"{predictor}_probability_{index}" for index in range(3)]
                ].to_numpy()
            )
            expected = evaluate(predictions["y_true"].to_numpy(), p, n_bootstrap=7)
            actual = table[(table["task"] == task) & (table["predictor"] == predictor)]
            pd.testing.assert_frame_equal(
                expected,
                actual.drop(columns=["task", "predictor"]).reset_index(drop=True),
            )
