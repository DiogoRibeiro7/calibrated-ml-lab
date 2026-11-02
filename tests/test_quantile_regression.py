"""Validate quantile interval fitting, crossed bounds, and independent evaluation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from calibrated_ml_lab.models import quantile
from calibrated_ml_lab.models.quantile import QuantileIntervalRegressor


@pytest.fixture
def data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(19)
    features = rng.normal(size=(100, 2))
    targets = 2 * features[:, 0] + rng.normal(size=100) * (0.2 + np.abs(features[:, 1]))
    return features, targets


def test_constant_features_recover_empirical_quantiles() -> None:
    features = np.zeros((21, 1))
    targets = np.arange(21, dtype=float)
    model = QuantileIntervalRegressor(alpha=0.2, n_estimators=3)
    assert model.fit(features, targets) is model
    assert model.quantiles_ == (0.1, 0.9) and model.alpha_ == 0.2
    assert model.n_features_in_ == 1
    lower, upper = model.predict_quantiles(np.zeros((2, 1)))
    np.testing.assert_allclose(lower, [2, 2])
    np.testing.assert_allclose(upper, [18, 18])
    np.testing.assert_array_equal(model.predict_interval(np.zeros((2, 1))), [lower, upper])


def test_predictions_are_reproducible_and_inputs_are_not_changed(data) -> None:
    features, targets = data
    before_x, before_y = features.copy(), targets.copy()
    model = QuantileIntervalRegressor(n_estimators=10).fit(features, targets)
    expected = model.predict_quantiles(features)
    assert all(value.shape == (100,) and np.isfinite(value).all() for value in expected)
    np.testing.assert_array_equal(
        QuantileIntervalRegressor(n_estimators=10)
        .fit(features, targets)
        .predict_quantiles(features),
        expected,
    )
    model.fit(features, targets)
    np.testing.assert_array_equal(model.predict_quantiles(features), expected)
    np.testing.assert_array_equal(
        model.predict_quantiles(features[:1]), np.asarray(expected)[:, :1]
    )
    np.testing.assert_array_equal(
        model.predict_quantiles(features[::-1]), np.asarray(expected)[:, ::-1]
    )
    np.testing.assert_array_equal(features, before_x)
    np.testing.assert_array_equal(targets, before_y)


def test_numeric_objects_and_pandas_inputs_use_positional_feature_columns(data) -> None:
    features, targets = data
    model = QuantileIntervalRegressor(n_estimators=3).fit(features, targets)
    expected = model.predict_quantiles(features)
    frame = pd.DataFrame(features, columns=["second", "first"], index=np.arange(100)[::-1])
    np.testing.assert_array_equal(model.predict_quantiles(frame), expected)
    model.fit(features.astype(object), targets.astype(object))
    np.testing.assert_array_equal(model.predict_quantiles(features.astype(object)), expected)


def test_constant_targets_single_row_and_refitting_with_different_features() -> None:
    model = QuantileIntervalRegressor(n_estimators=2).fit(np.zeros((1, 1)), np.array([7.0]))
    np.testing.assert_array_equal(model.predict_interval(np.ones((2, 1))), [[7, 7], [7, 7]])
    model.alpha = 0.4
    # Options change future fits; current fitted quantile identities remain available.
    assert model.alpha_ == 0.1 and model.quantiles_ == (0.05, 0.95)
    model.fit(np.zeros((4, 3)), np.ones(4))
    assert model.n_features_in_ == 3 and model.alpha_ == 0.4 and model.quantiles_ == (0.2, 0.8)
    np.testing.assert_array_equal(model.predict_interval(np.zeros((2, 3))), np.ones((2, 2)))
    with pytest.raises(ValueError, match="same number of columns"):
        model.predict_interval(np.zeros((1, 1)))


def test_crossed_quantiles_remain_visible_and_interval_bounds_are_ordered(
    monkeypatch, data
) -> None:
    model = QuantileIntervalRegressor(n_estimators=2).fit(*data)
    raw_lower, raw_upper = np.array([3.0, 1, 2]), np.array([1.0, 4, 2])
    monkeypatch.setattr(model.lower_model_, "predict", lambda values: raw_lower)
    monkeypatch.setattr(model.upper_model_, "predict", lambda values: raw_upper)
    values = np.zeros((3, 2))
    np.testing.assert_array_equal(model.predict_quantiles(values), [[3, 1, 2], [1, 4, 2]])
    lower, upper = model.predict_interval(values)
    np.testing.assert_array_equal(lower, [1, 1, 2])
    np.testing.assert_array_equal(upper, [3, 4, 2])
    np.testing.assert_array_equal(raw_lower, [3, 1, 2])
    np.testing.assert_array_equal(raw_upper, [1, 4, 2])
    assert np.mean(raw_lower > raw_upper) == 1 / 3


@pytest.mark.parametrize("method", ["predict_quantiles", "predict_interval"])
def test_prediction_requires_a_successful_fit(method: str) -> None:
    with pytest.raises(RuntimeError, match="must be fitted"):
        getattr(QuantileIntervalRegressor(), method)(np.zeros((1, 1)))


@pytest.mark.parametrize(
    "features",
    [
        [],
        1,
        [1, 2],
        np.empty((0, 2)),
        np.empty((2, 0)),
        [[[1, 2]]],
        [[np.nan, 1]],
        [[np.inf, 1]],
        [[-np.inf, 1]],
        [[1j, 2]],
        [["1", "2"]],
        np.array([[1, "2"]], dtype=object),
        [[None, 1]],
        [[1e100, 0]],
    ],
)
@pytest.mark.parametrize("operation", ["fit", "predict_quantiles", "predict_interval"])
def test_invalid_features_preserve_a_previous_fit(features, operation, data) -> None:
    model = QuantileIntervalRegressor(n_estimators=2).fit(*data)
    before = model.predict_interval(data[0])
    models = (model.lower_model_, model.upper_model_)
    with pytest.raises(ValueError):
        if operation == "fit":
            model.fit(np.asarray(features), np.array([0]))
        else:
            getattr(model, operation)(np.asarray(features))
    assert (model.lower_model_, model.upper_model_) == models
    assert model.n_features_in_ == 2 and model.alpha_ == 0.1
    np.testing.assert_array_equal(model.predict_interval(data[0]), before)


@pytest.mark.parametrize(
    "targets",
    [
        [],
        1,
        [[1], [2]],
        [1],
        [0, np.nan],
        [0, np.inf],
        [0, 1j],
        ["1", "2"],
        np.array([1, "2"], dtype=object),
    ],
)
def test_invalid_targets_leave_new_instance_unfitted(targets) -> None:
    model = QuantileIntervalRegressor(n_estimators=2)
    with pytest.raises(ValueError):
        model.fit(np.zeros((2, 1)), np.asarray(targets))
    assert model.lower_model_ is None and model.upper_model_ is None
    assert model.alpha_ is None and model.quantiles_ is None and model.n_features_in_ is None


@pytest.mark.parametrize(
    "alpha",
    [
        0,
        1,
        -0.1,
        1.1,
        np.nan,
        np.inf,
        True,
        np.array(True, dtype=object),
        "0.1",
        [0.1],
        0.1j,
        np.nextafter(0.0, 1.0),
    ],
)
def test_invalid_or_unrepresentable_quantile_levels_are_rejected(alpha) -> None:
    with pytest.raises(ValueError, match="alpha"):
        QuantileIntervalRegressor(alpha=alpha)


@pytest.mark.parametrize("option", ["n_estimators", "max_depth", "min_samples_leaf"])
@pytest.mark.parametrize("value", [0, -1, 2.5, True, "3", [3]])
def test_invalid_tree_options_are_rejected(option: str, value) -> None:
    with pytest.raises(ValueError, match=option):
        QuantileIntervalRegressor(**{option: value})


@pytest.mark.parametrize("learning_rate", [0, -1, np.nan, np.inf, True, "0.1", [0.1]])
def test_invalid_learning_rate_is_rejected(learning_rate) -> None:
    with pytest.raises(ValueError, match="learning_rate"):
        QuantileIntervalRegressor(learning_rate=learning_rate)


@pytest.mark.parametrize("seed", [-1, 2**32, 2.5, True, None, "42", [42]])
def test_invalid_random_state_is_rejected(seed) -> None:
    with pytest.raises(ValueError, match="random_state"):
        QuantileIntervalRegressor(random_state=seed)


def test_failed_refit_validation_keeps_predictions_and_metadata(data) -> None:
    model = QuantileIntervalRegressor(n_estimators=2).fit(*data)
    expected = model.predict_quantiles(data[0])
    with pytest.raises(ValueError):
        model.fit(np.ones((2, 3)), np.array([1, np.nan]))
    model.alpha = 0
    with pytest.raises(ValueError, match="alpha"):
        model.fit(*data)
    assert model.n_features_in_ == 2 and model.alpha_ == 0.1 and model.quantiles_ == (0.05, 0.95)
    np.testing.assert_array_equal(model.predict_quantiles(data[0]), expected)


@pytest.mark.parametrize("already_fitted", [False, True])
def test_failure_of_second_model_keeps_both_previous_models(
    monkeypatch, data, already_fitted
) -> None:
    model = QuantileIntervalRegressor(n_estimators=2)
    expected = None
    if already_fitted:
        model.fit(*data)
        expected = model.predict_quantiles(data[0])
    previous = (
        model.lower_model_,
        model.upper_model_,
        model.n_features_in_,
        model.quantiles_,
        model.alpha_,
    )
    real_regressor = quantile.GradientBoostingRegressor
    fitted_levels = []

    def new_regressor(**kwargs):
        candidate = real_regressor(**kwargs)
        real_fit = candidate.fit

        def fit(values, targets):
            fitted_levels.append(candidate.alpha)
            if candidate.alpha > 0.5:
                raise RuntimeError("upper model failed")
            return real_fit(values, targets)

        monkeypatch.setattr(candidate, "fit", fit)
        return candidate

    monkeypatch.setattr(quantile, "GradientBoostingRegressor", new_regressor)
    model.alpha = 0.2
    with pytest.raises(RuntimeError, match="upper model failed"):
        model.fit(np.zeros((5, 3)), np.arange(5))
    assert fitted_levels == [0.1, 0.9]
    assert (
        model.lower_model_,
        model.upper_model_,
        model.n_features_in_,
        model.quantiles_,
        model.alpha_,
    ) == previous
    if already_fitted:
        np.testing.assert_array_equal(model.predict_quantiles(data[0]), expected)


@pytest.mark.parametrize("predictions", [[np.nan, 0], [np.inf, 0], [[1], [2]], [1], ["1", "2"]])
def test_invalid_model_predictions_are_rejected(monkeypatch, data, predictions) -> None:
    model = QuantileIntervalRegressor(n_estimators=2).fit(*data)
    monkeypatch.setattr(model.upper_model_, "predict", lambda values: np.asarray(predictions))
    with pytest.raises(RuntimeError, match="invalid predictions"):
        model.predict_interval(np.zeros((2, 2)))


def test_nonfinite_new_model_output_is_not_committed(monkeypatch, data) -> None:
    model = QuantileIntervalRegressor(n_estimators=2).fit(*data)
    before = model.lower_model_, model.upper_model_
    real_regressor = quantile.GradientBoostingRegressor

    def new_regressor(**kwargs):
        candidate = real_regressor(**kwargs)
        monkeypatch.setattr(candidate, "predict", lambda values: np.full(len(values), np.nan))
        return candidate

    monkeypatch.setattr(quantile, "GradientBoostingRegressor", new_regressor)
    with pytest.raises(RuntimeError, match="invalid predictions"):
        model.fit(*data)
    assert (model.lower_model_, model.upper_model_) == before


def test_unsafe_target_arithmetic_preserves_previous_fit(data) -> None:
    model = QuantileIntervalRegressor(n_estimators=2).fit(*data)
    before = model.predict_quantiles(data[0])
    with pytest.raises(RuntimeError, match="failed numerically"):
        model.fit(np.array([[0.0], [1.0]]), np.array([-1e308, 1e308]))
    assert model.n_features_in_ == 2 and model.quantiles_ == (0.05, 0.95)
    np.testing.assert_array_equal(model.predict_quantiles(data[0]), before)


def test_offline_quantile_example_reports_test_coverage_and_adaptive_widths(tmp_path: Path) -> None:
    example = Path(__file__).resolve().parents[1] / "examples/run_quantile_regression.py"
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
    assert report["train_rows"] == 2000 and report["test_rows"] == 1000
    assert report["nominal_coverage"] == 0.9 and report["quantile_levels"] == [0.05, 0.95]
    assert 0 <= report["test_crossing_rate"] <= 1
    assert report["test_lower_pinball_loss"] >= 0 and report["test_upper_pinball_loss"] >= 0
    intervals = report["test_intervals"]
    assert 0 <= intervals["coverage"] <= 1
    assert 0 <= intervals["min_width"] <= intervals["average_width"] <= intervals["max_width"]
    groups = report["test_by_noise_region"]
    assert set(groups) == {"lower_noise", "higher_noise"}
    assert groups["higher_noise"]["average_width"] > groups["lower_noise"]["average_width"]
    assert sum(group["count"] for group in groups.values()) == report["test_rows"]
    for metric in ("coverage", "average_width"):
        weighted = (
            sum(group["count"] * group[metric] for group in groups.values()) / report["test_rows"]
        )
        assert weighted == pytest.approx(intervals[metric])
