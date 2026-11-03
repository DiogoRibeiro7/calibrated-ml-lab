"""Check residual resampling, predictive noise, estimator isolation, and evaluation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from calibrated_ml_lab.models import bootstrap
from calibrated_ml_lab.models.bootstrap import ResidualBootstrapRegressor


@pytest.fixture
def data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(19)
    features = rng.normal(size=(40, 2))
    targets = 2 + features @ [1, -0.5] + rng.normal(size=40)
    return features, targets


def test_hand_calculated_residual_refits_and_prediction_noise(monkeypatch) -> None:
    # Mean fit = 3, residuals = [-3, 0, 3]. Bootstrap fits = 1 and 5.
    # Adding independent residual draws [-3, 3] gives predictive draws [-2, 8].
    samples = [np.array([-3, -3, 0]), np.array([0, 3, 3]), np.array([-3, 3])]

    class ScriptedRng:
        def choice(self, values, size):
            np.testing.assert_array_equal(values, [-3, 0, 3])
            sample = samples.pop(0)
            assert size == len(sample)
            return sample.copy()

    monkeypatch.setattr(bootstrap.np.random, "default_rng", lambda seed: ScriptedRng())
    model = ResidualBootstrapRegressor(DummyRegressor(), n_bootstrap=2, alpha=0.5).fit(
        np.arange(3)[:, None], np.array([0, 3, 6])
    )
    assert not samples
    assert [m.constant_.item() for m in model.bootstrap_estimators_] == [1, 5]
    np.testing.assert_array_equal(model.predict(np.zeros((2, 1))), [3, 3])
    np.testing.assert_array_equal(
        model.predict_interval(np.zeros((2, 1))), [[0.5, 0.5], [5.5, 5.5]]
    )


def test_future_noise_does_not_collapse_when_all_refitted_means_are_identical() -> None:
    model = ResidualBootstrapRegressor(
        DummyRegressor(strategy="constant", constant=0), n_bootstrap=100
    ).fit(np.zeros((4, 1)), np.array([-2, -1, 1, 2]))
    assert all(m.predict([[0]]).item() == 0 for m in model.bootstrap_estimators_)
    np.testing.assert_array_equal(model.predict_interval(np.zeros((2, 1))), [[-2, -2], [2, 2]])


def test_residuals_are_centered_even_for_a_biased_base_estimator() -> None:
    model = ResidualBootstrapRegressor(
        DummyRegressor(strategy="constant", constant=0), n_bootstrap=4
    ).fit(np.zeros((3, 1)), np.array([1, 2, 6]))
    np.testing.assert_array_equal(model.residuals_, [-2, -1, 3])
    assert set(model.noise_draws_).issubset({-2, -1, 3})


def test_base_predictions_match_linear_regression_and_input_estimator_is_untouched(data) -> None:
    features, targets = data
    template = LinearRegression(copy_X=False).fit(features.copy(), targets.copy())
    saved_coef, saved_intercept = template.coef_.copy(), template.intercept_
    saved_x, saved_y = features.copy(), targets.copy()
    expected = template.predict(features)
    model = ResidualBootstrapRegressor(template, n_bootstrap=8).fit(features, targets)
    np.testing.assert_allclose(model.predict(features), expected)
    assert model.estimator_ is not template
    assert len({id(m) for m in (model.estimator_, *model.bootstrap_estimators_)}) == 9
    np.testing.assert_array_equal(template.coef_, saved_coef)
    assert template.intercept_ == saved_intercept
    np.testing.assert_array_equal(features, saved_x)
    np.testing.assert_array_equal(targets, saved_y)


def test_repeatable_fits_calls_batches_and_row_order_without_global_rng_changes(data) -> None:
    features, targets = data
    global_before = np.random.get_state()
    model = ResidualBootstrapRegressor(n_bootstrap=12).fit(features, targets)
    expected = model.predict_interval(features)
    np.testing.assert_array_equal(model.predict_interval(features), expected)
    np.testing.assert_allclose(
        model.predict_interval(features[::-1]), np.asarray(expected)[:, ::-1]
    )
    batches = [model.predict_interval(features[:13]), model.predict_interval(features[13:])]
    np.testing.assert_allclose(np.concatenate(batches, axis=1), expected)
    np.testing.assert_array_equal(
        ResidualBootstrapRegressor(n_bootstrap=12)
        .fit(features, targets)
        .predict_interval(features),
        expected,
    )
    model.fit(features, targets)
    np.testing.assert_array_equal(model.predict_interval(features), expected)
    other = ResidualBootstrapRegressor(n_bootstrap=12, random_state=21).fit(features, targets)
    assert not np.array_equal(other.predict_interval(features), expected)
    global_after = np.random.get_state()
    for before, after in zip(global_before, global_after, strict=True):
        np.testing.assert_array_equal(before, after)


def test_nested_stochastic_estimators_are_seeded_without_mutating_the_template(data) -> None:
    template = make_pipeline(
        StandardScaler(), DecisionTreeRegressor(max_depth=2, splitter="random")
    )
    first = ResidualBootstrapRegressor(template, n_bootstrap=6, random_state=7).fit(*data)
    second = ResidualBootstrapRegressor(template, n_bootstrap=6, random_state=7).fit(*data)
    assert template[-1].random_state is None and not hasattr(template[-1], "tree_")
    assert all(m[-1].random_state == 7 for m in (first.estimator_, *first.bootstrap_estimators_))
    np.testing.assert_array_equal(first.predict_interval(data[0]), second.predict_interval(data[0]))


def test_constant_targets_and_zero_residuals_produce_singleton_intervals() -> None:
    model = ResidualBootstrapRegressor(n_bootstrap=4).fit(np.zeros((4, 1)), np.full(4, 7.0))
    np.testing.assert_array_equal(model.residuals_, np.zeros(4))
    np.testing.assert_array_equal(model.predict_interval(np.ones((2, 1))), np.full((2, 2), 7.0))


def test_interpolating_fit_exposes_zero_residual_limitation(data) -> None:
    model = ResidualBootstrapRegressor(DecisionTreeRegressor(), n_bootstrap=4).fit(*data)
    np.testing.assert_array_equal(model.residuals_, np.zeros(len(data[1])))
    lower, upper = model.predict_interval(data[0])
    np.testing.assert_array_equal(lower, upper)


def test_pandas_inputs_are_positional_and_numeric_objects_are_supported(data) -> None:
    features, targets = data
    model = ResidualBootstrapRegressor(n_bootstrap=4).fit(features, targets)
    expected = model.predict_interval(features)
    frame = pd.DataFrame(features, columns=["b", "a"], index=np.arange(40)[::-1])
    labels = pd.Series(targets, index=np.arange(40) + 100)
    model.fit(frame, labels)
    np.testing.assert_array_equal(model.predict_interval(frame), expected)
    model.fit(features.astype(object), targets.astype(object))
    np.testing.assert_array_equal(model.predict_interval(features.astype(object)), expected)


@pytest.mark.parametrize("method", ["predict", "predict_interval"])
def test_prediction_requires_a_successful_fit(method) -> None:
    with pytest.raises(RuntimeError, match="must be fitted"):
        getattr(ResidualBootstrapRegressor(), method)(np.zeros((1, 1)))


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
        [[None, 1]],
    ],
)
@pytest.mark.parametrize("operation", ["fit", "predict", "predict_interval"])
def test_invalid_features_preserve_the_previous_fit(features, operation, data) -> None:
    model = ResidualBootstrapRegressor(n_bootstrap=2).fit(*data)
    before = model.predict_interval(data[0])
    estimators = model.bootstrap_estimators_
    with pytest.raises(ValueError):
        if operation == "fit":
            model.fit(features, np.zeros(2))
        else:
            getattr(model, operation)(features)
    assert model.bootstrap_estimators_ is estimators
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
        [0, None],
    ],
)
def test_invalid_targets_leave_a_new_instance_unfitted(targets) -> None:
    model = ResidualBootstrapRegressor(n_bootstrap=2)
    with pytest.raises(ValueError):
        model.fit(np.zeros((2, 1)), targets)
    assert model.estimator_ is None and model.bootstrap_estimators_ == ()
    assert model.residuals_ is None and model.noise_draws_ is None


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
        False,
        "0.1",
        [0.1],
        0.1j,
        np.nextafter(0.0, 1.0),
    ],
)
def test_invalid_quantile_levels_are_rejected(alpha) -> None:
    with pytest.raises(ValueError, match="alpha"):
        ResidualBootstrapRegressor(alpha=alpha)


@pytest.mark.parametrize("repetitions", [0, 1, -1, 2.5, True, "3", [3]])
def test_invalid_bootstrap_counts_are_rejected(repetitions) -> None:
    with pytest.raises(ValueError, match="n_bootstrap"):
        ResidualBootstrapRegressor(n_bootstrap=repetitions)


@pytest.mark.parametrize("seed", [-1, 2**32, 2.5, True, None, "42", [42]])
def test_invalid_random_state_is_rejected(seed) -> None:
    with pytest.raises(ValueError, match="random_state"):
        ResidualBootstrapRegressor(random_state=seed)


@pytest.mark.parametrize(
    "estimator", [0, object(), LinearRegression, LogisticRegression(), StandardScaler()]
)
def test_invalid_estimator_is_rejected(estimator) -> None:
    with pytest.raises(ValueError, match="estimator"):
        ResidualBootstrapRegressor(estimator)


def test_clone_that_returns_original_estimator_is_rejected(data) -> None:
    class FrozenRegressor(RegressorMixin, BaseEstimator):
        def __sklearn_clone__(self):
            return self

    with pytest.raises(ValueError, match="distinct, unfitted"):
        ResidualBootstrapRegressor(FrozenRegressor(), n_bootstrap=2).fit(*data)


def test_failed_refit_and_changed_options_preserve_fitted_state(data) -> None:
    model = ResidualBootstrapRegressor(n_bootstrap=4).fit(*data)
    before = model.predict_interval(data[0])
    model.alpha = 0.4
    model.n_bootstrap = 6
    np.testing.assert_array_equal(model.predict_interval(data[0]), before)
    with pytest.raises(ValueError, match="at least two"):
        model.fit(np.zeros((1, 3)), np.zeros(1))
    assert (
        model.alpha_ == 0.1 and len(model.bootstrap_estimators_) == 4 and model.n_features_in_ == 2
    )
    model.alpha = 0
    with pytest.raises(ValueError, match="alpha"):
        model.fit(*data)
    np.testing.assert_array_equal(model.predict_interval(data[0]), before)
    model.alpha = 0.4
    model.fit(np.zeros((4, 3)), np.ones(4))
    assert (
        model.alpha_ == 0.4 and len(model.bootstrap_estimators_) == 6 and model.n_features_in_ == 3
    )
    for method in [model.predict, model.predict_interval]:
        with pytest.raises(ValueError, match="same number of columns"):
            method(np.zeros((2, 2)))


@pytest.mark.parametrize("already_fitted", [False, True])
def test_failure_during_later_bootstrap_fit_is_transactional(
    monkeypatch, data, already_fitted
) -> None:
    model = ResidualBootstrapRegressor(n_bootstrap=4)
    if already_fitted:
        model.fit(*data)
    before = model.__dict__.copy()
    real_clone, calls = bootstrap._clone_regressor, []

    def clone(template, seed):
        candidate = real_clone(template, seed)
        calls.append(candidate)
        if len(calls) == 3:

            def fail(*args):
                raise RuntimeError("later fit failed")

            monkeypatch.setattr(candidate, "fit", fail)
        return candidate

    monkeypatch.setattr(bootstrap, "_clone_regressor", clone)
    with pytest.raises(RuntimeError, match="later fit failed"):
        model.fit(*data)
    assert len(calls) == 3
    assert all(model.__dict__[key] is value for key, value in before.items())


@pytest.mark.parametrize("predictions", [[np.nan, 0], [np.inf, 0], [[1], [2]], [1], ["1", "2"]])
@pytest.mark.parametrize("method", ["predict", "predict_interval"])
def test_malformed_estimator_output_is_rejected(monkeypatch, data, predictions, method) -> None:
    model = ResidualBootstrapRegressor(n_bootstrap=2).fit(*data)
    estimator = model.estimator_ if method == "predict" else model.bootstrap_estimators_[1]
    monkeypatch.setattr(estimator, "predict", lambda values: np.asarray(predictions))
    with pytest.raises(RuntimeError, match="invalid predictions"):
        getattr(model, method)(np.zeros((2, 2)))


def test_nonfinite_new_model_output_does_not_replace_successful_fit(monkeypatch, data) -> None:
    model = ResidualBootstrapRegressor(n_bootstrap=2).fit(*data)
    before = model.estimator_, model.bootstrap_estimators_
    real_clone = bootstrap._clone_regressor

    def clone(template, seed):
        candidate = real_clone(template, seed)
        monkeypatch.setattr(candidate, "predict", lambda values: np.full(len(values), np.nan))
        return candidate

    monkeypatch.setattr(bootstrap, "_clone_regressor", clone)
    with pytest.raises(RuntimeError, match="invalid predictions"):
        model.fit(*data)
    assert model.estimator_ is before[0] and model.bootstrap_estimators_ is before[1]


def test_unsafe_residual_arithmetic_preserves_successful_fit(data) -> None:
    model = ResidualBootstrapRegressor(n_bootstrap=2).fit(*data)
    expected = model.predict_interval(data[0])
    model.estimator = DummyRegressor(strategy="constant", constant=-1e308)
    with pytest.raises(RuntimeError, match="fitting failed numerically"):
        model.fit(np.zeros((2, 1)), np.full(2, 1e308))
    np.testing.assert_array_equal(model.predict_interval(data[0]), expected)


def test_unsafe_prediction_addition_is_rejected(monkeypatch, data) -> None:
    model = ResidualBootstrapRegressor(n_bootstrap=2).fit(*data)
    model.noise_draws_[:] = 1e308
    for estimator in model.bootstrap_estimators_:
        monkeypatch.setattr(estimator, "predict", lambda values: np.full(len(values), 1e308))
    with pytest.raises(RuntimeError, match="interval calculation failed numerically"):
        model.predict_interval(data[0])


def test_offline_example_exports_overall_and_feature_bin_coverage(tmp_path: Path) -> None:
    example = Path(__file__).resolve().parents[1] / "examples/run_bootstrap_regression.py"
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
    assert report["seed"] == 42 and report["n_bootstrap"] == 500
    assert (report["train_rows"], report["test_rows"], report["nominal_coverage"]) == (
        1000,
        2000,
        0.9,
    )
    table = pd.read_csv(report["table_path"])
    assert len(table) == 8 and set(table["scenario"]) == set(report["scenarios"])
    for scenario, group in table.groupby("scenario"):
        assert group["count"].sum() == 2000
        np.testing.assert_allclose(group["coverage"], group["covered_count"] / group["count"])
        for name in ["coverage", "average_width", "empty_set_rate"]:
            assert np.average(group[name], weights=group["count"]) == pytest.approx(
                report["scenarios"][scenario][name]
            )
        assert ((group["min_width"] > 0) & (group["min_width"] <= group["average_width"])).all()
        assert (group["average_width"] <= group["max_width"]).all()
    heterogeneous = table[table["scenario"] == "feature_dependent_noise"]
    assert heterogeneous.iloc[0]["coverage"] > heterogeneous.iloc[-1]["coverage"]
