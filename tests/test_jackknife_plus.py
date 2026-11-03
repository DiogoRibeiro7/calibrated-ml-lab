"""Check finite-sample ranks, genuine leave-one-out fits, and prediction contracts."""

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
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from calibrated_ml_lab.conformal import jackknife
from calibrated_ml_lab.conformal.jackknife import JackknifePlusRegressor, jackknife_plus_intervals


@pytest.fixture
def data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    features = rng.normal(size=(12, 2))
    return features, 1 + features @ [2, -1] + rng.normal(size=12)


def test_hand_calculated_order_statistics_keep_model_residual_pairing() -> None:
    predictions = np.array([[0, 10], [2, 8], [4, 6], [6, 4], [8, 2]])
    residuals = np.arange(5)
    # n=5, alpha=1/3: lower rank 2 and upper rank 4 (one-based).
    # Lower candidates: [0,1,2,3,4] and [10,7,4,1,-2].
    # Upper candidates: [0,3,6,9,12] and [10,9,8,7,6].
    expected = np.array([[1, 1], [9, 9]])
    np.testing.assert_array_equal(jackknife_plus_intervals(predictions, residuals, 1 / 3), expected)
    np.testing.assert_array_equal(
        jackknife_plus_intervals(predictions[::-1], residuals[::-1], 1 / 3), expected
    )
    assert not np.array_equal(
        jackknife_plus_intervals(predictions, residuals[::-1], 1 / 3), expected
    )
    np.testing.assert_array_equal(predictions[:, 0], [0, 2, 4, 6, 8])
    np.testing.assert_array_equal(residuals, np.arange(5))


@pytest.mark.parametrize("n,rank", [(4, 1), (4, 2), (9, 1), (48, 1), (48, 20), (99, 29)])
def test_rank_boundaries_and_neighboring_floats(n, rank) -> None:
    boundary = rank / (n + 1)
    predictions, residuals = np.arange(n)[:, None], np.zeros(n)
    for alpha, expected_rank in [
        (np.nextafter(boundary, 0), rank - 1),
        (boundary, rank),
        (np.nextafter(boundary, 1), rank),
    ]:
        lo, hi = jackknife_plus_intervals(predictions, residuals, alpha)
        if expected_rank == 0:
            assert np.isneginf(lo).all() and np.isposinf(hi).all()
        else:
            assert lo.item() == expected_rank - 1 and hi.item() == n - expected_rank


def test_exact_rank_avoids_interpolation_and_retains_ties() -> None:
    predictions = np.zeros((4, 1))
    residuals = np.array([1, 2, 2, 10])
    np.testing.assert_array_equal(
        jackknife_plus_intervals(predictions, residuals, 0.2), [[-10], [10]]
    )
    assert np.quantile(residuals, 0.8) < 10
    np.testing.assert_array_equal(
        jackknife_plus_intervals(predictions, residuals, 0.4), [[-2], [2]]
    )


def test_small_samples_and_tiny_alpha_retain_full_line() -> None:
    for alpha in [0.1, np.nextafter(0.0, 1.0)]:
        lower, upper = jackknife_plus_intervals(np.zeros((2, 3)), np.zeros(2), alpha)
        np.testing.assert_array_equal(lower, np.full(3, -np.inf))
        np.testing.assert_array_equal(upper, np.full(3, np.inf))


def test_candidate_overflow_yields_conservative_infinite_bounds() -> None:
    largest = np.finfo(float).max
    lower, upper = jackknife_plus_intervals(
        np.array([[-largest], [largest]]), np.full(2, largest), 0.4
    )
    assert lower.item() == -np.inf and upper.item() == np.inf


@pytest.mark.parametrize(
    "alpha", [0, -0.1, 0.5, 0.9, 1, np.nan, np.inf, True, False, "0.1", [0.1], 1j]
)
def test_invalid_alpha_is_rejected_by_both_apis(alpha) -> None:
    with pytest.raises(ValueError, match="alpha"):
        jackknife_plus_intervals(np.zeros((2, 1)), np.zeros(2), alpha)
    with pytest.raises(ValueError, match="alpha"):
        JackknifePlusRegressor(alpha=alpha)


@pytest.mark.parametrize(
    "predictions",
    [
        [],
        0,
        [0, 1],
        [[0]],
        np.empty((2, 0)),
        np.zeros((2, 1, 1)),
        [[0], [np.nan]],
        [[0], [np.inf]],
        [[0], [1j]],
        [["0"], ["1"]],
    ],
)
def test_malformed_prediction_matrices_are_rejected_even_for_infinite_ranks(predictions) -> None:
    with pytest.raises(ValueError, match="loo_predictions"):
        jackknife_plus_intervals(predictions, np.zeros(2))


@pytest.mark.parametrize(
    "residuals", [[], 0, [0], [[0], [1]], [0, -1], [0, np.inf], [0, np.nan], [0, 1j], ["0", "1"]]
)
def test_invalid_residuals_are_rejected_even_for_infinite_ranks(residuals) -> None:
    with pytest.raises(ValueError, match="loo_residuals"):
        jackknife_plus_intervals(np.zeros((2, 1)), residuals)


def test_numeric_objects_and_pandas_inputs_use_positional_pairing() -> None:
    predictions = pd.DataFrame([[0], [1], [2]], index=[8, 7, 6], dtype=object)
    residuals = pd.Series([1, 1, 1], index=[20, 21, 22], dtype=object)
    np.testing.assert_array_equal(
        jackknife_plus_intervals(predictions, residuals, 0.25), [[-1], [3]]
    )


def test_hand_calculated_omitted_means_and_residuals_differ_from_original_jackknife(
    monkeypatch,
) -> None:
    model = JackknifePlusRegressor(DummyRegressor(), alpha=0.25).fit(
        np.arange(3)[:, None], np.array([0, 2, 7])
    )
    assert [m.constant_.item() for m in model.loo_estimators_] == [4.5, 3.5, 1]
    np.testing.assert_array_equal(model.loo_residuals_, [4.5, 1.5, 6])
    query = np.zeros((2, 1))
    np.testing.assert_array_equal(model.predict(query), [3, 3])
    np.testing.assert_array_equal(model.predict_interval(query), [[-5, -5], [9, 9]])
    # The full-data prediction (and its [-3, 9] original-jackknife interval) is not used.
    monkeypatch.setattr(model.estimator_, "predict", lambda values: np.full(len(values), 1000))
    np.testing.assert_array_equal(model.predict_interval(query), [[-5, -5], [9, 9]])


def test_each_fit_excludes_exactly_its_own_response_and_refits_pipeline_preprocessing(data) -> None:
    class RecordingRegressor(RegressorMixin, BaseEstimator):
        def fit(self, values, targets):
            self.values_, self.targets_ = values.copy(), targets.copy()
            return self

        def predict(self, values):
            return np.full(len(values), np.mean(self.targets_))

    features, targets = data
    model = JackknifePlusRegressor(RecordingRegressor()).fit(features, targets)
    for index, estimator in enumerate(model.loo_estimators_):
        selected = np.arange(len(targets)) != index
        np.testing.assert_array_equal(estimator.values_, features[selected])
        np.testing.assert_array_equal(estimator.targets_, targets[selected])
        assert model.loo_residuals_[index] == abs(targets[index] - np.mean(targets[selected]))
    pipeline = JackknifePlusRegressor(make_pipeline(StandardScaler(), Ridge())).fit(
        features, targets
    )
    for index, estimator in enumerate(pipeline.loo_estimators_):
        expected_mean = np.delete(features, index, axis=0).mean(axis=0)
        np.testing.assert_allclose(estimator[0].mean_, expected_mean)
        assert estimator[0].n_samples_seen_ == len(features) - 1


def test_point_predictions_template_and_input_isolation(data) -> None:
    features, targets = data
    saved_x, saved_y = features.copy(), targets.copy()
    template = LinearRegression(copy_X=False).fit(features.copy(), targets.copy())
    saved_coef, expected = template.coef_.copy(), template.predict(features)
    model = JackknifePlusRegressor(template).fit(features, targets)
    np.testing.assert_allclose(model.predict(features), expected)
    assert len({id(m) for m in (model.estimator_, *model.loo_estimators_)}) == len(features) + 1
    assert model.estimator_ is not template
    np.testing.assert_array_equal(features, saved_x)
    np.testing.assert_array_equal(targets, saved_y)
    np.testing.assert_array_equal(template.coef_, saved_coef)


def test_repeated_calls_batches_and_training_permutations_agree_for_linear_learner(data) -> None:
    features, targets = data
    model = JackknifePlusRegressor().fit(*data)
    expected = model.predict_interval(features)
    np.testing.assert_array_equal(model.predict_interval(features), expected)
    np.testing.assert_allclose(
        model.predict_interval(features[::-1]), np.asarray(expected)[:, ::-1]
    )
    batches = [model.predict_interval(features[:4]), model.predict_interval(features[4:])]
    np.testing.assert_allclose(np.concatenate(batches, axis=1), expected)
    model.fit(features[::-1], targets[::-1])
    np.testing.assert_allclose(model.predict_interval(features), expected)
    model.fit(features, targets)
    np.testing.assert_allclose(model.predict_interval(features), expected)


def test_nested_stochastic_estimator_seeds_are_fixed_and_template_is_untouched(data) -> None:
    template = make_pipeline(
        StandardScaler(), DecisionTreeRegressor(max_depth=2, splitter="random")
    )
    first = JackknifePlusRegressor(template, random_state=7).fit(*data)
    second = JackknifePlusRegressor(template, random_state=7).fit(*data)
    assert template[-1].random_state is None and not hasattr(template[-1], "tree_")
    assert all(m[-1].random_state == 7 for m in (first.estimator_, *first.loo_estimators_))
    np.testing.assert_array_equal(first.predict_interval(data[0]), second.predict_interval(data[0]))


def test_constant_targets_and_two_training_rows() -> None:
    model = JackknifePlusRegressor(alpha=0.4).fit(np.zeros((2, 1)), np.full(2, 7))
    np.testing.assert_array_equal(model.loo_residuals_, [0, 0])
    np.testing.assert_array_equal(model.predict_interval(np.ones((2, 1))), np.full((2, 2), 7))
    model.alpha = 0.1
    model.fit(np.zeros((2, 1)), np.full(2, 7))
    lower, upper = model.predict_interval(np.ones((2, 1)))
    assert np.isneginf(lower).all() and np.isposinf(upper).all()


def test_regressor_accepts_positional_pandas_and_numeric_object_inputs(data) -> None:
    features, targets = data
    expected = JackknifePlusRegressor().fit(*data).predict_interval(features)
    frame = pd.DataFrame(features, columns=["b", "a"], index=np.arange(12)[::-1])
    model = JackknifePlusRegressor().fit(frame, pd.Series(targets, index=np.arange(12) + 100))
    np.testing.assert_array_equal(model.predict_interval(frame), expected)
    model.fit(features.astype(object), targets.astype(object))
    np.testing.assert_array_equal(model.predict_interval(features.astype(object)), expected)


@pytest.mark.parametrize("method", ["predict", "predict_interval"])
def test_prediction_requires_successful_fit(method) -> None:
    with pytest.raises(RuntimeError, match="must be fitted"):
        getattr(JackknifePlusRegressor(), method)(np.zeros((1, 1)))


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
        [[1j, 2]],
        [["1", "2"]],
        [[None, 1]],
    ],
)
@pytest.mark.parametrize("operation", ["fit", "predict", "predict_interval"])
def test_invalid_features_preserve_previous_fit(features, operation, data) -> None:
    model = JackknifePlusRegressor().fit(*data)
    previous, expected = model.loo_estimators_, model.predict_interval(data[0])
    with pytest.raises(ValueError):
        if operation == "fit":
            model.fit(features, np.zeros(2))
        else:
            getattr(model, operation)(features)
    assert model.loo_estimators_ is previous
    np.testing.assert_array_equal(model.predict_interval(data[0]), expected)


@pytest.mark.parametrize(
    "targets", [[], 0, [[0], [1]], [0], [0, np.nan], [0, np.inf], [0, 1j], ["0", "1"]]
)
def test_invalid_targets_leave_new_instance_unfitted(targets) -> None:
    model = JackknifePlusRegressor()
    with pytest.raises(ValueError):
        model.fit(np.zeros((2, 1)), targets)
    assert model.estimator_ is None and model.loo_estimators_ == ()
    assert model.loo_residuals_ is None and model.n_features_in_ is None and model.alpha_ is None


@pytest.mark.parametrize("seed", [-1, 2**32, 2.5, True, None, "42", [42]])
def test_invalid_seed_is_rejected(seed) -> None:
    with pytest.raises(ValueError, match="random_state"):
        JackknifePlusRegressor(random_state=seed)


@pytest.mark.parametrize(
    "estimator", [0, object(), LinearRegression, LogisticRegression(), StandardScaler()]
)
def test_invalid_estimator_is_rejected(estimator) -> None:
    with pytest.raises(ValueError, match="estimator"):
        JackknifePlusRegressor(estimator)


def test_option_changes_apply_only_after_successful_refit(data) -> None:
    model = JackknifePlusRegressor().fit(*data)
    expected = model.predict_interval(data[0])
    model.alpha = 0.25
    np.testing.assert_array_equal(model.predict_interval(data[0]), expected)
    with pytest.raises(ValueError, match="at least two"):
        model.fit(np.zeros((1, 3)), np.zeros(1))
    assert model.alpha_ == 0.1 and model.n_features_in_ == 2
    model.alpha = 0
    with pytest.raises(ValueError, match="alpha"):
        model.fit(*data)
    np.testing.assert_array_equal(model.predict_interval(data[0]), expected)
    model.alpha = 0.25
    model.fit(np.zeros((4, 3)), np.ones(4))
    assert model.alpha_ == 0.25 and model.n_features_in_ == 3 and len(model.loo_estimators_) == 4
    for method in [model.predict, model.predict_interval]:
        with pytest.raises(ValueError, match="same number of columns"):
            method(np.zeros((1, 2)))


@pytest.mark.parametrize("already_fitted", [False, True])
def test_later_omitted_fit_failure_is_transactional(monkeypatch, data, already_fitted) -> None:
    model = JackknifePlusRegressor()
    if already_fitted:
        model.fit(*data)
    previous = model.__dict__.copy()
    real_clone, calls = jackknife._clone_regressor, []

    def clone(template, seed):
        candidate = real_clone(template, seed)
        calls.append(candidate)
        if len(calls) == 3:

            def fail(*args):
                raise RuntimeError("later omission failed")

            monkeypatch.setattr(candidate, "fit", fail)
        return candidate

    monkeypatch.setattr(jackknife, "_clone_regressor", clone)
    with pytest.raises(RuntimeError, match="later omission failed"):
        model.fit(*data)
    assert len(calls) == 3
    assert all(model.__dict__[name] is value for name, value in previous.items())


@pytest.mark.parametrize("predictions", [[np.nan, 0], [np.inf, 0], [[1], [2]], [1], ["1", "2"]])
@pytest.mark.parametrize("method", ["predict", "predict_interval"])
def test_invalid_estimator_predictions_are_rejected(monkeypatch, data, predictions, method) -> None:
    model = JackknifePlusRegressor().fit(*data)
    estimator = model.estimator_ if method == "predict" else model.loo_estimators_[1]
    monkeypatch.setattr(estimator, "predict", lambda values: np.asarray(predictions))
    with pytest.raises(RuntimeError, match="invalid predictions"):
        getattr(model, method)(np.zeros((2, 2)))


def test_unsafe_held_out_residual_arithmetic_preserves_fit(data) -> None:
    model = JackknifePlusRegressor().fit(*data)
    previous = model.estimator_, model.loo_estimators_, model.loo_residuals_
    model.estimator = DummyRegressor(strategy="constant", constant=-1e308)
    with pytest.raises(RuntimeError, match="fitting failed numerically"):
        model.fit(np.zeros((2, 1)), np.full(2, 1e308))
    assert model.estimator_ is previous[0] and model.loo_estimators_ is previous[1]
    assert model.loo_residuals_ is previous[2]


def test_offline_example_reports_nominal_and_general_coverage_separately(tmp_path: Path) -> None:
    example = Path(__file__).resolve().parents[1] / "examples/run_jackknife_plus.py"
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
    assert report["seed"] == 42
    assert (report["train_rows"], report["test_rows"], report["fits_per_level"]) == (200, 2000, 201)
    assert [row["alpha"] for row in report["evaluations"]] == [0.1, 0.05]
    table = pd.read_csv(report["table_path"])
    assert len(table) == 8
    for evaluation in report["evaluations"]:
        alpha = evaluation["alpha"]
        assert evaluation["nominal_coverage"] == 1 - alpha
        assert evaluation["general_coverage_bound"] == 1 - 2 * alpha
        group = table[table["alpha"] == alpha]
        assert group["count"].sum() == 2000 and group["bin_index"].tolist() == [0, 1, 2, 3]
        assert group["nominal_coverage"].eq(1 - alpha).all()
        assert group["general_coverage_bound"].eq(1 - 2 * alpha).all()
        np.testing.assert_allclose(group["coverage"], group["covered_count"] / group["count"])
        for name in ["coverage", "average_width", "empty_set_rate"]:
            assert np.average(group[name], weights=group["count"]) == pytest.approx(
                evaluation["test_intervals"][name]
            )
        assert (group["min_width"] > 0).all() and np.isfinite(group["max_width"]).all()
