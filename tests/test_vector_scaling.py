"""Check vector scaling against exact distributions, gradients, and failure contracts."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import OptimizeResult, check_grad
from scipy.special import softmax
from sklearn.metrics import log_loss

from calibrated_ml_lab.calibration import vector
from calibrated_ml_lab.calibration.multiclass import MulticlassTemperatureScaler
from calibrated_ml_lab.calibration.vector import MulticlassVectorScaler


@pytest.fixture
def exact_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Four distinct logit rows with exact empirical class frequencies and a known map."""
    probabilities = np.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.1, 0.2, 0.7], [0.4, 0.1, 0.5]])
    logits = (np.log(probabilities) - [0.4, -0.6, 0.2]) / [0.5, 1.5, -0.8]
    labels = np.concatenate([np.repeat(np.arange(3), (100 * p).astype(int)) for p in probabilities])
    return np.repeat(logits, 100, axis=0), labels, np.repeat(probabilities, 100, axis=0)


def test_recovers_class_specific_weights_and_biases(exact_data) -> None:
    logits, labels, expected = exact_data
    original_logits, original_labels = logits.copy(), labels.copy()
    scaler = MulticlassVectorScaler(l2=0)
    assert scaler.fit(logits, labels) is scaler
    assert scaler.n_classes_ == 3
    np.testing.assert_allclose(scaler.weights_, [0.5, 1.5, -0.8], atol=2e-4)
    np.testing.assert_allclose(scaler.bias_, [0.4, -0.6, 0.2], atol=2e-4)
    np.testing.assert_allclose(scaler.transform(logits), expected, atol=2e-6)
    temperature = MulticlassTemperatureScaler().fit(logits, labels)
    assert log_loss(labels, expected) < log_loss(labels, temperature.transform(logits)) - 0.05
    assert log_loss(labels, scaler.transform(logits)) < log_loss(labels, softmax(logits, axis=1))
    np.testing.assert_array_equal(logits, original_logits)
    np.testing.assert_array_equal(labels, original_labels)


@pytest.mark.parametrize("l2", [0, 0.01, 10])
def test_analytic_gradient_matches_finite_differences(l2: float) -> None:
    logits = np.array([[2.0, 0.5, -1.0], [-0.2, 1.0, 0.1], [0.5, -0.4, 2.0]])
    labels = np.array([1, 2, 0])
    parameters = np.array([0.7, -0.3, 1.8, -0.4, 0.2, 0.9])
    error = check_grad(
        lambda p: vector._objective(p, logits, labels, l2)[0],
        lambda p: vector._objective(p, logits, labels, l2)[1],
        parameters,
    )
    assert error < 2e-6


def test_regularization_shrinks_towards_identity_without_increasing_fitting_nll(exact_data) -> None:
    logits, labels, _ = exact_data
    weak = MulticlassVectorScaler(l2=1e-3).fit(logits, labels)
    strong = MulticlassVectorScaler(l2=10).fit(logits, labels)
    weak_distance = np.linalg.norm(np.r_[weak.weights_ - 1, weak.bias_])
    strong_distance = np.linalg.norm(np.r_[strong.weights_ - 1, strong.bias_])
    assert strong_distance < weak_distance
    assert log_loss(labels, strong.transform(logits)) <= log_loss(labels, softmax(logits, axis=1))
    assert log_loss(labels, weak.transform(logits)) < log_loss(labels, strong.transform(logits))


def test_transform_uses_raw_logits_and_can_change_the_winning_class(exact_data) -> None:
    logits, labels, _ = exact_data
    scaler = MulticlassVectorScaler(l2=0).fit(logits, labels)
    new_logits = np.array([[0.0, 0.0, 0.0], [-1, 2, 3], [1, -2, 0], [-1e100, 0, 1e100]])
    original = new_logits.copy()
    probabilities = scaler.transform(new_logits)
    np.testing.assert_allclose(
        probabilities, softmax(new_logits * scaler.weights_ + scaler.bias_, axis=1)
    )
    assert probabilities.shape == new_logits.shape
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    assert (probabilities.argmax(axis=1) != new_logits.argmax(axis=1)).any()
    assert not np.allclose(scaler.transform(new_logits[:1] + 5), probabilities[:1])
    np.testing.assert_allclose(scaler.transform(new_logits[:1]), probabilities[:1])
    np.testing.assert_array_equal(new_logits, original)


def test_class_and_row_permutations_preserve_predictions(exact_data) -> None:
    logits, labels, _ = exact_data
    expected = MulticlassVectorScaler().fit(logits, labels).transform(logits)
    columns = np.array([2, 0, 1])
    actual = MulticlassVectorScaler().fit(logits[:, columns], np.argsort(columns)[labels])
    np.testing.assert_allclose(
        actual.transform(logits[:, columns]), expected[:, columns], atol=1e-6
    )
    reordered = MulticlassVectorScaler().fit(logits[::-1], labels[::-1])
    np.testing.assert_allclose(reordered.transform(logits), expected, atol=1e-6)


def test_uniform_logits_can_learn_class_priors_without_logit_variation() -> None:
    logits = np.zeros((100, 3))
    labels = np.repeat([0, 1, 2], [60, 30, 10])
    scaler = MulticlassVectorScaler(l2=0).fit(logits, labels)
    np.testing.assert_allclose(scaler.transform(logits)[0], [0.6, 0.3, 0.1], atol=1e-6)
    np.testing.assert_array_equal(scaler.weights_, np.ones(3))


def test_balanced_uniform_data_keeps_identity() -> None:
    scaler = MulticlassVectorScaler().fit(np.zeros((6, 3)), np.arange(6) % 3)
    np.testing.assert_array_equal(scaler.weights_, np.ones(3))
    np.testing.assert_array_equal(scaler.bias_, np.zeros(3))


@pytest.mark.parametrize("classes", [2, 4])
def test_single_class_calibration_and_one_row_are_supported(classes: int) -> None:
    scaler = MulticlassVectorScaler().fit(np.zeros((1, classes)), np.array([1]))
    probabilities = scaler.transform(np.zeros((2, classes)))
    assert scaler.n_classes_ == classes
    assert probabilities[0, 1] > 0.98
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    assert np.isfinite(scaler.weights_).all() and np.isfinite(scaler.bias_).all()


@pytest.mark.parametrize(
    "logits",
    [
        [],
        1,
        [1, 2],
        np.empty((0, 3)),
        np.empty((2, 0)),
        [[1], [2]],
        [[[1, 2]]],
        [[1, np.nan, 0]],
        [[1, np.inf, 0]],
        [[1, -np.inf, 0]],
        [[1j, 2, 3]],
        [["1", "2", "3"]],
        np.array([[1, "2", 3]], dtype=object),
        [[None, 2, 3]],
    ],
)
@pytest.mark.parametrize("operation", ["fit", "transform"])
def test_invalid_logits_preserve_successful_fit(logits, operation, exact_data) -> None:
    values, labels, _ = exact_data
    scaler = MulticlassVectorScaler().fit(values, labels)
    expected = scaler.transform(values)
    weights, bias = scaler.weights_.copy(), scaler.bias_.copy()
    with pytest.raises(ValueError):
        if operation == "fit":
            scaler.fit(np.asarray(logits), np.array([0]))
        else:
            scaler.transform(np.asarray(logits))
    assert scaler.n_classes_ == 3
    np.testing.assert_array_equal(scaler.weights_, weights)
    np.testing.assert_array_equal(scaler.bias_, bias)
    np.testing.assert_array_equal(scaler.transform(values), expected)


@pytest.mark.parametrize(
    "labels",
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
def test_invalid_labels_leave_a_new_scaler_unfitted(labels) -> None:
    scaler = MulticlassVectorScaler()
    with pytest.raises(ValueError):
        scaler.fit(np.array([[1, 2, 3], [3, 2, 1]]), np.asarray(labels))
    assert scaler.n_classes_ is None and scaler.weights_ is None and scaler.bias_ is None
    with pytest.raises(RuntimeError, match="must be fitted"):
        scaler.transform(np.ones((1, 3)))


def test_numeric_objects_and_successful_refits_can_change_class_count(exact_data) -> None:
    logits, labels, _ = exact_data
    scaler = MulticlassVectorScaler().fit(logits, labels)
    before = scaler.transform(logits)
    scaler.fit(logits.astype(object), labels.astype(object))
    np.testing.assert_array_equal(scaler.transform(logits.astype(object)), before)
    scaler.fit(np.zeros((4, 2)), np.array([True, False, True, False], dtype=object))
    assert scaler.n_classes_ == 2 and scaler.weights_.shape == scaler.bias_.shape == (2,)
    np.testing.assert_allclose(scaler.transform(np.zeros((1, 2))), 0.5)
    with pytest.raises(ValueError, match="same number of class columns"):
        scaler.transform(logits)


@pytest.mark.parametrize(
    "l2",
    [-1, np.nan, np.inf, -np.inf, True, np.array(True), np.array(True, dtype=object), "1", 1j, [1]],
)
def test_invalid_regularization_is_rejected(l2) -> None:
    with pytest.raises(ValueError, match="l2"):
        MulticlassVectorScaler(l2=l2)


@pytest.mark.parametrize("iterations", [0, -1, 1.5, True, np.bool_(True), "3", np.nan, [10]])
def test_invalid_iteration_limit_is_rejected(iterations) -> None:
    with pytest.raises(ValueError, match="max_iter"):
        MulticlassVectorScaler(max_iter=iterations)


def test_invalid_refit_labels_and_options_preserve_fit(exact_data) -> None:
    logits, labels, _ = exact_data
    scaler = MulticlassVectorScaler().fit(logits, labels)
    expected = scaler.transform(logits)
    with pytest.raises(ValueError):
        scaler.fit(np.zeros((2, 4)), np.array([0, 0.5]))
    scaler.l2 = -1
    with pytest.raises(ValueError, match="l2"):
        scaler.fit(logits, labels)
    assert scaler.n_classes_ == 3
    np.testing.assert_array_equal(scaler.transform(logits), expected)


@pytest.mark.parametrize(
    "result",
    [
        OptimizeResult(success=False),
        OptimizeResult(success=True, x=np.zeros(8), fun=np.inf),
        OptimizeResult(success=True, x=np.full(8, np.nan), fun=0),
        OptimizeResult(success=True, x=np.zeros(7), fun=0),
        OptimizeResult(success=True, x=np.zeros((2, 4)), fun=0),
    ],
)
@pytest.mark.parametrize("already_fitted", [False, True])
def test_failed_optimizer_preserves_fitted_state(
    monkeypatch, result, already_fitted, exact_data
) -> None:
    logits, labels, _ = exact_data
    scaler = MulticlassVectorScaler()
    expected = None
    if already_fitted:
        scaler.fit(logits, labels)
        expected = scaler.transform(logits)
    monkeypatch.setattr(vector, "minimize", lambda *args, **kwargs: result)
    with pytest.raises(RuntimeError, match="optimisation failed"):
        scaler.fit(np.zeros((4, 4)), np.arange(4))
    if already_fitted:
        assert scaler.n_classes_ == 3
        np.testing.assert_array_equal(scaler.transform(logits), expected)
    else:
        assert scaler.n_classes_ is None and scaler.weights_ is None and scaler.bias_ is None


def test_nonconvergence_preserves_previous_predictions(exact_data) -> None:
    logits, labels, _ = exact_data
    scaler = MulticlassVectorScaler().fit(logits, labels)
    expected = scaler.transform(logits)
    scaler.max_iter = 1
    with pytest.raises(RuntimeError, match="optimisation failed"):
        scaler.fit(logits, labels)
    np.testing.assert_array_equal(scaler.transform(logits), expected)


def test_successful_but_worse_candidate_uses_identity(monkeypatch, exact_data) -> None:
    logits, labels, _ = exact_data
    monkeypatch.setattr(
        vector,
        "minimize",
        lambda *args, **kwargs: OptimizeResult(
            success=True,
            x=np.array([1, 1, 1, 1000, 0, 0]),
            fun=0,
        ),
    )
    scaler = MulticlassVectorScaler().fit(logits, labels)
    np.testing.assert_array_equal(scaler.weights_, np.ones(3))
    np.testing.assert_array_equal(scaler.bias_, np.zeros(3))
    np.testing.assert_allclose(scaler.transform(logits), softmax(logits, axis=1))


def test_unsafe_arithmetic_raises_without_overwriting_previous_parameters(
    exact_data, monkeypatch
) -> None:
    logits, labels, _ = exact_data
    scaler = MulticlassVectorScaler(l2=0).fit(logits, labels)
    before = scaler.transform(logits)
    with pytest.raises(ValueError, match="too large"):
        scaler.fit(np.array([[-1e308, 1e308, 0]]), np.array([0]))
    with pytest.raises(ValueError, match="too large"):
        scaler.transform(np.array([[0, 1.7e308, 0]]))
    monkeypatch.setattr(
        vector,
        "minimize",
        lambda *args, **kwargs: OptimizeResult(
            success=True,
            x=np.full(6, 1e308),
            fun=0,
        ),
    )
    with pytest.raises(RuntimeError, match="failed numerically"):
        scaler.fit(logits, labels)
    np.testing.assert_array_equal(scaler.transform(logits), before)


def test_offline_vector_example_from_another_directory(tmp_path: Path) -> None:
    example = Path(__file__).resolve().parents[1] / "examples/run_vector_scaling.py"
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
    assert report["calibration_rows"] == 2000 and report["test_rows"] == 1000
    assert report["vector_l2"] == 1e-3
    assert len(report["vector_weights"]) == len(report["vector_bias"]) == 3
    metrics = report["test_metrics"]
    assert set(metrics) == {"uncalibrated", "temperature", "vector"}
    assert metrics["vector"]["log_loss"] < metrics["uncalibrated"]["log_loss"]
    assert metrics["vector"]["log_loss"] < metrics["temperature"]["log_loss"]
    for method in metrics.values():
        assert 0 <= method["accuracy"] <= 1
        assert method["log_loss"] >= 0
        assert all(
            0 <= method[key] <= 1 for key in ["confidence_ece", "top_label_ece", "classwise_ece"]
        )
