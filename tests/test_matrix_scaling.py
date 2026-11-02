"""Check full matrix calibration, its orientation, and failure behavior."""

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

from calibrated_ml_lab.calibration import matrix
from calibrated_ml_lab.calibration.matrix import MulticlassMatrixScaler
from calibrated_ml_lab.calibration.vector import MulticlassVectorScaler


@pytest.fixture
def exact_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A nonsymmetric affine map with exact class frequencies at six distinct inputs."""
    probabilities = np.array(
        [
            [0.6, 0.3, 0.1],
            [0.2, 0.5, 0.3],
            [0.1, 0.2, 0.7],
            [0.4, 0.1, 0.5],
            [0.1, 0.6, 0.3],
            [0.3, 0.2, 0.5],
        ]
    )
    weights = np.array([[1.2, 0.7, -0.2], [-0.4, 0.8, 0.5], [0.3, -0.6, 1.1]])
    bias = np.array([0.2, -0.5, 0.3])
    logits = np.linalg.solve(weights, (np.log(probabilities) - bias).T).T
    labels = np.concatenate([np.repeat(np.arange(3), (100 * p).astype(int)) for p in probabilities])
    return (
        np.repeat(logits, 100, axis=0),
        labels,
        np.repeat(probabilities, 100, axis=0),
        weights,
        bias,
    )


def test_recovers_nondiagonal_transformation_and_correct_matrix_orientation(exact_data) -> None:
    logits, labels, expected, weights, bias = exact_data
    original_logits, original_labels = logits.copy(), labels.copy()
    scaler = MulticlassMatrixScaler(l2=0)
    assert scaler.fit(logits, labels) is scaler
    assert scaler.n_classes_ == 3 and scaler.weights_.shape == (3, 3)
    assert scaler.bias_.shape == (3,)
    np.testing.assert_allclose(scaler.transform(logits), expected, atol=3e-6)
    # Softmax does not identify a common output-logit offset. Compare class contrasts.
    np.testing.assert_allclose(
        scaler.weights_ - scaler.weights_.mean(axis=0),
        weights - weights.mean(axis=0),
        atol=5e-4,
    )
    np.testing.assert_allclose(scaler.bias_ - scaler.bias_.mean(), bias - bias.mean(), atol=5e-4)
    diagonal = MulticlassVectorScaler(l2=0).fit(logits, labels)
    assert (
        log_loss(labels, scaler.transform(logits))
        < log_loss(labels, diagonal.transform(logits)) - 0.001
    )
    np.testing.assert_array_equal(logits, original_logits)
    np.testing.assert_array_equal(labels, original_labels)


@pytest.mark.parametrize("l2", [0, 0.01, 10])
def test_full_matrix_gradient_matches_finite_differences(l2: float) -> None:
    logits = np.array([[2.0, 0.5, -1.0], [-0.2, 1.0, 0.1], [0.5, -0.4, 2.0]])
    labels = np.array([1, 2, 0])
    parameters = np.array([0.7, -0.3, 1.8, -0.4, 0.2, 0.9, 1.1, 0.6, -0.5, 0.1, -0.2, 0.3])
    assert (
        check_grad(
            lambda p: matrix._objective(p, logits, labels, l2)[0],
            lambda p: matrix._objective(p, logits, labels, l2)[1],
            parameters,
        )
        < 3e-6
    )


def test_regularization_shrinks_full_matrix_to_identity(exact_data) -> None:
    logits, labels, *_ = exact_data
    weak = MulticlassMatrixScaler(l2=1e-3).fit(logits, labels)
    strong = MulticlassMatrixScaler(l2=10).fit(logits, labels)
    assert np.linalg.norm(
        np.r_[(strong.weights_ - np.eye(3)).ravel(), strong.bias_]
    ) < np.linalg.norm(np.r_[(weak.weights_ - np.eye(3)).ravel(), weak.bias_])
    assert log_loss(labels, strong.transform(logits)) <= log_loss(labels, softmax(logits, axis=1))
    assert log_loss(labels, weak.transform(logits)) < log_loss(labels, strong.transform(logits))


def test_transform_matches_full_affine_map_on_new_rows_and_preserves_inputs(exact_data) -> None:
    scaler = MulticlassMatrixScaler().fit(*exact_data[:2])
    logits = np.array([[0.0, 0.0, 0.0], [1, -2, 0], [-1, 2, 3], [-1e100, 0, 1e100]])
    original = logits.copy()
    actual = scaler.transform(logits)
    expected = softmax(logits @ scaler.weights_.T + scaler.bias_, axis=1)
    np.testing.assert_allclose(actual, expected)
    assert actual.shape == logits.shape and np.isfinite(actual).all()
    assert ((actual >= 0) & (actual <= 1)).all()
    np.testing.assert_allclose(actual.sum(axis=1), 1)
    np.testing.assert_allclose(scaler.transform(logits[:1]), actual[:1])
    assert (actual.argmax(axis=1) != logits.argmax(axis=1)).any()
    assert not np.allclose(scaler.transform(logits[:1] + 5), actual[:1])
    np.testing.assert_array_equal(logits, original)


def test_class_permutation_row_order_and_repeated_fits_preserve_predictions(exact_data) -> None:
    logits, labels, *_ = exact_data
    scaler = MulticlassMatrixScaler().fit(logits, labels)
    expected = scaler.transform(logits)
    scaler.fit(logits, labels)
    np.testing.assert_array_equal(scaler.transform(logits), expected)
    permutation = np.array([2, 0, 1])
    permuted = MulticlassMatrixScaler().fit(logits[:, permutation], np.argsort(permutation)[labels])
    np.testing.assert_allclose(
        permuted.transform(logits[:, permutation]), expected[:, permutation], atol=2e-6
    )
    reversed_rows = MulticlassMatrixScaler().fit(logits[::-1], labels[::-1])
    np.testing.assert_allclose(reversed_rows.transform(logits), expected, atol=2e-6)


def test_uniform_logits_learn_priors_and_successful_refits_change_class_count() -> None:
    scaler = MulticlassMatrixScaler(l2=0).fit(
        np.zeros((100, 3)), np.repeat([0, 1, 2], [60, 30, 10])
    )
    np.testing.assert_allclose(scaler.transform(np.zeros((1, 3))), [[0.6, 0.3, 0.1]], atol=1e-6)
    np.testing.assert_array_equal(scaler.weights_, np.eye(3))
    scaler.fit(np.zeros((4, 2), dtype=object), np.array([True, False, True, False], dtype=object))
    assert scaler.n_classes_ == 2 and scaler.weights_.shape == (2, 2) and scaler.bias_.shape == (2,)
    np.testing.assert_array_equal(scaler.weights_, np.eye(2))
    np.testing.assert_array_equal(scaler.bias_, np.zeros(2))
    np.testing.assert_allclose(scaler.transform(np.zeros((1, 2), dtype=object)), 0.5)
    with pytest.raises(ValueError, match="same number of class columns"):
        scaler.transform(np.zeros((1, 3)))


@pytest.mark.parametrize("classes", [2, 4])
def test_single_class_and_one_row_calibration_remain_finite(classes: int) -> None:
    scaler = MulticlassMatrixScaler().fit(np.zeros((1, classes)), np.array([1]))
    probabilities = scaler.transform(np.zeros((1, classes)))
    assert probabilities[0, 1] > 0.98
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    assert np.isfinite(scaler.weights_).all() and np.isfinite(scaler.bias_).all()


@pytest.mark.parametrize(
    "logits",
    [
        [],
        [1, 2],
        np.empty((0, 3)),
        [[1], [2]],
        [[[1, 2]]],
        [[1, np.nan, 0]],
        [[1, np.inf, 0]],
        [[1j, 2, 3]],
        [["1", "2", "3"]],
        np.array([[1, "2", 3]], dtype=object),
    ],
)
@pytest.mark.parametrize("operation", ["fit", "transform"])
def test_invalid_logits_preserve_previous_fit(logits, operation, exact_data) -> None:
    scaler = MulticlassMatrixScaler().fit(*exact_data[:2])
    expected = scaler.transform(exact_data[0])
    with pytest.raises(ValueError):
        if operation == "fit":
            scaler.fit(np.asarray(logits), np.array([0]))
        else:
            scaler.transform(np.asarray(logits))
    assert scaler.n_classes_ == 3
    np.testing.assert_array_equal(scaler.transform(exact_data[0]), expected)


@pytest.mark.parametrize(
    "labels",
    [
        [],
        0,
        [[0], [1]],
        [0],
        [0, 0.5],
        [-1, 1],
        [0, 3],
        [0, np.nan],
        [0, np.inf],
        [0, 1j],
        ["0", "1"],
    ],
)
def test_invalid_labels_leave_new_instance_unfitted(labels) -> None:
    scaler = MulticlassMatrixScaler()
    with pytest.raises(ValueError):
        scaler.fit(np.ones((2, 3)), np.asarray(labels))
    assert scaler.n_classes_ is None and scaler.weights_ is None and scaler.bias_ is None
    with pytest.raises(RuntimeError, match="must be fitted"):
        scaler.transform(np.ones((1, 3)))


@pytest.mark.parametrize(
    "l2", [-1, np.nan, np.inf, True, np.array(True, dtype=object), "1", 1j, [1]]
)
def test_invalid_regularization_is_rejected(l2) -> None:
    with pytest.raises(ValueError, match="l2"):
        MulticlassMatrixScaler(l2=l2)


@pytest.mark.parametrize("iterations", [0, -1, 1.5, True, "3", np.nan, [10]])
def test_invalid_iteration_limit_is_rejected(iterations) -> None:
    with pytest.raises(ValueError, match="max_iter"):
        MulticlassMatrixScaler(max_iter=iterations)


def test_failed_refit_labels_options_and_nonconvergence_keep_predictions(exact_data) -> None:
    scaler = MulticlassMatrixScaler().fit(*exact_data[:2])
    expected = scaler.transform(exact_data[0])
    with pytest.raises(ValueError):
        scaler.fit(np.zeros((2, 4)), np.array([0, 0.5]))
    scaler.l2 = -1
    with pytest.raises(ValueError, match="l2"):
        scaler.fit(*exact_data[:2])
    scaler.l2, scaler.max_iter = 1e-3, 1
    with pytest.raises(RuntimeError, match="optimisation failed"):
        scaler.fit(*exact_data[:2])
    assert scaler.n_classes_ == 3
    np.testing.assert_array_equal(scaler.transform(exact_data[0]), expected)


@pytest.mark.parametrize(
    "result",
    [
        OptimizeResult(success=False),
        OptimizeResult(success=True, x=np.zeros(20), fun=np.inf),
        OptimizeResult(success=True, x=np.full(20, np.nan), fun=0),
        OptimizeResult(success=True, x=np.zeros(19), fun=0),
        OptimizeResult(success=True, x=np.zeros((4, 5)), fun=0),
    ],
)
@pytest.mark.parametrize("already_fitted", [False, True])
def test_bad_optimizer_results_preserve_fitted_state(
    monkeypatch, result, already_fitted, exact_data
) -> None:
    scaler = MulticlassMatrixScaler()
    expected = None
    if already_fitted:
        scaler.fit(*exact_data[:2])
        expected = scaler.transform(exact_data[0])
    monkeypatch.setattr(matrix, "minimize", lambda *args, **kwargs: result)
    with pytest.raises(RuntimeError, match="optimisation failed"):
        scaler.fit(np.zeros((4, 4)), np.arange(4))
    if already_fitted:
        assert scaler.n_classes_ == 3
        np.testing.assert_array_equal(scaler.transform(exact_data[0]), expected)
    else:
        assert scaler.n_classes_ is None and scaler.weights_ is None and scaler.bias_ is None


def test_successful_but_worse_candidate_falls_back_to_identity(monkeypatch, exact_data) -> None:
    monkeypatch.setattr(
        matrix,
        "minimize",
        lambda *args, **kwargs: OptimizeResult(
            success=True,
            x=np.r_[np.eye(3).ravel(), [1000, 0, 0]],
            fun=0,
        ),
    )
    scaler = MulticlassMatrixScaler().fit(*exact_data[:2])
    np.testing.assert_array_equal(scaler.weights_, np.eye(3))
    np.testing.assert_array_equal(scaler.bias_, np.zeros(3))
    np.testing.assert_allclose(scaler.transform(exact_data[0]), softmax(exact_data[0], axis=1))


def test_unsafe_matrix_arithmetic_preserves_previous_fit(monkeypatch, exact_data) -> None:
    scaler = MulticlassMatrixScaler().fit(*exact_data[:2])
    expected = scaler.transform(exact_data[0])
    with pytest.raises(ValueError, match="too large"):
        scaler.fit(np.array([[-1e308, 1e308, 0]]), np.array([0]))
    # Same-sign products in one output must not overflow silently during matmul.
    unsafe = np.sign(scaler.weights_[0])[None, :] * 1.7e308
    with pytest.raises(ValueError, match="too large"):
        scaler.transform(unsafe)
    monkeypatch.setattr(
        matrix,
        "minimize",
        lambda *args, **kwargs: OptimizeResult(
            success=True,
            x=np.full(12, 1e308),
            fun=0,
        ),
    )
    with pytest.raises(RuntimeError, match="failed numerically"):
        scaler.fit(*exact_data[:2])
    np.testing.assert_array_equal(scaler.transform(exact_data[0]), expected)


def test_offline_matrix_example_runs_from_another_directory(tmp_path: Path) -> None:
    example = Path(__file__).resolve().parents[1] / "examples/run_matrix_scaling.py"
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
    assert report["matrix_l2"] == 1e-3
    assert np.shape(report["matrix_weights"]) == (3, 3) and len(report["matrix_bias"]) == 3
    metrics = report["test_metrics"]
    assert set(metrics) == {"uncalibrated", "temperature", "vector", "matrix"}
    assert metrics["matrix"]["log_loss"] < metrics["vector"]["log_loss"]
    assert metrics["matrix"]["log_loss"] < metrics["uncalibrated"]["log_loss"]
    for method in metrics.values():
        assert method["log_loss"] >= 0
        assert all(
            0 <= method[key] <= 1
            for key in ["accuracy", "confidence_ece", "top_label_ece", "classwise_ece"]
        )
