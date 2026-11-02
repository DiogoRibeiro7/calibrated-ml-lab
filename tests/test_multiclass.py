"""Validate multiclass temperature scaling against known distributions and invariants."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import OptimizeResult
from scipy.special import softmax
from sklearn.metrics import log_loss

from calibrated_ml_lab.calibration import multiclass
from calibrated_ml_lab.calibration.multiclass import MulticlassTemperatureScaler


@pytest.fixture
def calibration_data() -> tuple[np.ndarray, np.ndarray]:
    """An exact empirical distribution with a known temperature of three."""
    logits = np.tile(3.0 * np.log([0.7, 0.2, 0.1]), (100, 1))
    return logits, np.repeat([0, 1, 2], [70, 20, 10])


@pytest.mark.parametrize("known_temperature", [0.3, 1.0, 3.0])
def test_recovers_known_temperature_and_reduces_nll(
    calibration_data, known_temperature: float
) -> None:
    logits, labels = calibration_data
    logits = logits * (known_temperature / 3.0)
    original_logits, original_labels = logits.copy(), labels.copy()
    scaler = MulticlassTemperatureScaler()
    assert scaler.fit(logits, labels) is scaler
    assert scaler.temperature_ == pytest.approx(known_temperature, rel=1e-5)
    assert scaler.n_classes_ == 3
    probabilities = scaler.transform(logits)
    np.testing.assert_allclose(probabilities, np.tile([0.7, 0.2, 0.1], (100, 1)), atol=1e-6)
    original_loss = log_loss(labels, softmax(logits, axis=1))
    assert log_loss(labels, probabilities) <= original_loss + 1e-12
    if known_temperature != 1.0:
        assert log_loss(labels, probabilities) < original_loss
    np.testing.assert_array_equal(logits, original_logits)
    np.testing.assert_array_equal(labels, original_labels)


def test_predictions_are_normalized_preserve_winner_and_accept_new_rows(calibration_data) -> None:
    scaler = MulticlassTemperatureScaler().fit(*calibration_data)
    logits = np.array([[2, 1, -3], [-4, 3, 1], [-2, 2, 8], [0, 0, 0]], dtype=float)
    probabilities = scaler.transform(logits)
    assert probabilities.shape == logits.shape
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    np.testing.assert_array_equal(probabilities.argmax(axis=1), logits.argmax(axis=1))
    np.testing.assert_allclose(scaler.transform(logits[:1]), probabilities[:1])


def test_row_shifts_and_class_permutations_do_not_change_the_fit(calibration_data) -> None:
    logits, labels = calibration_data
    scaler = MulticlassTemperatureScaler().fit(logits, labels)
    shifted = logits + np.arange(len(logits))[:, None] * 100.0
    shifted_scaler = MulticlassTemperatureScaler().fit(shifted, labels)
    assert shifted_scaler.temperature_ == pytest.approx(scaler.temperature_, rel=1e-9)
    np.testing.assert_allclose(scaler.transform(shifted), scaler.transform(logits), atol=1e-12)
    permutation = np.array([2, 0, 1])
    permuted_labels = np.argsort(permutation)[labels]
    permuted = MulticlassTemperatureScaler().fit(logits[:, permutation], permuted_labels)
    assert permuted.temperature_ == pytest.approx(scaler.temperature_, rel=1e-9)
    np.testing.assert_allclose(
        permuted.transform(logits[:, permutation]), scaler.transform(logits)[:, permutation]
    )


def test_uniform_logits_keep_identity_temperature() -> None:
    logits = np.full((6, 3), 1e308)
    scaler = MulticlassTemperatureScaler().fit(logits, np.arange(6) % 3)
    assert scaler.temperature_ == 1.0
    np.testing.assert_allclose(scaler.transform(logits), 1 / 3)


@pytest.mark.parametrize("magnitude", [1e4, 1e300])
def test_large_logit_gaps_remain_finite(magnitude: float) -> None:
    logits = magnitude * np.array([[1, -1, 0], [-1, 1, 0], [0, -1, 1]])
    labels = np.array([1, 2, 0])
    scaler = MulticlassTemperatureScaler().fit(logits, labels)
    assert scaler.temperature_ == pytest.approx(np.exp(3.0))
    probabilities = scaler.transform(logits)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_two_columns_and_single_class_labels_are_supported() -> None:
    logits = np.tile([0.0, 0.1], (5, 1))
    scaler = MulticlassTemperatureScaler().fit(logits, np.ones(5))
    assert scaler.n_classes_ == 2
    assert scaler.temperature_ == pytest.approx(np.exp(-3.0))
    assert (scaler.transform(logits)[:, 1] > softmax(logits, axis=1)[:, 1]).all()


@pytest.mark.parametrize(
    "logits",
    [
        [],
        [1, 2, 3],
        1.0,
        np.empty((0, 3)),
        np.empty((2, 0)),
        [[1], [2]],
        [[[1, 2, 3]]],
        [[1, np.nan, 0]],
        [[1, np.inf, 0]],
        [[1, -np.inf, 0]],
        [[1 + 1j, 2, 3]],
        [["1", "2", "3"]],
        np.array([[1, "2", 3]], dtype=object),
        np.array([[None, 2, 3]], dtype=object),
        [[-1e308, 0, 1e308]],
        [[0, 1e308, 1]],
    ],
)
@pytest.mark.parametrize("operation", ["fit", "transform"])
def test_rejects_invalid_logits_without_changing_a_previous_fit(
    logits,
    operation: str,
    calibration_data,
) -> None:
    scaler = MulticlassTemperatureScaler().fit(*calibration_data)
    before = scaler.transform(calibration_data[0])
    temperature, n_classes = scaler.temperature_, scaler.n_classes_
    with pytest.raises(ValueError):
        if operation == "fit":
            scaler.fit(np.asarray(logits), np.array([0]))
        else:
            scaler.transform(np.asarray(logits))
    assert (scaler.temperature_, scaler.n_classes_) == (temperature, n_classes)
    np.testing.assert_array_equal(scaler.transform(calibration_data[0]), before)


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
def test_rejects_invalid_labels_without_fitting(labels) -> None:
    scaler = MulticlassTemperatureScaler()
    with pytest.raises(ValueError):
        scaler.fit(np.array([[1, 2, 3], [3, 2, 1]]), np.asarray(labels))
    assert scaler.temperature_ is None and scaler.n_classes_ is None
    with pytest.raises(RuntimeError, match="must be fitted"):
        scaler.transform(np.ones((2, 3)))


def test_transform_requires_matching_number_of_class_columns(calibration_data) -> None:
    scaler = MulticlassTemperatureScaler().fit(*calibration_data)
    with pytest.raises(ValueError, match="same number of class columns"):
        scaler.transform(np.ones((2, 4)))


def test_invalid_labels_preserve_previous_predictions(calibration_data) -> None:
    scaler = MulticlassTemperatureScaler().fit(*calibration_data)
    before = scaler.transform(calibration_data[0])
    with pytest.raises(ValueError, match="integer class indices"):
        scaler.fit(np.ones((2, 4)), np.array([0, 1.5]))
    assert scaler.n_classes_ == 3
    np.testing.assert_array_equal(scaler.transform(calibration_data[0]), before)


@pytest.mark.parametrize(
    "result",
    [
        OptimizeResult(success=False),
        OptimizeResult(success=True, x=np.nan, fun=0.1),
        OptimizeResult(success=True, x=0.0, fun=np.inf),
        OptimizeResult(success=True, x=4.0, fun=0.1),
    ],
)
@pytest.mark.parametrize("already_fitted", [False, True])
def test_optimizer_failures_preserve_fitted_state(
    monkeypatch: pytest.MonkeyPatch,
    calibration_data,
    result,
    already_fitted: bool,
) -> None:
    scaler = MulticlassTemperatureScaler()
    if already_fitted:
        scaler.fit(*calibration_data)
    before = (scaler.temperature_, scaler.n_classes_)
    monkeypatch.setattr(multiclass, "minimize_scalar", lambda *args, **kwargs: result)
    with pytest.raises(RuntimeError, match="optimisation failed"):
        scaler.fit(np.ones((2, 4)), np.array([0, 1]))
    assert (scaler.temperature_, scaler.n_classes_) == before


def test_repeated_fits_are_reproducible_and_can_change_class_count(calibration_data) -> None:
    scaler = MulticlassTemperatureScaler().fit(*calibration_data)
    before = scaler.transform(calibration_data[0])
    scaler.fit(*calibration_data)
    np.testing.assert_array_equal(scaler.transform(calibration_data[0]), before)
    scaler.fit(np.zeros((5, 4)), np.arange(5) % 4)
    assert scaler.n_classes_ == 4 and scaler.temperature_ == 1.0
    np.testing.assert_allclose(scaler.transform(np.zeros((2, 4))), 0.25)


def test_example_runs_offline_from_another_directory(tmp_path: Path) -> None:
    example = Path(__file__).resolve().parents[1] / "examples/run_multiclass.py"
    result = subprocess.run(
        [sys.executable, str(example)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    metrics = json.loads(result.stdout)
    assert (metrics["train_rows"], metrics["calibration_rows"], metrics["test_rows"]) == (
        1440,
        480,
        480,
    )
    assert metrics["temperature"] > 1.0
    assert metrics["test_log_loss_after"] < metrics["test_log_loss_before"]
    assert metrics["test_accuracy_after"] == metrics["test_accuracy_before"]
