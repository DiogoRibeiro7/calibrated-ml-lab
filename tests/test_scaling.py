from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from calibrated_ml_lab.calibration import scaling
from calibrated_ml_lab.calibration.scaling import IsotonicCalibrator, PlattScaler, TemperatureScaler

Calibrator = PlattScaler | IsotonicCalibrator | TemperatureScaler


@pytest.fixture(params=[PlattScaler, IsotonicCalibrator, TemperatureScaler])
def calibrator(request: pytest.FixtureRequest) -> Calibrator:
    """Exercise each calibrator's public fit/transform contract."""
    return request.param()


def test_platt_scaler_returns_probabilities():
    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([0.1, 0.2, 0.8, 0.7, 0.3, 0.9])
    scaler = PlattScaler().fit(p, y)
    out = scaler.transform(p)
    assert np.all((out >= 0) & (out <= 1))


def test_isotonic_calibrator_returns_probabilities():
    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([0.1, 0.2, 0.8, 0.7, 0.3, 0.9])
    scaler = IsotonicCalibrator().fit(p, y)
    out = scaler.transform(p)
    assert np.all((out >= 0) & (out <= 1))


def test_temperature_scaler_fits_positive_temperature():
    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([0.1, 0.2, 0.8, 0.7, 0.3, 0.9])
    scaler = TemperatureScaler().fit(p, y)
    assert scaler.temperature_ is not None
    assert scaler.temperature_ > 0


@pytest.mark.parametrize(
    "probabilities,labels",
    [
        ([0.2, 0.8], [0.5, 1.5]),
        ([0.2, 0.8], [0, 2]),
        ([0.2, 0.8], [-1, 1]),
        ([0.2, 0.8], [0, np.nan]),
        ([0.2, 0.8], [0, np.inf]),
        ([0.2, 0.8], [[0], [1]]),
        ([0.2, 0.8], [0]),
        ([], []),
        (0.5, 0),
        ([[0.2], [0.8]], [0, 1]),
        ([-0.1, 1.1], [0, 1]),
        ([0.2, np.nan], [0, 1]),
        ([0.2, np.inf], [0, 1]),
        ([0.2 + 1j, 0.8], [0, 1]),
        ([0.2, 0.8], ["0", "1"]),
        (["0.2", "0.8"], [0, 1]),
        ([0.2, 0.8], np.array([0, "1"], dtype=object)),
    ],
)
def test_fit_rejects_invalid_inputs(
    calibrator: Calibrator, probabilities: object, labels: object
) -> None:
    with pytest.raises(ValueError):
        calibrator.fit(np.asarray(probabilities), np.asarray(labels))
    with pytest.raises(RuntimeError, match="must be fitted"):
        calibrator.transform(np.array([0.5]))


@pytest.mark.parametrize(
    "probabilities",
    [
        [],
        0.5,
        [[0.5]],
        [-0.1],
        [1.1],
        [np.nan],
        [np.inf],
        [0.5 + 1j],
        ["0.5"],
        np.array(["0.5"], dtype=object),
    ],
)
def test_transform_rejects_invalid_probabilities(
    calibrator: Calibrator, probabilities: object
) -> None:
    calibrator.fit(np.array([0.2, 0.8]), np.array([0, 1]))
    with pytest.raises(ValueError):
        calibrator.transform(np.asarray(probabilities))


def test_transform_requires_a_successful_fit(calibrator: Calibrator) -> None:
    with pytest.raises(RuntimeError, match="must be fitted"):
        calibrator.transform(np.array([0.0, 0.5, 1.0]))


def test_failed_refit_preserves_previous_predictions(calibrator: Calibrator) -> None:
    probabilities, labels = np.array([0.2, 0.8]), np.array([0, 1])
    calibrator.fit(probabilities, labels)
    before = calibrator.transform(probabilities)
    with pytest.raises(ValueError):
        calibrator.fit(np.array([0.2, 0.8, 0.5]), labels)
    np.testing.assert_array_equal(calibrator.transform(probabilities), before)


@pytest.mark.parametrize("already_fitted", [False, True])
def test_backend_failure_does_not_replace_fitted_state(
    calibrator: Calibrator, already_fitted: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    probabilities, labels = np.array([0.2, 0.8]), np.array([0, 1])
    if already_fitted:
        calibrator.fit(probabilities, labels)
        before = calibrator.transform(probabilities)

    def fail_fit(*args: object, **kwargs: object) -> None:
        raise RuntimeError("fit failed")

    if isinstance(calibrator, TemperatureScaler):
        monkeypatch.setattr(
            scaling, "minimize_scalar", lambda *args, **kwargs: OptimizeResult(success=False)
        )
    else:
        backend = (
            scaling.LogisticRegression
            if isinstance(calibrator, PlattScaler)
            else scaling.IsotonicRegression
        )
        monkeypatch.setattr(backend, "fit", fail_fit)
    with pytest.raises(RuntimeError):
        calibrator.fit(probabilities, labels)
    if already_fitted:
        np.testing.assert_array_equal(calibrator.transform(probabilities), before)
    else:
        with pytest.raises(RuntimeError, match="must be fitted"):
            calibrator.transform(probabilities)


@pytest.mark.parametrize("probabilities", [[0.0, 5e-13], [1 - 5e-13, 1.0]])
def test_isotonic_preserves_distinct_scores_near_endpoints(probabilities: list[float]) -> None:
    p = np.array(probabilities)
    fitted = IsotonicCalibrator().fit(p, np.array([0, 1]))
    np.testing.assert_allclose(fitted.transform(p), [0.0, 1.0], atol=0)


def test_isotonic_interpolates_and_clips_to_fitted_domain() -> None:
    fitted = IsotonicCalibrator().fit(np.array([0.25, 0.75]), np.array([0, 1]))
    np.testing.assert_allclose(fitted.transform(np.array([0.0, 0.5, 1.0])), [0.0, 0.5, 1.0])


@pytest.mark.parametrize("label", [0, 1])
def test_single_class_behavior_is_explicit(calibrator: Calibrator, label: int) -> None:
    probabilities, labels = np.array([0.2, 0.8]), np.full(2, label)
    if isinstance(calibrator, PlattScaler):
        with pytest.raises(ValueError, match="both classes"):
            calibrator.fit(probabilities, labels)
    else:
        calibrator.fit(probabilities, labels)
        result = calibrator.transform(probabilities)
        assert np.isfinite(result).all() and ((result >= 0) & (result <= 1)).all()
        if isinstance(calibrator, IsotonicCalibrator):
            np.testing.assert_array_equal(result, labels)


def test_valid_fit_and_refit_are_reproducible_and_do_not_mutate_inputs(
    calibrator: Calibrator,
) -> None:
    p, y = np.array([0.0, 0.2, 0.8, 1.0]), np.array([0, 0, 1, 1])
    original_p, original_y = p.copy(), y.copy()
    assert calibrator.fit(p, y) is calibrator
    first = calibrator.transform(p)
    assert first.shape == p.shape
    assert np.isfinite(first).all() and ((first >= 0) & (first <= 1)).all()
    calibrator.fit(p, y)
    np.testing.assert_allclose(calibrator.transform(p), first, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(p, original_p)
    np.testing.assert_array_equal(y, original_y)
    calibrator.fit(p, 1 - y)
    fresh = type(calibrator)().fit(p, 1 - y)
    updated = calibrator.transform(p)
    np.testing.assert_allclose(updated, fresh.transform(p), rtol=0, atol=1e-12)
    assert not np.allclose(updated, first)
