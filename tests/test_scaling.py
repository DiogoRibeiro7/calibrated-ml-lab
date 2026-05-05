import numpy as np

from calibrated_ml_lab.calibration.scaling import IsotonicCalibrator, PlattScaler, TemperatureScaler


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
