"""Reject misleading binary evaluation inputs at each public entry point."""

from collections.abc import Callable

import numpy as np
import pytest

from calibrated_ml_lab.calibration.metrics import calibration_bins, probabilistic_metrics
from calibrated_ml_lab.uncertainty.thresholds import threshold_metrics


@pytest.mark.parametrize("evaluate", [calibration_bins, probabilistic_metrics, threshold_metrics])
@pytest.mark.parametrize(
    "labels,probabilities",
    [
        ([0.5, 1.5], [0.2, 0.8]),
        ([0, 2], [0.2, 0.8]),
        ([-1, 1], [0.2, 0.8]),
        ([0, np.nan], [0.2, 0.8]),
        ([0, np.inf], [0.2, 0.8]),
        ([0, 1], [np.nan, 0.8]),
        ([0, 1], [0.2, np.inf]),
        ([0, 1], [-0.1, 1.1]),
        ([0, 1], [0.2 + 1j, 0.8]),
        ([], []),
        (0, 0.5),
        ([[0], [1]], [[0.2], [0.8]]),
        ([0, 1], [0.2]),
    ],
)
def test_binary_evaluation_rejects_invalid_inputs(
    evaluate: Callable, labels: object, probabilities: object
) -> None:
    with pytest.raises(ValueError):
        evaluate(np.asarray(labels), np.asarray(probabilities))
