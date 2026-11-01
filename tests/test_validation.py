"""Check that array conversion cannot turn nonnumeric data into valid inputs."""

from decimal import Decimal

import numpy as np
import pytest

from calibrated_ml_lab._validation import as_vector


@pytest.mark.parametrize(
    "values",
    [
        np.array(["0", "1"]),
        np.array([b"0", b"1"]),
        np.array([0, "1"], dtype=object),
        np.array(["0", "1"], dtype=object),
        np.array([None], dtype=object),
        np.array([0.5 + 1j], dtype=object),
        np.array(["1970-01-01"], dtype="datetime64[D]"),
        np.array([1], dtype="timedelta64[D]"),
        np.array([np.timedelta64(1, "D")], dtype=object),
    ],
)
def test_nonnumeric_arrays_are_rejected_before_conversion(values: np.ndarray) -> None:
    with pytest.raises(ValueError, match="real numeric"):
        as_vector(values, "values")


@pytest.mark.parametrize(
    "values,expected",
    [
        (np.array([0, 1]), [0.0, 1.0]),
        (np.array([0.25, 0.75]), [0.25, 0.75]),
        (np.array([True, False]), [1.0, 0.0]),
        (np.array([np.int64(0), np.float64(1)], dtype=object), [0.0, 1.0]),
        (np.array([Decimal("0.25"), Decimal("0.75")], dtype=object), [0.25, 0.75]),
        (np.array([True, np.bool_(False)], dtype=object), [1.0, 0.0]),
    ],
)
def test_real_numeric_and_boolean_arrays_remain_supported(
    values: np.ndarray, expected: list[float]
) -> None:
    np.testing.assert_array_equal(as_vector(values, "values"), expected)
