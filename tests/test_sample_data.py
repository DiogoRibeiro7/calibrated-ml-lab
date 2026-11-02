from pathlib import Path

import pytest

from calibrated_ml_lab.sample_data import (
    generate_classification_data,
    generate_regression_data,
    write_sample_data,
)


def test_generate_classification_data_has_target():
    frame = generate_classification_data(rows=100, seed=1)
    assert "label" in frame.columns
    assert len(frame) == 100


def test_generate_regression_data_has_target():
    frame = generate_regression_data(rows=100, seed=1)
    assert "y" in frame.columns
    assert len(frame) == 100


def test_write_sample_data(tmp_path: Path):
    classification_path, regression_path = write_sample_data(tmp_path, rows=100, seed=1)
    assert classification_path.exists()
    assert regression_path.exists()


@pytest.mark.parametrize("rows,seed", [(0, 42), (100, -1)])
def test_invalid_sample_parameters_do_not_create_output(tmp_path: Path, rows: int, seed: int):
    output = tmp_path / "output"
    with pytest.raises(ValueError):
        write_sample_data(output, rows=rows, seed=seed)
    assert not output.exists()
