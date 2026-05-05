from pathlib import Path

from calibrated_ml_lab.sample_data import generate_classification_data, generate_regression_data, write_sample_data


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
