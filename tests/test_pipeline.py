"""Guard the boundaries between fitting, selection, and held-out evaluation."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from calibrated_ml_lab import pipeline
from calibrated_ml_lab.sample_data import generate_classification_data


@pytest.fixture(autouse=True)
def skip_plots(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep these pipeline tests focused on data flow; CLI tests exercise plots."""
    for name in (
        "plot_reliability_diagram",
        "plot_probability_histogram",
        "plot_threshold_tradeoff",
    ):
        monkeypatch.setattr(pipeline, name, lambda *args, **kwargs: None)


@pytest.mark.parametrize("rows", [1000, 1003])
def test_splits_are_disjoint_exhaustive_and_seeded(rows: int) -> None:
    frame = pd.DataFrame({"row_id": np.arange(rows), "label": np.arange(rows) % 2})
    frame.index = np.zeros(rows, dtype=int)  # Input indices need not identify unique rows.
    splits = pipeline._split_classification_data(frame, "label")
    repeated = pipeline._split_classification_data(frame, "label")
    changed = pipeline._split_classification_data(frame, "label", seed=7)
    row_sets = [set(part["row_id"]) for part in splits.values()]
    assert set.union(*row_sets) == set(range(rows))
    assert all(left.isdisjoint(right) for left, right in combinations(row_sets, 2))
    for name, fraction in zip(splits, [0.5, 0.1, 0.1, 0.1, 0.2], strict=True):
        assert abs(len(splits[name]) - rows * fraction) <= 2
        pd.testing.assert_frame_equal(splits[name], repeated[name])
    assert set(splits["test"]["row_id"]) != set(changed["test"]["row_id"])


@pytest.mark.parametrize("labels", [[0] * 100, [0, 2] * 50, [0, np.nan] * 50, [0.1, 1] * 50])
def test_split_rejects_invalid_binary_targets(labels: list[float]) -> None:
    with pytest.raises(ValueError, match="binary labels"):
        pipeline._split_classification_data(pd.DataFrame({"label": labels}), "label")


def test_split_rejects_insufficient_data_without_reusing_rows() -> None:
    with pytest.raises(ValueError, match="provide more rows"):
        pipeline._split_classification_data(pd.DataFrame({"label": [0, 1]}), "label")
    with pytest.raises(ValueError, match="each need both classes|provide more rows"):
        pipeline._split_classification_data(pd.DataFrame({"label": [0] * 399 + [1]}), "label")


def test_small_calibration_split_returns_both_classes(tmp_path: Path) -> None:
    frame = pd.DataFrame({"feature": np.arange(60), "label": [0, 1] * 30})
    outputs = pipeline.run_classification_calibration(frame, "label", tmp_path)
    summary = outputs["conformal_classification"].iloc[0]
    assert summary["qhat"] == np.inf
    assert summary["coverage"] == 1.0
    assert summary["average_set_size"] == 2.0
    assert summary["singleton_rate"] == 0.0
    scored = pd.read_csv(tmp_path / "scored_classification_test.csv")
    assert scored["conformal_include_0"].all() and scored["conformal_include_1"].all()


def test_small_regression_calibration_exports_unbounded_intervals(tmp_path: Path) -> None:
    frame = pd.DataFrame({"feature": np.arange(20), "y": np.arange(20) / 10})
    summary = pipeline.run_regression_conformal(frame, "y", tmp_path).iloc[0]
    assert summary["qhat"] == summary["average_width"] == np.inf
    assert summary["coverage"] == 1.0
    scored = pd.read_csv(tmp_path / "scored_regression_test.csv")
    assert np.isneginf(scored["interval_lower"]).all()
    assert np.isposinf(scored["interval_upper"]).all()
    assert (tmp_path / "figures/conformal_regression_intervals.png").is_file()


@pytest.mark.parametrize("changed_split", ["test", "conformal"])
def test_held_out_labels_cannot_select_or_fit_candidates(
    tmp_path: Path, changed_split: str
) -> None:
    frame = generate_classification_data(rows=400)
    splits = pipeline._split_classification_data(frame, "label")
    modified = frame.copy()
    held_out_indices = splits[changed_split].index
    modified.loc[held_out_indices, "label"] = (
        0 if changed_split == "test" else 1 - modified.loc[held_out_indices, "label"]
    )
    changed_splits = pipeline._split_classification_data(modified, "label")
    for name in ("train", "calibrator_fit", "selection"):
        pd.testing.assert_frame_equal(splits[name], changed_splits[name])

    original = pipeline.run_classification_calibration(frame, "label", tmp_path / "original")
    changed = pipeline.run_classification_calibration(modified, "label", tmp_path / "changed")
    pd.testing.assert_frame_equal(original["selection_metrics"], changed["selection_metrics"])
    original_scored = pd.read_csv(tmp_path / "original/scored_classification_test.csv")
    changed_scored = pd.read_csv(tmp_path / "changed/scored_classification_test.csv")
    np.testing.assert_allclose(
        original_scored.filter(like="probability_"), changed_scored.filter(like="probability_")
    )
    if changed_split == "test":
        assert original["conformal_classification"]["qhat"].iloc[0] == pytest.approx(
            changed["conformal_classification"]["qhat"].iloc[0]
        )
        pd.testing.assert_frame_equal(
            original_scored.filter(like="conformal_"), changed_scored.filter(like="conformal_")
        )
        assert not original["metrics"].equals(changed["metrics"])
        assert changed["metrics"]["roc_auc"].isna().all()
    else:
        pd.testing.assert_frame_equal(original["metrics"], changed["metrics"])
        # The conformal threshold may change, but tied scores can also leave it unchanged.


@pytest.mark.parametrize("tie", [False, True])
def test_validation_winner_and_stage_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tie: bool
) -> None:
    # Deliberately make Platt best on validation and uncalibrated best on test.
    names = ["train", "calibrator_fit", "selection", "conformal", "test"]
    splits = {
        name: pd.DataFrame({"row_id": np.arange(i * 20, (i + 1) * 20), "label": [0, 1] * 10})
        for i, name in enumerate(names)
    }
    splits["selection"]["label"] = [1] * 16 + [0] * 4
    splits["test"]["label"] = [0] * 16 + [1] * 4
    frame = pd.concat(splits.values(), ignore_index=True)
    monkeypatch.setattr(pipeline, "_split_classification_data", lambda *args, **kwargs: splits)
    observed: dict[str, list[int]] = {}
    last_predicted: list[int] = []

    class Model:
        def fit(self, features: pd.DataFrame, labels: pd.Series) -> Model:
            observed["train"] = features["row_id"].tolist()
            return self

        def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
            last_predicted[:] = features["row_id"].tolist()
            return np.tile([0.5, 0.5] if tie else [0.9, 0.1], (len(features), 1))

    class Calibrator:
        def __init__(self, name: str, probability: float) -> None:
            self.name = name
            self.probability = 0.5 if tie else probability

        def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> Calibrator:
            assert self.name not in observed, "Calibrators must not be refitted after selection"
            observed[self.name] = last_predicted.copy()
            return self

        def transform(self, probabilities: np.ndarray) -> np.ndarray:
            return np.full(len(probabilities), self.probability)

    monkeypatch.setattr(pipeline, "build_classification_pipeline", lambda spec: Model())
    monkeypatch.setattr(pipeline, "PlattScaler", lambda: Calibrator("platt", 0.9))
    monkeypatch.setattr(pipeline, "IsotonicCalibrator", lambda: Calibrator("isotonic", 0.5))
    monkeypatch.setattr(pipeline, "TemperatureScaler", lambda: Calibrator("temperature", 0.4))
    original_threshold = pipeline.conformal_classification_threshold

    def record_threshold(labels: np.ndarray, probabilities: np.ndarray, alpha: float) -> float:
        observed["conformal"] = last_predicted.copy()
        np.testing.assert_array_equal(labels, splits["conformal"]["label"])
        np.testing.assert_allclose(probabilities, 0.5 if tie else 0.9)
        return original_threshold(labels, probabilities, alpha)

    monkeypatch.setattr(pipeline, "conformal_classification_threshold", record_threshold)
    outputs = pipeline.run_classification_calibration(frame, "label", tmp_path)
    assert observed["train"] == splits["train"]["row_id"].tolist()
    for name in ("platt", "isotonic", "temperature"):
        assert observed[name] == splits["calibrator_fit"]["row_id"].tolist()
    assert observed["conformal"] == splits["conformal"]["row_id"].tolist()
    assert outputs["metrics"].sort_values("ece").iloc[0]["model"] == "uncalibrated"
    selected = "uncalibrated" if tie else "platt"
    for key in ("selection_metrics", "metrics"):
        assert outputs[key].loc[outputs[key]["selected"], "model"].tolist() == [selected]
    assert outputs["conformal_classification"].iloc[0]["method"] == f"split_conformal_{selected}"
    report_path = tmp_path / "report.md"
    pipeline.write_report(
        report_path,
        metrics=outputs["metrics"],
        conformal_classification=outputs["conformal_classification"],
        conformal_regression=pd.DataFrame(),
        threshold_metrics=outputs["threshold_metrics"],
        selection_metrics=outputs["selection_metrics"],
        classification_splits=outputs["classification_splits"],
    )
    assert f"Selected method: **{selected}**" in report_path.read_text(encoding="utf-8")
