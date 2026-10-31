"""End-to-end calibration and uncertainty workflow."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from calibrated_ml_lab.calibration.metrics import calibration_bins, probabilistic_metrics
from calibrated_ml_lab.calibration.scaling import IsotonicCalibrator, PlattScaler, TemperatureScaler
from calibrated_ml_lab.conformal.classification import (
    classification_set_coverage,
    conformal_classification_threshold,
    conformal_prediction_sets,
)
from calibrated_ml_lab.conformal.regression import (
    conformal_intervals,
    conformal_residual_quantile,
    regression_interval_metrics,
)
from calibrated_ml_lab.models.baseline import (
    build_classification_pipeline,
    build_regression_pipeline,
    infer_feature_spec,
)
from calibrated_ml_lab.reporting import write_report
from calibrated_ml_lab.uncertainty.thresholds import threshold_metrics
from calibrated_ml_lab.visualization.plots import (
    plot_conformal_regression_intervals,
    plot_probability_histogram,
    plot_reliability_diagram,
    plot_threshold_tradeoff,
)


def _split_classification_data(
    frame: pd.DataFrame, target: str, seed: int = 42
) -> dict[str, pd.DataFrame]:
    """Allocate roughly 50/10/10/10/20% to five disjoint classification stages."""

    if target not in frame:
        raise ValueError(f"target column '{target}' not found")
    if not frame[target].isin([0, 1]).all() or frame[target].nunique() != 2:
        raise ValueError("classification target must contain both binary labels 0 and 1")

    try:
        # Reserve evaluation rows without consulting their labels. Only development
        # data is stratified; conformal calibration and test remain random holdouts.
        development, held_out = train_test_split(frame, test_size=0.3, random_state=seed)
        conformal, test = train_test_split(held_out, test_size=2 / 3, random_state=seed)
        train, tuning = train_test_split(
            development, test_size=2 / 7, random_state=seed, stratify=development[target]
        )
        calibrator_fit, selection = train_test_split(
            tuning, test_size=0.5, random_state=seed, stratify=tuning[target]
        )
    except ValueError as error:
        raise ValueError(
            "Not enough classification data for five disjoint splits; provide more rows "
            "of each class for training, calibrator fitting, and method selection."
        ) from error

    splits = {
        "train": train,
        "calibrator_fit": calibrator_fit,
        "selection": selection,
        "conformal": conformal,
        "test": test,
    }
    if any(
        splits[name][target].nunique() != 2 for name in ("train", "calibrator_fit", "selection")
    ):
        raise ValueError(
            "training, calibrator fitting, and method selection each need both classes"
        )
    if len(conformal) < 9:
        raise ValueError("alpha=0.1 needs at least 9 conformal calibration rows; provide more data")
    return splits


def _probability_metrics(
    labels: np.ndarray, probabilities: dict[str, np.ndarray], n_bins: int
) -> pd.DataFrame:
    """Evaluate each candidate without choosing a method."""
    return pd.DataFrame(
        [
            {"model": name, **probabilistic_metrics(labels, values, n_bins=n_bins)}
            for name, values in probabilities.items()
        ]
    )


def run_classification_calibration(
    frame: pd.DataFrame, target: str, output_dir: Path, n_bins: int = 10, seed: int = 42
) -> dict[str, pd.DataFrame]:
    """Run classification calibration and conformal prediction workflow."""

    splits = _split_classification_data(frame, target=target, seed=seed)
    train, calibrator_fit, selection, conformal, test = (
        splits[name] for name in ("train", "calibrator_fit", "selection", "conformal", "test")
    )
    spec = infer_feature_spec(train, target=target)
    feature_columns = spec.numeric_features + spec.categorical_features

    model = build_classification_pipeline(spec)
    model.fit(train[feature_columns], train[target])

    fit_prob = model.predict_proba(calibrator_fit[feature_columns])[:, 1]
    calibrators: dict[str, PlattScaler | IsotonicCalibrator | TemperatureScaler] = {
        "platt": PlattScaler(),
        "isotonic": IsotonicCalibrator(),
        "temperature": TemperatureScaler(),
    }
    for calibrator in calibrators.values():
        calibrator.fit(fit_prob, calibrator_fit[target].to_numpy())

    def candidate_probabilities(data: pd.DataFrame) -> dict[str, np.ndarray]:
        """Apply the frozen model and calibrators to another split."""
        raw = model.predict_proba(data[feature_columns])[:, 1]
        return {"uncalibrated": raw, **{name: c.transform(raw) for name, c in calibrators.items()}}

    selection_metrics = _probability_metrics(
        selection[target].to_numpy(), candidate_probabilities(selection), n_bins
    )
    # A stable sort resolves exact ties in candidate order, favouring uncalibrated.
    selected_name = str(selection_metrics.sort_values("ece", kind="stable").iloc[0]["model"])
    selection_metrics["selected"] = selection_metrics["model"] == selected_name

    conformal_prob = candidate_probabilities(conformal)[selected_name]
    qhat = conformal_classification_threshold(
        conformal[target].to_numpy(), conformal_prob, alpha=0.1
    )

    # Freeze selection and qhat before inspecting any final test labels.
    probability_sets = candidate_probabilities(test)
    selected_test_prob = probability_sets[selected_name]

    y_test = test[target].to_numpy()
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    split_summary = pd.DataFrame(
        [{"split": name, "rows": len(data), "seed": seed} for name, data in splits.items()]
    )
    split_summary.to_csv(tables_dir / "classification_splits.csv", index=False)
    selection_metrics.to_csv(tables_dir / "selection_metrics.csv", index=False)
    for name, probabilities in probability_sets.items():
        calibration_bins(y_test, probabilities, n_bins=n_bins).to_csv(
            tables_dir / f"calibration_bins_{name}.csv", index=False
        )

    metrics = _probability_metrics(y_test, probability_sets, n_bins)
    metrics["selected"] = metrics["model"] == selected_name
    metrics.to_csv(tables_dir / "calibration_metrics.csv", index=False)

    prediction_sets = conformal_prediction_sets(selected_test_prob, threshold=qhat)
    conformal_summary = pd.DataFrame(
        [
            {
                "method": f"split_conformal_{selected_name}",
                "alpha": 0.1,
                "qhat": qhat,
                **classification_set_coverage(y_test, prediction_sets),
            }
        ]
    )
    conformal_summary.to_csv(tables_dir / "conformal_classification_summary.csv", index=False)

    thresholds = threshold_metrics(y_test, selected_test_prob)
    thresholds.to_csv(tables_dir / "threshold_metrics.csv", index=False)

    scored = test.copy()
    for name, probabilities in probability_sets.items():
        scored[f"probability_{name}"] = probabilities
    scored["conformal_include_0"] = prediction_sets["include_0"].to_numpy()
    scored["conformal_include_1"] = prediction_sets["include_1"].to_numpy()
    scored["conformal_set_size"] = prediction_sets["set_size"].to_numpy()
    scored.to_csv(output_dir / "scored_classification_test.csv", index=False)

    plot_reliability_diagram(
        y_test, probability_sets, figures_dir / "reliability_diagram.png", n_bins=n_bins
    )
    plot_probability_histogram(probability_sets, figures_dir / "probability_histogram.png")
    plot_threshold_tradeoff(thresholds, figures_dir / "threshold_tradeoff.png")

    return {
        "selection_metrics": selection_metrics,
        "classification_splits": split_summary,
        "metrics": metrics,
        "conformal_classification": conformal_summary,
        "threshold_metrics": thresholds,
    }


def run_regression_conformal(frame: pd.DataFrame, target: str, output_dir: Path) -> pd.DataFrame:
    """Run split-conformal regression intervals."""

    train, temp = train_test_split(frame, test_size=0.4, random_state=42)
    calibration, test = train_test_split(temp, test_size=0.5, random_state=42)
    feature_columns = [
        col for col in frame.columns if col != target and pd.api.types.is_numeric_dtype(frame[col])
    ]

    model = build_regression_pipeline(feature_columns)
    model.fit(train[feature_columns], train[target])
    cal_pred = model.predict(calibration[feature_columns])
    test_pred = model.predict(test[feature_columns])
    qhat = conformal_residual_quantile(calibration[target].to_numpy(), cal_pred, alpha=0.1)
    lower, upper = conformal_intervals(test_pred, qhat)
    summary = pd.DataFrame(
        [
            {
                "method": "split_conformal_regression",
                "alpha": 0.1,
                "qhat": qhat,
                **regression_interval_metrics(test[target].to_numpy(), lower, upper),
            }
        ]
    )

    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(tables_dir / "conformal_regression_summary.csv", index=False)

    scored = test.copy()
    scored["prediction"] = test_pred
    scored["interval_lower"] = lower
    scored["interval_upper"] = upper
    scored.to_csv(output_dir / "scored_regression_test.csv", index=False)

    plot_conformal_regression_intervals(
        test[target].to_numpy(),
        test_pred,
        lower,
        upper,
        figures_dir / "conformal_regression_intervals.png",
    )
    return summary


def run_workflow(
    classification_path: Path,
    regression_path: Path,
    output_dir: Path,
    target: str = "label",
    regression_target: str = "y",
) -> None:
    """Run the full calibration and uncertainty workflow."""

    output_dir.mkdir(parents=True, exist_ok=True)
    classification = pd.read_csv(classification_path)
    regression = pd.read_csv(regression_path)

    class_outputs = run_classification_calibration(
        classification, target=target, output_dir=output_dir
    )
    regression_summary = run_regression_conformal(
        regression, target=regression_target, output_dir=output_dir
    )

    write_report(
        output_dir / "calibration_report.md",
        metrics=class_outputs["metrics"],
        conformal_classification=class_outputs["conformal_classification"],
        conformal_regression=regression_summary,
        threshold_metrics=class_outputs["threshold_metrics"],
        selection_metrics=class_outputs["selection_metrics"],
        classification_splits=class_outputs["classification_splits"],
    )
