"""End-to-end calibration and uncertainty workflow."""

from __future__ import annotations

from pathlib import Path

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


def _split_three_way(
    frame: pd.DataFrame, target: str, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train, calibration, and test sets."""

    train, temp = train_test_split(frame, test_size=0.4, random_state=seed, stratify=frame[target])
    calibration, test = train_test_split(
        temp, test_size=0.5, random_state=seed, stratify=temp[target]
    )
    return (
        train.reset_index(drop=True),
        calibration.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def run_classification_calibration(
    frame: pd.DataFrame, target: str, output_dir: Path, n_bins: int = 10
) -> dict[str, pd.DataFrame]:
    """Run classification calibration and conformal prediction workflow."""

    train, calibration, test = _split_three_way(frame, target=target)
    spec = infer_feature_spec(frame, target=target)
    feature_columns = spec.numeric_features + spec.categorical_features

    model = build_classification_pipeline(spec)
    model.fit(train[feature_columns], train[target])

    cal_prob = model.predict_proba(calibration[feature_columns])[:, 1]
    test_prob = model.predict_proba(test[feature_columns])[:, 1]

    platt = PlattScaler().fit(cal_prob, calibration[target].to_numpy())
    isotonic = IsotonicCalibrator().fit(cal_prob, calibration[target].to_numpy())
    temperature = TemperatureScaler().fit(cal_prob, calibration[target].to_numpy())

    probability_sets = {
        "uncalibrated": test_prob,
        "platt": platt.transform(test_prob),
        "isotonic": isotonic.transform(test_prob),
        "temperature": temperature.transform(test_prob),
    }

    metrics_rows: list[dict[str, float | str]] = []
    y_test = test[target].to_numpy()
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    for name, probabilities in probability_sets.items():
        row: dict[str, float | str] = {
            "model": name,
            **probabilistic_metrics(y_test, probabilities, n_bins=n_bins),
        }
        metrics_rows.append(row)
        calibration_bins(y_test, probabilities, n_bins=n_bins).to_csv(
            tables_dir / f"calibration_bins_{name}.csv", index=False
        )

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(tables_dir / "calibration_metrics.csv", index=False)

    # Use the best-ECE probabilities for downstream conformal and threshold analysis.
    best_model_name = str(metrics.sort_values("ece").iloc[0]["model"])
    best_test_prob = probability_sets[best_model_name]
    best_cal_prob = {
        "uncalibrated": cal_prob,
        "platt": platt.transform(cal_prob),
        "isotonic": isotonic.transform(cal_prob),
        "temperature": temperature.transform(cal_prob),
    }[best_model_name]

    qhat = conformal_classification_threshold(
        calibration[target].to_numpy(), best_cal_prob, alpha=0.1
    )
    prediction_sets = conformal_prediction_sets(best_test_prob, threshold=qhat)
    conformal_summary = pd.DataFrame(
        [
            {
                "method": f"split_conformal_{best_model_name}",
                "alpha": 0.1,
                "qhat": qhat,
                **classification_set_coverage(y_test, prediction_sets),
            }
        ]
    )
    conformal_summary.to_csv(tables_dir / "conformal_classification_summary.csv", index=False)

    thresholds = threshold_metrics(y_test, best_test_prob)
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
    )
