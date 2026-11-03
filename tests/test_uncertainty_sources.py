"""Check the teaching example's posterior moments, comparisons, and exported diagnostics."""

from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/run_uncertainty_sources.py"


@pytest.fixture(scope="module")
def example() -> dict:
    return runpy.run_path(str(EXAMPLE))


def test_posterior_predictions_match_hand_calculated_two_point_problem(example) -> None:
    # Phi.T @ Phi = 2 I; prior precision = I/4; noise variance = 4.
    # Posterior covariance = (4/3) I and mean = [4/3, 2/3].
    table = example["_predictive_components"](
        np.array([-1.0, 1]),
        np.array([1.0, 3]),
        np.array([0.0, 1, 2]),
        noise_std=2.0,
        prior_variance=4.0,
    )
    np.testing.assert_allclose(table["mean_prediction"], [4 / 3, 2, 8 / 3])
    np.testing.assert_allclose(table["epistemic_variance"], [4 / 3, 8 / 3, 20 / 3])
    np.testing.assert_allclose(table["aleatoric_variance"], 4)
    np.testing.assert_allclose(table["total_variance"], [16 / 3, 20 / 3, 32 / 3])


def test_unobserved_slope_retains_prior_uncertainty_with_constant_training_inputs(example) -> None:
    table = example["_predictive_components"](
        np.zeros(2), np.array([1.0, 3]), np.array([0.0, 1]), noise_std=1, prior_variance=4
    )
    np.testing.assert_allclose(table["mean_prediction"], 16 / 9)
    np.testing.assert_allclose(table["epistemic_variance"], [4 / 9, 4 / 9 + 4])


def test_known_noise_posterior_variance_depends_on_design_not_observed_targets(example) -> None:
    predict = example["_predictive_components"]
    x = np.array([-1.0, 0, 1])
    first = predict(x, x, x, noise_std=0.4)
    second = predict(x, 5 + 3 * x, x, noise_std=0.4)
    assert not np.allclose(first["mean_prediction"], second["mean_prediction"])
    pd.testing.assert_frame_equal(
        first.drop(columns="mean_prediction"), second.drop(columns="mean_prediction")
    )


def test_more_training_reduces_epistemic_variance_and_noise_increases_both_sources(example) -> None:
    _, grid, _ = example["_run_experiment"]()
    for noise_std, rows in grid.groupby("noise_std"):
        epistemic = rows.pivot(index="x", columns="train_rows", values="epistemic_variance")
        assert (epistemic[20] > epistemic[80]).all()
        assert (epistemic[80] > epistemic[320]).all()
        np.testing.assert_allclose(rows["aleatoric_variance"], noise_std**2)
        for _, scenario in rows.groupby("train_rows"):
            near = scenario.loc[scenario["x"].abs() < 1, "epistemic_variance"].mean()
            far = scenario.loc[scenario["x"].abs() > 4, "epistemic_variance"].mean()
            assert far > near
    noise_comparison = grid.pivot(
        index=["x", "train_rows"], columns="noise_std", values="epistemic_variance"
    )
    assert (noise_comparison[1.2] > noise_comparison[0.4]).all()


def test_scenarios_are_reproducible_and_share_test_inputs_and_standardized_noise(example) -> None:
    first = example["_run_experiment"](seed=42)
    second = example["_run_experiment"](seed=42)
    for left, right in zip(first, second, strict=True):
        pd.testing.assert_frame_equal(left, right)
    _, _, test = first
    scenarios = list(test.groupby(["noise_std", "train_rows"]))
    reference = scenarios[0][1]
    reference_noise = (reference["y_true"] - 1 - 2 * reference["x"]) / 0.4
    for (noise_std, _), rows in scenarios:
        np.testing.assert_array_equal(rows["x"], reference["x"])
        np.testing.assert_allclose(
            (rows["y_true"] - 1 - 2 * rows["x"]) / noise_std, reference_noise
        )
    different = example["_run_experiment"](seed=43)[2]
    assert not np.array_equal(test["x"], different["x"])


def test_offline_exports_retain_predictions_and_recomputable_scenario_diagnostics(tmp_path) -> None:
    result = subprocess.run(
        [sys.executable, str(EXAMPLE), "--output-dir", str(tmp_path / "output with spaces")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["seed"] == 42 and report["prior_variance"] == 25
    assert report["train_sizes"] == [20, 80, 320]
    assert report["known_noise_stds"] == [0.4, 1.2]
    assert report["nominal_coverage"] == 0.9
    summary = pd.read_csv(report["summary_path"])
    grid = pd.read_csv(report["grid_path"])
    test = pd.read_csv(report["test_path"])
    assert len(summary) == 6 and len(grid) == 6 * 401 and len(test) == 6 * 1000
    pd.testing.assert_frame_equal(summary, pd.DataFrame(report["scenarios"]))
    for table in [grid, test]:
        assert np.isfinite(table.select_dtypes(include="number")).all().all()
        assert (table["epistemic_variance"] > 0).all()
        np.testing.assert_allclose(table["aleatoric_variance"], table["noise_std"] ** 2)
        np.testing.assert_allclose(
            table["total_variance"], table["epistemic_variance"] + table["aleatoric_variance"]
        )
        # Central 90% Gaussian intervals must include observation noise.
        half_width = 1.6448536269514722 * np.sqrt(table["total_variance"])
        np.testing.assert_allclose(table["lower"], table["mean_prediction"] - half_width)
        np.testing.assert_allclose(table["upper"], table["mean_prediction"] + half_width)
    for row in summary.itertuples():
        rows = test[(test["noise_std"] == row.noise_std) & (test["train_rows"] == row.train_rows)]
        covered = (rows["lower"] <= rows["y_true"]) & (rows["y_true"] <= rows["upper"])
        np.testing.assert_array_equal(rows["covered"], covered)
        assert row.test_rows == len(rows)
        assert row.nominal_coverage == 0.9
        assert row.test_coverage == pytest.approx(covered.mean())
        assert row.test_mean_squared_error == pytest.approx(
            np.mean((rows["y_true"] - rows["mean_prediction"]) ** 2)
        )
        assert row.test_mean_interval_width == pytest.approx((rows["upper"] - rows["lower"]).mean())
        for source in ["epistemic", "aleatoric", "total"]:
            assert getattr(row, f"test_mean_{source}_variance") == pytest.approx(
                rows[f"{source}_variance"].mean()
            )
    for _, rows in grid.groupby(["noise_std", "train_rows"]):
        np.testing.assert_allclose(rows["x"], np.linspace(-5, 5, 401))
    assert Path(report["plot_path"]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_plot_keeps_all_scenarios_visible_and_closes_its_figure(
    example, monkeypatch, tmp_path
) -> None:
    pyplot = example["plt"]
    original_subplots = pyplot.subplots
    figures = []

    def capture_subplots(*args, **kwargs):
        figure, axes = original_subplots(*args, **kwargs)
        figures.append(figure)
        return figure, axes

    monkeypatch.setattr(pyplot, "subplots", capture_subplots)
    summary, grid, _ = example["_run_experiment"]()
    example["_plot_components"](summary, grid, tmp_path / "components.png")
    assert len(figures) == 1
    figure = figures[0]
    assert not pyplot.fignum_exists(figure.number)
    assert len(figure.axes) == 4
    for axis in figure.axes:
        low, high = axis.get_ylim()
        for line in axis.get_lines():
            values = np.asarray(line.get_ydata())
            assert (values >= low).all() and (values <= high).all()
    assert figure.axes[2].get_yscale() == figure.axes[3].get_yscale() == "log"
    assert figure.axes[2].get_ylim() == figure.axes[3].get_ylim()
