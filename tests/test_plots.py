"""Keep unbounded conformal intervals visible in exported figures."""

from pathlib import Path

import numpy as np
import pytest
from matplotlib.figure import Figure

from calibrated_ml_lab.visualization.plots import plot_conformal_regression_intervals


@pytest.mark.parametrize("unbounded_rows", [0, 1, 3])
def test_regression_plot_labels_unbounded_intervals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unbounded_rows: int
) -> None:
    annotations: list[str] = []
    legend_labels: list[str] = []
    savefig = Figure.savefig

    def capture_figure(figure: Figure, path: Path, dpi: int) -> None:
        annotations.extend(text.get_text() for text in figure.axes[0].texts)
        legend_labels.extend(text.get_text() for text in figure.axes[0].get_legend().get_texts())
        savefig(figure, path, dpi=dpi)

    monkeypatch.setattr(Figure, "savefig", capture_figure)
    prediction = np.array([1.0, 2.0, 3.0])
    lower, upper = prediction - 1, prediction + 1
    lower[:unbounded_rows], upper[:unbounded_rows] = -np.inf, np.inf
    output = tmp_path / "intervals.png"
    plot_conformal_regression_intervals(prediction, prediction, lower, upper, output)
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    if unbounded_rows:
        assert annotations == [f"Unbounded intervals: {unbounded_rows} of 3 shown"]
        assert "Predictions (unbounded intervals)" in legend_labels
    else:
        assert not annotations
