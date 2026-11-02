# calibrated-ml-lab

A practical workbench for **probability calibration** and **uncertainty estimation** in machine learning.

This repository focuses on a common production problem: a model can have good discrimination while producing probabilities that are not reliable enough for decisions. The project implements calibration diagnostics, post-hoc calibration methods, conformal prediction, uncertainty-aware reports, and decision-oriented evaluation.

The goal is not to make every model look confident. The goal is to make uncertainty visible, measurable, and useful.

## What this repo demonstrates

- Binary probability calibration.
- Reliability diagrams and calibration tables.
- Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
- Brier score and negative log loss.
- Platt scaling.
- Isotonic regression calibration.
- Temperature scaling.
- Multiclass temperature scaling from logits (Python API).
- Multiclass confidence, top-label, and class-wise calibration diagnostics.
- Split conformal classification.
- Split conformal regression intervals.
- Coverage and interval-width diagnostics.
- Decision-threshold evaluation before and after calibration.
- Markdown reports and plots.
- CLI and notebook workflows.

## Repository layout

```text
calibrated-ml-lab/
├── README.md
├── ROADMAP.md
├── AGENTS.md
├── CITATION.cff
├── LICENSE
├── pyproject.toml
├── Makefile
├── setup.sh
├── docs/
├── notebooks/
├── prompts/
├── examples/
├── src/calibrated_ml_lab/
│   ├── calibration/
│   ├── conformal/
│   ├── models/
│   ├── uncertainty/
│   ├── visualization/
│   ├── sample_data.py
│   ├── pipeline.py
│   ├── reporting.py
│   └── cli.py
├── tests/
└── reports/sample_run/
```

## Quick start

```bash
poetry install --with dev
```

Generate synthetic classification and regression data:

```bash
poetry run calibrated-lab sample-data \
  --output-dir data/raw \
  --rows 5000
```

Run the full workflow:

```bash
poetry run calibrated-lab run \
  --input data/raw/classification.csv \
  --regression-input data/raw/regression.csv \
  --output-dir reports/sample_run \
  --target label \
  --regression-target y
```

Inputs must be readable CSV files; `--output-dir` must name a directory.
`sample-data` requires positive `--rows` and a nonnegative `--seed`.
Invalid command options exit with status 2. Expected data or filesystem failures
print an error to stderr and exit with status 1, without a traceback or success
message. See [the data model](docs/data_model.md) for target requirements.

The full workflow generates its report, tables, and plots in temporary storage.
It copies them to `--output-dir` only after every computation succeeds, preserving
an existing report if data validation, model fitting, or report generation fails.
A successful run overwrites only files produced by that run; other files,
including artifacts from older versions, are retained. Existing file and directory
permissions are preserved. The final copy is not atomic: a filesystem failure
during copying can leave partial output.
Before copying, the workflow rejects existing symbolic links or Windows reparse
points at the output directory, generated-file destinations, or their parents
within the output directory. Keep the destination unchanged while a run publishes.

## Multiclass calibration

Use `MulticlassTemperatureScaler` from
`calibrated_ml_lab.calibration.multiclass` to fit one temperature on a matrix of
calibration logits and integer class labels. Call `transform` on new logits to
obtain normalized probabilities for every class, keeping the class-column order
used during fitting.

Run the reproducible, offline example:

```bash
poetry run python examples/run_multiclass.py
```

The example uses separate model-training, calibration, and test sets. It deliberately
amplifies logits to simulate overconfidence, then reports temperature, test log loss,
and test accuracy before and after calibration. It also reports confidence ECE,
top-label ECE, and per-class ECE/MCE on the test set. In this example, log loss
improves while the binned calibration errors increase; the output keeps both results
visible. This illustrates the method;
calibration does not guarantee better performance on unseen data.
See [the multiclass API contract](docs/methodology.md#multiclass-temperature-scaling)
for label mapping, numerical limits, and fitting assumptions. The `calibrated-lab run`
reporting workflow currently handles binary classification.

The functions in `calibrated_ml_lab.calibration.multiclass_metrics` accept class-index
labels and a matrix of class probabilities:

- `confidence_calibration_bins`: pooled confidence versus prediction accuracy.
- `top_label_calibration_bins`: confidence versus accuracy within each predicted class.
- `classwise_calibration_errors`: one-versus-rest ECE/MCE for every class, with label counts.
- `multiclass_calibration_metrics`: confidence ECE, top-label ECE, and mean class-wise ECE.

See [the diagnostic definitions](docs/methodology.md#multiclass-calibration-diagnostics)
for bin weights, missing classes, and probability ties.

## Main outputs

```text
reports/sample_run/
├── calibration_report.md
├── tables/
│   ├── calibration_metrics.csv
│   ├── selection_metrics.csv
│   ├── classification_splits.csv
│   ├── calibration_bins_uncalibrated.csv
│   ├── calibration_bins_platt.csv
│   ├── calibration_bins_isotonic.csv
│   ├── calibration_bins_temperature.csv
│   ├── conformal_classification_summary.csv
│   ├── conformal_regression_summary.csv
│   └── threshold_metrics.csv
└── figures/
    ├── reliability_diagram.png
    ├── probability_histogram.png
    ├── conformal_regression_intervals.png
    └── threshold_tradeoff.png
```

## Responsible use

Calibration improves the interpretation of model probabilities, but it does not remove data bias, label noise, confounding, or poor problem framing. A calibrated model can still be wrong, unfair, or unsafe. Use calibration as one part of model risk management, not as a guarantee of correctness.

The classification workflow uses five disjoint splits: roughly 50% for model
training, 10% for calibrator fitting, 10% for method selection, 10% for conformal
calibration, and 20% for final testing. Validation ECE selects the method; final
test metrics never select or refit it. Reports include selection metrics and split
sizes separately from final test metrics. See [methodology](docs/methodology.md)
for splitting assumptions and small-sample requirements.

Conformal thresholds use the exact ranked calibration score. When there are too
few calibration rows for the requested coverage, reports show an infinite threshold:
classification sets contain both labels and regression intervals are unbounded.

Threshold curves describe the final test set and must not be used to claim
validated deployment performance for a threshold chosen from those curves.

## Development checks

Use Python 3.11 or 3.12 and Poetry 2.2.1. Commit dependency changes together
with `poetry.lock` so local runs and CI install the same resolved dependencies.

Python 3.10 is no longer supported: patched Pillow 13, used by Matplotlib,
requires Python 3.11 or newer. Select a supported interpreter with
`poetry env use 3.11` or `poetry env use 3.12` before reinstalling dependencies.
The package explicitly requires Pillow 13 to avoid selecting affected 12.x versions.

```bash
poetry install --with dev
poetry check --lock
poetry run ruff check .
poetry run ruff format --check .
poetry run mypy src
poetry run pytest -q
poetry build
```

CI runs these checks on each supported Python version. The tests exercise the
installed CLI from a temporary directory, generate synthetic data, and verify
that the full offline workflow produces its Markdown report, CSV tables, and PNG
figures. Markdown tables require the declared `tabulate` runtime dependency.
