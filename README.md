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
- Regularized multiclass vector scaling with class-specific weights and biases (Python API).
- Regularized multiclass matrix scaling for corrections that mix class logits (Python API).
- Multiclass confidence, top-label, and class-wise calibration diagnostics.
- Split conformal classification.
- Multiclass split-conformal prediction sets (Python API).
- Split conformal regression intervals.
- Gradient-boosted quantile regression intervals with feature-dependent widths (Python API).
- Residual-bootstrap prediction intervals with refit variation and future-observation noise.
- Conformalized quantile regression with signed corrections and explicit empty-set diagnostics.
- Coverage and interval-width diagnostics.
- Interval calibration tables and plots across nominal coverage levels.
- Empirical interval coverage by fixed feature bins, including counts and empty bins.
- Empirical multiclass coverage by true class and named data slices.
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

The example uses separate model-training, temperature-fitting, conformal-calibration,
and test sets (60/20/10/10%). It deliberately
amplifies logits to simulate overconfidence, then reports temperature, test log loss,
and test accuracy before and after calibration. It also reports confidence ECE,
top-label ECE, and per-class ECE/MCE on the test set. In this example, log loss
improves while some binned calibration errors increase; the output keeps both results
visible. This illustrates the method;
calibration does not guarantee better performance on unseen data.
See [the multiclass API contract](docs/methodology.md#multiclass-temperature-scaling)
for label mapping, numerical limits, and fitting assumptions. The `calibrated-lab run`
reporting workflow currently handles binary classification.

For class-specific corrections, use `MulticlassVectorScaler` from
`calibrated_ml_lab.calibration.vector`. It fits `softmax(logits * weights_ + bias_)`
with L2 regularization towards identity. It can change predicted classes and depends
on the original logit offsets, so keep the model's logit convention consistent.

```bash
poetry run python examples/run_vector_scaling.py
```

This separate offline example compares uncalibrated, temperature-scaled, and
vector-scaled predictions on independent synthetic calibration and test rows.
It reports log loss, accuracy, and all three ECE summaries without selecting a
method from test outcomes. See [vector scaling](docs/methodology.md#multiclass-vector-scaling)
for the objective, options, and limitations.

`MulticlassMatrixScaler` from `calibrated_ml_lab.calibration.matrix` fits a full
affine transformation, `softmax(logits @ weights_.T + bias_)`. Each output class can
use every input logit. The default penalty shrinks the weights towards the identity
matrix and biases towards zero. Its `K² + K` coefficients can overfit small
calibration samples, so choose the method and regularization on separate development data.

```bash
poetry run python examples/run_matrix_scaling.py
```

This offline example compares temperature, vector, and matrix scaling against
uncalibrated predictions under fixed mixing of class logits. See
[matrix scaling](docs/methodology.md#multiclass-matrix-scaling) for matrix orientation,
parameter ambiguity, numerical limits, and evaluation requirements.

The functions in `calibrated_ml_lab.calibration.multiclass_metrics` accept class-index
labels and a matrix of class probabilities:

- `confidence_calibration_bins`: pooled confidence versus prediction accuracy.
- `top_label_calibration_bins`: confidence versus accuracy within each predicted class.
- `classwise_calibration_errors`: one-versus-rest ECE/MCE for every class, with label counts.
- `multiclass_calibration_metrics`: confidence ECE, top-label ECE, and mean class-wise ECE.

See [the diagnostic definitions](docs/methodology.md#multiclass-calibration-diagnostics)
for bin weights, missing classes, and probability ties.

The functions in `calibrated_ml_lab.conformal.multiclass` compute a threshold from
fresh calibration probabilities, return boolean prediction-set membership matrices,
and summarize empirical coverage and set size. `multiclass_coverage_by_class` groups
observations by their true class; `multiclass_coverage_by_slice` accepts named binary
masks for custom groups. Both tables show counts and covered counts alongside rates,
and retain empty groups with undefined statistics.

The example reports overall, per-class, and feature-slice coverage at a 90% target,
including set sizes and empty-set rates. Its feature slices use a fixed zero cutoff
on the first input feature. These diagnostics expose differences between groups;
they do not establish class- or slice-conditional coverage guarantees. See
[multiclass conformal prediction](docs/methodology.md#multiclass-conformal-prediction)
and [grouped coverage](docs/methodology.md#coverage-by-class-and-slice)
for input requirements, data separation, and interpretation.

## Quantile regression intervals

Use `QuantileIntervalRegressor` from `calibrated_ml_lab.models.quantile` to fit
lower and upper conditional quantiles on numeric training features. The default
`alpha=0.1` fits the 5th and 95th percentiles for a nominal 90% interval.
`predict_quantiles` returns the raw estimates; `predict_interval` orders crossed
endpoints. Report the crossing rate and evaluate coverage on separate test rows.
These estimates do not have a finite-sample coverage guarantee.

```bash
poetry run python examples/run_quantile_regression.py
```

The offline example fits on 2000 rows and evaluates another 1000 rows with
feature-dependent noise. It reports coverage, interval widths, crossing rate,
and pinball losses, with coverage and widths for two fixed noise regions.
See [quantile regression intervals](docs/methodology.md#quantile-regression-intervals)
for the API, input requirements, and interpretation.

To calibrate these bounds on fresh observations, use the functions in
`calibrated_ml_lab.conformal.quantile`: `conformalized_quantile_threshold`,
`conformalized_quantile_intervals`, and `conformalized_quantile_metrics`.
The signed correction can widen or shrink intervals; reversed corrected bounds
represent empty sets and must retain their order. The summary reports their rate
and includes them as uncovered, zero-width sets.

```bash
poetry run python examples/run_conformalized_quantile.py
```

This example uses disjoint model-training, conformal-calibration, and test data,
and compares base and corrected coverage and widths. See
[conformalized quantile regression](docs/methodology.md#conformalized-quantile-regression)
for the exact finite-sample rule, empty-set handling, and marginal coverage assumptions.

## Bootstrap prediction intervals

`ResidualBootstrapRegressor` in `calibrated_ml_lab.models.bootstrap` fits a base
regressor and clones it on resampled, centered training residuals. Its prediction
intervals include a separate residual draw for future-observation noise. The default
model is linear regression; compatible scikit-learn regressors and numeric pipelines
can be supplied. `predict` returns the base model's point predictions and
`predict_interval` returns lower and upper bounds.

```bash
poetry run python examples/run_bootstrap_regression.py --output-dir .artifacts/bootstrap_regression
```

The example evaluates constant and feature-dependent noise on separate test rows,
prints overall metrics, and exports `bootstrap_coverage.csv` by feature bin. Pooled
residuals assume the same additive error distribution across features. They can
undercover with heteroskedasticity, small samples, or overfitting; bootstrap intervals
have no finite-sample coverage guarantee. See [residual bootstrap](docs/methodology.md#residual-bootstrap-prediction-intervals)
for the algorithm, reproducibility contract, and limitations.

## Interval calibration curves

Use `interval_calibration_table` from `calibrated_ml_lab.uncertainty.intervals` to
compare nominal and empirical coverage at several levels. The table retains
observation counts, covered counts, signed coverage gaps, interval widths, and
empty-set rates. `plot_interval_calibration` from
`calibrated_ml_lab.visualization.plots` exports a comparison of one or more methods.

```bash
poetry run python examples/run_interval_calibration.py --output-dir .artifacts/interval_calibration
```

The example exports `interval_calibration.png` and `interval_calibration.csv` for
quantile and conformalized quantile intervals at five preset levels, using separate
training, calibration, and test rows. Review the widths alongside the coverage
curve. See [interval calibration diagnostics](docs/methodology.md#interval-calibration-across-nominal-levels)
for input conventions and interpretation.

## Interval coverage by feature

Use `interval_coverage_by_feature` from `calibrated_ml_lab.uncertainty.intervals`
to inspect coverage, widths, and coverage gaps within fixed numeric feature bins.
It reports bin counts and preserves empty bins with undefined statistics. Every
evaluation row must fall within the supplied edges; rows are never silently dropped.

```bash
poetry run python examples/run_feature_coverage.py --output-dir .artifacts/feature_coverage
```

The example exports `feature_coverage.csv` for quantile and conformalized quantile
intervals at a 90% target, alongside overall summaries printed as JSON. Bins are
fixed before fitting and evaluation. These are descriptive diagnostics, without
conditional coverage guarantees. See [feature-bin coverage](docs/methodology.md#interval-coverage-by-feature-bins)
for boundary rules, empty-bin handling, and interpretation.

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
