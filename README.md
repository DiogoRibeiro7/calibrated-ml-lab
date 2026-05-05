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

## Main outputs

```text
reports/sample_run/
├── calibration_report.md
├── tables/
│   ├── calibration_metrics.csv
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
