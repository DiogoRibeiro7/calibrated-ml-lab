# Data model

## Classification input

Required columns:

- `label`: binary target, 0 or 1, with both classes present and no missing labels.
- numeric feature columns.
- optional categorical slice columns.

Use `--target` to choose another target column. Provide enough rows of each class
for model training, calibrator fitting, and method selection; see
[the splitting requirements](methodology.md). Conformal and final-test holdouts
may contain a single class. Columns prefixed with `true_` are
excluded from classification predictors.

The sample generator creates:

- `age`
- `income`
- `usage_30d`
- `prior_events`
- `risk_signal`
- `region`
- `segment`
- `label`

## Regression input

Required columns:

- `y`: finite numeric target with no missing values.
- at least one numeric feature column other than the target.

Use `--regression-target` to choose another target column. Categorical columns
are ignored by the regression model. Missing predictor values are imputed; missing,
infinite, or nonnumeric target values are rejected before regression fitting.

The sample generator creates:

- `x1`
- `x2`
- `x3`
- `y`
