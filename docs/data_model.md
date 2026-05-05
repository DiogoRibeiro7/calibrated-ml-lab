# Data model

## Classification input

Required columns:

- `label`: binary target, 0 or 1.
- numeric feature columns.
- optional categorical slice columns.

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

- `y`: continuous target.
- numeric feature columns.

The sample generator creates:

- `x1`
- `x2`
- `x3`
- `y`
