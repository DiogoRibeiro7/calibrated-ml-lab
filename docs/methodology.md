# Methodology

This project separates three related but different concepts.

## Discrimination

Discrimination asks whether the model ranks positive examples above negative examples. ROC AUC and average precision mainly measure this property.

## Calibration

Calibration asks whether predicted probabilities correspond to observed frequencies. If a group of samples receives predicted probability 0.70, roughly 70% of those samples should be positive.

## Uncertainty

Uncertainty estimation asks how much confidence we should place in predictions, intervals, or prediction sets. Conformal methods provide distribution-free finite-sample coverage under exchangeability assumptions.

## Implemented calibration methods

- Platt scaling: logistic regression fitted on model scores.
- Isotonic regression: monotonic non-parametric calibration.
- Temperature scaling: one-parameter logit rescaling to reduce negative log loss.

## Implemented conformal methods

- Split conformal classification using nonconformity score `1 - p_true`.
- Split conformal regression using absolute residual scores.

## Important limitations

Calibration can fail under distribution shift. Conformal coverage is marginal unless stronger assumptions or conditional methods are used. A low ECE does not imply fairness, safety, or causal validity.
