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

## Classification evaluation protocol

Five disjoint subsets have separate responsibilities (fractions are approximate
because row counts are rounded):

| Split | Fraction | Purpose |
| --- | --- | --- |
| Train | 50% | Fit preprocessing and the random forest. |
| Calibrator fit | 10% | Fit Platt, isotonic, and temperature scaling. |
| Selection | 10% | Choose the lowest-ECE candidate, including uncalibrated probabilities. |
| Conformal | 10% | Compute the selected method's conformal threshold. |
| Test | 20% | Evaluate frozen candidates and the selected conformal prediction sets. |

The default split seed is 42; `run_classification_calibration` also accepts `seed`.
First reserve 30% of rows without consulting labels, then divide those rows into
conformal calibration and final testing. Stratify only the remaining development
rows when allocating training, calibrator fitting, and method selection. Thus,
changing held-out labels cannot change the development partitions or fitted models.

Each development subset must contain both binary classes. Small or heavily
imbalanced datasets can fail this requirement and produce an explicit error;
the workflow never borrows held-out rows to make a fit succeed. At alpha 0.1,
at least nine conformal rows are required to estimate a finite conformal threshold.

Method selection uses equal-width-bin ECE with the same `n_bins` used for final
evaluation. Exact ties follow candidate order: uncalibrated, Platt, isotonic,
temperature. Calibrators are not refitted after selection. Selection ECE is a
tuning result, not an unbiased performance estimate; it can be noisy on small
samples and depends on the bin count.

`selection_metrics.csv` records validation results and the selected candidate.
`calibration_metrics.csv` records final test results with that same selected flag.
`classification_splits.csv` records row counts and the seed. Conformal calibration
uses only its own subset, never calibrator-fitting or method-selection rows.
The test threshold curve is descriptive; its minimum is not a validated deployment
threshold. Regression retains its separate 60/20/20 train/calibration/test protocol.

Keeping score fitting separate from conformal calibration follows the split
conformal setup in [Angelopoulos and Bates](https://arxiv.org/abs/2107.07511).
Coverage is reported empirically. Marginal conformal validity relies on
exchangeability and does not imply per-class or per-group coverage. Random row
splits are unsuitable for repeated entities or time-dependent observations without
an appropriate data-splitting design.

## Important limitations

Calibration can fail under distribution shift. Conformal coverage is marginal unless stronger assumptions or conditional methods are used. A low ECE does not imply fairness, safety, or causal validity.
