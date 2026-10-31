# Methodology

This project separates three related but different concepts.

## Discrimination

Discrimination asks whether the model ranks positive examples above negative examples. ROC AUC and average precision mainly measure this property.

## Calibration

Calibration asks whether predicted probabilities correspond to observed frequencies. If a group of samples receives predicted probability 0.70, roughly 70% of those samples should be positive.

## Binary metric and threshold inputs

Calibration metrics and decision-threshold analysis require nonempty,
one-dimensional arrays of matching lengths. Labels must be exactly 0 or 1 and
probabilities must be finite and in [0, 1]. Fractional labels, NaN, infinity,
complex values, and mismatched shapes raise `ValueError` instead of being
truncated or silently broadcast.

Calibration bins, ECE, MCE, Brier score, and ROC AUC use the original probabilities,
including exact 0 and 1. Only log-loss inputs are clipped to [1e-12, 1 - 1e-12]
to keep logarithms finite. Clipping scores before ROC AUC can introduce ties and
change the reported ranking. Single-class evaluation is supported: ROC AUC is
reported as NaN because it is undefined, while the other metrics remain available.

The bin count must be a positive integer. Bins include their left endpoint;
probability 1 belongs to the final bin. Empty bins keep zero counts, weights, and
calibration gaps, with NaN means and observed rates.

Decision thresholds must be a nonempty, finite vector in [0, 1]. The decision
rule is `probability >= threshold`, including equality and endpoint thresholds.
Supplied order and duplicates are preserved; the default is 19 thresholds from
0.05 through 0.95. False-positive and false-negative costs must be finite,
nonnegative scalars; zero costs are allowed. Expected cost is the weighted sum
of error rates, so finite costs do not overflow merely from summing across rows.

## Uncertainty

Uncertainty estimation asks how much confidence we should place in predictions, intervals, or prediction sets. Conformal methods provide distribution-free finite-sample coverage under exchangeability assumptions.

## Implemented calibration methods

- Platt scaling: logistic regression fitted on model scores.
- Isotonic regression: monotonic non-parametric calibration.
- Temperature scaling: one-parameter logit rescaling to reduce negative log loss.

## Implemented conformal methods

- Split conformal classification using nonconformity score `1 - p_true`.
- Split conformal regression using absolute residual scores.

Both methods use the score at rank `ceil((n + 1) * (1 - alpha))` in the sorted
calibration scores, counting from one. This is an order statistic, not an
interpolated quantile. If the rank exceeds `n`, the threshold is positive infinity:
classification sets contain both labels and regression intervals cover the entire
real line. Capping the rank at `n` would understate uncertainty in this case.
See the [coverage proof in Appendix D of Angelopoulos and Bates](https://arxiv.org/html/2107.07511v6)
for the ranked-score rule and the infinite-threshold case.

Conformal functions require nonempty, one-dimensional inputs of matching lengths.
Labels must be exactly 0 or 1; probabilities must be finite and in [0, 1]; regression
targets and predictions must be finite. Invalid values raise `ValueError` rather
than being truncated or silently broadcast. Interval metrics accept ordered
infinite bounds, report infinite widths, and reject NaN or reversed bounds.
Prediction-set metrics check that membership columns and set sizes agree.

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
fewer than nine conformal rows produce an infinite threshold and full prediction
sets. Reports retain that uncertainty explicitly instead of rejecting the run.

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
