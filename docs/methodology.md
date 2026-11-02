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

The binary calibrators validate aligned, nonempty one-dimensional fitting arrays before
training: labels must be exactly 0 or 1 and probabilities must be finite and in
[0, 1]. Numeric-looking strings and other nonnumeric array elements are rejected
before conversion; real numeric and boolean arrays remain supported, including
object arrays containing numeric values. Transform inputs use the same probability
validation. Invalid data raises
`ValueError`. A failed fit keeps any previous successful fit usable; a new instance
remains unfitted and raises `RuntimeError` if asked to transform probabilities.

Platt scaling requires both classes. Isotonic and temperature scaling also accept
single-class fitting data, although such a sample cannot establish calibration for
the missing class. The full classification workflow still requires both classes
in its development subsets.

Isotonic regression uses original probabilities, including exact endpoints and
distinct scores near 0 or 1. For valid probabilities outside the fitted score
range, its prediction is the nearest fitted boundary value. Platt and temperature
scaling retain clipping to [1e-12, 1 - 1e-12] for finite logit calculations.

## Multiclass temperature scaling

`MulticlassTemperatureScaler` in `calibrated_ml_lab.calibration.multiclass`
fits a single positive temperature and returns `softmax(logits / T)` for each row.
This follows [Guo et al. (2017)](https://proceedings.mlr.press/v70/guo17a.html).
The scalar temperature adjusts confidence and mathematically preserves class order
within each row; floating-point rounding can introduce ties between very close scores.

```python
from calibrated_ml_lab.calibration.multiclass import MulticlassTemperatureScaler

calibrator = MulticlassTemperatureScaler().fit(calibration_logits, calibration_labels)
test_probabilities = calibrator.transform(test_logits)
```

- Logits are finite real arrays with shape `(n_samples, n_classes)`, at least one
  row, and at least two class columns. Pass model logits, not softmax probabilities.
- Labels are a one-dimensional array of integer-valued class indices from `0` to
  `n_classes - 1`, with one label per row. Map arbitrary class names to logit-column
  indices first. Numeric-looking strings, fractional labels, NaN, and infinity
  are rejected.
- Transform inputs must retain the fitted number of columns and their order. The
  scaler checks the number of columns; callers are responsible for their meaning.
- Calibration labels may omit classes, including a single-class calibration sample.
  The number of classes comes from the logits. Such data cannot establish calibration
  for missing classes.

Fit uses a bounded search over `log(T)` in `[-3, 3]`, approximately
`T` in `[0.0498, 20.0855]`. It minimizes mean calibration negative log likelihood
using log-sum-exp directly, without clipping probabilities inside the objective.
The objective is divided by a fixed positive constant to stabilize optimization
arithmetic for large logit gaps. Identity scaling and both temperature endpoints
are evaluated as well; identity wins an exact loss tie. These choices keep fitted
calibration loss no worse than the identity candidate within numerical precision.
The bounds can limit the correction for severely miscalibrated scores.

Row maxima are subtracted before scaling so large common offsets do not overflow.
Logits whose differences cannot be represented safely across the temperature range
raise `ValueError`. Returned probabilities may round to exactly zero or one.
Invalid inputs and optimizer failures preserve a previous successful fit, including
its class count. Transforming an unfitted instance raises `RuntimeError`.

Fit the model, fit the calibrator, and evaluate final predictions on separate data.
The example in `examples/run_multiclass.py` uses 60/20/20 splits and a fixed seed,
with no external downloads. Improved calibration-set loss does not guarantee improved
test loss, ECE, class-wise calibration, or robustness under distribution shift.

## Multiclass calibration diagnostics

`calibrated_ml_lab.calibration.multiclass_metrics` evaluates probability matrices.
Unlike the temperature scaler, these functions take probabilities, not logits.
Labels are integer-valued column indices from `0` to `n_classes - 1`. Inputs must
have matching, nonzero row counts and at least two class columns. Every entry must
be finite and in `[0, 1]`; row sums must be within `1e-6` of one (absolute tolerance,
with no relative tolerance). Accepted values are preserved without clipping or
renormalization. Numeric-looking strings and fractional labels are rejected.

The three ECE summaries answer different questions:

| Summary | Binning and aggregation |
| --- | --- |
| `confidence_ece` | Bin the maximum probability against prediction correctness across all rows. Sum each bin's absolute confidence–accuracy gap weighted by its fraction of all rows. |
| `top_label_ece` | Bin confidence and correctness separately within each predicted class. Sum gaps weighted by each class/bin's fraction of all rows. |
| `classwise_ece` | For every class, bin its probability against the indicator that it is the true class, using all rows. Average the resulting one-versus-rest ECE values equally across all class columns. |

Conditioning on the predicted class distinguishes top-label calibration from pooled
confidence calibration, following
[Gupta and Ramdas](https://arxiv.org/abs/2107.08353). For example, two equally sized
groups predicted as different classes can both have confidence 0.75, but accuracies
1.0 and 0.5. Their pooled confidence ECE is zero, while top-label ECE is 0.25.
With the same bin edges, top-label ECE cannot be smaller than confidence ECE apart
from floating-point rounding. Neither summary examines the entire probability vector.
The class-wise calculation examines every column, including probabilities for classes
that are never the winning prediction.

`confidence_calibration_bins` returns `n_bins` rows; `top_label_calibration_bins`
returns `n_classes * n_bins` rows and adds `predicted_class`. Both tables include
bin edges, counts, global weights, mean confidence, accuracy, and absolute gaps.
All bins are equal-width over `[0, 1]`, include their left edge, and include exactly
one in the final bin. Empty bins have zero count, weight, and gap, with NaN means.
Predicted classes with no rows retain their empty bins. Probability ties select
the first column, matching NumPy's `argmax`; reordering tied columns can change
predictions and hence confidence and top-label results.

`classwise_calibration_errors` reports `class_index`, `count`, `positive_count`,
`ece`, and `mce` for every class. `count` is the total number of input rows for
each one-versus-rest calculation; MCE is the largest gap in a nonempty bin.
Classes absent from the labels remain included, with `positive_count = 0`.
The macro class-wise ECE includes every column equally, rather than weighting by
class frequency or omitting absent classes. Zero error for an absent class does
not demonstrate performance on positive examples of that class.

Use separate evaluation data and record the chosen bin count. ECE and MCE depend
on that choice and on sample size; MCE can be dominated by a bin with very few rows.
These summaries do not establish full multiclass calibration, fairness, or robustness
under distribution shift. The example reports these metrics separately from accuracy
and log loss, since optimizing log loss need not improve the binned errors.

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
