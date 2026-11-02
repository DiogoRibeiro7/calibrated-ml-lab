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
The example in `examples/run_multiclass.py` uses 60/20/10/10 splits for model training,
temperature fitting, conformal calibration, and final testing, with a fixed seed and
no external downloads. Improved calibration-set loss does not guarantee improved
test loss, ECE, class-wise calibration, or robustness under distribution shift.

## Multiclass vector scaling

`MulticlassVectorScaler` in `calibrated_ml_lab.calibration.vector` fits
`softmax(logits * weights_ + bias_)`, with one real weight and one bias per class.
This is the diagonal affine transformation described by
[Guo et al. (2017), section 4.2](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf).
Weights are unconstrained and can be negative. The transformation can change class
rankings and accuracy, so evaluate both discrimination and calibration afterwards.

```python
from calibrated_ml_lab.calibration.vector import MulticlassVectorScaler

calibrator = MulticlassVectorScaler(l2=1e-3, max_iter=1000)
calibrator.fit(calibration_logits, calibration_labels)
test_probabilities = calibrator.transform(test_logits)
```

The implementation adds a penalty towards identity to control the extra flexibility:

```text
mean negative log likelihood + l2 / 2 * (sum((weights - 1)**2) + sum(bias**2))
```

`l2` must be a finite nonnegative real scalar; it defaults to `1e-3`. Zero disables
the penalty. With no regularization, separable or missing-class data may have no
finite optimum. `max_iter` must be a positive integer and defaults to 1000.
Fitting starts deterministically from weights of one and biases of zero, using
L-BFGS-B with an analytic gradient. The objective uses log-sum-exp without clipping
probabilities. After a successful optimization, its objective is recomputed and
compared with identity; identity wins ties or a worse candidate. Since the penalty
is nonnegative and zero at identity, the accepted calibration-set NLL is no worse
than uncalibrated NLL within numerical precision.

Logits and labels follow the temperature scaler's shape, numeric-type, and class-index
requirements. The input matrix must be nonempty, finite, and have at least two class
columns. Numeric-looking strings, NaN, infinity, complex values, fractional labels,
and mismatched row counts are rejected. Transform inputs must retain the fitted
number and order of class columns. Absent classes and single-class fitting data
are accepted, but cannot establish calibration for missing classes.

Use the original model logits consistently. Subtracting row maxima **before** the
diagonal affine map would change the method: unequal weights make it sensitive to
common offsets in the original logits. The implementation subtracts row maxima
only **after** applying the map to stabilize the softmax. Probabilities can round
to zero or one. Arithmetic that cannot be represented safely raises `ValueError`
for input/transform calculations or `RuntimeError` during optimization, rather
than returning nonfinite probabilities. Nonconvergence also raises `RuntimeError`.
Failed fits preserve previous weights, biases, and class count; a new instance
remains unfitted.

The reproducible `examples/run_vector_scaling.py` draws independent synthetic
logit/label pairs from a fixed generator with known class-specific distortion.
It fits temperature and vector scaling on 2000 rows and evaluates both on another
1000 rows. The generator and regularization are fixed before evaluation; no model,
method, or hyperparameter is selected using test outcomes. It reports accuracy,
log loss, and confidence, top-label, and class-wise ECE for all three methods.
In this example vector scaling improves log loss over temperature scaling, while
its pooled confidence ECE is higher. Improvement in one metric does not imply
improvement in all calibration summaries.

For real data, train the underlying model separately, fit calibration parameters on
held-out rows, and choose methods or regularization on a separate selection split
or within development-data cross-validation. Leave final test data untouched until
these choices are fixed. Extra flexibility can overfit small calibration samples;
better fitting NLL does not guarantee better test NLL, ECE, or accuracy. If using
conformal prediction, compute a fresh threshold after fixing this calibrator.

## Multiclass matrix scaling

`MulticlassMatrixScaler` in `calibrated_ml_lab.calibration.matrix` fits a full
affine transformation of the original logits, following the matrix-scaling model
in [Guo et al. (2017), section 4.2](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf).
For a row-oriented input matrix, predictions are
`softmax(logits @ weights_.T + bias_, axis=1)`. The fitted `weights_` has shape
`(K, K)`: row `j` contains the weights from every input logit to output class `j`.
The fitted `bias_` has shape `(K,)`, and `n_classes_` records `K`.

```python
from calibrated_ml_lab.calibration.matrix import MulticlassMatrixScaler

calibrator = MulticlassMatrixScaler(l2=1e-3, max_iter=1000)
calibrator.fit(calibration_logits, calibration_labels)
test_probabilities = calibrator.transform(test_logits)
```

The implementation minimizes this regularized objective:

```text
mean negative log likelihood + l2 / 2 * (sum((weights - identity)**2) + sum(bias**2))
```

All weights, including off-diagonal entries, are unconstrained real numbers.
The penalty shrinks the complete matrix towards identity and biases towards zero.
Options and fitting safeguards match vector scaling: finite `l2 >= 0` (default
`1e-3`), integer `max_iter > 0` (default 1000), deterministic initialization at
identity, L-BFGS-B with an analytic gradient, and a recomputed objective comparison
against identity after successful optimization. The accepted calibration-set NLL
is no worse than identity within numerical precision. Zero regularization can
leave no finite optimum for separable or missing-class samples.

Softmax probabilities do not uniquely identify every matrix coefficient: adding
the same vector to every output row of `weights_`, or the same scalar to every
bias, adds a common output-logit offset and leaves probabilities unchanged.
The identity penalty favors a particular representative. Interpret predictions
and differences between output classes instead of treating unregularized weights
as unique explanations.

Input validation, class-column ordering, missing-class support, failed-refit
preservation, and numeric error behavior follow the vector-scaling contract.
Use finite nonempty logits with at least two class columns and aligned integer
class-index labels. Preserve the model's original logit convention: row offsets
are subtracted only after the affine map. Unsafe matrix products or optimizer
arithmetic raise errors; nonconvergence raises `RuntimeError`. Successful transforms
return finite normalized probability rows, which can include rounded zeros or ones.
As with vector scaling, the winning class and accuracy can change.

The `K² + K` stored coefficients increase memory, optimization cost, and the risk
of fitting noise in small calibration samples. More flexibility does not guarantee
better held-out log loss, accuracy, or calibration. Choose regularization and the
calibration method on separate selection data or development-data cross-validation.
Keep final test data untouched and fit any conformal threshold only after these
choices are fixed.

The offline `examples/run_matrix_scaling.py` uses a fixed invertible mixing matrix
to distort synthetic logits. It fits the three calibrators on 2000 independent
rows and reports their log loss, accuracy, and three ECE summaries on another
1000 rows alongside uncalibrated predictions. The generator and both L2 penalties
are fixed before evaluation; it does not select a method from test outcomes.

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

## Quantile regression intervals

`QuantileIntervalRegressor` in `calibrated_ml_lab.models.quantile` fits two
gradient-boosted regression models with quantile (pinball) loss. For residual
`r = y - prediction` and quantile level `tau`, this loss is
`max(tau * r, (tau - 1) * r)`. The two levels are `alpha / 2` and `1 - alpha / 2`;
the default `alpha=0.1` targets a central 90% prediction interval.
This uses scikit-learn's
[quantile-loss gradient boosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html).

```python
from calibrated_ml_lab.models.quantile import QuantileIntervalRegressor
from calibrated_ml_lab.conformal.regression import regression_interval_metrics

model = QuantileIntervalRegressor(alpha=0.1, random_state=42).fit(train_x, train_y)
raw_lower, raw_upper = model.predict_quantiles(test_x)
crossing_rate = (raw_lower > raw_upper).mean()
lower, upper = model.predict_interval(test_x)
summary = regression_interval_metrics(test_y, lower, upper)
```

Both prediction methods return a pair of one-dimensional arrays. The raw pair
retains the fitted quantile identities, including crossings. Independently fitted
quantile models can disagree about their ordering. `predict_interval` swaps the
endpoints wherever the raw lower estimate exceeds the raw upper estimate; tied
estimates produce zero-width intervals. This guarantees ordered bounds, without
guaranteeing coverage or that the swapped endpoints still estimate their original
quantile levels. Use raw predictions for level-specific pinball losses and report
the crossing rate alongside metrics on the ordered intervals.

The wrapper accepts finite, dense numeric feature matrices with at least one row
and column, and a finite one-dimensional target vector with one value per row.
Numeric-looking strings, complex values, NaN, infinity, and malformed shapes are
rejected. Features must fit within the finite float32 range used internally by
the trees; values also inherit float32 precision. Feature columns are positional,
including DataFrames: names and indices do not align or reorder inputs. Preserve
their count and meaning at prediction time. Encode categorical features and fit
any imputation or other preprocessing using training rows only.

`alpha` must yield two distinct representable levels strictly inside `(0, 1)`.
Tree settings are positive integers: `n_estimators=100`, `max_depth=3`, and
`min_samples_leaf=9`. `learning_rate=0.05` must be finite and positive.
`random_state=42` must be an integer from 0 through `2**32 - 1` for reproducibility.
Booleans and numeric strings are rejected as options. The two models use the same
settings and seed. Hyperparameters are revalidated on refit; changing an option
does not change an already fitted model. `alpha_`, `quantiles_`, and `n_features_in_`
describe the last successful fit. This is a small fit/prediction wrapper, rather
than a complete scikit-learn estimator with cloning and grid-search support.

Fit stages both models and validates their training predictions before replacing
any existing fit. If either fit fails, previous models and fitted metadata remain
usable. Unfitted prediction raises `RuntimeError`. Invalid user input raises
`ValueError`; detected numerical failures during training and malformed or
nonfinite model predictions raise `RuntimeError`.

The reproducible `examples/run_quantile_regression.py` generates a nonlinear mean
with increasing noise as its input feature increases. It fixes settings in advance,
fits on 2000 training rows, and reports coverage, widths, raw crossing rate, and
pinball losses on another 1000 independent rows. Two feature regions fixed at zero
also show how widths and observed coverage differ with noise. The example reaches
87.9% overall test coverage for a 90% target; the wider intervals in the noisier
region do not establish conditional coverage.

These are model-based intervals for future responses, not confidence intervals for
the conditional mean. A nominal target is not a distribution-free finite-sample
guarantee. Tune quantile-model settings using a validation split or development-data
cross-validation, then evaluate on untouched test rows. Underfitting, overfitting,
distribution shift, and inaccurate tail estimates can reduce coverage. Conformalized
quantile regression requires an additional independent calibration step and remains
a separate roadmap item.

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

Binary-classification and regression conformal functions require nonempty,
one-dimensional inputs of matching lengths.
Labels must be exactly 0 or 1; probabilities must be finite and in [0, 1]; regression
targets and predictions must be finite. Invalid values raise `ValueError` rather
than being truncated or silently broadcast. Interval metrics accept ordered
infinite bounds, report infinite widths, and reject NaN or reversed bounds.
Prediction-set metrics check that membership columns and set sizes agree.

## Multiclass conformal prediction

`calibrated_ml_lab.conformal.multiclass` implements split conformal prediction with
scores `1 - p_true_class`. Fit the model and any temperature scaler first, then
freeze them before computing scores on fresh conformal-calibration rows. The score
construction follows the classification example in
[Angelopoulos and Bates](https://arxiv.org/html/2107.07511v6#S1).

```python
from calibrated_ml_lab.conformal.multiclass import (
    multiclass_conformal_threshold,
    multiclass_prediction_sets,
    multiclass_set_coverage,
)

qhat = multiclass_conformal_threshold(conformal_labels, conformal_probabilities, alpha=0.1)
membership = multiclass_prediction_sets(test_probabilities, qhat)
summary = multiclass_set_coverage(test_labels, membership)
```

Probability inputs use the same validation as the multiclass diagnostics: nonempty
matrices with at least two class columns, finite entries in `[0, 1]`, and row sums
within absolute tolerance `1e-6` of one. No clipping or renormalization is applied.
Labels are integer-valued class-column indices; absent classes are allowed.
Keep the number and meaning of probability columns consistent across calibration,
prediction, and evaluation. The functional API does not store a class mapping.

The threshold is the score at rank `ceil((n + 1) * (1 - alpha))`, with positive
infinity when that rank exceeds the number of calibration rows. Set construction
uses `(1 - probabilities) <= qhat` directly, matching the calibration-score
arithmetic and including ties. It does not reconstruct a probability cutoff through
another subtraction, which could exclude boundary scores through floating-point
rounding. A threshold of infinity includes all classes.

The result is a boolean `(n_rows, n_classes)` matrix: entry `[i, k]` indicates that
class `k` belongs to row `i`'s prediction set. Row sums give set sizes. Empty sets
are retained, without forcing a top prediction into them. Coverage evaluation also
accepts numeric membership matrices containing only 0 and 1; other values are
rejected. It returns `coverage`, `average_set_size`, `singleton_rate`, and
`empty_set_rate`. Both labels and membership must be nonempty and aligned.

Conformal `alpha` values must be finite numeric scalars strictly between zero and
one. Thresholds must be nonnegative numeric scalars, allowing positive infinity.
Numeric-looking strings, complex values, and one-element vectors are rejected.

The example reserves 20% of rows without consulting labels, then divides that pool
equally into conformal calibration and final testing, also without stratification.
The remaining 80% is split into model-training and temperature-fitting subsets using
only development labels. All four subsets are disjoint. Changing conformal or test
labels cannot change the development rows or their labels, and neither held-out
subset fits the temperature. Predictions and the threshold are fixed before test
labels are used for evaluation.

The coverage guarantee is marginal and requires exchangeable conformal-calibration
and future observations with a fixed score function. It does not guarantee coverage
for every class, feature slice, or individual example. Empirical test coverage may
fall below the target on a finite sample. Random row splits do not address repeated
entities, time dependence, or distribution shift; use a suitable sampling design.

## Coverage by class and slice

An overall coverage rate can conceal poor coverage in a smaller group. For example,
90 covered observations from one class and 10 uncovered observations from another
give 90% overall coverage while the second class has zero coverage. Grouped empirical
diagnostics address this visibility problem; they do not change the conformal
threshold or add conditional coverage guarantees. This distinction follows the
[evaluation discussion in Angelopoulos and Bates](https://arxiv.org/html/2107.07511v6#S3).

```python
from calibrated_ml_lab.conformal.multiclass import (
    multiclass_coverage_by_class,
    multiclass_coverage_by_slice,
)

by_class = multiclass_coverage_by_class(test_labels, membership)
by_slice = multiclass_coverage_by_slice(test_labels, membership, {
    "feature_0_nonpositive": test_features[:, 0] <= 0,
    "feature_0_positive": test_features[:, 0] > 0,
})
```

Both functions validate all evaluation labels and set memberships before grouping,
using the same input contract as `multiclass_set_coverage`. Grouping is positional;
Pandas indices do not align or reorder the inputs. The overall input must remain
nonempty even if a requested group has no rows.

The resulting tables contain a group identifier and these statistics:

| Column | Meaning |
| --- | --- |
| `count` | Number of evaluation rows in the group. |
| `covered_count` | Number whose prediction set contains the true class. |
| `coverage` | `covered_count / count`. |
| `average_set_size` | Mean number of included classes, counting empty sets as size zero. |
| `singleton_rate` | Fraction of group rows with exactly one included class. |
| `empty_set_rate` | Fraction of group rows with no included classes. |

Class tables use `class_index` and contain every membership-matrix column in index
order. Grouping uses the **true class**, not the predicted winner. Absent classes
remain as rows with zero counts and NaN for all four statistics; their coverage is
undefined, not zero or one.

Slice tables use `slice` and preserve the mapping's iteration order. Names must be
nonblank strings, and masks must be one-dimensional booleans or numeric 0/1 vectors
with exactly one entry per evaluation row. All-false masks retain zero counts and
NaN statistics. An empty mapping returns an empty table with the usual columns.
The example serializes undefined table statistics as JSON `null`.

Slices can overlap or leave some rows unassigned. Do not pool their counts or average
their coverage as though they partitioned the data. For an exhaustive, disjoint
partition, weighting each nonempty group's statistics by `count` recovers the overall
statistics. Exclude zero-count groups from this calculation because their statistics
are undefined. The class table has this partition property; arbitrary slice tables may not.

Choose meaningful slices before inspecting test outcomes and retain their sample
counts. Small groups have noisy empirical coverage, and selecting slices after
seeing their errors makes the results exploratory. The example uses fixed feature
cutoffs only after fitting and prediction; no slice refits the temperature or the
conformal threshold. Per-class and per-slice rates describe the observed test rows,
not a promise about future coverage or fairness.

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
