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

## Bootstrap confidence intervals for calibration metrics

`calibrated_ml_lab.calibration.bootstrap` provides two fixed-prediction APIs:

| Function | Reported metrics |
|---|---|
| `bootstrap_calibration_metrics` | Binary `brier_score`, `log_loss`, `ece`, and `mce` |
| `bootstrap_multiclass_calibration_metrics` | Multiclass `confidence_ece`, `top_label_ece`, and macro `classwise_ece` |

Brier score and log loss measure overall probabilistic quality, not calibration
alone. ROC AUC is excluded because it measures discrimination and is undefined
for single-class resamples. The multiclass API uses the existing pooled-confidence,
conditional top-label, and one-versus-rest class-wise definitions; every probability
column stays in the macro average even when its label is absent from a resample.

```python
import numpy as np
from calibrated_ml_lab.calibration.bootstrap import (
    bootstrap_calibration_metrics,
    bootstrap_multiclass_calibration_metrics,
)

y = np.array([0, 0, 1, 1])
p = np.array([0.1, 0.4, 0.6, 0.9])
binary = bootstrap_calibration_metrics(
    y, p, n_bins=5, n_bootstrap=1000, confidence_level=0.95, random_state=42
)
multiclass = bootstrap_multiclass_calibration_metrics(
    y, np.column_stack([1 - p, p]), n_bins=5, random_state=42
)
```

Given `n` evaluation rows, each of `B = n_bootstrap` replicates draws `n` row indices
uniformly with replacement. The same indices select both labels and full probability
rows, preserving their association, and all metrics use the same resample. The method
does not stratify by class, discard single-class resamples, or refit any model or
calibrator. Bin edges stay fixed at the original equal-width partition of `[0, 1]`.
This is the paired percentile procedure described in the
[SciPy bootstrap documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html).

For metric `T`, the output estimate is `T(y, p)` on the original rows. With
`q = (1 - confidence_level) / 2`, the bounds are the `q` and `1 - q` quantiles of
the bootstrap metric values, using NumPy's linear interpolation. `standard_error`
is the sample standard deviation of these values (`ddof=1`), not that deviation
divided by `sqrt(B)`. Bounds are not forced to contain the original estimate.
The implementation uses a local random generator and stores one index vector
at a time plus the `B` by metric-count array; it does not materialize `B` datasets.

Both functions return a DataFrame with these columns:

| Column | Meaning |
|---|---|
| `metric` | Metric name, in the order listed above |
| `estimate` | Original evaluation-sample metric |
| `lower`, `upper` | Two-sided percentile endpoints |
| `standard_error` | Bootstrap sample standard deviation |
| `confidence_level` | Requested nominal confidence level |
| `n_samples` | Original evaluation row count, also the size of every resample |
| `n_bootstrap` | Number of resamples used |
| `n_bins` | Fixed equal-width bin count |
| `method` | Always `percentile` |

At least two evaluation rows are required. The binary API accepts nonempty,
aligned one-dimensional 0/1 labels and finite probabilities in `[0, 1]`. The
multiclass API accepts integer class indices and finite probability matrices with
at least two columns and row sums within `1e-6` of one. It retains all columns;
a two-column matrix uses multiclass confidence/top-label semantics, which differ
from binary positive-class calibration. Inputs are matched by position, including
Pandas inputs, and are not modified. Numeric objects and boolean arrays follow
the existing metric validators; strings, complex values, NaNs, and infinities are
rejected. Probability endpoints stay intact for every metric except binary log
loss, which uses the existing clipping to `[1e-12, 1 - 1e-12]`.

`n_bins` is a positive integer (default 10), `n_bootstrap` is an integer at least
two (default 1000), and `random_state` is an integer in `[0, 2**32 - 1]` (default
42). Boolean options are rejected; NumPy integer scalars are accepted. The
confidence level is a finite real scalar strictly between zero and one (default
0.95), and must yield distinct representable quantiles inside `(0, 1)`. Repeating
the same ordered inputs and settings reproduces results without altering global
NumPy random state. A small replicate count is allowed for quick checks, but gives
poor resolution of tail quantiles; assess stability with more resamples for analysis.

These are **marginal, approximate sampling intervals**, conditional on the fitted
predictor and chosen bins. They assume IID evaluation rows representative of the
target population. They do not include training/calibration-set uncertainty,
hyperparameter selection, bin selection, or future distribution shift. Clustered
or time-dependent observations need an appropriate resampling design. Resampling
training predictions does not repair optimistic evaluation or leakage.

Percentile intervals do not correct the finite-sample bias or nonsmoothness of
binned ECE/MCE. In particular, sparse bins and absolute gaps near zero can make
bootstrap coverage unreliable. A positive ECE lower bound is not a general test
that a predictor is miscalibrated. Identical or perfectly predicted observed rows
can produce zero-width intervals; those cannot establish certainty about unseen
data. Reporting several intervals does not give simultaneous coverage. Interval
overlap, even with shared draws, is not a confidence interval or significance test
for a difference between models. These APIs neither certify calibration nor
produce prediction intervals for individual outcomes.

The offline `examples/run_calibration_bootstrap.py` creates separate binary and
three-class datasets with seed 42. For each, it fits a standardized logistic model
on 1200 training rows, amplifies logits by three, fits temperature on 600 separate
calibration rows, then freezes both raw and calibrated predictions on 600 test
rows. The split, bin count, and interval settings are preset; the test responses
are used only for evaluation. The default 1000 resamples yield 95% intervals for
each predictor, using the same seed to pair the row draws across predictors.

`calibration_bootstrap_intervals.csv` contains 14 rows with additional `task` and
`predictor` columns. `binary_test_predictions.csv` and
`multiclass_test_predictions.csv` retain test labels and the raw/calibrated
probabilities so the intervals can be recomputed. The example prints paths and
settings as JSON and accepts `--n-bootstrap` for a smaller or larger run. It does
not select a predictor or interval method using these test results.

In the seeded binary example, temperature scaling lowers test log loss from about
0.173 to 0.0885 while ECE rises from about 0.0132 to 0.0207. Improvement in one
diagnostic need not imply improvement in another; the individual intervals do
not by themselves establish a significant difference between predictors.

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
distribution shift, and inaccurate tail estimates can reduce coverage.
[Conformalized quantile regression](#conformalized-quantile-regression) adds an
independent calibration step after the quantile models are fixed.

## Residual bootstrap prediction intervals

`ResidualBootstrapRegressor` in `calibrated_ml_lab.models.bootstrap` implements a
fixed-design residual bootstrap. It uses resampled residuals both to refit the
regressor and to represent noise in a future response. Percentiles of refitted
mean predictions alone describe a different source of variation and can be much
narrower than prediction intervals. Adding residual noise follows the construction
described in the [nlstools prediction-bootstrap documentation](https://search.r-project.org/CRAN/refmans/nlstools/html/nlsBootpredict.html).

Given training rows `(X, y)`, fit a base regressor `f`. Center its residuals as
`r_i = y_i - f(x_i) - mean(y - f(X))`. For each of `B = n_bootstrap` replicates:

1. Draw `n` centered residuals with replacement and form `y* = f(X) + r*` at the
   original training features. Fit a fresh clone `f_b` to `(X, y*)`.
2. Draw another residual `e_b` independently from the same centered pool.
3. At query `x`, form the predictive draw `f_b(x) + e_b`.

Return the `alpha / 2` and `1 - alpha / 2` percentiles across the `B` predictive
draws, using NumPy's `method="linear"` interpolation. These are approximate
pointwise prediction intervals, without a conformal rank correction. The point
prediction returned by `predict` is `f(x)`, not the bootstrap median.

```python
from sklearn.linear_model import Ridge
from calibrated_ml_lab.models.bootstrap import ResidualBootstrapRegressor
from calibrated_ml_lab.conformal.quantile import conformalized_quantile_metrics

model = ResidualBootstrapRegressor(
    Ridge(alpha=1.0), alpha=0.1, n_bootstrap=1000, random_state=42
).fit(train_x, train_y)
predictions = model.predict(test_x)
lower, upper = model.predict_interval(test_x)
summary = conformalized_quantile_metrics(test_y, lower, upper)
```

Omitting the estimator uses ordinary `LinearRegression`. Supply a cloneable
scikit-learn regressor instance (a `BaseEstimator` tagged as a regressor), including
a pipeline ending in a regressor. Its clones must fit scalar responses and return
one finite prediction per row. The wrapper accepts dense finite numeric feature
matrices and real target vectors, with at least two training rows. It does not
accept sparse, categorical, missing, or multi-output data. An estimator may impose
additional numeric restrictions. Pandas inputs use positional columns and rows;
column names and indices do not control alignment.

The supplied estimator is treated as a template; any previous fit is discarded
in its clones, while the original estimator and input arrays are preserved.
Clone implementations that return the original object are rejected. Preprocessing
inside a supplied pipeline is refitted with each clone. Tune its settings using
development data, and reserve independent test responses for final evaluation.
This wrapper implements `fit`, `predict`, and `predict_interval`, without exposing
the full scikit-learn estimator interface itself.

`alpha` must yield distinct representable quantile levels strictly between zero
and one. `n_bootstrap` is an integer at least two (default 200), and `random_state`
is an integer from zero to `2**32 - 1` (default 42). A small replicate count has
substantial Monte Carlo error, especially in the tails; increase it and assess
stability on development data. Fitting stores `B + 1` estimators, and interval
prediction uses memory proportional to `B` times the query batch size.

The seed controls residual sampling and is assigned to **all exposed estimator
`random_state` parameters**, including nested pipeline parameters and any explicit
seeds on the template. Every clone receives the same estimator seed, so changes
between clones come from resampled targets. This makes fits reproducible for
estimators whose randomness is controlled through those parameters; third-party
estimators with additional random sources may require their own controls.

The independent noise draws are stored at fit time and reused across query rows.
Prediction does not consume random numbers, and pointwise results are unchanged
by repeated calls, row ordering, or batching apart from numerical roundoff in the
underlying estimator. Shared residual draws do **not** define independent joint
future observations or a simultaneous prediction band. Fitted options remain in
effect until another successful fit. Invalid data, numerical failures, or failures
in a later bootstrap clone leave the previous fitted state intact.

The model exposes `estimator_`, `bootstrap_estimators_`, centered `residuals_`,
stored `noise_draws_`, `n_features_in_`, `alpha_`, and `quantiles_` for inspection.
Treat fitted attributes as read-only. Constant targets or an interpolating fit
can yield zero residuals and collapsed intervals; the implementation retains this
result rather than inserting an arbitrary noise floor.

Pooled residual resampling assumes an adequate mean model and IID additive errors
with a common distribution independent of features. These are raw centered fitted
residuals, without leverage or degrees-of-freedom adjustment. They may understate
noise in small samples or with flexible models. Increasing `B` reduces simulation
error but does not correct overfitting, model bias, heteroskedasticity, dependent
rows, or distribution shift. There is no finite-sample, marginal, or conditional
coverage guarantee. Report empirical coverage and widths on untouched data, and
inspect feature bins even when overall coverage is near the target.

The offline `examples/run_bootstrap_regression.py` fixes seed 42, the 90% target,
500 bootstrap replicates, and feature edges `[-inf, -1, 0, 1, inf]`. It uses 1000
training rows and 2000 separate test rows for each of two linear-mean scenarios.
With constant noise, observed coverage is 90.2%. With feature-dependent noise,
overall coverage is 90.95%, but coverage in the highest feature bin is 76.56%
(512 rows). The CSV and overall JSON summaries expose this failure of the pooled
noise assumption. These seeded results illustrate the method, not expected
coverage on other data. The bootstrap example needs no conformal-calibration set.

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

## Conformalized quantile regression

`calibrated_ml_lab.conformal.quantile` implements split conformalized quantile
regression (CQR), using the signed scores in
[Romano, Patterson, and Candès (2019), section 4](https://papers.neurips.cc/paper/8613-conformalized-quantile-regression.pdf).
First fit the quantile models and fix any bound-ordering rule on training data.
Then evaluate their bounds on fresh calibration observations:

```text
score[i] = max(lower_calibration[i] - y_calibration[i],
               y_calibration[i] - upper_calibration[i])
```

`conformalized_quantile_threshold` returns the score at one-based rank
`ceil((n + 1) * (1 - alpha))`, without interpolation. A required rank beyond the
calibration sample size returns positive infinity. Scores inside the base interval
can be negative; they are preserved. Applying the threshold yields
`[lower - qhat, upper + qhat]`. Positive thresholds widen the bounds; negative
thresholds shrink them. The infinite fallback returns the full real line.

```python
from calibrated_ml_lab.conformal.quantile import (
    conformalized_quantile_threshold,
    conformalized_quantile_intervals,
    conformalized_quantile_metrics,
)

calibration_lower, calibration_upper = model.predict_interval(calibration_x)
qhat = conformalized_quantile_threshold(
    calibration_y, calibration_lower, calibration_upper, alpha=0.1
)
test_lower, test_upper = model.predict_interval(test_x)
lower, upper = conformalized_quantile_intervals(test_lower, test_upper, qhat)
summary = conformalized_quantile_metrics(test_y, lower, upper)
```

Calibration responses and base bounds must be aligned, nonempty, finite real
vectors. Base bounds must be ordered; raw crossed quantiles are rejected. Use the
same fixed ordering rule before calibration and prediction, as `predict_interval`
does above. Fractional real regression responses are accepted; strings, complex
values, NaN, infinity, and mismatched shapes are rejected. `alpha` is a real scalar
strictly between zero and one. The correction accepts any finite real scalar or
positive infinity, but rejects NaN and negative infinity. Nonrepresentable final
scores raise `ValueError`; endpoint arithmetic may produce infinite bounds.

Shrinkage can produce a corrected lower bound above its upper bound. This denotes
an **empty set**, not a reversed interval to repair. Sorting corrected endpoints
would change the prediction set defined by the score rule. Equal finite endpoints
remain a singleton, which can cover an exactly matching response.

Use `conformalized_quantile_metrics` for corrected intervals. It reports `coverage`,
`average_width`, `median_width`, `min_width`, `max_width`, and `empty_set_rate`.
Empty sets contribute zero coverage and zero width to the full-sample summaries;
they are never dropped. Unbounded intervals can have infinite width. NaN bounds
and purported singleton intervals at positive or negative infinity are rejected.
The existing `regression_interval_metrics` keeps its ordered-bound contract.

With quantile models, tuning choices, and preprocessing fixed independently of
calibration observations, exchangeability of calibration and future observations
gives marginal coverage at least `1 - alpha` under the usual conformal argument.
This does not guarantee the realized coverage of a particular finite test sample,
or coverage within every feature region. Avoid reusing calibration labels for
model selection, changing the quantile models afterwards, or treating dependent
time-series or repeated-entity observations as independent random rows.

The offline example uses 2000 training rows, 1000 independent calibration rows,
and 1000 untouched test rows from a fixed heteroskedastic generator. It reports
base and corrected coverage, widths, empty-set rates, and two fixed noise regions.
In this seeded run overall test coverage changes from 90.3% to 91.2%, with a small
positive correction. These results illustrate the procedure; empirical coverage
can differ in either direction on other data.

## Interval calibration across nominal levels

An interval calibration curve compares each requested coverage level with the
fraction of evaluation responses actually inside its prediction sets. The diagonal
marks equal nominal and empirical coverage. Points below it indicate empirical
undercoverage; points above it indicate empirical overcoverage. Compare widths
alongside coverage: arbitrarily wide intervals can cover every response.

`interval_calibration_table` in `calibrated_ml_lab.uncertainty.intervals` takes
evaluation responses and a mapping from nominal coverage to `(lower, upper)` arrays:

```python
from pathlib import Path
from calibrated_ml_lab.uncertainty.intervals import interval_calibration_table
from calibrated_ml_lab.visualization.plots import plot_interval_calibration

intervals = {0.8: bounds_80, 0.9: bounds_90, 0.95: bounds_95}
table = interval_calibration_table(test_y, intervals)
plot_interval_calibration(test_y, {"My method": intervals}, Path("calibration.png"))
```

The mapping must be nonempty. Its keys must be real numeric scalars strictly
between zero and one, remaining distinct after conversion to float. Each value
must be a tuple or list containing two one-dimensional bound arrays, with one
entry per evaluation response. Responses must be nonempty and finite; bounds may
include infinity but must not include NaN. Numeric strings, complex values, and
mismatched shapes are rejected. Alignment is positional, including Pandas inputs.
Rows are sorted by nominal level; neither input arrays nor mapping order are changed.

Each table row reports:

| Column | Meaning |
| --- | --- |
| `nominal_coverage` | Requested level associated with the supplied interval arrays. |
| `count` | Number of evaluation responses, including those with empty prediction sets. |
| `covered_count` | Number contained in their prediction sets, including finite boundary equality. |
| `coverage` | `covered_count / count`. |
| `coverage_gap` | Empirical minus nominal coverage; negative values indicate undercoverage. |
| `average_width`, `median_width`, `min_width`, `max_width` | Width summaries across every evaluation row. |
| `empty_set_rate` | Fraction with a corrected lower bound exceeding its upper bound. |

The set convention matches CQR: reversed bounds denote empty sets and contribute
zero width and zero coverage. Equal finite endpoints are singletons; unbounded
sets remain included, with potentially infinite widths. Singleton infinities are
invalid. Order raw crossed quantile estimates before computing calibration scores,
then retain any reversed bounds produced by a signed conformal correction. The
table cannot infer whether reversed input bounds were intentional.

`plot_interval_calibration` takes a nonempty mapping from nonblank method names
to their interval mappings. Methods may use different sets of nominal levels,
but all must use the same evaluation responses. The plot includes a diagonal,
the observed points, and the evaluation count. It connects points in nominal-level
order without smoothing rates or enforcing monotonicity. Connecting lines are
visual guides, not predictions at unmeasured levels or evidence of nested intervals.
All methods are validated before output is created; the figure is closed after
saving or a plotting failure. The filename extension selects the Matplotlib output
format, including PNG and SVG.

The offline `examples/run_interval_calibration.py` fixes levels 0.5, 0.7, 0.8, 0.9,
and 0.95 before evaluation. At each level it fits quantile models on 2000 training
rows and computes a CQR threshold on 1000 separate calibration rows. It evaluates
both methods on the same 1000 untouched test rows, then exports a PNG and a CSV
with the plotted rates and their accompanying width diagnostics. It uses a fixed
seed and a noninteractive plotting backend, so it runs without downloads or GUI windows.

Rates at different levels are correlated because they reuse the same test rows.
The curve is descriptive and has no uncertainty band; proximity to the diagonal
does not prove conditional coverage or identify the best method. Choose models,
levels, and tuning settings on development data, then use this plot for final
evaluation. A method selected from this test plot needs a new independent evaluation.

## Interval coverage by feature bins

Overall coverage can hide undercoverage in a feature region. For example, covering
nine responses in one region and missing the only response in another gives 90%
overall coverage, but zero empirical coverage in the second region.
`interval_coverage_by_feature` in `calibrated_ml_lab.uncertainty.intervals` reports
coverage and widths within a fixed partition of a numeric feature:

```python
import numpy as np
from calibrated_ml_lab.uncertainty.intervals import interval_coverage_by_feature

# Fix these edges from domain knowledge or training data before evaluation.
edges = np.array([-np.inf, -1.0, 0.0, 1.0, np.inf])
by_feature = interval_coverage_by_feature(
    test_y, lower, upper, test_x[:, 0], edges, nominal_coverage=0.9
)
```

Responses, bounds, and feature values must be nonempty one-dimensional arrays of
equal length. Alignment is positional, including Pandas inputs; all rows are
validated before grouping. Responses and feature values must be finite. Bounds
follow the [CQR set convention](#conformalized-quantile-regression): reversed bounds
denote empty prediction sets, equal finite endpoints are singletons, and infinite
bounds are allowed for unbounded sets. NaN and singleton infinities are rejected.
Empty prediction sets contribute zero coverage and zero width within their bins.

Edges must be real numeric values, strictly increasing after conversion to float,
with at least two entries. Infinite outer endpoints are allowed. Each bin includes
its left edge and excludes its right edge, except the final bin includes both.
Internal ties go to the bin on their right. Every feature value must lie between
the outer edges: out-of-range rows cause an error, rather than being dropped or
clipped. Infinite outer edges make the partition exhaustive for finite values.
Constant features are supported; duplicate edges from tied quantiles must be
resolved before calling the function. No bins are learned or merged automatically.

The table has one row for every bin, in edge order:

| Column | Meaning |
| --- | --- |
| `bin_index`, `bin_lower`, `bin_upper` | Zero-based bin index and its supplied boundaries. |
| `right_closed` | Whether the right endpoint is included; true only for the final bin. |
| `count`, `fraction` | Number of evaluation rows in the bin and its share of all rows. |
| `covered_count` | Number of responses covered in the bin. |
| `nominal_coverage` | Required reference target, strictly between zero and one. |
| `coverage`, `coverage_gap` | Empirical coverage and empirical minus nominal coverage. |
| `average_width`, `median_width`, `min_width`, `max_width` | Width summaries within the bin. |
| `empty_set_rate` | Fraction of the bin's prediction sets that are empty. |

An empty **bin** has zero `count`, `covered_count`, and `fraction`, but NaN for
coverage, coverage gap, all widths, and empty-set rate. It provides no evidence
about coverage. This differs from a populated bin containing only empty
prediction sets: its coverage and widths are zero and its empty-set rate is one.
CSV exports represent undefined statistics as blank cells.

Bin counts sum to the evaluation count. Summing covered counts and dividing by
the total count reconstructs overall coverage; averaging bin coverage without
count weights does not. Ignore empty bins when weighting coverage, average width,
or empty-set rate. Medians generally cannot be combined by taking a weighted
average of bin medians. Different feature partitions describe the same rows, so
their counts must not be pooled together.

The offline `examples/run_feature_coverage.py` uses seed 42, 2000 training rows,
1000 separate conformal-calibration rows, and 1000 test rows. Before fitting it
fixes the 90% target and edges `[-inf, -1, 0, 1, inf]`. It exports a CSV comparing
quantile and conformalized quantile intervals in all four bins and prints JSON
with the corresponding overall metrics. No fitting or bin selection uses test labels.

These are descriptive feature-conditional diagnostics, not conditional coverage
guarantees. Small bins have noisy observed rates; this table supplies counts but
no uncertainty intervals or multiple-comparison correction. Prespecify features
and edges, or choose them using separate development data. Selecting bins or
methods after inspecting test results requires fresh evaluation data. A marginal
conformal guarantee does not ensure coverage in every bin or at every feature value.

## Jackknife-plus prediction intervals

`JackknifePlusRegressor` in `calibrated_ml_lab.conformal.jackknife` implements
jackknife-plus. For each training row `i`, fit `f_-i` without that row and calculate
`r_i = abs(y_i - f_-i(x_i))`. At query `x`, form paired candidates
`f_-i(x) - r_i` and `f_-i(x) + r_i`. Select these one-based order statistics:

- Lower endpoint: `floor((n + 1) * alpha)`-th lower candidate.
- Upper endpoint: `ceil((n + 1) * (1 - alpha))`-th upper candidate.

This is equation (9) of [Barber, Candès, Ramdas, and Tibshirani](https://arxiv.org/html/1905.02928v3#S2).
Theorem 1 gives a marginal coverage lower bound of `1 - 2 * alpha`, assuming
exchangeable training/test observations and a learner invariant to training-row
order. The conventional nominal target is `1 - alpha`. Thus `alpha=0.1` targets
90% but has an 80% general bound; use `alpha=0.05` for a 90% bound under those
assumptions. This is not a guarantee for each test sample or feature group.

```python
from sklearn.linear_model import Ridge
from calibrated_ml_lab.conformal.jackknife import JackknifePlusRegressor
from calibrated_ml_lab.conformal.quantile import conformalized_quantile_metrics

model = JackknifePlusRegressor(Ridge(alpha=1.0), alpha=0.1).fit(train_x, train_y)
predictions = model.predict(test_x)
lower, upper = model.predict_interval(test_x)
summary = conformalized_quantile_metrics(test_y, lower, upper)
```

The wrapper fits `n + 1` cloned estimators: `n` leave-one-out models and a full-data
model used only for point prediction. It defaults to `LinearRegression`. An
estimator must be a cloneable scikit-learn `BaseEstimator` tagged as a regressor,
return one finite prediction per query row, and support fitting `n - 1` rows.
Feature matrices must be dense, finite, and numeric. Targets are finite real
vectors. Training requires at least two aligned rows and one feature column.
Pandas row indices and column names do not affect positional alignment.

The original estimator and input arrays are preserved. Preprocessing inside a
pipeline is refitted separately for every omitted row, including scalers and
feature selection. Fix hyperparameters in advance or tune on separate development
data. Preprocessing or tuning on all training responses before leave-one-out
fitting can leak the omitted response. This implementation needs no dedicated
conformal-calibration split, but test responses must remain reserved for evaluation.

The required symmetry concerns the complete fitting procedure, including
preprocessing. Setting `random_state` does not make an order-dependent learner
permutation invariant. The wrapper sets every exposed estimator `random_state`
parameter, including nested pipeline options, to its own seed on each clone.
This controls reproducibility for compatible learners; users must still assess
the symmetry assumption for their chosen algorithm. Third-party random sources
not exposed through these parameters need separate controls.

`alpha` is restricted to `0 < alpha < 0.5`, keeping a positive general coverage
bound and ordered endpoints. `random_state` is an integer from zero through
`2**32 - 1` (default 42). Fitted `alpha_` remains in effect until a successful
refit. Validation failures, numerical failures, or a failed later omitted fit
preserve the entire previous fitted state. Fitted attributes `estimator_`,
`loo_estimators_`, `loo_residuals_`, `n_features_in_`, and `alpha_` are available
for inspection and should be treated as read-only. Leave-one-out model `i` and
residual `i` correspond to the same original training row.

For externally computed leave-one-out results, the standalone
`jackknife_plus_intervals(loo_predictions, loo_residuals, alpha=0.1)` accepts a
finite matrix with shape `(n_training_rows, n_query_rows)` and a finite nonnegative
residual vector. At least two training rows and one query row are required. It
validates shapes and values but cannot verify that the supplied fits truly omit
their corresponding rows, or that model/residual pairing is correct.

Ranks use order statistics without quantile interpolation. Floating-point rank
boundaries are compared with the represented values `k / (n + 1)`, preserving
the distinction between a boundary and its adjacent floating-point values. When
`alpha < 1 / (n + 1)`, both endpoints are missing and the result is `(-inf, inf)`.
The code never caps that interval to the observed extrema. Finite candidate
arithmetic overflowing outward also retains infinite endpoints. Nonfinite model
predictions and overflow when computing a held-out residual are rejected.

Prediction evaluates all `n` omitted models and uses memory proportional to
`n * n_query_rows`. Queries can be processed in batches; predictions agree up to
the learner's numerical roundoff. There is no parallel fitting, automatic
subsampling, or CV+ approximation. Choose training sizes and learners with the
cost of `n + 1` complete fits in mind. The wrapper itself does not implement the
full scikit-learn estimator interface.

The offline `examples/run_jackknife_plus.py` fixes seed 42, 200 training rows,
2000 untouched test rows, four feature bins, and error levels 0.1 and 0.05. The
linear mean has feature-dependent noise. The example prints nominal targets,
general coverage bounds, and empirical coverage separately, and exports a CSV
with feature-bin counts and widths. In this seeded run, empirical coverage is
88.9% and 96.15%, respectively. These observations do not establish conditional
coverage or determine the best error level on new data.

## Regression ensemble variance decomposition

`ensemble_regression_summary` in `calibrated_ml_lab.uncertainty.ensemble` summarizes
an equally weighted mixture of regression predictive distributions. Each member
supplies a predictive mean `mu_m(x)` and predictive variance `v_m(x)` at each query.
The law of total variance gives:

```text
mean_prediction(x)        = mean_m(mu_m(x))
within_model_variance(x)  = mean_m(v_m(x))
between_model_variance(x) = mean_m((mu_m(x) - mean_prediction(x))**2)
total_variance(x)         = within_model_variance(x) + between_model_variance(x)
total_std(x)              = sqrt(total_variance(x))
```

These are the mixture moments used for regression in
[Lakshminarayanan, Pritzel, and Blundell, section 2.4](https://arxiv.org/pdf/1612.01474).
Computing the moments does not require Gaussian components. In particular, the
resulting mixture need not be Gaussian, and these summaries alone do not define
prediction quantiles or justify a Gaussian interval.

```python
import numpy as np
from calibrated_ml_lab.uncertainty.ensemble import ensemble_regression_summary

# Two members, each predicting the same two query rows.
means = np.array([[0.0, 10.0], [2.0, 14.0]])
variances = np.array([[1.0, 4.0], [9.0, 0.0]])
summary = ensemble_regression_summary(means, variances)
# mean_prediction: [1, 12]; within: [5, 2]; between: [1, 4]; total: [6, 6].
```

Both inputs must be nonempty finite real matrices with identical shape
`(n_members, n_query_rows)`. Variances must be nonnegative and expressed in squared
target units, not standard deviations. Scalars, vectors, implicit broadcasting,
numeric strings, complex values, missing values, and infinite inputs are rejected.
Pandas inputs are aligned positionally; member rows and query columns must already
match. The function returns a DataFrame with one row per query, preserving query
order and using a fresh integer index. Inputs are not modified.

The five columns shown above are the complete output schema. All members receive
equal mixture weights; custom weights and multi-output covariance matrices are
not supported. The between-model term uses denominator `M`, not `M - 1`, because
it is the variance of this finite mixture. It is not the sampling variance or
standard error of an ensemble average, and is not divided by `M` again. Replicating
every member equally leaves all mixture moments unchanged.

Predictive variances are mandatory. Passing explicit zeros means the component
distributions are point masses; it does not mean observation noise is unknown or
has been estimated to vanish. A single member can have nonzero predictive variance
despite zero disagreement. Identical deterministic members yield zero total
variance even if all predictions share a large bias.

The implementation centers and scales calculations to reduce cancellation and
intermediate overflow, including small disagreements around large common offsets.
Normal floating-point roundoff still applies. Output variances beyond the float
range become positive infinity. `total_std` is calculated without first squaring
the disagreement scale, so it can remain finite even when `total_variance`
overflows. Negative input variances are rejected rather than clipped.

Within-model predictive variance is often interpreted as an aleatoric component,
and disagreement as an epistemic component, but that interpretation depends on
how the members and their variances were obtained. Supplied predictive variances
can already contain parameter uncertainty; correlated or similarly misspecified
members can agree while being wrong. This API reports mixture components without
claiming to recover the true uncertainty sources, a Bayesian posterior, calibration,
or coverage guarantees. Fit models and choose their settings using training or
development data; evaluate predictive performance on untouched responses.

The offline `examples/run_ensemble_uncertainty.py` fixes seed 42, draws 80 training
rows from a linear model with constant noise, and fits 40 linear regressors on
bootstrap resamples. Each component uses its own training residual sum of squares
divided by `n - 2` as a constant predictive noise variance. This is an illustrative
homoskedastic plug-in estimate, not a general variance-learning procedure. Test
labels are used only to evaluate 1000 separate test observations.

The example exports `ensemble_test_predictions.csv`, `ensemble_uncertainty.csv`
on a grid extending beyond the training range, and `ensemble_uncertainty.png`.
The PNG stacks within-model and between-model variance, with the generating
training range marked. In the seeded run, test MSE is about 0.681 and average
predicted total variance is about 0.577. The mismatch illustrates why computing
the decomposition does not establish calibration; agreement between those two
aggregate statistics would not establish conditional calibration either.

## Epistemic and aleatoric uncertainty example

`examples/run_uncertainty_sources.py` illustrates the distinction in a model
where the two components have explicit meanings. Assume a linear conditional
mean, independent Gaussian observation errors with known constant variance
`sigma**2`, and a Gaussian prior on the intercept and slope:

```text
phi(x) = [1, x]
y = phi(x) @ beta + epsilon
epsilon ~ Normal(0, sigma**2)
beta ~ Normal([0, 0], tau**2 * I)
```

With training design matrix `Phi` (one row per observation), the posterior and
predictive moments are:

```text
S = (I / tau**2 + Phi.T @ Phi / sigma**2)**(-1)
m = S @ Phi.T @ y / sigma**2

mean_prediction(x)    = phi(x) @ m
epistemic_variance(x) = phi(x) @ S @ phi(x).T
aleatoric_variance(x) = sigma**2
total_variance(x)     = epistemic_variance(x) + aleatoric_variance(x)
```

These follow the Gaussian linear-model posterior and predictive distribution in
[Rasmussen and Williams, chapter 2, section 2.1](https://gaussianprocess.org/gpml/chapters/RW2.pdf).
The epistemic term is posterior uncertainty about the latent mean. The aleatoric
term is future observation noise conditional on the parameters. The example uses
linear solves to compute the moments, without posterior sampling or a finite
ensemble approximation. Its helpers are local to the example, not a new public
regression estimator API.

The experiment fixes seed 42 and generates `y = 1 + 2*x + sigma*z`, with
`x ~ Uniform(-2, 2)` and independent standard-normal noise `z`. It uses a fixed
prior variance `tau**2 = 25`, noise standard deviations 0.4 and 1.2, and training
sizes 20, 80, and 320. Smaller training sets are prefixes of the largest one.
Training inputs and standardized noise draws are reused across the two noise
levels, making the comparison controlled rather than six independent datasets.

The 1000 test inputs and standardized test errors are drawn separately from
training and reused across scenarios. Test responses enter only the evaluation
metrics; they do not fit parameters, estimate noise, select the prior, or choose
the scenarios. A separate 401-point query grid spans `[-5, 5]` to show extrapolation.
All six scenarios are fixed before evaluation.

The example writes:

- `uncertainty_sources_summary.csv`: one row per scenario, including training and
  test counts, known noise level, nominal coverage, test MSE, mean component and
  total variances, empirical predictive coverage, and mean interval width.
- `uncertainty_sources_grid.csv`: grid inputs, predicted means, variance
  components, and predictive bounds, with training-size and noise-level identifiers.
- `uncertainty_sources_test.csv`: the same quantities on independent test rows,
  with observed responses and a boolean coverage indicator.
- `uncertainty_sources.png`: component means against training size, plus variance
  curves across the query grid. Feature panels use a shared logarithmic variance
  scale and mark the generating training-input domain.

The reported intervals use `mean_prediction +/- z_0.95 * sqrt(total_variance)`.
They are central 90% posterior predictive intervals for future noisy responses
under this Gaussian model. Using only the epistemic term would instead describe
uncertainty about the latent mean. These are not conformal intervals, and empirical
test coverage need not equal 90% in a finite run. The fixed generating coefficients
are not sampled from the prior, so posterior credibility should not be read as an
exact repeated-sample coverage guarantee for those coefficients.

In the seeded run, mean epistemic variance at noise SD 0.4 falls from about
0.0174 with 20 training rows to 0.00102 with 320; aleatoric variance remains 0.16.
Observed test coverage is 83.9%, 88.1%, and 89.0% for the three training sizes at
both noise levels. The repeated percentages arise in this paired experiment;
they are not independent replications or a general coverage result.

For nested training designs with fixed prior and known noise, adding observations
reduces posterior covariance; higher noise at a fixed design increases parameter
uncertainty as well as observation variance. The aleatoric term stays constant
across sample sizes. In this example epistemic variance also grows away from the
observed input region. These comparisons are conditional on the specified linear
model. They do not account for an incorrect mean function, unknown noise, missing
features, or a change in the data-generating process. More data need not reveal
shared model bias. Aleatoric uncertainty is irreducible relative to the chosen
information and observation process, not necessarily to every possible measurement.

The earlier `ensemble_regression_summary` can summarize approximations to such a
posterior predictive mixture, but arbitrary bootstrap members and plug-in
variances do not automatically recover these Bayesian uncertainty components.

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
