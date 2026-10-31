# Calibration and uncertainty report

## Executive summary

The workflow compared uncalibrated model probabilities with Platt scaling,
isotonic regression, and temperature scaling. Calibration was evaluated separately
from discrimination.

- Selected method: **platt**, chosen by lowest validation ECE
  (**0.0827**) before final test evaluation.
- Selected method's final test ECE: **0.0482**.
- Selected method's final test log loss: **0.3592**.

## Classification data splits

Training, calibrator fitting, method selection, conformal calibration, and final
testing use disjoint rows. The selected method and conformal threshold are frozen
before final test evaluation. No calibrator is refitted after method selection.

| split          |   rows |   seed |
|:---------------|-------:|-------:|
| train          |    600 |     42 |
| calibrator_fit |    120 |     42 |
| selection      |    120 |     42 |
| conformal      |    120 |     42 |
| test           |    240 |     42 |

## Calibration-method selection

These metrics use the selection (validation) split. Exact ECE ties use candidate
order: uncalibrated, Platt, isotonic, then temperature scaling.

| model        |   brier_score |   log_loss |       ece |      mce |   roc_auc | selected   |
|:-------------|--------------:|-----------:|----------:|---------:|----------:|:-----------|
| uncalibrated |      0.185451 |   0.55074  | 0.199776  | 0.536528 |    0.6255 | False      |
| platt        |      0.145386 |   0.462723 | 0.0827155 | 0.530314 |    0.6255 | True       |
| isotonic     |      0.158662 |   0.904114 | 0.126176  | 0.6      |    0.583  | False      |
| temperature  |      0.188987 |   0.563546 | 0.142258  | 0.845837 |    0.6255 | False      |

## Calibration metrics

These are final test metrics. The selected flag records the validation choice;
test rankings do not change that choice. ECE, MCE, Brier score, and log loss
describe probability quality; ROC AUC describes discrimination.

| model        |   brier_score |   log_loss |       ece |      mce |   roc_auc | selected   |
|:-------------|--------------:|-----------:|----------:|---------:|----------:|:-----------|
| uncalibrated |      0.166938 |   0.509829 | 0.236213  | 0.339053 |  0.777422 | False      |
| platt        |      0.108819 |   0.359206 | 0.0481576 | 0.295231 |  0.777422 | True       |
| isotonic     |      0.118722 |   0.78718  | 0.0704225 | 0.5      |  0.758955 | False      |
| temperature  |      0.153854 |   0.466872 | 0.168856  | 0.412509 |  0.777422 | False      |

## Split-conformal classification

| method                |   alpha |     qhat |   coverage |   average_set_size |   singleton_rate |   empty_set_rate |
|:----------------------|--------:|---------:|-----------:|-------------------:|-----------------:|-----------------:|
| split_conformal_platt |     0.1 | 0.836201 |   0.970833 |            1.48333 |         0.516667 |                0 |

## Split-conformal regression

| method                     |   alpha |    qhat |   coverage |   average_width |   median_width |   min_width |   max_width |
|:---------------------------|--------:|--------:|-----------:|----------------:|---------------:|------------:|------------:|
| split_conformal_regression |     0.1 | 1.34087 |   0.758333 |         2.68173 |        2.68173 |     2.68173 |     2.68173 |

## Decision-threshold analysis

This table describes the selected method on the final test set. Its lowest-cost
threshold is exploratory, not a validated deployment recommendation. Choosing a
deployment threshold requires separate validation data and a new test evaluation.

|   threshold |   precision |   recall |       f1 |   tp |   fp |   tn |   fn |   expected_cost |   alert_rate |
|------------:|------------:|---------:|---------:|-----:|-----:|-----:|-----:|----------------:|-------------:|
|        0.25 |    0.328767 | 0.685714 | 0.444444 |   24 |   49 |  156 |   11 |        0.433333 |    0.304167  |
|        0.2  |    0.273684 | 0.742857 | 0.4      |   26 |   69 |  136 |    9 |        0.475    |    0.395833  |
|        0.3  |    0.36     | 0.514286 | 0.423529 |   18 |   32 |  173 |   17 |        0.4875   |    0.208333  |
|        0.35 |    0.416667 | 0.428571 | 0.422535 |   15 |   21 |  184 |   20 |        0.504167 |    0.15      |
|        0.15 |    0.237705 | 0.828571 | 0.369427 |   29 |   93 |  112 |    6 |        0.5125   |    0.508333  |
|        0.4  |    0.48     | 0.342857 | 0.4      |   12 |   13 |  192 |   23 |        0.533333 |    0.104167  |
|        0.1  |    0.210526 | 0.914286 | 0.342246 |   32 |  120 |   85 |    3 |        0.5625   |    0.633333  |
|        0.45 |    0.5      | 0.2      | 0.285714 |    7 |    7 |  198 |   28 |        0.6125   |    0.0583333 |
|        0.5  |    0.5      | 0.171429 | 0.255319 |    6 |    6 |  199 |   29 |        0.629167 |    0.05      |
|        0.55 |    0.5      | 0.114286 | 0.186047 |    4 |    4 |  201 |   31 |        0.6625   |    0.0333333 |

## Figures

- `figures/reliability_diagram.png`
- `figures/probability_histogram.png`
- `figures/conformal_regression_intervals.png`
- `figures/threshold_tradeoff.png`

## Interpretation notes

Calibration answers whether predicted probabilities match observed frequencies.
It does not prove causal validity, fairness, clinical safety, or robustness under
distribution shift. Reported coverage is empirical and can differ from the nominal
90% target. Conformal validity requires exchangeable conformal-calibration and
future examples; random row splits do not address time dependence, repeated
entities, or distribution shift.
