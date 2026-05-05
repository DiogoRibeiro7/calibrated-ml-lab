# Calibration and uncertainty report

## Executive summary

The workflow compared uncalibrated model probabilities with Platt scaling, isotonic regression, and temperature scaling. Calibration was evaluated separately from discrimination.

- Best ECE: **platt** with ECE **0.0525**.
- Best log loss: **platt** with log loss **0.4109**.
- Lowest expected-cost threshold for the selected calibrated model: **0.20**.

## Calibration metrics

| model        |   brier_score |   log_loss |       ece |      mce |   roc_auc |
|:-------------|--------------:|-----------:|----------:|---------:|----------:|
| uncalibrated |      0.169365 |   0.515239 | 0.199717  | 0.704139 |  0.734526 |
| platt        |      0.126202 |   0.410946 | 0.0524735 | 0.212705 |  0.734526 |
| isotonic     |      0.129241 |   0.62723  | 0.0812414 | 0.447904 |  0.722883 |
| temperature  |      0.159506 |   0.488129 | 0.144439  | 0.501136 |  0.734526 |

## Split-conformal classification

| method                |   alpha |     qhat |   coverage |   average_set_size |   singleton_rate |   empty_set_rate |
|:----------------------|--------:|---------:|-----------:|-------------------:|-----------------:|-----------------:|
| split_conformal_platt |     0.1 | 0.723492 |   0.916667 |             1.2375 |           0.7625 |                0 |

## Split-conformal regression

| method                     |   alpha |    qhat |   coverage |   average_width |   median_width |   min_width |   max_width |
|:---------------------------|--------:|--------:|-----------:|----------------:|---------------:|------------:|------------:|
| split_conformal_regression |     0.1 | 1.34087 |   0.758333 |         2.68173 |        2.68173 |     2.68173 |     2.68173 |

## Decision-threshold analysis

|   threshold |   precision |   recall |       f1 |   tp |   fp |   tn |   fn |   expected_cost |   alert_rate |
|------------:|------------:|---------:|---------:|-----:|-----:|-----:|-----:|----------------:|-------------:|
|        0.2  |    0.313953 | 0.658537 | 0.425197 |   27 |   59 |  140 |   14 |        0.5375   |    0.358333  |
|        0.25 |    0.347826 | 0.585366 | 0.436364 |   24 |   45 |  154 |   17 |        0.541667 |    0.2875    |
|        0.15 |    0.267857 | 0.731707 | 0.392157 |   30 |   82 |  117 |   11 |        0.570833 |    0.466667  |
|        0.3  |    0.367347 | 0.439024 | 0.4      |   18 |   31 |  168 |   23 |        0.608333 |    0.204167  |
|        0.1  |    0.238806 | 0.780488 | 0.365714 |   32 |  102 |   97 |    9 |        0.6125   |    0.558333  |
|        0.05 |    0.211111 | 0.926829 | 0.343891 |   38 |  142 |   57 |    3 |        0.654167 |    0.75      |
|        0.35 |    0.393939 | 0.317073 | 0.351351 |   13 |   20 |  179 |   28 |        0.666667 |    0.1375    |
|        0.4  |    0.423077 | 0.268293 | 0.328358 |   11 |   15 |  184 |   30 |        0.6875   |    0.108333  |
|        0.45 |    0.5625   | 0.219512 | 0.315789 |    9 |    7 |  192 |   32 |        0.695833 |    0.0666667 |
|        0.5  |    0.7      | 0.170732 | 0.27451  |    7 |    3 |  196 |   34 |        0.720833 |    0.0416667 |

## Figures

- `figures/reliability_diagram.png`
- `figures/probability_histogram.png`
- `figures/conformal_regression_intervals.png`
- `figures/threshold_tradeoff.png`

## Interpretation notes

Calibration answers whether predicted probabilities match observed frequencies. It does not prove causal validity, fairness, clinical safety, or robustness under distribution shift. Conformal coverage is empirical in this workflow and relies on the calibration/test split being representative of the deployment setting.
