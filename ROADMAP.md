# Roadmap

## Immediate engineering priorities

- [x] Establish reproducible dependencies and green lint, formatting, type, test,
  and package-build checks on Python 3.11–3.12, with patched Pillow and pytest.
- [x] Exercise CLI help, data generation, and the complete offline report in CI.
- [x] Separate calibration-method selection, calibrator fitting, conformal
  calibration, and final evaluation to remove classification test leakage.
- [x] Use exact conformal order statistics, preserve infinite thresholds for small
  calibration samples, and validate conformal inputs and coverage summaries.
- [x] Preserve probability endpoints and score rankings in binary metrics, and
  validate metric inputs, decision thresholds, and error costs consistently.
- [x] Validate calibrator fitting and transform inputs, preserve successful fits
  after failed refits, and avoid clipping isotonic training scores.
- [x] Report expected CLI input errors clearly and stage complete workflow artifacts
  before publishing them, preserving existing reports when computation fails.

## v0.1 — Local calibration and uncertainty workbench

- [x] Synthetic binary classification dataset.
- [x] Synthetic regression dataset.
- [x] Train/calibration/test split.
- [x] Baseline probabilistic classifier.
- [x] ECE, MCE, Brier score, log loss, AUC.
- [x] Platt scaling.
- [x] Isotonic calibration.
- [x] Temperature scaling.
- [x] Reliability diagrams.
- [x] Probability histograms.
- [x] Split conformal classification.
- [x] Split conformal regression.
- [x] Coverage and interval-width diagnostics.
- [x] Decision-threshold analysis.
- [x] Markdown report generation.
- [x] CLI and notebook workflow.
- [x] Tests.

## v0.2 — Multi-class calibration

- [x] Add multi-class temperature scaling for softmax logits, with a Python API
  and a reproducible example using separate calibration and test data.
- [x] Add regularized vector scaling with class-specific weights and biases, a
  Python API, and an offline comparison on independent calibration and test rows.
- [x] Add regularized matrix scaling with a full affine logit transformation, a
  Python API, and an offline comparison on independent calibration and test rows.
- [x] Add pooled confidence and top-label calibration diagnostics with bin tables.
- [x] Add one-versus-rest class-wise ECE/MCE and macro class-wise ECE.
- [x] Add multi-class split-conformal prediction sets and empirical coverage summaries,
  with separate temperature-fitting, conformal-calibration, and test rows.
- [x] Add empirical coverage by true class and named slices, including counts,
  overlapping masks, and explicit undefined statistics for empty groups.

## v0.3 — Regression uncertainty

- [x] Add gradient-boosted quantile regression intervals with raw crossing diagnostics,
  ordered bounds, and an offline evaluation on independent heteroskedastic test rows.
- [ ] Add bootstrap prediction intervals.
- [x] Add conformalized quantile regression with exact signed score thresholds,
  empty-set summaries, and separate model-training, calibration, and test rows.
- [ ] Add interval calibration plots.
- [ ] Add conditional coverage diagnostics by feature bins.

## v0.4 — Model-agnostic uncertainty

- [ ] Add jackknife-plus intervals.
- [ ] Add ensemble uncertainty decomposition.
- [ ] Add epistemic vs aleatoric uncertainty examples.
- [ ] Add bootstrap confidence intervals for calibration metrics.

## v0.5 — Monitoring integration

- [ ] Compare calibration between reference and production periods.
- [ ] Add calibration drift metrics.
- [ ] Add alert rules for ECE, Brier score, and coverage degradation.
- [ ] Add slice-level calibration monitoring.

## v0.6 — Decision-aware calibration

- [ ] Add expected utility curves.
- [ ] Add cost-sensitive threshold optimisation.
- [ ] Add abstention and defer-to-human policies.
- [ ] Add selective classification metrics.

## v1.0 — Stable calibration toolkit

- [ ] Stable public API.
- [ ] Full documentation site.
- [ ] More examples for healthcare, fraud, churn, and credit risk.
- [ ] CI with coverage gate.
- [ ] Versioned releases.
