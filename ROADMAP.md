# Roadmap

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

- [ ] Extend temperature scaling to multi-class softmax outputs.
- [ ] Add vector scaling and matrix scaling.
- [ ] Add top-label calibration diagnostics.
- [ ] Add class-wise ECE.
- [ ] Add multi-class conformal prediction sets.
- [ ] Add coverage by class and slice.

## v0.3 — Regression uncertainty

- [ ] Add quantile regression intervals.
- [ ] Add bootstrap prediction intervals.
- [ ] Add conformalized quantile regression.
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
