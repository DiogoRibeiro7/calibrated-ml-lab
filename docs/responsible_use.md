# Responsible use

Calibration and conformal prediction are useful for making model uncertainty visible. They are not guarantees of correctness.

Avoid these mistakes:

- Treating calibrated probabilities as causal probabilities.
- Assuming calibration transfers across populations without monitoring.
- Reporting only global calibration while ignoring subgroup calibration.
- Using conformal coverage claims when the calibration and deployment data are not exchangeable.
- Using calibrated scores for high-stakes decisions without human review and domain validation.
