# Instructions for coding agents

This project is a Python calibration and uncertainty-estimation workbench.

## Style

- Use typed Python.
- Keep modules small and testable.
- Prefer clear functions over clever abstractions.
- Add docstrings to public functions and classes.
- Never hide uncertainty or overstate model reliability.

## Testing

Run:

```bash
poetry run pytest -q
poetry run ruff check .
poetry run mypy src
```

## Design rules

- Calibration methods must expose `fit`, `predict_proba`, or `transform` as appropriate.
- Metrics must handle edge cases such as empty bins, degenerate labels, and probabilities at 0 or 1.
- Reports must distinguish discrimination from calibration.
- Conformal intervals must report empirical coverage.
- All generated outputs should be reproducible with a seed.
