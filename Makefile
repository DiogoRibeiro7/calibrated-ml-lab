.PHONY: install test lint typecheck sample run

install:
	poetry install --with dev

test:
	poetry run pytest -q

lint:
	poetry run ruff check .

typecheck:
	poetry run mypy src

sample:
	poetry run calibrated-lab sample-data --output-dir data/raw --rows 5000

run:
	poetry run calibrated-lab run --input data/raw/classification.csv --regression-input data/raw/regression.csv --output-dir reports/sample_run --target label --regression-target y
