.PHONY: install install-all test lint typecheck check benchmark dashboard

install:
	python -m pip install -e ".[dev]"

install-all:
	python -m pip install -e ".[dev,quantum,dashboard]"

test:
	pytest

lint:
	ruff check .

typecheck:
	mypy src

check: lint typecheck test

benchmark:
	qob benchmark --config configs/baseline.yaml

dashboard:
	streamlit run app.py
