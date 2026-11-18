PY ?= python

.PHONY: install lint format typecheck test fake-demo suites validate-kl

install:
	$(PY) -m pip install -e ".[dev]"

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

typecheck:
	mypy day0

test:
	pytest -q tests/unit tests/integration

fake-demo:
	day0 run examples/fake.yaml
	@echo "Card: $$(ls -td results/fake-demo-* | head -1)/card.md"

suites:
	$(PY) scripts/build_suites.py
	$(PY) scripts/build_fake_suites.py

validate-kl:
	$(PY) scripts/validate_kl.py
