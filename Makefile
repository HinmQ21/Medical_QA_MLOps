PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: install test

.venv:
	python3.12 -m venv .venv
	$(PIP) install -U pip

install: .venv
	$(PIP) install -e ".[dev]"

test:
	.venv/bin/pytest --cov=medical_qa_platform --cov-report=term-missing
