PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: install install-pipeline test smoke-pipeline smoke-pipeline-local mlflow-register-dry-run register-model dvc-status

.venv:
	python3.12 -m venv .venv
	$(PIP) install -U pip

install: .venv
	$(PIP) install -e ".[dev]"

install-pipeline: .venv
	$(PIP) install -e ".[dev,pipeline]"

test:
	.venv/bin/pytest --cov=medical_qa_platform --cov-report=term-missing

smoke-pipeline:
	.venv/bin/dvc repro

smoke-pipeline-local:
	$(PY) -m mlops.pipelines.build_smoke_kg --profile smoke
	$(PY) -m mlops.pipelines.eval_smoke --profile smoke
	$(PY) -m mlops.mlflow_register --profile smoke --dry-run

mlflow-register-dry-run:
	$(PY) -m mlops.mlflow_register --profile smoke --dry-run

register-model:
	$(PY) -m mlops.mlflow_register --profile smoke

dvc-status:
	.venv/bin/dvc status
