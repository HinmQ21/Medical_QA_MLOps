SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

PY := .venv/bin/python
PIP := .venv/bin/pip
DEPLOY_TOOLS_DIR := .tools
HELM := $(DEPLOY_TOOLS_DIR)/bin/helm
KUBECTL := $(DEPLOY_TOOLS_DIR)/bin/kubectl

.PHONY: install install-pipeline install-deploy-tools test smoke-pipeline smoke-pipeline-local mlflow-register-dry-run register-model dvc-status helm-lint helm-template helm-dry-run docker-build full-pipeline full-pipeline-dry-run smoke-full

.venv:
	python3.12 -m venv .venv
	$(PIP) install -U pip

install: .venv
	$(PIP) install -e ".[dev]"

install-pipeline: .venv
	$(PIP) install -e ".[dev,pipeline]"

install-deploy-tools: $(HELM) $(KUBECTL)

$(HELM) $(KUBECTL) &: scripts/install_deploy_tools.py | .venv
	$(PY) scripts/install_deploy_tools.py --bin-dir $(DEPLOY_TOOLS_DIR)/bin

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

helm-lint: $(HELM) $(KUBECTL)
	$(HELM) lint deploy/helm/api
	$(HELM) lint deploy/helm/retrieval
	$(HELM) lint deploy/helm/nginx
	$(HELM) lint deploy/helm/kserve

helm-template: $(HELM) $(KUBECTL)
	$(HELM) template medical-qa-api deploy/helm/api >/dev/null
	$(HELM) template medical-qa-retrieval deploy/helm/retrieval >/dev/null
	$(HELM) template medical-qa-nginx deploy/helm/nginx >/dev/null
	$(HELM) template medical-qa-kserve deploy/helm/kserve >/dev/null

helm-dry-run: $(HELM) $(KUBECTL)
	$(HELM) template medical-qa-api deploy/helm/api | $(KUBECTL) apply --dry-run=client --validate=false -f -
	$(HELM) template medical-qa-retrieval deploy/helm/retrieval | $(KUBECTL) apply --dry-run=client --validate=false -f -
	$(HELM) template medical-qa-nginx deploy/helm/nginx | $(KUBECTL) apply --dry-run=client --validate=false -f -

docker-build:
	docker buildx build --platform linux/amd64 -f docker/api.Dockerfile -t medical-qa-api:local --load .
	docker buildx build --platform linux/amd64 -f docker/retrieval.Dockerfile -t medical-qa-retrieval:local --load .
	docker buildx build --platform linux/amd64 -f docker/kserve-mock.Dockerfile -t medical-qa-kserve-mock:local --load .
	docker buildx build --platform linux/amd64 -f docker/pipeline-init.Dockerfile -t medical-qa-pipeline-init:local --load .

full-pipeline-dry-run:
	$(PY) -m mlops.pipelines.run_full --profile full --dry-run

full-pipeline:
	$(PY) -m mlops.pipelines.run_full --profile full

smoke-full:
	$(PY) -m mlops.pipelines.run_full --profile smoke_full
