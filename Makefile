SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

PY := .venv/bin/python
PIP := .venv/bin/pip
DEPLOY_TOOLS_DIR := .tools
HELM := $(DEPLOY_TOOLS_DIR)/bin/helm
KUBECTL := $(DEPLOY_TOOLS_DIR)/bin/kubectl

.PHONY: install install-pipeline install-deploy-tools test smoke-pipeline smoke-pipeline-local mlflow-register-dry-run register-model dvc-status helm-lint helm-template helm-dry-run docker-build full-pipeline full-pipeline-dry-run smoke-full cloud-provision cloud-gcs-dvc cloud-workload-identity cloud-github-oidc cloud-secrets cloud-deploy cloud-smoke cloud-teardown demo-ui

.venv:
	python3.12 -m venv .venv
	$(PIP) install -U pip uv

install: .venv
	.venv/bin/uv pip install --python .venv/bin/python -e ".[dev]"

install-pipeline: .venv
	.venv/bin/uv pip install --python .venv/bin/python -e ".[dev,pipeline]"

install-deploy-tools: $(HELM) $(KUBECTL)

$(HELM) $(KUBECTL) &: scripts/install_deploy_tools.py | .venv
	$(PY) scripts/install_deploy_tools.py --bin-dir $(DEPLOY_TOOLS_DIR)/bin

test:
	.venv/bin/pytest --cov=medical_qa_platform --cov-report=term-missing

smoke-pipeline:
	.venv/bin/dvc repro build_kg_smoke eval_smoke register_smoke_model

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

cloud-provision:
	bash scripts/cloud/provision_gke.sh

cloud-gcs-dvc:
	bash scripts/cloud/setup_gcs_dvc_remote.sh

cloud-workload-identity:
	bash scripts/cloud/setup_workload_identity.sh

cloud-github-oidc:
	bash scripts/cloud/setup_github_oidc.sh

cloud-secrets:
	bash scripts/cloud/create_secrets.sh

cloud-deploy:
	bash scripts/cloud/deploy.sh

cloud-smoke:
	bash scripts/cloud/smoke_cloud.sh

cloud-teardown:
	bash scripts/cloud/teardown.sh

demo-ui:
	.venv/bin/python -m pip install -e '.[demo]'
	.venv/bin/streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
