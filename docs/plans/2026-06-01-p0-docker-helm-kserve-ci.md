# MLOps P0 — Docker / Helm / KServe / CI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Package the runtime API, retrieval service, KServe smoke predictor, and DVC-backed deployment metadata into locally verifiable Docker, Helm, KServe, and GitHub Actions artifacts.

**Architecture:** Plan 3 stays inside `mlops-platform/` and does not import from the sibling `baseline/` tree. The runtime images are API, retrieval, and KServe mock predictor; a fourth small support image exists only for the retrieval chart initContainer because `dvc pull` requires DVC but the retrieval runtime image intentionally installs only `.[runtime]`. Helm charts render app-serving resources only; live GKE, real RunPod, GCS secret wiring, observability, and MLflow server deployment remain Plan 4.

**Tech Stack:** Python 3.12, FastAPI, Docker, Helm, Kubernetes manifests, KServe `InferenceService`, GitHub Actions, PyYAML, pytest.

---

## Scope Notes

- Implement the approved spec at `docs/specs/2026-06-01-p0-docker-helm-kserve-ci-design.md`.
- Preserve Plan 2 targets, especially `make dvc-status`.
- Add a support `docker/pipeline-init.Dockerfile` because the spec requires a DVC-capable init image while keeping the retrieval runtime image lean.
- Do not run local `docker build` on the aarch64 host as a completion gate. Dockerfiles are structurally tested locally; real image builds happen in CI on x86.
- Do not execute `helm upgrade`, access GKE, configure GCS credentials, or validate KServe against a live CRD.

## File Structure

**Create:**
- `scripts/install_deploy_tools.py` — downloads pinned `helm` and `kubectl` into `.tools/bin`.
- `src/medical_qa_platform/serving/__init__.py` — serving package marker.
- `src/medical_qa_platform/serving/kserve_mock_app.py` — KServe-compatible mock predictor app.
- `docker/api.Dockerfile`
- `docker/retrieval.Dockerfile`
- `docker/kserve-mock.Dockerfile`
- `docker/pipeline-init.Dockerfile`
- `deploy/helm/api/{Chart.yaml,values.yaml,templates/configmap.yaml,templates/deployment.yaml,templates/service.yaml,templates/hpa.yaml}`
- `deploy/helm/retrieval/{Chart.yaml,values.yaml,templates/configmap.yaml,templates/pvc.yaml,templates/deployment.yaml,templates/service.yaml,templates/hpa.yaml}`
- `deploy/helm/nginx/{Chart.yaml,values.yaml,templates/configmap.yaml,templates/secret.yaml,templates/deployment.yaml,templates/service.yaml}`
- `deploy/helm/kserve/{Chart.yaml,values.yaml,templates/inferenceservice.yaml}`
- `.github/workflows/ci.yml`
- `.github/workflows/deploy.yml`
- `tests/serving/test_kserve_mock_app.py`
- `tests/deploy/test_makefile_deploy_tools.py`
- `tests/deploy/test_dockerfiles.py`
- `tests/deploy/helm_helpers.py`
- `tests/deploy/test_helm_api_chart.py`
- `tests/deploy/test_helm_retrieval_chart.py`
- `tests/deploy/test_helm_nginx_chart.py`
- `tests/deploy/test_kserve_chart.py`
- `tests/deploy/test_ci_workflows.py`
- `tests/test_readme_deploy_docs.py`

**Modify:**
- `Makefile` — add deploy tool, Helm, dry-run, and Docker targets while preserving Plan 2 targets.
- `README.md` — document Docker/Helm/KServe/CI usage and the aarch64-to-x86 build note.
- `.gitignore` — ignore `.tools/`.

---

## Task 1: Deploy Tooling + Make Targets

**Files:**
- Create: `scripts/install_deploy_tools.py`
- Create: `tests/deploy/test_makefile_deploy_tools.py`
- Modify: `Makefile`
- Modify: `.gitignore`

- [ ] **Step 1: Write the failing tests**

Create `tests/deploy/test_makefile_deploy_tools.py`:

```python
from pathlib import Path


ROOT = Path(__file__).parents[2]


def test_makefile_exposes_deploy_targets_and_preserves_pipeline_targets():
    text = (ROOT / "Makefile").read_text()
    for target in [
        "install-deploy-tools:",
        "helm-lint:",
        "helm-template:",
        "helm-dry-run:",
        "docker-build:",
        "dvc-status:",
    ]:
        assert target in text
    assert ".venv/bin/dvc status" in text
    assert "deploy/helm/api" in text
    assert "deploy/helm/retrieval" in text
    assert "deploy/helm/nginx" in text
    assert "deploy/helm/kserve" in text
    assert "docker/api.Dockerfile" in text
    assert "docker/retrieval.Dockerfile" in text
    assert "docker/kserve-mock.Dockerfile" in text
    assert "docker/pipeline-init.Dockerfile" in text


def test_deploy_tool_installer_pins_helm_and_kubectl():
    text = (ROOT / "scripts/install_deploy_tools.py").read_text()
    assert 'HELM_VERSION = "v3.15.4"' in text
    assert 'KUBECTL_VERSION = "v1.30.4"' in text
    assert "get.helm.sh" in text
    assert "dl.k8s.io" in text
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/deploy/test_makefile_deploy_tools.py -v
```

Expected: FAIL because the deploy targets and installer script do not exist yet.

- [ ] **Step 3: Create the deploy tool installer**

Create `scripts/install_deploy_tools.py`:

```python
"""Install pinned Helm and kubectl binaries into a project-local tool directory."""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import stat
import tarfile
import tempfile
import urllib.request
from pathlib import Path


HELM_VERSION = "v3.15.4"
KUBECTL_VERSION = "v1.30.4"


def _linux_arch() -> str:
    machine = platform.machine().lower()
    if machine in {"x86_64", "amd64"}:
        return "amd64"
    if machine in {"aarch64", "arm64"}:
        return "arm64"
    raise RuntimeError(f"unsupported Linux architecture: {machine}")


def _download(url: str, destination: Path) -> None:
    with urllib.request.urlopen(url, timeout=60) as response:
        destination.write_bytes(response.read())


def _make_executable(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _install_helm(bin_dir: Path, arch: str) -> None:
    archive_name = f"helm-{HELM_VERSION}-linux-{arch}.tar.gz"
    url = f"https://get.helm.sh/{archive_name}"
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        archive = tmpdir / archive_name
        _download(url, archive)
        with tarfile.open(archive) as handle:
            handle.extractall(tmpdir)
        shutil.copy2(tmpdir / f"linux-{arch}" / "helm", bin_dir / "helm")
    _make_executable(bin_dir / "helm")


def _install_kubectl(bin_dir: Path, arch: str) -> None:
    url = (
        f"https://dl.k8s.io/release/{KUBECTL_VERSION}/bin/linux/{arch}/kubectl"
    )
    _download(url, bin_dir / "kubectl")
    _make_executable(bin_dir / "kubectl")


def install(bin_dir: Path) -> None:
    if platform.system().lower() != "linux":
        raise RuntimeError("this installer supports Linux developer/CI hosts only")
    arch = _linux_arch()
    bin_dir.mkdir(parents=True, exist_ok=True)
    _install_helm(bin_dir, arch)
    _install_kubectl(bin_dir, arch)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin-dir", default=".tools/bin")
    args = parser.parse_args()
    install(Path(args.bin_dir))
    checks = {
        "helm": "version --short",
        "kubectl": "version --client=true",
    }
    for name, command in checks.items():
        path = Path(args.bin_dir) / name
        os.system(f"{path} {command} >/dev/null")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Replace `Makefile` with the complete target set**

Use this exact `Makefile` content so the Plan 2 targets remain intact:

```makefile
PY := .venv/bin/python
PIP := .venv/bin/pip
DEPLOY_TOOLS_DIR := .tools
HELM := $(DEPLOY_TOOLS_DIR)/bin/helm
KUBECTL := $(DEPLOY_TOOLS_DIR)/bin/kubectl

.PHONY: install install-pipeline install-deploy-tools test smoke-pipeline smoke-pipeline-local mlflow-register-dry-run register-model dvc-status helm-lint helm-template helm-dry-run docker-build

.venv:
	python3.12 -m venv .venv
	$(PIP) install -U pip

install: .venv
	$(PIP) install -e ".[dev]"

install-pipeline: .venv
	$(PIP) install -e ".[dev,pipeline]"

install-deploy-tools: .venv
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

helm-lint: install-deploy-tools
	$(HELM) lint deploy/helm/api
	$(HELM) lint deploy/helm/retrieval
	$(HELM) lint deploy/helm/nginx
	$(HELM) lint deploy/helm/kserve

helm-template: install-deploy-tools
	$(HELM) template medical-qa-api deploy/helm/api >/dev/null
	$(HELM) template medical-qa-retrieval deploy/helm/retrieval >/dev/null
	$(HELM) template medical-qa-nginx deploy/helm/nginx >/dev/null
	$(HELM) template medical-qa-kserve deploy/helm/kserve >/dev/null

helm-dry-run: install-deploy-tools
	$(HELM) template medical-qa-api deploy/helm/api | $(KUBECTL) apply --dry-run=client --validate=false -f -
	$(HELM) template medical-qa-retrieval deploy/helm/retrieval | $(KUBECTL) apply --dry-run=client --validate=false -f -
	$(HELM) template medical-qa-nginx deploy/helm/nginx | $(KUBECTL) apply --dry-run=client --validate=false -f -

docker-build:
	docker buildx build --platform linux/amd64 -f docker/api.Dockerfile -t medical-qa-api:local --load .
	docker buildx build --platform linux/amd64 -f docker/retrieval.Dockerfile -t medical-qa-retrieval:local --load .
	docker buildx build --platform linux/amd64 -f docker/kserve-mock.Dockerfile -t medical-qa-kserve-mock:local --load .
	docker buildx build --platform linux/amd64 -f docker/pipeline-init.Dockerfile -t medical-qa-pipeline-init:local --load .
```

- [ ] **Step 5: Update `.gitignore`**

Append:

```gitignore
/.tools/
```

- [ ] **Step 6: Run tests to verify pass**

Run:

```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/deploy/test_makefile_deploy_tools.py -v
```

Expected: 2 passed.

- [ ] **Step 7: Commit**

```bash
cd /home/vcsai/minhlbq/mlops-platform
git add Makefile .gitignore scripts/install_deploy_tools.py tests/deploy/test_makefile_deploy_tools.py
git commit -m "chore: add deploy tooling targets"
```

---

## Task 2: KServe Mock Predictor App

**Files:**
- Create: `src/medical_qa_platform/serving/__init__.py`
- Create: `src/medical_qa_platform/serving/kserve_mock_app.py`
- Create: `tests/serving/test_kserve_mock_app.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/serving/test_kserve_mock_app.py`:

```python
from fastapi.testclient import TestClient

from medical_qa_platform.serving.kserve_mock_app import create_app


def test_kserve_mock_health_and_ready():
    client = TestClient(create_app())
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json() == {"status": "ready"}


def test_kserve_mock_predict_matches_kserve_backend_contract():
    client = TestClient(create_app())
    response = client.post(
        "/v1/models/medical-qa-smoke:predict",
        json={"instances": [{"messages": [{"role": "user", "content": "pick A"}]}]},
    )
    assert response.status_code == 200
    assert response.json() == {
        "predictions": [{"text": "<think>Mock reasoning for 1 messages.</think><answer>A</answer>"}]
    }


def test_kserve_mock_predict_short_path_for_local_smoke():
    client = TestClient(create_app())
    response = client.post(
        "/predict",
        json={"instances": [{"messages": [{"role": "user", "content": "pick A"}]}]},
    )
    assert response.status_code == 200
    assert response.json()["predictions"][0]["text"].endswith("<answer>A</answer>")
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/serving/test_kserve_mock_app.py -v
```

Expected: FAIL because `medical_qa_platform.serving.kserve_mock_app` does not exist.

- [ ] **Step 3: Add the serving package marker**

Create `src/medical_qa_platform/serving/__init__.py`:

```python
"""Serving entrypoints for deployable model backends."""
```

- [ ] **Step 4: Implement the mock predictor**

Create `src/medical_qa_platform/serving/kserve_mock_app.py`:

```python
"""KServe-compatible mock predictor used for CPU smoke serving."""

from fastapi import FastAPI
from pydantic import BaseModel, Field

from ..inference.mock_backend import MockBackend


class PredictInstance(BaseModel):
    messages: list[dict] = Field(default_factory=list)


class PredictRequest(BaseModel):
    instances: list[PredictInstance]


class Prediction(BaseModel):
    text: str


class PredictResponse(BaseModel):
    predictions: list[Prediction]


def create_app(backend: MockBackend | None = None) -> FastAPI:
    app = FastAPI(title="Medical QA KServe Mock Predictor")
    model = backend or MockBackend()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        return {"status": "ready"}

    def _predict(req: PredictRequest) -> PredictResponse:
        predictions = [
            Prediction(text=model.generate(instance.messages))
            for instance in req.instances
        ]
        return PredictResponse(predictions=predictions)

    @app.post("/v1/models/{model_name}:predict", response_model=PredictResponse)
    def predict_v1(model_name: str, req: PredictRequest):
        return _predict(req)

    @app.post("/predict", response_model=PredictResponse)
    def predict_short(req: PredictRequest):
        return _predict(req)

    return app
```

- [ ] **Step 5: Run tests to verify pass**

Run:

```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/serving/test_kserve_mock_app.py -v
```

Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
cd /home/vcsai/minhlbq/mlops-platform
git add src/medical_qa_platform/serving tests/serving/test_kserve_mock_app.py
git commit -m "feat: add KServe mock predictor app"
```

---

## Task 3: Dockerfiles

**Files:**
- Create: `docker/api.Dockerfile`
- Create: `docker/retrieval.Dockerfile`
- Create: `docker/kserve-mock.Dockerfile`
- Create: `docker/pipeline-init.Dockerfile`
- Create: `tests/deploy/test_dockerfiles.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/deploy/test_dockerfiles.py`:

```python
from pathlib import Path


ROOT = Path(__file__).parents[2]


def _read(name: str) -> str:
    return (ROOT / "docker" / name).read_text()


def test_api_dockerfile_is_lean_non_root_and_binds_all_interfaces():
    text = _read("api.Dockerfile")
    assert "FROM python:3.12-slim" in text
    assert "pip install --no-cache-dir ." in text
    assert ".[runtime]" not in text
    assert ".[pipeline]" not in text
    assert "USER app" in text
    assert "EXPOSE 8000" in text
    assert '"--host", "0.0.0.0"' in text
    assert '"--port", "8000"' in text
    assert "medical_qa_platform.api.app:create_app" in text


def test_retrieval_dockerfile_installs_runtime_only_and_does_not_bake_artifacts():
    text = _read("retrieval.Dockerfile")
    assert "pip install --no-cache-dir '.[runtime]'" in text
    assert "USER app" in text
    assert "EXPOSE 8001" in text
    assert "RETRIEVAL_DEVICE=cpu" in text
    assert "HF_HOME=/mnt/artifacts/hf" in text
    assert "SENTENCE_TRANSFORMERS_HOME=/mnt/artifacts/hf/sentence-transformers" in text
    assert '"--host", "0.0.0.0"' in text
    assert '"--port", "8001"' in text
    forbidden = ["baseline", "medical_hg.json", "index_hyperedge.bin", "index_entity.bin", "hedge_ids.npy", "entity_names.npy"]
    for needle in forbidden:
        assert needle not in text


def test_kserve_mock_dockerfile_runs_mock_predictor_on_8080():
    text = _read("kserve-mock.Dockerfile")
    assert "pip install --no-cache-dir ." in text
    assert "USER app" in text
    assert "EXPOSE 8080" in text
    assert "medical_qa_platform.serving.kserve_mock_app:create_app" in text
    assert '"--host", "0.0.0.0"' in text
    assert '"--port", "8080"' in text


def test_pipeline_init_dockerfile_contains_dvc_but_not_runtime_ml_dependencies():
    text = _read("pipeline-init.Dockerfile")
    assert "pip install --no-cache-dir '.[pipeline]'" in text
    assert ".[runtime]" not in text
    assert "USER app" in text
    assert 'CMD ["dvc", "--version"]' in text
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/deploy/test_dockerfiles.py -v
```

Expected: FAIL because `docker/` files do not exist.

- [ ] **Step 3: Add `docker/api.Dockerfile`**

```dockerfile
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN addgroup --system app && adduser --system --ingroup app app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

USER app
EXPOSE 8000

CMD ["uvicorn", "medical_qa_platform.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 4: Add `docker/retrieval.Dockerfile`**

```dockerfile
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RETRIEVAL_DEVICE=cpu \
    KG_DATA_DIR=/mnt/artifacts/smoke/kg \
    HF_HOME=/mnt/artifacts/hf \
    SENTENCE_TRANSFORMERS_HOME=/mnt/artifacts/hf/sentence-transformers

WORKDIR /app

RUN addgroup --system app && adduser --system --ingroup app app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir '.[runtime]'

USER app
EXPOSE 8001

CMD ["uvicorn", "medical_qa_platform.retrieval.service:create_retrieval_service", "--factory", "--host", "0.0.0.0", "--port", "8001"]
```

- [ ] **Step 5: Add `docker/kserve-mock.Dockerfile`**

```dockerfile
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN addgroup --system app && adduser --system --ingroup app app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

USER app
EXPOSE 8080

CMD ["uvicorn", "medical_qa_platform.serving.kserve_mock_app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080"]
```

- [ ] **Step 6: Add `docker/pipeline-init.Dockerfile`**

```dockerfile
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

RUN addgroup --system app && adduser --system --ingroup app app && \
    mkdir -p /workspace/.dvc /workspace/artifacts && \
    chown -R app:app /workspace

COPY pyproject.toml README.md ./
COPY src ./src
COPY mlops ./mlops

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir '.[pipeline]'

USER app

CMD ["dvc", "--version"]
```

- [ ] **Step 7: Run tests to verify pass**

Run:

```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/deploy/test_dockerfiles.py -v
```

Expected: 4 passed.

- [ ] **Step 8: Commit**

```bash
cd /home/vcsai/minhlbq/mlops-platform
git add docker tests/deploy/test_dockerfiles.py
git commit -m "chore: add deploy Dockerfiles"
```

---

## Task 4: Helm Test Helpers

**Files:**
- Create: `tests/deploy/helm_helpers.py`

- [ ] **Step 1: Create Helm render helper**

Create `tests/deploy/helm_helpers.py`:

```python
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).parents[2]
HELM = ROOT / ".tools/bin/helm"


def require_helm() -> Path:
    if not HELM.exists():
        pytest.skip("project-local helm is not installed; run make install-deploy-tools")
    return HELM


def render_chart(chart_name: str) -> list[dict]:
    helm = require_helm()
    chart = ROOT / "deploy/helm" / chart_name
    result = subprocess.run(
        [str(helm), "template", f"test-{chart_name}", str(chart)],
        check=True,
        capture_output=True,
        text=True,
    )
    return [
        doc
        for doc in yaml.safe_load_all(result.stdout)
        if isinstance(doc, dict) and doc
    ]


def find_kind(resources: list[dict], kind: str, name: str | None = None) -> dict:
    for resource in resources:
        if resource.get("kind") != kind:
            continue
        if name is None or resource.get("metadata", {}).get("name") == name:
            return resource
    available = [
        f"{resource.get('kind')}/{resource.get('metadata', {}).get('name')}"
        for resource in resources
    ]
    raise AssertionError(f"missing {kind}/{name}; available={available}")
```

- [ ] **Step 2: Commit**

```bash
cd /home/vcsai/minhlbq/mlops-platform
git add tests/deploy/helm_helpers.py
git commit -m "test: add Helm chart render helpers"
```

---

## Task 5: API Helm Chart

**Files:**
- Create: `deploy/helm/api/Chart.yaml`
- Create: `deploy/helm/api/values.yaml`
- Create: `deploy/helm/api/templates/configmap.yaml`
- Create: `deploy/helm/api/templates/deployment.yaml`
- Create: `deploy/helm/api/templates/service.yaml`
- Create: `deploy/helm/api/templates/hpa.yaml`
- Create: `tests/deploy/test_helm_api_chart.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/deploy/test_helm_api_chart.py`:

```python
from pathlib import Path

import yaml

from tests.deploy.helm_helpers import find_kind, render_chart


ROOT = Path(__file__).parents[2]


def test_api_values_configure_retrieval_service_not_localhost():
    values = yaml.safe_load((ROOT / "deploy/helm/api/values.yaml").read_text())
    assert values["env"]["retrievalUrl"] == "http://medical-qa-retrieval:8001"
    assert "localhost" not in values["env"]["retrievalUrl"]


def test_api_chart_renders_deployment_service_configmap_and_hpa():
    resources = render_chart("api")
    config = find_kind(resources, "ConfigMap", "medical-qa-api-config")
    deployment = find_kind(resources, "Deployment", "medical-qa-api")
    service = find_kind(resources, "Service", "medical-qa-api")
    hpa = find_kind(resources, "HorizontalPodAutoscaler", "medical-qa-api")

    assert config["data"]["RETRIEVAL_URL"] == "http://medical-qa-retrieval:8001"
    assert config["data"]["MODEL_BACKEND"] == "mock"
    assert service["spec"]["ports"][0]["port"] == 8000
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    assert container["ports"][0]["containerPort"] == 8000
    assert container["livenessProbe"]["httpGet"]["path"] == "/health"
    assert container["readinessProbe"]["httpGet"]["path"] == "/ready"
    assert hpa["apiVersion"] == "autoscaling/v2"
    assert hpa["spec"]["metrics"][0]["resource"]["target"]["averageUtilization"] == 70
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/deploy/test_helm_api_chart.py -v
```

Expected: FAIL because the API chart does not exist. If Helm is not installed, the render test skips, but `test_api_values_configure_retrieval_service_not_localhost` still fails.

- [ ] **Step 3: Create API chart files**

Create `deploy/helm/api/Chart.yaml`:

```yaml
apiVersion: v2
name: medical-qa-api
description: FastAPI pre/post-processing API for Medical QA
type: application
version: 0.1.0
appVersion: "0.1.0"
```

Create `deploy/helm/api/values.yaml`:

```yaml
replicaCount: 1

image:
  repository: ghcr.io/hinmq21/medical-qa-api
  tag: latest
  pullPolicy: IfNotPresent

service:
  port: 8000

env:
  modelBackend: mock
  modelVersion: smoke-dev
  retrievalUrl: http://medical-qa-retrieval:8001
  runpodBaseUrl: ""
  runpodModel: ""
  kserveUrl: http://medical-qa-kserve/v1/models/medical-qa-smoke:predict
  topK: "5"

resources:
  requests:
    cpu: 100m
    memory: 256Mi
  limits:
    cpu: 500m
    memory: 512Mi

autoscaling:
  minReplicas: 1
  maxReplicas: 3
  targetCPUUtilizationPercentage: 70
```

Create `deploy/helm/api/templates/configmap.yaml`:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: medical-qa-api-config
data:
  MODEL_BACKEND: {{ .Values.env.modelBackend | quote }}
  MODEL_VERSION: {{ .Values.env.modelVersion | quote }}
  RETRIEVAL_URL: {{ .Values.env.retrievalUrl | quote }}
  RUNPOD_BASE_URL: {{ .Values.env.runpodBaseUrl | quote }}
  RUNPOD_MODEL: {{ .Values.env.runpodModel | quote }}
  KSERVE_URL: {{ .Values.env.kserveUrl | quote }}
  TOP_K: {{ .Values.env.topK | quote }}
```

Create `deploy/helm/api/templates/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: medical-qa-api
  labels:
    app.kubernetes.io/name: medical-qa-api
spec:
  replicas: {{ .Values.replicaCount }}
  selector:
    matchLabels:
      app.kubernetes.io/name: medical-qa-api
  template:
    metadata:
      labels:
        app.kubernetes.io/name: medical-qa-api
    spec:
      containers:
        - name: api
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          envFrom:
            - configMapRef:
                name: medical-qa-api-config
          ports:
            - name: http
              containerPort: 8000
          livenessProbe:
            httpGet:
              path: /health
              port: http
            initialDelaySeconds: 10
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /ready
              port: http
            initialDelaySeconds: 5
            periodSeconds: 10
          resources:
{{ toYaml .Values.resources | indent 12 }}
```

Create `deploy/helm/api/templates/service.yaml`:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: medical-qa-api
spec:
  type: ClusterIP
  selector:
    app.kubernetes.io/name: medical-qa-api
  ports:
    - name: http
      port: {{ .Values.service.port }}
      targetPort: http
```

Create `deploy/helm/api/templates/hpa.yaml`:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: medical-qa-api
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: medical-qa-api
  minReplicas: {{ .Values.autoscaling.minReplicas }}
  maxReplicas: {{ .Values.autoscaling.maxReplicas }}
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: {{ .Values.autoscaling.targetCPUUtilizationPercentage }}
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/deploy/test_helm_api_chart.py -v
```

Expected without `.tools/bin/helm`: 1 passed, 1 skipped. Expected after `make install-deploy-tools`: 2 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/vcsai/minhlbq/mlops-platform
git add deploy/helm/api tests/deploy/test_helm_api_chart.py
git commit -m "feat: add API Helm chart"
```

---

## Task 6: Retrieval Helm Chart

**Files:**
- Create: `deploy/helm/retrieval/Chart.yaml`
- Create: `deploy/helm/retrieval/values.yaml`
- Create: `deploy/helm/retrieval/templates/configmap.yaml`
- Create: `deploy/helm/retrieval/templates/pvc.yaml`
- Create: `deploy/helm/retrieval/templates/deployment.yaml`
- Create: `deploy/helm/retrieval/templates/service.yaml`
- Create: `deploy/helm/retrieval/templates/hpa.yaml`
- Create: `tests/deploy/test_helm_retrieval_chart.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/deploy/test_helm_retrieval_chart.py`:

```python
from pathlib import Path

import yaml

from tests.deploy.helm_helpers import find_kind, render_chart


ROOT = Path(__file__).parents[2]


def test_retrieval_values_use_runtime_and_dvc_init_images():
    values = yaml.safe_load((ROOT / "deploy/helm/retrieval/values.yaml").read_text())
    assert values["image"]["repository"].endswith("medical-qa-retrieval")
    assert values["initImage"]["repository"].endswith("medical-qa-pipeline-init")
    assert values["env"]["retrievalDevice"] == "cpu"


def test_retrieval_chart_renders_dvc_init_container_pvc_and_hpa():
    resources = render_chart("retrieval")
    config = find_kind(resources, "ConfigMap", "medical-qa-retrieval-dvc")
    pvc = find_kind(resources, "PersistentVolumeClaim", "medical-qa-retrieval-artifacts")
    deployment = find_kind(resources, "Deployment", "medical-qa-retrieval")
    service = find_kind(resources, "Service", "medical-qa-retrieval")
    hpa = find_kind(resources, "HorizontalPodAutoscaler", "medical-qa-retrieval")

    assert {"dvc-config", "dvc-yaml", "dvc-lock", "artifacts-dvc"} <= set(config["data"])
    assert pvc["spec"]["resources"]["requests"]["storage"] == "5Gi"
    pod_spec = deployment["spec"]["template"]["spec"]
    init = pod_spec["initContainers"][0]
    container = pod_spec["containers"][0]
    assert init["name"] == "dvc-pull"
    assert init["image"].endswith("medical-qa-pipeline-init:latest")
    assert "dvc pull --no-run-cache" in init["args"][0]
    mount_names = {mount["name"] for mount in init["volumeMounts"]}
    assert {"artifacts", "dvc-metadata"} <= mount_names
    env = {item["name"]: item["value"] for item in container["env"]}
    assert env["RETRIEVAL_DEVICE"] == "cpu"
    assert env["KG_DATA_DIR"] == "/mnt/artifacts/smoke/kg"
    assert env["HF_HOME"] == "/mnt/artifacts/hf"
    assert env["SENTENCE_TRANSFORMERS_HOME"] == "/mnt/artifacts/hf/sentence-transformers"
    assert service["spec"]["ports"][0]["port"] == 8001
    assert hpa["spec"]["metrics"][0]["resource"]["target"]["averageUtilization"] == 70
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/deploy/test_helm_retrieval_chart.py -v
```

Expected: FAIL because the retrieval chart does not exist. If Helm is not installed, the render test skips, but the values test still fails.

- [ ] **Step 3: Create retrieval chart files**

Create `deploy/helm/retrieval/Chart.yaml`:

```yaml
apiVersion: v2
name: medical-qa-retrieval
description: Medical KG retrieval service with DVC-populated artifacts
type: application
version: 0.1.0
appVersion: "0.1.0"
```

Create `deploy/helm/retrieval/values.yaml`:

```yaml
replicaCount: 1

image:
  repository: ghcr.io/hinmq21/medical-qa-retrieval
  tag: latest
  pullPolicy: IfNotPresent

initImage:
  repository: ghcr.io/hinmq21/medical-qa-pipeline-init
  tag: latest
  pullPolicy: IfNotPresent

service:
  port: 8001

persistence:
  storageClassName: ""
  size: 5Gi
  mountPath: /mnt/artifacts

env:
  retrievalBackend: kg
  retrievalDevice: cpu
  kgDataDir: /mnt/artifacts/smoke/kg
  hfHome: /mnt/artifacts/hf
  sentenceTransformersHome: /mnt/artifacts/hf/sentence-transformers

dvc:
  config: |
    [core]
        remote = gcsremote
    ['remote "gcsremote"']
        url = gs://medical-qa-plan4-wiring/dvc
  yaml: |
    stages: {}
  lock: |
    schema: '2.0'
  artifactsDvc: |
    outs: []

resources:
  requests:
    cpu: 500m
    memory: 2Gi
  limits:
    cpu: "2"
    memory: 4Gi

autoscaling:
  minReplicas: 1
  maxReplicas: 3
  targetCPUUtilizationPercentage: 70
```

Create `deploy/helm/retrieval/templates/configmap.yaml`:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: medical-qa-retrieval-dvc
data:
  dvc-config: |-
{{ .Values.dvc.config | indent 4 }}
  dvc-yaml: |-
{{ .Values.dvc.yaml | indent 4 }}
  dvc-lock: |-
{{ .Values.dvc.lock | indent 4 }}
  artifacts-dvc: |-
{{ .Values.dvc.artifactsDvc | indent 4 }}
```

Create `deploy/helm/retrieval/templates/pvc.yaml`:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: medical-qa-retrieval-artifacts
spec:
  accessModes:
    - ReadWriteOnce
{{- if .Values.persistence.storageClassName }}
  storageClassName: {{ .Values.persistence.storageClassName | quote }}
{{- end }}
  resources:
    requests:
      storage: {{ .Values.persistence.size }}
```

Create `deploy/helm/retrieval/templates/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: medical-qa-retrieval
  labels:
    app.kubernetes.io/name: medical-qa-retrieval
spec:
  replicas: {{ .Values.replicaCount }}
  selector:
    matchLabels:
      app.kubernetes.io/name: medical-qa-retrieval
  template:
    metadata:
      labels:
        app.kubernetes.io/name: medical-qa-retrieval
    spec:
      initContainers:
        - name: dvc-pull
          image: "{{ .Values.initImage.repository }}:{{ .Values.initImage.tag }}"
          imagePullPolicy: {{ .Values.initImage.pullPolicy }}
          workingDir: /workspace
          command:
            - /bin/sh
            - -c
          args:
            - dvc pull --no-run-cache
          volumeMounts:
            - name: artifacts
              mountPath: /workspace/artifacts
            - name: dvc-metadata
              mountPath: /workspace/.dvc/config
              subPath: dvc-config
            - name: dvc-metadata
              mountPath: /workspace/dvc.yaml
              subPath: dvc-yaml
            - name: dvc-metadata
              mountPath: /workspace/dvc.lock
              subPath: dvc-lock
            - name: dvc-metadata
              mountPath: /workspace/artifacts.dvc
              subPath: artifacts-dvc
      containers:
        - name: retrieval
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          env:
            - name: RETRIEVAL_BACKEND
              value: {{ .Values.env.retrievalBackend | quote }}
            - name: RETRIEVAL_DEVICE
              value: {{ .Values.env.retrievalDevice | quote }}
            - name: KG_DATA_DIR
              value: {{ .Values.env.kgDataDir | quote }}
            - name: HF_HOME
              value: {{ .Values.env.hfHome | quote }}
            - name: SENTENCE_TRANSFORMERS_HOME
              value: {{ .Values.env.sentenceTransformersHome | quote }}
          ports:
            - name: http
              containerPort: 8001
          volumeMounts:
            - name: artifacts
              mountPath: {{ .Values.persistence.mountPath }}
          livenessProbe:
            httpGet:
              path: /health
              port: http
            initialDelaySeconds: 10
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /ready
              port: http
            initialDelaySeconds: 20
            periodSeconds: 10
          resources:
{{ toYaml .Values.resources | indent 12 }}
      volumes:
        - name: artifacts
          persistentVolumeClaim:
            claimName: medical-qa-retrieval-artifacts
        - name: dvc-metadata
          configMap:
            name: medical-qa-retrieval-dvc
```

Create `deploy/helm/retrieval/templates/service.yaml`:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: medical-qa-retrieval
spec:
  type: ClusterIP
  selector:
    app.kubernetes.io/name: medical-qa-retrieval
  ports:
    - name: http
      port: {{ .Values.service.port }}
      targetPort: http
```

Create `deploy/helm/retrieval/templates/hpa.yaml`:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: medical-qa-retrieval
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: medical-qa-retrieval
  minReplicas: {{ .Values.autoscaling.minReplicas }}
  maxReplicas: {{ .Values.autoscaling.maxReplicas }}
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: {{ .Values.autoscaling.targetCPUUtilizationPercentage }}
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/deploy/test_helm_retrieval_chart.py -v
```

Expected without `.tools/bin/helm`: 1 passed, 1 skipped. Expected after `make install-deploy-tools`: 2 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/vcsai/minhlbq/mlops-platform
git add deploy/helm/retrieval tests/deploy/test_helm_retrieval_chart.py
git commit -m "feat: add retrieval Helm chart"
```

---

## Task 7: NGINX Gateway Helm Chart

**Files:**
- Create: `deploy/helm/nginx/Chart.yaml`
- Create: `deploy/helm/nginx/values.yaml`
- Create: `deploy/helm/nginx/templates/configmap.yaml`
- Create: `deploy/helm/nginx/templates/secret.yaml`
- Create: `deploy/helm/nginx/templates/deployment.yaml`
- Create: `deploy/helm/nginx/templates/service.yaml`
- Create: `tests/deploy/test_helm_nginx_chart.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/deploy/test_helm_nginx_chart.py`:

```python
from pathlib import Path

from tests.deploy.helm_helpers import find_kind, render_chart


ROOT = Path(__file__).parents[2]


def test_nginx_configmap_template_does_not_contain_literal_default_key():
    text = (ROOT / "deploy/helm/nginx/templates/configmap.yaml").read_text()
    assert "${API_KEY}" in text
    assert "change-me-dev-key" not in text
    assert "proxy_pass http://medical-qa-api:8000" in text


def test_nginx_chart_renders_configmap_secret_deployment_and_service():
    resources = render_chart("nginx")
    config = find_kind(resources, "ConfigMap", "medical-qa-nginx-config")
    secret = find_kind(resources, "Secret", "medical-qa-nginx-api-key")
    deployment = find_kind(resources, "Deployment", "medical-qa-nginx")
    service = find_kind(resources, "Service", "medical-qa-nginx")

    conf = config["data"]["default.conf.template"]
    assert "${API_KEY}" in conf
    assert "change-me-dev-key" not in conf
    assert "proxy_pass http://medical-qa-api:8000" in conf
    assert secret["stringData"]["API_KEY"] == "change-me-dev-key"
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    env = container["env"][0]
    assert env["name"] == "API_KEY"
    assert env["valueFrom"]["secretKeyRef"]["name"] == "medical-qa-nginx-api-key"
    assert service["spec"]["ports"][0]["port"] == 8080
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/deploy/test_helm_nginx_chart.py -v
```

Expected: FAIL because the NGINX chart does not exist. If Helm is not installed, the render test skips, but the ConfigMap-template test still fails.

- [ ] **Step 3: Create NGINX chart files**

Create `deploy/helm/nginx/Chart.yaml`:

```yaml
apiVersion: v2
name: medical-qa-nginx
description: NGINX gateway with API-key auth for Medical QA
type: application
version: 0.1.0
appVersion: "1.27"
```

Create `deploy/helm/nginx/values.yaml`:

```yaml
replicaCount: 1

image:
  repository: nginx
  tag: "1.27-alpine"
  pullPolicy: IfNotPresent

auth:
  apiKey: change-me-dev-key

service:
  port: 8080

upstream:
  apiUrl: http://medical-qa-api:8000

resources:
  requests:
    cpu: 50m
    memory: 64Mi
  limits:
    cpu: 250m
    memory: 128Mi
```

Create `deploy/helm/nginx/templates/configmap.yaml`:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: medical-qa-nginx-config
data:
  default.conf.template: |-
    map $http_x_api_key $api_key_valid {
      default 0;
      "${API_KEY}" 1;
    }

    server {
      listen 8080;

      location /health {
        return 200 "ok\n";
      }

      location / {
        if ($api_key_valid = 0) {
          return 401;
        }
        proxy_set_header Host $host;
        proxy_set_header X-Request-ID $request_id;
        proxy_pass {{ .Values.upstream.apiUrl }};
      }
    }
```

Create `deploy/helm/nginx/templates/secret.yaml`:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: medical-qa-nginx-api-key
type: Opaque
stringData:
  API_KEY: {{ .Values.auth.apiKey | quote }}
```

Create `deploy/helm/nginx/templates/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: medical-qa-nginx
  labels:
    app.kubernetes.io/name: medical-qa-nginx
spec:
  replicas: {{ .Values.replicaCount }}
  selector:
    matchLabels:
      app.kubernetes.io/name: medical-qa-nginx
  template:
    metadata:
      labels:
        app.kubernetes.io/name: medical-qa-nginx
    spec:
      containers:
        - name: nginx
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          env:
            - name: API_KEY
              valueFrom:
                secretKeyRef:
                  name: medical-qa-nginx-api-key
                  key: API_KEY
          ports:
            - name: http
              containerPort: 8080
          volumeMounts:
            - name: nginx-config
              mountPath: /etc/nginx/templates/default.conf.template
              subPath: default.conf.template
          livenessProbe:
            httpGet:
              path: /health
              port: http
            initialDelaySeconds: 5
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /health
              port: http
            initialDelaySeconds: 5
            periodSeconds: 10
          resources:
{{ toYaml .Values.resources | indent 12 }}
      volumes:
        - name: nginx-config
          configMap:
            name: medical-qa-nginx-config
```

Create `deploy/helm/nginx/templates/service.yaml`:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: medical-qa-nginx
spec:
  type: ClusterIP
  selector:
    app.kubernetes.io/name: medical-qa-nginx
  ports:
    - name: http
      port: {{ .Values.service.port }}
      targetPort: http
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/deploy/test_helm_nginx_chart.py -v
```

Expected without `.tools/bin/helm`: 1 passed, 1 skipped. Expected after `make install-deploy-tools`: 2 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/vcsai/minhlbq/mlops-platform
git add deploy/helm/nginx tests/deploy/test_helm_nginx_chart.py
git commit -m "feat: add NGINX gateway Helm chart"
```

---

## Task 8: KServe Helm Chart

**Files:**
- Create: `deploy/helm/kserve/Chart.yaml`
- Create: `deploy/helm/kserve/values.yaml`
- Create: `deploy/helm/kserve/templates/inferenceservice.yaml`
- Create: `tests/deploy/test_kserve_chart.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/deploy/test_kserve_chart.py`:

```python
from pathlib import Path

import yaml

from tests.deploy.helm_helpers import find_kind, render_chart


ROOT = Path(__file__).parents[2]


def test_kserve_values_default_to_scale_to_zero():
    values = yaml.safe_load((ROOT / "deploy/helm/kserve/values.yaml").read_text())
    assert values["minReplicas"] == 0
    assert values["image"]["repository"].endswith("medical-qa-kserve-mock")


def test_kserve_chart_renders_inferenceservice_with_min_replicas_zero():
    resources = render_chart("kserve")
    service = find_kind(resources, "InferenceService", "medical-qa-kserve")
    assert service["apiVersion"] == "serving.kserve.io/v1beta1"
    predictor = service["spec"]["predictor"]
    assert predictor["minReplicas"] == 0
    container = predictor["containers"][0]
    assert container["image"].endswith("medical-qa-kserve-mock:latest")
    assert container["ports"][0]["containerPort"] == 8080
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/deploy/test_kserve_chart.py -v
```

Expected: FAIL because the KServe chart does not exist. If Helm is not installed, the render test skips, but the values test still fails.

- [ ] **Step 3: Create KServe chart files**

Create `deploy/helm/kserve/Chart.yaml`:

```yaml
apiVersion: v2
name: medical-qa-kserve
description: KServe CPU smoke/mock InferenceService for Medical QA
type: application
version: 0.1.0
appVersion: "0.1.0"
```

Create `deploy/helm/kserve/values.yaml`:

```yaml
image:
  repository: ghcr.io/hinmq21/medical-qa-kserve-mock
  tag: latest
  pullPolicy: IfNotPresent

minReplicas: 0

resources:
  requests:
    cpu: 100m
    memory: 256Mi
  limits:
    cpu: 500m
    memory: 512Mi
```

Create `deploy/helm/kserve/templates/inferenceservice.yaml`:

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: medical-qa-kserve
spec:
  predictor:
    minReplicas: {{ .Values.minReplicas }}
    containers:
      - name: kserve-container
        image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
        imagePullPolicy: {{ .Values.image.pullPolicy }}
        ports:
          - containerPort: 8080
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 10
        resources:
{{ toYaml .Values.resources | indent 10 }}
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/deploy/test_kserve_chart.py -v
```

Expected without `.tools/bin/helm`: 1 passed, 1 skipped. Expected after `make install-deploy-tools`: 2 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/vcsai/minhlbq/mlops-platform
git add deploy/helm/kserve tests/deploy/test_kserve_chart.py
git commit -m "feat: add KServe mock Helm chart"
```

---

## Task 9: GitHub Actions CI and Deploy Skeleton

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `.github/workflows/deploy.yml`
- Create: `tests/deploy/test_ci_workflows.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/deploy/test_ci_workflows.py`:

```python
from pathlib import Path

import yaml


ROOT = Path(__file__).parents[2]


def _workflow(name: str) -> dict:
    return yaml.safe_load((ROOT / ".github/workflows" / name).read_text())


def _all_run_commands(workflow: dict) -> str:
    commands = []
    for job in workflow["jobs"].values():
        for step in job["steps"]:
            if "run" in step:
                commands.append(step["run"])
    return "\n".join(commands)


def test_ci_workflow_runs_tests_helm_checks_and_builds_images():
    workflow = _workflow("ci.yml")
    commands = _all_run_commands(workflow)
    assert "make install-pipeline" in commands
    assert "make install-deploy-tools" in commands
    assert "make helm-lint" in commands
    assert "make helm-template" in commands
    assert "--cov-fail-under=80" in commands
    assert "docker/api.Dockerfile" in commands
    assert "docker/retrieval.Dockerfile" in commands
    assert "docker/kserve-mock.Dockerfile" in commands
    assert "docker/pipeline-init.Dockerfile" in commands


def test_deploy_workflow_is_manual_dispatch_skeleton():
    workflow = _workflow("deploy.yml")
    triggers = workflow.get("on", workflow.get(True))
    assert "workflow_dispatch" in triggers
    commands = _all_run_commands(workflow)
    assert "helm upgrade --install medical-qa-api deploy/helm/api" in commands
    assert "helm upgrade --install medical-qa-retrieval deploy/helm/retrieval" in commands
    assert "helm upgrade --install medical-qa-nginx deploy/helm/nginx" in commands
    assert "helm upgrade --install medical-qa-kserve deploy/helm/kserve" in commands
    assert "/health" in commands
    assert "/predict" in commands
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/deploy/test_ci_workflows.py -v
```

Expected: FAIL because workflow files do not exist.

- [ ] **Step 3: Create CI workflow**

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

env:
  REGISTRY: ghcr.io/hinmq21

jobs:
  test-build:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install Python dependencies
        run: make install-pipeline

      - name: Install deploy tools
        run: make install-deploy-tools

      - name: Helm lint
        run: make helm-lint

      - name: Helm template
        run: make helm-template

      - name: Pytest
        run: .venv/bin/pytest --cov=medical_qa_platform --cov-report=term-missing --cov-fail-under=80

      - uses: docker/setup-buildx-action@v3

      - name: Build API image
        run: docker buildx build --platform linux/amd64 -f docker/api.Dockerfile -t $REGISTRY/medical-qa-api:${{ github.sha }} --load .

      - name: Build retrieval image
        run: docker buildx build --platform linux/amd64 -f docker/retrieval.Dockerfile -t $REGISTRY/medical-qa-retrieval:${{ github.sha }} --load .

      - name: Build KServe mock image
        run: docker buildx build --platform linux/amd64 -f docker/kserve-mock.Dockerfile -t $REGISTRY/medical-qa-kserve-mock:${{ github.sha }} --load .

      - name: Build DVC init image
        run: docker buildx build --platform linux/amd64 -f docker/pipeline-init.Dockerfile -t $REGISTRY/medical-qa-pipeline-init:${{ github.sha }} --load .

      - name: Login to GHCR
        if: github.ref == 'refs/heads/main'
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Push images
        if: github.ref == 'refs/heads/main'
        run: |
          docker buildx build --platform linux/amd64 -f docker/api.Dockerfile -t $REGISTRY/medical-qa-api:${{ github.sha }} -t $REGISTRY/medical-qa-api:latest --push .
          docker buildx build --platform linux/amd64 -f docker/retrieval.Dockerfile -t $REGISTRY/medical-qa-retrieval:${{ github.sha }} -t $REGISTRY/medical-qa-retrieval:latest --push .
          docker buildx build --platform linux/amd64 -f docker/kserve-mock.Dockerfile -t $REGISTRY/medical-qa-kserve-mock:${{ github.sha }} -t $REGISTRY/medical-qa-kserve-mock:latest --push .
          docker buildx build --platform linux/amd64 -f docker/pipeline-init.Dockerfile -t $REGISTRY/medical-qa-pipeline-init:${{ github.sha }} -t $REGISTRY/medical-qa-pipeline-init:latest --push .
```

- [ ] **Step 4: Create deploy workflow skeleton**

Create `.github/workflows/deploy.yml`:

```yaml
name: Manual Deploy

on:
  workflow_dispatch:
    inputs:
      image_tag:
        description: Image tag to deploy
        required: true
        default: latest
      namespace:
        description: Kubernetes namespace
        required: true
        default: medical-qa

jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: plan4-cluster
    steps:
      - uses: actions/checkout@v4

      - name: Install Python dependencies
        run: make install-pipeline

      - name: Install deploy tools
        run: make install-deploy-tools

      - name: Authenticate to cluster in Plan 4
        run: |
          echo "Plan 4 supplies cloud auth and kubeconfig before helm upgrade."

      - name: Helm upgrade app-serving charts
        run: |
          helm upgrade --install medical-qa-api deploy/helm/api --namespace "${{ inputs.namespace }}" --create-namespace --set image.tag="${{ inputs.image_tag }}"
          helm upgrade --install medical-qa-retrieval deploy/helm/retrieval --namespace "${{ inputs.namespace }}" --set image.tag="${{ inputs.image_tag }}" --set initImage.tag="${{ inputs.image_tag }}"
          helm upgrade --install medical-qa-nginx deploy/helm/nginx --namespace "${{ inputs.namespace }}"
          helm upgrade --install medical-qa-kserve deploy/helm/kserve --namespace "${{ inputs.namespace }}" --set image.tag="${{ inputs.image_tag }}"

      - name: Smoke tests
        run: |
          curl -fsS "$BASE_URL/health"
          curl -fsS -H "x-api-key: $API_KEY" "$BASE_URL/health"
          curl -fsS -X POST -H "x-api-key: $API_KEY" -H "content-type: application/json" "$BASE_URL/predict" -d '{"question":"Which drug is first-line for type 2 diabetes?","options":{"A":"Metformin","B":"Amoxicillin"}}'
```

- [ ] **Step 5: Run tests to verify pass**

Run:

```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/deploy/test_ci_workflows.py -v
```

Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
cd /home/vcsai/minhlbq/mlops-platform
git add .github/workflows tests/deploy/test_ci_workflows.py
git commit -m "ci: add test build and deploy workflows"
```

---

## Task 10: README Deployment Documentation

**Files:**
- Modify: `README.md`
- Create: `tests/test_readme_deploy_docs.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_readme_deploy_docs.py`:

```python
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_readme_documents_plan3_deploy_artifacts():
    text = (ROOT / "README.md").read_text()
    for expected in [
        "make install-deploy-tools",
        "make helm-lint",
        "make helm-template",
        "make helm-dry-run",
        "make docker-build",
        "docker/api.Dockerfile",
        "docker/retrieval.Dockerfile",
        "docker/kserve-mock.Dockerfile",
        "docker/pipeline-init.Dockerfile",
        "KServe",
        "NGINX",
        "aarch64",
        "linux/amd64",
    ]:
        assert expected in text
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/test_readme_deploy_docs.py -v
```

Expected: FAIL because README does not document Plan 3 deployment artifacts yet.

- [ ] **Step 3: Append deployment docs to `README.md`**

Append:

````markdown
## Docker, Helm, KServe, and CI

Plan 3 adds deployable app-serving artifacts for the runtime package:

- `docker/api.Dockerfile` builds the FastAPI pre/post-processing API.
- `docker/retrieval.Dockerfile` builds the KG retrieval service with the `runtime`
  extra, but does not bake KG artifacts or encoder weights into the image.
- `docker/kserve-mock.Dockerfile` builds the CPU KServe mock predictor.
- `docker/pipeline-init.Dockerfile` builds the DVC-capable init image used by the
  retrieval Helm chart to run `dvc pull`.

Install local Helm and kubectl tools:

```bash
make install-deploy-tools
```

Lint and render all charts:

```bash
make helm-lint
make helm-template
```

Run client-side dry-run validation for built-in Kubernetes resources:

```bash
make helm-dry-run
```

The KServe chart is structurally tested locally because `kubectl --dry-run=client`
cannot validate `serving.kserve.io` resources without the KServe CRD installed in a
cluster.

Build images on an x86 CI runner, or locally through `buildx` when needed:

```bash
make docker-build
```

The development host may be `aarch64`, while the target GKE nodes are `linux/amd64`.
The Docker build target and GitHub Actions workflow therefore pin
`--platform linux/amd64`.

The Helm charts cover API, retrieval, NGINX API-key gateway, and KServe mock
`InferenceService`. Live GKE deployment, GCS DVC remote credentials, real RunPod
configuration, and observability stacks are deferred to Plan 4.
````

- [ ] **Step 4: Run test to verify pass**

Run:

```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/test_readme_deploy_docs.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/vcsai/minhlbq/mlops-platform
git add README.md tests/test_readme_deploy_docs.py
git commit -m "docs: document deploy artifacts"
```

---

## Task 11: Full Verification

**Files:**
- No new files expected.
- DVC/MLflow artifacts remain unchanged.

- [ ] **Step 1: Install deploy tools**

Run:

```bash
cd /home/vcsai/minhlbq/mlops-platform && make install-deploy-tools
```

Expected: `.tools/bin/helm` and `.tools/bin/kubectl` exist and print client versions.

- [ ] **Step 2: Run the full test suite**

Run:

```bash
cd /home/vcsai/minhlbq/mlops-platform && make test
```

Expected: all tests pass and coverage remains above 80%.

- [ ] **Step 3: Run Helm lint**

Run:

```bash
cd /home/vcsai/minhlbq/mlops-platform && make helm-lint
```

Expected: all four charts lint cleanly.

- [ ] **Step 4: Run Helm render checks**

Run:

```bash
cd /home/vcsai/minhlbq/mlops-platform && make helm-template
```

Expected: all four charts render successfully.

- [ ] **Step 5: Run Kubernetes client dry-run for built-in resources**

Run:

```bash
cd /home/vcsai/minhlbq/mlops-platform && make helm-dry-run
```

Expected: API, retrieval, and NGINX rendered manifests pass `kubectl apply --dry-run=client --validate=false`. KServe is intentionally excluded because the CRD is absent locally.

- [ ] **Step 6: Confirm DVC pipeline still works**

Run:

```bash
cd /home/vcsai/minhlbq/mlops-platform && make dvc-status
```

Expected: `Data and pipelines are up to date.`

- [ ] **Step 7: Inspect git status**

Run:

```bash
cd /home/vcsai/minhlbq/mlops-platform && git status --short
```

Expected: clean worktree after the task commits.

- [ ] **Step 8: Commit verification-only fixes if any were required**

If any verification step required small fixes, commit only those files:

```bash
cd /home/vcsai/minhlbq/mlops-platform
git add Makefile README.md .gitignore scripts src docker deploy .github tests
git commit -m "test: verify deploy artifact toolchain"
```

Expected: commit exists only if verification revealed necessary changes after Task 10.

---

## Self-Review Notes

- **Spec coverage:** Tasks cover the KServe mock app, Dockerfiles, API/retrieval/NGINX/KServe Helm charts, CI/deploy workflows, Make targets, README docs, and final local verification gates.
- **Known implementation adjustment:** The spec requires retrieval initContainer `dvc pull` while keeping retrieval runtime free of DVC. This plan adds `docker/pipeline-init.Dockerfile` as a support image and includes it in CI and Make targets.
- **Repository boundary:** No task imports from `baseline/`; retrieval runtime uses the package-owned `medical_qa_platform.retrieval` modules.
- **Local validation boundary:** Built-in Kubernetes resources are checked with `kubectl --dry-run=client --validate=false`; KServe CRD validation is structural-only until Plan 4.
- **Type consistency:** The KServe mock app response matches `KServeBackend.generate()`, which reads `resp.json()["predictions"][0]["text"]`.
