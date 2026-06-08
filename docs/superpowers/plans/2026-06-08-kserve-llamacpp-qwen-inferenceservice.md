# KServe llama.cpp Qwen2.5-1.5B InferenceService Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the vestigial KServe *mock* predictor with a real Qwen2.5-1.5B-Instruct InferenceService served by upstream llama.cpp, deployable on a free-tier GKE Standard zonal cluster with KServe RawDeployment.

**Architecture:** The `medical-qa-kserve` Helm chart stops shipping our hand-built `medical-qa-kserve-mock` FastAPI image and instead runs the **upstream `ghcr.io/ggml-org/llama.cpp:server`** image (no custom build package — the misnamed one is deleted, not renamed). The container downloads `Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M` (~1.1 GB) at startup via `-hf` and exposes the OpenAI-compatible `/v1` API. Because llama.cpp speaks OpenAI, the API talks to it through the **existing `vllm` backend** (just point `LLM_BASE_URL` at the in-cluster predictor) — so the bespoke KServe v1-`:predict` Python backend (`kserve_backend.py`) and the FastAPI mock (`kserve_mock_app.py`) become dead code and are removed. The model answers naturally (no constrained decoding); the existing API answer-parser extracts the letter from the model's `<answer>…</answer>` output.

**Tech Stack:** Helm, KServe `serving.kserve.io/v1beta1` InferenceService (RawDeployment mode), llama.cpp `llama-server`, GGUF (Q4_K_M), Python 3.12 (httpx/FastAPI for the parts being deleted), pytest.

**Brainstorm-approved config (do not re-litigate):**
- Model: **Qwen2.5-1.5B-Instruct, Q4_K_M GGUF** (~1.1 GB, working set ~2 GB).
- Image: **`ghcr.io/ggml-org/llama.cpp:server`** (upstream, no custom build).
- Deploy: **KServe RawDeployment** on **GKE Standard zonal**, `minReplicas: 1` (no scale-to-zero without KEDA).
- Pod resources: **requests cpu `2` / mem `2Gi`; limits cpu `4` / mem `3Gi`** (fits the 8-vCPU free-trial cap alongside the existing ~750m of services).
- llama-server flags: `--ctx-size 4096 --threads 2 --cont-batching --jinja` (model answers naturally; no constrained decoding).
- API reaches it via the existing `vllm`/OpenAI backend (`LLM_BASE_URL` → in-cluster predictor). No new Python backend code.

**Execution prerequisites:**
- This is the **`mlops-platform`** repo (`/home/vcsai/minhlbq/mlops-platform`), a separate git repo from `baseline/`.
- Work on a feature branch (e.g. `feat/kserve-llamacpp-qwen`). `main` is local-only; do **not** push unless asked.
- Project-local helm is required for the chart tests: run `make install-deploy-tools` once if `.tools/bin/helm` is missing (the helm tests `pytest.skip` without it — install it so they actually run).
- Test runner: `.venv/bin/pytest`. Full gate: `.venv/bin/pytest --cov=medical_qa_platform --cov-report=term-missing --cov-fail-under=80`.

---

## File Structure

**Deleted (dead mock path + misnamed build package):**
- `src/medical_qa_platform/inference/kserve_backend.py` — KServe v1 `:predict` client, superseded by the OpenAI `vllm` backend.
- `src/medical_qa_platform/serving/kserve_mock_app.py` — FastAPI mock predictor.
- `src/medical_qa_platform/serving/__init__.py` — package becomes empty.
- `docker/kserve-mock.Dockerfile` — the `medical-qa-kserve-mock` build package.
- `tests/inference/test_kserve_backend.py`, `tests/serving/test_kserve_mock_app.py` — tests for the deleted code.

**Modified:**
- `src/medical_qa_platform/inference/__init__.py` — drop the `kserve` branch from `get_backend`.
- `deploy/helm/kserve/values.yaml` — llama.cpp image + model + resources + RawDeployment.
- `deploy/helm/kserve/templates/inferenceservice.yaml` — real container args, probes, and model-cache volume.
- `deploy/helm/kserve/Chart.yaml` — description.
- `deploy/helm/api/templates/configmap.yaml`, `deploy/helm/api/values.yaml` — remove `KSERVE_URL`/`kserveUrl`.
- `.github/workflows/ci.yml` — drop the `kserve-mock` build-matrix entry.
- `Makefile` — drop the `kserve-mock` docker-build line.
- `scripts/cloud/deploy.sh` — comment + drop the now-irrelevant `--set image.tag` for the kserve chart.
- `README.md` — describe the real llama.cpp InferenceService instead of the mock.
- Tests pinning the above: `tests/inference/test_mock_backend.py`, `tests/deploy/test_helm_api_chart.py`, `tests/deploy/test_kserve_chart.py`, `tests/deploy/test_ci_workflows.py`, `tests/deploy/test_dockerfiles.py`, `tests/deploy/test_makefile_deploy_tools.py`, `tests/test_readme_deploy_docs.py`.

---

## Task 1: Remove the dead KServe Python backend + mock predictor

**Files:**
- Modify: `src/medical_qa_platform/inference/__init__.py:19-22`
- Modify: `tests/inference/test_mock_backend.py` (add a regression test)
- Delete: `src/medical_qa_platform/inference/kserve_backend.py`
- Delete: `src/medical_qa_platform/serving/kserve_mock_app.py`
- Delete: `src/medical_qa_platform/serving/__init__.py`
- Delete: `tests/inference/test_kserve_backend.py`
- Delete: `tests/serving/test_kserve_mock_app.py`

- [ ] **Step 1: Write the failing test** — mirror the existing `test_factory_no_longer_knows_runpod` for `kserve`. Append to `tests/inference/test_mock_backend.py`:

```python
def test_factory_no_longer_knows_kserve():
    with pytest.raises(ValueError):
        get_backend("kserve")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/pytest tests/inference/test_mock_backend.py::test_factory_no_longer_knows_kserve -v`
Expected: FAIL — `DID NOT RAISE ValueError` (the `kserve` branch still constructs a backend; `KSERVE_URL` defaults to `""` so it returns a `KServeBackend` instead of raising).

- [ ] **Step 3: Remove the `kserve` branch from the factory.** In `src/medical_qa_platform/inference/__init__.py`, delete these lines:

```python
    if name == "kserve":
        from .kserve_backend import KServeBackend

        return KServeBackend.from_env()
```

The file should now end:

```python
    if name == "vllm":
        from .vllm_backend import VllmBackend

        return VllmBackend.from_env()
    raise ValueError(f"unknown MODEL_BACKEND: {name!r}")
```

- [ ] **Step 4: Delete the dead modules and their tests**

```bash
git rm src/medical_qa_platform/inference/kserve_backend.py \
       src/medical_qa_platform/serving/kserve_mock_app.py \
       src/medical_qa_platform/serving/__init__.py \
       tests/inference/test_kserve_backend.py \
       tests/serving/test_kserve_mock_app.py
rmdir src/medical_qa_platform/serving tests/serving 2>/dev/null || true
```

- [ ] **Step 5: Verify nothing else imports the deleted modules**

Run: `grep -rnI "kserve_backend\|kserve_mock_app\|KServeBackend\|from .*serving" src/ tests/ app/`
Expected: no matches (empty output).

- [ ] **Step 6: Run the factory + inference tests**

Run: `.venv/bin/pytest tests/inference/ -v`
Expected: PASS (including the new `test_factory_no_longer_knows_kserve`); no import errors from the removed `kserve_backend`.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor: drop dead KServe v1-predict backend and FastAPI mock predictor

The KServe path now serves a real llama.cpp (OpenAI) endpoint reached via the
vllm backend, so the bespoke KServeBackend (instances/:predict protocol) and the
kserve_mock_app predictor are dead code."
```

---

## Task 2: Drop `KSERVE_URL` / `kserveUrl` from the api chart

**Files:**
- Modify: `deploy/helm/api/templates/configmap.yaml:11`
- Modify: `deploy/helm/api/values.yaml:17`
- Modify: `tests/deploy/test_helm_api_chart.py` (assert the key is gone)

- [ ] **Step 1: Write the failing assertion.** In `tests/deploy/test_helm_api_chart.py`, inside `test_api_chart_renders_deployment_service_configmap_and_hpa`, next to the existing `assert "RUNPOD_BASE_URL" not in config["data"]` line, add:

```python
    assert "KSERVE_URL" not in config["data"]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/pytest tests/deploy/test_helm_api_chart.py::test_api_chart_renders_deployment_service_configmap_and_hpa -v`
Expected: FAIL — `KSERVE_URL` is still rendered into the ConfigMap. (If it `SKIP`s, run `make install-deploy-tools` first so project-local helm exists.)

- [ ] **Step 3: Remove the key from the ConfigMap template.** In `deploy/helm/api/templates/configmap.yaml`, delete the line:

```yaml
  KSERVE_URL: {{ .Values.env.kserveUrl | quote }}
```

- [ ] **Step 4: Remove the value default.** In `deploy/helm/api/values.yaml`, delete the line:

```yaml
  kserveUrl: http://medical-qa-kserve/v1/models/medical-qa-smoke:predict
```

- [ ] **Step 5: Run the api chart test + helm template**

Run: `.venv/bin/pytest tests/deploy/test_helm_api_chart.py -v && .tools/bin/helm template medical-qa-api deploy/helm/api >/dev/null && echo TEMPLATE_OK`
Expected: PASS and `TEMPLATE_OK`.

- [ ] **Step 6: Commit**

```bash
git add deploy/helm/api/templates/configmap.yaml deploy/helm/api/values.yaml tests/deploy/test_helm_api_chart.py
git commit -m "chore(api): drop KSERVE_URL env; api reaches llama.cpp via vllm backend"
```

---

## Task 3: Rewrite the kserve chart to a real llama.cpp InferenceService

**Files:**
- Modify: `deploy/helm/kserve/values.yaml` (full rewrite)
- Modify: `deploy/helm/kserve/templates/inferenceservice.yaml` (full rewrite)
- Modify: `deploy/helm/kserve/Chart.yaml:2-3` (description)
- Modify: `tests/deploy/test_kserve_chart.py` (full rewrite)

- [ ] **Step 1: Rewrite the chart test to pin the real-model shape.** Replace the entire contents of `tests/deploy/test_kserve_chart.py` with:

```python
from pathlib import Path

import yaml

from tests.deploy.helm_helpers import find_kind, render_chart


ROOT = Path(__file__).parents[2]


def test_kserve_values_serve_real_qwen_llamacpp_not_a_mock():
    values = yaml.safe_load((ROOT / "deploy/helm/kserve/values.yaml").read_text())
    # RawDeployment has no scale-to-zero without KEDA, so we hold one replica.
    assert values["minReplicas"] == 1
    assert values["deploymentMode"] == "RawDeployment"
    # upstream image, no hand-built mock package
    assert values["image"]["repository"] == "ghcr.io/ggml-org/llama.cpp"
    assert values["image"]["tag"] == "server"
    assert "kserve-mock" not in values["image"]["repository"]
    assert values["model"]["hfRepo"] == "Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M"
    # fits the 8-vCPU free-trial cap: request 2 / burst to 4
    assert values["resources"]["requests"]["cpu"] == "2"
    assert values["resources"]["requests"]["memory"] == "2Gi"
    assert values["resources"]["limits"]["cpu"] == "4"
    assert values["resources"]["limits"]["memory"] == "3Gi"


def test_kserve_chart_renders_raw_llamacpp_inferenceservice():
    resources = render_chart("kserve")
    isvc = find_kind(resources, "InferenceService", "medical-qa-kserve")
    assert isvc["apiVersion"] == "serving.kserve.io/v1beta1"
    assert (
        isvc["metadata"]["annotations"]["serving.kserve.io/deploymentMode"]
        == "RawDeployment"
    )
    predictor = isvc["spec"]["predictor"]
    assert predictor["minReplicas"] == 1
    container = predictor["containers"][0]
    assert container["image"] == "ghcr.io/ggml-org/llama.cpp:server"
    assert container["ports"][0]["containerPort"] == 8080
    # OpenAI-compatible health gate, not the old /ready mock probe
    assert container["readinessProbe"]["httpGet"]["path"] == "/health"
    assert container["startupProbe"]["httpGet"]["path"] == "/health"
    # serves the Qwen2.5-1.5B GGUF, downloaded at startup
    assert "-hf" in container["args"]
    assert "Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M" in container["args"]
    assert "--ctx-size" in container["args"]
    # model cache lives on a writable volume (container image user-agnostic)
    mounts = {m["name"]: m["mountPath"] for m in container["volumeMounts"]}
    assert mounts["model-cache"] == "/models"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/pytest tests/deploy/test_kserve_chart.py -v`
Expected: FAIL — current values still point at `medical-qa-kserve-mock` with `minReplicas: 0` and no `model`/`deploymentMode` keys.

- [ ] **Step 3: Rewrite `deploy/helm/kserve/values.yaml`.** Replace the entire file with:

```yaml
# Real Qwen2.5-1.5B-Instruct InferenceService served by the upstream
# llama.cpp:server image (OpenAI-compatible /v1). No custom build package: the
# image is pulled from ghcr.io/ggml-org and the GGUF is downloaded at startup via
# -hf. The API reaches this endpoint through the vllm backend (LLM_BASE_URL).

# RawDeployment = plain Deployment/Service, no Knative/Istio. Required because the
# free-tier GKE Standard cluster runs KServe without the Serverless stack.
deploymentMode: RawDeployment

image:
  repository: ghcr.io/ggml-org/llama.cpp
  tag: server
  pullPolicy: IfNotPresent

# Must be >= 1: RawDeployment has no scale-to-zero without KEDA.
minReplicas: 1

model:
  # public GGUF repo; the :Q4_K_M tag selects the ~1.1 GB quant.
  hfRepo: "Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M"
  # served model name advertised on /v1/models (the OpenAI `model` field).
  servedName: "qwen2.5-1.5b-instruct"
  contextSize: 4096
  # e2 vCPUs are hyperthreads; pin llama.cpp threads to physical cores (vCPU/2).
  threads: 2

# request 2 / burst to 4 vCPU keeps total cluster requests under the 8-vCPU
# free-trial cap; ~2 GB working set fits 2Gi request / 3Gi limit.
resources:
  requests:
    cpu: "2"
    memory: 2Gi
  limits:
    cpu: "4"
    memory: 3Gi
```

- [ ] **Step 4: Rewrite `deploy/helm/kserve/templates/inferenceservice.yaml`.** Replace the entire file with:

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: medical-qa-kserve
  annotations:
    serving.kserve.io/deploymentMode: {{ .Values.deploymentMode | quote }}
spec:
  predictor:
    minReplicas: {{ .Values.minReplicas }}
    containers:
      - name: kserve-container
        image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
        imagePullPolicy: {{ .Values.image.pullPolicy }}
        # The image's entrypoint is llama-server; these are appended as flags.
        args:
          - "-hf"
          - {{ .Values.model.hfRepo | quote }}
          - "--alias"
          - {{ .Values.model.servedName | quote }}
          - "--host"
          - "0.0.0.0"
          - "--port"
          - "8080"
          - "--ctx-size"
          - {{ .Values.model.contextSize | quote }}
          - "--threads"
          - {{ .Values.model.threads | quote }}
          - "--cont-batching"
          - "--jinja"
        env:
          # -hf downloads into $LLAMA_CACHE; keep it on a writable volume so the
          # image's default user (whatever it is) can write the GGUF.
          - name: LLAMA_CACHE
            value: /models
        ports:
          - containerPort: 8080
        startupProbe:
          httpGet:
            path: /health
            port: 8080
          # GGUF download (~1.1 GB) + load can take minutes on a cold node:
          # 10 + 60*10 = up to ~10 min before the pod is failed.
          initialDelaySeconds: 10
          periodSeconds: 10
          failureThreshold: 60
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          periodSeconds: 10
        volumeMounts:
          - name: model-cache
            mountPath: /models
        resources:
{{ toYaml .Values.resources | indent 10 }}
    volumes:
      - name: model-cache
        emptyDir: {}
```

- [ ] **Step 5: Update `deploy/helm/kserve/Chart.yaml` description.** Change line 3 from:

```yaml
description: KServe CPU smoke/mock InferenceService for Medical QA
```

to:

```yaml
description: KServe CPU InferenceService serving Qwen2.5-1.5B-Instruct via llama.cpp
```

- [ ] **Step 6: Run the chart test + helm lint/template**

Run: `.venv/bin/pytest tests/deploy/test_kserve_chart.py -v && .tools/bin/helm lint deploy/helm/kserve && .tools/bin/helm template medical-qa-kserve deploy/helm/kserve >/dev/null && echo CHART_OK`
Expected: PASS, lint clean, and `CHART_OK`.

- [ ] **Step 7: Commit**

```bash
git add deploy/helm/kserve tests/deploy/test_kserve_chart.py
git commit -m "feat(kserve): serve real Qwen2.5-1.5B via upstream llama.cpp (RawDeployment)

Replaces the mock predictor image with ghcr.io/ggml-org/llama.cpp:server pulling
Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M at startup. minReplicas=1, request 2/limit
4 vCPU, 2Gi/3Gi, /health probes, model cache on an emptyDir."
```

---

## Task 4: Delete the `medical-qa-kserve-mock` build package

**Files:**
- Delete: `docker/kserve-mock.Dockerfile`
- Modify: `.github/workflows/ci.yml:36,52` (comment + matrix entry)
- Modify: `Makefile:69` (docker-build line)
- Modify: `tests/deploy/test_ci_workflows.py:38-44`
- Modify: `tests/deploy/test_dockerfiles.py:62-69`
- Modify: `tests/deploy/test_makefile_deploy_tools.py:42`

- [ ] **Step 1: Update the failing tests first.** Make three edits.

(a) In `tests/deploy/test_ci_workflows.py`, remove the `kserve-mock` line from the expected dockerfile set so it reads:

```python
    assert dockerfiles == {
        "docker/api.Dockerfile",
        "docker/retrieval.Dockerfile",
        "docker/pipeline-init.Dockerfile",
        "docker/ui.Dockerfile",
    }
```

(b) In `tests/deploy/test_dockerfiles.py`, delete the whole function `test_kserve_mock_dockerfile_runs_mock_predictor_on_8080` (lines 62-69):

```python
def test_kserve_mock_dockerfile_runs_mock_predictor_on_8080():
    text = _read("kserve-mock.Dockerfile")
    assert "uv pip install --system --no-cache ." in text
    assert "USER app" in text
    assert "EXPOSE 8080" in text
    assert "medical_qa_platform.serving.kserve_mock_app:create_app" in text
    assert '"--host", "0.0.0.0"' in text
    assert '"--port", "8080"' in text
```

(c) In `tests/deploy/test_makefile_deploy_tools.py`, inside `test_makefile_exposes_deploy_targets_and_preserves_pipeline_targets`, delete the line:

```python
    assert "docker/kserve-mock.Dockerfile" in text
```

- [ ] **Step 2: Run the updated tests to verify they now fail (against the still-present mock build)**

Run: `.venv/bin/pytest tests/deploy/test_ci_workflows.py tests/deploy/test_makefile_deploy_tools.py::test_makefile_exposes_deploy_targets_and_preserves_pipeline_targets -v`
Expected: FAIL — the CI matrix still includes `docker/kserve-mock.Dockerfile` and the Makefile still lists it, so the set/string assertions mismatch.

- [ ] **Step 3: Remove the matrix entry from `.github/workflows/ci.yml`.** Delete the line:

```yaml
          - { image: kserve-mock, dockerfile: docker/kserve-mock.Dockerfile }
```

The build-job comment on line 36 already says "the four images" — now accurate; leave it.

- [ ] **Step 4: Remove the docker-build line from `Makefile`.** Delete the line:

```makefile
	docker buildx build --platform linux/amd64 -f docker/kserve-mock.Dockerfile -t medical-qa-kserve-mock:local --load .
```

- [ ] **Step 5: Delete the Dockerfile**

```bash
git rm docker/kserve-mock.Dockerfile
```

- [ ] **Step 6: Verify no stale references remain**

Run: `grep -rnI "kserve-mock\|kserve_mock" . --exclude-dir=.git --exclude-dir=docs`
Expected: no matches (the plan doc under `docs/` is excluded).

- [ ] **Step 7: Run the affected tests**

Run: `.venv/bin/pytest tests/deploy/test_ci_workflows.py tests/deploy/test_dockerfiles.py tests/deploy/test_makefile_deploy_tools.py -v`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "chore: delete medical-qa-kserve-mock build package

The KServe InferenceService now uses upstream ghcr.io/ggml-org/llama.cpp:server,
so the hand-built mock predictor image and its CI/Makefile build entries are gone.
Build matrix drops from 5 to 4 images."
```

---

## Task 5: Update README + deploy.sh for the real InferenceService

**Files:**
- Modify: `README.md:67,90-92,104-105` (+ a short RawDeployment note)
- Modify: `scripts/cloud/deploy.sh:64-70`
- Modify: `tests/test_readme_deploy_docs.py:9-23`

- [ ] **Step 1: Update the README doc test first.** In `tests/test_readme_deploy_docs.py`, remove the `docker/kserve-mock.Dockerfile` entry from the expected list and add `llama.cpp` so the list reads:

```python
    for expected in [
        "make install-deploy-tools",
        "make helm-lint",
        "make helm-template",
        "make helm-dry-run",
        "make docker-build",
        "docker/api.Dockerfile",
        "docker/retrieval.Dockerfile",
        "docker/pipeline-init.Dockerfile",
        "KServe",
        "llama.cpp",
        "NGINX",
        "aarch64",
        "linux/amd64",
    ]:
        assert expected in text
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/pytest tests/test_readme_deploy_docs.py -v`
Expected: FAIL — README still references `docker/kserve-mock.Dockerfile` and does not yet contain `llama.cpp`.

- [ ] **Step 3: Update the Dockerfile bullet in `README.md`.** Replace the line:

```markdown
- `docker/kserve-mock.Dockerfile` builds the CPU KServe mock predictor.
```

with:

```markdown
- The KServe InferenceService uses the upstream `ghcr.io/ggml-org/llama.cpp:server`
  image directly (no custom build) and pulls `Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M`
  at startup, exposing the OpenAI-compatible `/v1` API.
```

- [ ] **Step 4: Update the KServe paragraph in `README.md`.** Replace the paragraph:

```markdown
The KServe chart is structurally tested locally because `kubectl --dry-run=client`
cannot validate `serving.kserve.io` resources without the KServe CRD installed in a
cluster.
```

with:

```markdown
The KServe chart is structurally tested locally because `kubectl --dry-run=client`
cannot validate `serving.kserve.io` resources without the KServe CRD installed in a
cluster. The `medical-qa-kserve` `InferenceService` runs in **RawDeployment** mode
(plain Deployment/Service, no Knative/Istio) so it deploys on a lean free-tier
**GKE Standard zonal** cluster with KServe installed; `minReplicas: 1` (RawDeployment
has no scale-to-zero without KEDA). It serves Qwen2.5-1.5B-Instruct on CPU via
llama.cpp; the model answers naturally and the API's answer-parser extracts the letter
from its `<answer>…</answer>` output. The API consumes it
through the `vllm` backend: set `MODEL_BACKEND=vllm` and point `LLM_BASE_URL` at the
in-cluster predictor Service (e.g. `http://medical-qa-kserve-predictor/v1`; confirm the
exact Service name with `kubectl get svc` after deploy).
```

- [ ] **Step 5: Update the chart-coverage sentence in `README.md`.** Replace:

```markdown
The Helm charts cover API, retrieval, NGINX API-key gateway, and KServe mock
`InferenceService`. Live GKE deployment, GCS DVC remote credentials, and
```

with:

```markdown
The Helm charts cover API, retrieval, NGINX API-key gateway, and a KServe
llama.cpp `InferenceService` (Qwen2.5-1.5B). Live GKE deployment, GCS DVC remote
credentials, and
```

- [ ] **Step 6: Update `scripts/cloud/deploy.sh`.** Replace the comment block + install array (lines 64-70) — from:

```bash
# KServe is optional. The mock-backend demo needs no InferenceService, and a vanilla
# Autopilot cluster ships without KServe CRDs — a hard install there aborts with
# "no matches for kind InferenceService" and fails the whole deploy. Install the chart
# only when the CRD is present; otherwise skip (non-fatal).
KSERVE_INSTALL=("$HELM" upgrade --install medical-qa-kserve deploy/helm/kserve \
  --namespace "$K8S_NAMESPACE" \
  --set image.tag="$IMAGE_TAG")
```

to:

```bash
# KServe is optional. The default demo (mock or vllm->DGX backend) needs no
# InferenceService, and a vanilla Autopilot cluster ships without KServe CRDs — a hard
# install there aborts with "no matches for kind InferenceService" and fails the whole
# deploy. When the CRD IS present (GKE Standard zonal with KServe installed), this
# deploys the real Qwen2.5-1.5B llama.cpp InferenceService; the image tag is pinned in
# the chart (upstream ghcr.io/ggml-org/llama.cpp:server), so IMAGE_TAG does not apply.
# To route the API at it, deploy with MODEL_BACKEND=vllm and LLM_BASE_URL pointing at
# the in-cluster predictor Service.
KSERVE_INSTALL=("$HELM" upgrade --install medical-qa-kserve deploy/helm/kserve \
  --namespace "$K8S_NAMESPACE")
```

- [ ] **Step 7: Verify deploy.sh still parses and dry-runs**

Run: `bash -n scripts/cloud/deploy.sh && PATH=/usr/bin:/bin GCP_PROJECT=demo bash scripts/cloud/deploy.sh --dry-run | grep "medical-qa-kserve" && echo DEPLOY_OK`
Expected: prints the `upgrade --install medical-qa-kserve deploy/helm/kserve` line and `DEPLOY_OK` (exit 0).

- [ ] **Step 8: Run the README + deploy.sh tests**

Run: `.venv/bin/pytest tests/test_readme_deploy_docs.py tests/cloud/test_deploy_sh.py -v`
Expected: PASS (all deploy.sh dry-run assertions still hold; README test passes).

- [ ] **Step 9: Commit**

```bash
git add README.md scripts/cloud/deploy.sh tests/test_readme_deploy_docs.py
git commit -m "docs: describe the real Qwen2.5-1.5B llama.cpp InferenceService

README + deploy.sh now reflect RawDeployment on GKE Standard, the upstream
llama.cpp image, and routing the API via the vllm backend. Drop the stale
IMAGE_TAG --set on the kserve install (upstream tag is pinned in the chart)."
```

---

## Task 6: Full-suite verification

**Files:** none (verification + final state check only)

- [ ] **Step 1: Run the full test suite with the coverage gate**

Run: `.venv/bin/pytest --cov=medical_qa_platform --cov-report=term-missing --cov-fail-under=80`
Expected: all tests PASS, coverage ≥ 80%. If coverage dipped below 80 because the deleted modules removed covered lines, investigate which module lost coverage (`term-missing`) — do **not** lower the threshold; if a genuinely under-tested module surfaced, note it for follow-up rather than papering over it.

- [ ] **Step 2: Lint and render every chart (CI parity)**

Run: `make helm-lint && make helm-template && echo HELM_OK`
Expected: lint clean for all five charts, all templates render, `HELM_OK`.

- [ ] **Step 3: Confirm no `kserve-mock` / dead-module references survive anywhere**

Run: `grep -rnI "kserve-mock\|kserve_mock\|KServeBackend\|kserveUrl\|KSERVE_URL" . --exclude-dir=.git --exclude-dir=docs --exclude-dir=.venv --exclude-dir=.tools`
Expected: no matches.

- [ ] **Step 4: Confirm the working tree is clean and review the branch diff**

Run: `git status && git log --oneline main..HEAD`
Expected: clean tree; five commits (Tasks 1–5) on the feature branch.

---

## Out of Scope / Follow-ups (do NOT implement now)

- **Auto-wiring deploy.sh to the in-cluster predictor URL.** The predictor Service name under KServe RawDeployment (`medical-qa-kserve-predictor` vs `-predictor-default`) varies by KServe version; left as a documented manual `LLM_BASE_URL` step rather than hard-coded, to avoid a brittle guess.
- **Persisting the GGUF across pod restarts** (PVC instead of emptyDir). For the demo the ~1.1 GB re-download on restart is acceptable; a PVC is a later optimization.
- **Actually provisioning the GKE Standard zonal cluster + installing KServe.** This plan ships the chart + wiring; cluster provisioning (and validating real `llama-bench` CPU throughput on the e2 node) is a separate operational task.
- **Baking the GGUF into a pinned `medical-qa-llamacpp` image** for reproducibility/air-gap. Only needed if startup `-hf` download proves unreliable.
- **GBNF constrained decoding** (`--grammar-file` forcing `<answer>[ABCD]</answer>`). Deliberately dropped so the model answers naturally; revisit only if format drift on the small base model starts breaking the answer-parser in practice.

---

## Self-Review

- **Spec coverage:** (1) replace mock with real Qwen2.5-1.5B llama.cpp InferenceService → Task 3; (2) "especially the build package name" → Task 4 deletes the misnamed `medical-qa-kserve-mock` package entirely (upstream image, no custom build); (3) RawDeployment/GKE Standard/minReplicas:1/resources → Tasks 3 + 5; (4) model answers naturally (no constrained decoding) → no grammar task by design; (5) reuse the existing vllm/OpenAI backend, no new backend code → enabled by Task 1 (remove dead `kserve` backend) + Task 2 (drop `KSERVE_URL`) + documented in Task 5. All approved-config items map to a task.
- **Placeholder scan:** every code/YAML step contains the full literal content; no TBD/"add error handling"/"similar to Task N". Commands have expected output.
- **Type/name consistency:** Values keys (`deploymentMode`, `image.repository/tag`, `minReplicas`, `model.hfRepo/servedName/contextSize/threads`, `resources`) match between values.yaml, the template, and the chart tests. The InferenceService container args end at `--jinja` (no grammar flag) and mount only `model-cache` at `/models` — consistent across the Task 3 template and its test. InferenceService name `medical-qa-kserve` unchanged (chart/test/deploy.sh agree).
