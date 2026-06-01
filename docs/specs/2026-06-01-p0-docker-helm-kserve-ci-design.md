# Plan 3 Design — Docker / Helm / KServe / CI

**Status:** Approved (brainstorming) — 2026-06-01
**Spec reference:** repo-root `MLOPS_IMPLEMENTATION_PLAN.md` sections 5.1–5.4, 5.6 (decisions #1–#5, #8, #9), 6, 7, 11 (P0 tier), Day 3.
**Predecessors:** Plan 1 (runtime package, ✓), Plan 2 (DVC + MLflow smoke pipeline, ✓).

## Goal

Package the runtime inference package (plan 1) and smoke pipeline (plan 2) into
deployable artifacts: Docker images, Helm charts for app-serving, a KServe mock
`InferenceService` with scale-to-zero, an NGINX gateway with API-key auth, and a
GitHub Actions CI workflow (test → build). Everything is verifiable locally via
`helm lint` / `helm template` / `kubectl --dry-run=client` plus Python structural
tests. No GKE provisioning, no live RunPod, no observability/MLflow deploy.

## Scope Boundary (fixed by Plan 1's scope note)

- **Plan 3 (this):** Docker / Helm / KServe / CI authoring + local verification.
- **Plan 4 (next):** GKE provisioning, live RunPod endpoint, monitoring/MLflow/
  Evidently deploy, and *executing* the manual-deploy workflow against a real
  cluster.

This plan does **not** provision cloud infrastructure, run a real cluster, push
to a live registry from a developer machine, or execute `helm upgrade` against
GKE. The Helm `retrieval` chart templates the GCS-backed PVC + `dvc pull`
initContainer, but the GCS remote/secret wiring and execution belong to plan 4.

## Environment Constraints (verified 2026-06-01)

- `docker` client + daemon present, but host is **aarch64**; GKE is x86. Per
  master-plan decision #3, images build in CI (x86) or via
  `buildx --platform linux/amd64`. **Plan 3 does not run `docker build`
  locally** — Dockerfiles are verified structurally; the real build runs in CI.
- `helm`, `kubectl`, `kind` are **not installed**. Plan 3 installs `helm` +
  `kubectl` into a project tool dir (`install-deploy-tools` Make target) and uses
  `helm lint` + `helm template` + `kubectl apply --dry-run=client`. No `kind`
  cluster (live cluster apply is plan 4).
- `PyYAML` is available in `.venv` for structural manifest tests.

## Components

### 1. Docker (`docker/`)

- `api.Dockerfile` — lean API image: `pip install .` (no ML extras — only
  fastapi/httpx/prometheus-client/pydantic), non-root `USER`, `EXPOSE 8000`,
  `CMD uvicorn medical_qa_platform.api.app:create_app --factory --host 0.0.0.0
  --port 8000`. The explicit `--host 0.0.0.0` is required — uvicorn's default
  `127.0.0.1` is unreachable by K8s Services/probes.
- `retrieval.Dockerfile` — `pip install .[runtime]` (torch/faiss/
  sentence-transformers). **Does not bake** the encoder or KG artifacts (pulled
  via PVC at runtime, decision #2); sets `HF_HOME`/`SENTENCE_TRANSFORMERS_HOME`
  to the PVC cache path and `RETRIEVAL_DEVICE=cpu`. Non-root user. `EXPOSE 8001`;
  CMD binds `--host 0.0.0.0 --port 8001`.
- `kserve-mock.Dockerfile` — packages the mock predictor module (below) and runs
  it bound to `--host 0.0.0.0 --port 8080`. Lean, no ML deps.

### 1b. KServe mock predictor module (`src/medical_qa_platform/serving/`)

The KServe `InferenceService` needs a real HTTP app, not just a Dockerfile.
`KServeBackend` (verified in `inference/kserve_backend.py`) POSTs
`{"instances": [{"messages": [...]}]}` and reads `resp.json()["predictions"][0]
["text"]`. Add `serving/kserve_mock_app.py` — a tiny FastAPI app exposing a
predict endpoint that reuses `MockBackend` and returns exactly
`{"predictions": [{"text": <answer>}]}`, plus `/health`/`/ready`. Unit-tested via
`TestClient` against that contract.

### 2. Helm charts (`deploy/helm/`) — app-serving only

Monitoring / MLflow charts are plan 4.

- `api/` — Deployment + Service + ConfigMap (`MODEL_BACKEND`, `MODEL_VERSION`,
  **`RETRIEVAL_URL=http://<retrieval-service>:8001`**, `RUNPOD_BASE_URL`,
  `RUNPOD_MODEL`, `KSERVE_URL`) + **HPA** (`autoscaling/v2`, CPU-utilization).
  `RETRIEVAL_URL` is mandatory: `config.py` defaults it to `http://localhost:8001`,
  so without it API pods would call themselves instead of the retrieval Service.
  Probes hit `/health` (liveness) and `/ready` (readiness).
- `retrieval/` — Deployment with an **initContainer that runs `dvc pull`** + PVC
  mount + `HF_HOME`/`SENTENCE_TRANSFORMERS_HOME`/`RETRIEVAL_DEVICE=cpu` env +
  Service + HPA. The initContainer uses a **DVC-capable image** (configurable
  `initImage` in values, defaulting to an image built from `.[pipeline]`) — the
  runtime `.[runtime]` extra does **not** include DVC. It mounts the DVC metadata
  (`.dvc/config`, `dvc.yaml`, `dvc.lock`, and the artifact `*.dvc` files) so
  `dvc pull` resolves. `/ready` returns 200 only after artifacts load. (GCS
  remote + secret wiring = plan 4; plan 3 templates the mechanism with
  placeholders.)
- `nginx/` — Deployment + ConfigMap + **Secret** + Service. Plain nginx reverse
  proxy (**not** an ingress-nginx controller — that needs a cluster → plan 4).
  The ConfigMap holds only the `nginx.conf` **template/logic** (API-key check via
  `map`, `proxy_pass` to the api Service); the actual key value lives in a
  Kubernetes **Secret** (decision #5) injected via env + an `envsubst` entrypoint.
  No literal API key in the ConfigMap.
- `kserve/` — `InferenceService` with **`minReplicas: 0`** (scale-to-zero),
  pointing at the kserve-mock image.

### 3. CI (`.github/workflows/`)

- `ci.yml` — on PR/push: setup Python 3.12 → `pip install .[dev,pipeline]` →
  **`install-deploy-tools` (helm + kubectl) → `helm lint` + `helm template` every
  chart** → `pytest --cov-fail-under=80` → `docker build` of all three images on
  an x86 runner → push to registry on `main` only. Installing the deploy tools in
  CI is required so the chart tests actually run instead of skipping — otherwise a
  broken chart passes the PR gate.
- `deploy.yml` — `workflow_dispatch` **skeleton**: auth → `helm upgrade` → smoke
  tests (`/health`, `/search`, `/predict` mock). Structured but marked
  verify-in-plan-4; not executed here.

### 4. Makefile additions

- `install-deploy-tools` — download pinned `helm` + `kubectl` into a tool dir.
- `helm-lint` — `helm lint` every chart.
- `helm-template` — `helm template` every chart (render check).
- `docker-build` — `buildx --platform linux/amd64` build of the three images
  (intended for CI/x86; documented as such).

## Verification (TDD, per plan 2 pattern)

Each test is written failing first, then satisfied by the authored files.

- `tests/serving/test_kserve_mock_app.py` — `TestClient` against the mock
  predictor: POST `{"instances":[{"messages":[...]}]}` returns
  `{"predictions":[{"text": ...}]}`; `/health` + `/ready` OK. (No helm/kubectl
  needed — pure Python.)
- `tests/deploy/test_dockerfiles.py` — base image pins, non-root `USER`, `COPY`
  of `src`, **retrieval Dockerfile does not COPY encoder/KG weights**, all three
  CMDs bind `--host 0.0.0.0`, correct `EXPOSE` ports.
- `tests/deploy/test_helm_charts.py` — run `helm lint` + `helm template`, parse
  rendered YAML: expected `kind`s, **API ConfigMap has `RETRIEVAL_URL` pointing
  at the retrieval Service** (not localhost), API HPA
  `targetCPUUtilizationPercentage`, retrieval initContainer runs `dvc pull` with a
  DVC-capable `initImage` + PVC mount + DVC-metadata mounts + `HF_HOME` env,
  **nginx ConfigMap contains the API-key check logic but no literal key, and a
  Secret carries the key**.
- `tests/deploy/test_kserve.py` — render kserve chart and assert via parsed YAML:
  `InferenceService` with `minReplicas: 0`. KServe is **not** dry-run-validated
  locally (CRD absent) — structural assertions only; live validation = plan 4.
- `tests/deploy/test_ci_workflows.py` — parse workflow YAML: `ci.yml` installs
  deploy tools, runs `helm lint`/`helm template`, has `--cov-fail-under=80`, and
  builds three images; `deploy.yml` has `workflow_dispatch`.
- Final gate: `helm lint` clean for all four charts; `helm template <chart> |
  kubectl apply --dry-run=client --validate=false` passes for the three
  built-in-kind charts (api/retrieval/nginx). The KServe chart is verified by the
  structural test only, because client dry-run cannot resolve the
  `serving.kserve.io` CRD without a cluster.

Tests that require `helm`/`kubectl` skip cleanly (pytest skip) when the tool dir
is absent, so the pure-Python suite (including the mock-predictor test) still runs
without `install-deploy-tools`. CI installs the tools so the chart tests run for
real rather than skipping.

## File Structure (new under `mlops-platform/`)

```
src/medical_qa_platform/serving/{__init__.py,kserve_mock_app.py}
docker/{api,retrieval,kserve-mock}.Dockerfile
deploy/helm/{api,retrieval,nginx,kserve}/{Chart.yaml,values.yaml,templates/*.yaml}
.github/workflows/{ci.yml,deploy.yml}
tests/serving/test_kserve_mock_app.py
tests/deploy/test_{dockerfiles,helm_charts,kserve,ci_workflows}.py
Makefile          (add install-deploy-tools, helm-lint, helm-template, docker-build)
README.md         (add Docker/Helm/KServe/CI section + aarch64→x86 note)
```

## Grading-criteria coverage (P0 lines advanced)

- API + retrieval Docker images build (in CI, x86).
- CI: test + build images on PR, coverage gate > 80%.
- Helm deploys API + retrieval (charts authored + render-verified).
- NGINX gateway + API-key auth.
- KServe `InferenceService` (mock) + scale-to-zero (manifest + render-verified;
  live demo = plan 4).

## Out of Scope (→ plan 4)

GKE cluster provisioning; live RunPod 3B endpoint; Prometheus/Grafana/Loki/Tempo
and MLflow server Helm charts; Evidently drift dashboard; executing the deploy
workflow; GCS DVC remote wiring; HPA load demonstration (`hey`/`locust`).
