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
  `CMD uvicorn medical_qa_platform.api.app:create_app --factory`.
- `retrieval.Dockerfile` — `pip install .[runtime]` (torch/faiss/
  sentence-transformers). **Does not bake** the encoder or KG artifacts (pulled
  via PVC at runtime, decision #2); sets `HF_HOME`/`SENTENCE_TRANSFORMERS_HOME`
  to the PVC cache path and `RETRIEVAL_DEVICE=cpu`. Non-root user.
- `kserve-mock.Dockerfile` — small CPU mock predictor reusing `MockBackend`,
  serving a KServe-compatible predict endpoint. Lean, no ML deps.

### 2. Helm charts (`deploy/helm/`) — app-serving only

Monitoring / MLflow charts are plan 4.

- `api/` — Deployment + Service + ConfigMap (`MODEL_BACKEND`, `MODEL_VERSION`,
  `RUNPOD_BASE_URL`, `RUNPOD_MODEL`, `KSERVE_URL`) + **HPA** (`autoscaling/v2`,
  CPU-utilization). Probes hit `/health` (liveness) and `/ready` (readiness).
- `retrieval/` — Deployment with **initContainer `dvc pull`** + PVC mount +
  `HF_HOME`/`SENTENCE_TRANSFORMERS_HOME`/`RETRIEVAL_DEVICE=cpu` env + Service +
  HPA. `/ready` returns 200 only after artifacts load. (GCS remote/secret = plan 4.)
- `nginx/` — Deployment + ConfigMap (`nginx.conf`: API-key check via `map`/`if`,
  `proxy_pass` to the api Service) + Service. Plain nginx reverse proxy, **not**
  an ingress-nginx controller (controller needs a cluster → plan 4).
- `kserve/` — `InferenceService` with **`minReplicas: 0`** (scale-to-zero),
  pointing at the kserve-mock image.

### 3. CI (`.github/workflows/`)

- `ci.yml` — on PR/push: setup Python 3.12 → `pip install .[dev,pipeline]` →
  `pytest --cov-fail-under=80` → `docker build` of all three images on an x86
  runner → push to registry on `main` only.
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

- `tests/deploy/test_dockerfiles.py` — base image pins, non-root `USER`, `COPY`
  of `src`, **retrieval Dockerfile does not COPY encoder/KG weights**, correct
  `CMD`/`EXPOSE`.
- `tests/deploy/test_helm_charts.py` — run `helm lint` + `helm template`, parse
  rendered YAML: expected `kind`s, API HPA `targetCPUUtilizationPercentage`,
  retrieval initContainer runs `dvc pull` + PVC mount + `HF_HOME` env, nginx
  ConfigMap contains the API-key check.
- `tests/deploy/test_kserve.py` — render kserve chart; assert `InferenceService`
  with `minReplicas: 0`.
- `tests/deploy/test_ci_workflows.py` — parse workflow YAML: `ci.yml` has
  `--cov-fail-under=80` and builds three images; `deploy.yml` has
  `workflow_dispatch`.
- Final gate: `helm lint` clean and `helm template | kubectl apply
  --dry-run=client` passes for all four charts.

Tests that require `helm`/`kubectl` skip cleanly (pytest skip) when the tool dir
is absent, so the pure-Python suite still runs without `install-deploy-tools`;
the final gate run installs the tools and exercises the real lint/template path.

## File Structure (new under `mlops-platform/`)

```
docker/{api,retrieval,kserve-mock}.Dockerfile
deploy/helm/{api,retrieval,nginx,kserve}/{Chart.yaml,values.yaml,templates/*.yaml}
.github/workflows/{ci.yml,deploy.yml}
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
