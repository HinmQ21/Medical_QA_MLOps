# Medical QA — MLOps Platform

A production-shaped **MLOps serving + pipeline** platform for a Medical QA system.
It ships an agentic inference API, a knowledge-graph retrieval service, a
DVC + MLflow pipeline, Helm/Docker/KServe deployment, keyless CI/CD onto GKE, a
Streamlit demo UI, and drift + Prometheus observability.

> **Package:** `medical_qa_platform` (src layout) · Python ≥ 3.12 · managed with
> `uv` in `.venv`.

The runtime package (`src/medical_qa_platform/`) is **self-contained**: a guard
test (`tests/retrieval/test_kg_backend_self_contained.py`) keeps `src/` free of
out-of-tree imports. The full training pipeline orchestrates its training scripts
out-of-process (subprocess), so the serving and pipeline code stay decoupled.

## Architecture

```
                         ┌─────────────── GKE (demo) ───────────────┐
Browser ─ UI (Streamlit :8501) ─x-api-key─> nginx gateway ──────────┐
                                                                    │ (x-api-key server-side)
                                          ┌─────────────────────────▼────────────┐
                                          │  api (FastAPI) — /predict             │
                                          │  agentic tool-call loop               │
                                          └──┬────────────────────────────────┬───┘
                          tool: search_medical_knowledge                  ModelBackend
                                             │                                │
                               retrieval service (FAISS + MedEmbed-small)     │
                               ranker retrieve_v1 → contract.format_evidence  │
                                                                              │
                           MODEL_BACKEND=mock ── MockBackend                  │
                           MODEL_BACKEND=llm  ── LLMBackend (OpenAI /v1) ─────┤
                                                    ├── DGX vLLM (3B model) via Cloudflare Tunnel
                                                    └── KServe llama.cpp (Qwen2.5-1.5B) in-cluster

Offline:  params.yaml ─▶ mlops/pipelines ─▶ DVC stages ─▶ artifacts/ ─▶ MLflow registry
          (the full pipeline drives training scripts via subprocess — decoupled from serving)
```

## `/predict` — agentic tool-call loop

`/predict` is a **model-driven agentic loop**, not one-shot RAG. The model decides
when to call `search_medical_knowledge` (native OpenAI `tools=`); the loop runs the
tool, feeds results back, and repeats until the model answers or `MAX_TOOL_ITERATIONS`
(default 2) is hit. This mirrors the training rollout.

- **Request:** `{ "question": "<question with A) B) C) D) inline>" }`.
- **Response:** `answer` (best-effort letter parse, may be `null`), `raw_output`,
  `evidence[]`, `trace[]` (per-turn), `backend`, `model_version`, `contract_version`,
  `latency_ms`, `trace_id`.

## Model backends (`MODEL_BACKEND`)

| Backend | Use | Notes |
|---------|-----|-------|
| `mock` *(default)* | CI / offline plumbing | `MockBackend` |
| `llm` | real model | `LLMBackend`, generic OpenAI `/v1` client (`vllm` is a back-compat alias) |

The `llm` backend points at one of two targets:

| Target | When | Config |
|--------|------|--------|
| **DGX vLLM + Cloudflare Tunnel** | real 3B model | `LLM_BASE_URL=https://llm.<domain>/v1`, `LLM_MODEL=medical-qa-llama-gdpo`, `LLM_API_KEY=…` — see `docs/runbooks/dgx-vllm-cloudflare.md` |
| **KServe llama.cpp in-cluster** | GKE demo, no DGX | `LLM_BASE_URL=http://medical-qa-kserve-predictor.<ns>.svc.cluster.local/v1`, `LLM_MODEL=qwen2.5-1.5b-instruct`, `LLM_API_KEY` unset |

> `LLM_BASE_URL` **must end in `/v1`**, and deploys should set `MAX_TOKENS ≥ 2048`
> (the 512 default truncates `<think>…</think><answer>`, losing `<answer>`).

## Retrieval parity

The retrieval ranker `retrieve_v1` (8-term dual-retrieval fusion) lives in
`retrieval/ranker.py`. Encoder: **`abhinand/MedEmbed-small-v0.1` (384-d)**; contract
version: **`v1-medembed-small`** (`retrieval/contract.py`).

Changing the ranker or encoder requires regenerating the golden parity fixtures
(`scripts/gen_retrieval_golden.py`), bumping `RETRIEVAL_CONTRACT_VERSION`, and (if the
encoder changed) re-training. Parity is tested at L1 (CI, replay FAISS hits) and L2
(opt-in, full backend).

The production KG retrieval backend expects these local artifact files (populated by
DVC or mounted into the retrieval container) in `KG_DATA_DIR`:

```
index_hyperedge.bin  index_entity.bin  hedge_ids.npy  entity_names.npy  medical_hg.json
```

## Setup

```bash
cd /home/vcsai/minhlbq/mlops-platform   # always work from here

make install              # .venv + uv pip install -e ".[dev]"
make install-pipeline     # add [pipeline] extra (dvc[gs], mlflow)
make install-deploy-tools # fetch helm + kubectl into .tools/bin
```

Optional extras: `[pipeline]` (DVC/MLflow), `[demo]` (Streamlit),
`[runtime]` (faiss-cpu, sentence-transformers — retrieval container only).

## Test

```bash
make test    # pytest + coverage, fail_under=80 (~95% in practice)
```

Some tests are SKIPped outside CI because they need the `[runtime]` or `[demo]`
extras — install the relevant extra and re-run to verify them for real.

## Run the API locally

```bash
# mock backend (offline)
MODEL_BACKEND=mock .venv/bin/uvicorn medical_qa_platform.api.app:create_app --factory --reload

# real model
MODEL_BACKEND=llm LLM_BASE_URL=https://llm.<domain>/v1 LLM_MODEL=… LLM_API_KEY=… \
  .venv/bin/uvicorn medical_qa_platform.api.app:create_app --factory
```

Endpoints: `/health` · `/ready` · `/predict` · `/metrics` · `/version`.

## Streamlit demo UI

```bash
make demo-ui    # installs [demo], then streamlit run app/streamlit_app.py :8501
```

## Offline pipeline (DVC + MLflow)

```bash
make smoke-pipeline-local   # build_smoke_kg → eval_smoke → mlflow register (no DVC)
make smoke-pipeline         # same stages through DVC (dvc repro)
make full-pipeline-dry-run  # print the full-pipeline plan (structural in CI)
make full-pipeline          # REAL: orchestrate the training pipeline (needs free GB10 GPU + vLLM)
make smoke-full             # full pipeline with tiny caps (gate before a real run)
make dvc-status
```

`params.yaml` defines the `smoke` / `full` / `smoke_full` profiles; `dvc.yaml` +
`dvc.lock` track stage lineage. The full pipeline orchestrates training via
subprocess and registers the result through `mlops/mlflow_register.py`.

## Docker & Helm

```bash
make docker-build   # 4 images, --platform linux/amd64 (cross-build from aarch64 host)
make helm-lint      # lint 6 charts (api/retrieval/nginx/kserve/ui/monitoring)
make helm-template  # render charts
make helm-dry-run   # kubectl apply --dry-run for standard resources
```

The KServe `InferenceService` runs in **RawDeployment** mode (plain Deployment/Service,
no Knative/Istio) so it deploys on a lean GKE Standard zonal cluster; it serves
Qwen2.5-1.5B-Instruct on CPU via llama.cpp and is consumed through the `llm` backend.
The KServe chart is structurally tested locally because `kubectl --dry-run=client`
cannot validate `serving.kserve.io` resources without the CRD installed.

> Redeploys must use the image **sha tag** (`IMAGE_TAG=<sha> bash scripts/cloud/deploy.sh`)
> — charts use `pullPolicy: IfNotPresent`, so `:latest` won't pull new code. GHCR
> packages must be **public** (charts carry no imagePullSecret).

## Cloud deploy — GKE demo

One-click GitHub workflows with keyless (WIF/OIDC) CI/CD. The CI demo uses the `mock`
backend; flip to `MODEL_BACKEND=llm` for a real model. Retrieval is always real. Full
runbook: [`docs/cloud-setup.md`](docs/cloud-setup.md).

```bash
# one-time bootstrap (durable)
make cloud-gcs-dvc            # GCS DVC remote
make cloud-workload-identity  # WI for retrieval to pull the KG
make cloud-github-oidc        # keyless GitHub → GCP (WIF)

# demo lifecycle (also available as GitHub workflows)
make cloud-provision          # GKE Autopilot
make cloud-secrets            # nginx api-key + LLM key
make cloud-deploy             # helm upgrade --install
make cloud-smoke              # nginx → api → retrieval → model
make cloud-teardown           # delete cluster + LB (keeps the bucket)

# GKE Standard + in-cluster llama.cpp, one command:
GCP_PROJECT=… NGINX_API_KEY=$(openssl rand -hex 24) bash scripts/cloud/demo_up_llm.sh
```

- **Demo Up / Demo Down (GKE)** — manual workflows to bring the demo up
  (provision + deploy + smoke) and tear it down.
- **Auto Deploy** — pushes to `main` auto-roll new images onto the Autopilot
  `medical-qa` cluster while it is up, and skip green when it is down. (The manual
  GKE Standard `medical-qa-llm` cluster is deployed by hand with a sha tag.)

## Observability

Drift collection (`drift/collector.py`) and Prometheus metrics
(`observability/`) ship with the platform; a Helm `monitoring` chart deploys the
monitoring stack. See [`docs/runbooks/monitoring.md`](docs/runbooks/monitoring.md).

## Repository layout

```
src/medical_qa_platform/   runtime package (api, inference, retrieval, drift, observability)
mlops/                     pipelines (smoke / full), mlflow_register, smoke_data
deploy/helm/               6 Helm charts (api, retrieval, nginx, kserve, ui, monitoring)
docker/                    4 Dockerfiles (api, retrieval, pipeline-init, ui)
app/                       Streamlit demo UI
scripts/                   cloud/*.sh, gen_retrieval_golden.py, install_deploy_tools.py
tests/                     pytest suite (~95% coverage)
docs/                      specs, plans, runbooks, cloud-setup.md
params.yaml · dvc.yaml     pipeline profiles + stage lineage
```
