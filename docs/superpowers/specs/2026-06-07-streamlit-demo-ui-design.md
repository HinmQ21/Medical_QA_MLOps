# Streamlit Demo UI — Design

**Date:** 2026-06-07
**Status:** Approved
**Topic:** A browser demo frontend for the Medical QA platform, deployed to GKE alongside the API.

## Problem

The platform is API-only: `POST /predict` takes `{question, options{A..}}` and returns
`{answer, evidence[], backend, model_version, latency_ms, trace_id}`, reachable through the
public nginx LoadBalancer (`http://<nginx-lb>:8080`) behind an `x-api-key` gate. There is no
human-facing frontend — demoing the system means hand-crafting `curl` calls. We want a clean
Streamlit UI to enter a multiple-choice medical question and see the predicted answer plus the
KG evidence the model was given.

## Goals

- A focused MCQ demo: type a question + options A–D (preset example buttons), submit, see the
  highlighted answer letter, the KG evidence list, and response metadata badges.
- Deployed **to GKE** alongside the existing services (the operator chose the full deploy path,
  not local-only).
- Exercise the **real gateway + auth path** the way an external client would.
- **Zero changes** to the verified `/predict` end-to-end path (nginx → api → retrieval/vLLM).

## Non-Goals (YAGNI)

- No benchmark-dataset loader, no metrics/observability dashboard, no request history.
- No session persistence, no user accounts.
- No change to the API contract, the nginx gateway, or the retrieval/vLLM path.

## Architecture

Current topology (unchanged by this work):

```
Internet ─► nginx LB :8080 ──(x-api-key gate)──► medical-qa-api:8000 ─► medical-qa-retrieval:8001
                                                                       └► vllm → Tailscale → DGX vLLM
```

This work adds one service, `medical-qa-ui`, with its **own LoadBalancer**:

```
Internet ─► medical-qa-ui LB :8501  (Streamlit)              ◄── NEW, separate IP
Internet ─► medical-qa-nginx LB :8080 (API gateway)          ◄── untouched

medical-qa-ui pod ─► http://medical-qa-nginx:8080/predict    (x-api-key header, server-side)
                     key injected from Secret medical-qa-nginx-api-key (key: API_KEY)
```

### Decision A — UI exposed via its own LoadBalancer (chosen)

`medical-qa-ui` Service is `type: LoadBalancer` (via a `values-prod.yaml` overlay, mirroring
nginx), port 8501. Streamlit websockets work natively on a dedicated LB, and the existing API
gateway and the verified `/predict` e2e stay byte-for-byte unchanged. Cost: one extra cloud LB
IP — acceptable for a demo.

*Rejected:* folding the UI into the existing nginx (UI at `/`, API moved to `/api`). One IP, but
requires nginx websocket-upgrade config, **breaks the `/predict` path and `smoke_cloud.sh`**, and
drags in Streamlit `baseUrlPath`/XSRF configuration. More risk than a demo warrants.

### Decision B — UI reaches the API in-cluster through the nginx gateway, with the key (chosen)

The UI pod calls `http://medical-qa-nginx:8080/predict` (the in-cluster ClusterIP of the same
gateway external clients hit), attaching `x-api-key` read from the existing
`medical-qa-nginx-api-key` Secret. Single source of truth for the key; the key lives
**server-side in the UI pod and is never sent to the browser** (Streamlit runs Python
server-side). Demonstrates the real auth path.

*Rejected:* UI → `medical-qa-api:8000` directly (no key). Simpler, but bypasses the
gateway/auth and is less representative of the real system.

## Components

### New files

| File | Purpose |
|------|---------|
| `app/__init__.py` | Make `app` importable (pytest already sets `pythonpath = ["."]`). |
| `app/client.py` | **Pure, testable** HTTP client. Depends only on `httpx` (a core dependency) — **no Streamlit import**, so it runs in the normal venv. Functions: `build_payload(question, options) -> dict`, `predict(base_url, api_key, payload, timeout) -> PredictResult`, `fetch_version(base_url, api_key, timeout) -> dict`. Raises typed errors (`PredictError`) for 401 / timeout / connection failures so the UI can render friendly messages. |
| `app/streamlit_app.py` | The UI. Imports `app.client` + `streamlit`. Layout, widgets, rendering only — no business logic. |
| `docker/ui.Dockerfile` | `python:3.12-slim`; `uv pip install --system --no-cache ".[demo]"`; copy `app/`; non-root; `EXPOSE 8501`; `CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]`. |
| `deploy/helm/ui/Chart.yaml` | Chart metadata, mirroring `api/`. |
| `deploy/helm/ui/values.yaml` | `image.repository: ghcr.io/hinmq21/medical-qa-ui`, `service.type: ClusterIP`, `service.port: 8501`, `env.apiBaseUrl: http://medical-qa-nginx:8080`, `auth.existingSecret: medical-qa-nginx-api-key`, resources. |
| `deploy/helm/ui/values-prod.yaml` | `service.type: LoadBalancer` (cloud overlay, like nginx). |
| `deploy/helm/ui/templates/configmap.yaml` | `API_BASE_URL` from `.Values.env.apiBaseUrl`. |
| `deploy/helm/ui/templates/deployment.yaml` | Container on 8501; `envFrom` the configmap; `env: API_KEY` from `secretKeyRef` `medical-qa-nginx-api-key` key `API_KEY`; liveness/readiness `httpGet` on `/_stcore/health`. |
| `deploy/helm/ui/templates/service.yaml` | `type: {{ .Values.service.type }}`, port 8501 → targetPort `http`. |
| `tests/app/__init__.py` | Test package marker. |
| `tests/app/test_client.py` | Unit tests via `httpx.MockTransport`: payload build (validates 2–10 options and A–Z keys to match the server contract), success parse, 401 → typed error, timeout/connection → typed error. |
| `tests/deploy/test_ui_chart.py` | Render the chart with `helm template` (or read the rendered YAML) and assert: LB type under the prod overlay, port 8501, `API_BASE_URL` in the configmap, `API_KEY` `secretKeyRef` → `medical-qa-nginx-api-key`, probe path `/_stcore/health`. |

### Modified files

| File | Change |
|------|--------|
| `pyproject.toml` | Add optional extra `demo = ["streamlit>=1.40", "httpx>=0.27"]`. |
| `Makefile` | Add `deploy/helm/ui` to `helm-lint` and `helm-template`; add a `demo-ui` target (`streamlit run app/streamlit_app.py`) for local runs. |
| `scripts/cloud/deploy.sh` | `helm upgrade --install medical-qa-ui deploy/helm/ui -f deploy/helm/ui/values-prod.yaml` (after the nginx install). |
| `scripts/cloud/smoke_cloud.sh` | Add `medical-qa-ui` to the rollout-wait loop so deploy gates on the UI rolling out. |
| `tests/cloud/test_smoke_cloud.py` | Assert the smoke script waits for `medical-qa-ui` rollout. |
| `.github/workflows/ci.yml` | Add build-matrix entry `{ image: ui, dockerfile: docker/ui.Dockerfile }`. |
| `.github/workflows/demo-up.yml` | Also print the `medical-qa-ui` LoadBalancer IP at the end. |

## Data Flow

1. Browser opens `http://<ui-lb-ip>:8501`.
2. User types a question + options A–D, or clicks a preset (T2DM first-line, MI marker), then
   clicks **Chẩn đoán**.
3. Streamlit server-side calls `app.client.build_payload(...)` then
   `predict("http://medical-qa-nginx:8080", API_KEY, payload, timeout)` — `POST /predict` with the
   `x-api-key` header. The key comes from the pod's `API_KEY` env and is **never sent to the browser**.
4. The UI renders: the answer letter highlighted among the options; the KG evidence list; metadata
   badges (`backend`, `model_version`, `latency_ms`, `trace_id`).
5. The sidebar shows connection status from `/version` + `/health` on load (backend / model /
   contract version); the endpoint is editable (advanced); a key-override field is empty and of
   type `password`.
6. On timeout / 401 / connection error, the UI shows a friendly inline message — never a stack trace.

## Error Handling

- `app.client` converts httpx errors and non-2xx responses into a typed `PredictError` carrying a
  human-readable reason (e.g. "unauthorized — check the gateway key", "timed out after Ns",
  "could not reach the API"). `streamlit_app.py` catches it and renders `st.error(...)`.
- A `null` answer from the API (model produced no parseable letter) is shown as
  "no answer parsed" rather than crashing the answer-highlight logic.

## Testing

- Business logic lives in `app/client.py` (pure) so tests run without launching Streamlit and
  without the `[demo]` extra (`httpx` is already a core dependency). Streamlit rendering itself is
  not unit-tested (standard for Streamlit).
- The helm chart is verified by `tests/deploy/test_ui_chart.py` and by CI's existing `helm-lint` /
  `helm-template` targets (extended to include `deploy/helm/ui`).
- Coverage: `app/` sits outside the measured `medical_qa_platform` package, so it neither raises
  nor lowers the ≥80% `fail_under` gate — no regression to the existing suite.

## Deploy / CI

- `deploy.sh` installs the UI chart, so `demo-up.yml` and the Auto-Deploy `deploy.yml` pick it up
  for free (both call `deploy.sh`). No new workflow inputs are needed.
- A second LoadBalancer IP appears for the UI. The API gateway path is unchanged, so the verified
  `/predict` e2e (GKE nginx → api → Tailscale → DGX vLLM) carries no risk from this work.

## Open Questions / Risks

- **Streamlit image size:** installing `.[demo]` also pulls the package's core deps (fastapi etc.)
  the UI doesn't import. Acceptable for a demo; keeps a single dependency source. Revisit only if
  build time becomes a problem.
- **Second LB cost:** one extra GCP LoadBalancer for the demo window. Torn down with the rest by
  `demo-down.yml` (the UI Service is in the same namespace).
- **`model_version: smoke-dev`:** the UI surfaces whatever `/version` reports; the pre-existing
  `MODEL_VERSION` plumbing gap (deferred follow-up) is out of scope here.
