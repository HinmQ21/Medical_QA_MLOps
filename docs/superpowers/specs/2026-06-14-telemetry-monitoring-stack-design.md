# Telemetry & Monitoring Stack — Design

**Date:** 2026-06-14
**Status:** Approved (brainstorming complete)
**Repo:** `mlops-platform`

## Problem

The platform exposes Prometheus metrics but **nobody collects them**. Both the API and
retrieval services serve `/metrics` (`observability/metrics.py`), yet there is no
ServiceMonitor, no `prometheus.io` scrape annotation, no Prometheus server, and no Grafana
in any cluster. The `/metrics` endpoint is a dead end in the demo.

The in-process instrumentation is also thin and partly broken for observability purposes:

- `trace_id` is generated per `/predict` (`api/app.py`) but **never logged** — the JSON
  logger (`observability/logging.py`) supports a `trace_id` field that nothing populates.
- `tool_call_count` is computed in `LoopResult` (`inference/agent.py`) but **never surfaced**
  to any metric or log.
- Latency is only measured end-to-end (`mqa_request_latency_seconds`); there is **no split**
  between model time and retrieval time, so a slow demo can't be attributed.
- The drift signal **conflates two distinct cases**: `DriftCollector` and
  `mqa_retrieval_no_result_total` both treat "model never called the tool" the same as
  "tool was called but returned empty."
- `/predict` has **no error handling** — an exception inside the loop 500s with no metric
  and no logged context.

This design (a) **expands in-process instrumentation** so the signals above exist and are
correct, and (b) **stands up a real collection + visualization + alerting pipeline**
(Prometheus + Grafana + Alertmanager) in-cluster, portable across both demo clusters.

## Decisions (locked during brainstorming)

1. **Scope = full stack.** Both app instrumentation expansion AND collection/dashboard/alert
   infrastructure.
2. **Stack = `kube-prometheus-stack` (self-hosted)** — Prometheus Operator + Grafana +
   Alertmanager. Portable across GKE Autopilot (`medical-qa`, mock) and GKE Standard
   (`medical-qa-llm`, in-cluster llama.cpp). Chosen over Google Managed Prometheus
   (GCP lock-in, weaker Grafana story) and a plain no-operator install (no ServiceMonitor).
3. **Three pillars = metrics + dashboards + alerts only.** Logs stay JSON-on-stdout
   (viewed via `kubectl logs` / Cloud Logging), correlated by `trace_id`. **No Loki, no
   distributed tracing (Tempo/OTel)** — YAGNI for a 2-service system.
4. **Install pattern mirrors KServe.** `kube-prometheus-stack` is **not vendored** as a Helm
   dependency (avoids `helm dependency build` needing internet in CI); it is installed by a
   pinned `scripts/cloud/install_monitoring.sh` (same shape as `install_kserve.sh`). The repo
   owns only its own thin chart (rules + dashboards) plus ServiceMonitor templates.
5. **Alertmanager runs in-cluster with a null/demo receiver** — alerts are visible in the
   Alertmanager UI and Grafana's alerts panel; no external Slack/email wiring.
6. **One spec, plan in two phases.** Phase 1 (app instrumentation) is pure `src/`, fully
   CI-tested, mergeable independently. Phase 2 (infra) is helm/deploy, live-validated.
7. **Metric naming keeps the `mqa_` prefix.**

## Architecture

```
                          ┌──────────────── GKE cluster ────────────────────┐
                          │                                                  │
  api  /metrics ◀── scrape ┤  Prometheus (Operator)  ──remote──▶  Alertmanager (null receiver)
  retrieval /metrics ◀──── ┤        │  selects ServiceMonitors (label release)
                          │        │  evaluates PrometheusRule (our alerts)   │
                          │        ▼                                          │
                          │     Grafana ── sidecar auto-imports ── dashboard ConfigMap
                          └──────────────────────────────────────────────────┘

  install_monitoring.sh:  helm upgrade --install kube-prometheus-stack (PINNED) + our values
  deploy/helm/monitoring/: OUR chart  → PrometheusRule + Grafana dashboard ConfigMap
  deploy/helm/api,retrieval/: + ServiceMonitor template (gated by .Values.monitoring.enabled)
```

ServiceMonitor manifests render via `helm template` with no CRD present; `deploy.sh` only
`apply`s the monitoring contributions when CRD `servicemonitors.monitoring.coreos.com`
exists (mirrors the KServe CRD check) — skip is **non-fatal** on clusters without it.

## Phase 1 — App instrumentation (code, CI-testable)

### New / changed metrics (`observability/metrics.py`)

| Metric | Type | Labels | Meaning |
|--------|------|--------|---------|
| `mqa_model_latency_seconds` | Histogram | `backend` | Time inside `backend.chat()` (includes HTTP to vLLM/llama.cpp) |
| `mqa_tool_calls_per_request` | Histogram | — | Distribution of `tool_call_count` per `/predict` |
| `mqa_tool_outcome_total` | Counter | `outcome` ∈ {`not_called`,`empty`,`hit`} | Disambiguates the old conflated no-result signal |
| `mqa_build_info` | Gauge (=1) | `model_version`,`contract_version`,`backend` | Surfaces deployed versions to Prometheus |
| `mqa_requests_total` | Counter (existing) | + `status="error"` branch | Error path now counted |

Existing metrics (`mqa_requests_total`, `mqa_request_latency_seconds`,
`mqa_retrieval_latency_seconds`, `mqa_retrieval_no_result_total`) are **kept as-is**.
`mqa_retrieval_no_result_total` stays for backward compatibility; `mqa_tool_outcome_total`
is the new, correct signal.

New emitter helpers alongside the existing `observe_request` / `observe_retrieval`:

```python
def observe_model(backend: str, latency_s: float) -> None: ...
def observe_tool(tool_call_count: int, outcome: str) -> None: ...   # outcome: not_called|empty|hit
def set_build_info(model_version, contract_version, backend) -> None: ...  # gauge .labels(...).set(1)
```

### Loop changes (`inference/agent.py`)

`LoopResult` gains timing/count fields so the loop measures at the call site and `app.py`
emits (keeps the loop logic-pure, app-emits-metrics pattern intact):

```python
@dataclass
class LoopResult:
    trace: list[dict] = ...
    final_content: str = ""
    evidence: list[str] = ...
    tool_call_count: int = 0
    model_latency_s: float = 0.0     # NEW: summed time across backend.chat() calls
    iterations: int = 0              # NEW: number of loop rounds executed
```

`model_latency_s` accumulates `perf_counter()` deltas around each `backend.chat()`.
`iterations` counts rounds entered. Retrieval timing already lives in the retrieval service
(`mqa_retrieval_latency_seconds`); no new retrieval-side metric.

The **tool outcome** is derived once per request (not per tool call) for the
`mqa_tool_outcome_total` counter:
- `not_called` — `tool_call_count == 0`
- `empty` — tool called, but total `evidence` is empty
- `hit` — tool called and evidence non-empty

### `app.py` `/predict` changes

- Wrap the handler body in `try/except`: on exception, emit `observe_request(..., status="error", ...)`,
  log the exception with `trace_id`, and return HTTP 500.
- Emit `observe_model(backend, result.model_latency_s)` and
  `observe_tool(result.tool_call_count, outcome)`.
- Emit one structured request log via the existing logger:
  `logger.info("prediction", extra={"trace_id": trace_id, "latency_ms":..., "backend":...,
  "tool_call_count":..., "n_evidence":..., "status":...})`.
- Call `set_build_info(model_version, contract_version, backend)` once at startup (so the gauge
  is present even before the first request).

### Drift fix (`drift/collector.py`)

`record()` row replaces the conflated `no_result` boolean with explicit fields:

```python
row = {
    "q_token_len": ...,
    "answer": ...,
    "tool_call_count": tool_call_count,   # NEW
    "tool_called": tool_call_count > 0,   # NEW
    "n_evidence": n_evidence,             # NEW (was only used to compute no_result)
    "tool_outcome": outcome,              # NEW: not_called|empty|hit
    "latency_ms": ...,
}
```

`record()` signature gains `tool_call_count` (and derives `outcome`); the file-append
best-effort behavior is unchanged. Callers in `app.py` pass the new arg.

## Phase 2 — Monitoring infrastructure (helm/deploy, live-validated)

### `scripts/cloud/install_monitoring.sh` (mirrors `install_kserve.sh`)

`helm repo add prometheus-community ... && helm upgrade --install monitoring
prometheus-community/kube-prometheus-stack --version <PINNED> -n monitoring --create-namespace
-f <our values>`. Values: enable Grafana dashboard sidecar, set `serviceMonitorSelector` to
match `release: monitoring` (default), Alertmanager null receiver, modest resource requests.
Version pinned (new vars in `scripts/cloud/config.sh`, e.g. `KUBE_PROM_STACK_VERSION`).

### `deploy/helm/monitoring/` (our chart)

- `templates/prometheusrule.yaml` — 4 alerts:
  - `HighErrorRate` — `error` ratio of `mqa_requests_total` > X% over 5m
  - `HighLatencyP95` — p95 of `mqa_request_latency_seconds{endpoint="/predict"}` > threshold
    (configurable via values; default high because llama.cpp CPU is slow)
  - `RetrievalEmptyRateHigh` — `empty` ratio of `mqa_tool_outcome_total` high over 15m
  - `TargetDown` — `up{job=~"medical-qa-.*"} == 0`
- `templates/dashboard-configmap.yaml` — ConfigMap labeled `grafana_dashboard: "1"` carrying
  the "Medical QA Serving" dashboard JSON.
- `values.yaml` — alert thresholds, namespace, `enabled` toggle.

### ServiceMonitor templates (in existing `api` + `retrieval` charts)

`deploy/helm/{api,retrieval}/templates/servicemonitor.yaml`, gated by
`.Values.monitoring.enabled` (default false so existing installs are unchanged), with
`release: monitoring` label so the stack selects them. Scrapes the existing `/metrics` port.

### Grafana dashboard "Medical QA Serving"

Panels: request rate by status; latency p50/p95/p99 (total / model / retrieval); tool-calls
per request distribution; `mqa_tool_outcome_total` breakdown (not_called/empty/hit); error
rate; retrieval no-result rate; `mqa_build_info` table.

### `deploy.sh` wiring + runbook

`deploy.sh` applies the `monitoring` chart + ServiceMonitors **only if** the ServiceMonitor
CRD exists (non-fatal skip otherwise). New runbook `docs/runbooks/monitoring.md`: how to
install the stack, port-forward Grafana, default creds, where the dashboard/alerts live,
live-validation steps.

## Testing

**Phase 1 (unit, CI):**
- Parse `/metrics` output and assert the new metric names are present after a `/predict`.
- `observe_model` / `observe_tool` / `set_build_info` update the right series.
- `LoopResult.model_latency_s` and `iterations` are populated (monkeypatch `perf_counter`
  or assert `>= 0` and round count).
- `/predict` error path: a backend that raises increments `status="error"` and returns 500.
- `/predict` logs a `prediction` line carrying `trace_id` (capture via `caplog`).
- `DriftCollector.record` writes the new fields and correct `tool_outcome` for all three cases.
- Self-contained guard (`tests/retrieval/test_kg_backend_self_contained.py`) still passes —
  no `baseline` tokens introduced in `src/`.

**Phase 2 (helm, CI):**
- `helm lint` + `helm template` the `monitoring` chart and the ServiceMonitor templates.
- Assert ServiceMonitor renders with the `release: monitoring` selector label and the
  correct `/metrics` endpoint; assert PrometheusRule renders the 4 named alerts.
- Validate the Grafana dashboard JSON parses (`json.load`).

**Live (manual, per project convention):** no Prometheus in CI. Verify on GKE Standard
(`medical-qa-llm`, in-cluster llama.cpp) — the cluster where latency telemetry is most
meaningful: install stack, confirm targets `up`, fire a `/predict`, see panels populate,
trip an alert by lowering a threshold.

## Boundaries & non-goals

- **No baseline import** — all instrumentation is pure `src/` code; the guard test stands.
- **No Loki / no distributed tracing** — explicitly out of scope (decision #3).
- **No external alert routing** (Slack/email) — null receiver only (decision #5).
- **No new backend, no ranker/encoder/contract change** — retrieval parity untouched.
- **Autopilot (`medical-qa`) auto-deploy is unchanged** — monitoring is opt-in via
  `monitoring.enabled` + the install script; the existing CI deploy path is not forced to
  install Prometheus.

## Open questions / risks

- **kube-prometheus-stack resource footprint** on GKE Standard (Prometheus TSDB + Grafana +
  operator) — may need trimmed retention / requests; tune in the install values.
- **GHCR / image pull** is irrelevant here (upstream chart pulls from quay.io/grafana.com) —
  but confirm those registries are reachable from the cluster.
- **Pinned chart version drift** — pin a known-good `kube-prometheus-stack` version; document
  the upgrade path in the runbook.
- Dashboard JSON is verbose; keep it minimal and readable rather than importing a giant
  community dashboard.
