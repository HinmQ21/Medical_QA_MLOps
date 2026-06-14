# Telemetry & Monitoring Stack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand in-process serving telemetry (model latency, tool-call outcome, build info, trace_id logging, error tracking, drift fix) and stand up a self-hosted Prometheus + Grafana + Alertmanager pipeline in-cluster.

**Architecture:** Phase 1 adds metrics/helpers in `observability/metrics.py`, timing fields in `inference/agent.py`'s `LoopResult`, corrected drift rows in `drift/collector.py`, and wires emission + structured logging + error handling into `api/app.py` — all pure `src/` code, fully unit-tested in CI. Phase 2 installs `kube-prometheus-stack` via a pinned `scripts/cloud/install_monitoring.sh` (mirroring `install_kserve.sh`) and adds a new `deploy/helm/monitoring/` chart holding the ServiceMonitors, PrometheusRule alerts, and a Grafana dashboard ConfigMap. `deploy.sh` installs that chart only when the ServiceMonitor CRD is present (non-fatal skip otherwise), exactly like the KServe block.

**Tech Stack:** Python 3.12, `prometheus_client`, FastAPI, pytest, Helm, `kube-prometheus-stack` (Prometheus Operator + Grafana + Alertmanager), bash.

**Spec:** `docs/superpowers/specs/2026-06-14-telemetry-monitoring-stack-design.md`

**Note on a spec refinement (locked):** The spec placed ServiceMonitor templates inside the `api`/`retrieval` charts. This plan instead puts **all** Prometheus-Operator CRD-typed resources (ServiceMonitor ×2, PrometheusRule, dashboard ConfigMap) inside the single new `monitoring` chart. This makes deploy-gating a single conditional chart install (like KServe), leaves the `api`/`retrieval` charts untouched, and is functionally identical (the ServiceMonitors still select the services by their existing `app.kubernetes.io/name` label).

---

## File Structure

**Phase 1 (code):**
- Modify: `src/medical_qa_platform/observability/metrics.py` — new metrics + emit helpers
- Modify: `src/medical_qa_platform/inference/agent.py` — `LoopResult.model_latency_s`, `.iterations`, timing
- Modify: `src/medical_qa_platform/drift/collector.py` — corrected drift row fields
- Modify: `src/medical_qa_platform/api/app.py` — emit metrics, structured log, error handling, build_info
- Modify tests: `tests/observability/test_metrics.py`, `tests/inference/test_agent.py`, `tests/drift/test_collector.py`, `tests/api/test_app.py`

**Phase 2 (infra):**
- Modify: `scripts/cloud/config.sh` — pinned `KUBE_PROM_STACK_VERSION`
- Create: `scripts/cloud/install_monitoring.sh`
- Create: `deploy/helm/monitoring/Chart.yaml`, `values.yaml`, `templates/servicemonitor-api.yaml`, `templates/servicemonitor-retrieval.yaml`, `templates/prometheusrule.yaml`, `templates/dashboard-configmap.yaml`, `dashboards/medical-qa-serving.json`
- Modify: `scripts/cloud/deploy.sh` — conditional monitoring chart install
- Modify: `Makefile` — add monitoring to `helm-lint`/`helm-template`
- Create: `docs/runbooks/monitoring.md`
- Create tests: `tests/cloud/test_install_monitoring.py`, `tests/deploy/test_helm_monitoring_chart.py`; Modify: `tests/cloud/test_deploy_sh.py`

---

# PHASE 1 — App Instrumentation

## Task 1: Model-latency metric + `observe_model`

**Files:**
- Modify: `src/medical_qa_platform/observability/metrics.py`
- Test: `tests/observability/test_metrics.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/observability/test_metrics.py`:

```python
def test_observe_model_appears_in_render():
    metrics.observe_model(backend="mock", latency_s=0.05)
    text = metrics.render_metrics()[0].decode()
    assert "mqa_model_latency_seconds" in text
    assert 'backend="mock"' in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/observability/test_metrics.py::test_observe_model_appears_in_render -v`
Expected: FAIL with `AttributeError: module ... has no attribute 'observe_model'`

- [ ] **Step 3: Implement**

In `src/medical_qa_platform/observability/metrics.py`, add the metric after `RETRIEVAL_NO_RESULT` (keep existing imports; `Counter, Histogram` already imported):

```python
MODEL_LATENCY = Histogram(
    "mqa_model_latency_seconds",
    "Time spent inside model backend.chat() calls per request.",
    ["backend"],
)
```

And add the emit helper after `observe_retrieval`:

```python
def observe_model(backend: str, latency_s: float) -> None:
    MODEL_LATENCY.labels(backend=backend).observe(latency_s)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/observability/test_metrics.py::test_observe_model_appears_in_render -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/medical_qa_platform/observability/metrics.py tests/observability/test_metrics.py
git commit -m "feat(observability): mqa_model_latency_seconds + observe_model"
```

---

## Task 2: Tool-outcome counter + tool-calls histogram + `observe_tool`

**Files:**
- Modify: `src/medical_qa_platform/observability/metrics.py`
- Test: `tests/observability/test_metrics.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/observability/test_metrics.py`:

```python
def test_observe_tool_records_outcome_and_count():
    metrics.observe_tool(tool_call_count=2, outcome="hit")
    metrics.observe_tool(tool_call_count=0, outcome="not_called")
    text = metrics.render_metrics()[0].decode()
    assert "mqa_tool_outcome_total" in text
    assert 'outcome="hit"' in text
    assert 'outcome="not_called"' in text
    assert "mqa_tool_calls_per_request" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/observability/test_metrics.py::test_observe_tool_records_outcome_and_count -v`
Expected: FAIL with `AttributeError: ... 'observe_tool'`

- [ ] **Step 3: Implement**

In `metrics.py`, add after `MODEL_LATENCY`:

```python
TOOL_OUTCOME = Counter(
    "mqa_tool_outcome_total",
    "Per-request tool usage outcome.",
    ["outcome"],  # not_called | empty | hit
)
TOOL_CALLS_PER_REQUEST = Histogram(
    "mqa_tool_calls_per_request",
    "Number of tool calls the model made in one /predict.",
    buckets=(0, 1, 2, 3, 4, 5),
)
```

And the helper after `observe_model`:

```python
def observe_tool(tool_call_count: int, outcome: str) -> None:
    TOOL_OUTCOME.labels(outcome=outcome).inc()
    TOOL_CALLS_PER_REQUEST.observe(tool_call_count)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/observability/test_metrics.py::test_observe_tool_records_outcome_and_count -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/medical_qa_platform/observability/metrics.py tests/observability/test_metrics.py
git commit -m "feat(observability): mqa_tool_outcome_total + mqa_tool_calls_per_request"
```

---

## Task 3: Build-info gauge + `set_build_info`

**Files:**
- Modify: `src/medical_qa_platform/observability/metrics.py`
- Test: `tests/observability/test_metrics.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/observability/test_metrics.py` (add `Gauge` is internal; test only via render):

```python
def test_set_build_info_exposes_versions():
    metrics.set_build_info(
        model_version="test-v1",
        contract_version="v1-medembed-small",
        backend="mock",
    )
    text = metrics.render_metrics()[0].decode()
    assert "mqa_build_info" in text
    assert 'model_version="test-v1"' in text
    assert 'contract_version="v1-medembed-small"' in text
    assert 'backend="mock"' in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/observability/test_metrics.py::test_set_build_info_exposes_versions -v`
Expected: FAIL with `AttributeError: ... 'set_build_info'`

- [ ] **Step 3: Implement**

In `metrics.py`, change the import line to include `Gauge`:

```python
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
```

Add after `TOOL_CALLS_PER_REQUEST`:

```python
BUILD_INFO = Gauge(
    "mqa_build_info",
    "Deployed version info (value is always 1).",
    ["model_version", "contract_version", "backend"],
)
```

Add the helper after `observe_tool`:

```python
def set_build_info(model_version: str | None, contract_version: str, backend: str) -> None:
    BUILD_INFO.labels(
        model_version=model_version or "unknown",
        contract_version=contract_version,
        backend=backend,
    ).set(1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/observability/test_metrics.py::test_set_build_info_exposes_versions -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/medical_qa_platform/observability/metrics.py tests/observability/test_metrics.py
git commit -m "feat(observability): mqa_build_info gauge + set_build_info"
```

---

## Task 4: `LoopResult` timing + iteration count

**Files:**
- Modify: `src/medical_qa_platform/inference/agent.py`
- Test: `tests/inference/test_agent.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/inference/test_agent.py` (it already imports the loop; add this self-contained test):

```python
def test_loop_result_reports_iterations_and_model_latency():
    from medical_qa_platform.inference.agent import run_agentic_loop
    from medical_qa_platform.inference.base import ChatTurn, ModelBackend
    from medical_qa_platform.retrieval.backends import FixtureRetrieval

    class _OneShot(ModelBackend):
        name = "x"

        def chat(self, messages, tools=None, tool_choice="auto", max_tokens=512, temperature=0.3):
            return ChatTurn(content="<answer>A</answer>", tool_calls=[], finish_reason="stop")

    result = run_agentic_loop(
        _OneShot(), FixtureRetrieval({}), "Q?",
        top_k=5, max_tokens=512, max_iterations=2,
    )
    assert result.iterations == 1            # answered on the first round
    assert result.model_latency_s >= 0.0
    assert result.tool_call_count == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/inference/test_agent.py::test_loop_result_reports_iterations_and_model_latency -v`
Expected: FAIL with `AttributeError: 'LoopResult' object has no attribute 'iterations'`

- [ ] **Step 3: Implement**

In `src/medical_qa_platform/inference/agent.py`:

(a) Add `from time import perf_counter` under the existing `import json`:

```python
import json
from dataclasses import dataclass, field
from time import perf_counter
```

(b) Extend `LoopResult`:

```python
@dataclass
class LoopResult:
    trace: list[dict] = field(default_factory=list)
    final_content: str = ""
    evidence: list[str] = field(default_factory=list)
    tool_call_count: int = 0
    model_latency_s: float = 0.0
    iterations: int = 0
```

(c) Inside the `for i in range(max_iterations + 1):` loop, count the iteration and time each `backend.chat()`. Replace the existing if/else that assigns `turn`:

```python
    for i in range(max_iterations + 1):
        result.iterations += 1
        _t = perf_counter()
        if i == max_iterations:
            # Final round: offer no tools so the model must answer in text.
            turn = backend.chat(messages, tools=None, max_tokens=max_tokens)
        else:
            turn = backend.chat(
                messages,
                tools=[MEDICAL_TOOL_DEF],
                tool_choice="auto",
                max_tokens=max_tokens,
            )
        result.model_latency_s += perf_counter() - _t
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/inference/test_agent.py::test_loop_result_reports_iterations_and_model_latency -v`
Expected: PASS

- [ ] **Step 5: Run the whole agent suite (no regressions)**

Run: `.venv/bin/pytest tests/inference/test_agent.py -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/medical_qa_platform/inference/agent.py tests/inference/test_agent.py
git commit -m "feat(inference): LoopResult model_latency_s + iterations"
```

---

## Task 5: Drift collector — disambiguate tool outcome

**Files:**
- Modify: `src/medical_qa_platform/drift/collector.py`
- Test: `tests/drift/test_collector.py`

- [ ] **Step 1: Rewrite the failing tests**

Replace the bodies of the three `no_result`-based tests in `tests/drift/test_collector.py`. Replace `test_record_computes_features`, `test_no_result_flag_true_when_zero_evidence`, and the `record(...)` calls in `test_record_does_not_raise_when_path_unwritable` / `test_record_appends_jsonl` so every `record()` call passes `tool_call_count`. The full updated test file:

```python
import json

from medical_qa_platform.api.schemas import PredictRequest, PredictResponse
from medical_qa_platform.drift.collector import DriftCollector


def _req():
    return PredictRequest(question="what treats diabetes now")


def _resp(answer, evidence):
    return PredictResponse(
        answer=answer,
        raw_output="<answer>x</answer>",
        evidence=evidence,
        backend="mock",
        model_version="dev",
        contract_version="v1-medembed-small",
        latency_ms=10.0,
        trace_id="t1",
    )


def test_record_computes_features():
    row = DriftCollector(path=None).record(
        _req(), _resp("A", ["e1"]), n_evidence=1, tool_call_count=1
    )
    assert row["q_token_len"] == 4
    assert row["answer"] == "A"
    assert row["tool_call_count"] == 1
    assert row["tool_called"] is True
    assert row["n_evidence"] == 1
    assert row["tool_outcome"] == "hit"
    assert row["latency_ms"] == 10.0
    assert "no_result" not in row


def test_outcome_not_called_when_model_skips_tool():
    row = DriftCollector(path=None).record(
        _req(), _resp("A", []), n_evidence=0, tool_call_count=0
    )
    assert row["tool_called"] is False
    assert row["tool_outcome"] == "not_called"


def test_outcome_empty_when_tool_called_but_no_evidence():
    row = DriftCollector(path=None).record(
        _req(), _resp(None, []), n_evidence=0, tool_call_count=1
    )
    assert row["tool_outcome"] == "empty"
    assert row["answer"] == "none"


def test_record_does_not_raise_when_path_unwritable():
    collector = DriftCollector(path="/no-such-dir-xyz/drift.jsonl")
    row = collector.record(_req(), _resp("A", ["e1"]), n_evidence=1, tool_call_count=1)
    assert row["answer"] == "A"


def test_record_appends_jsonl(tmp_path):
    path = tmp_path / "drift.jsonl"
    collector = DriftCollector(path=str(path))
    collector.record(_req(), _resp("A", ["e1"]), n_evidence=1, tool_call_count=1)
    collector.record(_req(), _resp("B", []), n_evidence=0, tool_call_count=0)
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["tool_outcome"] == "not_called"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/drift/test_collector.py -v`
Expected: FAIL with `TypeError: record() got an unexpected keyword argument 'tool_call_count'`

- [ ] **Step 3: Implement**

Replace the `record` method in `src/medical_qa_platform/drift/collector.py`:

```python
    def record(
        self,
        request: PredictRequest,
        response: PredictResponse,
        n_evidence: int,
        tool_call_count: int,
    ) -> dict:
        if tool_call_count == 0:
            outcome = "not_called"
        elif n_evidence == 0:
            outcome = "empty"
        else:
            outcome = "hit"
        row = {
            "q_token_len": len(request.question.split()),
            "answer": response.answer if response.answer is not None else "none",
            "tool_call_count": tool_call_count,
            "tool_called": tool_call_count > 0,
            "n_evidence": n_evidence,
            "tool_outcome": outcome,
            "latency_ms": response.latency_ms,
        }
        if self.path is not None:
            try:
                with open(self.path, "a") as handle:
                    handle.write(json.dumps(row) + "\n")
            except OSError:
                # Drift logging is best-effort observability; a write failure
                # (read-only/unwritable path) must never break the prediction.
                pass
        return row
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/drift/test_collector.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/medical_qa_platform/drift/collector.py tests/drift/test_collector.py
git commit -m "feat(drift): record tool_outcome (not_called/empty/hit) instead of conflated no_result"
```

---

## Task 6: Wire metric emission + drift arg into `/predict`

**Files:**
- Modify: `src/medical_qa_platform/api/app.py`
- Test: `tests/api/test_app.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/api/test_app.py`:

```python
def test_metrics_include_model_and_tool_and_build_info(tmp_path):
    client = _client(tmp_path)
    client.post("/predict", json={"question": "Q?"})
    text = client.get("/metrics").text
    assert "mqa_model_latency_seconds" in text
    assert "mqa_tool_outcome_total" in text
    assert 'outcome="not_called"' in text  # MockBackend makes no tool call
    assert "mqa_build_info" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/api/test_app.py::test_metrics_include_model_and_tool_and_build_info -v`
Expected: FAIL (`mqa_model_latency_seconds`/`mqa_build_info` absent)

- [ ] **Step 3: Implement**

In `src/medical_qa_platform/api/app.py`:

(a) Update the metrics import:

```python
from ..observability.metrics import (
    observe_model,
    observe_request,
    observe_tool,
    render_metrics,
    set_build_info,
)
```

(b) After `app.state.collector = DriftCollector(...)` and the `if backend is not None:` block, set build info when the backend is already known (test + injected-backend path):

```python
    if backend is not None:
        app.state.backend = backend
        set_build_info(app.state.model_version, RETRIEVAL_CONTRACT_VERSION, backend.name)
    if retrieval is not None:
        app.state.retrieval = retrieval
```

(c) In `_startup`, after `app.state.backend` is resolved (both branches), add the build-info call. Change the startup body so it ends with:

```python
    @app.on_event("startup")
    def _startup() -> None:
        if backend is not None:
            app.state.backend = backend
        else:
            from ..inference import get_backend

            app.state.backend = get_backend(settings.model_backend)
        if retrieval is not None:
            app.state.retrieval = retrieval
        else:
            from ..retrieval.client import RetrievalClient

            app.state.retrieval = RetrievalClient.from_env()
        set_build_info(
            app.state.model_version,
            RETRIEVAL_CONTRACT_VERSION,
            app.state.backend.name,
        )
```

(d) In `predict`, after `latency_ms` is computed and before/after `observe_request`, derive the outcome and emit the new metrics, and pass `tool_call_count` to the collector. Replace the tail of `predict` (from `observe_request(` through the `return resp`) with:

```python
        if result.tool_call_count == 0:
            outcome = "not_called"
        elif not result.evidence:
            outcome = "empty"
        else:
            outcome = "hit"
        status = "ok" if answer is not None else "no_answer"
        observe_request(
            endpoint="/predict",
            backend=app.state.backend.name,
            status=status,
            latency_s=latency_ms / 1000.0,
        )
        observe_model(app.state.backend.name, result.model_latency_s)
        observe_tool(result.tool_call_count, outcome)
        app.state.collector.record(
            req, resp, n_evidence=len(result.evidence), tool_call_count=result.tool_call_count
        )
        return resp
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/api/test_app.py::test_metrics_include_model_and_tool_and_build_info -v`
Expected: PASS

- [ ] **Step 5: Run the full api suite (drift call signature changed)**

Run: `.venv/bin/pytest tests/api/test_app.py -v`
Expected: all PASS (the existing `test_predict_writes_drift_row` still passes — the collector call now includes `tool_call_count`)

- [ ] **Step 6: Commit**

```bash
git add src/medical_qa_platform/api/app.py tests/api/test_app.py
git commit -m "feat(api): emit model/tool/build_info metrics; pass tool_call_count to drift"
```

---

## Task 7: Structured per-prediction log with `trace_id`

**Files:**
- Modify: `src/medical_qa_platform/api/app.py`
- Test: `tests/api/test_app.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/api/test_app.py` (own handler on the named logger — robust regardless of root config):

```python
def test_predict_logs_structured_trace_id(tmp_path):
    import logging

    records = []

    class _Cap(logging.Handler):
        def emit(self, record):
            records.append(record)

    logger = logging.getLogger("medical_qa_platform.api")
    handler = _Cap()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        client = _client(tmp_path)
        body = client.post("/predict", json={"question": "Q?"}).json()
    finally:
        logger.removeHandler(handler)

    logged = [r for r in records if getattr(r, "trace_id", None)]
    assert logged, "expected a prediction log carrying trace_id"
    assert logged[-1].trace_id == body["trace_id"]
    assert logged[-1].msg == "prediction"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/api/test_app.py::test_predict_logs_structured_trace_id -v`
Expected: FAIL (no `prediction` log emitted)

- [ ] **Step 3: Implement**

In `src/medical_qa_platform/api/app.py`:

(a) Add the logging import and a module-level logger near the top imports:

```python
from ..observability.logging import configure_logging, get_logger
```

After the imports, before `def create_app(`:

```python
logger = get_logger("medical_qa_platform.api")
```

(b) At the very top of `create_app`, configure logging (only touches the root logger; safe for tests that attach to the named logger):

```python
def create_app(
    ...
) -> FastAPI:
    configure_logging()
    settings = Settings.from_env()
```

(c) In `predict`, just before `return resp`, emit the structured log:

```python
        logger.info(
            "prediction",
            extra={
                "trace_id": trace_id,
                "latency_ms": latency_ms,
                "backend": app.state.backend.name,
                "tool_call_count": result.tool_call_count,
                "n_evidence": len(result.evidence),
                "status": status,
            },
        )
        return resp
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/api/test_app.py::test_predict_logs_structured_trace_id -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/medical_qa_platform/api/app.py tests/api/test_app.py
git commit -m "feat(api): structured per-prediction log carrying trace_id"
```

---

## Task 8: Error tracking on `/predict`

**Files:**
- Modify: `src/medical_qa_platform/api/app.py`
- Test: `tests/api/test_app.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/api/test_app.py`:

```python
def test_predict_error_increments_metric_and_returns_500(tmp_path):
    class _Boom(ModelBackend):
        name = "boom"

        def chat(self, messages, tools=None, tool_choice="auto", max_tokens=512, temperature=0.3):
            raise RuntimeError("backend exploded")

    app = create_app(
        backend=_Boom(),
        retrieval=FixtureRetrieval({}),
        model_version="x",
        drift_log_path=str(tmp_path / "d.jsonl"),
    )
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/predict", json={"question": "Q?"})
    assert resp.status_code == 500
    text = client.get("/metrics").text
    assert 'status="error"' in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/api/test_app.py::test_predict_error_increments_metric_and_returns_500 -v`
Expected: FAIL (unhandled exception / no `status="error"` series)

- [ ] **Step 3: Implement**

In `src/medical_qa_platform/api/app.py`:

(a) Add `HTTPException` to the fastapi import:

```python
from fastapi import FastAPI, HTTPException, Response
```

(b) Wrap the `predict` body in try/except. The handler becomes:

```python
    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        t0 = time.perf_counter()
        trace_id = uuid.uuid4().hex
        try:
            result = run_agentic_loop(
                app.state.backend,
                app.state.retrieval,
                req.question,
                top_k=app.state.top_k,
                max_tokens=app.state.max_tokens,
                max_iterations=app.state.max_tool_iterations,
            )
            answer = parse_answer(result.final_content)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            resp = PredictResponse(
                answer=answer,
                raw_output=result.final_content,
                evidence=result.evidence,
                trace=[Turn(**t) for t in result.trace],
                backend=app.state.backend.name,
                model_version=app.state.model_version,
                contract_version=RETRIEVAL_CONTRACT_VERSION,
                latency_ms=latency_ms,
                trace_id=trace_id,
            )
            if result.tool_call_count == 0:
                outcome = "not_called"
            elif not result.evidence:
                outcome = "empty"
            else:
                outcome = "hit"
            status = "ok" if answer is not None else "no_answer"
            observe_request(
                endpoint="/predict",
                backend=app.state.backend.name,
                status=status,
                latency_s=latency_ms / 1000.0,
            )
            observe_model(app.state.backend.name, result.model_latency_s)
            observe_tool(result.tool_call_count, outcome)
            app.state.collector.record(
                req, resp, n_evidence=len(result.evidence), tool_call_count=result.tool_call_count
            )
            logger.info(
                "prediction",
                extra={
                    "trace_id": trace_id,
                    "latency_ms": latency_ms,
                    "backend": app.state.backend.name,
                    "tool_call_count": result.tool_call_count,
                    "n_evidence": len(result.evidence),
                    "status": status,
                },
            )
            return resp
        except Exception:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            observe_request(
                endpoint="/predict",
                backend=app.state.backend.name,
                status="error",
                latency_s=latency_ms / 1000.0,
            )
            logger.exception("prediction_failed", extra={"trace_id": trace_id})
            raise HTTPException(status_code=500, detail="prediction failed")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/api/test_app.py::test_predict_error_increments_metric_and_returns_500 -v`
Expected: PASS

- [ ] **Step 5: Run the full api suite**

Run: `.venv/bin/pytest tests/api/test_app.py -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/medical_qa_platform/api/app.py tests/api/test_app.py
git commit -m "feat(api): error tracking on /predict (status=error metric + 500 + logged exception)"
```

---

## Task 9: Phase 1 gate — full suite + self-contained guard

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite with the coverage gate**

Run: `make test`
Expected: all pass, `1 skipped` (runtime/demo extras), coverage ≥ 80% (was ~96%).

- [ ] **Step 2: Confirm the self-contained guard still passes**

Run: `.venv/bin/pytest tests/retrieval/test_kg_backend_self_contained.py -v`
Expected: PASS (no `baseline` tokens introduced in `src/`).

- [ ] **Step 3: Confirm no stale `no_result` key consumer remains**

Run: `grep -rn '"no_result"\|\[.no_result.\]' src/ tests/`
Expected: only `mqa_retrieval_no_result_total` (the retrieval-service metric, unrelated) — no references to the removed DriftCollector `no_result` row key.

---

# PHASE 2 — Monitoring Infrastructure

## Task 10: Pin stack version + `install_monitoring.sh`

**Files:**
- Modify: `scripts/cloud/config.sh`
- Create: `scripts/cloud/install_monitoring.sh`
- Create test: `tests/cloud/test_install_monitoring.py`

- [ ] **Step 1: Write the failing test**

Create `tests/cloud/test_install_monitoring.py`:

```python
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts/cloud/install_monitoring.sh"
ENV = {"PATH": "/usr/bin:/bin", "GCP_PROJECT": "demo"}


def test_script_parses_with_bash_n():
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0


def test_has_strict_mode_and_dry_run():
    text = SCRIPT.read_text()
    assert "set -euo pipefail" in text
    assert "--dry-run" in text


def test_dry_run_adds_repo_then_installs_stack_pinned():
    out = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"], capture_output=True, text=True, env=ENV
    )
    assert out.returncode == 0, out.stderr
    o = out.stdout
    add = o.index("repo add prometheus-community")
    inst = o.index("upgrade --install monitoring prometheus-community/kube-prometheus-stack")
    assert add < inst
    assert "--version v65.5.1" in o          # pinned KUBE_PROM_STACK_VERSION
    assert "grafana.sidecar.dashboards.enabled=true" in o
    assert "grafana.sidecar.dashboards.searchNamespace=ALL" in o
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/cloud/test_install_monitoring.py -v`
Expected: FAIL (`install_monitoring.sh` does not exist)

- [ ] **Step 3: Implement — pin the version in `config.sh`**

In `scripts/cloud/config.sh`, after the `KSERVE_VERSION` line, add:

```bash
# Pinned kube-prometheus-stack chart version for install_monitoring.sh.
: "${KUBE_PROM_STACK_VERSION:=v65.5.1}"
```

- [ ] **Step 4: Implement — `install_monitoring.sh`**

Create `scripts/cloud/install_monitoring.sh`:

```bash
#!/usr/bin/env bash
# Install kube-prometheus-stack (Prometheus Operator + Grafana + Alertmanager) onto the
# current-context cluster. Idempotent (helm upgrade --install). Provides the
# servicemonitors.monitoring.coreos.com CRD that deploy.sh gates the monitoring chart on.
# Grafana's dashboard sidecar is enabled so the monitoring chart's dashboard ConfigMap
# is auto-imported. Version is pinned (Helm will not auto-resolve "latest").
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/cloud/config.sh
source "$HERE/config.sh"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
HELM="$REPO_ROOT/.tools/bin/helm"
KUBECTL="$REPO_ROOT/.tools/bin/kubectl"

DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

run() {
  if [ "$DRY_RUN" -eq 1 ]; then echo "+ $*"; else "$@"; fi
}

run "$HELM" repo add prometheus-community https://prometheus-community.github.io/helm-charts
run "$HELM" repo update
run "$HELM" upgrade --install monitoring prometheus-community/kube-prometheus-stack \
  --version "$KUBE_PROM_STACK_VERSION" \
  --namespace monitoring --create-namespace \
  --set grafana.sidecar.dashboards.enabled=true \
  --set grafana.sidecar.dashboards.searchNamespace=ALL \
  --set alertmanager.enabled=true
run "$KUBECTL" -n monitoring rollout status deploy/monitoring-kube-prometheus-operator --timeout=300s
echo "kube-prometheus-stack $KUBE_PROM_STACK_VERSION installed (Prometheus + Grafana + Alertmanager)."
```

Make it executable:

```bash
chmod +x scripts/cloud/install_monitoring.sh
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/pytest tests/cloud/test_install_monitoring.py -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/cloud/config.sh scripts/cloud/install_monitoring.sh tests/cloud/test_install_monitoring.py
git commit -m "feat(cloud): install_monitoring.sh (pinned kube-prometheus-stack)"
```

---

## Task 11: `monitoring` chart scaffold + ServiceMonitors

**Files:**
- Create: `deploy/helm/monitoring/Chart.yaml`, `deploy/helm/monitoring/values.yaml`
- Create: `deploy/helm/monitoring/templates/servicemonitor-api.yaml`, `templates/servicemonitor-retrieval.yaml`
- Create test: `tests/deploy/test_helm_monitoring_chart.py`

- [ ] **Step 1: Write the failing test**

Create `tests/deploy/test_helm_monitoring_chart.py`:

```python
from .helm_helpers import find_kind, render_chart


def test_servicemonitors_select_services_and_carry_release_label():
    resources = render_chart("monitoring")
    api_sm = find_kind(resources, "ServiceMonitor", "medical-qa-api")
    ret_sm = find_kind(resources, "ServiceMonitor", "medical-qa-retrieval")

    assert api_sm["metadata"]["labels"]["release"] == "monitoring"
    assert api_sm["spec"]["selector"]["matchLabels"]["app.kubernetes.io/name"] == "medical-qa-api"
    assert api_sm["spec"]["endpoints"][0]["port"] == "http"
    assert api_sm["spec"]["endpoints"][0]["path"] == "/metrics"

    assert ret_sm["spec"]["selector"]["matchLabels"]["app.kubernetes.io/name"] == "medical-qa-retrieval"
    assert ret_sm["spec"]["endpoints"][0]["path"] == "/metrics"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/deploy/test_helm_monitoring_chart.py -v`
Expected: FAIL (chart dir does not exist → `helm template` errors)

- [ ] **Step 3: Implement — `Chart.yaml`**

Create `deploy/helm/monitoring/Chart.yaml`:

```yaml
apiVersion: v2
name: medical-qa-monitoring
description: ServiceMonitors, alert rules, and Grafana dashboard for Medical QA serving.
type: application
version: 0.1.0
appVersion: "1.0"
```

- [ ] **Step 4: Implement — `values.yaml`**

Create `deploy/helm/monitoring/values.yaml`:

```yaml
# Label kube-prometheus-stack uses to select ServiceMonitors/PrometheusRules. The
# install_monitoring.sh release name is "monitoring", so the stack's default
# serviceMonitorSelector/ruleSelector matches release=monitoring.
releaseLabel: monitoring
scrapeInterval: 30s

alerts:
  errorRateThreshold: 0.05        # 5% of /predict requests erroring
  p95LatencySeconds: 60           # llama.cpp CPU is slow; high ceiling
  emptyRateThreshold: 0.5         # 50% of tool calls returning empty
```

- [ ] **Step 5: Implement — ServiceMonitor templates**

Create `deploy/helm/monitoring/templates/servicemonitor-api.yaml`:

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: medical-qa-api
  labels:
    release: {{ .Values.releaseLabel }}
spec:
  selector:
    matchLabels:
      app.kubernetes.io/name: medical-qa-api
  endpoints:
    - port: http
      path: /metrics
      interval: {{ .Values.scrapeInterval }}
```

Create `deploy/helm/monitoring/templates/servicemonitor-retrieval.yaml`:

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: medical-qa-retrieval
  labels:
    release: {{ .Values.releaseLabel }}
spec:
  selector:
    matchLabels:
      app.kubernetes.io/name: medical-qa-retrieval
  endpoints:
    - port: http
      path: /metrics
      interval: {{ .Values.scrapeInterval }}
```

- [ ] **Step 6: Run test to verify it passes**

Run: `.venv/bin/pytest tests/deploy/test_helm_monitoring_chart.py -v`
Expected: PASS (skips if project-local helm absent — run `make install-deploy-tools` first)

- [ ] **Step 7: Commit**

```bash
git add deploy/helm/monitoring/Chart.yaml deploy/helm/monitoring/values.yaml \
  deploy/helm/monitoring/templates/servicemonitor-api.yaml \
  deploy/helm/monitoring/templates/servicemonitor-retrieval.yaml \
  tests/deploy/test_helm_monitoring_chart.py
git commit -m "feat(helm): monitoring chart with api/retrieval ServiceMonitors"
```

---

## Task 12: PrometheusRule alerts

**Files:**
- Create: `deploy/helm/monitoring/templates/prometheusrule.yaml`
- Test: `tests/deploy/test_helm_monitoring_chart.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/deploy/test_helm_monitoring_chart.py`:

```python
def test_prometheusrule_defines_named_alerts():
    resources = render_chart("monitoring")
    rule = find_kind(resources, "PrometheusRule", "medical-qa-alerts")
    assert rule["metadata"]["labels"]["release"] == "monitoring"
    alert_names = {
        r["alert"]
        for group in rule["spec"]["groups"]
        for r in group["rules"]
    }
    assert alert_names == {
        "HighErrorRate",
        "HighLatencyP95",
        "RetrievalEmptyRateHigh",
        "TargetDown",
    }


def test_prometheusrule_thresholds_come_from_values():
    resources = render_chart("monitoring", set_values={"alerts.errorRateThreshold": "0.1"})
    rule = find_kind(resources, "PrometheusRule", "medical-qa-alerts")
    exprs = [
        r["expr"]
        for group in rule["spec"]["groups"]
        for r in group["rules"]
        if r["alert"] == "HighErrorRate"
    ]
    assert "0.1" in exprs[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/deploy/test_helm_monitoring_chart.py::test_prometheusrule_defines_named_alerts -v`
Expected: FAIL (`missing PrometheusRule/medical-qa-alerts`)

- [ ] **Step 3: Implement**

Create `deploy/helm/monitoring/templates/prometheusrule.yaml`:

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: medical-qa-alerts
  labels:
    release: {{ .Values.releaseLabel }}
spec:
  groups:
    - name: medical-qa-serving
      rules:
        - alert: HighErrorRate
          expr: >-
            sum(rate(mqa_requests_total{status="error"}[5m]))
            / clamp_min(sum(rate(mqa_requests_total[5m])), 1)
            > {{ .Values.alerts.errorRateThreshold }}
          for: 5m
          labels:
            severity: warning
          annotations:
            summary: "Medical QA /predict error rate is high"
        - alert: HighLatencyP95
          expr: >-
            histogram_quantile(0.95,
              sum(rate(mqa_request_latency_seconds_bucket{endpoint="/predict"}[5m])) by (le))
            > {{ .Values.alerts.p95LatencySeconds }}
          for: 10m
          labels:
            severity: warning
          annotations:
            summary: "Medical QA /predict p95 latency is high"
        - alert: RetrievalEmptyRateHigh
          expr: >-
            sum(rate(mqa_tool_outcome_total{outcome="empty"}[15m]))
            / clamp_min(sum(rate(mqa_tool_outcome_total[15m])), 1)
            > {{ .Values.alerts.emptyRateThreshold }}
          for: 15m
          labels:
            severity: info
          annotations:
            summary: "Tool calls frequently return empty evidence"
        - alert: TargetDown
          expr: up{job=~"medical-qa-.*"} == 0
          for: 5m
          labels:
            severity: critical
          annotations:
            summary: "A Medical QA scrape target is down"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/deploy/test_helm_monitoring_chart.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add deploy/helm/monitoring/templates/prometheusrule.yaml tests/deploy/test_helm_monitoring_chart.py
git commit -m "feat(helm): PrometheusRule with 4 serving alerts"
```

---

## Task 13: Grafana dashboard ConfigMap

**Files:**
- Create: `deploy/helm/monitoring/dashboards/medical-qa-serving.json`
- Create: `deploy/helm/monitoring/templates/dashboard-configmap.yaml`
- Test: `tests/deploy/test_helm_monitoring_chart.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/deploy/test_helm_monitoring_chart.py`:

```python
import json


def test_dashboard_configmap_carries_valid_json_and_grafana_label():
    resources = render_chart("monitoring")
    cm = find_kind(resources, "ConfigMap", "medical-qa-dashboard")
    assert cm["metadata"]["labels"]["grafana_dashboard"] == "1"
    raw = cm["data"]["medical-qa-serving.json"]
    dashboard = json.loads(raw)  # must be valid JSON
    titles = {p["title"] for p in dashboard["panels"]}
    assert "Request rate by status" in titles
    assert "Latency p95 (total/model)" in titles
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/deploy/test_helm_monitoring_chart.py::test_dashboard_configmap_carries_valid_json_and_grafana_label -v`
Expected: FAIL (`missing ConfigMap/medical-qa-dashboard`)

- [ ] **Step 3: Implement — dashboard JSON**

Create `deploy/helm/monitoring/dashboards/medical-qa-serving.json` (minimal, valid Grafana dashboard; Prometheus datasource by default UID `prometheus`):

```json
{
  "title": "Medical QA Serving",
  "uid": "medical-qa-serving",
  "schemaVersion": 39,
  "version": 1,
  "time": { "from": "now-6h", "to": "now" },
  "panels": [
    {
      "id": 1,
      "title": "Request rate by status",
      "type": "timeseries",
      "gridPos": { "h": 8, "w": 12, "x": 0, "y": 0 },
      "targets": [
        { "expr": "sum by (status) (rate(mqa_requests_total[5m]))", "legendFormat": "{{status}}" }
      ]
    },
    {
      "id": 2,
      "title": "Latency p95 (total/model)",
      "type": "timeseries",
      "gridPos": { "h": 8, "w": 12, "x": 12, "y": 0 },
      "targets": [
        { "expr": "histogram_quantile(0.95, sum(rate(mqa_request_latency_seconds_bucket{endpoint=\"/predict\"}[5m])) by (le))", "legendFormat": "total p95" },
        { "expr": "histogram_quantile(0.95, sum(rate(mqa_model_latency_seconds_bucket[5m])) by (le))", "legendFormat": "model p95" },
        { "expr": "histogram_quantile(0.95, sum(rate(mqa_retrieval_latency_seconds_bucket[5m])) by (le))", "legendFormat": "retrieval p95" }
      ]
    },
    {
      "id": 3,
      "title": "Tool outcome rate",
      "type": "timeseries",
      "gridPos": { "h": 8, "w": 12, "x": 0, "y": 8 },
      "targets": [
        { "expr": "sum by (outcome) (rate(mqa_tool_outcome_total[5m]))", "legendFormat": "{{outcome}}" }
      ]
    },
    {
      "id": 4,
      "title": "Build info",
      "type": "table",
      "gridPos": { "h": 8, "w": 12, "x": 12, "y": 8 },
      "targets": [
        { "expr": "mqa_build_info", "instant": true, "format": "table" }
      ]
    }
  ]
}
```

- [ ] **Step 4: Implement — ConfigMap template**

Create `deploy/helm/monitoring/templates/dashboard-configmap.yaml` (the `grafana_dashboard: "1"` label triggers the kube-prometheus-stack Grafana sidecar to auto-import):

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: medical-qa-dashboard
  labels:
    grafana_dashboard: "1"
data:
  medical-qa-serving.json: |-
{{ .Files.Get "dashboards/medical-qa-serving.json" | indent 4 }}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/deploy/test_helm_monitoring_chart.py -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add deploy/helm/monitoring/dashboards/medical-qa-serving.json \
  deploy/helm/monitoring/templates/dashboard-configmap.yaml \
  tests/deploy/test_helm_monitoring_chart.py
git commit -m "feat(helm): Grafana dashboard ConfigMap for serving telemetry"
```

---

## Task 14: Gate the monitoring chart in `deploy.sh`

**Files:**
- Modify: `scripts/cloud/deploy.sh`
- Test: `tests/cloud/test_deploy_sh.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/cloud/test_deploy_sh.py`. The module already defines `ROOT`, `SCRIPT`, and `ENV` at the top and the existing tests run `subprocess.run(["bash", str(SCRIPT), "--dry-run"], ..., env=ENV)` — reuse those exactly:

```python
def test_dry_run_installs_monitoring_chart():
    out = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"], capture_output=True, text=True, env=ENV
    )
    assert out.returncode == 0, out.stderr
    assert "upgrade --install medical-qa-monitoring deploy/helm/monitoring" in out.stdout


def test_monitoring_is_gated_on_servicemonitor_crd():
    # Same shape as the KServe guard: ServiceMonitor/PrometheusRule kinds only exist after
    # kube-prometheus-stack is installed, so the live install must be CRD-gated (non-fatal skip).
    assert "get crd servicemonitors.monitoring.coreos.com" in SCRIPT.read_text()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/cloud/test_deploy_sh.py -k monitoring -v`
Expected: FAIL (no monitoring install line)

- [ ] **Step 3: Implement**

In `scripts/cloud/deploy.sh`, after the existing KServe gated block (after its closing `fi` and before the final `echo "Deployed all charts...`), add a parallel block:

```bash
# Monitoring is optional, same shape as KServe: the ServiceMonitor/PrometheusRule
# kinds only exist once kube-prometheus-stack is installed (run install_monitoring.sh).
# On a cluster without it, skip non-fatally so the core demo still deploys.
MONITORING_INSTALL=("$HELM" upgrade --install medical-qa-monitoring deploy/helm/monitoring \
  --namespace "$K8S_NAMESPACE")
if [ "$DRY_RUN" -eq 1 ]; then
  echo "+ ${MONITORING_INSTALL[*]}"
elif "$KUBECTL" get crd servicemonitors.monitoring.coreos.com >/dev/null 2>&1; then
  "${MONITORING_INSTALL[@]}"
else
  echo "ServiceMonitor CRD (servicemonitors.monitoring.coreos.com) not found; skipping monitoring chart (run install_monitoring.sh first)."
fi
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/cloud/test_deploy_sh.py -k monitoring -v`
Expected: PASS

- [ ] **Step 5: Run the full deploy_sh suite (no regressions)**

Run: `.venv/bin/pytest tests/cloud/test_deploy_sh.py -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/cloud/deploy.sh tests/cloud/test_deploy_sh.py
git commit -m "feat(cloud): deploy.sh installs monitoring chart when ServiceMonitor CRD present"
```

---

## Task 15: Makefile helm targets + runbook

**Files:**
- Modify: `Makefile`
- Create: `docs/runbooks/monitoring.md`

- [ ] **Step 1: Add the monitoring chart to `helm-lint` and `helm-template`**

In `Makefile`, in the `helm-lint:` recipe add after the `ui` line:

```make
	$(HELM) lint deploy/helm/monitoring
```

In the `helm-template:` recipe add after the `ui` line:

```make
	$(HELM) template medical-qa-monitoring deploy/helm/monitoring >/dev/null
```

- [ ] **Step 2: Verify the charts lint and template**

Run: `make helm-lint && make helm-template`
Expected: all six charts lint clean and render (requires `make install-deploy-tools`).

- [ ] **Step 3: Write the runbook**

Create `docs/runbooks/monitoring.md`:

```markdown
# Runbook: Monitoring (Prometheus + Grafana + Alertmanager)

Self-hosted `kube-prometheus-stack` scrapes the api/retrieval `/metrics`, renders the
"Medical QA Serving" Grafana dashboard, and evaluates the serving alerts.

## Install (once per cluster)

```bash
cd /home/vcsai/minhlbq/mlops-platform
bash scripts/cloud/install_monitoring.sh          # pinned KUBE_PROM_STACK_VERSION
# then (re)deploy so the monitoring chart's ServiceMonitors/rules/dashboard land:
IMAGE_TAG=<sha> bash scripts/cloud/deploy.sh
```

`deploy.sh` installs `deploy/helm/monitoring` only when the
`servicemonitors.monitoring.coreos.com` CRD exists — so run `install_monitoring.sh` first.

## View Grafana

```bash
.tools/bin/kubectl -n monitoring port-forward svc/monitoring-grafana 3000:80
# open http://localhost:3000 — default user admin
.tools/bin/kubectl -n monitoring get secret monitoring-grafana \
  -o jsonpath='{.data.admin-password}' | base64 -d; echo
```

The "Medical QA Serving" dashboard is auto-imported by the Grafana dashboard sidecar
(ConfigMap label `grafana_dashboard: "1"`).

## Verify scrape targets + alerts

```bash
.tools/bin/kubectl -n monitoring port-forward svc/monitoring-kube-prometheus-prometheus 9090:9090
# http://localhost:9090/targets  → medical-qa-api / medical-qa-retrieval should be "up"
# http://localhost:9090/alerts   → HighErrorRate, HighLatencyP95, RetrievalEmptyRateHigh, TargetDown
```

## Notes

- Best validated on the GKE Standard in-cluster-LLM cluster (`medical-qa-llm`), where
  llama.cpp latency makes `mqa_model_latency_seconds` meaningful.
- Alert thresholds live in `deploy/helm/monitoring/values.yaml` (`alerts.*`).
- Alertmanager uses the stack default (no external receiver) — alerts are visible in the
  Alertmanager UI and Grafana, not routed to Slack/email.
- Pinned chart version is `KUBE_PROM_STACK_VERSION` in `scripts/cloud/config.sh`; confirm
  the version is reachable from `prometheus-community` before installing.
```

- [ ] **Step 4: Commit**

```bash
git add Makefile docs/runbooks/monitoring.md
git commit -m "docs(monitoring): add monitoring chart to helm targets + runbook"
```

---

## Task 16: Phase 2 gate — lint, template, full suite

**Files:** none (verification only)

- [ ] **Step 1: Lint + template all charts**

Run: `make helm-lint && make helm-template`
Expected: six charts (api, retrieval, nginx, kserve, ui, monitoring) lint and render with no errors.

- [ ] **Step 2: Full test suite with coverage gate**

Run: `make test`
Expected: all pass, coverage ≥ 80%. New tests for `install_monitoring.sh`, the monitoring chart, and the deploy.sh gate are green.

- [ ] **Step 3: Confirm install_monitoring.sh parses + dry-runs cleanly**

Run: `GCP_PROJECT=demo bash scripts/cloud/install_monitoring.sh --dry-run`
Expected: echoes repo-add then the pinned `upgrade --install monitoring ...` line; exit 0.

- [ ] **Step 4: Confirm clean tree**

Run: `git status -s`
Expected: clean (all work committed across the per-task commits).

---

## Notes for the implementer

- **Run everything from** `/home/vcsai/minhlbq/mlops-platform`.
- **Helm tests skip** if `.tools/bin/helm` is absent — run `make install-deploy-tools` first to actually exercise Tasks 11–13/16.
- **No `baseline` tokens** may enter `src/` — the guard test (`tests/retrieval/test_kg_backend_self_contained.py`) enforces this; everything here is generic observability code, so it stays clean.
- **`prometheus_client` uses a global registry** — metrics defined at import time persist across tests; the tests assert substring presence in `/metrics`, which is registry-state-tolerant.
- **Live validation is manual** (no Prometheus in CI): follow `docs/runbooks/monitoring.md` on the `medical-qa-llm` cluster after merge.
```
