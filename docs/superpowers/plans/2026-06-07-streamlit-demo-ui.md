# Streamlit Demo UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Streamlit demo frontend for `POST /predict`, deployed to GKE on its own LoadBalancer, calling the existing nginx gateway in-cluster with the `x-api-key` (server-side).

**Architecture:** A new `medical-qa-ui` service (Streamlit) with its own `LoadBalancer`. The UI pod posts to `http://medical-qa-nginx:8080/predict` with the gateway key read from the existing `medical-qa-nginx-api-key` Secret — the key stays server-side and never reaches the browser. The API/nginx/retrieval path is unchanged. All request logic lives in a pure, unit-tested `app/client.py` (no Streamlit import); `app/streamlit_app.py` is rendering only.

**Tech Stack:** Python 3.12, Streamlit, httpx (core dep), Helm, Docker, GKE, pytest, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-06-07-streamlit-demo-ui-design.md`

**Repo conventions discovered (follow these exactly):**
- Tests run with `.venv/bin/pytest`. Coverage gate `--cov-fail-under=80` measures only `medical_qa_platform`; `app/` is outside it, so it neither raises nor lowers coverage.
- Helm charts are unit-tested by rendering with the project-local `.tools/bin/helm` via `tests/deploy/helm_helpers.py` (`render_chart`, `find_kind`). Helm IS installed locally, so these tests run (don't skip).
- `pytest.ini` sets `pythonpath = ["."]`, so `import app.client` and `from tests.deploy.helm_helpers import ...` work via namespace packages. Test subdirs mostly have NO `__init__.py`; do not add one under `tests/app/`.
- `httpx` is 0.28.1 in `.venv` (already core); the `[demo]` extra is only needed to *run* Streamlit, not for the unit tests.

---

## Task 1: Pure HTTP client (`app/client.py`)

**Files:**
- Create: `app/__init__.py`
- Create: `app/client.py`
- Test: `tests/app/test_client.py`

- [ ] **Step 1: Write the failing test**

Create `tests/app/test_client.py`:

```python
import httpx
import pytest

from app.client import (
    PredictError,
    PredictResult,
    build_payload,
    fetch_version,
    predict,
)


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_build_payload_strips_and_keeps_letters():
    payload = build_payload("  first-line? ", {"A": "Metformin ", "B": " Insulin"})
    assert payload == {
        "question": "first-line?",
        "options": {"A": "Metformin", "B": "Insulin"},
    }


def test_build_payload_rejects_blank_question():
    with pytest.raises(ValueError):
        build_payload("   ", {"A": "x", "B": "y"})


def test_build_payload_drops_empty_options_then_requires_two():
    with pytest.raises(ValueError):
        build_payload("q", {"A": "only", "B": "   "})


def test_build_payload_rejects_more_than_ten_options():
    opts = {chr(ord("A") + i): str(i) for i in range(11)}
    with pytest.raises(ValueError):
        build_payload("q", opts)


def test_build_payload_rejects_non_letter_key():
    with pytest.raises(ValueError):
        build_payload("q", {"A": "x", "1": "y"})


def test_predict_parses_success_and_sends_key():
    def handler(request):
        assert request.url.path == "/predict"
        assert request.headers["x-api-key"] == "k"
        return httpx.Response(
            200,
            json={
                "answer": "A",
                "evidence": ["e1", "e2"],
                "backend": "vllm",
                "model_version": "smoke-dev",
                "contract_version": "v1",
                "latency_ms": 12.5,
                "trace_id": "abc",
            },
        )

    res = predict(
        "http://gw:8080/",
        "k",
        {"question": "q", "options": {"A": "x", "B": "y"}},
        client=_client(handler),
    )
    assert isinstance(res, PredictResult)
    assert res.answer == "A"
    assert res.evidence == ["e1", "e2"]
    assert res.backend == "vllm"
    assert res.trace_id == "abc"


def test_predict_raises_predicterror_on_401():
    def handler(request):
        return httpx.Response(401)

    with pytest.raises(PredictError, match="unauthorized"):
        predict(
            "http://gw:8080",
            "bad",
            {"question": "q", "options": {"A": "x", "B": "y"}},
            client=_client(handler),
        )


def test_predict_raises_predicterror_on_timeout():
    def handler(request):
        raise httpx.TimeoutException("slow")

    with pytest.raises(PredictError, match="timed out"):
        predict(
            "http://gw:8080",
            "k",
            {"question": "q", "options": {"A": "x", "B": "y"}},
            timeout=3,
            client=_client(handler),
        )


def test_predict_raises_predicterror_on_connection_error():
    def handler(request):
        raise httpx.ConnectError("nope")

    with pytest.raises(PredictError, match="could not reach"):
        predict(
            "http://gw:8080",
            "k",
            {"question": "q", "options": {"A": "x", "B": "y"}},
            client=_client(handler),
        )


def test_fetch_version_returns_json():
    def handler(request):
        assert request.url.path == "/version"
        return httpx.Response(
            200,
            json={"backend": "vllm", "model_version": "smoke-dev", "contract_version": "v1"},
        )

    out = fetch_version("http://gw:8080", "k", client=_client(handler))
    assert out["backend"] == "vllm"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/app/test_client.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'app'` (or `app.client`).

- [ ] **Step 3: Write minimal implementation**

Create `app/__init__.py` (empty file):

```python
```

Create `app/client.py`:

```python
"""Pure HTTP client for the Medical QA /predict API.

No Streamlit import — importable and unit-testable in the base venv. The UI
(``app.streamlit_app``) is the only Streamlit-aware module.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx


class PredictError(Exception):
    """A human-readable failure talking to the API (surfaced in the UI)."""


@dataclass
class PredictResult:
    answer: str | None
    evidence: list[str]
    backend: str
    model_version: str
    contract_version: str
    latency_ms: float
    trace_id: str


def build_payload(question: str, options: dict[str, str]) -> dict:
    """Validate UI inputs into the /predict request body (mirrors the server contract)."""
    question = question.strip()
    if not question:
        raise ValueError("Câu hỏi không được để trống.")
    clean = {key: value.strip() for key, value in options.items() if value.strip()}
    if not (2 <= len(clean) <= 10):
        raise ValueError("Cần 2–10 phương án không rỗng.")
    for key in clean:
        if len(key) != 1 or not ("A" <= key <= "Z"):
            raise ValueError(f"Khóa phương án {key!r} phải là một chữ cái A–Z.")
    return {"question": question, "options": clean}


def _headers(api_key: str | None) -> dict[str, str]:
    return {"x-api-key": api_key} if api_key else {}


def _request(
    method: str,
    base_url: str,
    path: str,
    api_key: str | None,
    timeout: float,
    client: httpx.Client | None,
    **kwargs,
) -> httpx.Response:
    url = base_url.rstrip("/") + path
    owned = client is None
    conn = client or httpx.Client(timeout=timeout)
    try:
        resp = conn.request(method, url, headers=_headers(api_key), **kwargs)
    except httpx.TimeoutException as exc:
        raise PredictError(f"Hết thời gian chờ sau {timeout:g}s.") from exc
    except httpx.HTTPError as exc:
        raise PredictError("Không thể kết nối tới API.") from exc
    finally:
        if owned:
            conn.close()
    if resp.status_code == 401:
        raise PredictError("Bị từ chối (401) — kiểm tra API key của gateway.")
    if resp.status_code != 200:
        raise PredictError(f"API trả về HTTP {resp.status_code}.")
    return resp


def predict(
    base_url: str,
    api_key: str | None,
    payload: dict,
    timeout: float = 30.0,
    client: httpx.Client | None = None,
) -> PredictResult:
    resp = _request("POST", base_url, "/predict", api_key, timeout, client, json=payload)
    data = resp.json()
    return PredictResult(
        answer=data.get("answer"),
        evidence=data.get("evidence", []),
        backend=data.get("backend", ""),
        model_version=data.get("model_version", ""),
        contract_version=data.get("contract_version", ""),
        latency_ms=data.get("latency_ms", 0.0),
        trace_id=data.get("trace_id", ""),
    )


def fetch_version(
    base_url: str,
    api_key: str | None,
    timeout: float = 5.0,
    client: httpx.Client | None = None,
) -> dict:
    resp = _request("GET", base_url, "/version", api_key, timeout, client)
    return resp.json()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/app/test_client.py -q`
Expected: PASS (10 passed).

- [ ] **Step 5: Commit**

```bash
git add app/__init__.py app/client.py tests/app/test_client.py
git commit -m "feat(ui): pure httpx client for /predict + /version with typed errors"
```

---

## Task 2: Streamlit app + `[demo]` extra + `make demo-ui`

**Files:**
- Create: `app/streamlit_app.py`
- Modify: `pyproject.toml` (add `demo` optional extra)
- Modify: `Makefile` (add `demo-ui` target)
- Test: `tests/app/test_demo_packaging.py`

- [ ] **Step 1: Write the failing test**

Create `tests/app/test_demo_packaging.py`:

```python
import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_demo_extra_declares_streamlit_and_httpx():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    demo = "\n".join(data["project"]["optional-dependencies"]["demo"])
    assert "streamlit" in demo
    assert "httpx" in demo


def test_makefile_exposes_demo_ui_target():
    text = (ROOT / "Makefile").read_text()
    assert "demo-ui:" in text
    assert "streamlit run app/streamlit_app.py" in text


def test_streamlit_app_is_render_only_and_imports_client():
    # Logic lives in app.client; the Streamlit module must not redefine it.
    text = (ROOT / "app/streamlit_app.py").read_text()
    assert "from app.client import" in text
    assert "import streamlit" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/app/test_demo_packaging.py -q`
Expected: FAIL — `KeyError: 'demo'` (extra not declared) / missing `Makefile` target / missing file.

- [ ] **Step 3a: Add the `demo` extra to `pyproject.toml`**

In `pyproject.toml`, under `[project.optional-dependencies]`, after the `dev = [...]` block, add:

```toml
demo = [
    "streamlit>=1.40",
    "httpx>=0.27",
]
```

- [ ] **Step 3b: Add the `demo-ui` Makefile target**

Append to `Makefile` (and add `demo-ui` to the `.PHONY` line):

```make
demo-ui:
	.venv/bin/python -m pip install -e '.[demo]'
	.venv/bin/streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

- [ ] **Step 3c: Create `app/streamlit_app.py`**

```python
"""Streamlit demo UI for the Medical QA platform.

Server-side only: the gateway API key is read from the pod environment and
attached to requests by ``app.client``; it never reaches the browser. All
request/response logic lives in ``app.client`` — this module is rendering only.
"""

from __future__ import annotations

import os

import streamlit as st

from app.client import PredictError, build_payload, fetch_version, predict

DEFAULT_BASE_URL = os.environ.get("API_BASE_URL", "http://medical-qa-nginx:8080")
ENV_API_KEY = os.environ.get("API_KEY", "")

PRESETS: dict[str, dict] = {
    "Đái tháo đường type 2 — first-line": {
        "question": "Which medication is first-line for type 2 diabetes mellitus?",
        "options": {"A": "Metformin", "B": "Amoxicillin", "C": "Atorvastatin", "D": "Furosemide"},
    },
    "Nhồi máu cơ tim — marker": {
        "question": "Which serum marker is most specific for acute myocardial infarction?",
        "options": {"A": "Troponin I", "B": "Amylase", "C": "ALT", "D": "Creatinine"},
    },
}


def _sidebar() -> tuple[str, str]:
    st.sidebar.header("Kết nối")
    base_url = st.sidebar.text_input("API endpoint", value=DEFAULT_BASE_URL)
    key_override = st.sidebar.text_input(
        "API key (override)",
        value="",
        type="password",
        help="Để trống sẽ dùng key tiêm sẵn trong pod (env API_KEY).",
    )
    api_key = key_override or ENV_API_KEY
    if st.sidebar.button("Kiểm tra kết nối"):
        try:
            info = fetch_version(base_url, api_key)
            st.sidebar.success(
                f"backend={info.get('backend')} · model={info.get('model_version')} · "
                f"contract={info.get('contract_version')}"
            )
        except PredictError as exc:
            st.sidebar.error(str(exc))
    return base_url, api_key


def main() -> None:
    st.set_page_config(page_title="Medical QA Demo", page_icon="🩺")
    st.title("🩺 Medical QA — Demo")
    st.caption("Nhập câu hỏi trắc nghiệm y khoa; hệ thống truy hồi tri thức (KG) rồi trả đáp án.")

    base_url, api_key = _sidebar()

    preset = st.selectbox("Câu hỏi mẫu", ["(tự nhập)", *PRESETS])
    seed = PRESETS.get(
        preset, {"question": "", "options": {"A": "", "B": "", "C": "", "D": ""}}
    )

    question = st.text_area("Câu hỏi", value=seed["question"], height=100)
    st.write("Phương án:")
    options: dict[str, str] = {}
    cols = st.columns(2)
    for index, letter in enumerate("ABCD"):
        with cols[index % 2]:
            value = st.text_input(
                letter, value=seed["options"].get(letter, ""), key=f"opt_{letter}"
            )
        if value.strip():
            options[letter] = value.strip()

    if not st.button("Chẩn đoán", type="primary"):
        return

    try:
        payload = build_payload(question, options)
    except ValueError as exc:
        st.warning(str(exc))
        return

    with st.spinner("Đang truy hồi tri thức và suy luận..."):
        try:
            result = predict(base_url, api_key, payload)
        except PredictError as exc:
            st.error(str(exc))
            return

    if result.answer is None:
        st.warning("Mô hình không trả về đáp án hợp lệ (no answer parsed).")
    else:
        st.success(f"Đáp án: **{result.answer}** — {options.get(result.answer, '')}")

    for letter, text in options.items():
        mark = "✅" if letter == result.answer else "▫️"
        st.write(f"{mark} **{letter}.** {text}")

    with st.expander(f"Bằng chứng KG ({len(result.evidence)})", expanded=True):
        if result.evidence:
            for i, evidence in enumerate(result.evidence, 1):
                st.markdown(f"{i}. {evidence}")
        else:
            st.write("(không có bằng chứng nào được truy hồi)")

    st.caption(
        f"backend={result.backend} · model={result.model_version} · "
        f"contract={result.contract_version} · latency={result.latency_ms:.0f}ms · "
        f"trace={result.trace_id}"
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4a: Verify the Streamlit module compiles (Streamlit not installed in base venv)**

Run: `.venv/bin/python -m py_compile app/streamlit_app.py && echo OK`
Expected: `OK` (no syntax errors). Importing it would require the `[demo]` extra; compilation is enough here.

- [ ] **Step 4b: Run the packaging tests**

Run: `.venv/bin/pytest tests/app/test_demo_packaging.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add app/streamlit_app.py pyproject.toml Makefile tests/app/test_demo_packaging.py
git commit -m "feat(ui): Streamlit demo app, [demo] extra, make demo-ui target"
```

---

## Task 3: UI Dockerfile (`docker/ui.Dockerfile`)

**Files:**
- Create: `docker/ui.Dockerfile`
- Test: `tests/deploy/test_dockerfiles.py` (add one test)

- [ ] **Step 1: Write the failing test**

Append to `tests/deploy/test_dockerfiles.py`:

```python
def test_ui_dockerfile_runs_streamlit_on_8501_non_root():
    text = _read("ui.Dockerfile")
    assert "FROM python:3.12-slim" in text
    assert "uv pip install --system --no-cache '.[demo]'" in text
    assert "COPY app ./app" in text
    assert "USER app" in text
    assert "EXPOSE 8501" in text
    assert '"streamlit", "run", "app/streamlit_app.py"' in text
    assert '"--server.port", "8501"' in text
    assert '"--server.address", "0.0.0.0"' in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/deploy/test_dockerfiles.py::test_ui_dockerfile_runs_streamlit_on_8501_non_root -q`
Expected: FAIL — `FileNotFoundError: .../docker/ui.Dockerfile`.

- [ ] **Step 3: Create `docker/ui.Dockerfile`**

```dockerfile
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN addgroup --system app && adduser --system --ingroup app app

COPY pyproject.toml README.md ./
COPY src ./src
COPY app ./app

RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache '.[demo]'

USER app
EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port", "8501", "--server.address", "0.0.0.0"]
```

(`COPY src` is required because installing `.[demo]` builds the `medical_qa_platform` package, whose source lives under `src/`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/deploy/test_dockerfiles.py -q`
Expected: PASS (all dockerfile tests, including the new one).

- [ ] **Step 5: Commit**

```bash
git add docker/ui.Dockerfile tests/deploy/test_dockerfiles.py
git commit -m "feat(ui): Dockerfile running Streamlit on 8501 (non-root)"
```

---

## Task 4: Helm chart (`deploy/helm/ui/`)

**Files:**
- Create: `deploy/helm/ui/Chart.yaml`
- Create: `deploy/helm/ui/values.yaml`
- Create: `deploy/helm/ui/values-prod.yaml`
- Create: `deploy/helm/ui/templates/configmap.yaml`
- Create: `deploy/helm/ui/templates/deployment.yaml`
- Create: `deploy/helm/ui/templates/service.yaml`
- Test: `tests/deploy/test_helm_ui_chart.py`

- [ ] **Step 1: Write the failing test**

Create `tests/deploy/test_helm_ui_chart.py`:

```python
import pytest
import yaml

from tests.deploy.helm_helpers import find_kind, render_chart


def test_ui_base_chart_renders_configmap_deployment_service_clusterip():
    resources = render_chart("ui")
    config = find_kind(resources, "ConfigMap", "medical-qa-ui-config")
    deployment = find_kind(resources, "Deployment", "medical-qa-ui")
    service = find_kind(resources, "Service", "medical-qa-ui")

    assert config["data"]["API_BASE_URL"] == "http://medical-qa-nginx:8080"
    assert service["spec"]["type"] == "ClusterIP"
    assert service["spec"]["ports"][0]["port"] == 8501

    container = deployment["spec"]["template"]["spec"]["containers"][0]
    assert container["ports"][0]["containerPort"] == 8501
    assert container["livenessProbe"]["httpGet"]["path"] == "/_stcore/health"
    assert container["readinessProbe"]["httpGet"]["path"] == "/_stcore/health"

    env_by_name = {e["name"]: e for e in container.get("env", [])}
    ref = env_by_name["API_KEY"]["valueFrom"]["secretKeyRef"]
    assert ref["name"] == "medical-qa-nginx-api-key"
    assert ref["key"] == "API_KEY"


def test_ui_prod_overlay_switches_service_to_loadbalancer():
    resources = render_chart("ui", values_files=["values-prod.yaml"])
    service = find_kind(resources, "Service", "medical-qa-ui")
    assert service["spec"]["type"] == "LoadBalancer"
    assert service["spec"]["ports"][0]["port"] == 8501


def test_ui_values_point_at_nginx_gateway_in_cluster():
    from pathlib import Path

    root = Path(__file__).parents[2]
    values = yaml.safe_load((root / "deploy/helm/ui/values.yaml").read_text())
    assert values["env"]["apiBaseUrl"] == "http://medical-qa-nginx:8080"
    assert values["auth"]["existingSecret"] == "medical-qa-nginx-api-key"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/deploy/test_helm_ui_chart.py -q`
Expected: FAIL — `helm template failed` / chart dir missing.

- [ ] **Step 3a: Create `deploy/helm/ui/Chart.yaml`**

```yaml
apiVersion: v2
name: medical-qa-ui
description: Streamlit demo UI for the Medical QA platform
type: application
version: 0.1.0
appVersion: "0.1.0"
```

- [ ] **Step 3b: Create `deploy/helm/ui/values.yaml`**

```yaml
replicaCount: 1

image:
  repository: ghcr.io/hinmq21/medical-qa-ui
  tag: latest
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 8501

env:
  # In-cluster ClusterIP of the nginx gateway; the UI calls /predict through it.
  apiBaseUrl: http://medical-qa-nginx:8080

auth:
  # The UI reuses the gateway's own API-key Secret (key: API_KEY). The key is
  # injected into the pod and attached server-side — it never reaches the browser.
  existingSecret: medical-qa-nginx-api-key

resources:
  requests:
    cpu: 100m
    memory: 256Mi
  limits:
    cpu: 500m
    memory: 512Mi
```

- [ ] **Step 3c: Create `deploy/helm/ui/values-prod.yaml`**

```yaml
service:
  type: LoadBalancer
```

- [ ] **Step 3d: Create `deploy/helm/ui/templates/configmap.yaml`**

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: medical-qa-ui-config
data:
  API_BASE_URL: {{ .Values.env.apiBaseUrl | quote }}
```

- [ ] **Step 3e: Create `deploy/helm/ui/templates/deployment.yaml`**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: medical-qa-ui
  labels:
    app.kubernetes.io/name: medical-qa-ui
spec:
  replicas: {{ .Values.replicaCount }}
  selector:
    matchLabels:
      app.kubernetes.io/name: medical-qa-ui
  template:
    metadata:
      labels:
        app.kubernetes.io/name: medical-qa-ui
    spec:
      containers:
        - name: ui
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          envFrom:
            - configMapRef:
                name: medical-qa-ui-config
          env:
            # Gateway API key, reused from the nginx Secret; attached server-side.
            - name: API_KEY
              valueFrom:
                secretKeyRef:
                  name: {{ .Values.auth.existingSecret }}
                  key: API_KEY
          ports:
            - name: http
              containerPort: 8501
          livenessProbe:
            httpGet:
              path: /_stcore/health
              port: http
            initialDelaySeconds: 10
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /_stcore/health
              port: http
            initialDelaySeconds: 5
            periodSeconds: 10
          resources:
{{ toYaml .Values.resources | indent 12 }}
```

- [ ] **Step 3f: Create `deploy/helm/ui/templates/service.yaml`**

```yaml
apiVersion: v1
kind: Service
metadata:
  name: medical-qa-ui
spec:
  type: {{ .Values.service.type }}
  selector:
    app.kubernetes.io/name: medical-qa-ui
  ports:
    - name: http
      port: {{ .Values.service.port }}
      targetPort: http
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/deploy/test_helm_ui_chart.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add deploy/helm/ui tests/deploy/test_helm_ui_chart.py
git commit -m "feat(ui): helm chart (ClusterIP base, LoadBalancer prod overlay)"
```

---

## Task 5: Wire the chart into Makefile helm-lint / helm-template / docker-build

**Files:**
- Modify: `Makefile` (`helm-lint`, `helm-template`, `docker-build`)
- Modify: `tests/deploy/test_makefile_deploy_tools.py` (assert `deploy/helm/ui`)

- [ ] **Step 1: Write the failing test**

In `tests/deploy/test_makefile_deploy_tools.py`, inside `test_makefile_exposes_deploy_targets_and_preserves_pipeline_targets`, after the line `assert "deploy/helm/kserve" in text`, add:

```python
    assert "deploy/helm/ui" in text
    assert "docker/ui.Dockerfile" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/deploy/test_makefile_deploy_tools.py::test_makefile_exposes_deploy_targets_and_preserves_pipeline_targets -q`
Expected: FAIL — `assert "deploy/helm/ui" in text`.

- [ ] **Step 3a: Add `ui` to `helm-lint` and `helm-template` in `Makefile`**

In the `helm-lint:` recipe, add after the kserve line:

```make
	$(HELM) lint deploy/helm/ui
```

In the `helm-template:` recipe, add after the kserve line:

```make
	$(HELM) template medical-qa-ui deploy/helm/ui >/dev/null
```

- [ ] **Step 3b: Add the UI image to `docker-build` in `Makefile`**

In the `docker-build:` recipe, add after the pipeline-init line:

```make
	docker buildx build --platform linux/amd64 -f docker/ui.Dockerfile -t medical-qa-ui:local --load .
```

- [ ] **Step 4a: Run the Makefile test**

Run: `.venv/bin/pytest tests/deploy/test_makefile_deploy_tools.py -q`
Expected: PASS.

- [ ] **Step 4b: Verify helm lint/template accept the new chart**

Run: `.tools/bin/helm lint deploy/helm/ui && .tools/bin/helm template medical-qa-ui deploy/helm/ui >/dev/null && echo OK`
Expected: `1 chart(s) linted, 0 chart(s) failed` then `OK`.

- [ ] **Step 5: Commit**

```bash
git add Makefile tests/deploy/test_makefile_deploy_tools.py
git commit -m "build(ui): lint/template/build the ui chart and image via Makefile"
```

---

## Task 6: Install the UI chart in `deploy.sh`

**Files:**
- Modify: `scripts/cloud/deploy.sh`
- Modify: `tests/cloud/test_deploy_sh.py` (add assertions)

- [ ] **Step 1: Write the failing test**

In `tests/cloud/test_deploy_sh.py`, inside `test_dry_run_upgrades_four_charts_mock_backend`, after `assert "upgrade --install medical-qa-kserve deploy/helm/kserve" in o`, add:

```python
    assert "upgrade --install medical-qa-ui deploy/helm/ui" in o
    assert "deploy/helm/ui/values-prod.yaml" in o
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/cloud/test_deploy_sh.py::test_dry_run_upgrades_four_charts_mock_backend -q`
Expected: FAIL — `assert "upgrade --install medical-qa-ui deploy/helm/ui" in o`.

- [ ] **Step 3: Add the UI chart install to `scripts/cloud/deploy.sh`**

After the nginx install block:

```bash
run "$HELM" upgrade --install medical-qa-nginx deploy/helm/nginx \
  --namespace "$K8S_NAMESPACE" \
  -f deploy/helm/nginx/values-prod.yaml
```

insert:

```bash
# Streamlit demo UI on its own LoadBalancer; reaches the api through the nginx
# gateway in-cluster, reusing the gateway API-key Secret (medical-qa-nginx-api-key).
run "$HELM" upgrade --install medical-qa-ui deploy/helm/ui \
  --namespace "$K8S_NAMESPACE" \
  --set image.tag="$IMAGE_TAG" \
  -f deploy/helm/ui/values-prod.yaml
```

- [ ] **Step 4a: Run the deploy.sh test**

Run: `.venv/bin/pytest tests/cloud/test_deploy_sh.py -q`
Expected: PASS (all deploy.sh tests).

- [ ] **Step 4b: Verify the script still parses**

Run: `bash -n scripts/cloud/deploy.sh && echo OK`
Expected: `OK`.

- [ ] **Step 5: Commit**

```bash
git add scripts/cloud/deploy.sh tests/cloud/test_deploy_sh.py
git commit -m "feat(ui): deploy.sh installs the ui chart with the LoadBalancer overlay"
```

---

## Task 7: Gate deploy on the UI rollout in `smoke_cloud.sh`

**Files:**
- Modify: `scripts/cloud/smoke_cloud.sh` (add `medical-qa-ui` to the rollout-wait loop)
- Modify: `tests/cloud/test_smoke_cloud.py` (assert the UI rollout wait)

- [ ] **Step 1: Write the failing test**

In `tests/cloud/test_smoke_cloud.py`, inside `test_dry_run_waits_for_rollout_before_probing`, after `assert "rollout status deploy/medical-qa-retrieval" in o`, add:

```python
    assert "rollout status deploy/medical-qa-ui" in o
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/cloud/test_smoke_cloud.py::test_dry_run_waits_for_rollout_before_probing -q`
Expected: FAIL — `assert "rollout status deploy/medical-qa-ui" in o`.

- [ ] **Step 3: Add `medical-qa-ui` to the rollout-wait loop in `scripts/cloud/smoke_cloud.sh`**

Change:

```bash
for dep in medical-qa-retrieval medical-qa-api medical-qa-nginx; do
```

to:

```bash
for dep in medical-qa-retrieval medical-qa-api medical-qa-nginx medical-qa-ui; do
```

- [ ] **Step 4a: Run the smoke test suite**

Run: `.venv/bin/pytest tests/cloud/test_smoke_cloud.py -q`
Expected: PASS.

- [ ] **Step 4b: Verify the script still parses**

Run: `bash -n scripts/cloud/smoke_cloud.sh && echo OK`
Expected: `OK`.

- [ ] **Step 5: Commit**

```bash
git add scripts/cloud/smoke_cloud.sh tests/cloud/test_smoke_cloud.py
git commit -m "test(ui): smoke gates on the medical-qa-ui rollout"
```

---

## Task 8: Build the UI image in CI + print its IP in demo-up

**Files:**
- Modify: `.github/workflows/ci.yml` (add `ui` to the build matrix)
- Modify: `tests/deploy/test_ci_workflows.py` (add `docker/ui.Dockerfile` to the expected set — MANDATORY, the test pins exact equality)
- Modify: `.github/workflows/demo-up.yml` (print the UI LoadBalancer IP)

- [ ] **Step 1: Write the failing test**

In `tests/deploy/test_ci_workflows.py`, in `test_ci_runs_tests_then_builds_all_images_in_parallel`, change the expected `dockerfiles` set to include the UI image:

```python
    assert dockerfiles == {
        "docker/api.Dockerfile",
        "docker/retrieval.Dockerfile",
        "docker/kserve-mock.Dockerfile",
        "docker/pipeline-init.Dockerfile",
        "docker/ui.Dockerfile",
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/deploy/test_ci_workflows.py::test_ci_runs_tests_then_builds_all_images_in_parallel -q`
Expected: FAIL — set inequality (`docker/ui.Dockerfile` missing from the workflow).

- [ ] **Step 3a: Add the `ui` matrix entry to `.github/workflows/ci.yml`**

In the `build` job's `strategy.matrix.include`, after the `pipeline-init` entry, add:

```yaml
          - { image: ui, dockerfile: docker/ui.Dockerfile }
```

- [ ] **Step 3b: Print the UI LoadBalancer IP in `.github/workflows/demo-up.yml`**

Find the step that prints `Demo is up at http://$IP:8080` (the nginx IP). Immediately after the line that echoes that URL, add a UI-IP lookup and echo. The full block becomes:

```bash
          IP=$(.tools/bin/kubectl get svc medical-qa-nginx -n medical-qa \
            -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
          echo "Demo is up at http://$IP:8080"
          UI_IP=$(.tools/bin/kubectl get svc medical-qa-ui -n medical-qa \
            -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || true)
          echo "Demo UI is up at http://$UI_IP:8501"
```

(Match the existing indentation and the existing nginx-IP lookup; only the `UI_IP` lookup + echo are new. If the existing block already assigns `IP` differently, keep that assignment and add only the two `UI_IP` lines.)

- [ ] **Step 4a: Run the CI workflow test**

Run: `.venv/bin/pytest tests/deploy/test_ci_workflows.py -q`
Expected: PASS.

- [ ] **Step 4b: Validate both workflow YAMLs parse**

Run: `.venv/bin/python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml')); yaml.safe_load(open('.github/workflows/demo-up.yml')); print('OK')"`
Expected: `OK`.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/ci.yml .github/workflows/demo-up.yml tests/deploy/test_ci_workflows.py
git commit -m "ci(ui): build the ui image in the matrix; demo-up prints the UI IP"
```

---

## Task 9: Full-suite verification

**Files:** none (verification only)

- [ ] **Step 1: Run the whole test suite with coverage**

Run: `.venv/bin/pytest --cov=medical_qa_platform --cov-report=term-missing --cov-fail-under=80 -q`
Expected: PASS, coverage ≥ 80% (unchanged — `app/` is outside the measured package).

- [ ] **Step 2: Run helm lint + template across all charts**

Run: `make helm-lint && make helm-template && echo ALL_OK`
Expected: each chart lints/templates cleanly, ending `ALL_OK`.

- [ ] **Step 3: Final no-op commit check**

Run: `git status --porcelain`
Expected: empty (everything committed). If anything is uncommitted, commit it with an appropriate message.

---

## Self-Review (completed during planning)

**Spec coverage:** UI client → Task 1; Streamlit app + `[demo]` extra + `make demo-ui` → Task 2; Dockerfile → Task 3; helm chart (ClusterIP base + LoadBalancer overlay, `API_BASE_URL`, `API_KEY` from `medical-qa-nginx-api-key`, `/_stcore/health` probes) → Task 4; Makefile lint/template/build → Task 5; `deploy.sh` install → Task 6; smoke rollout gate → Task 7; CI matrix + demo-up IP → Task 8. All spec sections mapped.

**Mandatory cross-test updates identified:** `test_ci_workflows.py` pins the dockerfile set by equality (Task 8 Step 1 updates it); `test_deploy_sh.py`, `test_smoke_cloud.py`, `test_makefile_deploy_tools.py`, `test_dockerfiles.py` get additive assertions in their respective tasks.

**Type/name consistency:** `PredictError`, `PredictResult`, `build_payload`, `predict`, `fetch_version` are defined in Task 1 and imported unchanged by the Streamlit app in Task 2. Service/Secret names (`medical-qa-ui`, `medical-qa-ui-config`, `medical-qa-nginx-api-key`/`API_KEY`), ports (8501), and probe path (`/_stcore/health`) match across the chart, its test, deploy.sh, and smoke.

**No placeholders:** every code/edit step contains the literal content.
