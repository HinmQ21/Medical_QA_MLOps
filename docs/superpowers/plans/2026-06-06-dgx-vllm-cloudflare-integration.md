# DGX-Spark vLLM + Cloudflare Tunnel Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the RunPod LLM backend with a self-hosted vLLM server on the DGX-Spark, exposed to the GKE-hosted API over a Cloudflare Tunnel.

**Architecture:** The mlops API already selects an OpenAI-compatible `ModelBackend` by env var. We rename the `runpod` backend to a provider-neutral `vllm` backend (`LLM_*` env), make generation length configurable (the trained model needs >512 tokens), make readiness reflect real LLM reachability, and wire the LLM API key into GKE as a Kubernetes Secret. The GKE-only mock demo stays green; flipping to the DGX is an opt-in deploy-time env. The DGX runs `vllm serve` (bound to loopback) plus `cloudflared` (named tunnel), both under systemd.

**Tech Stack:** Python 3.12, FastAPI, httpx, pytest, Helm, bash, vLLM, Cloudflare Tunnel (`cloudflared`).

**Working directory:** `/home/vcsai/minhlbq/mlops-platform` (all paths relative to it; all commands run from here).

**Test runner:** `.venv/bin/python -m pytest` (pyproject sets `pythonpath=["."]`, `addopts="-q"`). Helm-render tests auto-skip if `.tools/bin/helm` is absent.

---

## File Structure

Repo changes (TDD):

- `src/medical_qa_platform/inference/base.py` — MODIFY: add default `health_check()`.
- `src/medical_qa_platform/inference/vllm_backend.py` — CREATE: `VllmBackend` (renamed from RunpodBackend; `LLM_*` env; adds `health_check`).
- `src/medical_qa_platform/inference/runpod_backend.py` — DELETE.
- `src/medical_qa_platform/inference/__init__.py` — MODIFY: dispatch `"vllm"`, drop `"runpod"`.
- `src/medical_qa_platform/config.py` — MODIFY: add `max_tokens` from `MAX_TOKENS`.
- `src/medical_qa_platform/api/app.py` — MODIFY: plumb `max_tokens` into `generate`; `/ready` calls `health_check`.
- `tests/inference/test_vllm_backend.py` — CREATE (renamed from `test_runpod_backend.py`).
- `tests/inference/test_runpod_backend.py` — DELETE.
- `tests/inference/test_mock_backend.py` — MODIFY: add `get_backend("vllm")` dispatch test.
- `tests/test_config.py` — MODIFY: `runpod`→`vllm`, `MAX_TOKENS`.
- `tests/api/test_app.py` — MODIFY: max_tokens plumbing + unhealthy `/ready`.
- `deploy/helm/api/templates/configmap.yaml` — MODIFY: `LLM_BASE_URL`/`LLM_MODEL`/`MAX_TOKENS`, drop `RUNPOD_*`.
- `deploy/helm/api/values.yaml` — MODIFY: `llmBaseUrl`/`llmModel`/`maxTokens`.
- `deploy/helm/api/templates/deployment.yaml` — MODIFY: `LLM_API_KEY` from optional Secret.
- `tests/deploy/test_helm_api_chart.py` — MODIFY: assert new configmap keys + secret env.
- `scripts/cloud/create_secrets.sh` — MODIFY: conditionally create `medical-qa-llm-key`.
- `tests/cloud/test_create_secrets.py` — MODIFY: tests for the LLM secret.
- `scripts/cloud/deploy.sh` — MODIFY: opt-in `vllm` flip via env.
- `tests/cloud/test_deploy_sh.py` — MODIFY: tests for the flip.
- `README.md` — MODIFY: replace stale RunPod mentions.

Ops (non-TDD runbook):

- `docs/runbooks/dgx-vllm-cloudflare.md` — CREATE: DGX `vllm serve` + `cloudflared` systemd + deploy command.

---

## Task 1: Rename backend to `VllmBackend` + add `health_check`

**Files:**
- Modify: `src/medical_qa_platform/inference/base.py`
- Create: `src/medical_qa_platform/inference/vllm_backend.py`
- Delete: `src/medical_qa_platform/inference/runpod_backend.py`
- Create test: `tests/inference/test_vllm_backend.py`
- Delete test: `tests/inference/test_runpod_backend.py`

- [ ] **Step 1: Write the failing test**

Create `tests/inference/test_vllm_backend.py`:

```python
import json

import httpx

from medical_qa_platform.inference.vllm_backend import VllmBackend


def _backend(captured: dict, *, status: int = 200, json_body=None):
    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["method"] = request.method
        captured["headers"] = dict(request.headers)
        if request.method == "POST":
            captured["body"] = json.loads(request.content)
        return httpx.Response(
            status,
            json=json_body
            if json_body is not None
            else {"choices": [{"message": {"content": "<answer>B</answer>"}}]},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return VllmBackend(base_url="http://pod/v1", model="m", api_key="k", client=client)


def test_name():
    assert _backend({}).name == "vllm"


def test_generate_returns_content():
    out = _backend({}).generate([{"role": "user", "content": "x"}])
    assert out == "<answer>B</answer>"


def test_posts_to_chat_completions_with_auth_model_and_max_tokens():
    captured = {}
    _backend(captured).generate([{"role": "user", "content": "x"}], max_tokens=2048)
    assert captured["url"] == "http://pod/v1/chat/completions"
    assert captured["headers"]["authorization"] == "Bearer k"
    assert captured["body"]["model"] == "m"
    assert captured["body"]["max_tokens"] == 2048
    assert captured["body"]["messages"] == [{"role": "user", "content": "x"}]


def test_from_env(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://x/v1")
    monkeypatch.setenv("LLM_MODEL", "llama")
    monkeypatch.setenv("LLM_API_KEY", "secret")
    backend = VllmBackend.from_env()
    assert backend.base_url == "http://x/v1"
    assert backend.model == "llama"
    assert backend.api_key == "secret"


def test_health_check_true_on_200_models():
    captured = {}
    ok = _backend(captured, json_body={"data": []}).health_check()
    assert ok is True
    assert captured["url"] == "http://pod/v1/models"
    assert captured["method"] == "GET"


def test_health_check_false_on_bad_status():
    backend = _backend({}, status=503, json_body={"error": "down"})
    assert backend.health_check() is False


def test_health_check_false_on_transport_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    backend = VllmBackend(base_url="http://pod/v1", model="m", client=client)
    assert backend.health_check() is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/inference/test_vllm_backend.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'medical_qa_platform.inference.vllm_backend'`.

- [ ] **Step 3: Add `health_check` to the base interface**

Edit `src/medical_qa_platform/inference/base.py` — add this method to `ModelBackend` after `generate`:

```python
    def health_check(self) -> bool:
        """Return True if the backend is reachable and ready to serve.

        Default: assume healthy (mock/in-process backends never go down).
        Network backends override this with a real probe.
        """
        return True
```

- [ ] **Step 4: Create `VllmBackend`**

Create `src/medical_qa_platform/inference/vllm_backend.py`:

```python
"""vLLM (OpenAI-compatible) chat-completions backend.

Talks to any server exposing the OpenAI ``/v1/chat/completions`` API. In this
project that is a self-hosted vLLM server on the DGX-Spark, reached over a
Cloudflare Tunnel. Replaces the former RunPod backend; the wire protocol is
identical, so only the env prefix and name changed (RUNPOD_* -> LLM_*).
"""

import os

import httpx

from .base import ModelBackend


class VllmBackend(ModelBackend):
    name = "vllm"

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        client: httpx.Client | None = None,
        timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self._client = client or httpx.Client(timeout=timeout)

    @classmethod
    def from_env(cls) -> "VllmBackend":
        return cls(
            base_url=os.environ.get("LLM_BASE_URL", ""),
            model=os.environ.get("LLM_MODEL", ""),
            api_key=os.environ.get("LLM_API_KEY", ""),
        )

    def _auth_headers(self) -> dict:
        if self.api_key:
            return {"Authorization": f"Bearer {self.api_key}"}
        return {}

    def generate(
        self,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> str:
        resp = self._client.post(
            f"{self.base_url}/chat/completions",
            headers=self._auth_headers(),
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def health_check(self) -> bool:
        try:
            resp = self._client.get(
                f"{self.base_url}/models",
                headers=self._auth_headers(),
                timeout=5.0,
            )
            return resp.status_code == 200
        except httpx.HTTPError:
            return False
```

- [ ] **Step 5: Delete the old backend and its test**

Run:
```bash
git rm src/medical_qa_platform/inference/runpod_backend.py tests/inference/test_runpod_backend.py
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/inference/test_vllm_backend.py tests/inference/test_mock_backend.py -v`
Expected: `test_vllm_backend.py` all PASS. `test_mock_backend.py` may FAIL on `test_factory_*` only if it referenced runpod — it does not yet, so it should still PASS here (the `get_backend` change comes in Task 2). If `test_mock_backend` errors on import, stop and recheck Step 3.

- [ ] **Step 7: Commit**

```bash
git add src/medical_qa_platform/inference/base.py \
        src/medical_qa_platform/inference/vllm_backend.py \
        tests/inference/test_vllm_backend.py
git commit -m "feat(inference): rename runpod backend to vllm + add health_check"
```

---

## Task 2: Dispatch `vllm` in `get_backend`

**Files:**
- Modify: `src/medical_qa_platform/inference/__init__.py`
- Modify test: `tests/inference/test_mock_backend.py`

- [ ] **Step 1: Write the failing test**

Edit `tests/inference/test_mock_backend.py` — add at the end:

```python
def test_factory_returns_vllm_by_name():
    from medical_qa_platform.inference.vllm_backend import VllmBackend

    assert isinstance(get_backend("vllm"), VllmBackend)


def test_factory_no_longer_knows_runpod():
    with pytest.raises(ValueError):
        get_backend("runpod")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/inference/test_mock_backend.py::test_factory_returns_vllm_by_name -v`
Expected: FAIL with `ValueError: unknown MODEL_BACKEND: 'vllm'`.

- [ ] **Step 3: Update the dispatch**

Edit `src/medical_qa_platform/inference/__init__.py` — replace the `runpod` branch:

```python
    if name == "vllm":
        from .vllm_backend import VllmBackend

        return VllmBackend.from_env()
```

(Remove the entire `if name == "runpod": ... RunpodBackend.from_env()` block. Keep `mock` and `kserve`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/inference/ -v`
Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add src/medical_qa_platform/inference/__init__.py tests/inference/test_mock_backend.py
git commit -m "feat(inference): dispatch vllm backend, drop runpod"
```

---

## Task 3: Configurable `max_tokens`

**Files:**
- Modify: `src/medical_qa_platform/config.py`
- Modify test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Edit `tests/test_config.py` — replace the whole file with:

```python
from medical_qa_platform.config import Settings


def test_defaults(monkeypatch):
    for var in [
        "MODEL_BACKEND",
        "RETRIEVAL_URL",
        "MODEL_VERSION",
        "TOP_K",
        "DRIFT_LOG_PATH",
        "MAX_TOKENS",
    ]:
        monkeypatch.delenv(var, raising=False)
    settings = Settings.from_env()
    assert settings.model_backend == "mock"
    assert settings.retrieval_url == "http://localhost:8001"
    assert settings.model_version == "dev"
    assert settings.top_k == 5
    assert settings.max_tokens == 512


def test_reads_env(monkeypatch):
    monkeypatch.setenv("MODEL_BACKEND", "vllm")
    monkeypatch.setenv("MODEL_VERSION", "v6.1")
    monkeypatch.setenv("TOP_K", "8")
    monkeypatch.setenv("MAX_TOKENS", "2048")
    settings = Settings.from_env()
    assert settings.model_backend == "vllm"
    assert settings.model_version == "v6.1"
    assert settings.top_k == 8
    assert settings.max_tokens == 2048
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_config.py -v`
Expected: FAIL — `AttributeError: 'Settings' object has no attribute 'max_tokens'`.

- [ ] **Step 3: Add the field**

Edit `src/medical_qa_platform/config.py`:

Add to the dataclass fields (after `drift_log_path`):
```python
    max_tokens: int = 512
```

Add to `from_env()` return kwargs (after `drift_log_path=...`):
```python
            max_tokens=int(os.environ.get("MAX_TOKENS", "512")),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_config.py -v`
Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add src/medical_qa_platform/config.py tests/test_config.py
git commit -m "feat(config): add MAX_TOKENS setting"
```

---

## Task 4: Plumb `max_tokens` into `/predict` and gate `/ready` on backend health

**Files:**
- Modify: `src/medical_qa_platform/api/app.py`
- Modify test: `tests/api/test_app.py`

- [ ] **Step 1: Write the failing tests**

Edit `tests/api/test_app.py` — add this import near the top (after the existing imports):

```python
from medical_qa_platform.inference.base import ModelBackend
```

Then append these tests and helpers to the end of the file:

```python
class _RecordingBackend(ModelBackend):
    name = "rec"

    def __init__(self):
        self.max_tokens_calls = []

    def generate(self, messages, max_tokens=512, temperature=0.3):
        self.max_tokens_calls.append(max_tokens)
        return "<answer>A</answer>"


class _UnhealthyBackend(ModelBackend):
    name = "down"

    def generate(self, messages, max_tokens=512, temperature=0.3):
        return "<answer>A</answer>"

    def health_check(self):
        return False


def test_predict_passes_configured_max_tokens(tmp_path):
    backend = _RecordingBackend()
    app = create_app(
        backend=backend,
        retrieval=FixtureRetrieval({}),
        max_tokens=2048,
        drift_log_path=str(tmp_path / "d.jsonl"),
    )
    TestClient(app).post(
        "/predict", json={"question": "Q?", "options": {"A": "a", "B": "b"}}
    )
    assert backend.max_tokens_calls == [2048]


def test_ready_returns_503_when_backend_unhealthy(tmp_path):
    app = create_app(
        backend=_UnhealthyBackend(),
        retrieval=FixtureRetrieval({}),
        drift_log_path=str(tmp_path / "d.jsonl"),
    )
    assert TestClient(app).get("/ready").status_code == 503
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/api/test_app.py::test_predict_passes_configured_max_tokens tests/api/test_app.py::test_ready_returns_503_when_backend_unhealthy -v`
Expected: `test_predict_passes_configured_max_tokens` FAILs (`create_app() got an unexpected keyword argument 'max_tokens'`); `test_ready_returns_503...` FAILs (returns 200 — `/ready` ignores health).

- [ ] **Step 3: Update `create_app`**

Edit `src/medical_qa_platform/api/app.py`:

(a) Add `max_tokens` to the signature — change:
```python
    top_k: int | None = None,
) -> FastAPI:
```
to:
```python
    top_k: int | None = None,
    max_tokens: int | None = None,
) -> FastAPI:
```

(b) Set state after the `app.state.top_k = ...` line:
```python
    app.state.max_tokens = (
        max_tokens if max_tokens is not None else settings.max_tokens
    )
```

(c) Pass it to `generate` — change:
```python
        raw = app.state.backend.generate(messages)
```
to:
```python
        raw = app.state.backend.generate(messages, max_tokens=app.state.max_tokens)
```

(d) Gate `/ready` on health — replace the `ready` handler body:
```python
    @app.get("/ready")
    def ready(response: Response):
        backend = getattr(app.state, "backend", None)
        if backend is None or not backend.health_check():
            response.status_code = 503
            return {"status": "not ready"}
        return {"status": "ready"}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/api/test_app.py -v`
Expected: ALL PASS (including the existing `test_ready` — MockBackend's default `health_check` returns True).

- [ ] **Step 5: Commit**

```bash
git add src/medical_qa_platform/api/app.py tests/api/test_app.py
git commit -m "feat(api): configurable max_tokens + readiness gated on backend health"
```

---

## Task 5: Helm — rename config keys, add MAX_TOKENS, wire LLM_API_KEY Secret

**Files:**
- Modify: `deploy/helm/api/templates/configmap.yaml`
- Modify: `deploy/helm/api/values.yaml`
- Modify: `deploy/helm/api/templates/deployment.yaml`
- Modify test: `tests/deploy/test_helm_api_chart.py`

- [ ] **Step 1: Write the failing test**

Edit `tests/deploy/test_helm_api_chart.py` — inside `test_api_chart_renders_deployment_service_configmap_and_hpa`, add after the existing `config["data"]` asserts:

```python
    assert config["data"]["LLM_BASE_URL"] == ""
    assert config["data"]["LLM_MODEL"] == ""
    assert config["data"]["MAX_TOKENS"] == "2048"
    assert "RUNPOD_BASE_URL" not in config["data"]
    assert "RUNPOD_MODEL" not in config["data"]
    env_by_name = {e["name"]: e for e in container.get("env", [])}
    ref = env_by_name["LLM_API_KEY"]["valueFrom"]["secretKeyRef"]
    assert ref["name"] == "medical-qa-llm-key"
    assert ref["key"] == "API_KEY"
    assert ref["optional"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/deploy/test_helm_api_chart.py -v`
Expected: FAIL with `KeyError: 'LLM_BASE_URL'` (or SKIP if `.tools/bin/helm` is missing — if skipped, run `make install-deploy-tools` first, then rerun).

- [ ] **Step 3: Update the ConfigMap**

Replace the `data:` block of `deploy/helm/api/templates/configmap.yaml` with:

```yaml
data:
  MODEL_BACKEND: {{ .Values.env.modelBackend | quote }}
  MODEL_VERSION: {{ .Values.env.modelVersion | quote }}
  RETRIEVAL_URL: {{ .Values.env.retrievalUrl | quote }}
  LLM_BASE_URL: {{ .Values.env.llmBaseUrl | quote }}
  LLM_MODEL: {{ .Values.env.llmModel | quote }}
  KSERVE_URL: {{ .Values.env.kserveUrl | quote }}
  TOP_K: {{ .Values.env.topK | quote }}
  MAX_TOKENS: {{ .Values.env.maxTokens | quote }}
  DRIFT_LOG_PATH: {{ .Values.env.driftLogPath | quote }}
```

- [ ] **Step 4: Update values.yaml**

In `deploy/helm/api/values.yaml`, replace the `env:` block with:

```yaml
env:
  modelBackend: mock
  modelVersion: smoke-dev
  retrievalUrl: http://medical-qa-retrieval:8001
  llmBaseUrl: ""
  llmModel: ""
  kserveUrl: http://medical-qa-kserve/v1/models/medical-qa-smoke:predict
  topK: "5"
  # the trained model emits <think>...</think><answer> sequences far longer than
  # 512 tokens; truncation drops the <answer> tag and yields a null answer.
  maxTokens: "2048"
  # the container runs non-root with a root-owned /app; write drift telemetry to a
  # writable dir so /predict doesn't 500 on the log append.
  driftLogPath: /tmp/drift_log.jsonl
```

- [ ] **Step 5: Wire the LLM key into the deployment**

In `deploy/helm/api/templates/deployment.yaml`, add an `env:` block immediately after the `envFrom:` block (i.e. after the `configMapRef` lines, before `ports:`):

```yaml
          env:
            # Optional: only the vllm backend needs it. The mock demo leaves the
            # secret absent; optional=true keeps the pod schedulable without it.
            - name: LLM_API_KEY
              valueFrom:
                secretKeyRef:
                  name: medical-qa-llm-key
                  key: API_KEY
                  optional: true
```

- [ ] **Step 6: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/deploy/test_helm_api_chart.py -v`
Expected: ALL PASS.

- [ ] **Step 7: Commit**

```bash
git add deploy/helm/api/templates/configmap.yaml deploy/helm/api/values.yaml \
        deploy/helm/api/templates/deployment.yaml tests/deploy/test_helm_api_chart.py
git commit -m "feat(helm): LLM_* config keys, MAX_TOKENS, optional LLM_API_KEY secret"
```

---

## Task 6: `create_secrets.sh` — create `medical-qa-llm-key` when provided

**Files:**
- Modify: `scripts/cloud/create_secrets.sh`
- Modify test: `tests/cloud/test_create_secrets.py`

- [ ] **Step 1: Write the failing tests**

Edit `tests/cloud/test_create_secrets.py` — append:

```python
def test_dry_run_creates_llm_secret_when_key_set():
    env = {**ENV, "LLM_API_KEY": "dgx-key"}
    out = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"], capture_output=True, text=True, env=env
    )
    assert out.returncode == 0, out.stderr
    o = out.stdout
    assert "kubectl create secret generic medical-qa-llm-key" in o
    assert "--from-literal=API_KEY=dgx-key" in o


def test_dry_run_skips_llm_secret_when_key_unset():
    out = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"], capture_output=True, text=True, env=ENV
    )
    assert out.returncode == 0, out.stderr
    assert "medical-qa-llm-key" not in out.stdout
```

(The existing `test_no_runpod_references` must keep passing — do **not** introduce the string `RUNPOD` in the script.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/cloud/test_create_secrets.py -v`
Expected: `test_dry_run_creates_llm_secret_when_key_set` FAILs (no such output); the skip test PASSes already.

- [ ] **Step 3: Add the conditional secret**

In `scripts/cloud/create_secrets.sh`, after the existing
```bash
apply_secret medical-qa-nginx-api-key \
  "--from-literal=API_KEY=$NGINX_API_KEY"

echo "Secret medical-qa-nginx-api-key applied in namespace $K8S_NAMESPACE."
```
add:
```bash
# Optional LLM backend key (self-hosted vLLM on DGX-Spark via Cloudflare Tunnel).
# Created only when provided; the mock-backend demo leaves it unset and the api
# deployment treats the secret as optional.
if [ -n "${LLM_API_KEY:-}" ]; then
  apply_secret medical-qa-llm-key \
    "--from-literal=API_KEY=$LLM_API_KEY"
  echo "Secret medical-qa-llm-key applied in namespace $K8S_NAMESPACE."
else
  echo "LLM_API_KEY not set; skipping medical-qa-llm-key (mock-backend demo)."
fi
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/cloud/test_create_secrets.py -v`
Expected: ALL PASS (including `test_no_runpod_references` and `test_missing_api_key_fails_fast`).

- [ ] **Step 5: Commit**

```bash
git add scripts/cloud/create_secrets.sh tests/cloud/test_create_secrets.py
git commit -m "feat(cloud): create medical-qa-llm-key secret when LLM_API_KEY is set"
```

---

## Task 7: `deploy.sh` — opt-in flip to the vllm backend

**Files:**
- Modify: `scripts/cloud/deploy.sh`
- Modify test: `tests/cloud/test_deploy_sh.py`

- [ ] **Step 1: Write the failing tests**

Edit `tests/cloud/test_deploy_sh.py` — append:

```python
def test_dry_run_flips_to_vllm_backend_when_requested():
    env = {
        **ENV,
        "MODEL_BACKEND": "vllm",
        "LLM_BASE_URL": "https://llm.example/v1",
        "LLM_MODEL": "medical-qa-llama-gdpo",
    }
    out = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"], capture_output=True, text=True, env=env
    )
    assert out.returncode == 0, out.stderr
    o = out.stdout
    assert "env.modelBackend=vllm" in o
    assert "env.llmBaseUrl=https://llm.example/v1" in o
    assert "env.llmModel=medical-qa-llama-gdpo" in o


def test_vllm_backend_requires_base_url_and_model():
    env = {**ENV, "MODEL_BACKEND": "vllm"}
    out = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"], capture_output=True, text=True, env=env
    )
    assert out.returncode != 0
```

(The existing `test_dry_run_does_not_use_runpod_or_api_prod_overlay` runs with no `MODEL_BACKEND`, so it stays on the mock path and must keep passing.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/cloud/test_deploy_sh.py -v`
Expected: `test_dry_run_flips_to_vllm_backend_when_requested` FAILs (no such `--set` output).

- [ ] **Step 3: Update deploy.sh**

In `scripts/cloud/deploy.sh`, replace the api install block:
```bash
run "$HELM" upgrade --install medical-qa-api deploy/helm/api \
  --namespace "$K8S_NAMESPACE" \
  --set image.tag="$IMAGE_TAG"
```
with:
```bash
# api defaults to the mock backend (GKE-only demo). Set MODEL_BACKEND=vllm plus
# LLM_BASE_URL / LLM_MODEL to point at the self-hosted vLLM server on the DGX-Spark
# (reached via Cloudflare Tunnel); the LLM_API_KEY secret is created separately.
API_SET=(--set image.tag="$IMAGE_TAG")
BACKEND="${MODEL_BACKEND:-mock}"
if [ "$BACKEND" = "vllm" ]; then
  : "${LLM_BASE_URL:?set LLM_BASE_URL (e.g. https://llm.example/v1) for the vllm backend}"
  : "${LLM_MODEL:?set LLM_MODEL (vllm --served-model-name) for the vllm backend}"
  API_SET+=(--set env.modelBackend=vllm \
            --set env.llmBaseUrl="$LLM_BASE_URL" \
            --set env.llmModel="$LLM_MODEL")
fi

run "$HELM" upgrade --install medical-qa-api deploy/helm/api \
  --namespace "$K8S_NAMESPACE" \
  "${API_SET[@]}"
```

Then change the final echo:
```bash
echo "Deployed all charts (api=mock) to namespace $K8S_NAMESPACE."
```
to:
```bash
echo "Deployed all charts (api=$BACKEND) to namespace $K8S_NAMESPACE."
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/cloud/test_deploy_sh.py -v`
Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/cloud/deploy.sh tests/cloud/test_deploy_sh.py
git commit -m "feat(cloud): opt-in vllm backend flip in deploy.sh"
```

---

## Task 8: Reconcile README RunPod mentions

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update the two stale lines**

In `README.md`, replace the line containing `real RunPod` (~line 105):
```
`InferenceService`. Live GKE deployment, GCS DVC remote credentials, real RunPod
```
with:
```
`InferenceService`. Live GKE deployment, GCS DVC remote credentials, a self-hosted
```

And replace the line containing `(no RunPod)` (~line 111):
```
CI/CD. The model uses the **mock** backend (no RunPod); retrieval is real. See the
```
with:
```
CI/CD. The CI demo uses the **mock** backend; flip to `MODEL_BACKEND=vllm`
(self-hosted vLLM on the DGX-Spark via Cloudflare Tunnel) for a real model —
see `docs/runbooks/dgx-vllm-cloudflare.md`. Retrieval is always real. See the
```

- [ ] **Step 2: Verify no test regressed**

Run: `.venv/bin/python -m pytest tests/test_readme_cloud_docs.py tests/test_readme_deploy_docs.py tests/test_readme_pipeline_docs.py -v`
Expected: ALL PASS (these assert pipeline/deploy doc structure, not RunPod strings).

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs(readme): replace RunPod mentions with self-hosted vLLM path"
```

---

## Task 9: DGX runbook (vLLM serve + Cloudflare Tunnel)

**Files:**
- Create: `docs/runbooks/dgx-vllm-cloudflare.md`

This task is an operational runbook (no unit tests). Create the file verbatim:

```markdown
# Runbook: DGX-Spark vLLM behind Cloudflare Tunnel

Self-host the trained model on the DGX-Spark and expose it to the GKE API as the
`vllm` backend. Two long-lived processes run on the DGX under systemd: the vLLM
OpenAI server (loopback only) and `cloudflared` (named tunnel).

## Prerequisites
- Merged checkpoint on disk: `baseline/outputs/gdpo_llama32_3b_stage2_v6_1_vllm_merged`
- `vllm_venv312` present (see baseline/CLAUDE.md)
- A domain added to Cloudflare (free plan is fine)
- `cloudflared` installed on the DGX

## 1. vLLM OpenAI server (loopback)

Pick a strong key and export it once for the unit:
```bash
echo "DGX_LLM_KEY=$(openssl rand -hex 24)" | sudo tee /etc/medqa-llm.env
```

`/etc/systemd/system/medqa-vllm.service`:
```ini
[Unit]
Description=medqa vLLM OpenAI server
After=network-online.target

[Service]
User=vcsai
EnvironmentFile=/etc/medqa-llm.env
WorkingDirectory=/home/vcsai/minhlbq/baseline
ExecStart=/home/vcsai/minhlbq/baseline/vllm_venv312/bin/python -m vllm.entrypoints.openai.api_server \
  --model outputs/gdpo_llama32_3b_stage2_v6_1_vllm_merged \
  --served-model-name medical-qa-llama-gdpo \
  --host 127.0.0.1 --port 8001 \
  --api-key ${DGX_LLM_KEY} \
  --gpu-memory-utilization 0.6 --max-model-len 4096
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Note: `--served-model-name medical-qa-llama-gdpo` MUST equal the GKE `LLM_MODEL`.

```bash
sudo systemctl daemon-reload && sudo systemctl enable --now medqa-vllm
curl -s -H "Authorization: Bearer $(grep -oP '(?<=DGX_LLM_KEY=).*' /etc/medqa-llm.env)" \
  http://127.0.0.1:8001/v1/models   # expect 200 + medical-qa-llama-gdpo
```

## 2. Cloudflare named tunnel

```bash
cloudflared tunnel login
cloudflared tunnel create medqa-llm          # writes ~/.cloudflared/<UUID>.json
cloudflared tunnel route dns medqa-llm llm.<your-domain>
```

`~/.cloudflared/config.yml`:
```yaml
tunnel: medqa-llm
credentials-file: /home/vcsai/.cloudflared/<UUID>.json
ingress:
  - hostname: llm.<your-domain>
    service: http://127.0.0.1:8001
  - service: http_status:404
```

`/etc/systemd/system/medqa-cloudflared.service`:
```ini
[Unit]
Description=medqa cloudflared tunnel
After=network-online.target medqa-vllm.service

[Service]
User=vcsai
ExecStart=/usr/bin/cloudflared tunnel run medqa-llm
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload && sudo systemctl enable --now medqa-cloudflared
curl -s -H "Authorization: Bearer <DGX_LLM_KEY>" https://llm.<your-domain>/v1/models
```

## 3. Point GKE at the DGX

```bash
# create the in-cluster secret from the same key
export LLM_API_KEY="<DGX_LLM_KEY>"
bash scripts/cloud/create_secrets.sh        # also (re)creates the nginx key

# deploy the api on the vllm backend (note the /v1 suffix)
export MODEL_BACKEND=vllm
export LLM_BASE_URL="https://llm.<your-domain>/v1"
export LLM_MODEL="medical-qa-llama-gdpo"
bash scripts/cloud/deploy.sh
```

Verify: `kubectl -n <ns> rollout status deploy/medical-qa-api`; the pod becomes
Ready only when `/ready` -> `health_check()` -> `GET <base_url>/models` succeeds,
i.e. the tunnel and vLLM are both up.

## Gotchas
- `LLM_BASE_URL` MUST end in `/v1` (backend appends `/chat/completions`).
- Cloudflare free tier drops origin requests after ~100s (error 524); a 3B model
  at <=2048 tokens answers in seconds, so this is only a concern under overload.
- Keep `--api-key` on; without it the tunnel URL is an open inference endpoint.
- Accuracy here is single-shot RAG (platform pre-retrieves evidence), not the
  agentic tool loop used in training — do not quote the 60.09% MedQA number for
  this serving path; measure it separately.
```

- [ ] **Step 2: Commit**

```bash
git add docs/runbooks/dgx-vllm-cloudflare.md
git commit -m "docs(runbook): DGX vLLM + Cloudflare Tunnel setup"
```

---

## Task 10: Full suite + coverage gate

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `.venv/bin/python -m pytest`
Expected: all PASS (helm-render tests SKIP only if `.tools/bin/helm` is absent). Coverage gate is `fail_under = 80`.

- [ ] **Step 2: Grep for leftover RunPod references in code/config**

Run:
```bash
grep -rn "runpod\|RUNPOD\|Runpod" src/ deploy/ scripts/ tests/ --exclude-dir=.venv
```
Expected: only the negative-assertion test lines (`test_no_runpod_references`, `test_dry_run_does_not_use_runpod...`, `test_factory_no_longer_knows_runpod`, `RUNPOD" not in text`). No production code or config should match.

- [ ] **Step 3: Final commit (if grep surfaced stragglers, fix then commit)**

```bash
git add -A
git commit -m "chore: complete runpod->vllm (DGX + Cloudflare Tunnel) integration" || echo "nothing to commit"
```

---

## Self-Review Notes

- **Spec coverage:** vLLM backend rename (T1–T2), max_tokens fix for truncated `<answer>` (T3–T4), readiness `/models` probe (T4), LLM_API_KEY Secret wiring incl. mock-safe `optional:true` (T5–T6), opt-in deploy flip preserving the mock demo (T7), docs (T8–T9), verification (T10). Cloudflare Access was explicitly out of scope.
- **Naming consistency:** backend name `vllm`; env `LLM_BASE_URL`/`LLM_MODEL`/`LLM_API_KEY`/`MAX_TOKENS`; helm values `llmBaseUrl`/`llmModel`/`maxTokens`; secret `medical-qa-llm-key` key `API_KEY`; served model `medical-qa-llama-gdpo` — used identically across code, helm, scripts, and runbook.
- **Demo safety:** with no `MODEL_BACKEND`/`LLM_*` set, deploy.sh stays mock, the secret is skipped, the deployment's secret ref is `optional`, and `smoke_cloud.sh` still gets a deterministic mock answer — the existing `demo-up.yml` CI path is unaffected.
