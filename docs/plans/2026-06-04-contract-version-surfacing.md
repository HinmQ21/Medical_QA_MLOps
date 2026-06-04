# Contract-Version Surfacing (SP4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the serving stack report `RETRIEVAL_CONTRACT_VERSION` (and the configured encoder/KG) so a deployment can be matched to the model version registered by the SP3 pipeline.

**Architecture:** Add a `contract_version` field to `PredictResponse`, a `GET /version` endpoint to the API, and a `GET /version` endpoint to the retrieval service. Report-only; no enforcement, no change to retrieval/generation logic.

**Tech Stack:** Python 3.12, FastAPI, Pydantic v2, pytest (FastAPI `TestClient`).

**Spec:** `docs/specs/2026-06-04-contract-version-surfacing-design.md`

**Working dir:** `/home/vcsai/minhlbq/mlops-platform` (`.venv/bin/python`).

---

## File Structure

**Modify:**
- `src/medical_qa_platform/api/schemas.py` — add `contract_version: str` to `PredictResponse`.
- `src/medical_qa_platform/api/app.py` — import the constant; populate `/predict`; add `GET /version`.
- `src/medical_qa_platform/retrieval/service.py` — add `GET /version`.
- `tests/api/test_app.py` — `/version` + `/predict` field tests.
- `tests/retrieval/test_service.py` — retrieval `/version` test.

---

## Task 1: API — `contract_version` in `/predict` + `GET /version`

**Files:**
- Modify: `src/medical_qa_platform/api/schemas.py`, `src/medical_qa_platform/api/app.py`
- Test: `tests/api/test_app.py`

- [ ] **Step 1: Write the failing tests** — append to `tests/api/test_app.py`:

```python
def test_predict_response_includes_contract_version():
    from fastapi.testclient import TestClient

    from medical_qa_platform.api.app import create_app
    from medical_qa_platform.retrieval.backends import FixtureRetrieval
    from medical_qa_platform.retrieval.contract import RETRIEVAL_CONTRACT_VERSION

    class _Backend:
        name = "mock"

        def generate(self, messages):
            return "<answer>A</answer>"

    app = create_app(
        backend=_Backend(),
        retrieval=FixtureRetrieval({"q": ["fact"]}),
        model_version="testv",
        top_k=1,
    )
    client = TestClient(app)
    resp = client.post("/predict", json={"question": "q", "options": {"A": "x", "B": "y"}})
    assert resp.status_code == 200
    assert resp.json()["contract_version"] == RETRIEVAL_CONTRACT_VERSION


def test_version_endpoint_reports_contract_and_model():
    from fastapi.testclient import TestClient

    from medical_qa_platform.api.app import create_app
    from medical_qa_platform.retrieval.backends import FixtureRetrieval
    from medical_qa_platform.retrieval.contract import RETRIEVAL_CONTRACT_VERSION

    class _Backend:
        name = "mock"

        def generate(self, messages):
            return "<answer>A</answer>"

    app = create_app(backend=_Backend(), retrieval=FixtureRetrieval({}), model_version="testv")
    client = TestClient(app)
    body = client.get("/version").json()
    assert body["contract_version"] == RETRIEVAL_CONTRACT_VERSION
    assert body["model_version"] == "testv"
    assert body["backend"] == "mock"
```

(If `tests/api/test_app.py` already imports `TestClient`/`create_app` at module level, you may use those instead of the local imports — but local imports are safe and self-contained.)

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/api/test_app.py -k "contract_version or version_endpoint" -q`
Expected: FAIL — `PredictResponse` has no `contract_version`; `/version` returns 404.

- [ ] **Step 3a: Add the field to `schemas.py`** — in `PredictResponse`, add `contract_version`:

```python
class PredictResponse(BaseModel):
    answer: str | None
    evidence: list[str]
    backend: str
    model_version: str
    contract_version: str
    latency_ms: float
    trace_id: str
```

- [ ] **Step 3b: Edit `app.py`** — add the import (after the `from .schemas import ...` line):

```python
from ..retrieval.contract import RETRIEVAL_CONTRACT_VERSION
```

In the `/predict` handler, add `contract_version` to the `PredictResponse(...)` construction (alongside `model_version=app.state.model_version,`):

```python
            model_version=app.state.model_version,
            contract_version=RETRIEVAL_CONTRACT_VERSION,
```

Add a `/version` endpoint (place it next to `/metrics`, before `return app`):

```python
    @app.get("/version")
    def version():
        backend = getattr(app.state, "backend", None)
        return {
            "contract_version": RETRIEVAL_CONTRACT_VERSION,
            "model_version": app.state.model_version,
            "backend": backend.name if backend is not None else None,
        }
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/api/test_app.py -q`
Expected: PASS (all `test_app.py` tests, including the two new ones).

- [ ] **Step 5: Commit**

```bash
git add src/medical_qa_platform/api/schemas.py src/medical_qa_platform/api/app.py tests/api/test_app.py
git commit -m "feat(api): surface contract_version in /predict + GET /version"
```

---

## Task 2: Retrieval service — `GET /version`

**Files:**
- Modify: `src/medical_qa_platform/retrieval/service.py`
- Test: `tests/retrieval/test_service.py`

- [ ] **Step 1: Write the failing test** — append to `tests/retrieval/test_service.py`:

```python
def test_retrieval_version_reports_contract_encoder_and_kg_dir(monkeypatch):
    from fastapi.testclient import TestClient

    from medical_qa_platform.retrieval.backends import FixtureRetrieval
    from medical_qa_platform.retrieval.contract import RETRIEVAL_CONTRACT_VERSION
    from medical_qa_platform.retrieval.service import create_retrieval_service

    monkeypatch.setenv("KG_ENCODER_MODEL", "abhinand/MedEmbed-small-v0.1")
    monkeypatch.setenv("KG_DATA_DIR", "/mnt/artifacts/kg")
    app = create_retrieval_service(backend=FixtureRetrieval({}))
    client = TestClient(app)
    body = client.get("/version").json()
    assert body["contract_version"] == RETRIEVAL_CONTRACT_VERSION
    assert body["encoder_model"] == "abhinand/MedEmbed-small-v0.1"
    assert body["kg_data_dir"] == "/mnt/artifacts/kg"


def test_retrieval_version_defaults_to_small_encoder(monkeypatch):
    from fastapi.testclient import TestClient

    from medical_qa_platform.retrieval.backends import FixtureRetrieval
    from medical_qa_platform.retrieval.service import create_retrieval_service

    monkeypatch.delenv("KG_ENCODER_MODEL", raising=False)
    app = create_retrieval_service(backend=FixtureRetrieval({}))
    client = TestClient(app)
    assert client.get("/version").json()["encoder_model"] == "abhinand/MedEmbed-small-v0.1"
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/retrieval/test_service.py -k version -q`
Expected: FAIL — `/version` returns 404.

- [ ] **Step 3: Edit `service.py`** — add `import os` (top, with the other stdlib imports) and the contract import (with the other `from .` imports):

```python
import os
```
```python
from .contract import RETRIEVAL_CONTRACT_VERSION
```

Add a `/version` route inside `create_retrieval_service` (next to the other `@app.get` routes, before `return app`):

```python
    @app.get("/version")
    def version():
        return {
            "contract_version": RETRIEVAL_CONTRACT_VERSION,
            "encoder_model": os.environ.get(
                "KG_ENCODER_MODEL", "abhinand/MedEmbed-small-v0.1"
            ),
            "kg_data_dir": os.environ.get("KG_DATA_DIR", "data/"),
        }
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/retrieval/test_service.py -q`
Expected: PASS (all service tests, including the two new ones).

- [ ] **Step 5: Commit**

```bash
git add src/medical_qa_platform/retrieval/service.py tests/retrieval/test_service.py
git commit -m "feat(retrieval): GET /version reports contract + encoder + kg_dir"
```

---

## Task 3: Full-suite + coverage gate

**Files:** none (verification only).

- [ ] **Step 1: Full suite + coverage**

Run: `.venv/bin/python -m pytest --cov --cov-report= -q 2>&1 | tail -3`
Expected: all pass; coverage ≥ 80%.

- [ ] **Step 2: Confirm self-contained guard still green**

Run: `.venv/bin/python -m pytest tests/retrieval/test_kg_backend_self_contained.py -q`
Expected: PASS — no `baseline` token leaked into `src/` (the new imports reference `..retrieval.contract` / `.contract`, not baseline).

- [ ] **Step 3: Stop** — `git status` clean. SP4 complete on branch `contract-version-surfacing`.

---

## Notes for the implementer

- **Report-only.** `/version` and the new field describe the running contract; they do not validate or enforce a match. No change to `/search`, retrieval, or generation.
- **Field order in `PredictResponse`** does not affect JSON consumers, but place `contract_version` after `model_version` for readability.
- **The encoder default** (`abhinand/MedEmbed-small-v0.1`) is intentionally duplicated between `KGRetrieval` and the retrieval `/version` — both must stay in sync; the spec notes centralizing if it ever drifts.
