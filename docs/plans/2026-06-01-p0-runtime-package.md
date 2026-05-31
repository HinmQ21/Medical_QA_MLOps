# MLOps P0 — Runtime Inference Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `medical_qa_platform` Python package inside `mlops-platform/` — a FastAPI pre/post-processing API with a pluggable model-backend abstraction (mock/runpod/kserve), a retrieval client + standalone retrieval service, observability hooks, and a drift-row collector — all locally testable to >80% coverage with no GPU.

**Architecture:** A clean, installable package under `mlops-platform/src/medical_qa_platform/`. The API orchestrates: validate MCQ → call retrieval (over an injectable `search(query, top_k)` interface) → call a model backend (over an injectable `generate(messages)` interface) → parse the answer → return structured response + record a drift row. Heavy runtime adapters (the FAISS/encoder-backed retrieval, real HTTP calls) sit behind interfaces so unit tests use deterministic fakes and `httpx.MockTransport` — never real models or network. The one heavy adapter (`KGRetrieval`, which imports the existing `baseline` tool) is isolated in its own module and excluded from coverage.

**Tech Stack:** Python 3.12, FastAPI, Pydantic v2, httpx (with `MockTransport` for tests), prometheus-client, pytest + pytest-cov. Sync tests via `fastapi.testclient.TestClient`. Heavy ML deps (torch/faiss/sentence-transformers) are an optional `[runtime]` extra used only by the retrieval container, never in unit tests.

**Spec reference:** repo-root `MLOPS_IMPLEMENTATION_PLAN.md` sections 5.1, 5.2, 5.6 (resolved decisions #1, #4, #7, #9), 6, 11 (P0 tier).

**Workspace convention (per user):** ALL new code AND this plan live under `mlops-platform/`. The package root is `mlops-platform/` (its own `pyproject.toml`, `Makefile`, `.venv`, `src/`, `tests/`). Run `make`/`pytest` from inside `mlops-platform/`. The **only** file touched outside `mlops-platform/` is the single backward-compatible line edit in `baseline/scripts/serve/retrieval_tool.py` (decision #1, Task 10). Git commits run from repo root `/home/vcsai/minhlbq`.

**Scope note:** This is plan 1 of 4 for P0. It delivers the runtime package + retrieval service code (`make install && make test` green, ≥80% coverage; local `/predict` works with mock backend + fixture retrieval). It does **not** include DVC/MLflow/Makefile-pipeline (plan 2), Docker/Helm/KServe/CI (plan 3), or GKE/RunPod/observability-deploy (plan 4).

---

## File Structure

All paths below are relative to repo root `/home/vcsai/minhlbq`. Package source lives under `mlops-platform/src/`.

**Create (all under `mlops-platform/`):**
- `mlops-platform/pyproject.toml` — package metadata, deps, dev/runtime extras, pytest + coverage config.
- `mlops-platform/Makefile` — `install` and `test` targets (full DVC/MLflow targets are plan 2).
- `mlops-platform/src/medical_qa_platform/__init__.py` — package version marker.
- `mlops-platform/src/medical_qa_platform/config.py` — `Settings` dataclass read from env.
- `mlops-platform/src/medical_qa_platform/api/__init__.py`
- `mlops-platform/src/medical_qa_platform/api/parser.py` — `parse_answer()`.
- `mlops-platform/src/medical_qa_platform/api/prompt.py` — `build_prompt()` + `SYSTEM_PROMPT`.
- `mlops-platform/src/medical_qa_platform/api/schemas.py` — `PredictRequest`, `PredictResponse`.
- `mlops-platform/src/medical_qa_platform/api/app.py` — `create_app()`, the FastAPI pre/post API.
- `mlops-platform/src/medical_qa_platform/inference/__init__.py` — `get_backend()` factory.
- `mlops-platform/src/medical_qa_platform/inference/base.py` — `ModelBackend` ABC.
- `mlops-platform/src/medical_qa_platform/inference/mock_backend.py` — `MockBackend`.
- `mlops-platform/src/medical_qa_platform/inference/runpod_backend.py` — `RunpodBackend`.
- `mlops-platform/src/medical_qa_platform/inference/kserve_backend.py` — `KServeBackend`.
- `mlops-platform/src/medical_qa_platform/retrieval/__init__.py`
- `mlops-platform/src/medical_qa_platform/retrieval/device.py` — `resolve_device()`.
- `mlops-platform/src/medical_qa_platform/retrieval/backends.py` — `RetrievalBackend` ABC, `SupportsSearch` Protocol, `FixtureRetrieval`.
- `mlops-platform/src/medical_qa_platform/retrieval/kg_backend.py` — `KGRetrieval` (heavy; coverage-omitted).
- `mlops-platform/src/medical_qa_platform/retrieval/client.py` — `RetrievalClient` (HTTP, injectable transport).
- `mlops-platform/src/medical_qa_platform/retrieval/service.py` — `create_retrieval_service()`, the standalone retrieval FastAPI app.
- `mlops-platform/src/medical_qa_platform/observability/__init__.py`
- `mlops-platform/src/medical_qa_platform/observability/metrics.py` — Prometheus metrics + `render_metrics()`.
- `mlops-platform/src/medical_qa_platform/observability/logging.py` — `configure_logging()`, `get_logger()`.
- `mlops-platform/src/medical_qa_platform/drift/__init__.py`
- `mlops-platform/src/medical_qa_platform/drift/collector.py` — `DriftCollector`.
- `mlops-platform/tests/...` — mirrors the package (one test module per source module).

**Modify (the one allowed exception outside `mlops-platform/`):**
- `baseline/scripts/serve/retrieval_tool.py:55,64-66` — make encoder device configurable (decision #1), default stays `cuda`.

**Deferred to later plans:** `observability/tracing.py` (P2), real-model prompt reconciliation (plan 4), Docker/Helm/CI, PVC+`dvc pull` encoder delivery (plan 3 — `HF_HOME` points at the PVC; nothing baked).

---

## Task 1: Project scaffolding & tooling

**Files:**
- Create: `mlops-platform/pyproject.toml`
- Create: `mlops-platform/Makefile`
- Create: `mlops-platform/src/medical_qa_platform/__init__.py`
- Create: `mlops-platform/src/medical_qa_platform/{api,inference,retrieval,observability,drift}/__init__.py`
- Test: `mlops-platform/tests/test_package_imports.py`

- [ ] **Step 1: Write `mlops-platform/pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "medical_qa_platform"
version = "0.1.0"
description = "Medical QA MLOps inference platform"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.110",
    "uvicorn>=0.29",
    "pydantic>=2.6",
    "httpx>=0.27",
    "prometheus-client>=0.20",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
]
# Heavy adapters used only by the retrieval container; never in unit tests.
runtime = [
    "faiss-cpu>=1.8",
    "sentence-transformers>=2.7",
    "numpy>=1.26",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"

[tool.coverage.run]
source = ["medical_qa_platform"]
omit = ["*/kg_backend.py", "*/__main__.py"]

[tool.coverage.report]
exclude_lines = ["pragma: no cover", "if __name__", "raise NotImplementedError"]
```

- [ ] **Step 2: Write `mlops-platform/Makefile`** (recipe lines are TAB-indented, not spaces)

```makefile
PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: install test

.venv:
	python3.12 -m venv .venv
	$(PIP) install -U pip

install: .venv
	$(PIP) install -e ".[dev]"

test:
	.venv/bin/pytest --cov=medical_qa_platform --cov-report=term-missing
```

- [ ] **Step 3: Create the package `__init__.py` files**

`mlops-platform/src/medical_qa_platform/__init__.py`:

```python
"""Medical QA MLOps inference platform."""

__version__ = "0.1.0"
```

`mlops-platform/src/medical_qa_platform/api/__init__.py`:

```python
"""FastAPI pre/post-processing API."""
```

`mlops-platform/src/medical_qa_platform/inference/__init__.py` — placeholder (replaced with the real factory in Task 5; needed now so the package imports):

```python
"""Model backend abstraction."""
```

`mlops-platform/src/medical_qa_platform/retrieval/__init__.py`:

```python
"""Retrieval client + service."""
```

`mlops-platform/src/medical_qa_platform/observability/__init__.py`:

```python
"""Metrics and logging."""
```

`mlops-platform/src/medical_qa_platform/drift/__init__.py`:

```python
"""Drift-row collection."""
```

- [ ] **Step 4: Write the smoke test** `mlops-platform/tests/test_package_imports.py`

```python
def test_package_imports():
    import medical_qa_platform

    assert medical_qa_platform.__version__ == "0.1.0"
```

- [ ] **Step 5: Install and run the smoke test** (from inside `mlops-platform/`)

Run:
```bash
cd /home/vcsai/minhlbq/mlops-platform && make install && .venv/bin/pytest tests/test_package_imports.py -v
```
Expected: install succeeds; `test_package_imports PASSED`.

- [ ] **Step 6: Commit** (from repo root)

```bash
cd /home/vcsai/minhlbq
printf '.venv/\n*.egg-info/\n__pycache__/\n.coverage\n' > mlops-platform/.gitignore
git add mlops-platform/pyproject.toml mlops-platform/Makefile \
        mlops-platform/src/medical_qa_platform mlops-platform/.gitignore \
        mlops-platform/tests/test_package_imports.py
git commit -m "chore: scaffold medical_qa_platform package"
```

---

## Task 2: Answer parser

**Files:**
- Create: `mlops-platform/src/medical_qa_platform/api/parser.py`
- Test: `mlops-platform/tests/api/test_parser.py`

- [ ] **Step 1: Write the failing tests** `mlops-platform/tests/api/test_parser.py`

```python
from medical_qa_platform.api.parser import parse_answer


def test_parses_answer_tag():
    assert parse_answer("<think>x</think><answer>B</answer>") == "B"


def test_parses_lowercase_and_strips():
    assert parse_answer("blah <answer> c </answer> blah") == "C"


def test_fallback_answer_is_phrase():
    assert parse_answer("The answer is D.") == "D"


def test_fallback_answer_colon():
    assert parse_answer("Answer: A") == "A"


def test_returns_none_when_absent():
    assert parse_answer("no letter here") is None


def test_respects_valid_letters():
    assert parse_answer("<answer>F</answer>", valid_letters={"A", "B"}) is None


def test_prefers_tag_over_fallback_phrase():
    assert parse_answer("The answer is A <answer>B</answer>") == "B"
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/api/test_parser.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'medical_qa_platform.api.parser'`

- [ ] **Step 3: Implement** `mlops-platform/src/medical_qa_platform/api/parser.py`

```python
"""Extract the chosen option letter from raw model output."""

import re

_TAG_RE = re.compile(r"<answer>\s*([A-Za-z])\s*</answer>", re.IGNORECASE)
_PHRASE_RE = re.compile(
    r"(?:the\s+answer\s+is|answer\s*:)\s*\(?([A-Za-z])\)?",
    re.IGNORECASE,
)


def parse_answer(text: str, valid_letters: set[str] | None = None) -> str | None:
    """Return the answer letter (uppercase) or None.

    Tries the canonical ``<answer>X</answer>`` tag first, then a loose
    "The answer is X" / "Answer: X" fallback. If ``valid_letters`` is given,
    a parsed letter outside that set is rejected (returns None).
    """
    letter = None
    match = _TAG_RE.search(text)
    if match is None:
        match = _PHRASE_RE.search(text)
    if match is not None:
        letter = match.group(1).upper()
    if letter is not None and valid_letters is not None and letter not in valid_letters:
        return None
    return letter
```

- [ ] **Step 4: Run to verify pass**

Run: `cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/api/test_parser.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/vcsai/minhlbq
git add mlops-platform/src/medical_qa_platform/api/parser.py mlops-platform/tests/api/test_parser.py
git commit -m "feat: answer parser for model output"
```

---

## Task 3: Prompt builder

**Files:**
- Create: `mlops-platform/src/medical_qa_platform/api/prompt.py`
- Test: `mlops-platform/tests/api/test_prompt.py`

- [ ] **Step 1: Write the failing tests** `mlops-platform/tests/api/test_prompt.py`

```python
from medical_qa_platform.api.prompt import build_prompt, SYSTEM_PROMPT


def test_returns_system_and_user_messages():
    msgs = build_prompt("Q?", {"A": "opt a", "B": "opt b"}, [])
    assert msgs[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert msgs[1]["role"] == "user"
    assert len(msgs) == 2


def test_user_message_contains_question_and_options():
    msgs = build_prompt("What is X?", {"A": "alpha", "B": "beta"}, [])
    user = msgs[1]["content"]
    assert "What is X?" in user
    assert "A. alpha" in user
    assert "B. beta" in user


def test_evidence_block_included_when_present():
    msgs = build_prompt("Q?", {"A": "a", "B": "b"}, ["fact one", "fact two"])
    user = msgs[1]["content"]
    assert "fact one" in user
    assert "fact two" in user


def test_no_evidence_block_when_empty():
    msgs = build_prompt("Q?", {"A": "a", "B": "b"}, [])
    assert "Evidence" not in msgs[1]["content"]


def test_options_sorted_by_letter():
    msgs = build_prompt("Q?", {"B": "beta", "A": "alpha"}, [])
    user = msgs[1]["content"]
    assert user.index("A. alpha") < user.index("B. beta")
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/api/test_prompt.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement** `mlops-platform/src/medical_qa_platform/api/prompt.py`

```python
"""Build chat messages for the medical MCQ task.

NOTE: SYSTEM_PROMPT must be reconciled with the canonical eval prompt in
baseline/scripts/benchmark/grpo_eval/grpo_eval.py before the real-model demo
(plan 4). For mock/local testing the exact wording is not asserted.
"""

SYSTEM_PROMPT = (
    "You are a medical question-answering assistant. Read the question and "
    "options, reason step by step inside <think></think>, then give the single "
    "best option letter inside <answer></answer>."
)


def build_prompt(
    question: str,
    options: dict[str, str],
    evidence: list[str],
) -> list[dict]:
    """Return OpenAI-style chat messages for the given MCQ."""
    lines = []
    if evidence:
        lines.append("Evidence:")
        lines.extend(f"- {e}" for e in evidence)
        lines.append("")
    lines.append(f"Question: {question}")
    for letter in sorted(options):
        lines.append(f"{letter}. {options[letter]}")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]
```

- [ ] **Step 4: Run to verify pass**

Run: `cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/api/test_prompt.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/vcsai/minhlbq
git add mlops-platform/src/medical_qa_platform/api/prompt.py mlops-platform/tests/api/test_prompt.py
git commit -m "feat: MCQ prompt builder"
```

---

## Task 4: Request/response schemas

**Files:**
- Create: `mlops-platform/src/medical_qa_platform/api/schemas.py`
- Test: `mlops-platform/tests/api/test_schemas.py`

- [ ] **Step 1: Write the failing tests** `mlops-platform/tests/api/test_schemas.py`

```python
import pytest
from pydantic import ValidationError

from medical_qa_platform.api.schemas import PredictRequest, PredictResponse


def test_valid_request():
    req = PredictRequest(question="Q?", options={"A": "a", "B": "b"})
    assert req.options["A"] == "a"


def test_rejects_single_option():
    with pytest.raises(ValidationError):
        PredictRequest(question="Q?", options={"A": "a"})


def test_rejects_non_letter_key():
    with pytest.raises(ValidationError):
        PredictRequest(question="Q?", options={"A": "a", "1": "b"})


def test_rejects_multichar_key():
    with pytest.raises(ValidationError):
        PredictRequest(question="Q?", options={"A": "a", "AB": "b"})


def test_rejects_empty_question():
    with pytest.raises(ValidationError):
        PredictRequest(question="   ", options={"A": "a", "B": "b"})


def test_rejects_too_many_options():
    opts = {chr(ord("A") + i): str(i) for i in range(11)}  # 11 options
    with pytest.raises(ValidationError):
        PredictRequest(question="Q?", options=opts)


def test_response_round_trip():
    resp = PredictResponse(
        answer="A",
        evidence=["e1"],
        backend="mock",
        model_version="dev",
        latency_ms=12.5,
        trace_id="abc",
    )
    assert resp.model_dump()["answer"] == "A"
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/api/test_schemas.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement** `mlops-platform/src/medical_qa_platform/api/schemas.py`

```python
"""Pydantic request/response models for the predict API."""

import re

from pydantic import BaseModel, Field, field_validator

_LETTER_RE = re.compile(r"^[A-Z]$")


class PredictRequest(BaseModel):
    question: str = Field(min_length=1)
    options: dict[str, str]

    @field_validator("question")
    @classmethod
    def _question_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be blank")
        return v

    @field_validator("options")
    @classmethod
    def _validate_options(cls, v: dict[str, str]) -> dict[str, str]:
        if not (2 <= len(v) <= 10):
            raise ValueError("options must have between 2 and 10 entries")
        for key, val in v.items():
            if not _LETTER_RE.match(key):
                raise ValueError(f"option key {key!r} must be a single A-Z letter")
            if not val.strip():
                raise ValueError(f"option {key} must not be blank")
        return v


class PredictResponse(BaseModel):
    answer: str | None
    evidence: list[str]
    backend: str
    model_version: str
    latency_ms: float
    trace_id: str
```

- [ ] **Step 4: Run to verify pass**

Run: `cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/api/test_schemas.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/vcsai/minhlbq
git add mlops-platform/src/medical_qa_platform/api/schemas.py mlops-platform/tests/api/test_schemas.py
git commit -m "feat: predict request/response schemas with validation"
```

---

## Task 5: Backend base, mock backend, and factory

**Files:**
- Create: `mlops-platform/src/medical_qa_platform/inference/base.py`
- Create: `mlops-platform/src/medical_qa_platform/inference/mock_backend.py`
- Modify: `mlops-platform/src/medical_qa_platform/inference/__init__.py`
- Test: `mlops-platform/tests/inference/test_mock_backend.py`

- [ ] **Step 1: Write the failing tests** `mlops-platform/tests/inference/test_mock_backend.py`

```python
import pytest

from medical_qa_platform.inference import get_backend
from medical_qa_platform.inference.base import ModelBackend
from medical_qa_platform.inference.mock_backend import MockBackend


def test_mock_is_a_model_backend():
    assert isinstance(MockBackend(), ModelBackend)


def test_mock_name():
    assert MockBackend().name == "mock"


def test_mock_generate_is_deterministic():
    backend = MockBackend(answer="C")
    out1 = backend.generate([{"role": "user", "content": "x"}])
    out2 = backend.generate([{"role": "user", "content": "x"}])
    assert out1 == out2
    assert "<answer>C</answer>" in out1


def test_factory_returns_mock_by_name():
    assert isinstance(get_backend("mock"), MockBackend)


def test_factory_reads_env(monkeypatch):
    monkeypatch.setenv("MODEL_BACKEND", "mock")
    assert isinstance(get_backend(), MockBackend)


def test_factory_unknown_raises():
    with pytest.raises(ValueError):
        get_backend("nope")
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/inference/test_mock_backend.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement** `mlops-platform/src/medical_qa_platform/inference/base.py`

```python
"""Model backend interface."""

from abc import ABC, abstractmethod


class ModelBackend(ABC):
    """A backend that turns chat messages into raw model text."""

    name: str = "base"

    @abstractmethod
    def generate(
        self,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> str:
        """Return the raw text completion for the given chat messages."""
        raise NotImplementedError
```

- [ ] **Step 4: Implement** `mlops-platform/src/medical_qa_platform/inference/mock_backend.py`

```python
"""Deterministic backend for tests and CI (no GPU, no network)."""

from .base import ModelBackend


class MockBackend(ModelBackend):
    name = "mock"

    def __init__(self, answer: str = "A"):
        self._answer = answer.upper()

    def generate(
        self,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> str:
        return (
            f"<think>Mock reasoning for {len(messages)} messages.</think>"
            f"<answer>{self._answer}</answer>"
        )
```

- [ ] **Step 5: Replace** `mlops-platform/src/medical_qa_platform/inference/__init__.py` with the factory

```python
"""Model backend abstraction."""

import os

from .base import ModelBackend


def get_backend(name: str | None = None) -> ModelBackend:
    """Construct a backend by name (defaults to env MODEL_BACKEND, then 'mock')."""
    name = (name or os.environ.get("MODEL_BACKEND", "mock")).lower()
    if name == "mock":
        from .mock_backend import MockBackend

        return MockBackend()
    if name == "runpod":
        from .runpod_backend import RunpodBackend

        return RunpodBackend.from_env()
    if name == "kserve":
        from .kserve_backend import KServeBackend

        return KServeBackend.from_env()
    raise ValueError(f"unknown MODEL_BACKEND: {name!r}")
```

Note: the `runpod`/`kserve` branches reference classes added in Tasks 6 and 7 and are imported lazily, so the module imports cleanly now; only the `mock` branch is exercised by this task's tests.

- [ ] **Step 6: Run to verify pass**

Run: `cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/inference/test_mock_backend.py -v`
Expected: 6 passed.

- [ ] **Step 7: Commit**

```bash
cd /home/vcsai/minhlbq
git add mlops-platform/src/medical_qa_platform/inference/base.py \
        mlops-platform/src/medical_qa_platform/inference/mock_backend.py \
        mlops-platform/src/medical_qa_platform/inference/__init__.py \
        mlops-platform/tests/inference/test_mock_backend.py
git commit -m "feat: model backend base, mock backend, factory"
```

---

## Task 6: RunPod backend (OpenAI-compatible)

**Files:**
- Create: `mlops-platform/src/medical_qa_platform/inference/runpod_backend.py`
- Test: `mlops-platform/tests/inference/test_runpod_backend.py`

- [ ] **Step 1: Write the failing tests** `mlops-platform/tests/inference/test_runpod_backend.py`

```python
import json

import httpx

from medical_qa_platform.inference.runpod_backend import RunpodBackend


def _backend(captured: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "<answer>B</answer>"}}]},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return RunpodBackend(
        base_url="http://pod/v1", model="m", api_key="k", client=client
    )


def test_name():
    assert _backend({}).name == "runpod"


def test_generate_returns_content():
    out = _backend({}).generate([{"role": "user", "content": "x"}])
    assert out == "<answer>B</answer>"


def test_posts_to_chat_completions_with_auth_and_model():
    captured = {}
    _backend(captured).generate([{"role": "user", "content": "x"}])
    assert captured["url"] == "http://pod/v1/chat/completions"
    assert captured["headers"]["authorization"] == "Bearer k"
    assert captured["body"]["model"] == "m"
    assert captured["body"]["messages"] == [{"role": "user", "content": "x"}]


def test_from_env(monkeypatch):
    monkeypatch.setenv("RUNPOD_BASE_URL", "http://x/v1")
    monkeypatch.setenv("RUNPOD_MODEL", "qwen")
    monkeypatch.setenv("RUNPOD_API_KEY", "secret")
    backend = RunpodBackend.from_env()
    assert backend.base_url == "http://x/v1"
    assert backend.model == "qwen"
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/inference/test_runpod_backend.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement** `mlops-platform/src/medical_qa_platform/inference/runpod_backend.py`

```python
"""RunPod vLLM backend via the OpenAI-compatible chat-completions API."""

import os

import httpx

from .base import ModelBackend


class RunpodBackend(ModelBackend):
    name = "runpod"

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
    def from_env(cls) -> "RunpodBackend":
        return cls(
            base_url=os.environ.get("RUNPOD_BASE_URL", ""),
            model=os.environ.get("RUNPOD_MODEL", ""),
            api_key=os.environ.get("RUNPOD_API_KEY", ""),
        )

    def generate(
        self,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> str:
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        resp = self._client.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
```

- [ ] **Step 4: Run to verify pass**

Run: `cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/inference/test_runpod_backend.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/vcsai/minhlbq
git add mlops-platform/src/medical_qa_platform/inference/runpod_backend.py \
        mlops-platform/tests/inference/test_runpod_backend.py
git commit -m "feat: runpod OpenAI-compatible backend"
```

---

## Task 7: KServe backend

**Files:**
- Create: `mlops-platform/src/medical_qa_platform/inference/kserve_backend.py`
- Test: `mlops-platform/tests/inference/test_kserve_backend.py`

KServe contract for our CPU mock model: request `{"instances": [{"messages": [...]}]}`, response `{"predictions": [{"text": "..."}]}`.

- [ ] **Step 1: Write the failing tests** `mlops-platform/tests/inference/test_kserve_backend.py`

```python
import json

import httpx

from medical_qa_platform.inference.kserve_backend import KServeBackend


def _backend(captured: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"predictions": [{"text": "<answer>A</answer>"}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return KServeBackend(url="http://ks/v1/models/mock:predict", client=client)


def test_name():
    assert _backend({}).name == "kserve"


def test_generate_returns_text():
    out = _backend({}).generate([{"role": "user", "content": "x"}])
    assert out == "<answer>A</answer>"


def test_posts_instances_payload():
    captured = {}
    _backend(captured).generate([{"role": "user", "content": "x"}])
    assert captured["url"] == "http://ks/v1/models/mock:predict"
    assert captured["body"]["instances"][0]["messages"] == [
        {"role": "user", "content": "x"}
    ]


def test_from_env(monkeypatch):
    monkeypatch.setenv("KSERVE_URL", "http://ks/v1/models/m:predict")
    assert KServeBackend.from_env().url == "http://ks/v1/models/m:predict"
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/inference/test_kserve_backend.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement** `mlops-platform/src/medical_qa_platform/inference/kserve_backend.py`

```python
"""KServe CPU smoke/mock backend (v1 predict protocol)."""

import os

import httpx

from .base import ModelBackend


class KServeBackend(ModelBackend):
    name = "kserve"

    def __init__(
        self,
        url: str,
        client: httpx.Client | None = None,
        timeout: float = 30.0,
    ):
        self.url = url
        self._client = client or httpx.Client(timeout=timeout)

    @classmethod
    def from_env(cls) -> "KServeBackend":
        return cls(url=os.environ.get("KSERVE_URL", ""))

    def generate(
        self,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> str:
        resp = self._client.post(
            self.url,
            json={"instances": [{"messages": messages}]},
        )
        resp.raise_for_status()
        return resp.json()["predictions"][0]["text"]
```

- [ ] **Step 4: Run to verify pass**

Run: `cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/inference/test_kserve_backend.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/vcsai/minhlbq
git add mlops-platform/src/medical_qa_platform/inference/kserve_backend.py \
        mlops-platform/tests/inference/test_kserve_backend.py
git commit -m "feat: kserve predict backend"
```

---

## Task 8: Retrieval device resolver + backends

**Files:**
- Create: `mlops-platform/src/medical_qa_platform/retrieval/device.py`
- Create: `mlops-platform/src/medical_qa_platform/retrieval/backends.py`
- Test: `mlops-platform/tests/retrieval/test_device.py`
- Test: `mlops-platform/tests/retrieval/test_backends.py`

- [ ] **Step 1: Write the failing tests** `mlops-platform/tests/retrieval/test_device.py`

```python
from medical_qa_platform.retrieval.device import resolve_device


def test_defaults_to_cpu(monkeypatch):
    monkeypatch.delenv("RETRIEVAL_DEVICE", raising=False)
    assert resolve_device() == "cpu"


def test_reads_env(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_DEVICE", "cuda")
    assert resolve_device() == "cuda"


def test_explicit_arg_wins(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_DEVICE", "cuda")
    assert resolve_device("cpu") == "cpu"
```

- [ ] **Step 2: Write the failing tests** `mlops-platform/tests/retrieval/test_backends.py`

```python
from medical_qa_platform.retrieval.backends import FixtureRetrieval, RetrievalBackend


def test_fixture_is_a_retrieval_backend():
    assert isinstance(FixtureRetrieval({}), RetrievalBackend)


def test_fixture_returns_canned_results():
    fx = FixtureRetrieval({"diabetes": ["metformin treats diabetes"]})
    assert fx.search("diabetes", top_k=5) == ["metformin treats diabetes"]


def test_fixture_truncates_to_top_k():
    fx = FixtureRetrieval({"q": ["a", "b", "c"]})
    assert fx.search("q", top_k=2) == ["a", "b"]


def test_fixture_unknown_query_returns_empty():
    assert FixtureRetrieval({"q": ["a"]}).search("other", top_k=5) == []
```

- [ ] **Step 3: Run to verify failure**

Run: `cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/retrieval/test_device.py tests/retrieval/test_backends.py -v`
Expected: FAIL — modules not found.

- [ ] **Step 4: Implement** `mlops-platform/src/medical_qa_platform/retrieval/device.py`

```python
"""Resolve the compute device for the retrieval encoder (decision #1)."""

import os


def resolve_device(explicit: str | None = None) -> str:
    """Return the device string. Defaults to CPU for the GKE retrieval service.

    Precedence: explicit arg > RETRIEVAL_DEVICE env > "cpu".
    """
    if explicit:
        return explicit
    return os.environ.get("RETRIEVAL_DEVICE", "cpu")
```

- [ ] **Step 5: Implement** `mlops-platform/src/medical_qa_platform/retrieval/backends.py`

```python
"""Retrieval backend interface + a deterministic fixture for tests."""

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable


@runtime_checkable
class SupportsSearch(Protocol):
    """Anything the API can call to fetch evidence (backend or HTTP client)."""

    def search(self, query: str, top_k: int) -> list[str]: ...


class RetrievalBackend(ABC):
    @abstractmethod
    def search(self, query: str, top_k: int) -> list[str]:
        raise NotImplementedError


class FixtureRetrieval(RetrievalBackend):
    """In-memory query→results map. No FAISS, no encoder, fully deterministic."""

    def __init__(self, data: dict[str, list[str]]):
        self._data = data

    def search(self, query: str, top_k: int) -> list[str]:
        return list(self._data.get(query, []))[:top_k]
```

- [ ] **Step 6: Run to verify pass**

Run: `cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/retrieval/test_device.py tests/retrieval/test_backends.py -v`
Expected: 7 passed.

- [ ] **Step 7: Commit**

```bash
cd /home/vcsai/minhlbq
git add mlops-platform/src/medical_qa_platform/retrieval/device.py \
        mlops-platform/src/medical_qa_platform/retrieval/backends.py \
        mlops-platform/tests/retrieval/test_device.py \
        mlops-platform/tests/retrieval/test_backends.py
git commit -m "feat: retrieval device resolver and fixture backend"
```

---

## Task 9: KGRetrieval adapter (heavy; coverage-omitted)

**Files:**
- Create: `mlops-platform/src/medical_qa_platform/retrieval/kg_backend.py`
- Test: `mlops-platform/tests/retrieval/test_kg_backend_import.py`

This module wraps the existing `baseline` tool. It imports torch/faiss/sentence-transformers lazily and is excluded from coverage (`omit = ["*/kg_backend.py"]`). The only unit test confirms the symbol exists without importing the heavy deps. It passes `device=resolve_device()` (defaults to CPU) into the baseline `load()` whose new `device` kwarg is added in Task 10.

- [ ] **Step 1: Write the failing test** `mlops-platform/tests/retrieval/test_kg_backend_import.py`

```python
def test_kg_backend_symbol_exists():
    from medical_qa_platform.retrieval import kg_backend

    assert hasattr(kg_backend, "KGRetrieval")
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/retrieval/test_kg_backend_import.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement** `mlops-platform/src/medical_qa_platform/retrieval/kg_backend.py`

```python
"""Production retrieval backed by the existing baseline KG tool (decision #1, #2).

Heavy: imports the baseline MedicalKnowledgeTool (faiss + sentence-transformers).
Used only inside the retrieval container, never in unit tests. Excluded from
coverage via pyproject `omit`. The encoder runs on CPU by default (decision #1);
KG artifacts and encoder weights are delivered to the container via a PVC
populated by `dvc pull` (decision #2) with HF_HOME pointing at that PVC cache —
nothing is baked into the image.
"""

import os
import sys

from .backends import RetrievalBackend
from .device import resolve_device


class KGRetrieval(RetrievalBackend):  # pragma: no cover
    def __init__(self, data_dir: str | None = None, device: str | None = None):
        self.data_dir = data_dir or os.environ.get("KG_DATA_DIR", "data/")
        self.device = resolve_device(device)
        baseline_root = os.environ.get(
            "BASELINE_ROOT", "/home/vcsai/minhlbq/baseline"
        )
        if baseline_root not in sys.path:
            sys.path.insert(0, baseline_root)
        from scripts.serve.retrieval_tool import MedicalKnowledgeTool

        self._tool = MedicalKnowledgeTool.load(self.data_dir, device=self.device)

    def search(self, query: str, top_k: int) -> list[str]:
        return self._tool.retrieve(query, top_k)
```

- [ ] **Step 4: Run to verify pass**

Run: `cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/retrieval/test_kg_backend_import.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/vcsai/minhlbq
git add mlops-platform/src/medical_qa_platform/retrieval/kg_backend.py \
        mlops-platform/tests/retrieval/test_kg_backend_import.py
git commit -m "feat: KGRetrieval adapter over baseline tool"
```

---

## Task 10: Make baseline encoder device configurable (decision #1)

**Files:**
- Modify: `baseline/scripts/serve/retrieval_tool.py` — the `load` classmethod signature (line 55) and encoder construction (lines 64-66)

This is the ONLY file edited outside `mlops-platform/`. Backward compatible: default stays `cuda`, so all 10 existing callers (`grpo_train*.py`, `grpo_eval.py`, `retrieval_api.py`, `eval_sft.py`, `gen_data_groq.py`, `test_infer.py`, module-level) are unaffected. It cannot be unit-tested without sentence-transformers, so it is verified by reading the edit back.

- [ ] **Step 1: Read the current `load` signature and encoder lines**

Run: `cd /home/vcsai/minhlbq && sed -n '55p;64,66p' baseline/scripts/serve/retrieval_tool.py`
Expected: shows `def load(cls, data_dir: str = "data/") -> "MedicalKnowledgeTool":` and the `SentenceTransformer('abhinand/MedEmbed-large-v0.1', device='cuda')` construction.

- [ ] **Step 2: Edit the signature** — replace

```python
    def load(cls, data_dir: str = "data/") -> "MedicalKnowledgeTool":
```

with

```python
    def load(cls, data_dir: str = "data/", device: str = "cuda") -> "MedicalKnowledgeTool":
```

- [ ] **Step 3: Edit the encoder construction** — replace

```python
        inst.encoder = SentenceTransformer(
            'abhinand/MedEmbed-large-v0.1', device='cuda'
        )
```

with

```python
        inst.encoder = SentenceTransformer(
            'abhinand/MedEmbed-large-v0.1', device=device
        )
```

- [ ] **Step 4: Verify the edit by reading it back**

Run: `cd /home/vcsai/minhlbq && grep -n 'def load\|device=device' baseline/scripts/serve/retrieval_tool.py`
Expected: shows the new `device: str = "cuda"` signature and `device=device` in the encoder call.

- [ ] **Step 5: Confirm no existing caller breaks**

Run: `cd /home/vcsai/minhlbq && grep -rn 'MedicalKnowledgeTool.load(' baseline/scripts | grep -v 'retrieval_tool.py'`
Expected: all callers pass only `data_dir` (positional or keyword) or nothing — none pass a 2nd positional arg, so the new optional `device` keeps them valid.

- [ ] **Step 6: Commit**

```bash
cd /home/vcsai/minhlbq
git add baseline/scripts/serve/retrieval_tool.py
git commit -m "feat: make retrieval encoder device configurable (default cuda)"
```

---

## Task 11: Retrieval HTTP client

**Files:**
- Create: `mlops-platform/src/medical_qa_platform/retrieval/client.py`
- Test: `mlops-platform/tests/retrieval/test_client.py`

- [ ] **Step 1: Write the failing tests** `mlops-platform/tests/retrieval/test_client.py`

```python
import json

import httpx

from medical_qa_platform.retrieval.client import RetrievalClient


def _client(captured: dict, results):
    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200, json=[{"query": captured["body"]["queries"][0], "results": results}]
        )

    http = httpx.Client(transport=httpx.MockTransport(handler))
    return RetrievalClient(base_url="http://ret:8001", client=http)


def test_search_returns_results_list():
    out = _client({}, ["evidence a", "evidence b"]).search("diabetes", top_k=5)
    assert out == ["evidence a", "evidence b"]


def test_search_posts_expected_body():
    captured = {}
    _client(captured, []).search("q", top_k=3)
    assert captured["url"] == "http://ret:8001/search"
    assert captured["body"] == {"queries": ["q"], "top_k": 3}


def test_search_empty_response_returns_empty():
    def handler(request):
        return httpx.Response(200, json=[])

    http = httpx.Client(transport=httpx.MockTransport(handler))
    client = RetrievalClient(base_url="http://ret:8001", client=http)
    assert client.search("q", top_k=5) == []
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/retrieval/test_client.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement** `mlops-platform/src/medical_qa_platform/retrieval/client.py`

```python
"""HTTP client the API uses to call the retrieval service /search endpoint."""

import os

import httpx


class RetrievalClient:
    def __init__(
        self,
        base_url: str,
        client: httpx.Client | None = None,
        timeout: float = 10.0,
    ):
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=timeout)

    @classmethod
    def from_env(cls) -> "RetrievalClient":
        return cls(base_url=os.environ.get("RETRIEVAL_URL", "http://localhost:8001"))

    def search(self, query: str, top_k: int) -> list[str]:
        resp = self._client.post(
            f"{self.base_url}/search",
            json={"queries": [query], "top_k": top_k},
        )
        resp.raise_for_status()
        payload = resp.json()
        if not payload:
            return []
        return payload[0]["results"]
```

- [ ] **Step 4: Run to verify pass**

Run: `cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/retrieval/test_client.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/vcsai/minhlbq
git add mlops-platform/src/medical_qa_platform/retrieval/client.py \
        mlops-platform/tests/retrieval/test_client.py
git commit -m "feat: retrieval HTTP client"
```

---

## Task 12: Observability — metrics + structured logging

**Files:**
- Create: `mlops-platform/src/medical_qa_platform/observability/metrics.py`
- Create: `mlops-platform/src/medical_qa_platform/observability/logging.py`
- Test: `mlops-platform/tests/observability/test_metrics.py`
- Test: `mlops-platform/tests/observability/test_logging.py`

- [ ] **Step 1: Write the failing tests** `mlops-platform/tests/observability/test_metrics.py`

```python
from medical_qa_platform.observability import metrics


def test_render_returns_text_and_content_type():
    body, content_type = metrics.render_metrics()
    assert isinstance(body, bytes)
    assert "text/plain" in content_type


def test_observe_request_appears_in_render():
    metrics.observe_request(endpoint="/predict", backend="mock", status="ok", latency_s=0.01)
    body, _ = metrics.render_metrics()
    text = body.decode()
    assert "mqa_requests_total" in text
    assert "mqa_request_latency_seconds" in text


def test_observe_retrieval_no_result():
    metrics.observe_retrieval(latency_s=0.02, no_result=True)
    text = metrics.render_metrics()[0].decode()
    assert "mqa_retrieval_no_result_total" in text
```

- [ ] **Step 2: Write the failing tests** `mlops-platform/tests/observability/test_logging.py`

```python
import json
import logging

from medical_qa_platform.observability.logging import JsonFormatter, get_logger


def test_formatter_emits_json():
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="t", level=logging.INFO, pathname=__file__, lineno=1,
        msg="hello", args=(), exc_info=None,
    )
    record.trace_id = "abc"
    parsed = json.loads(formatter.format(record))
    assert parsed["message"] == "hello"
    assert parsed["level"] == "INFO"
    assert parsed["trace_id"] == "abc"


def test_get_logger_returns_logger():
    assert isinstance(get_logger("x"), logging.Logger)
```

- [ ] **Step 3: Run to verify failure**

Run: `cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/observability/ -v`
Expected: FAIL — modules not found.

- [ ] **Step 4: Implement** `mlops-platform/src/medical_qa_platform/observability/metrics.py`

```python
"""Prometheus metrics for the API and retrieval service."""

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

REQUEST_COUNT = Counter(
    "mqa_requests_total",
    "Total prediction requests.",
    ["endpoint", "backend", "status"],
)
REQUEST_LATENCY = Histogram(
    "mqa_request_latency_seconds",
    "Request latency in seconds.",
    ["endpoint"],
)
RETRIEVAL_LATENCY = Histogram(
    "mqa_retrieval_latency_seconds",
    "Retrieval latency in seconds.",
)
RETRIEVAL_NO_RESULT = Counter(
    "mqa_retrieval_no_result_total",
    "Retrieval calls that returned zero evidence.",
)


def observe_request(endpoint: str, backend: str, status: str, latency_s: float) -> None:
    REQUEST_COUNT.labels(endpoint=endpoint, backend=backend, status=status).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency_s)


def observe_retrieval(latency_s: float, no_result: bool) -> None:
    RETRIEVAL_LATENCY.observe(latency_s)
    if no_result:
        RETRIEVAL_NO_RESULT.inc()


def render_metrics() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
```

- [ ] **Step 5: Implement** `mlops-platform/src/medical_qa_platform/observability/logging.py`

```python
"""Structured JSON logging."""

import json
import logging
import sys


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        trace_id = getattr(record, "trace_id", None)
        if trace_id is not None:
            payload["trace_id"] = trace_id
        return json.dumps(payload)


def configure_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
```

- [ ] **Step 6: Run to verify pass**

Run: `cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/observability/ -v`
Expected: 5 passed.

- [ ] **Step 7: Commit**

```bash
cd /home/vcsai/minhlbq
git add mlops-platform/src/medical_qa_platform/observability/metrics.py \
        mlops-platform/src/medical_qa_platform/observability/logging.py \
        mlops-platform/tests/observability/
git commit -m "feat: prometheus metrics and structured logging"
```

---

## Task 13: Drift collector

**Files:**
- Create: `mlops-platform/src/medical_qa_platform/drift/collector.py`
- Test: `mlops-platform/tests/drift/test_collector.py`

Features per decision #7: question token-length, option count, mean option length, predicted answer letter, no-result flag, latency.

- [ ] **Step 1: Write the failing tests** `mlops-platform/tests/drift/test_collector.py`

```python
import json

from medical_qa_platform.api.schemas import PredictRequest, PredictResponse
from medical_qa_platform.drift.collector import DriftCollector


def _req():
    return PredictRequest(question="what treats diabetes now", options={"A": "aa", "B": "bbbb"})


def _resp(answer, evidence):
    return PredictResponse(
        answer=answer, evidence=evidence, backend="mock",
        model_version="dev", latency_ms=10.0, trace_id="t1",
    )


def test_record_computes_features():
    row = DriftCollector(path=None).record(_req(), _resp("A", ["e1"]), n_evidence=1)
    assert row["q_token_len"] == 4
    assert row["n_options"] == 2
    assert row["mean_option_len"] == 3.0  # (2 + 4) / 2
    assert row["answer"] == "A"
    assert row["no_result"] is False
    assert row["latency_ms"] == 10.0


def test_no_result_flag_true_when_zero_evidence():
    row = DriftCollector(path=None).record(_req(), _resp(None, []), n_evidence=0)
    assert row["no_result"] is True
    assert row["answer"] == "none"


def test_record_appends_jsonl(tmp_path):
    path = tmp_path / "drift.jsonl"
    collector = DriftCollector(path=str(path))
    collector.record(_req(), _resp("A", ["e1"]), n_evidence=1)
    collector.record(_req(), _resp("B", []), n_evidence=0)
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["answer"] == "B"
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/drift/test_collector.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement** `mlops-platform/src/medical_qa_platform/drift/collector.py`

```python
"""Collect per-request rows for offline Evidently drift analysis (decision #7)."""

import json

from ..api.schemas import PredictRequest, PredictResponse


class DriftCollector:
    def __init__(self, path: str | None):
        """If path is None, rows are computed but not persisted (test mode)."""
        self.path = path

    def record(
        self,
        request: PredictRequest,
        response: PredictResponse,
        n_evidence: int,
    ) -> dict:
        option_lengths = [len(v) for v in request.options.values()]
        row = {
            "q_token_len": len(request.question.split()),
            "n_options": len(request.options),
            "mean_option_len": sum(option_lengths) / len(option_lengths),
            "answer": response.answer if response.answer is not None else "none",
            "no_result": n_evidence == 0,
            "latency_ms": response.latency_ms,
        }
        if self.path is not None:
            with open(self.path, "a") as f:
                f.write(json.dumps(row) + "\n")
        return row
```

- [ ] **Step 4: Run to verify pass**

Run: `cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/drift/test_collector.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/vcsai/minhlbq
git add mlops-platform/src/medical_qa_platform/drift/collector.py \
        mlops-platform/tests/drift/test_collector.py
git commit -m "feat: drift-row collector"
```

---

## Task 14: Config settings

**Files:**
- Create: `mlops-platform/src/medical_qa_platform/config.py`
- Test: `mlops-platform/tests/test_config.py`

- [ ] **Step 1: Write the failing tests** `mlops-platform/tests/test_config.py`

```python
from medical_qa_platform.config import Settings


def test_defaults(monkeypatch):
    for var in [
        "MODEL_BACKEND", "RETRIEVAL_URL", "MODEL_VERSION", "TOP_K", "DRIFT_LOG_PATH",
    ]:
        monkeypatch.delenv(var, raising=False)
    s = Settings.from_env()
    assert s.model_backend == "mock"
    assert s.retrieval_url == "http://localhost:8001"
    assert s.model_version == "dev"
    assert s.top_k == 5


def test_reads_env(monkeypatch):
    monkeypatch.setenv("MODEL_BACKEND", "runpod")
    monkeypatch.setenv("MODEL_VERSION", "v6.1")
    monkeypatch.setenv("TOP_K", "8")
    s = Settings.from_env()
    assert s.model_backend == "runpod"
    assert s.model_version == "v6.1"
    assert s.top_k == 8
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/test_config.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement** `mlops-platform/src/medical_qa_platform/config.py`

```python
"""Runtime configuration read from environment variables."""

import os
from dataclasses import dataclass


@dataclass
class Settings:
    model_backend: str = "mock"
    retrieval_url: str = "http://localhost:8001"
    model_version: str = "dev"
    top_k: int = 5
    drift_log_path: str = "drift_log.jsonl"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            model_backend=os.environ.get("MODEL_BACKEND", "mock"),
            retrieval_url=os.environ.get("RETRIEVAL_URL", "http://localhost:8001"),
            model_version=os.environ.get("MODEL_VERSION", "dev"),
            top_k=int(os.environ.get("TOP_K", "5")),
            drift_log_path=os.environ.get("DRIFT_LOG_PATH", "drift_log.jsonl"),
        )
```

- [ ] **Step 4: Run to verify pass**

Run: `cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/test_config.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/vcsai/minhlbq
git add mlops-platform/src/medical_qa_platform/config.py mlops-platform/tests/test_config.py
git commit -m "feat: env-based settings"
```

---

## Task 15: Retrieval service app

**Files:**
- Create: `mlops-platform/src/medical_qa_platform/retrieval/service.py`
- Test: `mlops-platform/tests/retrieval/test_service.py`

- [ ] **Step 1: Write the failing tests** `mlops-platform/tests/retrieval/test_service.py`

```python
from fastapi.testclient import TestClient

from medical_qa_platform.retrieval.backends import FixtureRetrieval
from medical_qa_platform.retrieval.service import create_retrieval_service


def _client():
    backend = FixtureRetrieval({"diabetes": ["metformin treats diabetes", "insulin"]})
    app = create_retrieval_service(backend=backend)
    return TestClient(app)


def test_health_ok():
    assert _client().get("/health").json()["status"] == "ok"


def test_ready_ok():
    assert _client().get("/ready").status_code == 200


def test_search_returns_results():
    resp = _client().post("/search", json={"queries": ["diabetes"], "top_k": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert body[0]["query"] == "diabetes"
    assert body[0]["results"] == ["metformin treats diabetes", "insulin"]


def test_search_top_k_upper_bound_rejected():
    resp = _client().post("/search", json={"queries": ["diabetes"], "top_k": 21})
    assert resp.status_code == 422


def test_search_top_k_lower_bound_rejected():
    resp = _client().post("/search", json={"queries": ["diabetes"], "top_k": 0})
    assert resp.status_code == 422


def test_metrics_endpoint():
    resp = _client().get("/metrics")
    assert resp.status_code == 200
    assert "mqa_retrieval" in resp.text
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/retrieval/test_service.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement** `mlops-platform/src/medical_qa_platform/retrieval/service.py`

```python
"""Standalone retrieval FastAPI service (decision #4: lifespan loading)."""

import os
import time

from fastapi import FastAPI, Response
from pydantic import BaseModel, Field

from ..observability.metrics import observe_retrieval, render_metrics
from .backends import SupportsSearch


class SearchRequest(BaseModel):
    queries: list[str]
    top_k: int = Field(default=5, ge=1, le=20)


class SearchResult(BaseModel):
    query: str
    results: list[str]


def _build_backend_from_env() -> SupportsSearch:  # pragma: no cover
    kind = os.environ.get("RETRIEVAL_BACKEND", "kg").lower()
    if kind == "kg":
        from .kg_backend import KGRetrieval

        return KGRetrieval()
    raise ValueError(f"unknown RETRIEVAL_BACKEND: {kind!r}")


def create_retrieval_service(backend: SupportsSearch | None = None) -> FastAPI:
    app = FastAPI(title="Medical Knowledge Retrieval Service")

    @app.on_event("startup")
    def _startup() -> None:
        app.state.backend = backend if backend is not None else _build_backend_from_env()

    @app.post("/search", response_model=list[SearchResult])
    def search(req: SearchRequest):
        out = []
        for q in req.queries:
            t0 = time.perf_counter()
            results = app.state.backend.search(q, req.top_k)
            observe_retrieval(time.perf_counter() - t0, no_result=len(results) == 0)
            out.append(SearchResult(query=q, results=results))
        return out

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready(response: Response):
        if getattr(app.state, "backend", None) is None:
            response.status_code = 503
            return {"status": "not ready"}
        return {"status": "ready"}

    @app.get("/metrics")
    def metrics():
        body, content_type = render_metrics()
        return Response(content=body, media_type=content_type)

    return app
```

- [ ] **Step 4: Run to verify pass**

Run: `cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/retrieval/test_service.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/vcsai/minhlbq
git add mlops-platform/src/medical_qa_platform/retrieval/service.py \
        mlops-platform/tests/retrieval/test_service.py
git commit -m "feat: standalone retrieval service"
```

---

## Task 16: Main predict API app

**Files:**
- Create: `mlops-platform/src/medical_qa_platform/api/app.py`
- Test: `mlops-platform/tests/api/test_app.py`

`create_app()` accepts injected `backend` and `retrieval` (anything matching `SupportsSearch`) so tests use `MockBackend` + `FixtureRetrieval` with no network. If not injected, it builds them from env at startup.

- [ ] **Step 1: Write the failing tests** `mlops-platform/tests/api/test_app.py`

```python
from fastapi.testclient import TestClient

from medical_qa_platform.api.app import create_app
from medical_qa_platform.inference.mock_backend import MockBackend
from medical_qa_platform.retrieval.backends import FixtureRetrieval


def _client(tmp_path):
    app = create_app(
        backend=MockBackend(answer="B"),
        retrieval=FixtureRetrieval({"Q?": ["evidence one", "evidence two"]}),
        model_version="test-v1",
        drift_log_path=str(tmp_path / "drift.jsonl"),
    )
    return TestClient(app)


def test_health(tmp_path):
    assert _client(tmp_path).get("/health").json()["status"] == "ok"


def test_ready(tmp_path):
    assert _client(tmp_path).get("/ready").status_code == 200


def test_predict_returns_all_fields(tmp_path):
    client = _client(tmp_path)
    resp = client.post(
        "/predict", json={"question": "Q?", "options": {"A": "a", "B": "b"}}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "B"
    assert body["evidence"] == ["evidence one", "evidence two"]
    assert body["backend"] == "mock"
    assert body["model_version"] == "test-v1"
    assert body["latency_ms"] >= 0
    assert body["trace_id"]


def test_predict_rejects_invalid_options(tmp_path):
    resp = _client(tmp_path).post(
        "/predict", json={"question": "Q?", "options": {"A": "a"}}
    )
    assert resp.status_code == 422


def test_predict_answer_constrained_to_options(tmp_path):
    # Backend insists on "D" but options are only A/B -> parser rejects -> answer null
    app = create_app(
        backend=MockBackend(answer="D"),
        retrieval=FixtureRetrieval({}),
        model_version="x",
        drift_log_path=str(tmp_path / "d.jsonl"),
    )
    resp = TestClient(app).post(
        "/predict", json={"question": "Q?", "options": {"A": "a", "B": "b"}}
    )
    assert resp.json()["answer"] is None


def test_predict_writes_drift_row(tmp_path):
    path = tmp_path / "drift.jsonl"
    app = create_app(
        backend=MockBackend(answer="A"),
        retrieval=FixtureRetrieval({"Q?": ["e"]}),
        model_version="x",
        drift_log_path=str(path),
    )
    TestClient(app).post(
        "/predict", json={"question": "Q?", "options": {"A": "a", "B": "b"}}
    )
    assert path.exists()
    assert path.read_text().strip()


def test_metrics_endpoint(tmp_path):
    client = _client(tmp_path)
    client.post("/predict", json={"question": "Q?", "options": {"A": "a", "B": "b"}})
    resp = client.get("/metrics")
    assert "mqa_requests_total" in resp.text
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/api/test_app.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement** `mlops-platform/src/medical_qa_platform/api/app.py`

```python
"""FastAPI pre/post-processing API (spec 5.1)."""

import time
import uuid

from fastapi import FastAPI, Response

from ..config import Settings
from ..drift.collector import DriftCollector
from ..observability.metrics import observe_request, render_metrics
from ..retrieval.backends import SupportsSearch
from .parser import parse_answer
from .prompt import build_prompt
from .schemas import PredictRequest, PredictResponse


def create_app(
    backend=None,
    retrieval: SupportsSearch | None = None,
    model_version: str | None = None,
    drift_log_path: str | None = None,
    top_k: int | None = None,
) -> FastAPI:
    settings = Settings.from_env()
    app = FastAPI(title="Medical QA API")
    app.state.top_k = top_k if top_k is not None else settings.top_k
    app.state.model_version = (
        model_version if model_version is not None else settings.model_version
    )
    app.state.collector = DriftCollector(
        drift_log_path if drift_log_path is not None else settings.drift_log_path
    )

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

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready(response: Response):
        if getattr(app.state, "backend", None) is None:
            response.status_code = 503
            return {"status": "not ready"}
        return {"status": "ready"}

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        t0 = time.perf_counter()
        trace_id = uuid.uuid4().hex
        evidence = app.state.retrieval.search(req.question, app.state.top_k)
        messages = build_prompt(req.question, req.options, evidence)
        raw = app.state.backend.generate(messages)
        answer = parse_answer(raw, valid_letters=set(req.options))
        latency_ms = (time.perf_counter() - t0) * 1000.0
        resp = PredictResponse(
            answer=answer,
            evidence=evidence,
            backend=app.state.backend.name,
            model_version=app.state.model_version,
            latency_ms=latency_ms,
            trace_id=trace_id,
        )
        observe_request(
            endpoint="/predict",
            backend=app.state.backend.name,
            status="ok" if answer is not None else "no_answer",
            latency_s=latency_ms / 1000.0,
        )
        app.state.collector.record(req, resp, n_evidence=len(evidence))
        return resp

    @app.get("/metrics")
    def metrics():
        body, content_type = render_metrics()
        return Response(content=body, media_type=content_type)

    return app
```

- [ ] **Step 4: Run to verify pass**

Run: `cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/api/test_app.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/vcsai/minhlbq
git add mlops-platform/src/medical_qa_platform/api/app.py mlops-platform/tests/api/test_app.py
git commit -m "feat: predict API wiring retrieval, backend, parser, drift"
```

---

## Task 17: Coverage gate + full-suite verification

**Files:**
- Modify: `mlops-platform/pyproject.toml` (add `fail_under = 80`)

- [ ] **Step 1: Add the coverage gate to `mlops-platform/pyproject.toml`** — under `[tool.coverage.report]`, add `fail_under = 80` so the section reads:

```toml
[tool.coverage.report]
exclude_lines = ["pragma: no cover", "if __name__", "raise NotImplementedError"]
fail_under = 80
```

- [ ] **Step 2: Run the full suite with coverage**

Run: `cd /home/vcsai/minhlbq/mlops-platform && make test`
Expected: all tests pass; coverage report prints; total coverage ≥ 80% and the command exits 0.

- [ ] **Step 3: If coverage < 80%, inspect the gap**

Run: `cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest --cov=medical_qa_platform --cov-report=term-missing | tail -40`
Read the "Missing" column. For any genuinely untestable-without-heavy-deps lines NOT already in `kg_backend.py`, add a targeted unit test using the existing fake/`MockTransport` patterns (do NOT add `# pragma: no cover` to runtime logic). Re-run `make test` until ≥ 80%.

- [ ] **Step 4: Commit**

```bash
cd /home/vcsai/minhlbq
git add mlops-platform/pyproject.toml
git commit -m "chore: enforce 80% coverage gate"
```

---

## Self-Review Notes (for the executor)

- **`tests/` package dirs:** pytest with `testpaths=["tests"]` + the `src/` layout editable install resolves imports without `__init__.py` in test dirs. All test module basenames are unique per directory. If collection ever errors on duplicate basenames, add empty `__init__.py` files under `tests/` and each subdir.
- **`on_event("startup")`:** used for broad FastAPI-version compatibility. A deprecation warning is acceptable for P0; migrating to `lifespan` is a non-behavioral change deferred to plan 3 hardening.
- **Encoder delivery (decision #2, no-bake):** the runtime container is NOT built in this plan. `KGRetrieval` already defaults the encoder to CPU and relies on `HF_HOME`/`SENTENCE_TRANSFORMERS_HOME` (set in the Helm chart, plan 3) pointing at the PVC populated by `dvc pull`. Nothing in this plan bakes the 1.3 GB encoder into an image.
- **Spec coverage check:** 5.1 API (Tasks 4,16) ✓; 5.2 retrieval service (Tasks 8,11,15) ✓; decision #1 device (Tasks 8,10) ✓; #4 lifespan/`/ready`/`top_k` bound (Tasks 15,16) ✓; #7 drift features (Task 13) ✓; #9 `model_version` from env at deploy (Tasks 14,16) ✓; backends mock/runpod/kserve (Tasks 5,6,7) ✓; P0 `make test` >80% (Task 17) ✓. Correctly deferred to later plans: #2 PVC/dvc-pull wiring, #3 buildx, #5 secrets, #6 MLflow, #8 HPA, tracing.
- **Type consistency:** `search(query, top_k) -> list[str]` is identical across `RetrievalBackend`, `FixtureRetrieval`, `KGRetrieval`, and `RetrievalClient`; `generate(messages, max_tokens, temperature) -> str` is identical across all backends; `DriftCollector.record(request, response, n_evidence)` matches its single caller in `app.py`.
