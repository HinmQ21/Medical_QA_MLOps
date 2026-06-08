# Free-text `/predict` + raw model output — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `/predict` accept a single free-text `question` (options inline, no structured `options`), return the raw model generation as `raw_output`, and collapse the Streamlit UI's question + 4 option boxes into one text area that shows the raw output.

**Architecture:** A coherent contract change. The request schema drops `options`; the response gains `raw_output`. `build_prompt` no longer formats options (they live inside the question text); `parse_answer` is called with no `valid_letters` (best-effort letter extraction). The HTTP client + Streamlit UI mirror the new contract. Pydantic v2 ignores extra fields, so an old caller still sending `options` is not rejected.

**Tech Stack:** Python 3.12, FastAPI + Pydantic v2, Streamlit (`streamlit.testing.v1.AppTest`), httpx (`MockTransport`), pytest. Tests run with `.venv/bin/pytest`; full suite + coverage via `make test`.

**Spec:** `docs/superpowers/specs/2026-06-08-predict-freetext-raw-output-design.md`

---

## File Structure

**Source (modify):**
- `src/medical_qa_platform/api/schemas.py` — `PredictRequest` loses `options` + validator; `PredictResponse` gains `raw_output`.
- `src/medical_qa_platform/api/prompt.py` — `build_prompt(question, evidence)` (drop `options`).
- `src/medical_qa_platform/api/app.py` — `/predict` wiring: build_prompt 2-arg, `parse_answer(raw)`, return `raw_output`.
- `src/medical_qa_platform/drift/collector.py` — drop `n_options`/`mean_option_len` from the row.
- `app/client.py` — `build_payload(question)`; `PredictResult.raw_output`.
- `app/streamlit_app.py` — single `text_area`, combined `PRESETS`, raw-output expander.
- `scripts/cloud/smoke_cloud.sh` — free-text `PAYLOAD`.

**Source (NOT modified):** `src/medical_qa_platform/api/parser.py` (`valid_letters` already optional), `prompt.SYSTEM_PROMPT` (stays verbatim-synced with training).

**Tests (modify):** `tests/api/test_schemas.py`, `tests/api/test_app.py`, `tests/api/test_prompt.py`, `tests/drift/test_collector.py`, `tests/app/test_ui_client.py`, `tests/app/test_streamlit_presets.py`, `tests/cloud/test_smoke_cloud.py`.

**Tests (unchanged):** `tests/api/test_parser.py`, `tests/api/test_system_prompt_sync.py`, `tests/app/test_streamlit_launch_path.py`, `tests/app/test_demo_packaging.py`, everything under `tests/mlops/`, `tests/deploy/`, `tests/retrieval/`.

---

## Task 1: Server contract — free-text request, `raw_output`, drop `options`

These four source files reference `req.options` / `build_prompt(…, options, …)` / `request.options`, so they must change together to keep the suite green. Update the tests first (red), then the sources (green), then one commit.

**Files:**
- Modify: `src/medical_qa_platform/api/schemas.py`
- Modify: `src/medical_qa_platform/api/prompt.py`
- Modify: `src/medical_qa_platform/api/app.py:74-87`
- Modify: `src/medical_qa_platform/drift/collector.py:13-27`
- Test: `tests/api/test_schemas.py`, `tests/api/test_prompt.py`, `tests/api/test_app.py`, `tests/drift/test_collector.py`

- [ ] **Step 1: Rewrite `tests/api/test_schemas.py` to the new contract**

```python
import pytest
from pydantic import ValidationError

from medical_qa_platform.api.schemas import PredictRequest, PredictResponse


def test_valid_request_is_free_text():
    req = PredictRequest(question="What is first-line for T2DM? A) Metformin B) Insulin")
    assert "Metformin" in req.question


def test_rejects_empty_question():
    with pytest.raises(ValidationError):
        PredictRequest(question="   ")


def test_ignores_stray_options_field():
    # Approach A: options is gone; pydantic v2 ignores extra input fields, so an old
    # caller still sending `options` gets no 422 — the field is just dropped.
    req = PredictRequest(question="Q?", options={"A": "a"})
    assert not hasattr(req, "options")


def test_response_round_trip_includes_raw_output():
    resp = PredictResponse(
        answer="A",
        raw_output="<think>r</think><answer>A</answer>",
        evidence=["e1"],
        backend="mock",
        model_version="dev",
        contract_version="v1-medembed-small",
        latency_ms=12.5,
        trace_id="abc",
    )
    dumped = resp.model_dump()
    assert dumped["answer"] == "A"
    assert dumped["raw_output"] == "<think>r</think><answer>A</answer>"
    assert dumped["contract_version"] == "v1-medembed-small"
```

- [ ] **Step 2: Rewrite `tests/api/test_prompt.py` (drop options arg + sorting test)**

```python
from medical_qa_platform.api.prompt import SYSTEM_PROMPT, build_prompt


def test_returns_system_and_user_messages():
    msgs = build_prompt("Q?", [])
    assert msgs[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert msgs[1]["role"] == "user"
    assert len(msgs) == 2


def test_user_message_contains_question_with_inline_options():
    msgs = build_prompt("What is X?\nA) alpha\nB) beta", [])
    user = msgs[1]["content"]
    assert "What is X?" in user
    assert "A) alpha" in user
    assert "B) beta" in user


def test_evidence_block_included_when_present():
    msgs = build_prompt("Q?", ["fact one", "fact two"])
    user = msgs[1]["content"]
    assert "fact one" in user
    assert "fact two" in user


def test_no_evidence_block_when_empty():
    msgs = build_prompt("Q?", [])
    assert "Evidence" not in msgs[1]["content"]
```

- [ ] **Step 3: Rewrite `tests/drift/test_collector.py` (`_req` no options, `_resp` has raw_output, row drops option fields)**

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
    row = DriftCollector(path=None).record(_req(), _resp("A", ["e1"]), n_evidence=1)
    assert row["q_token_len"] == 4
    assert row["answer"] == "A"
    assert row["no_result"] is False
    assert row["latency_ms"] == 10.0
    assert "n_options" not in row
    assert "mean_option_len" not in row


def test_no_result_flag_true_when_zero_evidence():
    row = DriftCollector(path=None).record(_req(), _resp(None, []), n_evidence=0)
    assert row["no_result"] is True
    assert row["answer"] == "none"


def test_record_does_not_raise_when_path_unwritable():
    # Drift logging is observability; a write failure (e.g. read-only/root-owned dir
    # in the container) must never break the prediction path with a 500.
    collector = DriftCollector(path="/no-such-dir-xyz/drift.jsonl")
    row = collector.record(_req(), _resp("A", ["e1"]), n_evidence=1)  # must not raise
    assert row["answer"] == "A"


def test_record_appends_jsonl(tmp_path):
    path = tmp_path / "drift.jsonl"
    collector = DriftCollector(path=str(path))
    collector.record(_req(), _resp("A", ["e1"]), n_evidence=1)
    collector.record(_req(), _resp("B", []), n_evidence=0)
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["answer"] == "B"
```

- [ ] **Step 4: Rewrite `tests/api/test_app.py` (free-text payloads, raw_output, no options-constraint)**

```python
from fastapi.testclient import TestClient

from medical_qa_platform.api.app import create_app
from medical_qa_platform.inference.base import ModelBackend
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
    resp = client.post("/predict", json={"question": "Q?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "B"
    assert "<answer>B</answer>" in body["raw_output"]
    assert body["evidence"] == ["evidence one", "evidence two"]
    assert body["backend"] == "mock"
    assert body["model_version"] == "test-v1"
    assert body["latency_ms"] >= 0
    assert body["trace_id"]


def test_predict_accepts_free_text_without_options(tmp_path):
    resp = _client(tmp_path).post("/predict", json={"question": "Q?"})
    assert resp.status_code == 200


def test_predict_answer_not_constrained_to_letters(tmp_path):
    # valid_letters is gone; any single-letter <answer> is returned verbatim.
    app = create_app(
        backend=MockBackend(answer="D"),
        retrieval=FixtureRetrieval({}),
        model_version="x",
        drift_log_path=str(tmp_path / "d.jsonl"),
    )
    resp = TestClient(app).post("/predict", json={"question": "Q?"})
    assert resp.json()["answer"] == "D"


def test_predict_writes_drift_row(tmp_path):
    path = tmp_path / "drift.jsonl"
    app = create_app(
        backend=MockBackend(answer="A"),
        retrieval=FixtureRetrieval({"Q?": ["e"]}),
        model_version="x",
        drift_log_path=str(path),
    )
    TestClient(app).post("/predict", json={"question": "Q?"})
    assert path.exists()
    assert path.read_text().strip()


def test_metrics_endpoint(tmp_path):
    client = _client(tmp_path)
    client.post("/predict", json={"question": "Q?"})
    resp = client.get("/metrics")
    assert "mqa_requests_total" in resp.text


def test_predict_response_includes_contract_version(tmp_path):
    from medical_qa_platform.retrieval.contract import RETRIEVAL_CONTRACT_VERSION

    client = _client(tmp_path)
    resp = client.post("/predict", json={"question": "Q?"})
    assert resp.json()["contract_version"] == RETRIEVAL_CONTRACT_VERSION


def test_version_endpoint_reports_contract_and_model(tmp_path):
    from medical_qa_platform.retrieval.contract import RETRIEVAL_CONTRACT_VERSION

    body = _client(tmp_path).get("/version").json()
    assert body["contract_version"] == RETRIEVAL_CONTRACT_VERSION
    assert body["model_version"] == "test-v1"
    assert body["backend"] == "mock"


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
    TestClient(app).post("/predict", json={"question": "Q?"})
    assert backend.max_tokens_calls == [2048]


def test_ready_returns_503_when_backend_unhealthy(tmp_path):
    app = create_app(
        backend=_UnhealthyBackend(),
        retrieval=FixtureRetrieval({}),
        drift_log_path=str(tmp_path / "d.jsonl"),
    )
    assert TestClient(app).get("/ready").status_code == 503
```

- [ ] **Step 5: Run the four test files to confirm they FAIL**

Run: `.venv/bin/pytest tests/api/test_schemas.py tests/api/test_prompt.py tests/api/test_app.py tests/drift/test_collector.py -q`
Expected: FAIL — `PredictResponse` missing `raw_output`, `PredictRequest` still has `options`, `build_prompt()` arity mismatch, `record()` still returns `n_options`.

- [ ] **Step 6: Rewrite `src/medical_qa_platform/api/schemas.py`**

```python
"""Pydantic request/response models for the predict API."""

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    question: str = Field(min_length=1)

    @field_validator("question")
    @classmethod
    def _question_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value


class PredictResponse(BaseModel):
    answer: str | None
    raw_output: str
    evidence: list[str]
    backend: str
    model_version: str
    contract_version: str
    latency_ms: float
    trace_id: str
```

- [ ] **Step 7: Rewrite `build_prompt` in `src/medical_qa_platform/api/prompt.py`**

Replace the `build_prompt` function (keep `SYSTEM_PROMPT` and the module docstring exactly as-is):

```python
def build_prompt(
    question: str,
    evidence: list[str],
) -> list[dict]:
    """Return OpenAI-style chat messages for the given question.

    The question is free-text and already contains any answer options inline,
    so no separate options block is rendered.
    """
    lines = []
    if evidence:
        lines.append("Evidence:")
        lines.extend(f"- {item}" for item in evidence)
        lines.append("")
    lines.append(f"Question: {question}")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]
```

- [ ] **Step 8: Update the `/predict` body in `src/medical_qa_platform/api/app.py`**

Replace lines 74-87 (from `evidence = …` through the `PredictResponse(...)` block) with:

```python
        evidence = app.state.retrieval.search(req.question, app.state.top_k)
        messages = build_prompt(req.question, evidence)
        raw = app.state.backend.generate(messages, max_tokens=app.state.max_tokens)
        answer = parse_answer(raw)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        resp = PredictResponse(
            answer=answer,
            raw_output=raw,
            evidence=evidence,
            backend=app.state.backend.name,
            model_version=app.state.model_version,
            contract_version=RETRIEVAL_CONTRACT_VERSION,
            latency_ms=latency_ms,
            trace_id=trace_id,
        )
```

- [ ] **Step 9: Update `record()` in `src/medical_qa_platform/drift/collector.py`**

Replace the `row = {…}` dict (lines 19-27, including the `option_lengths` line above it) with:

```python
        row = {
            "q_token_len": len(request.question.split()),
            "answer": response.answer if response.answer is not None else "none",
            "no_result": n_evidence == 0,
            "latency_ms": response.latency_ms,
        }
```

(Delete the `option_lengths = [...]` line entirely.)

- [ ] **Step 10: Run the four test files to confirm they PASS**

Run: `.venv/bin/pytest tests/api/test_schemas.py tests/api/test_prompt.py tests/api/test_app.py tests/drift/test_collector.py -q`
Expected: PASS (all green).

- [ ] **Step 11: Commit**

```bash
git add src/medical_qa_platform/api/schemas.py src/medical_qa_platform/api/prompt.py \
        src/medical_qa_platform/api/app.py src/medical_qa_platform/drift/collector.py \
        tests/api/test_schemas.py tests/api/test_prompt.py tests/api/test_app.py \
        tests/drift/test_collector.py
git commit -m "feat: free-text /predict request + raw_output in response"
```

---

## Task 2: HTTP client — `build_payload(question)` + `PredictResult.raw_output`

`app/client.py` is Streamlit-free and unit-tested via httpx `MockTransport`, so it is independent of Task 1's server code.

**Files:**
- Modify: `app/client.py`
- Test: `tests/app/test_ui_client.py`

- [ ] **Step 1: Rewrite `tests/app/test_ui_client.py`**

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


def test_build_payload_strips_question():
    assert build_payload("  first-line? ") == {"question": "first-line?"}


def test_build_payload_rejects_blank_question():
    with pytest.raises(ValueError):
        build_payload("   ")


def test_predict_parses_success_and_sends_key():
    def handler(request):
        assert request.url.path == "/predict"
        assert request.headers["x-api-key"] == "k"
        return httpx.Response(
            200,
            json={
                "answer": "A",
                "raw_output": "<think>r</think><answer>A</answer>",
                "evidence": ["e1", "e2"],
                "backend": "llm",
                "model_version": "smoke-dev",
                "contract_version": "v1",
                "latency_ms": 12.5,
                "trace_id": "abc",
            },
        )

    res = predict("http://gw:8080/", "k", {"question": "q"}, client=_client(handler))
    assert isinstance(res, PredictResult)
    assert res.answer == "A"
    assert res.raw_output == "<think>r</think><answer>A</answer>"
    assert res.evidence == ["e1", "e2"]
    assert res.backend == "llm"
    assert res.trace_id == "abc"


def test_predict_raises_predicterror_on_401():
    def handler(request):
        return httpx.Response(401)

    with pytest.raises(PredictError, match="unauthorized"):
        predict("http://gw:8080", "bad", {"question": "q"}, client=_client(handler))


def test_predict_raises_predicterror_on_timeout():
    def handler(request):
        raise httpx.TimeoutException("slow")

    with pytest.raises(PredictError, match="timed out"):
        predict("http://gw:8080", "k", {"question": "q"}, timeout=3, client=_client(handler))


def test_predict_raises_predicterror_on_connection_error():
    def handler(request):
        raise httpx.ConnectError("nope")

    with pytest.raises(PredictError, match="could not reach"):
        predict("http://gw:8080", "k", {"question": "q"}, client=_client(handler))


def test_fetch_version_returns_json():
    def handler(request):
        assert request.url.path == "/version"
        return httpx.Response(
            200,
            json={"backend": "llm", "model_version": "smoke-dev", "contract_version": "v1"},
        )

    out = fetch_version("http://gw:8080", "k", client=_client(handler))
    assert out["backend"] == "llm"


def test_fetch_version_raises_predicterror_on_401():
    def handler(request):
        return httpx.Response(401)

    with pytest.raises(PredictError, match="unauthorized"):
        fetch_version("http://gw:8080", "k", client=_client(handler))


def test_fetch_version_raises_predicterror_on_connection_error():
    def handler(request):
        raise httpx.ConnectError("down")

    with pytest.raises(PredictError, match="could not reach"):
        fetch_version("http://gw:8080", "k", client=_client(handler))


def test_predict_raises_predicterror_on_non_json_body():
    def handler(request):
        return httpx.Response(200, text="<html>oops</html>")

    with pytest.raises(PredictError):
        predict("http://gw:8080", "k", {"question": "q"}, client=_client(handler))
```

- [ ] **Step 2: Run to confirm FAIL**

Run: `.venv/bin/pytest tests/app/test_ui_client.py -q`
Expected: FAIL — `build_payload()` still requires `options`; `PredictResult` has no `raw_output`.

- [ ] **Step 3: Update `build_payload`, `PredictResult`, and `predict` in `app/client.py`**

Replace the `PredictResult` dataclass and `build_payload` function with:

```python
@dataclass
class PredictResult:
    answer: str | None
    raw_output: str
    evidence: list[str]
    backend: str
    model_version: str
    contract_version: str
    latency_ms: float
    trace_id: str


def build_payload(question: str) -> dict:
    """Validate the UI input into the /predict request body (mirrors the server contract)."""
    question = question.strip()
    if not question:
        raise ValueError("Câu hỏi không được để trống.")
    return {"question": question}
```

Then, inside `predict()`, add `raw_output` to the returned `PredictResult` (right after `answer=...`):

```python
    return PredictResult(
        answer=data.get("answer"),
        raw_output=data.get("raw_output", ""),
        evidence=data.get("evidence", []),
        backend=data.get("backend", ""),
        model_version=data.get("model_version", ""),
        contract_version=data.get("contract_version", ""),
        latency_ms=data.get("latency_ms", 0.0),
        trace_id=data.get("trace_id", ""),
    )
```

- [ ] **Step 4: Run to confirm PASS**

Run: `.venv/bin/pytest tests/app/test_ui_client.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/client.py tests/app/test_ui_client.py
git commit -m "feat: client build_payload(question) + PredictResult.raw_output"
```

---

## Task 3: Streamlit UI — single text area, combined presets, raw expander

Depends on Task 2 (`build_payload(question)` single-arg signature).

**Files:**
- Modify: `app/streamlit_app.py`
- Test: `tests/app/test_streamlit_presets.py`

- [ ] **Step 1: Rewrite `tests/app/test_streamlit_presets.py`**

```python
import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest


def test_selecting_a_preset_fills_the_single_question_box():
    at = AppTest.from_file("app/streamlit_app.py").run()
    # index 0 is "(tự nhập)"; pick the first real preset
    at.selectbox[0].select_index(1).run()

    value = at.text_area[0].value
    assert value.strip(), "preset should fill the single question box"
    assert "A)" in value, "preset embeds the options inline in the one box"

    # the separate A/B/C/D option inputs are gone
    opt_inputs = [ti for ti in at.text_input if ti.label in ("A", "B", "C", "D")]
    assert opt_inputs == []
```

- [ ] **Step 2: Run to confirm FAIL**

Run: `.venv/bin/pytest tests/app/test_streamlit_presets.py -q`
Expected: FAIL (old test name gone / option inputs still present) — or an error because the app still renders 4 option `text_input`s.

- [ ] **Step 3: Rewrite `PRESETS` and `main()` in `app/streamlit_app.py`**

Replace the `PRESETS` dict with combined single strings:

```python
PRESETS: dict[str, str] = {
    "Đái tháo đường type 2 — first-line": (
        "Which medication is first-line for type 2 diabetes mellitus?\n"
        "A) Metformin\n"
        "B) Amoxicillin\n"
        "C) Atorvastatin\n"
        "D) Furosemide"
    ),
    "Nhồi máu cơ tim — marker": (
        "Which serum marker is most specific for acute myocardial infarction?\n"
        "A) Troponin I\n"
        "B) Amylase\n"
        "C) ALT\n"
        "D) Creatinine"
    ),
}
```

Replace the body of `main()` from the `preset = …` line down to the end of the function with:

```python
    preset = st.selectbox("Câu hỏi mẫu", ["(tự nhập)", *PRESETS])
    seed = PRESETS.get(preset, "")

    question = st.text_area(
        "Câu hỏi (kèm phương án)",
        value=seed,
        height=200,
        help="Dán nguyên câu hỏi trắc nghiệm kèm các phương án, ví dụ 'A) ... B) ...'.",
    )

    if not st.button("Chẩn đoán", type="primary"):
        return

    try:
        payload = build_payload(question)
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
        st.warning("Mô hình không trả về đáp án dạng chữ cái (no letter parsed); xem phản hồi thô bên dưới.")
    else:
        st.success(f"Đáp án: **{result.answer}**")

    with st.expander("🧠 Phản hồi thô của mô hình", expanded=False):
        st.code(result.raw_output or "(rỗng)")

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
```

(The four `st.text_input(letter, …)` option widgets, the `st.write("Phương án:")` line, the `st.columns(2)` block, and the per-option highlight loop are all removed.)

- [ ] **Step 4: Run to confirm PASS**

Run: `.venv/bin/pytest tests/app/test_streamlit_presets.py tests/app/test_streamlit_launch_path.py -q`
Expected: PASS (or `skipped` if the `[demo]` extra / streamlit isn't installed — `importorskip`).

- [ ] **Step 5: Commit**

```bash
git add app/streamlit_app.py tests/app/test_streamlit_presets.py
git commit -m "feat: UI single free-text box + raw-output expander"
```

---

## Task 4: Smoke script — free-text payload

**Files:**
- Modify: `scripts/cloud/smoke_cloud.sh:55`
- Test: `tests/cloud/test_smoke_cloud.py`

- [ ] **Step 1: Update `tests/cloud/test_smoke_cloud.py` (drop the `"options"` assertion)**

In `test_dry_run_resolves_lb_ip_and_hits_health_version_and_predict`, replace:

```python
    assert '"question"' in o
    assert '"options"' in o
```

with:

```python
    assert '"question"' in o
    assert '"options"' not in o
```

- [ ] **Step 2: Run to confirm FAIL**

Run: `.venv/bin/pytest tests/cloud/test_smoke_cloud.py -q`
Expected: FAIL — current `PAYLOAD` still contains `"options"`.

- [ ] **Step 3: Update `PAYLOAD` in `scripts/cloud/smoke_cloud.sh`**

Replace line 55:

```bash
PAYLOAD='{"question":"Which medication is first-line for type 2 diabetes? A) Metformin B) Amoxicillin"}'
```

- [ ] **Step 4: Run to confirm PASS (and bash still parses)**

Run: `.venv/bin/pytest tests/cloud/test_smoke_cloud.py -q && bash -n scripts/cloud/smoke_cloud.sh`
Expected: PASS, and `bash -n` exits 0.

- [ ] **Step 5: Commit**

```bash
git add scripts/cloud/smoke_cloud.sh tests/cloud/test_smoke_cloud.py
git commit -m "test: smoke payload uses free-text question (no options)"
```

---

## Task 5: Full-suite verification

**Files:** none (verification only).

- [ ] **Step 1: Run the full suite with coverage**

Run: `make test`
Expected: all pass (≤1 pre-existing skip), coverage roughly unchanged from the prior ~95%. If anything fails, fix the offending file in its own task above and re-run.

- [ ] **Step 2: Grep for any stale `options` / 3-arg `build_prompt` references**

Run: `grep -rn "req.options\|request.options\|build_prompt([^)]*,[^)]*,[^)]*)\|build_payload([^)]*,[^)]*)" src/ app/ tests/ scripts/`
Expected: no matches (empty output). Any hit is a missed update — fix it.

- [ ] **Step 3: Final commit if Step 2 required fixes (otherwise skip)**

```bash
git add -A
git commit -m "chore: remove stale options references"
```

---

## Self-Review (completed during planning)

- **Spec coverage:** request drops `options` (Task 1, schemas) ✓; response gains `raw_output` (Task 1) ✓; `build_prompt` drops options (Task 1, prompt) ✓; best-effort `parse_answer` (Task 1, app) ✓; drift row drops option fields (Task 1, collector) ✓; client `build_payload(question)` + `raw_output` (Task 2) ✓; single-box UI + presets + raw expander (Task 3) ✓; smoke free-text (Task 4) ✓; non-goals untouched (parser, SYSTEM_PROMPT, contract version) ✓.
- **Placeholder scan:** none — every step has the real code/command.
- **Type consistency:** `raw_output: str` defined identically in `PredictResponse` (schema) and `PredictResult` (client); `build_prompt(question, evidence)` and `build_payload(question)` signatures match across source + every test; `MockBackend` raw asserted as substring `<answer>B</answer>` (matches its actual `<think>…</think><answer>{answer}</answer>` output).
