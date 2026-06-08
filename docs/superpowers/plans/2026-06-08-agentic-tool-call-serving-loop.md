# Agentic Tool-Call Serving Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the serving `/predict` one-shot RAG with a model-driven agentic tool-call loop (native OpenAI `tools=`), and render the per-turn transcript in the UI instead of a parsed answer letter.

**Architecture:** A new loop in the API layer calls the OpenAI `/v1/chat/completions` backend with `tools=[search_medical_knowledge]`; when the model returns `tool_calls`, the loop executes retrieval via the existing HTTP client, feeds the result back as a `tool` message, and repeats up to `MAX_TOOL_ITERATIONS` (default 2) rounds before forcing a final answer. The backend interface grows a structured `chat()` returning `ChatTurn(content, tool_calls, finish_reason)`; `generate()` stays as a text-only wrapper. The response gains a `trace` of turns; the Streamlit UI renders each turn as a labeled block.

**Tech Stack:** Python 3.12, FastAPI, Pydantic v2, httpx, Streamlit, pytest. llama.cpp server (already launched with `--jinja`, so it returns OpenAI-style tool calls — no chart change).

**Spec:** `docs/superpowers/specs/2026-06-08-agentic-tool-call-serving-loop-design.md`

**Test command:** `.venv/bin/pytest` (baseline before this work: 250 passed, 1 skipped).

**Branch:** `feat/agentic-serving-loop` (already created).

---

## File Structure

**Create:**
- `src/medical_qa_platform/inference/tools.py` — the `search_medical_knowledge` tool schema constant.
- `src/medical_qa_platform/inference/agent.py` — `LoopResult` + `run_agentic_loop`.
- `tests/inference/test_tools.py` — tool-schema shape.
- `tests/inference/test_agent.py` — loop behavior.
- `tests/app/test_streamlit_transcript.py` — UI wiring guard.

**Modify:**
- `src/medical_qa_platform/inference/base.py` — add `ChatTurn`; `chat()` abstract; `generate()` wrapper.
- `src/medical_qa_platform/inference/llm_backend.py` — implement `chat()`; drop bespoke `generate()`.
- `src/medical_qa_platform/inference/mock_backend.py` — implement `chat()`; drop bespoke `generate()`.
- `src/medical_qa_platform/config.py` — add `max_tool_iterations`.
- `src/medical_qa_platform/api/schemas.py` — add `Turn`; add `trace` to `PredictResponse`.
- `src/medical_qa_platform/api/app.py` — call `run_agentic_loop`; wire `max_tool_iterations`.
- `app/client.py` — `PredictResult.trace`; parse it; 120s timeout; `build_transcript_blocks`.
- `app/streamlit_app.py` — render the transcript.
- `tests/inference/test_llm_backend.py`, `tests/inference/test_mock_backend.py`, `tests/api/test_app.py`, `tests/test_config.py`, `tests/api/test_schemas.py`, `tests/app/test_ui_client.py`, `tests/deploy/test_kserve_chart.py` — updated/extended tests.

**Task order keeps the suite green at every commit:** backend `chat()` (1) → tool def (2) → loop (3) → config (4) → schema (5) → API wiring (6) → UI client (7) → Streamlit (8) → chart guard (9).

---

### Task 1: Backend `chat()` interface + `ChatTurn`

**Files:**
- Modify: `src/medical_qa_platform/inference/base.py`
- Modify: `src/medical_qa_platform/inference/llm_backend.py`
- Modify: `src/medical_qa_platform/inference/mock_backend.py`
- Modify: `tests/inference/test_llm_backend.py`
- Modify: `tests/inference/test_mock_backend.py`
- Modify: `tests/api/test_app.py` (the two in-file test backends must implement `chat()`)

- [ ] **Step 1: Write the failing tests for `LLMBackend.chat()`**

Append to `tests/inference/test_llm_backend.py`:

```python
def test_chat_posts_tools_and_tool_choice():
    captured = {}
    _backend(captured).chat(
        [{"role": "user", "content": "x"}],
        tools=[{"type": "function", "function": {"name": "t"}}],
        tool_choice="auto",
    )
    assert captured["body"]["tools"][0]["function"]["name"] == "t"
    assert captured["body"]["tool_choice"] == "auto"


def test_chat_omits_tools_when_none():
    captured = {}
    _backend(captured).chat([{"role": "user", "content": "x"}])
    assert "tools" not in captured["body"]
    assert "tool_choice" not in captured["body"]


def test_chat_parses_tool_calls_and_finish_reason():
    body = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "search_medical_knowledge",
                                "arguments": '{"query": "metformin"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }
    turn = _backend({}, json_body=body).chat([{"role": "user", "content": "x"}])
    assert turn.finish_reason == "tool_calls"
    assert turn.content is None
    assert turn.tool_calls[0]["function"]["name"] == "search_medical_knowledge"


def test_chat_defaults_finish_reason_and_tool_calls_when_absent():
    # the default mock body has neither finish_reason nor tool_calls
    turn = _backend({}).chat([{"role": "user", "content": "x"}])
    assert turn.content == "<answer>B</answer>"
    assert turn.tool_calls == []
    assert turn.finish_reason == "stop"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/inference/test_llm_backend.py -q`
Expected: FAIL — `AttributeError: 'LLMBackend' object has no attribute 'chat'`.

- [ ] **Step 3: Add `ChatTurn` and the new interface to `base.py`**

Replace the entire contents of `src/medical_qa_platform/inference/base.py` with:

```python
"""Model backend interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ChatTurn:
    """One assistant turn from a chat-completions call.

    ``tool_calls`` holds the raw OpenAI tool-call objects
    (``{"id", "type", "function": {"name", "arguments"}}``) so the agentic loop
    can echo them back verbatim in the next request.
    """

    content: str | None
    tool_calls: list[dict] = field(default_factory=list)
    finish_reason: str = "stop"


class ModelBackend(ABC):
    """A backend that turns chat messages into an assistant turn."""

    name: str = "base"

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str = "auto",
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> ChatTurn:
        """Return the assistant turn (content + any tool calls) for these messages."""
        raise NotImplementedError

    def generate(
        self,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> str:
        """Convenience wrapper: return just the assistant text (no tools)."""
        return (
            self.chat(messages, max_tokens=max_tokens, temperature=temperature).content
            or ""
        )

    def health_check(self) -> bool:
        """Return True if the backend is reachable and ready to serve.

        Default: assume healthy (mock/in-process backends never go down).
        Network backends override this with a real probe.
        """
        return True
```

- [ ] **Step 4: Implement `chat()` in `llm_backend.py`**

In `src/medical_qa_platform/inference/llm_backend.py`, change the import line `from .base import ModelBackend` to:

```python
from .base import ChatTurn, ModelBackend
```

Then replace the existing `generate(...)` method (the whole method, lines defining `def generate` through its `return`) with:

```python
    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str = "auto",
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> ChatTurn:
        body: dict = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools is not None:
            body["tools"] = tools
            body["tool_choice"] = tool_choice
        resp = self._client.post(
            f"{self.base_url}/chat/completions",
            headers=self._auth_headers(),
            json=body,
        )
        resp.raise_for_status()
        choice = resp.json()["choices"][0]
        message = choice.get("message", {})
        return ChatTurn(
            content=message.get("content"),
            tool_calls=message.get("tool_calls") or [],
            finish_reason=choice.get("finish_reason") or "stop",
        )
```

(`generate()` is now inherited from `ModelBackend` and still returns the content string.)

- [ ] **Step 5: Implement `chat()` in `mock_backend.py`**

Replace the entire contents of `src/medical_qa_platform/inference/mock_backend.py` with:

```python
"""Deterministic backend for tests and CI (no GPU, no network)."""

from .base import ChatTurn, ModelBackend


class MockBackend(ModelBackend):
    name = "mock"

    def __init__(self, answer: str = "A"):
        self._answer = answer.upper()

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str = "auto",
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> ChatTurn:
        return ChatTurn(
            content=(
                f"<think>Mock reasoning for {len(messages)} messages.</think>"
                f"<answer>{self._answer}</answer>"
            ),
            tool_calls=[],
            finish_reason="stop",
        )
```

- [ ] **Step 6: Add a `chat()` test to `test_mock_backend.py`**

Append to `tests/inference/test_mock_backend.py`:

```python
def test_mock_chat_returns_chatturn():
    from medical_qa_platform.inference.base import ChatTurn

    turn = MockBackend(answer="C").chat([{"role": "user", "content": "x"}])
    assert isinstance(turn, ChatTurn)
    assert "<answer>C</answer>" in (turn.content or "")
    assert turn.tool_calls == []
    assert turn.finish_reason == "stop"
```

- [ ] **Step 7: Update the in-file test backends in `test_app.py` to implement `chat()`**

In `tests/api/test_app.py`, change the import block near the top to include `ChatTurn`:

```python
from medical_qa_platform.inference.base import ChatTurn, ModelBackend
```

Replace the `_RecordingBackend` and `_UnhealthyBackend` classes with:

```python
class _RecordingBackend(ModelBackend):
    name = "rec"

    def __init__(self):
        self.max_tokens_calls = []

    def chat(self, messages, tools=None, tool_choice="auto", max_tokens=512, temperature=0.3):
        self.max_tokens_calls.append(max_tokens)
        return ChatTurn(content="<answer>A</answer>", tool_calls=[], finish_reason="stop")


class _UnhealthyBackend(ModelBackend):
    name = "down"

    def chat(self, messages, tools=None, tool_choice="auto", max_tokens=512, temperature=0.3):
        return ChatTurn(content="<answer>A</answer>", tool_calls=[], finish_reason="stop")

    def health_check(self):
        return False
```

- [ ] **Step 8: Run the affected suites to verify they pass**

Run: `.venv/bin/pytest tests/inference tests/api/test_app.py -q`
Expected: PASS. (`generate()` tests still pass via the inherited wrapper; `app.py` still calls `generate()` and pre-retrieval, so `test_predict_returns_all_fields` still sees the fixture evidence.)

- [ ] **Step 9: Run the full suite (interface change — verify nothing else broke)**

Run: `.venv/bin/pytest -q`
Expected: PASS (250 passed, 1 skipped).

- [ ] **Step 10: Commit**

```bash
git add src/medical_qa_platform/inference/base.py \
        src/medical_qa_platform/inference/llm_backend.py \
        src/medical_qa_platform/inference/mock_backend.py \
        tests/inference/test_llm_backend.py \
        tests/inference/test_mock_backend.py \
        tests/api/test_app.py
git commit -m "$(printf 'feat(inference): add structured chat() returning ChatTurn\n\nchat() is the new abstract backend method (content + tool_calls +\nfinish_reason); generate() becomes a text-only wrapper. Prepares the\nbackend for native OpenAI tool-calling.\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

### Task 2: `search_medical_knowledge` tool definition

**Files:**
- Create: `src/medical_qa_platform/inference/tools.py`
- Create: `tests/inference/test_tools.py`

- [ ] **Step 1: Write the failing test**

Create `tests/inference/test_tools.py`:

```python
from medical_qa_platform.inference.tools import MEDICAL_TOOL_DEF, SEARCH_TOOL_NAME


def test_tool_def_shape_matches_training_schema():
    assert SEARCH_TOOL_NAME == "search_medical_knowledge"
    assert MEDICAL_TOOL_DEF["type"] == "function"
    fn = MEDICAL_TOOL_DEF["function"]
    assert fn["name"] == SEARCH_TOOL_NAME
    assert fn["parameters"]["properties"]["query"]["type"] == "string"
    assert fn["parameters"]["required"] == ["query"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/inference/test_tools.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'medical_qa_platform.inference.tools'`.

- [ ] **Step 3: Create the tool module**

Create `src/medical_qa_platform/inference/tools.py`:

```python
"""The tool schema advertised to the model for native OpenAI function-calling.

Mirrors the training/eval tool definition
(baseline/scripts/utils/model_adapter.py:MEDICAL_TOOL_DEF) so serving behaves
like the agentic rollout the model was trained against. Kept in its own module
so a Llama-family variant can be added later without touching the loop.
"""

SEARCH_TOOL_NAME = "search_medical_knowledge"

MEDICAL_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": SEARCH_TOOL_NAME,
        "description": (
            "Search the medical knowledge base for relevant clinical information "
            "about diseases, drugs, symptoms, and treatments."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Medical query to search for.",
                }
            },
            "required": ["query"],
        },
    },
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/pytest tests/inference/test_tools.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/medical_qa_platform/inference/tools.py tests/inference/test_tools.py
git commit -m "$(printf 'feat(inference): add search_medical_knowledge tool schema\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

### Task 3: Agentic loop `run_agentic_loop`

**Files:**
- Create: `src/medical_qa_platform/inference/agent.py`
- Create: `tests/inference/test_agent.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/inference/test_agent.py`:

```python
import json

from medical_qa_platform.inference.agent import run_agentic_loop
from medical_qa_platform.inference.base import ChatTurn, ModelBackend
from medical_qa_platform.retrieval.backends import FixtureRetrieval


class ScriptedBackend(ModelBackend):
    """Returns pre-scripted ChatTurns in order; records the messages and tools flag per call."""

    name = "scripted"

    def __init__(self, turns):
        self._turns = list(turns)
        self.calls = []

    def chat(self, messages, tools=None, tool_choice="auto", max_tokens=512, temperature=0.3):
        self.calls.append({"messages": [dict(m) for m in messages], "tools_none": tools is None})
        return self._turns.pop(0)


def _tool_call(call_id, query):
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": "search_medical_knowledge",
            "arguments": json.dumps({"query": query}),
        },
    }


def test_no_tool_turn_ends_loop_with_no_retrieval():
    backend = ScriptedBackend([ChatTurn(content="<answer>A</answer>", finish_reason="stop")])
    retrieval = FixtureRetrieval({"x": ["unused"]})
    res = run_agentic_loop(backend, retrieval, "Q?", top_k=5, max_tokens=512, max_iterations=2)
    assert res.final_content == "<answer>A</answer>"
    assert res.evidence == []
    assert res.tool_call_count == 0
    assert len(backend.calls) == 1


def test_tool_call_executes_and_feeds_result_back():
    backend = ScriptedBackend([
        ChatTurn(content="<think>need facts</think>",
                 tool_calls=[_tool_call("c1", "metformin")], finish_reason="tool_calls"),
        ChatTurn(content="<answer>B</answer>", finish_reason="stop"),
    ])
    retrieval = FixtureRetrieval({"metformin": ["first-line for T2DM"]})
    res = run_agentic_loop(backend, retrieval, "Q?", top_k=5, max_tokens=512, max_iterations=2)
    assert res.final_content == "<answer>B</answer>"
    assert res.evidence == ["first-line for T2DM"]
    assert res.tool_call_count == 1
    second_call_msgs = backend.calls[1]["messages"]
    assert any(m["role"] == "tool" and "first-line for T2DM" in m["content"]
               for m in second_call_msgs)
    # assistant tool_calls echoed verbatim into the resent messages
    assert any(m["role"] == "assistant" and m.get("tool_calls") for m in second_call_msgs)


def test_max_iterations_final_round_omits_tools():
    backend = ScriptedBackend([
        ChatTurn(content="t1", tool_calls=[_tool_call("c1", "q1")], finish_reason="tool_calls"),
        ChatTurn(content="t2", tool_calls=[_tool_call("c2", "q2")], finish_reason="tool_calls"),
        ChatTurn(content="<answer>C</answer>", finish_reason="stop"),
    ])
    retrieval = FixtureRetrieval({"q1": ["a"], "q2": ["b"]})
    res = run_agentic_loop(backend, retrieval, "Q?", top_k=5, max_tokens=512, max_iterations=2)
    assert res.final_content == "<answer>C</answer>"
    assert res.tool_call_count == 2
    assert res.evidence == ["a", "b"]
    assert len(backend.calls) == 3
    assert backend.calls[2]["tools_none"] is True


def test_malformed_tool_args_degrade_to_empty_query():
    bad = {"id": "c1", "type": "function",
           "function": {"name": "search_medical_knowledge", "arguments": "{not json"}}
    backend = ScriptedBackend([
        ChatTurn(content="t", tool_calls=[bad], finish_reason="tool_calls"),
        ChatTurn(content="<answer>D</answer>", finish_reason="stop"),
    ])

    class _Recorder(FixtureRetrieval):
        def __init__(self):
            super().__init__({})
            self.queries = []

        def search(self, query, top_k):
            self.queries.append(query)
            return super().search(query, top_k)

    retrieval = _Recorder()
    res = run_agentic_loop(backend, retrieval, "Q?", top_k=5, max_tokens=512, max_iterations=2)
    assert res.final_content == "<answer>D</answer>"
    assert retrieval.queries == []  # empty query → retrieval not called
    assert res.tool_call_count == 1
    assert any(t["role"] == "tool" and "No relevant knowledge found." in t["content"]
               for t in res.trace)


def test_trace_has_assistant_then_tool_then_assistant():
    backend = ScriptedBackend([
        ChatTurn(content="<think>x</think>",
                 tool_calls=[_tool_call("c1", "q")], finish_reason="tool_calls"),
        ChatTurn(content="<answer>A</answer>", finish_reason="stop"),
    ])
    retrieval = FixtureRetrieval({"q": ["fact"]})
    res = run_agentic_loop(backend, retrieval, "Q?", top_k=5, max_tokens=512, max_iterations=2)
    assert [t["role"] for t in res.trace] == ["assistant", "tool", "assistant"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/inference/test_agent.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'medical_qa_platform.inference.agent'`.

- [ ] **Step 3: Implement the loop**

Create `src/medical_qa_platform/inference/agent.py`:

```python
"""Model-driven agentic tool-call loop for serving.

The model decides when to call ``search_medical_knowledge``; this loop executes
the call against the retrieval service, feeds the result back, and repeats until
the model answers or the tool-round budget is exhausted. Mirrors the training/eval
rollout (baseline/scripts/benchmark/grpo_eval/grpo_eval.py).
"""

import json
from dataclasses import dataclass, field

from ..api.prompt import build_prompt
from ..retrieval.backends import SupportsSearch
from .base import ModelBackend
from .tools import MEDICAL_TOOL_DEF


@dataclass
class LoopResult:
    trace: list[dict] = field(default_factory=list)
    final_content: str = ""
    evidence: list[str] = field(default_factory=list)
    tool_call_count: int = 0


def _query_of(tool_call: dict) -> str:
    """Extract the ``query`` argument from a raw OpenAI tool call; "" on bad JSON."""
    args = tool_call.get("function", {}).get("arguments", "")
    if isinstance(args, dict):
        return str(args.get("query", ""))
    try:
        return str(json.loads(args).get("query", ""))
    except (json.JSONDecodeError, AttributeError, TypeError):
        return ""


def run_agentic_loop(
    backend: ModelBackend,
    retrieval: SupportsSearch,
    question: str,
    *,
    top_k: int,
    max_tokens: int,
    max_iterations: int,
) -> LoopResult:
    messages = build_prompt(question, [])
    result = LoopResult()

    turn = None
    for i in range(max_iterations + 1):
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

        assistant_msg = {"role": "assistant", "content": turn.content or ""}
        if turn.tool_calls:
            assistant_msg["tool_calls"] = turn.tool_calls
        messages.append(assistant_msg)
        result.trace.append(
            {
                "role": "assistant",
                "content": turn.content or "",
                "tool_calls": turn.tool_calls or None,
            }
        )

        if turn.tool_calls and i < max_iterations:
            for tc in turn.tool_calls:
                result.tool_call_count += 1
                query = _query_of(tc)
                hits = retrieval.search(query, top_k) if query else []
                result.evidence.extend(hits)
                content = "\n".join(f"- {h}" for h in hits) or "No relevant knowledge found."
                messages.append(
                    {"role": "tool", "tool_call_id": tc.get("id", ""), "content": content}
                )
                result.trace.append({"role": "tool", "content": content, "tool_calls": None})
            continue

        break

    result.final_content = (turn.content if turn else "") or ""
    return result
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/inference/test_agent.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/medical_qa_platform/inference/agent.py tests/inference/test_agent.py
git commit -m "$(printf 'feat(inference): model-driven agentic tool-call loop\n\nrun_agentic_loop drives search_medical_knowledge over multiple turns,\nfeeding tool results back; forces a final answer once the round budget\nis spent. Robust to no-tool turns and malformed tool args.\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

### Task 4: Config `max_tool_iterations`

**Files:**
- Modify: `src/medical_qa_platform/config.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Update the failing tests**

In `tests/test_config.py`, add `"MAX_TOOL_ITERATIONS"` to the `delenv` list in `test_defaults` and assert the default, then assert the env read in `test_reads_env`. The two functions become:

```python
def test_defaults(monkeypatch):
    for var in [
        "MODEL_BACKEND",
        "RETRIEVAL_URL",
        "MODEL_VERSION",
        "TOP_K",
        "DRIFT_LOG_PATH",
        "MAX_TOKENS",
        "MAX_TOOL_ITERATIONS",
    ]:
        monkeypatch.delenv(var, raising=False)
    settings = Settings.from_env()
    assert settings.model_backend == "mock"
    assert settings.retrieval_url == "http://localhost:8001"
    assert settings.model_version == "dev"
    assert settings.top_k == 5
    assert settings.max_tokens == 512
    assert settings.max_tool_iterations == 2


def test_reads_env(monkeypatch):
    monkeypatch.setenv("MODEL_BACKEND", "llm")
    monkeypatch.setenv("MODEL_VERSION", "v6.1")
    monkeypatch.setenv("TOP_K", "8")
    monkeypatch.setenv("MAX_TOKENS", "2048")
    monkeypatch.setenv("MAX_TOOL_ITERATIONS", "3")
    settings = Settings.from_env()
    assert settings.model_backend == "llm"
    assert settings.model_version == "v6.1"
    assert settings.top_k == 8
    assert settings.max_tokens == 2048
    assert settings.max_tool_iterations == 3
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_config.py -q`
Expected: FAIL — `AttributeError: 'Settings' object has no attribute 'max_tool_iterations'`.

- [ ] **Step 3: Add the field**

In `src/medical_qa_platform/config.py`, add the field after `max_tokens: int = 512`:

```python
    max_tool_iterations: int = 2
```

and add to the `from_env` return (after the `max_tokens=` line):

```python
            max_tool_iterations=int(os.environ.get("MAX_TOOL_ITERATIONS", "2")),
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_config.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/medical_qa_platform/config.py tests/test_config.py
git commit -m "$(printf 'feat(config): add MAX_TOOL_ITERATIONS (default 2)\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

### Task 5: Response schema — `Turn` + `trace`

**Files:**
- Modify: `src/medical_qa_platform/api/schemas.py`
- Modify: `tests/api/test_schemas.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/api/test_schemas.py`:

```python
def test_response_carries_trace_of_turns():
    from medical_qa_platform.api.schemas import Turn

    resp = PredictResponse(
        answer="A",
        raw_output="<answer>A</answer>",
        evidence=[],
        trace=[
            Turn(
                role="assistant",
                content="<think>t</think>",
                tool_calls=[{"function": {"name": "search_medical_knowledge"}}],
            ),
            Turn(role="tool", content="- fact"),
        ],
        backend="mock",
        model_version="dev",
        contract_version="v1",
        latency_ms=1.0,
        trace_id="abc",
    )
    dumped = resp.model_dump()
    assert dumped["trace"][0]["role"] == "assistant"
    assert dumped["trace"][1]["tool_calls"] is None


def test_response_trace_defaults_to_empty():
    resp = PredictResponse(
        answer="A", raw_output="x", evidence=[], backend="mock",
        model_version="dev", contract_version="v1", latency_ms=1.0, trace_id="t",
    )
    assert resp.trace == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/api/test_schemas.py -q`
Expected: FAIL — `ImportError: cannot import name 'Turn'`.

- [ ] **Step 3: Add `Turn` and `trace`**

In `src/medical_qa_platform/api/schemas.py`, add a `Turn` model before `PredictResponse`, and add the `trace` field. The `Field` import is already present. The file becomes:

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


class Turn(BaseModel):
    """One turn of the agentic transcript, for display."""

    role: str
    content: str
    tool_calls: list[dict] | None = None


class PredictResponse(BaseModel):
    answer: str | None
    raw_output: str
    evidence: list[str]
    trace: list[Turn] = Field(default_factory=list)
    backend: str
    model_version: str
    contract_version: str
    latency_ms: float
    trace_id: str
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/api/test_schemas.py -q`
Expected: PASS (existing tests still pass — `trace` defaults to `[]`).

- [ ] **Step 5: Commit**

```bash
git add src/medical_qa_platform/api/schemas.py tests/api/test_schemas.py
git commit -m "$(printf 'feat(api): add Turn model and trace to PredictResponse\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

### Task 6: Wire `/predict` to the agentic loop

**Files:**
- Modify: `src/medical_qa_platform/api/app.py`
- Modify: `tests/api/test_app.py`

- [ ] **Step 1: Update / add the failing API tests**

In `tests/api/test_app.py`, replace `test_predict_returns_all_fields` with the version below (the mock makes no tool call, so evidence is now empty and the headline is the trace):

```python
def test_predict_returns_all_fields(tmp_path):
    client = _client(tmp_path)
    resp = client.post("/predict", json={"question": "Q?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "B"
    assert "<answer>B</answer>" in body["raw_output"]
    # pure model-driven: the mock backend makes no tool call, so no evidence gathered
    assert body["evidence"] == []
    assert body["trace"][-1]["role"] == "assistant"
    assert "<answer>B</answer>" in body["trace"][-1]["content"]
    assert body["backend"] == "mock"
    assert body["model_version"] == "test-v1"
    assert body["latency_ms"] >= 0
    assert body["trace_id"]
```

Then append two new tests:

```python
def test_predict_runs_tool_loop_and_returns_trace(tmp_path):
    class _ToolBackend(ModelBackend):
        name = "tool"

        def __init__(self):
            self._turns = [
                ChatTurn(
                    content="<think>need facts</think>",
                    tool_calls=[{
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "search_medical_knowledge",
                            "arguments": '{"query": "T2DM"}',
                        },
                    }],
                    finish_reason="tool_calls",
                ),
                ChatTurn(content="<answer>A</answer>", finish_reason="stop"),
            ]

        def chat(self, messages, tools=None, tool_choice="auto", max_tokens=512, temperature=0.3):
            return self._turns.pop(0)

    app = create_app(
        backend=_ToolBackend(),
        retrieval=FixtureRetrieval({"T2DM": ["metformin is first-line"]}),
        model_version="x",
        drift_log_path=str(tmp_path / "d.jsonl"),
    )
    body = TestClient(app).post("/predict", json={"question": "Q?"}).json()
    assert body["answer"] == "A"
    assert body["evidence"] == ["metformin is first-line"]
    assert [t["role"] for t in body["trace"]] == ["assistant", "tool", "assistant"]


def test_predict_skips_retrieval_when_model_makes_no_tool_call(tmp_path):
    class _SpyRetrieval(FixtureRetrieval):
        def __init__(self):
            super().__init__({"Q?": ["should-not-be-used"]})
            self.called = False

        def search(self, query, top_k):
            self.called = True
            return super().search(query, top_k)

    spy = _SpyRetrieval()
    app = create_app(
        backend=MockBackend(answer="B"),
        retrieval=spy,
        model_version="x",
        drift_log_path=str(tmp_path / "d.jsonl"),
    )
    TestClient(app).post("/predict", json={"question": "Q?"})
    assert spy.called is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/api/test_app.py -q`
Expected: FAIL — `test_predict_returns_all_fields` (evidence still injected), `test_predict_runs_tool_loop_and_returns_trace` (`trace` missing / no loop), `test_predict_skips_retrieval...` (pre-retrieval still calls search).

- [ ] **Step 3: Rewrite the `/predict` handler to use the loop**

In `src/medical_qa_platform/api/app.py`:

Change the imports block — remove `from .prompt import build_prompt`, and update the schema + add the loop import. The import section becomes:

```python
from ..config import Settings
from ..drift.collector import DriftCollector
from ..inference.agent import run_agentic_loop
from ..observability.metrics import observe_request, render_metrics
from ..retrieval.backends import SupportsSearch
from ..retrieval.contract import RETRIEVAL_CONTRACT_VERSION
from .parser import parse_answer
from .schemas import PredictRequest, PredictResponse, Turn
```

Add the `max_tool_iterations` parameter to `create_app` (after `max_tokens: int | None = None,`):

```python
    max_tool_iterations: int | None = None,
```

Add the state assignment after the `app.state.max_tokens = ...` block:

```python
    app.state.max_tool_iterations = (
        max_tool_iterations
        if max_tool_iterations is not None
        else settings.max_tool_iterations
    )
```

Replace the whole `predict` handler body with:

```python
    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        t0 = time.perf_counter()
        trace_id = uuid.uuid4().hex
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
        observe_request(
            endpoint="/predict",
            backend=app.state.backend.name,
            status="ok" if answer is not None else "no_answer",
            latency_s=latency_ms / 1000.0,
        )
        app.state.collector.record(req, resp, n_evidence=len(result.evidence))
        return resp
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/api/test_app.py -q`
Expected: PASS. (`test_predict_passes_configured_max_tokens` still passes: the recording backend returns no tool call, so the loop calls `chat` once with `max_tokens=2048`.)

- [ ] **Step 5: Run the full suite (handler change touches metrics/drift)**

Run: `.venv/bin/pytest -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/medical_qa_platform/api/app.py tests/api/test_app.py
git commit -m "$(printf 'feat(api): drive /predict with the agentic tool-call loop\n\n/predict now runs run_agentic_loop (no forced pre-retrieval); the model\ndecides when to search. Response carries the per-turn trace; the answer\nletter is still parsed server-side for metrics/drift.\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

### Task 7: UI client — `trace`, transcript blocks, longer timeout

**Files:**
- Modify: `app/client.py`
- Modify: `tests/app/test_ui_client.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/app/test_ui_client.py`:

```python
def test_predict_parses_trace():
    def handler(request):
        return httpx.Response(200, json={
            "answer": "A",
            "raw_output": "<answer>A</answer>",
            "evidence": [],
            "trace": [
                {"role": "assistant", "content": "<think>t</think>",
                 "tool_calls": [{"function": {"name": "search_medical_knowledge",
                                              "arguments": '{"query": "metformin"}'}}]},
                {"role": "tool", "content": "- fact", "tool_calls": None},
                {"role": "assistant", "content": "<answer>A</answer>", "tool_calls": None},
            ],
            "backend": "llm", "model_version": "v", "contract_version": "c",
            "latency_ms": 1.0, "trace_id": "t",
        })

    res = predict("http://gw/", "k", {"question": "q"}, client=_client(handler))
    assert [t["role"] for t in res.trace] == ["assistant", "tool", "assistant"]


def test_predict_trace_defaults_to_empty_when_absent():
    def handler(request):
        return httpx.Response(200, json={
            "answer": "A", "raw_output": "x", "evidence": [], "backend": "llm",
            "model_version": "v", "contract_version": "c", "latency_ms": 1.0, "trace_id": "t",
        })

    res = predict("http://gw/", "k", {"question": "q"}, client=_client(handler))
    assert res.trace == []


def test_build_transcript_blocks_labels_tool_calls_and_results():
    from app.client import build_transcript_blocks

    trace = [
        {"role": "assistant", "content": "<think>t</think>",
         "tool_calls": [{"function": {"name": "search_medical_knowledge",
                                      "arguments": '{"query": "metformin"}'}}]},
        {"role": "tool", "content": "- metformin is first-line", "tool_calls": None},
        {"role": "assistant", "content": "<answer>A</answer>", "tool_calls": None},
    ]
    blocks = build_transcript_blocks(trace)
    assert len(blocks) == 3
    assert "metformin" in blocks[0][0]            # query surfaced in the assistant label
    assert blocks[1][0].startswith("📚")           # tool result block
    assert blocks[2][1] == "<answer>A</answer>"    # final assistant body
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/app/test_ui_client.py -q`
Expected: FAIL — `PredictResult` has no `trace`; `build_transcript_blocks` not importable.

- [ ] **Step 3: Update `app/client.py`**

Add `import json` to the imports at the top (after `from dataclasses import dataclass`):

```python
import json
```

Add `trace: list[dict]` to the `PredictResult` dataclass, immediately after the `evidence` field:

```python
    trace: list[dict]
```

In `predict(...)`, change the default timeout from `30.0` to `120.0`:

```python
def predict(
    base_url: str,
    api_key: str | None,
    payload: dict,
    timeout: float = 120.0,
    client: httpx.Client | None = None,
) -> PredictResult:
```

and add `trace=data.get("trace", [])` to the returned `PredictResult(...)`, immediately after the `evidence=` line:

```python
        trace=data.get("trace", []),
```

Add the pure transcript-builder function at the end of the file:

```python
def build_transcript_blocks(trace: list[dict]) -> list[tuple[str, str]]:
    """Turn the API trace into (label, body) blocks for display.

    Assistant turns are labelled with the search query they issued (if any);
    tool turns are labelled as KG results. Pure — no Streamlit import — so it is
    unit-testable in the base venv.
    """
    blocks: list[tuple[str, str]] = []
    for turn in trace:
        role = turn.get("role")
        content = turn.get("content") or ""
        if role == "tool":
            blocks.append(("📚 Kết quả tri thức (KG)", content))
            continue
        label = "🤖 Trợ lý"
        calls = turn.get("tool_calls") or []
        if calls:
            queries = []
            for call in calls:
                args = call.get("function", {}).get("arguments", "")
                try:
                    parsed = json.loads(args) if isinstance(args, str) else args
                    query = parsed.get("query", "") if isinstance(parsed, dict) else ""
                except (json.JSONDecodeError, AttributeError, TypeError):
                    query = args if isinstance(args, str) else ""
                queries.append(str(query))
            label = "🤖 Trợ lý · 🔎 search_medical_knowledge(" + "; ".join(queries) + ")"
        blocks.append((label, content))
    return blocks
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/app/test_ui_client.py -q`
Expected: PASS. (`test_predict_parses_success_and_sends_key` still passes — `trace` defaults to `[]`.)

- [ ] **Step 5: Commit**

```bash
git add app/client.py tests/app/test_ui_client.py
git commit -m "$(printf 'feat(ui): parse trace + build_transcript_blocks; 120s timeout\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

### Task 8: Streamlit — render the transcript

**Files:**
- Modify: `app/streamlit_app.py`
- Create: `tests/app/test_streamlit_transcript.py`

- [ ] **Step 1: Write the failing wiring test**

Create `tests/app/test_streamlit_transcript.py`:

```python
import pytest

pytest.importorskip("streamlit")


def test_streamlit_app_uses_transcript_builder():
    import app.streamlit_app as ui
    from app.client import build_transcript_blocks

    # the UI must render via the pure builder (so rendering logic stays testable)
    assert ui.build_transcript_blocks is build_transcript_blocks
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/app/test_streamlit_transcript.py -q`
Expected: FAIL — `AttributeError: module 'app.streamlit_app' has no attribute 'build_transcript_blocks'`.

- [ ] **Step 3: Update the import and rendering in `app/streamlit_app.py`**

Change the `from app.client import (...)` block to include `build_transcript_blocks`:

```python
from app.client import (  # noqa: E402  (after sys.path bootstrap above)
    PredictError,
    build_payload,
    build_transcript_blocks,
    fetch_version,
    predict,
)
```

Replace the result-rendering block (from `if result.answer is None:` through the final `st.caption(...)` call, i.e. the current lines that show the success/warning, the "Phản hồi thô" expander, the evidence expander, and the metadata caption) with:

```python
    # The headline is the per-turn transcript now; the parsed letter is a small badge.
    if result.answer:
        st.caption(f"Đáp án (parse tự động): **{result.answer}**")

    st.subheader("Diễn tiến suy luận")
    blocks = build_transcript_blocks(result.trace)
    if blocks:
        for label, body in blocks:
            st.markdown(f"**{label}**")
            st.code(body or "(rỗng)")
    else:
        # No trace (older API / single-turn): fall back to the raw final output.
        st.code(result.raw_output or "(rỗng)")

    with st.expander(f"Bằng chứng KG ({len(result.evidence)})", expanded=False):
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

- [ ] **Step 4: Run the test to verify it passes, and confirm the app still imports**

Run: `.venv/bin/pytest tests/app/test_streamlit_transcript.py tests/app/test_streamlit_presets.py -q`
Expected: PASS (the presets AppTest confirms the app still runs end-to-end).

- [ ] **Step 5: Commit**

```bash
git add app/streamlit_app.py tests/app/test_streamlit_transcript.py
git commit -m "$(printf 'feat(ui): render per-turn transcript instead of a single raw block\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

### Task 9: Guard the `--jinja` server flag

**Files:**
- Modify: `tests/deploy/test_kserve_chart.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/deploy/test_kserve_chart.py`:

```python
def test_kserve_server_runs_with_jinja_for_tool_calling():
    # --jinja enables OpenAI tool-calling via the model's chat template; the
    # serving agentic loop depends on the server returning structured tool_calls.
    resources = render_chart("kserve")
    isvc = find_kind(resources, "InferenceService", "medical-qa-kserve")
    args = isvc["spec"]["predictor"]["containers"][0]["args"]
    assert "--jinja" in args
```

- [ ] **Step 2: Run the test to verify it passes immediately**

Run: `.venv/bin/pytest tests/deploy/test_kserve_chart.py -q`
Expected: PASS — `--jinja` is already in the chart args; this test is a regression guard so a future edit can't silently break tool calling.

(There is no implementation step: this task only adds the guard. If the test unexpectedly FAILS, the chart lost `--jinja` and must be restored in `deploy/helm/kserve/templates/inferenceservice.yaml`.)

- [ ] **Step 3: Commit**

```bash
git add tests/deploy/test_kserve_chart.py
git commit -m "$(printf 'test(deploy): guard --jinja flag the tool-calling loop depends on\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

### Final verification

- [ ] **Run the full suite**

Run: `.venv/bin/pytest -q`
Expected: PASS — baseline 250 passed/1 skipped, plus the new tests (Task 1: +5, Task 2: +1, Task 3: +5, Task 4: 0 net new, Task 5: +2, Task 6: +2, Task 7: +3, Task 8: +1, Task 9: +1).

- [ ] **Sanity-check the import graph** (no circular import from `inference.agent` → `api.prompt`)

Run: `.venv/bin/python -c "import medical_qa_platform.api.app; import medical_qa_platform.inference.agent; print('ok')"`
Expected: prints `ok`.

- [ ] Hand off to **superpowers:finishing-a-development-branch**.
