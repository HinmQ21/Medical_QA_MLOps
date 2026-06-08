# Agentic Tool-Call Serving Loop — Design

**Date:** 2026-06-08
**Status:** Approved (brainstorming complete)
**Repo:** `mlops-platform`

## Problem

The serving `/predict` flow is **one-shot RAG**: retrieve KG evidence once, inject it
into the prompt, call the model once, parse the `<answer>` letter, return. This diverges
structurally from the **training/eval flow** (`baseline/scripts/benchmark/grpo_eval/grpo_eval.py`),
where the model itself drives retrieval — it emits a tool call, the harness executes
`search_medical_knowledge`, injects the result back, and the model continues over multiple
turns until it produces `<answer>`. The existing serving prompt even documents this gap
(`api/prompt.py` docstring: "tracked as a separate sub-project").

This redesign makes serving match training: a **model-driven agentic tool-call loop**.
It also changes the UI to show the **per-turn transcript** as raw blocks instead of
surfacing a parsed answer letter as the headline.

## Decisions (locked during brainstorming)

1. **Model target:** in-cluster **Qwen2.5-1.5B-Instruct-GGUF** (llama.cpp), the current GKE
   demo model. It is a base instruct model (not the RL checkpoint), but Qwen2.5-Instruct was
   post-trained for Hermes-style tool use, so it can emit tool calls — the loop must tolerate
   turns where it skips the tool.
2. **Tool-calling mechanism:** **native OpenAI function-calling** (`tools=` param). The server
   returns structured `tool_calls` + `finish_reason`. The kserve llama.cpp server **already runs
   with `--jinja`** (`deploy/helm/kserve/templates/inferenceservice.yaml`), which enables
   OpenAI-compatible tool calling via the model's embedded chat template — **no chart change needed**.
3. **UI display:** **per-turn transcript** — the API returns a `trace` array of `{role, content,
   tool_calls}` messages; the UI renders each turn as a labeled raw block. The parsed letter
   drops to a small secondary badge.
4. **Answer field:** **keep parsing server-side.** `parse_answer()` still runs on the final turn;
   the `answer` field stays in the response so metrics and the drift collector are unchanged.
5. **No-tool turn:** **pure model-driven.** If the model answers without calling the tool, evidence
   is empty and that is accepted. No forced first-pass retrieval fallback.

## Architecture

New flow for `POST /predict`:

```
messages = [system, user(question)]          # NO forced pre-retrieval — model decides
trace = [], evidence = []
for i in range(max_iterations + 1):          # max_iterations default 3 (matches training)
    # Last round offers no tools (and no tool_choice) so the model must answer in text.
    # Sending tool_choice without tools is rejected by some OpenAI servers, so omit both.
    if i == max_iterations:
        turn = backend.chat(messages, tools=None, max_tokens=max_tokens, temperature=0.3)
    else:
        turn = backend.chat(messages, tools=[MEDICAL_TOOL_DEF], tool_choice="auto",
                            max_tokens=max_tokens, temperature=0.3)
    messages.append(assistant_message(turn))  # echoes raw tool_calls verbatim
    trace.append(turn_view(assistant))
    if turn.tool_calls and i < max_iterations:
        for tc in turn.tool_calls:
            query = json-parse tc.function.arguments -> .get("query", "")  # "" on bad JSON
            results = retrieval.search(query, top_k)   # existing HTTP RetrievalClient
            evidence.extend(results)
            content = "\n".join(f"- {r}" for r in results) or "No relevant knowledge found."
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})
            trace.append(turn_view(tool))
        continue
    break                                       # finish_reason stop / no tool call / cap reached
final_content = turn.content or ""
answer = parse_answer(final_content)            # kept server-side
return PredictResponse(answer, raw_output=final_content, evidence=evidence,
                       trace=trace, backend, model_version, contract_version,
                       latency_ms, trace_id)
```

The loop lives entirely in the API layer. The retrieval service, the llama.cpp
InferenceService, and the metrics/drift schemas are untouched.

**Behavioral change to be explicit about:** there is no longer a forced first-pass RAG.
Grounding happens only when the model calls the tool. If it never calls the tool, `evidence`
is `[]` and the answer is from the model's parametric knowledge alone — faithful to the
training agentic intent.

## Components

### Backend interface — `src/medical_qa_platform/inference/base.py`, `llm_backend.py`, `mock_backend.py`

Native function-calling needs the structured response, not just text.

- New dataclass `ChatTurn`:
  ```python
  @dataclass
  class ChatTurn:
      content: str | None
      tool_calls: list[dict]   # raw OpenAI objects: {id, type, function:{name, arguments}}
      finish_reason: str
  ```
  `tool_calls` holds the **raw** server objects so the loop can echo them verbatim in the
  next request (OpenAI requires the assistant message's `tool_calls` and each following
  `tool` message's `tool_call_id` to match).
- `ModelBackend.chat(messages, tools=None, tool_choice="auto", max_tokens=512, temperature=0.3) -> ChatTurn`
  becomes the abstract method. `generate(...) -> str` stays as a concrete default wrapper
  (`return self.chat(messages, max_tokens=max_tokens, temperature=temperature).content or ""`)
  so existing callers and tests keep working.
- `LLMBackend.chat()` POSTs `tools`/`tool_choice` (omitting `tools` when `None`) to
  `/v1/chat/completions`, reads `choices[0].message.content`, `choices[0].message.tool_calls`
  (default `[]`), and `choices[0].finish_reason`, returns a `ChatTurn`.
- `MockBackend.chat()` returns a deterministic no-tool turn:
  `ChatTurn(content="<think>Mock reasoning for N messages.</think><answer>A</answer>",
  tool_calls=[], finish_reason="stop")`.

### Tool definition — `src/medical_qa_platform/inference/tools.py` (new)

Mirrors the training schema (`baseline/scripts/utils/model_adapter.py:MEDICAL_TOOL_DEF`):

```python
SEARCH_TOOL_NAME = "search_medical_knowledge"
MEDICAL_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": SEARCH_TOOL_NAME,
        "description": ("Search the medical knowledge base for relevant clinical "
                        "information about diseases, drugs, symptoms, and treatments."),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string",
                                     "description": "Medical query to search for."}},
            "required": ["query"],
        },
    },
}
```

Isolated in its own module so a Llama-family variant can be added later without touching the loop.

### Agentic loop — `src/medical_qa_platform/inference/agent.py` (new)

```python
@dataclass
class LoopResult:
    trace: list[dict]        # [{role, content, tool_calls?}] for the UI
    final_content: str
    evidence: list[str]      # union of all retrieved facts across tool calls
    tool_call_count: int

def run_agentic_loop(backend, retrieval, question, *, top_k, max_tokens, max_iterations) -> LoopResult
```

- Initial messages come from `build_prompt(question, [])` (reuse, empty evidence).
- Robustness for the base 1.5B model:
  - tool-call arguments that aren't valid JSON → `query = ""` (do not crash).
  - a turn with no tool calls → final, break.
  - hard cap at `max_iterations`; on the last round `tools` is omitted (and `tool_choice` with it), forcing a text answer.
  - `trace` entries store `content or ""` (an assistant turn carrying only `tool_calls` has
    `content=None`); the OpenAI `tool_call_id` and raw `tool_calls` are preserved in `messages`,
    not the simplified `trace` view.
  - tool result formatting matches training: `"\n".join(f"- {r}")`, or `"No relevant knowledge found."` when empty.
- Graceful degradation: if the server returns no `tool_calls` at all (e.g. `--jinja` off, or the
  model declines), the loop breaks after the first turn — a single-turn answer, no crash.

### Response + UI schema

- `src/medical_qa_platform/api/schemas.py`:
  ```python
  class Turn(BaseModel):
      role: str
      content: str
      tool_calls: list[dict] | None = None

  class PredictResponse(BaseModel):
      answer: str | None
      raw_output: str            # = final assistant content
      evidence: list[str]        # = union of all retrieved facts
      trace: list[Turn]          # NEW
      backend: str
      model_version: str
      contract_version: str
      latency_ms: float
      trace_id: str
  ```
- `src/medical_qa_platform/api/app.py`: `predict()` calls `run_agentic_loop(...)`, then
  `parse_answer(result.final_content)`; drops the pre-retrieval `search` + `build_prompt` call
  (the loop owns message construction). `observe_request(status="ok" if answer else "no_answer")`
  and `collector.record(...)` keep working (answer + evidence still present).

### UI — `app/client.py`, `app/streamlit_app.py`

- `PredictResult` dataclass gains `trace: list[dict]`; `predict()` reads `data.get("trace", [])`.
  Raise the `predict` default timeout from `30.0` → `120.0` (multi-turn CPU generation is slow:
  up to `max_iterations + 1` generations on a ~14–20 tok/s 1.5B).
- `streamlit_app.py`: replace the single "raw output" expander with a **transcript** section:
  - each `assistant` turn → a labeled block showing `content`, plus a
    `🔎 search_medical_knowledge(query=…)` line per tool call;
  - each `tool` turn → a `📚 KG result` block showing `content`.
  - The parsed letter becomes a small secondary badge (kept, de-emphasized), not the headline.
  - Keep the evidence expander (now the gathered facts) and the metadata caption.

### Config — `src/medical_qa_platform/config.py`

`Settings` gains `max_tool_iterations: int = 3` (env `MAX_TOOL_ITERATIONS`). Wired through
`create_app(max_tool_iterations=None)` → `app.state.max_tool_iterations`.

## Testing

- **Backend:** `chat()` posts `tools`/`tool_choice` and parses `ChatTurn` (content, raw
  tool_calls, finish_reason); `generate()` still returns content. Mock `chat()` returns the
  deterministic no-tool turn.
- **Agent loop (new `tests/inference/test_agent.py`):** executes a tool call → injects the
  result → continues; stops at `finish_reason="stop"`; respects `max_iterations` (final round
  forces an answer by omitting `tools`); a no-tool first turn ends the loop; malformed
  tool args degrade to `query=""`; `evidence` accumulates across rounds; assistant `tool_calls`
  are echoed verbatim into the resent messages.
- **API:** `/predict` returns a `trace`; `answer` is still parsed; retrieval is called **only**
  when the model emits a tool call (assert `retrieval.search` not called on a no-tool turn);
  evidence reflects gathered facts. Update `test_predict_returns_all_fields` (no more forced
  pre-retrieval injection).
- **UI:** client parses `trace`; transcript rendering shows assistant + tool blocks.
- **Chart guard:** assert `--jinja` stays present in the kserve InferenceService args
  (`tests/deploy/test_kserve_chart.py`) so tool calling can't silently regress.

## Non-goals

- No Llama-family tool path (Qwen-only now; tool def isolated so it's easy to add later).
- No InferenceService / llama.cpp change (`--jinja` already present).
- No forced pre-retrieval fallback (pure model-driven).
- No streaming — the full trace is returned at once.
- No change to the retrieval service, the metrics, or the drift schema.
- No GBNF/constrained decoding.
