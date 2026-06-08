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
    assert retrieval.queries == []  # empty query -> retrieval not called
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


def test_multiple_tool_calls_in_one_turn_each_get_a_tool_message():
    backend = ScriptedBackend([
        ChatTurn(content="<think>two lookups</think>",
                 tool_calls=[_tool_call("c1", "q1"), _tool_call("c2", "q2")],
                 finish_reason="tool_calls"),
        ChatTurn(content="<answer>A</answer>", finish_reason="stop"),
    ])
    retrieval = FixtureRetrieval({"q1": ["fact one"], "q2": ["fact two"]})
    res = run_agentic_loop(backend, retrieval, "Q?", top_k=5, max_tokens=512, max_iterations=2)
    assert res.tool_call_count == 2
    assert res.evidence == ["fact one", "fact two"]
    second_call_msgs = backend.calls[1]["messages"]
    tool_ids = [m["tool_call_id"] for m in second_call_msgs if m["role"] == "tool"]
    assert tool_ids == ["c1", "c2"]


def test_max_iterations_zero_answers_in_one_no_tools_call():
    backend = ScriptedBackend([ChatTurn(content="<answer>B</answer>", finish_reason="stop")])
    retrieval = FixtureRetrieval({"q": ["unused"]})
    res = run_agentic_loop(backend, retrieval, "Q?", top_k=5, max_tokens=512, max_iterations=0)
    assert res.final_content == "<answer>B</answer>"
    assert res.tool_call_count == 0
    assert len(backend.calls) == 1
    assert backend.calls[0]["tools_none"] is True
