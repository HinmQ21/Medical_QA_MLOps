from fastapi.testclient import TestClient

from medical_qa_platform.api.app import create_app
from medical_qa_platform.inference.base import ChatTurn, ModelBackend
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
    # pure model-driven: the mock backend makes no tool call, so no evidence gathered
    assert body["evidence"] == []
    assert body["trace"][-1]["role"] == "assistant"
    assert "<answer>B</answer>" in body["trace"][-1]["content"]
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

    def chat(self, messages, tools=None, tool_choice="auto", max_tokens=512, temperature=0.3):
        self.max_tokens_calls.append(max_tokens)
        return ChatTurn(content="<answer>A</answer>", tool_calls=[], finish_reason="stop")


class _UnhealthyBackend(ModelBackend):
    name = "down"

    def chat(self, messages, tools=None, tool_choice="auto", max_tokens=512, temperature=0.3):
        return ChatTurn(content="<answer>A</answer>", tool_calls=[], finish_reason="stop")

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
    assert backend.max_tokens_calls  # at least one chat call happened
    assert all(t == 2048 for t in backend.max_tokens_calls)


def test_ready_returns_503_when_backend_unhealthy(tmp_path):
    app = create_app(
        backend=_UnhealthyBackend(),
        retrieval=FixtureRetrieval({}),
        drift_log_path=str(tmp_path / "d.jsonl"),
    )
    assert TestClient(app).get("/ready").status_code == 503


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


def test_metrics_include_model_and_tool_and_build_info(tmp_path):
    client = _client(tmp_path)
    client.post("/predict", json={"question": "Q?"})
    text = client.get("/metrics").text
    assert "mqa_model_latency_seconds" in text
    assert "mqa_tool_outcome_total" in text
    assert 'outcome="not_called"' in text  # MockBackend makes no tool call
    assert "mqa_build_info" in text
