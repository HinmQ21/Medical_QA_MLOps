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


def test_predict_response_includes_contract_version(tmp_path):
    from medical_qa_platform.retrieval.contract import RETRIEVAL_CONTRACT_VERSION

    client = _client(tmp_path)
    resp = client.post(
        "/predict", json={"question": "Q?", "options": {"A": "a", "B": "b"}}
    )
    assert resp.json()["contract_version"] == RETRIEVAL_CONTRACT_VERSION


def test_version_endpoint_reports_contract_and_model(tmp_path):
    from medical_qa_platform.retrieval.contract import RETRIEVAL_CONTRACT_VERSION

    body = _client(tmp_path).get("/version").json()
    assert body["contract_version"] == RETRIEVAL_CONTRACT_VERSION
    assert body["model_version"] == "test-v1"
    assert body["backend"] == "mock"
