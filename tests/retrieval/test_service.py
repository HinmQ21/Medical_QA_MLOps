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
