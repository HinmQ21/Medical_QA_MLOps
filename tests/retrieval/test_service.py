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


def test_retrieval_version_reports_contract_encoder_and_kg_dir(monkeypatch):
    from medical_qa_platform.retrieval.contract import RETRIEVAL_CONTRACT_VERSION

    monkeypatch.setenv("KG_ENCODER_MODEL", "abhinand/MedEmbed-small-v0.1")
    monkeypatch.setenv("KG_DATA_DIR", "/mnt/artifacts/kg")
    body = _client().get("/version").json()
    assert body["contract_version"] == RETRIEVAL_CONTRACT_VERSION
    assert body["encoder_model"] == "abhinand/MedEmbed-small-v0.1"
    assert body["kg_data_dir"] == "/mnt/artifacts/kg"


def test_retrieval_version_defaults_to_small_encoder(monkeypatch):
    monkeypatch.delenv("KG_ENCODER_MODEL", raising=False)
    assert _client().get("/version").json()["encoder_model"] == "abhinand/MedEmbed-small-v0.1"
