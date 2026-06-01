from fastapi.testclient import TestClient

from medical_qa_platform.inference.kserve_backend import KServeBackend
from medical_qa_platform.serving.kserve_mock_app import create_app


def test_kserve_mock_health_and_ready():
    client = TestClient(create_app())
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json() == {"status": "ready"}


def test_kserve_mock_predict_matches_kserve_backend_contract():
    client = TestClient(create_app())
    backend = KServeBackend(
        url="/v1/models/medical-qa-smoke:predict",
        client=client,
    )

    text = backend.generate([{"role": "user", "content": "pick A"}])

    assert text == "<think>Mock reasoning for 1 messages.</think><answer>A</answer>"


def test_kserve_mock_predict_response_shape():
    client = TestClient(create_app())
    response = client.post(
        "/v1/models/medical-qa-smoke:predict",
        json={"instances": [{"messages": [{"role": "user", "content": "pick A"}]}]},
    )
    assert response.status_code == 200
    assert response.json() == {
        "predictions": [{"text": "<think>Mock reasoning for 1 messages.</think><answer>A</answer>"}]
    }


def test_kserve_mock_predict_short_path_for_local_smoke():
    client = TestClient(create_app())
    response = client.post(
        "/predict",
        json={"instances": [{"messages": [{"role": "user", "content": "pick A"}]}]},
    )
    assert response.status_code == 200
    assert response.json()["predictions"][0]["text"].endswith("<answer>A</answer>")
