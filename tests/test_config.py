from medical_qa_platform.config import Settings


def test_defaults(monkeypatch):
    for var in [
        "MODEL_BACKEND",
        "RETRIEVAL_URL",
        "MODEL_VERSION",
        "TOP_K",
        "DRIFT_LOG_PATH",
    ]:
        monkeypatch.delenv(var, raising=False)
    settings = Settings.from_env()
    assert settings.model_backend == "mock"
    assert settings.retrieval_url == "http://localhost:8001"
    assert settings.model_version == "dev"
    assert settings.top_k == 5


def test_reads_env(monkeypatch):
    monkeypatch.setenv("MODEL_BACKEND", "runpod")
    monkeypatch.setenv("MODEL_VERSION", "v6.1")
    monkeypatch.setenv("TOP_K", "8")
    settings = Settings.from_env()
    assert settings.model_backend == "runpod"
    assert settings.model_version == "v6.1"
    assert settings.top_k == 8
