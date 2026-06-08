from medical_qa_platform.config import Settings


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
