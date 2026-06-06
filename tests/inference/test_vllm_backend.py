import json

import httpx

from medical_qa_platform.inference.vllm_backend import VllmBackend


def _backend(captured: dict, *, status: int = 200, json_body=None):
    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["method"] = request.method
        captured["headers"] = dict(request.headers)
        if request.method == "POST":
            captured["body"] = json.loads(request.content)
        return httpx.Response(
            status,
            json=json_body
            if json_body is not None
            else {"choices": [{"message": {"content": "<answer>B</answer>"}}]},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return VllmBackend(base_url="http://pod/v1", model="m", api_key="k", client=client)


def test_name():
    assert _backend({}).name == "vllm"


def test_generate_returns_content():
    out = _backend({}).generate([{"role": "user", "content": "x"}])
    assert out == "<answer>B</answer>"


def test_posts_to_chat_completions_with_auth_model_and_max_tokens():
    captured = {}
    _backend(captured).generate([{"role": "user", "content": "x"}], max_tokens=2048)
    assert captured["url"] == "http://pod/v1/chat/completions"
    assert captured["headers"]["authorization"] == "Bearer k"
    assert captured["body"]["model"] == "m"
    assert captured["body"]["max_tokens"] == 2048
    assert captured["body"]["messages"] == [{"role": "user", "content": "x"}]


def test_from_env(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://x/v1")
    monkeypatch.setenv("LLM_MODEL", "llama")
    monkeypatch.setenv("LLM_API_KEY", "secret")
    backend = VllmBackend.from_env()
    assert backend.base_url == "http://x/v1"
    assert backend.model == "llama"
    assert backend.api_key == "secret"


def test_health_check_true_on_200_models():
    captured = {}
    ok = _backend(captured, json_body={"data": []}).health_check()
    assert ok is True
    assert captured["url"] == "http://pod/v1/models"
    assert captured["method"] == "GET"
    assert captured["headers"]["authorization"] == "Bearer k"


def test_health_check_false_on_bad_status():
    backend = _backend({}, status=503, json_body={"error": "down"})
    assert backend.health_check() is False


def test_health_check_false_on_transport_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    backend = VllmBackend(base_url="http://pod/v1", model="m", client=client)
    assert backend.health_check() is False
