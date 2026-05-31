import json

import httpx

from medical_qa_platform.inference.runpod_backend import RunpodBackend


def _backend(captured: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "<answer>B</answer>"}}]},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return RunpodBackend(
        base_url="http://pod/v1", model="m", api_key="k", client=client
    )


def test_name():
    assert _backend({}).name == "runpod"


def test_generate_returns_content():
    out = _backend({}).generate([{"role": "user", "content": "x"}])
    assert out == "<answer>B</answer>"


def test_posts_to_chat_completions_with_auth_and_model():
    captured = {}
    _backend(captured).generate([{"role": "user", "content": "x"}])
    assert captured["url"] == "http://pod/v1/chat/completions"
    assert captured["headers"]["authorization"] == "Bearer k"
    assert captured["body"]["model"] == "m"
    assert captured["body"]["messages"] == [{"role": "user", "content": "x"}]


def test_from_env(monkeypatch):
    monkeypatch.setenv("RUNPOD_BASE_URL", "http://x/v1")
    monkeypatch.setenv("RUNPOD_MODEL", "qwen")
    monkeypatch.setenv("RUNPOD_API_KEY", "secret")
    backend = RunpodBackend.from_env()
    assert backend.base_url == "http://x/v1"
    assert backend.model == "qwen"
