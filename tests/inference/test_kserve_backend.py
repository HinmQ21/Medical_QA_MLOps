import json

import httpx

from medical_qa_platform.inference.kserve_backend import KServeBackend


def _backend(captured: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"predictions": [{"text": "<answer>A</answer>"}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return KServeBackend(url="http://ks/v1/models/mock:predict", client=client)


def test_name():
    assert _backend({}).name == "kserve"


def test_generate_returns_text():
    out = _backend({}).generate([{"role": "user", "content": "x"}])
    assert out == "<answer>A</answer>"


def test_posts_instances_payload():
    captured = {}
    _backend(captured).generate([{"role": "user", "content": "x"}])
    assert captured["url"] == "http://ks/v1/models/mock:predict"
    assert captured["body"]["instances"][0]["messages"] == [
        {"role": "user", "content": "x"}
    ]


def test_from_env(monkeypatch):
    monkeypatch.setenv("KSERVE_URL", "http://ks/v1/models/m:predict")
    assert KServeBackend.from_env().url == "http://ks/v1/models/m:predict"
