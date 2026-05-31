import json

import httpx

from medical_qa_platform.retrieval.client import RetrievalClient


def _client(captured: dict, results):
    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200, json=[{"query": captured["body"]["queries"][0], "results": results}]
        )

    http = httpx.Client(transport=httpx.MockTransport(handler))
    return RetrievalClient(base_url="http://ret:8001", client=http)


def test_search_returns_results_list():
    out = _client({}, ["evidence a", "evidence b"]).search("diabetes", top_k=5)
    assert out == ["evidence a", "evidence b"]


def test_search_posts_expected_body():
    captured = {}
    _client(captured, []).search("q", top_k=3)
    assert captured["url"] == "http://ret:8001/search"
    assert captured["body"] == {"queries": ["q"], "top_k": 3}


def test_search_empty_response_returns_empty():
    def handler(request):
        return httpx.Response(200, json=[])

    http = httpx.Client(transport=httpx.MockTransport(handler))
    client = RetrievalClient(base_url="http://ret:8001", client=http)
    assert client.search("q", top_k=5) == []
