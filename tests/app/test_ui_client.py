import httpx
import pytest

from app.client import (
    PredictError,
    PredictResult,
    build_payload,
    fetch_version,
    predict,
)


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_build_payload_strips_question():
    assert build_payload("  first-line? ") == {"question": "first-line?"}


def test_build_payload_rejects_blank_question():
    with pytest.raises(ValueError):
        build_payload("   ")


def test_predict_parses_success_and_sends_key():
    def handler(request):
        assert request.url.path == "/predict"
        assert request.headers["x-api-key"] == "k"
        return httpx.Response(
            200,
            json={
                "answer": "A",
                "raw_output": "<think>r</think><answer>A</answer>",
                "evidence": ["e1", "e2"],
                "backend": "llm",
                "model_version": "smoke-dev",
                "contract_version": "v1",
                "latency_ms": 12.5,
                "trace_id": "abc",
            },
        )

    res = predict("http://gw:8080/", "k", {"question": "q"}, client=_client(handler))
    assert isinstance(res, PredictResult)
    assert res.answer == "A"
    assert res.raw_output == "<think>r</think><answer>A</answer>"
    assert res.evidence == ["e1", "e2"]
    assert res.backend == "llm"
    assert res.trace_id == "abc"


def test_predict_raises_predicterror_on_401():
    def handler(request):
        return httpx.Response(401)

    with pytest.raises(PredictError, match="unauthorized"):
        predict("http://gw:8080", "bad", {"question": "q"}, client=_client(handler))


def test_predict_raises_predicterror_on_timeout():
    def handler(request):
        raise httpx.TimeoutException("slow")

    with pytest.raises(PredictError, match="timed out"):
        predict("http://gw:8080", "k", {"question": "q"}, timeout=3, client=_client(handler))


def test_predict_raises_predicterror_on_connection_error():
    def handler(request):
        raise httpx.ConnectError("nope")

    with pytest.raises(PredictError, match="could not reach"):
        predict("http://gw:8080", "k", {"question": "q"}, client=_client(handler))


def test_fetch_version_returns_json():
    def handler(request):
        assert request.url.path == "/version"
        return httpx.Response(
            200,
            json={"backend": "llm", "model_version": "smoke-dev", "contract_version": "v1"},
        )

    out = fetch_version("http://gw:8080", "k", client=_client(handler))
    assert out["backend"] == "llm"


def test_fetch_version_raises_predicterror_on_401():
    def handler(request):
        return httpx.Response(401)

    with pytest.raises(PredictError, match="unauthorized"):
        fetch_version("http://gw:8080", "k", client=_client(handler))


def test_fetch_version_raises_predicterror_on_connection_error():
    def handler(request):
        raise httpx.ConnectError("down")

    with pytest.raises(PredictError, match="could not reach"):
        fetch_version("http://gw:8080", "k", client=_client(handler))


def test_predict_raises_predicterror_on_non_json_body():
    def handler(request):
        return httpx.Response(200, text="<html>oops</html>")

    with pytest.raises(PredictError):
        predict("http://gw:8080", "k", {"question": "q"}, client=_client(handler))
