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


def test_build_payload_strips_and_keeps_letters():
    payload = build_payload("  first-line? ", {"A": "Metformin ", "B": " Insulin"})
    assert payload == {
        "question": "first-line?",
        "options": {"A": "Metformin", "B": "Insulin"},
    }


def test_build_payload_rejects_blank_question():
    with pytest.raises(ValueError):
        build_payload("   ", {"A": "x", "B": "y"})


def test_build_payload_drops_empty_options_then_requires_two():
    with pytest.raises(ValueError):
        build_payload("q", {"A": "only", "B": "   "})


def test_build_payload_rejects_more_than_ten_options():
    opts = {chr(ord("A") + i): str(i) for i in range(11)}
    with pytest.raises(ValueError):
        build_payload("q", opts)


def test_build_payload_rejects_non_letter_key():
    with pytest.raises(ValueError):
        build_payload("q", {"A": "x", "1": "y"})


def test_predict_parses_success_and_sends_key():
    def handler(request):
        assert request.url.path == "/predict"
        assert request.headers["x-api-key"] == "k"
        return httpx.Response(
            200,
            json={
                "answer": "A",
                "evidence": ["e1", "e2"],
                "backend": "vllm",
                "model_version": "smoke-dev",
                "contract_version": "v1",
                "latency_ms": 12.5,
                "trace_id": "abc",
            },
        )

    res = predict(
        "http://gw:8080/",
        "k",
        {"question": "q", "options": {"A": "x", "B": "y"}},
        client=_client(handler),
    )
    assert isinstance(res, PredictResult)
    assert res.answer == "A"
    assert res.evidence == ["e1", "e2"]
    assert res.backend == "vllm"
    assert res.trace_id == "abc"


def test_predict_raises_predicterror_on_401():
    def handler(request):
        return httpx.Response(401)

    with pytest.raises(PredictError, match="unauthorized"):
        predict(
            "http://gw:8080",
            "bad",
            {"question": "q", "options": {"A": "x", "B": "y"}},
            client=_client(handler),
        )


def test_predict_raises_predicterror_on_timeout():
    def handler(request):
        raise httpx.TimeoutException("slow")

    with pytest.raises(PredictError, match="timed out"):
        predict(
            "http://gw:8080",
            "k",
            {"question": "q", "options": {"A": "x", "B": "y"}},
            timeout=3,
            client=_client(handler),
        )


def test_predict_raises_predicterror_on_connection_error():
    def handler(request):
        raise httpx.ConnectError("nope")

    with pytest.raises(PredictError, match="could not reach"):
        predict(
            "http://gw:8080",
            "k",
            {"question": "q", "options": {"A": "x", "B": "y"}},
            client=_client(handler),
        )


def test_fetch_version_returns_json():
    def handler(request):
        assert request.url.path == "/version"
        return httpx.Response(
            200,
            json={"backend": "vllm", "model_version": "smoke-dev", "contract_version": "v1"},
        )

    out = fetch_version("http://gw:8080", "k", client=_client(handler))
    assert out["backend"] == "vllm"
