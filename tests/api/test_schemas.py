import pytest
from pydantic import ValidationError

from medical_qa_platform.api.schemas import PredictRequest, PredictResponse


def test_valid_request():
    req = PredictRequest(question="Q?", options={"A": "a", "B": "b"})
    assert req.options["A"] == "a"


def test_rejects_single_option():
    with pytest.raises(ValidationError):
        PredictRequest(question="Q?", options={"A": "a"})


def test_rejects_non_letter_key():
    with pytest.raises(ValidationError):
        PredictRequest(question="Q?", options={"A": "a", "1": "b"})


def test_rejects_multichar_key():
    with pytest.raises(ValidationError):
        PredictRequest(question="Q?", options={"A": "a", "AB": "b"})


def test_rejects_empty_question():
    with pytest.raises(ValidationError):
        PredictRequest(question="   ", options={"A": "a", "B": "b"})


def test_rejects_too_many_options():
    opts = {chr(ord("A") + i): str(i) for i in range(11)}
    with pytest.raises(ValidationError):
        PredictRequest(question="Q?", options=opts)


def test_response_round_trip():
    resp = PredictResponse(
        answer="A",
        evidence=["e1"],
        backend="mock",
        model_version="dev",
        contract_version="v1-medembed-small",
        latency_ms=12.5,
        trace_id="abc",
    )
    assert resp.model_dump()["answer"] == "A"
    assert resp.model_dump()["contract_version"] == "v1-medembed-small"
