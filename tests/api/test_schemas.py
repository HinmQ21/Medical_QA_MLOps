import pytest
from pydantic import ValidationError

from medical_qa_platform.api.schemas import PredictRequest, PredictResponse


def test_valid_request_is_free_text():
    req = PredictRequest(question="What is first-line for T2DM? A) Metformin B) Insulin")
    assert "Metformin" in req.question


def test_rejects_empty_question():
    with pytest.raises(ValidationError):
        PredictRequest(question="   ")


def test_ignores_stray_options_field():
    # Approach A: options is gone; pydantic v2 ignores extra input fields, so an old
    # caller still sending `options` gets no 422 — the field is just dropped.
    req = PredictRequest(question="Q?", options={"A": "a"})
    assert not hasattr(req, "options")


def test_response_round_trip_includes_raw_output():
    resp = PredictResponse(
        answer="A",
        raw_output="<think>r</think><answer>A</answer>",
        evidence=["e1"],
        backend="mock",
        model_version="dev",
        contract_version="v1-medembed-small",
        latency_ms=12.5,
        trace_id="abc",
    )
    dumped = resp.model_dump()
    assert dumped["answer"] == "A"
    assert dumped["raw_output"] == "<think>r</think><answer>A</answer>"
    assert dumped["contract_version"] == "v1-medembed-small"
