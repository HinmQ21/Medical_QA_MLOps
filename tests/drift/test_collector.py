import json

from medical_qa_platform.api.schemas import PredictRequest, PredictResponse
from medical_qa_platform.drift.collector import DriftCollector


def _req():
    return PredictRequest(question="what treats diabetes now")


def _resp(answer, evidence):
    return PredictResponse(
        answer=answer,
        raw_output="<answer>x</answer>",
        evidence=evidence,
        backend="mock",
        model_version="dev",
        contract_version="v1-medembed-small",
        latency_ms=10.0,
        trace_id="t1",
    )


def test_record_computes_features():
    row = DriftCollector(path=None).record(
        _req(), _resp("A", ["e1"]), n_evidence=1, tool_call_count=1
    )
    assert row["q_token_len"] == 4
    assert row["answer"] == "A"
    assert row["tool_call_count"] == 1
    assert row["tool_called"] is True
    assert row["n_evidence"] == 1
    assert row["tool_outcome"] == "hit"
    assert row["latency_ms"] == 10.0
    assert "no_result" not in row


def test_outcome_not_called_when_model_skips_tool():
    row = DriftCollector(path=None).record(
        _req(), _resp("A", []), n_evidence=0, tool_call_count=0
    )
    assert row["tool_called"] is False
    assert row["tool_outcome"] == "not_called"


def test_outcome_empty_when_tool_called_but_no_evidence():
    row = DriftCollector(path=None).record(
        _req(), _resp(None, []), n_evidence=0, tool_call_count=1
    )
    assert row["tool_outcome"] == "empty"
    assert row["answer"] == "none"


def test_record_does_not_raise_when_path_unwritable():
    collector = DriftCollector(path="/no-such-dir-xyz/drift.jsonl")
    row = collector.record(_req(), _resp("A", ["e1"]), n_evidence=1, tool_call_count=1)
    assert row["answer"] == "A"


def test_record_appends_jsonl(tmp_path):
    path = tmp_path / "drift.jsonl"
    collector = DriftCollector(path=str(path))
    collector.record(_req(), _resp("A", ["e1"]), n_evidence=1, tool_call_count=1)
    collector.record(_req(), _resp("B", []), n_evidence=0, tool_call_count=0)
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["tool_outcome"] == "not_called"
