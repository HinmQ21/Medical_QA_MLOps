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
    row = DriftCollector(path=None).record(_req(), _resp("A", ["e1"]), n_evidence=1)
    assert row["q_token_len"] == 4
    assert row["answer"] == "A"
    assert row["no_result"] is False
    assert row["latency_ms"] == 10.0
    assert "n_options" not in row
    assert "mean_option_len" not in row


def test_no_result_flag_true_when_zero_evidence():
    row = DriftCollector(path=None).record(_req(), _resp(None, []), n_evidence=0)
    assert row["no_result"] is True
    assert row["answer"] == "none"


def test_record_does_not_raise_when_path_unwritable():
    # Drift logging is observability; a write failure (e.g. read-only/root-owned dir
    # in the container) must never break the prediction path with a 500.
    collector = DriftCollector(path="/no-such-dir-xyz/drift.jsonl")
    row = collector.record(_req(), _resp("A", ["e1"]), n_evidence=1)  # must not raise
    assert row["answer"] == "A"


def test_record_appends_jsonl(tmp_path):
    path = tmp_path / "drift.jsonl"
    collector = DriftCollector(path=str(path))
    collector.record(_req(), _resp("A", ["e1"]), n_evidence=1)
    collector.record(_req(), _resp("B", []), n_evidence=0)
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["answer"] == "B"
