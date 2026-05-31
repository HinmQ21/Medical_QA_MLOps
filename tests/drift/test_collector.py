import json

from medical_qa_platform.api.schemas import PredictRequest, PredictResponse
from medical_qa_platform.drift.collector import DriftCollector


def _req():
    return PredictRequest(question="what treats diabetes now", options={"A": "aa", "B": "bbbb"})


def _resp(answer, evidence):
    return PredictResponse(
        answer=answer,
        evidence=evidence,
        backend="mock",
        model_version="dev",
        latency_ms=10.0,
        trace_id="t1",
    )


def test_record_computes_features():
    row = DriftCollector(path=None).record(_req(), _resp("A", ["e1"]), n_evidence=1)
    assert row["q_token_len"] == 4
    assert row["n_options"] == 2
    assert row["mean_option_len"] == 3.0
    assert row["answer"] == "A"
    assert row["no_result"] is False
    assert row["latency_ms"] == 10.0


def test_no_result_flag_true_when_zero_evidence():
    row = DriftCollector(path=None).record(_req(), _resp(None, []), n_evidence=0)
    assert row["no_result"] is True
    assert row["answer"] == "none"


def test_record_appends_jsonl(tmp_path):
    path = tmp_path / "drift.jsonl"
    collector = DriftCollector(path=str(path))
    collector.record(_req(), _resp("A", ["e1"]), n_evidence=1)
    collector.record(_req(), _resp("B", []), n_evidence=0)
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["answer"] == "B"
