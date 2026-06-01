import json
from pathlib import Path


ROOT = Path(__file__).parents[2]


def test_smoke_dataset_has_two_valid_mcq_rows():
    path = ROOT / "mlops/smoke_data/medical_mcq.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 2
    for row in rows:
        assert row["id"]
        assert row["question"]
        assert set(row["options"]) == {"A", "B"}
        assert row["answer"] in row["options"]
        assert row["mock_answer"] == row["answer"]
        assert isinstance(row["retrieval_query"], str)
