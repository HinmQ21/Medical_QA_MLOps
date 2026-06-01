import json
from pathlib import Path

from mlops.pipelines.build_smoke_kg import build_smoke_kg
from mlops.pipelines.eval_smoke import evaluate_smoke
from mlops.pipelines.profiles import PipelineProfile


def _profile(tmp_path: Path) -> PipelineProfile:
    data_dir = tmp_path / "smoke_data"
    data_dir.mkdir()
    (data_dir / "medical_mcq.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "q1",
                        "question": "Which drug treats type 2 diabetes?",
                        "options": {"A": "Metformin", "B": "Amoxicillin"},
                        "answer": "A",
                        "mock_answer": "A",
                        "retrieval_query": "type 2 diabetes first line medication",
                    }
                ),
                json.dumps(
                    {
                        "id": "q2",
                        "question": "Which hormone is deficient in type 1 diabetes?",
                        "options": {"A": "Insulin", "B": "Thyroxine"},
                        "answer": "A",
                        "mock_answer": "A",
                        "retrieval_query": "type 1 diabetes deficient hormone",
                    }
                ),
            ]
        )
        + "\n"
    )
    artifact_dir = tmp_path / "artifacts/smoke"
    return PipelineProfile(
        name="smoke",
        data_dir=data_dir,
        artifact_dir=artifact_dir,
        kg_dir=artifact_dir / "kg",
        retrieval_fixture_path=artifact_dir / "kg/retrieval_fixture.json",
        predictions_path=artifact_dir / "eval_predictions.jsonl",
        metrics_path=artifact_dir / "eval_metrics.json",
        mlflow_receipt_path=artifact_dir / "mlflow_register_dry_run.json",
        top_k=2,
        model_backend="mock",
        model_version="smoke-dev",
        registered_model_name="medical-qa-smoke",
    )


def test_evaluate_smoke_writes_metrics_and_predictions(tmp_path):
    profile = _profile(tmp_path)
    build_smoke_kg(profile)
    metrics = evaluate_smoke(profile)
    assert metrics["n_examples"] == 2
    assert metrics["accuracy"] == 1.0
    assert metrics["retrieval_no_result_rate"] == 0.0
    assert profile.metrics_path.exists()
    assert profile.predictions_path.exists()
    rows = [json.loads(line) for line in profile.predictions_path.read_text().splitlines()]
    assert rows[0]["predicted_answer"] == "A"
    assert rows[0]["is_correct"] is True
    assert rows[0]["evidence"]
