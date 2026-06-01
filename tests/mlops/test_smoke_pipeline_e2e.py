import json
from pathlib import Path

from mlops.mlflow_register import register_smoke_model
from mlops.pipelines.build_smoke_kg import build_smoke_kg
from mlops.pipelines.eval_smoke import evaluate_smoke
from mlops.pipelines.profiles import PipelineProfile


def test_smoke_pipeline_e2e(tmp_path):
    data_dir = tmp_path / "smoke_data"
    data_dir.mkdir()
    (data_dir / "medical_mcq.jsonl").write_text(
        json.dumps(
            {
                "id": "q1",
                "question": "Which drug treats type 2 diabetes?",
                "options": {"A": "Metformin", "B": "Amoxicillin"},
                "answer": "A",
                "mock_answer": "A",
                "retrieval_query": "type 2 diabetes first line medication",
            }
        )
        + "\n"
    )
    artifact_dir = tmp_path / "artifacts/smoke"
    profile = PipelineProfile(
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

    build_smoke_kg(profile)
    metrics = evaluate_smoke(profile)
    receipt = register_smoke_model(profile, dry_run=True)

    assert metrics["accuracy"] == 1.0
    assert receipt["metrics"]["accuracy"] == 1.0
    assert profile.mlflow_receipt_path.exists()
