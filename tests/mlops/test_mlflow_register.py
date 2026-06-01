import json
from pathlib import Path

from mlops.mlflow_register import register_smoke_model
from mlops.pipelines.profiles import PipelineProfile


def _profile(tmp_path: Path) -> PipelineProfile:
    artifact_dir = tmp_path / "artifacts/smoke"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "eval_metrics.json").write_text(
        json.dumps({"accuracy": 1.0, "n_examples": 2, "latency_ms_avg": 1.5})
    )
    (artifact_dir / "eval_predictions.jsonl").write_text(
        json.dumps({"id": "q1", "predicted_answer": "A"}) + "\n"
    )
    return PipelineProfile(
        name="smoke",
        data_dir=tmp_path / "smoke_data",
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


def test_register_smoke_model_dry_run_writes_receipt(tmp_path):
    profile = _profile(tmp_path)
    receipt = register_smoke_model(profile, dry_run=True)
    assert receipt["dry_run"] is True
    assert receipt["registered_model_name"] == "medical-qa-smoke"
    assert receipt["metrics"]["accuracy"] == 1.0
    assert profile.mlflow_receipt_path.exists()
    saved = json.loads(profile.mlflow_receipt_path.read_text())
    assert saved["model_version"] == "smoke-dev"
