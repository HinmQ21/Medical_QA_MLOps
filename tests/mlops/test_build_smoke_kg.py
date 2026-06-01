import json
from pathlib import Path

from mlops.pipelines.build_smoke_kg import build_smoke_kg
from mlops.pipelines.profiles import PipelineProfile


def _profile(tmp_path: Path) -> PipelineProfile:
    artifact_dir = tmp_path / "artifacts/smoke"
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


def test_build_smoke_kg_writes_expected_artifacts(tmp_path):
    manifest = build_smoke_kg(_profile(tmp_path))
    kg_dir = tmp_path / "artifacts/smoke/kg"
    assert (kg_dir / "medical_hg.json").exists()
    assert (kg_dir / "hedge_ids.json").exists()
    assert (kg_dir / "entity_names.json").exists()
    assert (kg_dir / "retrieval_fixture.json").exists()
    assert (kg_dir / "manifest.json").exists()
    assert manifest["artifact_type"] == "smoke_kg"
    assert manifest["n_hyperedges"] == 2


def test_retrieval_fixture_maps_queries_to_evidence(tmp_path):
    build_smoke_kg(_profile(tmp_path))
    fixture = json.loads((tmp_path / "artifacts/smoke/kg/retrieval_fixture.json").read_text())
    assert fixture["type 2 diabetes first line medication"] == [
        "Metformin is commonly used first-line for type 2 diabetes."
    ]
    assert fixture["type 1 diabetes deficient hormone"] == [
        "Type 1 diabetes is characterized by insulin deficiency."
    ]
