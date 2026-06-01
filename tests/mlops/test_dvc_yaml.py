from pathlib import Path

import yaml


ROOT = Path(__file__).parents[2]


def test_params_yaml_defines_smoke_profile_paths():
    params = yaml.safe_load((ROOT / "params.yaml").read_text())
    smoke = params["smoke"]
    assert smoke["data_dir"] == "mlops/smoke_data"
    assert smoke["artifact_dir"] == "artifacts/smoke"
    assert smoke["kg_dir"] == "artifacts/smoke/kg"
    assert smoke["predictions_path"] == "artifacts/smoke/eval_predictions.jsonl"
    assert smoke["metrics_path"] == "artifacts/smoke/eval_metrics.json"
    assert smoke["mlflow_receipt_path"] == "artifacts/smoke/mlflow_register_dry_run.json"
    assert smoke["top_k"] == 2
    assert smoke["model_backend"] == "mock"
    assert smoke["model_version"] == "smoke-dev"
    assert smoke["registered_model_name"] == "medical-qa-smoke"


def test_dvc_yaml_has_expected_smoke_stages():
    dvc = yaml.safe_load((ROOT / "dvc.yaml").read_text())
    stages = dvc["stages"]
    assert set(stages) == {"build_kg_smoke", "eval_smoke", "register_smoke_model"}
    assert stages["build_kg_smoke"]["cmd"] == (
        "python -m mlops.pipelines.build_smoke_kg --profile smoke"
    )
    assert stages["eval_smoke"]["cmd"] == (
        "python -m mlops.pipelines.eval_smoke --profile smoke"
    )
    assert stages["register_smoke_model"]["cmd"] == (
        "python -m mlops.mlflow_register --profile smoke --dry-run"
    )
    assert "artifacts/smoke/kg/manifest.json" in stages["build_kg_smoke"]["outs"]
    assert "artifacts/smoke/eval_metrics.json" in stages["eval_smoke"]["outs"]
    assert (
        "artifacts/smoke/mlflow_register_dry_run.json"
        in stages["register_smoke_model"]["outs"]
    )
