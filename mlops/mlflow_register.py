"""Log smoke pipeline outputs to MLflow, with a deterministic dry-run mode."""

import argparse
import json
from pathlib import Path

from .pipelines.profiles import PipelineProfile, load_profile


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def register_smoke_model(profile: PipelineProfile, dry_run: bool = False) -> dict:
    metrics = _read_json(profile.metrics_path)
    receipt = {
        "profile": profile.name,
        "dry_run": dry_run,
        "registered_model_name": profile.registered_model_name,
        "model_version": profile.model_version,
        "metrics": metrics,
        "artifacts": {
            "metrics_path": str(profile.metrics_path),
            "predictions_path": str(profile.predictions_path),
        },
    }

    if dry_run:
        _write_json(profile.mlflow_receipt_path, receipt)
        return receipt

    import mlflow

    mlflow.set_experiment("medical-qa-smoke")
    with mlflow.start_run(run_name=f"{profile.name}-{profile.model_version}") as run:
        mlflow.log_params(
            {
                "profile": profile.name,
                "model_backend": profile.model_backend,
                "model_version": profile.model_version,
                "top_k": profile.top_k,
            }
        )
        for key, value in metrics.items():
            if isinstance(value, int | float):
                mlflow.log_metric(key, float(value))
        mlflow.log_artifact(str(profile.metrics_path))
        mlflow.log_artifact(str(profile.predictions_path))
        receipt["run_id"] = run.info.run_id
        receipt["model_uri"] = f"runs:/{run.info.run_id}/smoke-model"

    _write_json(profile.mlflow_receipt_path, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="smoke")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    register_smoke_model(load_profile(args.profile), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
