"""Orchestrate the full training pipeline: parent MLflow run + per-stage runs."""

import argparse
import json
from pathlib import Path

from .full_config import load_full_config, stage_paths
from .full_stages import BUILDERS, STAGE_ORDER
from .stage_runner import run_stage


def _contract_version() -> str:
    try:
        from medical_qa_platform.retrieval.contract import RETRIEVAL_CONTRACT_VERSION

        return RETRIEVAL_CONTRACT_VERSION
    except Exception:
        return "unknown"


def _parent_tags(cfg) -> dict:
    return {
        "profile": cfg.name,
        "model_family": cfg.model_family,
        "kg_version": cfg.kg_version,
        "retrieval_contract_version": _contract_version(),
    }


def run_full(profile: str, dry_run: bool, stages: list[str] | None = None,
             receipt_path: str | None = None, params_path: str = "params.yaml") -> dict:
    cfg = load_full_config(profile, params_path)
    paths = stage_paths(cfg)
    selected = stages or STAGE_ORDER
    tags = _parent_tags(cfg)

    pipeline_receipt = {"profile": cfg.name, "dry_run": dry_run, "tags": tags, "stages": []}

    if dry_run:
        for name in selected:
            spec = BUILDERS[name](cfg, paths)
            pipeline_receipt["stages"].append(run_stage(spec, dry_run=True))
    else:  # pragma: no cover
        _run_real(cfg, paths, selected, tags, pipeline_receipt)

    if receipt_path:
        Path(receipt_path).parent.mkdir(parents=True, exist_ok=True)
        Path(receipt_path).write_text(json.dumps(pipeline_receipt, indent=2, sort_keys=True))
    return pipeline_receipt


def _run_real(cfg, paths, selected, tags, pipeline_receipt):  # pragma: no cover
    import mlflow

    mlflow.set_tracking_uri(f"file:{cfg.mlruns_dir}")
    mlflow.set_experiment(cfg.experiment)
    with mlflow.start_run(run_name=f"{cfg.name}-pipeline") as parent:
        mlflow.set_tags(tags)
        for name in selected:
            spec = BUILDERS[name](cfg, paths)
            if name == "register":
                _register(cfg, paths, spec)
                pipeline_receipt["stages"].append({"stage": "register", "dry_run": False})
                continue
            pipeline_receipt["stages"].append(run_stage(spec, dry_run=False, mlflow_active=True))
        pipeline_receipt["parent_run_id"] = parent.info.run_id


def _register(cfg, paths, spec):  # pragma: no cover
    import mlflow

    with mlflow.start_run(run_name="register", nested=True):
        mlflow.set_tags(spec.tags)
        mlflow.log_params(spec.params)
        mlflow.set_tag("artifact.model", str(paths["stage2_merged"]))
        from .full_stages import parse_eval_metrics

        if paths["eval_out"].exists():
            for key, value in parse_eval_metrics(str(paths["eval_out"])).items():
                mlflow.log_metric(key, value)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="full")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stages", default=None, help="comma-separated subset")
    args = ap.parse_args()
    stages = args.stages.split(",") if args.stages else None
    cfg = load_full_config(args.profile)
    receipt = str((cfg.artifact_root / "run_full_receipt.json"))
    run_full(profile=args.profile, dry_run=args.dry_run, stages=stages, receipt_path=receipt)


if __name__ == "__main__":
    main()
