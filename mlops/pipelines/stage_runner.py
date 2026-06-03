"""Run one pipeline stage: build a receipt (dry-run) or execute + log MLflow."""

import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class StageSpec:
    name: str
    commands: list[list[str]]  # one or more argv lists, run in order
    cwd: str
    params: dict = field(default_factory=dict)
    tags: dict = field(default_factory=dict)
    artifact_pointers: dict = field(default_factory=dict)
    eval_metrics_path: str | None = None  # if set, parsed for metrics in real mode


def _receipt(spec: StageSpec, dry_run: bool) -> dict:
    return {
        "stage": spec.name,
        "dry_run": dry_run,
        "commands": [list(c) for c in spec.commands],
        "cwd": spec.cwd,
        "params": dict(spec.params),
        "tags": dict(spec.tags),
        "artifact_pointers": dict(spec.artifact_pointers),
        "eval_metrics_path": spec.eval_metrics_path,
    }


def run_stage(spec: StageSpec, dry_run: bool, mlflow_active: bool = False) -> dict:
    """Dry-run: return the receipt, run nothing. Real: execute commands, log MLflow."""
    receipt = _receipt(spec, dry_run)
    if dry_run:
        return receipt
    _execute_and_log(spec, receipt, mlflow_active)  # pragma: no cover
    return receipt


def _execute_and_log(spec: StageSpec, receipt: dict, mlflow_active: bool) -> None:  # pragma: no cover
    Path(spec.cwd)  # cwd is expected to exist
    logs = []
    for argv in spec.commands:
        proc = subprocess.run(argv, cwd=spec.cwd, capture_output=True, text=True)
        logs.append({"argv": argv, "returncode": proc.returncode})
        if proc.returncode != 0:
            receipt["failed_command"] = argv
            receipt["stderr_tail"] = proc.stderr[-2000:]
            raise RuntimeError(f"stage {spec.name!r} failed: {argv}")
    receipt["command_logs"] = logs

    if not mlflow_active:
        return
    import mlflow

    from .full_stages import parse_eval_metrics

    with mlflow.start_run(run_name=spec.name, nested=True):
        mlflow.set_tags(spec.tags)
        mlflow.log_params(spec.params)
        for ptr_name, ptr_val in spec.artifact_pointers.items():
            mlflow.set_tag(f"artifact.{ptr_name}", str(ptr_val))
        if spec.eval_metrics_path and Path(spec.eval_metrics_path).exists():
            for key, value in parse_eval_metrics(spec.eval_metrics_path).items():
                mlflow.log_metric(key, value)
