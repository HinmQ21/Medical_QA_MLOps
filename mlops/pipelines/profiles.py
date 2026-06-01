"""Load typed pipeline profiles from params.yaml."""

from dataclasses import dataclass
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PipelineProfile:
    name: str
    data_dir: Path
    artifact_dir: Path
    kg_dir: Path
    retrieval_fixture_path: Path
    predictions_path: Path
    metrics_path: Path
    mlflow_receipt_path: Path
    top_k: int
    model_backend: str
    model_version: str
    registered_model_name: str


def _path(value: str) -> Path:
    return Path(value)


def load_profile(name: str, params_path: Path | str = "params.yaml") -> PipelineProfile:
    params_file = Path(params_path)
    data = yaml.safe_load(params_file.read_text())
    if name not in data:
        raise ValueError(f"unknown pipeline profile: {name!r}")
    raw = data[name]
    return PipelineProfile(
        name=name,
        data_dir=_path(raw["data_dir"]),
        artifact_dir=_path(raw["artifact_dir"]),
        kg_dir=_path(raw["kg_dir"]),
        retrieval_fixture_path=_path(raw["retrieval_fixture_path"]),
        predictions_path=_path(raw["predictions_path"]),
        metrics_path=_path(raw["metrics_path"]),
        mlflow_receipt_path=_path(raw["mlflow_receipt_path"]),
        top_k=int(raw["top_k"]),
        model_backend=raw["model_backend"],
        model_version=raw["model_version"],
        registered_model_name=raw["registered_model_name"],
    )
