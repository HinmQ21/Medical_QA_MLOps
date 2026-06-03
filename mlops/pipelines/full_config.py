"""Typed config for the full training pipeline (full / smoke_full profiles)."""

from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class FullPipelineConfig:
    name: str
    baseline_root: Path
    train_venv: Path
    vllm_venv: Path
    artifact_root: Path
    mlruns_dir: Path
    experiment: str
    registered_model_name: str
    model_family: str
    kg_version: str
    report_to: str
    build_kg: dict
    stage1: dict
    stage1_5: dict
    stage2: dict
    eval: dict
    caps: dict


def load_full_config(name: str, params_path: Path | str = "params.yaml") -> FullPipelineConfig:
    data = yaml.safe_load(Path(params_path).read_text())
    if name not in data:
        raise ValueError(f"unknown pipeline profile: {name!r}")
    raw = data[name]
    baseline_root = Path(raw["baseline_root"])
    return FullPipelineConfig(
        name=name,
        baseline_root=baseline_root,
        train_venv=baseline_root / raw["train_venv"],
        vllm_venv=baseline_root / raw["vllm_venv"],
        artifact_root=(ROOT / raw["artifact_root"]).resolve(),
        mlruns_dir=(ROOT / raw["mlruns_dir"]).resolve(),
        experiment=raw["experiment"],
        registered_model_name=raw["registered_model_name"],
        model_family=raw["model_family"],
        kg_version=raw["kg_version"],
        report_to=raw.get("report_to", "none"),
        build_kg=raw["build_kg"],
        stage1=raw["stage1"],
        stage1_5=raw["stage1_5"],
        stage2=raw["stage2"],
        eval=raw["eval"],
        caps=raw.get("caps", {}),
    )


def stage_paths(cfg: FullPipelineConfig) -> dict[str, Path]:
    """All pipeline-produced paths, absolute, under artifact_root."""
    a = cfg.artifact_root
    outputs = a / "outputs"
    return {
        "kg_dir": a / "kg",
        "stage1_out": outputs / "stage1",
        "stage1_5_out": outputs / "stage1_5",
        "stage1_5_adapter": outputs / "stage1_5" / "final",
        "stage1_5_merged": outputs / "stage1_5_merged",
        "stage1_5_llama_data": a / "stage1_5_sft_llama.jsonl",
        "stage2_out": outputs / "stage2",
        "stage2_adapter": outputs / "stage2" / "final",
        "stage2_merged": outputs / "stage2_merged",
        "eval_out": a / "eval" / "eval.json",
        "receipt": a / "run_full_receipt.json",
    }
