"""Build StageSpec command(s) for each pipeline stage, matching baseline argv.

mlops orchestrates baseline by subprocess (never import). All produced paths are
absolute under artifact_root; baseline inputs (base model, datasets, SFT traces)
are passed relative to baseline_root (the stage cwd).
"""

import dataclasses
import json
from pathlib import Path

from .full_config import FullPipelineConfig
from .stage_runner import StageSpec


def _with_family(cfg: FullPipelineConfig, family: str) -> FullPipelineConfig:
    return dataclasses.replace(cfg, model_family=family)


def _train_caps_argv(cfg: FullPipelineConfig) -> list[str]:
    """Smoke caps shared by Stage 1 / Stage 1.5 full-FT/SFT trainers."""
    caps = cfg.caps
    argv: list[str] = []
    if "max_train_samples" in caps:
        argv += ["--max-train-samples", str(caps["max_train_samples"])]
    if "max_eval_samples" in caps:
        argv += ["--max-eval-samples", str(caps["max_eval_samples"])]
    if "num_train_epochs" in caps:
        argv += ["--num-train-epochs", str(caps["num_train_epochs"])]
    if "save_steps" in caps:
        argv += ["--save-steps", str(caps["save_steps"])]
    return argv


def build_kg(cfg: FullPipelineConfig, p: dict[str, Path]) -> StageSpec:
    argv = [
        str(cfg.train_venv), "-m", "scripts.build_kg.run_pipeline",
        "--data-dir", str(p["kg_dir"]),
        "--embed-model", cfg.build_kg["embed_model"],
        "--embed-device", cfg.build_kg["embed_device"],
    ]
    return StageSpec(
        name="build_kg",
        commands=[argv],
        cwd=str(cfg.baseline_root),
        params={"embed_model": cfg.build_kg["embed_model"], "embed_device": cfg.build_kg["embed_device"]},
        tags={"stage": "build_kg", "kg_version": cfg.kg_version},
        artifact_pointers={"kg_dir": str(p["kg_dir"])},
    )


def stage1(cfg: FullPipelineConfig, p: dict[str, Path]) -> StageSpec:
    argv = [
        str(cfg.train_venv), "-m",
        "scripts.finetune.medreason.qwen25_medreason_full_trainer_think_tag",
        "--model-path", cfg.stage1["base_model"],
        "--model-family", cfg.model_family,
        "--output-dir", str(p["stage1_out"]),
        "--report-to", cfg.report_to,
    ] + _train_caps_argv(cfg)
    return StageSpec(
        name="stage1",
        commands=[argv],
        cwd=str(cfg.baseline_root),
        params={"base_model": cfg.stage1["base_model"], "model_family": cfg.model_family},
        tags={"stage": "stage1", "model_family": cfg.model_family},
        artifact_pointers={"stage1_out": str(p["stage1_out"])},
    )


def stage1_5(cfg: FullPipelineConfig, p: dict[str, Path]) -> StageSpec:
    commands: list[list[str]] = []
    sft_data = cfg.stage1_5["sft_data"]
    if cfg.model_family == "llama":
        commands.append([
            str(cfg.train_venv), "-m", "scripts.stage1_5.convert_data_llama",
            "--input", sft_data,
            "--output", str(p["stage1_5_llama_data"]),
        ])
        sft_data = str(p["stage1_5_llama_data"])
    commands.append([
        str(cfg.train_venv), "-m", "scripts.stage1_5.sft_train",
        "--model-path", str(p["stage1_out"]),
        "--model-family", cfg.model_family,
        "--data-path", sft_data,
        "--output-dir", str(p["stage1_5_out"]),
        "--lora-r", str(cfg.stage1_5["lora_r"]),
        "--lora-alpha", str(cfg.stage1_5["lora_alpha"]),
        "--report-to", cfg.report_to,
    ] + (["--num-train-epochs", str(cfg.caps["num_train_epochs"])] if "num_train_epochs" in cfg.caps else []))
    commands.append([
        str(cfg.train_venv), "-m", "scripts.finetune.merge_peft_adapter",
        "--base-model-path", str(p["stage1_out"]),
        "--adapter-path", str(p["stage1_5_adapter"]),
        "--output-dir", str(p["stage1_5_merged"]),
    ])
    return StageSpec(
        name="stage1_5",
        commands=commands,
        cwd=str(cfg.baseline_root),
        params={"lora_r": cfg.stage1_5["lora_r"], "lora_alpha": cfg.stage1_5["lora_alpha"], "model_family": cfg.model_family},
        tags={"stage": "stage1_5", "model_family": cfg.model_family},
        artifact_pointers={"merged": str(p["stage1_5_merged"])},
    )
