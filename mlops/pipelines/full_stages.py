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


def stage2(cfg: FullPipelineConfig, p: dict[str, Path]) -> StageSpec:
    s2 = cfg.stage2
    venv = str(cfg.vllm_venv) if s2.get("use_vllm") else str(cfg.train_venv)
    train = [
        venv, "-m", "scripts.train_rl.grpo_train_vllm",
        "--model-path", str(p["stage1_5_merged"]),
        "--model-family", cfg.model_family,
        "--data-dir", str(p["kg_dir"]),
        "--output-dir", str(p["stage2_out"]),
        "--lora-r", str(s2["lora_r"]),
        "--lora-alpha", str(s2["lora_alpha"]),
        "--num-generations", str(s2["num_generations"]),
        "--report-to", cfg.report_to,
    ]
    if s2.get("variant") == "gdpo":
        train.append("--use-gdpo")
    if s2.get("use_vllm"):
        train += ["--use-vllm", "--vllm-gpu-mem-util", str(s2["vllm_gpu_mem_util"])]
    if "max_steps" in cfg.caps:
        train += ["--max-steps", str(cfg.caps["max_steps"])]
    if "max_train_samples" in cfg.caps:
        train += ["--max-train-samples", str(cfg.caps["max_train_samples"])]
    merge = [
        str(cfg.train_venv), "-m", "scripts.finetune.merge_peft_adapter",
        "--base-model-path", str(p["stage1_5_merged"]),
        "--adapter-path", str(p["stage2_adapter"]),
        "--output-dir", str(p["stage2_merged"]),
    ]
    return StageSpec(
        name="stage2",
        commands=[train, merge],
        cwd=str(cfg.baseline_root),
        params={"variant": s2.get("variant"), "num_generations": s2["num_generations"], "lora_alpha": s2["lora_alpha"]},
        tags={"stage": "stage2", "variant": s2.get("variant"), "model_family": cfg.model_family},
        artifact_pointers={"merged": str(p["stage2_merged"])},
    )


def eval_stage(cfg: FullPipelineConfig, p: dict[str, Path]) -> StageSpec:
    ev = cfg.eval
    venv = str(cfg.vllm_venv) if ev.get("use_vllm") else str(cfg.train_venv)
    argv = [
        venv, "-m", "scripts.benchmark.grpo_eval.grpo_eval",
        "--model-path", str(p["stage2_merged"]),
        "--model-family", cfg.model_family,
        "--data-dir", str(p["kg_dir"]),
        "--benchmarks", *ev["benchmarks"],
        "--output", str(p["eval_out"]),
    ]
    if ev.get("use_vllm"):
        argv += ["--use-vllm", "--vllm-gpu-mem", str(ev["vllm_gpu_mem"])]
    if "n_samples" in cfg.caps:
        argv += ["--n-samples", str(cfg.caps["n_samples"])]
    return StageSpec(
        name="eval",
        commands=[argv],
        cwd=str(cfg.baseline_root),
        params={"benchmarks": ev["benchmarks"], "use_vllm": ev.get("use_vllm", False)},
        tags={"stage": "eval", "model_family": cfg.model_family},
        artifact_pointers={"eval_json": str(p["eval_out"])},
        eval_metrics_path=str(p["eval_out"]),
    )


def register_stage(cfg: FullPipelineConfig, p: dict[str, Path]) -> StageSpec:
    return StageSpec(
        name="register",
        commands=[],  # in-process registration handled by run_full
        cwd=str(cfg.baseline_root),
        params={"registered_model_name": cfg.registered_model_name, "kg_version": cfg.kg_version},
        tags={"stage": "register", "model_family": cfg.model_family},
        artifact_pointers={"model": str(p["stage2_merged"]), "eval_json": str(p["eval_out"])},
    )


def parse_eval_metrics(eval_json_path: str) -> dict[str, float]:
    """Flatten the eval JSON's per-benchmark numeric fields to MLflow metric keys."""
    data = json.loads(Path(eval_json_path).read_text())
    metrics: dict[str, float] = {}
    for bench, bench_data in data.get("benchmarks", {}).items():
        if not isinstance(bench_data, dict):
            continue
        safe = bench.replace("/", "_")
        for key, value in bench_data.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metrics[f"{safe}.{key}"] = float(value)
    return metrics


BUILDERS = {
    "build_kg": build_kg,
    "stage1": stage1,
    "stage1_5": stage1_5,
    "stage2": stage2,
    "eval": eval_stage,
    "register": register_stage,
}

STAGE_ORDER = ["build_kg", "stage1", "stage1_5", "stage2", "eval", "register"]
