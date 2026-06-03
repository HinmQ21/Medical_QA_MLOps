from pathlib import Path

from mlops.pipelines.full_config import load_full_config, stage_paths


def test_load_full_profile_defaults():
    cfg = load_full_config("full")
    assert cfg.model_family == "qwen"
    assert cfg.stage2["variant"] == "grpo"
    assert cfg.report_to == "none"
    assert cfg.build_kg["embed_model"] == "abhinand/MedEmbed-large-v0.1"
    assert cfg.eval["vllm_gpu_mem"] == 0.6
    assert cfg.caps == {}


def test_load_smoke_full_has_caps():
    cfg = load_full_config("smoke_full")
    assert cfg.artifact_root.name == "smoke_full"
    assert cfg.caps["max_steps"] == 2
    assert cfg.caps["n_samples"] == 4


def test_stage_paths_are_absolute_and_under_artifact_root():
    cfg = load_full_config("full")
    p = stage_paths(cfg)
    assert p["kg_dir"].is_absolute()
    assert p["kg_dir"].name == "kg"
    assert p["stage1_out"].as_posix().endswith("artifacts/full/outputs/stage1")
    assert p["stage1_5_adapter"].as_posix().endswith("outputs/stage1_5/final")
    assert p["stage1_5_merged"].as_posix().endswith("outputs/stage1_5_merged")
    assert p["stage2_adapter"].as_posix().endswith("outputs/stage2/final")
    assert p["stage2_merged"].as_posix().endswith("outputs/stage2_merged")
    assert p["eval_out"].as_posix().endswith("artifacts/full/eval/eval.json")


def test_unknown_profile_raises():
    import pytest

    with pytest.raises(ValueError):
        load_full_config("nope")
