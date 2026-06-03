from mlops.pipelines import full_stages
from mlops.pipelines.full_config import load_full_config, stage_paths


def _flat(spec):
    return [tok for cmd in spec.commands for tok in cmd]


def test_build_kg_passes_embed_model_and_kg_out():
    cfg = load_full_config("full")
    p = stage_paths(cfg)
    spec = full_stages.build_kg(cfg, p)
    flat = _flat(spec)
    assert "scripts.build_kg.run_pipeline" in flat
    assert "--embed-model" in flat and "abhinand/MedEmbed-large-v0.1" in flat
    assert "--data-dir" in flat and str(p["kg_dir"]) in flat
    assert spec.cwd == str(cfg.baseline_root)
    assert str(cfg.train_venv) == spec.commands[0][0]


def test_stage1_qwen_argv():
    cfg = load_full_config("full")
    p = stage_paths(cfg)
    spec = full_stages.stage1(cfg, p)
    flat = _flat(spec)
    assert "scripts.finetune.medreason.qwen25_medreason_full_trainer_think_tag" in flat
    assert "--model-family" in flat and "qwen" in flat
    assert "--model-path" in flat and "models/Qwen2.5-3B-Instruct" in flat
    assert "--output-dir" in flat and str(p["stage1_out"]) in flat
    assert "--report-to" in flat and "none" in flat


def test_stage1_applies_smoke_caps():
    cfg = load_full_config("smoke_full")
    p = stage_paths(cfg)
    flat = _flat(full_stages.stage1(cfg, p))
    assert "--max-train-samples" in flat and "8" in flat
    assert "--num-train-epochs" in flat and "1" in flat


def test_stage1_5_qwen_is_train_then_merge():
    cfg = load_full_config("full")
    p = stage_paths(cfg)
    spec = full_stages.stage1_5(cfg, p)
    assert len(spec.commands) == 2  # sft_train, merge
    train, merge = spec.commands
    assert "scripts.stage1_5.sft_train" in train
    assert "--model-path" in train and str(p["stage1_out"]) in train
    assert "--data-path" in train and "data/stage1_5_sft_v2.jsonl" in train
    assert "scripts.finetune.merge_peft_adapter" in merge
    assert "--base-model-path" in merge and str(p["stage1_out"]) in merge
    assert "--adapter-path" in merge and str(p["stage1_5_adapter"]) in merge
    assert "--output-dir" in merge and str(p["stage1_5_merged"]) in merge


def test_stage1_5_llama_prepends_convert():
    cfg = load_full_config("full")
    cfg = full_stages._with_family(cfg, "llama")
    p = stage_paths(cfg)
    spec = full_stages.stage1_5(cfg, p)
    assert len(spec.commands) == 3  # convert, sft_train, merge
    convert = spec.commands[0]
    assert "scripts.stage1_5.convert_data_llama" in convert
    assert "--output" in convert and str(p["stage1_5_llama_data"]) in convert
    assert "--model-family" in spec.commands[1] and "llama" in spec.commands[1]
