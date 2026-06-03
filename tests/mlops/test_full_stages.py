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


def test_stage2_grpo_train_then_merge_with_vllm():
    cfg = load_full_config("full")
    p = stage_paths(cfg)
    spec = full_stages.stage2(cfg, p)
    assert len(spec.commands) == 2  # grpo_train_vllm, merge
    train, merge = spec.commands
    assert "scripts.train_rl.grpo_train_vllm" in train
    assert "--model-path" in train and str(p["stage1_5_merged"]) in train
    assert "--data-dir" in train and str(p["kg_dir"]) in train
    assert "--use-vllm" in train
    assert "--vllm-gpu-mem-util" in train and "0.5" in train
    assert "--num-generations" in train and "8" in train
    assert "--use-gdpo" not in train  # grpo variant
    assert "--output-dir" in train and str(p["stage2_out"]) in train
    assert "--base-model-path" in merge and str(p["stage1_5_merged"]) in merge
    assert "--adapter-path" in merge and str(p["stage2_adapter"]) in merge
    assert "--output-dir" in merge and str(p["stage2_merged"]) in merge


def test_stage2_gdpo_adds_flag_and_smoke_caps_max_steps():
    cfg = load_full_config("smoke_full")
    cfg = dataclasses_replace_variant(cfg, "gdpo")
    p = stage_paths(cfg)
    train = full_stages.stage2(cfg, p).commands[0]
    assert "--use-gdpo" in train
    assert "--max-steps" in train and "2" in train
    assert "--num-generations" in train and "4" in train


def test_eval_stage_argv_and_metrics_path():
    cfg = load_full_config("full")
    p = stage_paths(cfg)
    spec = full_stages.eval_stage(cfg, p)
    flat = [t for c in spec.commands for t in c]
    assert "scripts.benchmark.grpo_eval.grpo_eval" in flat
    assert "--model-path" in flat and str(p["stage2_merged"]) in flat
    assert "--data-dir" in flat and str(p["kg_dir"]) in flat
    assert "--benchmarks" in flat and "dataset/MedQA/test" in flat
    assert "--use-vllm" in flat
    assert "--vllm-gpu-mem" in flat and "0.6" in flat
    assert "--output" in flat and str(p["eval_out"]) in flat
    assert spec.eval_metrics_path == str(p["eval_out"])


def test_eval_stage_applies_n_samples_cap():
    cfg = load_full_config("smoke_full")
    p = stage_paths(cfg)
    flat = [t for c in full_stages.eval_stage(cfg, p).commands for t in c]
    assert "--n-samples" in flat and "4" in flat


def test_register_stage_records_model_and_tags():
    cfg = load_full_config("full")
    p = stage_paths(cfg)
    spec = full_stages.register_stage(cfg, p)
    assert spec.commands == []  # register is in-process (no subprocess)
    assert spec.params["registered_model_name"] == "medical-qa-full"
    assert spec.artifact_pointers["model"] == str(p["stage2_merged"])


def test_parse_eval_metrics_flattens_benchmarks(tmp_path):
    import json
    eval_json = tmp_path / "eval.json"
    eval_json.write_text(json.dumps({
        "model_path": "x",
        "benchmarks": {
            "MedQA/test": {"accuracy": 0.6, "n": 100, "answer_rate": 0.95},
            "PubMedQA/train": {"accuracy": 0.73},
        },
    }))
    metrics = full_stages.parse_eval_metrics(str(eval_json))
    assert metrics["MedQA_test.accuracy"] == 0.6
    assert metrics["MedQA_test.n"] == 100.0
    assert metrics["PubMedQA_train.accuracy"] == 0.73


def dataclasses_replace_variant(cfg, variant):
    new_stage2 = dict(cfg.stage2)
    new_stage2["variant"] = variant
    import dataclasses
    return dataclasses.replace(cfg, stage2=new_stage2)
