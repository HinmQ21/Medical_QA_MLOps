import yaml


def _stages():
    return yaml.safe_load(open("dvc.yaml"))["stages"]


def test_smoke_stages_preserved():
    stages = _stages()
    for name in ["build_kg_smoke", "eval_smoke", "register_smoke_model"]:
        assert name in stages


def test_full_stages_exist_in_order_with_run_full_cmd():
    stages = _stages()
    for name in ["full_build_kg", "full_stage1", "full_stage1_5", "full_stage2", "full_eval", "full_register"]:
        assert name in stages, name
        assert "mlops.pipelines.run_full" in stages[name]["cmd"]
        assert "--profile full" in stages[name]["cmd"]


def test_full_train_outputs_are_not_cached():
    stages = _stages()
    # checkpoint dirs must be cache: false (no GB caching in DVC)
    outs = stages["full_stage1"]["outs"]
    assert outs, "full_stage1 has no outs"
    entry = outs[0]
    assert isinstance(entry, dict), "out entry should be a mapping with a cache flag"
    out_cfg = list(entry.values())[0]
    assert out_cfg.get("cache") is False
