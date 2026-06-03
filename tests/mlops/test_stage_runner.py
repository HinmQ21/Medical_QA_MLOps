from mlops.pipelines.stage_runner import StageSpec, run_stage


def _spec():
    return StageSpec(
        name="demo",
        commands=[["/venv/python", "-m", "scripts.x", "--flag", "1"]],
        cwd="/base",
        params={"flag": 1},
        tags={"stage": "demo"},
        artifact_pointers={"out": "/base/out"},
        eval_metrics_path=None,
    )


def test_run_stage_dry_run_returns_receipt_without_executing():
    receipt = run_stage(_spec(), dry_run=True)
    assert receipt["stage"] == "demo"
    assert receipt["dry_run"] is True
    assert receipt["commands"] == [["/venv/python", "-m", "scripts.x", "--flag", "1"]]
    assert receipt["cwd"] == "/base"
    assert receipt["params"] == {"flag": 1}
    assert receipt["tags"]["stage"] == "demo"
    assert receipt["artifact_pointers"] == {"out": "/base/out"}


def test_run_stage_dry_run_does_not_touch_subprocess(monkeypatch):
    called = {"n": 0}

    def boom(*a, **k):
        called["n"] += 1
        raise AssertionError("subprocess must not run in dry-run")

    monkeypatch.setattr("mlops.pipelines.stage_runner.subprocess.run", boom)
    run_stage(_spec(), dry_run=True)
    assert called["n"] == 0
