import json

from mlops.pipelines.run_full import run_full


def test_dry_run_smoke_full_emits_all_stages_in_order(tmp_path):
    receipt = run_full(profile="smoke_full", dry_run=True, receipt_path=str(tmp_path / "r.json"))
    stages = [s["stage"] for s in receipt["stages"]]
    assert stages == ["build_kg", "stage1", "stage1_5", "stage2", "eval", "register"]
    assert receipt["profile"] == "smoke_full"
    assert receipt["tags"]["model_family"] == "qwen"
    assert "retrieval_contract_version" in receipt["tags"]
    bk = receipt["stages"][0]
    assert any("--embed-model" in cmd for cmd in bk["commands"])
    saved = json.loads((tmp_path / "r.json").read_text())
    assert saved == receipt


def test_dry_run_subset_runs_only_selected(tmp_path):
    receipt = run_full(profile="smoke_full", dry_run=True, stages=["eval"], receipt_path=str(tmp_path / "r.json"))
    assert [s["stage"] for s in receipt["stages"]] == ["eval"]


def test_dry_run_does_not_import_mlflow(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "mlflow", None)  # importing mlflow would raise
    receipt = run_full(profile="smoke_full", dry_run=True, receipt_path=None)
    assert len(receipt["stages"]) == 6
