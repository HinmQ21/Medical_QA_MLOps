# Full MLflow/DVC Training Pipeline (SP3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a parameterised, runnable pipeline in `mlops-platform/` that drives the full `baseline/` training lifecycle (KG build → Stage 1 → Stage 1.5 → Stage 2 → eval → register) locally on the GB10, logging every stage to MLflow and tracking lineage in DVC — orchestrating baseline scripts by subprocess (no import).

**Architecture:** A typed config (`full`/`smoke_full` profiles) feeds six pure command-builders that emit the exact baseline argv (verified against the scripts' argparse). A generic `stage_runner` runs a stage's commands (or records them in `--dry-run`) and logs an MLflow nested run; `run_full` opens a parent run and iterates stages. DVC stages call `run_full` per stage for lineage; the real multi-day run is operator-triggered. CI uses dry-run + structural tests; a one-off `smoke_full` subset run proves wiring.

**Tech Stack:** Python 3.12, PyYAML, MLflow, DVC, pytest. Orchestration is subprocess-only; `mlops/` may reference `baseline/` paths (it is outside `src/`, so the SP1 self-contained guard does not apply) but must not import baseline Python.

**Spec:** `docs/specs/2026-06-03-full-mlflow-pipeline-design.md`

**Working directory:** all commands run from `/home/vcsai/minhlbq/mlops-platform`. Package venv: `.venv/bin/python`.

**Authoritative baseline flags** (extracted from the scripts' argparse — note these correct stale CLAUDE.md docs):
- `scripts.build_kg.run_pipeline`: `--data-dir` (KG **output** dir), `--embed-model`, `--embed-device`.
- `scripts.finetune.medreason.qwen25_medreason_full_trainer_think_tag`: `--model-path`, `--model-family`, `--output-dir`, `--max-train-samples`, `--max-eval-samples`, `--num-train-epochs`, `--save-steps`, `--report-to`. Saves final model at `<output-dir>` root.
- `scripts.stage1_5.sft_train`: `--model-path`, `--model-family`, `--data-path`, `--output-dir`, `--lora-r`, `--lora-alpha`, `--num-train-epochs`, `--report-to`. Saves adapter at `<output-dir>/final`.
- `scripts.stage1_5.convert_data_llama`: `--input`, `--output`.
- `scripts.train_rl.grpo_train_vllm`: `--model-path`, `--model-family`, `--data-dir` (KG), `--output-dir`, `--use-gdpo`, `--use-vllm`, `--vllm-gpu-mem-util`, `--num-generations`, `--lora-r`, `--lora-alpha`, `--max-steps`, `--max-train-samples`, `--max-eval-samples`, `--report-to`. Saves adapter at `<output-dir>/final`.
- `scripts.benchmark.grpo_eval.grpo_eval`: `--model-path`, `--model-family`, `--data-dir` (KG), `--benchmarks` (nargs), `--use-vllm`, `--vllm-gpu-mem` (NOT `-util`), `--n-samples`, `--output`. Writes a JSON whose top level has a `benchmarks` dict keyed by `"<Dataset>/<split>"`, each a dict of numeric metrics.
- `scripts.finetune.merge_peft_adapter`: `--base-model-path`, `--adapter-path`, `--output-dir`.

**Deterministic inter-stage paths** (no runtime checkpoint globbing needed):
- Stage 1 model → `<stage1_out>` (root). Stage 1.5 adapter → `<stage1_5_out>/final`. Stage 2 adapter → `<stage2_out>/final`. Merges write to `<*_merged>`.

---

## File Structure

**Create:**
- `mlops/pipelines/full_config.py` — `FullPipelineConfig` dataclass, `load_full_config`, `stage_paths` (derived absolute paths).
- `mlops/pipelines/stage_runner.py` — `StageSpec`, `run_stage` (dry-run + real), `_log_nested_run` (real, `# pragma: no cover`).
- `mlops/pipelines/full_stages.py` — six builders (`build_kg`, `stage1`, `stage1_5`, `stage2`, `eval_stage`, `register_stage`) + `parse_eval_metrics`.
- `mlops/pipelines/run_full.py` — orchestrator + CLI.
- `tests/mlops/test_full_config.py`, `tests/mlops/test_stage_runner.py`, `tests/mlops/test_full_stages.py`, `tests/mlops/test_run_full_dry_run.py`, `tests/mlops/test_dvc_full_stages.py`, `tests/mlops/test_makefile_full_targets.py`.

**Modify:**
- `params.yaml` — add `full` and `smoke_full` profiles.
- `dvc.yaml` — add six full stages.
- `Makefile` — add `full-pipeline`, `full-pipeline-dry-run`, `smoke-full` targets.
- `.gitignore` — ignore `artifacts/` and `mlruns/` (if not already).

---

## Task 1: Config (`full`/`smoke_full` profiles + loader)

**Files:** Modify `params.yaml`; Create `mlops/pipelines/full_config.py`, `tests/mlops/test_full_config.py`.

- [ ] **Step 1: Add profiles to `params.yaml`** (append; keep the existing `smoke:` block):

```yaml
full:
  baseline_root: /home/vcsai/minhlbq/baseline
  train_venv: training_venv312/bin/python
  vllm_venv: vllm_venv312/bin/python
  artifact_root: artifacts/full
  mlruns_dir: mlruns
  experiment: medical-qa-full
  registered_model_name: medical-qa-full
  model_family: qwen
  kg_version: v0
  report_to: none
  build_kg:
    embed_model: abhinand/MedEmbed-large-v0.1
    embed_device: cpu
  stage1:
    base_model: models/Qwen2.5-3B-Instruct
  stage1_5:
    sft_data: data/stage1_5_sft_v2.jsonl
    lora_r: 32
    lora_alpha: 32
  stage2:
    variant: grpo
    use_vllm: true
    vllm_gpu_mem_util: 0.5
    num_generations: 8
    lora_r: 32
    lora_alpha: 64
  eval:
    benchmarks:
      - dataset/MedQA/test
      - dataset/MedMCQA_4options_fixed/validation
      - dataset/PubMedQA/train
      - dataset/MedXpertQA_Text/test
    use_vllm: true
    vllm_gpu_mem: 0.6
  caps: {}

smoke_full:
  baseline_root: /home/vcsai/minhlbq/baseline
  train_venv: training_venv312/bin/python
  vllm_venv: vllm_venv312/bin/python
  artifact_root: artifacts/smoke_full
  mlruns_dir: mlruns
  experiment: medical-qa-smoke-full
  registered_model_name: medical-qa-smoke-full
  model_family: qwen
  kg_version: v0
  report_to: none
  build_kg:
    embed_model: abhinand/MedEmbed-large-v0.1
    embed_device: cpu
  stage1:
    base_model: models/Qwen2.5-3B-Instruct
  stage1_5:
    sft_data: data/stage1_5_sft_v2.jsonl
    lora_r: 32
    lora_alpha: 32
  stage2:
    variant: grpo
    use_vllm: true
    vllm_gpu_mem_util: 0.5
    num_generations: 4
    lora_r: 32
    lora_alpha: 64
  eval:
    benchmarks:
      - dataset/MedQA/test
    use_vllm: true
    vllm_gpu_mem: 0.6
  caps:
    max_train_samples: 8
    max_eval_samples: 4
    num_train_epochs: 1
    save_steps: 1
    max_steps: 2
    n_samples: 4
```

- [ ] **Step 2: Write the failing test** — create `tests/mlops/test_full_config.py`:

```python
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/mlops/test_full_config.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'mlops.pipelines.full_config'`.

- [ ] **Step 4: Write `mlops/pipelines/full_config.py`:**

```python
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/mlops/test_full_config.py -q`
Expected: PASS (4 passed).

- [ ] **Step 6: Commit**

```bash
git add params.yaml mlops/pipelines/full_config.py tests/mlops/test_full_config.py
git commit -m "feat(pipeline): full/smoke_full profiles + typed config loader"
```

---

## Task 2: Stage runner (`StageSpec` + `run_stage` dry-run)

**Files:** Create `mlops/pipelines/stage_runner.py`, `tests/mlops/test_stage_runner.py`.

- [ ] **Step 1: Write the failing test** — create `tests/mlops/test_stage_runner.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/mlops/test_stage_runner.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'mlops.pipelines.stage_runner'`.

- [ ] **Step 3: Write `mlops/pipelines/stage_runner.py`:**

```python
"""Run one pipeline stage: build a receipt (dry-run) or execute + log MLflow."""

import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class StageSpec:
    name: str
    commands: list[list[str]]  # one or more argv lists, run in order
    cwd: str
    params: dict = field(default_factory=dict)
    tags: dict = field(default_factory=dict)
    artifact_pointers: dict = field(default_factory=dict)
    eval_metrics_path: str | None = None  # if set, parsed for metrics in real mode


def _receipt(spec: StageSpec, dry_run: bool) -> dict:
    return {
        "stage": spec.name,
        "dry_run": dry_run,
        "commands": [list(c) for c in spec.commands],
        "cwd": spec.cwd,
        "params": dict(spec.params),
        "tags": dict(spec.tags),
        "artifact_pointers": dict(spec.artifact_pointers),
        "eval_metrics_path": spec.eval_metrics_path,
    }


def run_stage(spec: StageSpec, dry_run: bool, mlflow_active: bool = False) -> dict:
    """Dry-run: return the receipt, run nothing. Real: execute commands, log MLflow."""
    receipt = _receipt(spec, dry_run)
    if dry_run:
        return receipt
    _execute_and_log(spec, receipt, mlflow_active)  # pragma: no cover
    return receipt


def _execute_and_log(spec: StageSpec, receipt: dict, mlflow_active: bool) -> None:  # pragma: no cover
    Path(spec.cwd)  # cwd is expected to exist
    logs = []
    for argv in spec.commands:
        proc = subprocess.run(argv, cwd=spec.cwd, capture_output=True, text=True)
        logs.append({"argv": argv, "returncode": proc.returncode})
        if proc.returncode != 0:
            receipt["failed_command"] = argv
            receipt["stderr_tail"] = proc.stderr[-2000:]
            raise RuntimeError(f"stage {spec.name!r} failed: {argv}")
    receipt["command_logs"] = logs

    if not mlflow_active:
        return
    import mlflow

    from .full_stages import parse_eval_metrics

    with mlflow.start_run(run_name=spec.name, nested=True):
        mlflow.set_tags(spec.tags)
        mlflow.log_params(spec.params)
        for ptr_name, ptr_val in spec.artifact_pointers.items():
            mlflow.set_tag(f"artifact.{ptr_name}", str(ptr_val))
        if spec.eval_metrics_path and Path(spec.eval_metrics_path).exists():
            for key, value in parse_eval_metrics(spec.eval_metrics_path).items():
                mlflow.log_metric(key, value)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/mlops/test_stage_runner.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add mlops/pipelines/stage_runner.py tests/mlops/test_stage_runner.py
git commit -m "feat(pipeline): generic stage runner with dry-run receipts"
```

---

## Task 3: Builders part 1 — `build_kg`, `stage1`, `stage1_5`

**Files:** Create `mlops/pipelines/full_stages.py`; Create `tests/mlops/test_full_stages.py`.

- [ ] **Step 1: Write the failing test** — create `tests/mlops/test_full_stages.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/mlops/test_full_stages.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'mlops.pipelines.full_stages'`.

- [ ] **Step 3: Write `mlops/pipelines/full_stages.py`** (this task adds the header, helpers, and the first three builders; Task 4 appends the rest):

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/mlops/test_full_stages.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add mlops/pipelines/full_stages.py tests/mlops/test_full_stages.py
git commit -m "feat(pipeline): build_kg/stage1/stage1_5 command builders"
```

---

## Task 4: Builders part 2 — `stage2`, `eval_stage`, `register_stage`, `parse_eval_metrics`

**Files:** Modify `mlops/pipelines/full_stages.py`; Modify `tests/mlops/test_full_stages.py`.

- [ ] **Step 1: Append failing tests** to `tests/mlops/test_full_stages.py`:

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/mlops/test_full_stages.py -q`
Expected: FAIL — `AttributeError: module 'mlops.pipelines.full_stages' has no attribute 'stage2'`.

- [ ] **Step 3: Append to `mlops/pipelines/full_stages.py`:**

```python
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/mlops/test_full_stages.py -q`
Expected: PASS (11 passed).

- [ ] **Step 5: Commit**

```bash
git add mlops/pipelines/full_stages.py tests/mlops/test_full_stages.py
git commit -m "feat(pipeline): stage2/eval/register builders + eval metric parser"
```

---

## Task 5: Orchestrator (`run_full.py`)

**Files:** Create `mlops/pipelines/run_full.py`, `tests/mlops/test_run_full_dry_run.py`.

- [ ] **Step 1: Write the failing test** — create `tests/mlops/test_run_full_dry_run.py`:

```python
import json

from mlops.pipelines.run_full import run_full


def test_dry_run_smoke_full_emits_all_stages_in_order(tmp_path):
    receipt = run_full(profile="smoke_full", dry_run=True, receipt_path=str(tmp_path / "r.json"))
    stages = [s["stage"] for s in receipt["stages"]]
    assert stages == ["build_kg", "stage1", "stage1_5", "stage2", "eval", "register"]
    assert receipt["profile"] == "smoke_full"
    assert receipt["tags"]["model_family"] == "qwen"
    assert "retrieval_contract_version" in receipt["tags"]
    # build_kg argv carries the embed model
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/mlops/test_run_full_dry_run.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'mlops.pipelines.run_full'`.

- [ ] **Step 3: Write `mlops/pipelines/run_full.py`:**

```python
"""Orchestrate the full training pipeline: parent MLflow run + per-stage runs."""

import argparse
import json
from pathlib import Path

from .full_config import load_full_config, stage_paths
from .full_stages import BUILDERS, STAGE_ORDER
from .stage_runner import run_stage


def _contract_version() -> str:
    try:
        from medical_qa_platform.retrieval.contract import RETRIEVAL_CONTRACT_VERSION

        return RETRIEVAL_CONTRACT_VERSION
    except Exception:
        return "unknown"


def _parent_tags(cfg) -> dict:
    return {
        "profile": cfg.name,
        "model_family": cfg.model_family,
        "kg_version": cfg.kg_version,
        "retrieval_contract_version": _contract_version(),
    }


def run_full(profile: str, dry_run: bool, stages: list[str] | None = None,
             receipt_path: str | None = None, params_path: str = "params.yaml") -> dict:
    cfg = load_full_config(profile, params_path)
    paths = stage_paths(cfg)
    selected = stages or STAGE_ORDER
    tags = _parent_tags(cfg)

    pipeline_receipt = {"profile": cfg.name, "dry_run": dry_run, "tags": tags, "stages": []}

    if dry_run:
        for name in selected:
            spec = BUILDERS[name](cfg, paths)
            pipeline_receipt["stages"].append(run_stage(spec, dry_run=True))
    else:  # pragma: no cover
        _run_real(cfg, paths, selected, tags, pipeline_receipt)

    if receipt_path:
        Path(receipt_path).parent.mkdir(parents=True, exist_ok=True)
        Path(receipt_path).write_text(json.dumps(pipeline_receipt, indent=2, sort_keys=True))
    return pipeline_receipt


def _run_real(cfg, paths, selected, tags, pipeline_receipt):  # pragma: no cover
    import mlflow

    mlflow.set_tracking_uri(f"file:{cfg.mlruns_dir}")
    mlflow.set_experiment(cfg.experiment)
    with mlflow.start_run(run_name=f"{cfg.name}-pipeline") as parent:
        mlflow.set_tags(tags)
        for name in selected:
            spec = BUILDERS[name](cfg, paths)
            if name == "register":
                _register(cfg, paths, spec)
                pipeline_receipt["stages"].append({"stage": "register", "dry_run": False})
                continue
            pipeline_receipt["stages"].append(run_stage(spec, dry_run=False, mlflow_active=True))
        pipeline_receipt["parent_run_id"] = parent.info.run_id


def _register(cfg, paths, spec):  # pragma: no cover
    import mlflow

    with mlflow.start_run(run_name="register", nested=True):
        mlflow.set_tags(spec.tags)
        mlflow.log_params(spec.params)
        mlflow.set_tag("artifact.model", str(paths["stage2_merged"]))
        from .full_stages import parse_eval_metrics

        if paths["eval_out"].exists():
            for key, value in parse_eval_metrics(str(paths["eval_out"])).items():
                mlflow.log_metric(key, value)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="full")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stages", default=None, help="comma-separated subset")
    args = ap.parse_args()
    stages = args.stages.split(",") if args.stages else None
    cfg = load_full_config(args.profile)
    receipt = str((cfg.artifact_root / "run_full_receipt.json"))
    run_full(profile=args.profile, dry_run=args.dry_run, stages=stages, receipt_path=receipt)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/mlops/test_run_full_dry_run.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add mlops/pipelines/run_full.py tests/mlops/test_run_full_dry_run.py
git commit -m "feat(pipeline): run_full orchestrator with MLflow parent/nested runs"
```

---

## Task 6: DVC full stages

**Files:** Modify `dvc.yaml`; Create `tests/mlops/test_dvc_full_stages.py`.

- [ ] **Step 1: Write the failing test** — create `tests/mlops/test_dvc_full_stages.py`:

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/mlops/test_dvc_full_stages.py -q`
Expected: FAIL — `KeyError: 'full_build_kg'` (or assertion error).

- [ ] **Step 3: Append the full stages to `dvc.yaml`** (under the existing `stages:` map, after the smoke stages):

```yaml
  full_build_kg:
    cmd: .venv/bin/python -m mlops.pipelines.run_full --profile full --stages build_kg
    deps:
      - mlops/pipelines/run_full.py
      - mlops/pipelines/full_stages.py
      - mlops/pipelines/full_config.py
      - params.yaml
    outs:
      - artifacts/full/kg:
          cache: false

  full_stage1:
    cmd: .venv/bin/python -m mlops.pipelines.run_full --profile full --stages stage1
    deps:
      - mlops/pipelines/full_stages.py
      - params.yaml
    outs:
      - artifacts/full/outputs/stage1:
          cache: false

  full_stage1_5:
    cmd: .venv/bin/python -m mlops.pipelines.run_full --profile full --stages stage1_5
    deps:
      - mlops/pipelines/full_stages.py
      - params.yaml
      - artifacts/full/outputs/stage1
    outs:
      - artifacts/full/outputs/stage1_5_merged:
          cache: false

  full_stage2:
    cmd: .venv/bin/python -m mlops.pipelines.run_full --profile full --stages stage2
    deps:
      - mlops/pipelines/full_stages.py
      - params.yaml
      - artifacts/full/outputs/stage1_5_merged
      - artifacts/full/kg
    outs:
      - artifacts/full/outputs/stage2_merged:
          cache: false

  full_eval:
    cmd: .venv/bin/python -m mlops.pipelines.run_full --profile full --stages eval
    deps:
      - mlops/pipelines/full_stages.py
      - params.yaml
      - artifacts/full/outputs/stage2_merged
      - artifacts/full/kg
    outs:
      - artifacts/full/eval/eval.json:
          cache: false

  full_register:
    cmd: .venv/bin/python -m mlops.pipelines.run_full --profile full --stages register
    deps:
      - mlops/pipelines/run_full.py
      - params.yaml
      - artifacts/full/eval/eval.json
    outs:
      - artifacts/full/run_full_receipt.json:
          cache: false
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/mlops/test_dvc_full_stages.py -q`
Expected: PASS (3 passed). Also confirm DVC can parse the file: `.venv/bin/dvc status 2>&1 | head -3` does not raise a YAML/schema error (it may report stages as "changed"/"not in cache" — that is fine; we are not running them).

- [ ] **Step 5: Commit**

```bash
git add dvc.yaml tests/mlops/test_dvc_full_stages.py
git commit -m "feat(pipeline): DVC full-pipeline stages (lineage; checkpoints uncached)"
```

---

## Task 7: Makefile targets + `.gitignore`

**Files:** Modify `Makefile`, `.gitignore`; Create `tests/mlops/test_makefile_full_targets.py`.

- [ ] **Step 1: Write the failing test** — create `tests/mlops/test_makefile_full_targets.py`:

```python
def test_makefile_has_full_targets():
    text = open("Makefile").read()
    for target in ["full-pipeline:", "full-pipeline-dry-run:", "smoke-full:"]:
        assert target in text


def test_full_dry_run_target_uses_run_full_dry_run():
    text = open("Makefile").read()
    assert "mlops.pipelines.run_full --profile full --dry-run" in text


def test_gitignore_excludes_generated_dirs():
    text = open(".gitignore").read()
    assert "artifacts/" in text
    assert "mlruns/" in text
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/mlops/test_makefile_full_targets.py -q`
Expected: FAIL.

- [ ] **Step 3a: Add targets to `Makefile`** — add to the `.PHONY` line the names `full-pipeline full-pipeline-dry-run smoke-full`, and append these targets:

```makefile
full-pipeline-dry-run:
	$(PY) -m mlops.pipelines.run_full --profile full --dry-run

full-pipeline:
	$(PY) -m mlops.pipelines.run_full --profile full

smoke-full:
	$(PY) -m mlops.pipelines.run_full --profile smoke_full
```

- [ ] **Step 3b: Ensure `.gitignore` ignores generated dirs.** If `.gitignore` does not already contain them, append:

```gitignore
artifacts/
mlruns/
```

(If `artifacts/` is already ignored for the smoke pipeline, only add `mlruns/`. Do not remove existing entries.)

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/mlops/test_makefile_full_targets.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add Makefile .gitignore tests/mlops/test_makefile_full_targets.py
git commit -m "feat(pipeline): make targets (full-pipeline, dry-run, smoke-full) + gitignore"
```

---

## Task 8: Full-suite + coverage gate

**Files:** none (verification only).

- [ ] **Step 1: Run the full suite with coverage**

Run: `.venv/bin/python -m pytest --cov --cov-report=term-missing -q 2>&1 | tail -25`
Expected: all tests pass; total coverage ≥ 80%. NOTE: coverage `source = ["medical_qa_platform"]` only, so the new `mlops/` modules are not measured (same as the existing smoke pipeline) — the gate stays satisfied by the unchanged `medical_qa_platform` package. The `# pragma: no cover` markers on the real subprocess/MLflow paths are hygiene for when/if `mlops` is added to the coverage source later.

- [ ] **Step 2: Confirm the full dry-run actually produces a complete receipt**

Run: `.venv/bin/python -m mlops.pipelines.run_full --profile full --dry-run && .venv/bin/python -c "import json; d=json.load(open('artifacts/full/run_full_receipt.json')); print([s['stage'] for s in d['stages']]); print('tags:', d['tags'])"`
Expected: prints `['build_kg', 'stage1', 'stage1_5', 'stage2', 'eval', 'register']` and the parent tags including `retrieval_contract_version`.

- [ ] **Step 3: Confirm the smoke pipeline + prior tests still pass**

Run: `.venv/bin/python -m pytest tests/mlops -q`
Expected: all green (existing smoke tests + the six new full-pipeline test files).

- [ ] **Step 4: Commit (if any incidental change); otherwise stop**

```bash
git status
# clean -> SP3 code complete on branch full-mlflow-pipeline
```

---

## Notes for the implementer

- **Never run real training in CI.** Every test here is dry-run/structural and runs without GPU, baseline venvs, or MLflow. The real `make full-pipeline` and the one-off `make smoke-full` are run by the operator/controller on the GB10.
- **Authoritative flags only.** The builders use the flags listed in the header (extracted from the scripts' argparse). Do not "correct" them against CLAUDE.md — CLAUDE.md is stale in places (e.g. merge uses `--output-dir`, not `--output-path`; eval uses `--vllm-gpu-mem`, not `--vllm-gpu-mem-util`).
- **`mlops/` may reference `baseline/` paths** (it is outside `src/`); it must never `import` baseline Python — orchestration is subprocess only.
- **Deterministic paths:** Stage 1 model = `<stage1_out>` (root); Stage 1.5 / Stage 2 adapters = `<*_out>/final`. No runtime checkpoint globbing.
- **`smoke_full` real run** is the wiring gate (controller-run once, on the GB10): run a cheap representative subset — e.g. `make smoke-full` limited to `--stages stage2,eval` reusing an existing Stage-1.5 merged checkpoint — to prove the baseline scripts + MLflow logging actually run. Not in CI.
