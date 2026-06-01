# MLOps P0 — DVC + MLflow Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a self-contained smoke MLOps pipeline under `mlops-platform/` that uses DVC for artifact lineage, Makefile targets for reproducible execution, and MLflow for smoke metrics/model-version registration.

**Architecture:** The pipeline is package-owned and does not import from a sibling `baseline/` checkout. A tiny deterministic smoke dataset and KG fixture are generated locally, evaluated through the existing prompt/parser/mock backend code, then logged through a dry-run-capable MLflow registration script. Heavy/full training remains deferred; this plan proves the DVC + MLflow spine with fast local tests.

**Tech Stack:** Python 3.12, DVC, MLflow, PyYAML, pytest, existing `medical_qa_platform` runtime package.

---

## Scope

This is plan 2 of 4 for P0. It delivers:

- `params.yaml` and `dvc.yaml` for a smoke pipeline.
- `mlops/` pipeline modules for profile loading, smoke KG artifact generation, smoke evaluation, and MLflow registration.
- Make targets: `install-pipeline`, `smoke-pipeline`, `smoke-pipeline-local`, `mlflow-register-dry-run`, `register-model`, `dvc-status`.
- Tests for profile parsing, DVC stage shape, smoke KG generation, smoke eval, MLflow dry-run registration, and Makefile targets.

This plan does not build Docker images, Helm charts, KServe manifests, GKE infrastructure, or a real RunPod endpoint. Those are plan 3 and plan 4.

## File Structure

**Create:**
- `mlops-platform/params.yaml` — smoke pipeline parameters.
- `mlops-platform/dvc.yaml` — DVC stages for build KG, eval, and register dry-run.
- `mlops-platform/mlops/__init__.py` — package marker.
- `mlops-platform/mlops/pipelines/__init__.py` — package marker.
- `mlops-platform/mlops/pipelines/profiles.py` — typed profile loader.
- `mlops-platform/mlops/pipelines/build_smoke_kg.py` — deterministic smoke KG/fixture artifact generator.
- `mlops-platform/mlops/pipelines/eval_smoke.py` — deterministic MCQ smoke evaluator.
- `mlops-platform/mlops/mlflow_register.py` — MLflow logging/register script with dry-run mode.
- `mlops-platform/mlops/smoke_data/medical_mcq.jsonl` — tiny smoke MCQ dataset.
- `mlops-platform/tests/mlops/test_profiles.py`
- `mlops-platform/tests/mlops/test_dvc_yaml.py`
- `mlops-platform/tests/mlops/test_build_smoke_kg.py`
- `mlops-platform/tests/mlops/test_eval_smoke.py`
- `mlops-platform/tests/mlops/test_mlflow_register.py`
- `mlops-platform/tests/test_pipeline_tooling.py`

**Modify:**
- `mlops-platform/pyproject.toml` — add `pipeline` optional dependencies and PyYAML test dependency.
- `mlops-platform/Makefile` — add pipeline targets.
- `mlops-platform/README.md` — document smoke pipeline usage and artifact flow.

---

## Task 1: Pipeline Dependencies + Make Targets

**Files:**
- Modify: `mlops-platform/pyproject.toml`
- Modify: `mlops-platform/Makefile`
- Test: `mlops-platform/tests/test_pipeline_tooling.py`

- [ ] **Step 1: Write the failing tests** `mlops-platform/tests/test_pipeline_tooling.py`

```python
from pathlib import Path
import tomllib


ROOT = Path(__file__).parents[1]


def test_pipeline_extra_contains_dvc_mlflow_and_pyyaml():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    optional = data["project"]["optional-dependencies"]
    pipeline = "\n".join(optional["pipeline"])
    assert "dvc" in pipeline
    assert "mlflow" in pipeline
    assert "PyYAML" in pipeline


def test_dev_extra_contains_pyyaml_for_yaml_shape_tests():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    dev = "\n".join(data["project"]["optional-dependencies"]["dev"])
    assert "PyYAML" in dev


def test_makefile_exposes_pipeline_targets():
    text = (ROOT / "Makefile").read_text()
    for target in [
        "install-pipeline:",
        "smoke-pipeline:",
        "smoke-pipeline-local:",
        "mlflow-register-dry-run:",
        "register-model:",
        "dvc-status:",
    ]:
        assert target in text
```

- [ ] **Step 2: Run to verify failure**

Run:
```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/test_pipeline_tooling.py -v
```
Expected: FAIL because `pipeline` optional dependencies and Make targets do not exist.

- [ ] **Step 3: Update `mlops-platform/pyproject.toml`**

Edit `[project.optional-dependencies]` so it contains:

```toml
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "PyYAML>=6.0",
]
pipeline = [
    "dvc>=3.50",
    "mlflow>=2.12",
    "PyYAML>=6.0",
]
# Heavy adapters used only by the retrieval container; never in unit tests.
runtime = [
    "faiss-cpu>=1.8",
    "sentence-transformers>=2.7",
    "numpy>=1.26",
]
```

- [ ] **Step 4: Update `mlops-platform/Makefile`**

Replace the file with:

```makefile
PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: install install-pipeline test smoke-pipeline smoke-pipeline-local mlflow-register-dry-run register-model dvc-status

.venv:
	python3.12 -m venv .venv
	$(PIP) install -U pip

install: .venv
	$(PIP) install -e ".[dev]"

install-pipeline: .venv
	$(PIP) install -e ".[dev,pipeline]"

test:
	.venv/bin/pytest --cov=medical_qa_platform --cov-report=term-missing

smoke-pipeline:
	.venv/bin/dvc repro

smoke-pipeline-local:
	$(PY) -m mlops.pipelines.build_smoke_kg --profile smoke
	$(PY) -m mlops.pipelines.eval_smoke --profile smoke
	$(PY) -m mlops.mlflow_register --profile smoke --dry-run

mlflow-register-dry-run:
	$(PY) -m mlops.mlflow_register --profile smoke --dry-run

register-model:
	$(PY) -m mlops.mlflow_register --profile smoke

dvc-status:
	.venv/bin/dvc status
```

- [ ] **Step 5: Run to verify pass**

Run:
```bash
cd /home/vcsai/minhlbq/mlops-platform && make install && .venv/bin/pytest tests/test_pipeline_tooling.py -v
```
Expected: installs the refreshed dev dependencies, then 3 passed.

- [ ] **Step 6: Commit**

```bash
cd /home/vcsai/minhlbq/mlops-platform
git add pyproject.toml Makefile tests/test_pipeline_tooling.py
git commit -m "chore: add pipeline dependencies and make targets"
```

---

## Task 2: Params and DVC Stage Definitions

**Files:**
- Create: `mlops-platform/params.yaml`
- Create: `mlops-platform/dvc.yaml`
- Test: `mlops-platform/tests/mlops/test_dvc_yaml.py`

- [ ] **Step 1: Write the failing tests** `mlops-platform/tests/mlops/test_dvc_yaml.py`

```python
from pathlib import Path

import yaml


ROOT = Path(__file__).parents[2]


def test_params_yaml_defines_smoke_profile_paths():
    params = yaml.safe_load((ROOT / "params.yaml").read_text())
    smoke = params["smoke"]
    assert smoke["data_dir"] == "mlops/smoke_data"
    assert smoke["artifact_dir"] == "artifacts/smoke"
    assert smoke["kg_dir"] == "artifacts/smoke/kg"
    assert smoke["predictions_path"] == "artifacts/smoke/eval_predictions.jsonl"
    assert smoke["metrics_path"] == "artifacts/smoke/eval_metrics.json"
    assert smoke["mlflow_receipt_path"] == "artifacts/smoke/mlflow_register_dry_run.json"
    assert smoke["top_k"] == 2
    assert smoke["model_backend"] == "mock"
    assert smoke["model_version"] == "smoke-dev"
    assert smoke["registered_model_name"] == "medical-qa-smoke"


def test_dvc_yaml_has_expected_smoke_stages():
    dvc = yaml.safe_load((ROOT / "dvc.yaml").read_text())
    stages = dvc["stages"]
    assert set(stages) == {"build_kg_smoke", "eval_smoke", "register_smoke_model"}
    assert stages["build_kg_smoke"]["cmd"] == (
        "python -m mlops.pipelines.build_smoke_kg --profile smoke"
    )
    assert stages["eval_smoke"]["cmd"] == (
        "python -m mlops.pipelines.eval_smoke --profile smoke"
    )
    assert stages["register_smoke_model"]["cmd"] == (
        "python -m mlops.mlflow_register --profile smoke --dry-run"
    )
    assert "artifacts/smoke/kg/manifest.json" in stages["build_kg_smoke"]["outs"]
    assert "artifacts/smoke/eval_metrics.json" in stages["eval_smoke"]["outs"]
    assert (
        "artifacts/smoke/mlflow_register_dry_run.json"
        in stages["register_smoke_model"]["outs"]
    )
```

- [ ] **Step 2: Run to verify failure**

Run:
```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/mlops/test_dvc_yaml.py -v
```
Expected: FAIL because `params.yaml` and `dvc.yaml` do not exist.

- [ ] **Step 3: Create `mlops-platform/params.yaml`**

```yaml
smoke:
  data_dir: mlops/smoke_data
  artifact_dir: artifacts/smoke
  kg_dir: artifacts/smoke/kg
  retrieval_fixture_path: artifacts/smoke/kg/retrieval_fixture.json
  predictions_path: artifacts/smoke/eval_predictions.jsonl
  metrics_path: artifacts/smoke/eval_metrics.json
  mlflow_receipt_path: artifacts/smoke/mlflow_register_dry_run.json
  top_k: 2
  model_backend: mock
  model_version: smoke-dev
  registered_model_name: medical-qa-smoke
```

- [ ] **Step 4: Create `mlops-platform/dvc.yaml`**

```yaml
stages:
  build_kg_smoke:
    cmd: python -m mlops.pipelines.build_smoke_kg --profile smoke
    deps:
      - mlops/pipelines/build_smoke_kg.py
      - mlops/pipelines/profiles.py
      - params.yaml
    outs:
      - artifacts/smoke/kg/medical_hg.json
      - artifacts/smoke/kg/hedge_ids.json
      - artifacts/smoke/kg/entity_names.json
      - artifacts/smoke/kg/retrieval_fixture.json
      - artifacts/smoke/kg/manifest.json

  eval_smoke:
    cmd: python -m mlops.pipelines.eval_smoke --profile smoke
    deps:
      - mlops/pipelines/eval_smoke.py
      - mlops/pipelines/profiles.py
      - mlops/smoke_data/medical_mcq.jsonl
      - artifacts/smoke/kg/retrieval_fixture.json
      - params.yaml
    outs:
      - artifacts/smoke/eval_predictions.jsonl
      - artifacts/smoke/eval_metrics.json

  register_smoke_model:
    cmd: python -m mlops.mlflow_register --profile smoke --dry-run
    deps:
      - mlops/mlflow_register.py
      - artifacts/smoke/eval_metrics.json
      - artifacts/smoke/eval_predictions.jsonl
      - params.yaml
    outs:
      - artifacts/smoke/mlflow_register_dry_run.json
```

- [ ] **Step 5: Run to verify pass**

Run:
```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/mlops/test_dvc_yaml.py -v
```
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
cd /home/vcsai/minhlbq/mlops-platform
git add params.yaml dvc.yaml tests/mlops/test_dvc_yaml.py
git commit -m "chore: define smoke DVC pipeline"
```

---

## Task 3: Typed Pipeline Profile Loader

**Files:**
- Create: `mlops-platform/mlops/__init__.py`
- Create: `mlops-platform/mlops/pipelines/__init__.py`
- Create: `mlops-platform/mlops/pipelines/profiles.py`
- Test: `mlops-platform/tests/mlops/test_profiles.py`

- [ ] **Step 1: Write the failing tests** `mlops-platform/tests/mlops/test_profiles.py`

```python
from pathlib import Path

import pytest

from mlops.pipelines.profiles import PipelineProfile, load_profile


def test_load_smoke_profile_from_params():
    profile = load_profile("smoke")
    assert isinstance(profile, PipelineProfile)
    assert profile.name == "smoke"
    assert profile.top_k == 2
    assert profile.model_backend == "mock"
    assert profile.model_version == "smoke-dev"
    assert profile.registered_model_name == "medical-qa-smoke"
    assert profile.metrics_path == Path("artifacts/smoke/eval_metrics.json")


def test_unknown_profile_raises_clear_error():
    with pytest.raises(ValueError, match="unknown pipeline profile"):
        load_profile("missing")
```

- [ ] **Step 2: Run to verify failure**

Run:
```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/mlops/test_profiles.py -v
```
Expected: FAIL because `mlops.pipelines.profiles` does not exist.

- [ ] **Step 3: Create package marker files**

`mlops-platform/mlops/__init__.py`:

```python
"""MLOps pipeline helpers for the Medical QA platform."""
```

`mlops-platform/mlops/pipelines/__init__.py`:

```python
"""Offline smoke/full pipeline entrypoints."""
```

- [ ] **Step 4: Implement `mlops-platform/mlops/pipelines/profiles.py`**

```python
"""Load typed pipeline profiles from params.yaml."""

from dataclasses import dataclass
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PipelineProfile:
    name: str
    data_dir: Path
    artifact_dir: Path
    kg_dir: Path
    retrieval_fixture_path: Path
    predictions_path: Path
    metrics_path: Path
    mlflow_receipt_path: Path
    top_k: int
    model_backend: str
    model_version: str
    registered_model_name: str


def _path(value: str) -> Path:
    return Path(value)


def load_profile(name: str, params_path: Path | str = "params.yaml") -> PipelineProfile:
    params_file = Path(params_path)
    data = yaml.safe_load(params_file.read_text())
    if name not in data:
        raise ValueError(f"unknown pipeline profile: {name!r}")
    raw = data[name]
    return PipelineProfile(
        name=name,
        data_dir=_path(raw["data_dir"]),
        artifact_dir=_path(raw["artifact_dir"]),
        kg_dir=_path(raw["kg_dir"]),
        retrieval_fixture_path=_path(raw["retrieval_fixture_path"]),
        predictions_path=_path(raw["predictions_path"]),
        metrics_path=_path(raw["metrics_path"]),
        mlflow_receipt_path=_path(raw["mlflow_receipt_path"]),
        top_k=int(raw["top_k"]),
        model_backend=raw["model_backend"],
        model_version=raw["model_version"],
        registered_model_name=raw["registered_model_name"],
    )
```

- [ ] **Step 5: Run to verify pass**

Run:
```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/mlops/test_profiles.py -v
```
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
cd /home/vcsai/minhlbq/mlops-platform
git add mlops/__init__.py mlops/pipelines/__init__.py \
        mlops/pipelines/profiles.py tests/mlops/test_profiles.py
git commit -m "feat: typed pipeline profile loader"
```

---

## Task 4: Smoke Dataset

**Files:**
- Create: `mlops-platform/mlops/smoke_data/medical_mcq.jsonl`
- Test: `mlops-platform/tests/mlops/test_smoke_data.py`

- [ ] **Step 1: Write the failing tests** `mlops-platform/tests/mlops/test_smoke_data.py`

```python
import json
from pathlib import Path


ROOT = Path(__file__).parents[2]


def test_smoke_dataset_has_two_valid_mcq_rows():
    path = ROOT / "mlops/smoke_data/medical_mcq.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 2
    for row in rows:
        assert row["id"]
        assert row["question"]
        assert set(row["options"]) == {"A", "B"}
        assert row["answer"] in row["options"]
        assert row["mock_answer"] == row["answer"]
        assert isinstance(row["retrieval_query"], str)
```

- [ ] **Step 2: Run to verify failure**

Run:
```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/mlops/test_smoke_data.py -v
```
Expected: FAIL because `mlops/smoke_data/medical_mcq.jsonl` does not exist.

- [ ] **Step 3: Create `mlops-platform/mlops/smoke_data/medical_mcq.jsonl`**

```jsonl
{"id":"smoke-001","question":"Which medication is commonly used first-line for type 2 diabetes?","options":{"A":"Metformin","B":"Amoxicillin"},"answer":"A","mock_answer":"A","retrieval_query":"type 2 diabetes first line medication"}
{"id":"smoke-002","question":"Which hormone is deficient in type 1 diabetes?","options":{"A":"Insulin","B":"Thyroxine"},"answer":"A","mock_answer":"A","retrieval_query":"type 1 diabetes deficient hormone"}
```

- [ ] **Step 4: Run to verify pass**

Run:
```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/mlops/test_smoke_data.py -v
```
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/vcsai/minhlbq/mlops-platform
git add mlops/smoke_data/medical_mcq.jsonl tests/mlops/test_smoke_data.py
git commit -m "testdata: add smoke MCQ dataset"
```

---

## Task 5: Smoke KG Artifact Generator

**Files:**
- Create: `mlops-platform/mlops/pipelines/build_smoke_kg.py`
- Test: `mlops-platform/tests/mlops/test_build_smoke_kg.py`

- [ ] **Step 1: Write the failing tests** `mlops-platform/tests/mlops/test_build_smoke_kg.py`

```python
import json
from pathlib import Path

from mlops.pipelines.build_smoke_kg import build_smoke_kg
from mlops.pipelines.profiles import PipelineProfile


def _profile(tmp_path: Path) -> PipelineProfile:
    artifact_dir = tmp_path / "artifacts/smoke"
    return PipelineProfile(
        name="smoke",
        data_dir=tmp_path / "smoke_data",
        artifact_dir=artifact_dir,
        kg_dir=artifact_dir / "kg",
        retrieval_fixture_path=artifact_dir / "kg/retrieval_fixture.json",
        predictions_path=artifact_dir / "eval_predictions.jsonl",
        metrics_path=artifact_dir / "eval_metrics.json",
        mlflow_receipt_path=artifact_dir / "mlflow_register_dry_run.json",
        top_k=2,
        model_backend="mock",
        model_version="smoke-dev",
        registered_model_name="medical-qa-smoke",
    )


def test_build_smoke_kg_writes_expected_artifacts(tmp_path):
    manifest = build_smoke_kg(_profile(tmp_path))
    kg_dir = tmp_path / "artifacts/smoke/kg"
    assert (kg_dir / "medical_hg.json").exists()
    assert (kg_dir / "hedge_ids.json").exists()
    assert (kg_dir / "entity_names.json").exists()
    assert (kg_dir / "retrieval_fixture.json").exists()
    assert (kg_dir / "manifest.json").exists()
    assert manifest["artifact_type"] == "smoke_kg"
    assert manifest["n_hyperedges"] == 2


def test_retrieval_fixture_maps_queries_to_evidence(tmp_path):
    build_smoke_kg(_profile(tmp_path))
    fixture = json.loads((tmp_path / "artifacts/smoke/kg/retrieval_fixture.json").read_text())
    assert fixture["type 2 diabetes first line medication"] == [
        "Metformin is commonly used first-line for type 2 diabetes."
    ]
    assert fixture["type 1 diabetes deficient hormone"] == [
        "Type 1 diabetes is characterized by insulin deficiency."
    ]
```

- [ ] **Step 2: Run to verify failure**

Run:
```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/mlops/test_build_smoke_kg.py -v
```
Expected: FAIL because `mlops.pipelines.build_smoke_kg` does not exist.

- [ ] **Step 3: Implement `mlops-platform/mlops/pipelines/build_smoke_kg.py`**

```python
"""Generate tiny deterministic KG artifacts for the smoke pipeline."""

import argparse
import json
from pathlib import Path

from .profiles import PipelineProfile, load_profile


HYPEREDGES = [
    {
        "id": "h-metformin-diabetes",
        "description": "Metformin is commonly used first-line for type 2 diabetes.",
        "relation": "treats",
        "type": "drug",
        "anchor": "Metformin",
        "entities": ["type 2 diabetes"],
    },
    {
        "id": "h-insulin-diabetes",
        "description": "Type 1 diabetes is characterized by insulin deficiency.",
        "relation": "deficient_in",
        "type": "hormone",
        "anchor": "Insulin",
        "entities": ["type 1 diabetes"],
    },
]

ENTITIES = {
    "Metformin": {"type": "drug"},
    "Insulin": {"type": "hormone"},
    "type 1 diabetes": {"type": "disease"},
    "type 2 diabetes": {"type": "disease"},
}

ENTITY_TO_HEDGES = {
    "Metformin": ["h-metformin-diabetes"],
    "Insulin": ["h-insulin-diabetes"],
    "type 1 diabetes": ["h-insulin-diabetes"],
    "type 2 diabetes": ["h-metformin-diabetes"],
}

RETRIEVAL_FIXTURE = {
    "type 2 diabetes first line medication": [
        "Metformin is commonly used first-line for type 2 diabetes."
    ],
    "type 1 diabetes deficient hormone": [
        "Type 1 diabetes is characterized by insulin deficiency."
    ],
}


def _write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_smoke_kg(profile: PipelineProfile) -> dict:
    graph = {
        "hyperedges": HYPEREDGES,
        "entities": ENTITIES,
        "entity_to_hedges": ENTITY_TO_HEDGES,
    }
    _write_json(profile.kg_dir / "medical_hg.json", graph)
    _write_json(profile.kg_dir / "hedge_ids.json", [item["id"] for item in HYPEREDGES])
    _write_json(profile.kg_dir / "entity_names.json", list(ENTITIES))
    _write_json(profile.retrieval_fixture_path, RETRIEVAL_FIXTURE)

    manifest = {
        "artifact_type": "smoke_kg",
        "profile": profile.name,
        "n_hyperedges": len(HYPEREDGES),
        "n_entities": len(ENTITIES),
        "retrieval_fixture_path": str(profile.retrieval_fixture_path),
    }
    _write_json(profile.kg_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="smoke")
    args = parser.parse_args()
    build_smoke_kg(load_profile(args.profile))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

Run:
```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/mlops/test_build_smoke_kg.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/vcsai/minhlbq/mlops-platform
git add mlops/pipelines/build_smoke_kg.py tests/mlops/test_build_smoke_kg.py
git commit -m "feat: generate smoke KG artifacts"
```

---

## Task 6: Deterministic Smoke Evaluator

**Files:**
- Create: `mlops-platform/mlops/pipelines/eval_smoke.py`
- Test: `mlops-platform/tests/mlops/test_eval_smoke.py`

- [ ] **Step 1: Write the failing tests** `mlops-platform/tests/mlops/test_eval_smoke.py`

```python
import json
from pathlib import Path

from mlops.pipelines.build_smoke_kg import build_smoke_kg
from mlops.pipelines.eval_smoke import evaluate_smoke
from mlops.pipelines.profiles import PipelineProfile


def _profile(tmp_path: Path) -> PipelineProfile:
    data_dir = tmp_path / "smoke_data"
    data_dir.mkdir()
    (data_dir / "medical_mcq.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "q1",
                        "question": "Which drug treats type 2 diabetes?",
                        "options": {"A": "Metformin", "B": "Amoxicillin"},
                        "answer": "A",
                        "mock_answer": "A",
                        "retrieval_query": "type 2 diabetes first line medication",
                    }
                ),
                json.dumps(
                    {
                        "id": "q2",
                        "question": "Which hormone is deficient in type 1 diabetes?",
                        "options": {"A": "Insulin", "B": "Thyroxine"},
                        "answer": "A",
                        "mock_answer": "A",
                        "retrieval_query": "type 1 diabetes deficient hormone",
                    }
                ),
            ]
        )
        + "\n"
    )
    artifact_dir = tmp_path / "artifacts/smoke"
    return PipelineProfile(
        name="smoke",
        data_dir=data_dir,
        artifact_dir=artifact_dir,
        kg_dir=artifact_dir / "kg",
        retrieval_fixture_path=artifact_dir / "kg/retrieval_fixture.json",
        predictions_path=artifact_dir / "eval_predictions.jsonl",
        metrics_path=artifact_dir / "eval_metrics.json",
        mlflow_receipt_path=artifact_dir / "mlflow_register_dry_run.json",
        top_k=2,
        model_backend="mock",
        model_version="smoke-dev",
        registered_model_name="medical-qa-smoke",
    )


def test_evaluate_smoke_writes_metrics_and_predictions(tmp_path):
    profile = _profile(tmp_path)
    build_smoke_kg(profile)
    metrics = evaluate_smoke(profile)
    assert metrics["n_examples"] == 2
    assert metrics["accuracy"] == 1.0
    assert metrics["retrieval_no_result_rate"] == 0.0
    assert profile.metrics_path.exists()
    assert profile.predictions_path.exists()
    rows = [json.loads(line) for line in profile.predictions_path.read_text().splitlines()]
    assert rows[0]["predicted_answer"] == "A"
    assert rows[0]["is_correct"] is True
    assert rows[0]["evidence"]
```

- [ ] **Step 2: Run to verify failure**

Run:
```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/mlops/test_eval_smoke.py -v
```
Expected: FAIL because `mlops.pipelines.eval_smoke` does not exist.

- [ ] **Step 3: Implement `mlops-platform/mlops/pipelines/eval_smoke.py`**

```python
"""Run a deterministic smoke evaluation and write metrics/artifacts."""

import argparse
import json
import time
from pathlib import Path

from medical_qa_platform.api.parser import parse_answer
from medical_qa_platform.api.prompt import build_prompt
from medical_qa_platform.inference.mock_backend import MockBackend
from medical_qa_platform.retrieval.backends import FixtureRetrieval

from .profiles import PipelineProfile, load_profile


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def evaluate_smoke(profile: PipelineProfile) -> dict:
    data_path = profile.data_dir / "medical_mcq.jsonl"
    examples = _read_jsonl(data_path)
    fixture = json.loads(profile.retrieval_fixture_path.read_text())
    retrieval = FixtureRetrieval(fixture)

    predictions = []
    correct = 0
    no_result = 0
    latencies_ms = []

    for example in examples:
        started = time.perf_counter()
        evidence = retrieval.search(example["retrieval_query"], profile.top_k)
        backend = MockBackend(answer=example["mock_answer"])
        messages = build_prompt(example["question"], example["options"], evidence)
        raw = backend.generate(messages)
        predicted = parse_answer(raw, valid_letters=set(example["options"]))
        latency_ms = (time.perf_counter() - started) * 1000.0
        is_correct = predicted == example["answer"]
        correct += int(is_correct)
        no_result += int(len(evidence) == 0)
        latencies_ms.append(latency_ms)
        predictions.append(
            {
                "id": example["id"],
                "question": example["question"],
                "expected_answer": example["answer"],
                "predicted_answer": predicted,
                "is_correct": is_correct,
                "evidence": evidence,
                "latency_ms": latency_ms,
                "model_version": profile.model_version,
            }
        )

    profile.predictions_path.parent.mkdir(parents=True, exist_ok=True)
    profile.predictions_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions)
    )

    n_examples = len(examples)
    metrics = {
        "profile": profile.name,
        "model_backend": profile.model_backend,
        "model_version": profile.model_version,
        "n_examples": n_examples,
        "accuracy": correct / n_examples if n_examples else 0.0,
        "retrieval_no_result_rate": no_result / n_examples if n_examples else 0.0,
        "latency_ms_avg": sum(latencies_ms) / n_examples if n_examples else 0.0,
    }
    _write_json(profile.metrics_path, metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="smoke")
    args = parser.parse_args()
    evaluate_smoke(load_profile(args.profile))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

Run:
```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/mlops/test_eval_smoke.py -v
```
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/vcsai/minhlbq/mlops-platform
git add mlops/pipelines/eval_smoke.py tests/mlops/test_eval_smoke.py
git commit -m "feat: add deterministic smoke evaluator"
```

---

## Task 7: MLflow Register Script with Dry-Run

**Files:**
- Create: `mlops-platform/mlops/mlflow_register.py`
- Test: `mlops-platform/tests/mlops/test_mlflow_register.py`

- [ ] **Step 1: Write the failing tests** `mlops-platform/tests/mlops/test_mlflow_register.py`

```python
import json
from pathlib import Path

from mlops.mlflow_register import register_smoke_model
from mlops.pipelines.profiles import PipelineProfile


def _profile(tmp_path: Path) -> PipelineProfile:
    artifact_dir = tmp_path / "artifacts/smoke"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "eval_metrics.json").write_text(
        json.dumps({"accuracy": 1.0, "n_examples": 2, "latency_ms_avg": 1.5})
    )
    (artifact_dir / "eval_predictions.jsonl").write_text(
        json.dumps({"id": "q1", "predicted_answer": "A"}) + "\n"
    )
    return PipelineProfile(
        name="smoke",
        data_dir=tmp_path / "smoke_data",
        artifact_dir=artifact_dir,
        kg_dir=artifact_dir / "kg",
        retrieval_fixture_path=artifact_dir / "kg/retrieval_fixture.json",
        predictions_path=artifact_dir / "eval_predictions.jsonl",
        metrics_path=artifact_dir / "eval_metrics.json",
        mlflow_receipt_path=artifact_dir / "mlflow_register_dry_run.json",
        top_k=2,
        model_backend="mock",
        model_version="smoke-dev",
        registered_model_name="medical-qa-smoke",
    )


def test_register_smoke_model_dry_run_writes_receipt(tmp_path):
    profile = _profile(tmp_path)
    receipt = register_smoke_model(profile, dry_run=True)
    assert receipt["dry_run"] is True
    assert receipt["registered_model_name"] == "medical-qa-smoke"
    assert receipt["metrics"]["accuracy"] == 1.0
    assert profile.mlflow_receipt_path.exists()
    saved = json.loads(profile.mlflow_receipt_path.read_text())
    assert saved["model_version"] == "smoke-dev"
```

- [ ] **Step 2: Run to verify failure**

Run:
```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/mlops/test_mlflow_register.py -v
```
Expected: FAIL because `mlops.mlflow_register` does not exist.

- [ ] **Step 3: Implement `mlops-platform/mlops/mlflow_register.py`**

```python
"""Log smoke pipeline outputs to MLflow, with a deterministic dry-run mode."""

import argparse
import json
from pathlib import Path

from .pipelines.profiles import PipelineProfile, load_profile


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def register_smoke_model(profile: PipelineProfile, dry_run: bool = False) -> dict:
    metrics = _read_json(profile.metrics_path)
    receipt = {
        "profile": profile.name,
        "dry_run": dry_run,
        "registered_model_name": profile.registered_model_name,
        "model_version": profile.model_version,
        "metrics": metrics,
        "artifacts": {
            "metrics_path": str(profile.metrics_path),
            "predictions_path": str(profile.predictions_path),
        },
    }

    if dry_run:
        _write_json(profile.mlflow_receipt_path, receipt)
        return receipt

    import mlflow

    mlflow.set_experiment("medical-qa-smoke")
    with mlflow.start_run(run_name=f"{profile.name}-{profile.model_version}") as run:
        mlflow.log_params(
            {
                "profile": profile.name,
                "model_backend": profile.model_backend,
                "model_version": profile.model_version,
                "top_k": profile.top_k,
            }
        )
        for key, value in metrics.items():
            if isinstance(value, int | float):
                mlflow.log_metric(key, float(value))
        mlflow.log_artifact(str(profile.metrics_path))
        mlflow.log_artifact(str(profile.predictions_path))
        receipt["run_id"] = run.info.run_id
        receipt["model_uri"] = f"runs:/{run.info.run_id}/smoke-model"

    _write_json(profile.mlflow_receipt_path, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="smoke")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    register_smoke_model(load_profile(args.profile), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

Run:
```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/mlops/test_mlflow_register.py -v
```
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/vcsai/minhlbq/mlops-platform
git add mlops/mlflow_register.py tests/mlops/test_mlflow_register.py
git commit -m "feat: add MLflow smoke registration script"
```

---

## Task 8: Local Smoke Pipeline End-to-End Test

**Files:**
- Test: `mlops-platform/tests/mlops/test_smoke_pipeline_e2e.py`

- [ ] **Step 1: Write the failing test** `mlops-platform/tests/mlops/test_smoke_pipeline_e2e.py`

```python
import json
from pathlib import Path

from mlops.mlflow_register import register_smoke_model
from mlops.pipelines.build_smoke_kg import build_smoke_kg
from mlops.pipelines.eval_smoke import evaluate_smoke
from mlops.pipelines.profiles import PipelineProfile


def test_smoke_pipeline_e2e(tmp_path):
    data_dir = tmp_path / "smoke_data"
    data_dir.mkdir()
    (data_dir / "medical_mcq.jsonl").write_text(
        json.dumps(
            {
                "id": "q1",
                "question": "Which drug treats type 2 diabetes?",
                "options": {"A": "Metformin", "B": "Amoxicillin"},
                "answer": "A",
                "mock_answer": "A",
                "retrieval_query": "type 2 diabetes first line medication",
            }
        )
        + "\n"
    )
    artifact_dir = tmp_path / "artifacts/smoke"
    profile = PipelineProfile(
        name="smoke",
        data_dir=data_dir,
        artifact_dir=artifact_dir,
        kg_dir=artifact_dir / "kg",
        retrieval_fixture_path=artifact_dir / "kg/retrieval_fixture.json",
        predictions_path=artifact_dir / "eval_predictions.jsonl",
        metrics_path=artifact_dir / "eval_metrics.json",
        mlflow_receipt_path=artifact_dir / "mlflow_register_dry_run.json",
        top_k=2,
        model_backend="mock",
        model_version="smoke-dev",
        registered_model_name="medical-qa-smoke",
    )

    build_smoke_kg(profile)
    metrics = evaluate_smoke(profile)
    receipt = register_smoke_model(profile, dry_run=True)

    assert metrics["accuracy"] == 1.0
    assert receipt["metrics"]["accuracy"] == 1.0
    assert profile.mlflow_receipt_path.exists()
```

- [ ] **Step 2: Run to verify pass**

Run:
```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/mlops/test_smoke_pipeline_e2e.py -v
```
Expected: 1 passed after Tasks 5–7 are implemented.

- [ ] **Step 3: Run the local Make target**

Run:
```bash
cd /home/vcsai/minhlbq/mlops-platform && make smoke-pipeline-local
```
Expected: command exits 0 and creates:
- `artifacts/smoke/kg/manifest.json`
- `artifacts/smoke/eval_metrics.json`
- `artifacts/smoke/eval_predictions.jsonl`
- `artifacts/smoke/mlflow_register_dry_run.json`

- [ ] **Step 4: Commit**

```bash
cd /home/vcsai/minhlbq/mlops-platform
git add tests/mlops/test_smoke_pipeline_e2e.py artifacts/smoke/.gitignore
git commit -m "test: cover smoke pipeline end to end"
```

If `artifacts/smoke/.gitignore` does not exist, create it with:

```gitignore
*
!.gitignore
```

---

## Task 9: README Pipeline Documentation

**Files:**
- Modify: `mlops-platform/README.md`
- Test: `mlops-platform/tests/test_readme_pipeline_docs.py`

- [ ] **Step 1: Write the failing test** `mlops-platform/tests/test_readme_pipeline_docs.py`

```python
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_readme_documents_smoke_pipeline_commands():
    text = (ROOT / "README.md").read_text()
    assert "make install-pipeline" in text
    assert "make smoke-pipeline-local" in text
    assert "make smoke-pipeline" in text
    assert "make mlflow-register-dry-run" in text
    assert "DVC" in text
    assert "MLflow" in text
```

- [ ] **Step 2: Run to verify failure**

Run:
```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/test_readme_pipeline_docs.py -v
```
Expected: FAIL because README does not document pipeline commands yet.

- [ ] **Step 3: Update `mlops-platform/README.md`**

Append:

````markdown
## Smoke MLOps Pipeline

Plan 2 adds a self-contained smoke pipeline. It does not depend on a sibling
`baseline/` checkout.

Install pipeline dependencies:

```bash
make install-pipeline
```

Run the pipeline locally without DVC orchestration:

```bash
make smoke-pipeline-local
```

Run the same stages through DVC:

```bash
make smoke-pipeline
```

Dry-run MLflow registration:

```bash
make mlflow-register-dry-run
```

The smoke pipeline writes artifacts under `artifacts/smoke/`. DVC tracks stage
lineage through `dvc.yaml`; MLflow receives metrics and artifacts through
`mlops/mlflow_register.py`.
````

- [ ] **Step 4: Run to verify pass**

Run:
```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/test_readme_pipeline_docs.py -v
```
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/vcsai/minhlbq/mlops-platform
git add README.md tests/test_readme_pipeline_docs.py
git commit -m "docs: document smoke MLOps pipeline"
```

---

## Task 10: DVC and Full Verification

**Files:**
- Modify: `mlops-platform/.gitignore`

- [ ] **Step 1: Update `.gitignore`**

Append:

```gitignore
/artifacts/
/.dvc/cache/
```

Do not ignore `dvc.yaml`, `params.yaml`, or `*.dvc` files.

- [ ] **Step 2: Install pipeline dependencies**

Run:
```bash
cd /home/vcsai/minhlbq/mlops-platform && make install-pipeline
```
Expected: installs package with `[dev,pipeline]`.

- [ ] **Step 3: Run all tests**

Run:
```bash
cd /home/vcsai/minhlbq/mlops-platform && make test
```
Expected: all tests pass and coverage remains above 80%.

- [ ] **Step 4: Run local smoke pipeline**

Run:
```bash
cd /home/vcsai/minhlbq/mlops-platform && make smoke-pipeline-local
```
Expected: exits 0 and writes smoke artifacts under `artifacts/smoke/`.

- [ ] **Step 5: Run DVC smoke pipeline**

Run:
```bash
cd /home/vcsai/minhlbq/mlops-platform && make smoke-pipeline
```
Expected: DVC reproduces `build_kg_smoke`, `eval_smoke`, and `register_smoke_model` or reports all stages are unchanged.

- [ ] **Step 6: Inspect DVC status**

Run:
```bash
cd /home/vcsai/minhlbq/mlops-platform && make dvc-status
```
Expected: no changed DVC stages immediately after `make smoke-pipeline`.

- [ ] **Step 7: Commit**

```bash
cd /home/vcsai/minhlbq/mlops-platform
git add .gitignore dvc.yaml params.yaml mlops tests README.md Makefile pyproject.toml
git add dvc.lock || true
git commit -m "feat: add DVC and MLflow smoke pipeline"
```

---

## Self-Review Notes

- **Spec coverage:** This plan covers `MLOPS_IMPLEMENTATION_PLAN.md` section 5.5 and the P0 DoD items "MLflow tracking + ≥1 registered model; `make smoke-pipeline` logs to MLflow" and "DVC tracks smoke artifacts + configured GCS remote" for the local smoke case. GCS remote configuration is deferred until cloud/deploy credentials exist; Plan 3 or Plan 4 should add `dvc remote add -d gcsremote gs://...`.
- **Repository boundary:** No task imports from `baseline/`; the smoke pipeline uses package-owned data and code.
- **Type consistency:** `PipelineProfile` path fields are `pathlib.Path` across generator/evaluator/register scripts.
- **Testing:** All behavior is testable without DVC or MLflow services except the final verification task, which installs DVC/MLflow and runs `dvc repro`. MLflow has a dry-run path for CI/local tests.
- **Artifacts:** Generated `artifacts/` are ignored by git. DVC metadata files stay tracked.
