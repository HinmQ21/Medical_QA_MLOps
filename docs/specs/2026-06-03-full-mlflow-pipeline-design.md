# Full MLflow/DVC Training Pipeline (SP3) — Design

**Sub-project 3 of the train↔serve parity / MLOps effort.** It wraps the
existing `baseline/` training lifecycle (KG build → Stage 1 → Stage 1.5 →
Stage 2 → eval → register) in a reproducible, MLflow-tracked, DVC-lineaged
pipeline owned by `mlops-platform/`, **without importing** the `baseline/`
package — orchestration is process-level (subprocess).

## Goal

Give the project a single, parameterised, runnable pipeline that drives the
full real training lifecycle on the local DGX-Spark GB10, logging every step to
MLflow and tracking lineage in DVC — built and verified without running the
multi-day training (dry-run + structural tests in CI; one tiny `smoke_full` real
run proves the wiring). The full multi-day run is operator-triggered.

## Context

### Where training runs (execution topology)

All training — both the tiny `smoke_full` proof and the full run — executes
**locally on the GB10**, exactly where it runs today. SP3 does not move training
anywhere; it adds an orchestration + tracking layer around the existing local
training. The orchestrator (`mlops/`) is lightweight Python that builds commands
and `subprocess.run`s the `baseline/` training scripts in their venvs
(`training_venv312`, `vllm_venv312`); the GPU work happens on the GB10. This is
distinct from **serving** (Plan 4: GKE CPU + RunPod GPU) — SP3 is the
**training** world and never touches GKE/RunPod.

```
[GB10, local]
make full-pipeline  (or: dvc repro)
  -> .venv/bin/python -m mlops.pipelines.run_full --profile full   (orchestrator; light)
       per stage: subprocess.run(argv, cwd=baseline_root, env=...)
       build_kg  -> training_venv312 -m scripts.build_kg.run_pipeline ...
       stage1    -> training_venv312 -m scripts.finetune.medreason... ]
       stage1_5  -> training_venv312 -m scripts.stage1_5.sft_train (+ merge)  } GPU on GB10
       stage2    -> vllm_venv312 -m scripts.train_rl.grpo_train_vllm (+ merge)]
       eval      -> vllm_venv312 -m scripts.benchmark...grpo_eval ...
       register  -> log + register into mlruns/ (local file store)
```

### Existing patterns SP3 extends

- `mlops/pipelines/profiles.py` (typed profile from `params.yaml`),
  `mlops/mlflow_register.py` (MLflow with a deterministic **dry-run** mode that
  writes a receipt), the smoke `dvc.yaml`, `Makefile` targets, and structural
  tests in `tests/mlops/`. SP3 mirrors this philosophy: everything is testable
  via dry-run / structure without heavy execution.
- `mlops/` is **outside** `src/medical_qa_platform/`, so the SP1 self-contained
  guard (which scans only `src/`) does not apply — `mlops/` may reference
  `baseline/` paths. It still must not *import* baseline Python; it shells out.
- `mlops/` may import the installed `medical_qa_platform` package (the smoke
  pipeline already does) — used to read `RETRIEVAL_CONTRACT_VERSION`.

### Decisions (locked during brainstorming)

1. **Orchestration:** a reusable Python **stage-runner** builds the exact
   baseline command, `subprocess.run`s it (or records it in `--dry-run`), parses
   outputs, and logs an MLflow nested run. DVC stages call the runner for
   lineage.
2. **Execution scope:** deliver a **runnable** pipeline; verify in CI via
   dry-run + structural tests; prove real wiring with one tiny **`smoke_full`**
   run (`--max-steps` small, few samples) on the GB10. The full multi-day run is
   operator-triggered. SP3 does **not** run full training.
3. **MLflow:** local **file store** (`mlruns/`); one **parent run** per pipeline
   execution with **nested runs per stage**; artifacts logged as **pointers**
   (checkpoint paths/URIs + metadata), never multi-GB weights. Tracking-server
   deploy is Plan 5.
4. **Encoder-agnostic build_kg:** the `build_kg` stage is parameterised by the
   encoder, so SP2's encoder swap is a config change, not a code change here.
5. **Defaults:** the `full` profile defaults to `model_family: qwen` and Stage 2
   `variant: grpo` with vLLM rollout (both are config and trivially switchable to
   `llama`/`gdpo`).
6. **Artifact ownership — everything mlops produces lives under
   `mlops-platform/`:** an `artifact_root` (`artifacts/full/` for `full`,
   `artifacts/smoke_full/` for `smoke_full`) holds the pipeline-built KG
   (`artifact_root/kg`), all stage checkpoints (`artifact_root/outputs/<stage>`,
   passed as each baseline script's `--output-dir`/`--data-dir`), eval JSON,
   per-stage logs, and receipts. MLflow uses `mlops-platform/mlruns/`. **Inputs**
   that inherently belong to the reference tree — base model weights
   (`baseline/models/...`), raw benchmark datasets (`baseline/dataset/...`), and
   the teacher SFT traces (`baseline/data/stage1_5_sft_v2.jsonl`) — are **read**
   from `baseline/` (they are inputs, not outputs). `artifacts/` and `mlruns/`
   are generated and git-ignored.

## Architecture & Components

| File | Responsibility |
|------|----------------|
| `params.yaml` (+`full`, +`smoke_full`) | Full-pipeline config: `baseline_root`, `train_venv`/`vllm_venv` paths, `artifact_root` (under `mlops-platform/`), `mlruns_dir`, `model_family` (default `qwen`), `kg_version`; per-stage section (build_kg: encoder/kg-out-dir; stage1: base-model-path/out; stage1_5: sft-data-path/out; stage2: variant default `grpo`/use_vllm/num_generations/out; eval: benchmarks list/use_vllm/out; register: registered_model_name). Produced paths (KG, checkpoints, eval JSON) resolve under `artifact_root`; base models / datasets / SFT traces are read from `baseline/`. `smoke_full` = same shape, tiny (`max_steps`, `max_eval_samples`, `artifacts/smoke_full/`). |
| `mlops/pipelines/full_config.py` | `FullPipelineConfig` dataclass + `load_full_config(name, params_path)`. Separate from `PipelineProfile` (smoke) — different shape. |
| `mlops/pipelines/stage_runner.py` | `StageSpec(name, commands, cwd, env, params, tags, artifact_pointers, metrics_parser)` where `commands` is a list of argv lists (most stages = 1; `stage1_5`/`stage2` = train + merge). `run_stage(spec, dry_run, mlflow_parent)`: dry-run → return/record a receipt (the resolved argv + intended MLflow payload), no subprocess/mlflow; real → run each command (tee stdout to a per-stage log), parse metrics, log a nested MLflow run. |
| `mlops/pipelines/full/stages.py` | Six builders returning `StageSpec`, matching the canonical commands in CLAUDE.md, branching on `model_family`: `build_kg`, `stage1`, `stage1_5` (Llama: `convert_data_llama` → `sft_train` → `merge`; Qwen: `sft_train` → `merge`), `stage2` (`grpo_train_vllm` [+`--use-gdpo`] → `merge`), `eval` (`grpo_eval`), `register`. |
| `mlops/pipelines/run_full.py` | Orchestrator entrypoint. Opens an MLflow parent run, iterates the selected stages, calls `run_stage`, writes a pipeline receipt JSON. CLI: `--profile full\|smoke_full`, `--dry-run`, `--stages a,b,c` (subset; default all). Reads `RETRIEVAL_CONTRACT_VERSION` from the installed package and sets it as a parent-run tag. |
| `dvc.yaml` (+ full stages) | Lineage DAG `build_kg → stage1 → stage1_5 → stage2 → eval → register`; each `cmd` calls `run_full --profile full --stages <stage>`. Tracked outs = the small per-stage receipt/metrics JSON; checkpoint dirs are `cache: false` outs (lineage without caching GBs). Smoke stages stay unchanged. |
| `Makefile` (+ targets) | `full-pipeline-dry-run` (`run_full --profile full --dry-run`), `full-pipeline` (real; operator), `smoke-full` (`run_full --profile smoke_full`). |

### MLflow logging (full per step)

- Experiment `medical-qa-full`; **parent run** per `run_full` execution;
  **nested run per stage**.
- Parent-run tags: `model_family`, `kg_version`, `retrieval_contract_version`
  (from `medical_qa_platform.retrieval.contract`), `profile`.
- Each stage nested run logs: **params** (its hyperparameters), **tags**
  (`stage`, plus inherited family/version), **metrics** (eval: accuracy per
  benchmark parsed from the eval JSON; training stages: best-effort final loss
  from HF `trainer_state.json` if present), **artifacts = pointers** (checkpoint
  dir path/URI, eval JSON path, the per-stage receipt) — never the weights.
- `register` stage registers the final Stage 2 merged model into the local
  MLflow Model Registry with the contract/kg version tags.

## Data Flow

```
run_full(profile)
  -> open MLflow parent run (tags: family, kg_version, contract_version, profile)
  -> for stage in selected:
        spec = builders[stage](cfg)
        run_stage(spec, dry_run):
          dry_run -> receipt {resolved argv, mlflow payload}        (CI)
          real    -> subprocess.run each command (tee log)
                     -> parse metrics -> nested MLflow run          (GB10)
  -> register final model -> pipeline receipt JSON
```

## Testing Strategy / Completion Gates (no GPU, no MLflow server)

- **`test_full_config.py`** — load `full` and `smoke_full` configs; required keys
  present; types correct.
- **`test_stage_builders.py`** — each builder yields the expected argv for both
  `qwen` and `llama` (correct venv, module path, key flags; `--use-gdpo` only for
  the gdpo variant; Llama `stage1_5` includes the convert step; `stage1_5`/
  `stage2` include the merge command).
- **`test_run_full_dry_run.py`** — `run_full --profile smoke_full --dry-run`
  returns a receipt with all six stages in order and a well-formed MLflow payload
  (params/tags/metric-keys/artifact-pointers); asserts **no** subprocess or real
  MLflow call happened.
- **`test_dvc_full_stages.py`** — `dvc.yaml` parses; the six full stages exist
  with correct `cmd`/`deps`/`outs` shape and `cache: false` on checkpoint outs;
  smoke stages unchanged.
- **`test_makefile_full_targets.py`** — `full-pipeline`, `full-pipeline-dry-run`,
  `smoke-full` targets exist.
- **Gates:** `make test` green, coverage ≥ 80% (the new pure modules —
  `full_config`, `stage_runner` dry-run path, builders — are covered;
  subprocess/real-MLflow paths are `# pragma: no cover`). Existing smoke pipeline
  + all prior tests stay green.
- **`smoke_full` real run (manual/opt-in, controller-run once):** execute
  `run_full --profile smoke_full` on the GB10 with tiny caps to prove the
  baseline scripts + MLflow logging actually run end-to-end. Not in CI. Record
  the result in the PR.

## Out of Scope (later)

- The full multi-day training run (operator-triggered).
- MLflow **tracking-server** deploy, observability (Prometheus/Grafana/Loki/
  Tempo), Evidently drift — Plan 5.
- The encoder swap + KG re-index + re-eval — **SP2** (the `build_kg` stage is
  encoder-parameterised, so SP2 only changes `params.yaml`).
- Serving changes — SP4 / Plan 4.

## Risks / Open Questions

- **Baseline CLI drift:** builders hard-code the baseline scripts' flags; if
  those scripts change, the builders break. Mitigation: builders centralise the
  commands in one module; the `smoke_full` run catches drift before a full run.
- **DVC caching huge checkpoints:** checkpoint dirs are `cache: false` outs so
  DVC tracks lineage without copying GBs into its cache.
- **Training-stage metrics:** HF trainer loss is not always in a stable file;
  metric parsing for training stages is best-effort, and eval accuracy is the
  authoritative metric. State clearly what is logged.
- **Long-running subprocess:** the full run takes ~days; the operator runs it
  under `tmux`/`nohup`. The orchestrator `subprocess.run`s each stage
  sequentially and tees logs; a failed stage aborts the pipeline with the failing
  stage + log path in the receipt.
- **`model_family` branching:** Qwen vs Llama differ (data, `--model-family`,
  convert step). Builders branch on `model_family`; tests cover both.
- **kg_version provenance:** taken from config / the KG build manifest; recorded
  as a parent-run tag so every model version is traceable to a KG version.
