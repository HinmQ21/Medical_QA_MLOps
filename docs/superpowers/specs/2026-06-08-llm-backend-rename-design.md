# Design: Rename the serving `VllmBackend` → generic `LLMBackend`

**Date:** 2026-06-08
**Status:** Approved (brainstorming) — pending spec review before planning

## Problem

The FastAPI serving path selects a model backend via `MODEL_BACKEND`. One option is named `vllm` (`VllmBackend`, `name = "vllm"`), but the class is actually a **generic OpenAI-compatible chat-completions client** — it only POSTs to `{LLM_BASE_URL}/v1/chat/completions`, with nothing vLLM-specific. Since the KServe path now serves Qwen2.5-1.5B via **llama.cpp** (also OpenAI `/v1`), the same backend is reused against a non-vLLM server. The `vllm` label therefore misleads: operators set `MODEL_BACKEND=vllm` to talk to a llama.cpp server.

## Goal

Rename the serving backend to reflect what it is — a generic LLM/OpenAI client — without breaking any existing deployment that already uses `MODEL_BACKEND=vllm`.

## Scope boundary (critical)

"vllm" has **two unrelated meanings** in this repo. Only the first is in scope.

**IN scope — the serving ModelBackend** (`src/medical_qa_platform/inference/`):
- `VllmBackend` class + `vllm_backend.py` file
- the `MODEL_BACKEND=vllm` selector value and its defaults across deploy.sh, the GitHub workflows, and docs
- tests pinning the above

**OUT of scope — the genuine vLLM inference engine** used by the *baseline RL training/eval* pipeline and the DGX server. Do **not** rename any of these:
- `mlops/pipelines/full_stages.py`, `mlops/pipelines/full_config.py`, `params.yaml` — `vllm_venv`, `--use-vllm`, `--vllm-gpu-mem[-util]`, `grpo_train_vllm`
- `docs/runbooks/dgx-vllm-cloudflare.md` — describes a real vLLM server; only its single `export MODEL_BACKEND=vllm` selector line changes
- historical records `docs/plans/*`, `docs/specs/2026-06-03-full-mlflow-pipeline-design.md`

**No env-var changes:** `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY` are already generic and stay as-is.

## Design

### 1. The backend rename

- Move `src/medical_qa_platform/inference/vllm_backend.py` → `llm_backend.py`.
- Rename class `VllmBackend` → `LLMBackend`; `from_env` return annotation `"VllmBackend"` → `"LLMBackend"`.
- Change class attribute `name = "vllm"` → `name = "llm"`.
- De-vendor the module docstring: "Generic OpenAI-compatible chat-completions backend. Talks to any server exposing the OpenAI `/v1` API — a self-hosted vLLM server on the DGX-Spark, an in-cluster llama.cpp server, etc. Configured via `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`."
- No behavior change: `generate()` and `health_check()` are byte-for-byte identical (still POST `/chat/completions`, GET `/models`).

### 2. Factory with back-compat alias

In `src/medical_qa_platform/inference/__init__.py`, the branch becomes:

```python
    if name in ("llm", "vllm"):  # "vllm" is a back-compat alias for the generic LLM/OpenAI backend
        from .llm_backend import LLMBackend

        return LLMBackend.from_env()
```

- `llm` is canonical; `vllm` is accepted but maps to the same class.
- Because `LLMBackend.name == "llm"`, the `/predict` response `backend` field reports `llm` even when selected via `vllm` (the alias normalizes to canonical).
- **No runtime deprecation warning** for the alias — a code comment only (YAGNI for a demo/thesis).

### 3. Flip user-facing defaults + docs to `llm`

The alias guarantees nothing breaks; we still update the canonical surface so docs and configs are honest.

- `scripts/cloud/deploy.sh`: the "needs an LLM endpoint" check recognizes both — `if [ "$BACKEND" = "llm" ] || [ "$BACKEND" = "vllm" ]; then`. It continues to pass `--set env.modelBackend="$BACKEND"` (whatever the operator gave; the api factory's alias resolves it). Update the nearby comment from `MODEL_BACKEND=vllm` to `MODEL_BACKEND=llm`.
- `.github/workflows/deploy.yml`: `MODEL_BACKEND: ${{ vars.MODEL_BACKEND || 'llm' }}`.
- `.github/workflows/demo-up.yml`: the `backend` input dropdown options become `[llm, mock]`, default `llm`, with the description updated (e.g. "llm = real model via LLM_BASE_URL (DGX vLLM or in-cluster llama.cpp)"). The `vllm` value is still honored if set via repo var / env, but is no longer offered in the dropdown.
- `README.md`: change `MODEL_BACKEND=vllm` → `MODEL_BACKEND=llm` and "through the `vllm` backend" → "through the `llm` backend"; keep the parenthetical noting the server may be DGX vLLM or in-cluster llama.cpp.
- `docs/cloud-setup.md`: change the `vllm` selector references to `llm` (default `llm`, dropdown `llm`|`mock`), keeping the "real 3B model" prose.
- `docs/runbooks/dgx-vllm-cloudflare.md`: change ONLY the selector line `export MODEL_BACKEND=vllm` → `export MODEL_BACKEND=llm` (note `vllm` still works). Everything describing the vLLM server itself stays.

### 4. Tests

- Rename `tests/inference/test_vllm_backend.py` → `test_llm_backend.py`: import `LLMBackend` from `llm_backend`; assert `name == "llm"`; constructor/`from_env`/`generate`/`health_check` cases otherwise unchanged.
- `tests/inference/test_mock_backend.py`: rename `test_factory_returns_vllm_by_name` → `test_factory_returns_llm_by_name` (asserts `get_backend("llm")` is `LLMBackend`); add `test_factory_vllm_alias_maps_to_llm` (asserts `get_backend("vllm")` is also `LLMBackend`). Keep `test_factory_no_longer_knows_runpod` / `..._kserve`.
- `tests/cloud/test_deploy_sh.py`: add `test_dry_run_flips_to_llm_backend_when_requested` (`MODEL_BACKEND=llm` → `env.modelBackend=llm`); keep an alias case proving `MODEL_BACKEND=vllm` still flips (→ `env.modelBackend=vllm`); the `requires_base_url_and_model` guard must hold for both `llm` and `vllm`.
- `tests/deploy/test_cloud_workflows.py`: `test_demo_up_wires_..._backend_toggle` → default `llm`, options include `llm`; `test_deploy_auto_wires_..._and_ensures_llm_secret` → assert `|| 'llm'`.
- `tests/test_config.py`: `MODEL_BACKEND=llm` → `settings.model_backend == "llm"`.
- `tests/app/test_ui_client.py`: fixture `backend` values `"vllm"` → `"llm"` (the API now reports the canonical name).

### 5. Verification

- Full suite green: `.venv/bin/pytest --cov=medical_qa_platform --cov-fail-under=80`.
- `make helm-lint && make helm-template` clean (api chart still renders).
- `bash -n scripts/cloud/deploy.sh`; dry-run with `MODEL_BACKEND=llm` and with `MODEL_BACKEND=vllm` both flip to the LLM path.
- `grep -rnI "VllmBackend\|vllm_backend" src/ tests/` → no matches (class/file fully renamed). The only legitimate remaining `vllm` references in tracked, in-scope files are: the factory alias branch + its comment, and the deploy.sh alias condition (`= "vllm"`). Everything else matching `vllm` must be an OUT-of-scope occurrence — the training-pipeline engine (`mlops/`, `params.yaml`), the DGX runbook's vLLM-server prose, or historical `docs/plans|specs`.

## Out of scope / non-goals

- Renaming the genuine vLLM training-engine references (§ Scope boundary).
- Any change to backend *behavior*, request/response shape, or env-var names.
- A runtime deprecation warning for the `vllm` alias.
- Removing the `vllm` alias entirely (kept indefinitely as cheap back-compat).
