# Wire vLLM backend into CI/CD workflows — Design

## Goal
Make the GKE deploy workflows deploy the **real self-hosted vLLM backend** (DGX-Spark via Cloudflare Tunnel) instead of the mock backend, with mock retained as an opt-in escape hatch. After this, `git push` → CI → Auto Deploy lands the real model (no more silent revert-to-mock).

## Context
The previous feature (`feat/dgx-vllm-cloudflare`, merged) already gave the scripts a vLLM interface:
- `scripts/cloud/deploy.sh` flips to vllm when `MODEL_BACKEND=vllm` + `LLM_BASE_URL` + `LLM_MODEL` are set (defaults to `mock`).
- `scripts/cloud/create_secrets.sh` creates secret `medical-qa-llm-key` when `LLM_API_KEY` is set (skips otherwise).
- `deploy/helm/api` reads `LLM_API_KEY` from that secret (`optional: true`).
- `scripts/cloud/smoke_cloud.sh` is already backend-agnostic (asserts the `"answer"` key + `/version` contract; no specific letter).

So this work is **pure GitHub Actions YAML wiring** + tests + docs. No script logic changes.

## Decisions (locked)
- **Toggle, default vllm.** `demo-up.yml` gets a `workflow_dispatch` choice input `backend` (`vllm|mock`, default `vllm`). `deploy.yml` (auto, no inputs) defaults via repo Var: `MODEL_BACKEND: ${{ vars.MODEL_BACKEND || 'vllm' }}`. Mock remains reachable for plumbing/CI demos when the DGX is down.
- **Both workflows wired** (`demo-up.yml` + `deploy.yml`) for consistency — resolves the "next push reverts api to mock" gotcha.
- **Config via GitHub-native sources:** Secret `LLM_API_KEY`; Vars `LLM_BASE_URL`, `LLM_MODEL` (optional Var `MODEL_BACKEND`).

## Changes

### `.github/workflows/demo-up.yml`
- Add `workflow_dispatch` input `backend`: `type: choice`, `options: [vllm, mock]`, `default: vllm`.
- Add to job `env`:
  - `MODEL_BACKEND: ${{ inputs.backend }}`
  - `LLM_API_KEY: ${{ secrets.LLM_API_KEY }}`
  - `LLM_BASE_URL: ${{ vars.LLM_BASE_URL }}`
  - `LLM_MODEL: ${{ vars.LLM_MODEL }}`
- Existing steps unchanged in shape: `create_secrets.sh` now also creates the LLM secret (because `LLM_API_KEY` is in env); `deploy.sh` now flips to vllm.

### `.github/workflows/deploy.yml`
- Add to job `env`:
  - `MODEL_BACKEND: ${{ vars.MODEL_BACKEND || 'vllm' }}`
  - `LLM_API_KEY: ${{ secrets.LLM_API_KEY }}`
  - `LLM_BASE_URL: ${{ vars.LLM_BASE_URL }}`
  - `LLM_MODEL: ${{ vars.LLM_MODEL }}`
- **Add a `create_secrets.sh` step** (gated on `steps.cluster.outputs.up == 'true'`, before the deploy step). deploy.yml currently has none; the auto-deploy path must ensure `medical-qa-llm-key` exists/ is fresh or a vllm flip yields 401 (api sends no auth header). `create_secrets.sh` is idempotent (`kubectl apply`).

### `scripts/cloud/smoke_cloud.sh`
- Update the stale header comment (line 3) — it claims the mock backend returns a deterministic letter. Logic is unchanged (already backend-agnostic).

### `tests/deploy/test_cloud_workflows.py`
- Extend `test_demo_up_*`: assert the `backend` choice input (default `vllm`) and the `MODEL_BACKEND`/`LLM_API_KEY`/`LLM_BASE_URL`/`LLM_MODEL` env wiring.
- Extend `test_deploy_auto_*`: assert `MODEL_BACKEND` default-`vllm` expression, the `LLM_*` env wiring, and that a `create_secrets.sh` step is present.

### Docs
- `docs/cloud-setup.md` and `docs/runbooks/dgx-vllm-cloudflare.md`: document the required repo **Secret** `LLM_API_KEY` and **Vars** `LLM_BASE_URL`/`LLM_MODEL` (+ optional `MODEL_BACKEND`), the `backend` dispatch input, and that auto-deploy now lands vllm.

## Behavior when DGX is down
`backend=vllm` + DGX/tunnel down → api pod never becomes Ready (`/ready` → `health_check()` fails) → smoke `rollout status` times out → workflow fails loudly (correct). Run `demo-up` with `backend=mock` to demo plumbing without the DGX.

## Operator prerequisites (manual, outside this repo)
In GitHub repo settings: add Secret `LLM_API_KEY`; add Vars `LLM_BASE_URL` (e.g. `https://llm.<domain>/v1`), `LLM_MODEL` (e.g. `medical-qa-llama-gdpo`).

## Out of scope
- No changes to `ci.yml` (build/test unchanged).
- No script logic changes (interface already exists).
- Not adding the GitHub Vars/Secret (operator does this in the UI).

## Testing
`.venv/bin/python -m pytest tests/deploy/test_cloud_workflows.py` (PyYAML parse + assertions); full suite stays green. Workflows are validated by structure/text assertions, not live cloud calls.
