# Wire vLLM backend into CI/CD workflows — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `demo-up.yml` and `deploy.yml` deploy the real self-hosted vLLM backend (default), with mock as an opt-in fallback, so `git push` → CI → Auto Deploy lands the real model.

**Architecture:** Pure GitHub Actions YAML wiring. The deploy scripts already expose the interface (`MODEL_BACKEND=vllm` + `LLM_BASE_URL`/`LLM_MODEL` → `deploy.sh` flips; `LLM_API_KEY` → `create_secrets.sh` makes secret `medical-qa-llm-key`; helm reads it `optional`). We surface these via a `workflow_dispatch` choice input (demo-up) and a repo-Var-defaulted env (deploy), add a `create_secrets.sh` step to the auto-deploy path, refresh a stale smoke comment, extend the workflow tests, and document the GitHub Vars/Secret.

**Tech Stack:** GitHub Actions YAML, bash, pytest + PyYAML.

**Working directory:** `/home/vcsai/minhlbq/mlops-platform/.worktrees/vllm-workflow-wiring` (branch `feat/vllm-workflow-wiring`; all paths relative; commands run from here).

**Test runner:** `.venv/bin/python -m pytest`.

**Spec:** `docs/superpowers/specs/2026-06-06-vllm-workflow-wiring-design.md`.

---

## File Structure

- `tests/deploy/test_cloud_workflows.py` — MODIFY: assert vllm wiring in both workflows (tests first).
- `.github/workflows/demo-up.yml` — MODIFY: `backend` choice input + LLM env.
- `.github/workflows/deploy.yml` — MODIFY: default-vllm env + LLM env + a `create_secrets.sh` step.
- `scripts/cloud/smoke_cloud.sh` — MODIFY: refresh stale header comment (no logic change).
- `docs/cloud-setup.md` — MODIFY: document repo Vars/Secret + `backend` input + auto-deploy-vllm.
- `docs/runbooks/dgx-vllm-cloudflare.md` — MODIFY: add the workflow-driven path + GitHub config prerequisites.

---

## Task 1: Test the demo-up.yml vLLM wiring (TDD red)

**Files:**
- Modify test: `tests/deploy/test_cloud_workflows.py`

- [ ] **Step 1: Write the failing test**

In `tests/deploy/test_cloud_workflows.py`, append:

```python
def test_demo_up_wires_vllm_backend_toggle():
    text = (WF / "demo-up.yml").read_text()
    wf = _load("demo-up.yml")
    inputs = _triggers(wf)["workflow_dispatch"]["inputs"]
    assert inputs["backend"]["type"] == "choice"
    assert inputs["backend"]["default"] == "vllm"
    assert "vllm" in inputs["backend"]["options"]
    assert "mock" in inputs["backend"]["options"]
    assert "MODEL_BACKEND: ${{ inputs.backend }}" in text
    assert "LLM_API_KEY: ${{ secrets.LLM_API_KEY }}" in text
    assert "LLM_BASE_URL: ${{ vars.LLM_BASE_URL }}" in text
    assert "LLM_MODEL: ${{ vars.LLM_MODEL }}" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/deploy/test_cloud_workflows.py::test_demo_up_wires_vllm_backend_toggle -v`
Expected: FAIL — `KeyError: 'backend'` (input not defined yet).

- [ ] **Step 3: Commit the failing test**

```bash
git add tests/deploy/test_cloud_workflows.py
git commit -m "test(workflows): demo-up vllm backend toggle wiring (red)"
```

---

## Task 2: Wire demo-up.yml

**Files:**
- Modify: `.github/workflows/demo-up.yml`

- [ ] **Step 1: Add the `backend` choice input**

In `.github/workflows/demo-up.yml`, under `on.workflow_dispatch.inputs`, after the existing `namespace` input block, add:

```yaml
      backend:
        description: Model backend (vllm = real DGX model, mock = deterministic)
        required: true
        type: choice
        options:
          - vllm
          - mock
        default: vllm
```

- [ ] **Step 2: Add LLM env to the job**

In the same file, in the `demo-up` job's `env:` block (which currently has `GCP_PROJECT`, `IMAGE_TAG`, `K8S_NAMESPACE`, `NGINX_API_KEY`), add these four lines:

```yaml
      MODEL_BACKEND: ${{ inputs.backend }}
      LLM_API_KEY: ${{ secrets.LLM_API_KEY }}
      LLM_BASE_URL: ${{ vars.LLM_BASE_URL }}
      LLM_MODEL: ${{ vars.LLM_MODEL }}
```

- [ ] **Step 3: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/deploy/test_cloud_workflows.py -v`
Expected: ALL PASS (including the existing `test_demo_up_is_dispatch_and_uses_oidc_and_scripts`).

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/demo-up.yml
git commit -m "feat(ci): demo-up deploys vllm backend by default (toggle to mock)"
```

---

## Task 3: Test the deploy.yml vLLM wiring (TDD red)

**Files:**
- Modify test: `tests/deploy/test_cloud_workflows.py`

- [ ] **Step 1: Write the failing test**

In `tests/deploy/test_cloud_workflows.py`, append:

```python
def test_deploy_auto_wires_vllm_and_ensures_llm_secret():
    text = (WF / "deploy.yml").read_text()
    assert "MODEL_BACKEND: ${{ vars.MODEL_BACKEND || 'vllm' }}" in text
    assert "LLM_API_KEY: ${{ secrets.LLM_API_KEY }}" in text
    assert "LLM_BASE_URL: ${{ vars.LLM_BASE_URL }}" in text
    assert "LLM_MODEL: ${{ vars.LLM_MODEL }}" in text
    # auto-deploy must ensure the LLM secret exists before flipping to vllm
    assert "scripts/cloud/create_secrets.sh" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/deploy/test_cloud_workflows.py::test_deploy_auto_wires_vllm_and_ensures_llm_secret -v`
Expected: FAIL — the `MODEL_BACKEND` expression and `create_secrets.sh` are not in deploy.yml yet.

- [ ] **Step 3: Commit the failing test**

```bash
git add tests/deploy/test_cloud_workflows.py
git commit -m "test(workflows): deploy.yml vllm wiring + llm secret (red)"
```

---

## Task 4: Wire deploy.yml

**Files:**
- Modify: `.github/workflows/deploy.yml`

- [ ] **Step 1: Add LLM env to the job**

In `.github/workflows/deploy.yml`, in the `deploy` job's `env:` block (currently `GCP_PROJECT`, `IMAGE_TAG`, `K8S_NAMESPACE`, `NGINX_API_KEY`), add:

```yaml
      MODEL_BACKEND: ${{ vars.MODEL_BACKEND || 'vllm' }}
      LLM_API_KEY: ${{ secrets.LLM_API_KEY }}
      LLM_BASE_URL: ${{ vars.LLM_BASE_URL }}
      LLM_MODEL: ${{ vars.LLM_MODEL }}
```

- [ ] **Step 2: Add a `create_secrets.sh` step before the rolling deploy**

In the same file, insert this step immediately BEFORE the existing `- name: Rolling deploy` step (so it shares the same `if` gate and runs first):

```yaml
      - name: Ensure cluster secrets
        if: steps.cluster.outputs.up == 'true'
        run: bash scripts/cloud/create_secrets.sh
```

- [ ] **Step 3: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/deploy/test_cloud_workflows.py -v`
Expected: ALL PASS (including the existing `test_deploy_auto_uses_oidc_and_is_gated_on_ci_success`).

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/deploy.yml
git commit -m "feat(ci): auto-deploy lands vllm backend + ensures llm secret"
```

---

## Task 5: Refresh the stale smoke comment

**Files:**
- Modify: `scripts/cloud/smoke_cloud.sh`

- [ ] **Step 1: Update the header comment**

In `scripts/cloud/smoke_cloud.sh`, replace line 3:

```bash
# api serves the mock backend, so /predict returns a deterministic answer letter.
```
with:
```bash
# Backend-agnostic: asserts /predict returns an "answer" field + the /version
# contract, so it passes for both the mock and the real vllm backend.
```

- [ ] **Step 2: Verify the script still parses**

Run: `bash -n scripts/cloud/smoke_cloud.sh && echo OK`
Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
git add scripts/cloud/smoke_cloud.sh
git commit -m "docs(smoke): clarify smoke test is backend-agnostic"
```

---

## Task 6: Document the GitHub config + workflow-driven vllm

**Files:**
- Modify: `docs/cloud-setup.md`
- Modify: `docs/runbooks/dgx-vllm-cloudflare.md`

- [ ] **Step 1: Add a section to docs/cloud-setup.md**

In `docs/cloud-setup.md`, find the line that begins `- **Real 3B model (vLLM):**` (added by the prior feature) and replace that whole bullet with:

```markdown
- **Real 3B model (vLLM):** the deploy workflows default to the `vllm` backend.
  Set in GitHub repo settings: **Secret** `LLM_API_KEY` (the DGX vLLM api-key);
  **Vars** `LLM_BASE_URL` (e.g. `https://llm.<domain>/v1` — must end in `/v1`) and
  `LLM_MODEL` (e.g. `medical-qa-llama-gdpo`); optional Var `MODEL_BACKEND` to
  override the default. `Demo Up (GKE)` has a `backend` input (`vllm`|`mock`,
  default `vllm`); `Auto Deploy` uses `vllm` unless `MODEL_BACKEND` is set. Stand
  up the DGX server first — see [`runbooks/dgx-vllm-cloudflare.md`](runbooks/dgx-vllm-cloudflare.md).
  Run `Demo Up` with `backend: mock` to demo the plumbing when the DGX is offline.
```

- [ ] **Step 2: Add a "Driven by the GitHub workflows" subsection to the runbook**

In `docs/runbooks/dgx-vllm-cloudflare.md`, append this section at the end of the file:

```markdown
## Driving it from the GitHub workflows (instead of manual deploy)

The `Demo Up (GKE)` and `Auto Deploy` workflows deploy the `vllm` backend by
default. One-time GitHub repo config (Settings → Secrets and variables → Actions):

- **Secret** `LLM_API_KEY` — the DGX vLLM `--api-key`.
- **Var** `LLM_BASE_URL` — e.g. `https://llm.<your-domain>/v1` (must end in `/v1`).
- **Var** `LLM_MODEL` — e.g. `medical-qa-llama-gdpo` (= `--served-model-name`).
- **Var** `MODEL_BACKEND` (optional) — set to `mock` to make `Auto Deploy` skip the
  real model; otherwise it defaults to `vllm`.

Then: run **Demo Up (GKE)** (leave `backend: vllm`) to provision + deploy the real
model. After that, every push to `main` that passes CI triggers **Auto Deploy**,
which re-applies the secrets and rolls the api on `vllm` — provided the DGX server
and Cloudflare Tunnel are up (otherwise the api pod stays NotReady and the run
fails loudly). To demo the plumbing with the DGX offline, run **Demo Up** with
`backend: mock`.
```

- [ ] **Step 3: Verify README/doc tests still pass**

Run: `.venv/bin/python -m pytest tests/test_readme_cloud_docs.py tests/test_readme_deploy_docs.py -q`
Expected: ALL PASS.

- [ ] **Step 4: Commit**

```bash
git add docs/cloud-setup.md docs/runbooks/dgx-vllm-cloudflare.md
git commit -m "docs: document GitHub config for workflow-driven vllm deploy"
```

---

## Task 7: Full-suite verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all PASS (helm-render tests run via the symlinked `.tools/bin/helm`; the one pre-existing KG-parity skip is unrelated).

- [ ] **Step 2: Sanity-check both workflows parse as YAML**

Run:
```bash
.venv/bin/python -c "import yaml; [yaml.safe_load(open(f)) for f in ['.github/workflows/demo-up.yml','.github/workflows/deploy.yml']]; print('YAML OK')"
```
Expected: `YAML OK`.

- [ ] **Step 3: Final commit (if anything pending)**

```bash
git add -A && git commit -m "chore: complete vllm workflow wiring" || echo "nothing to commit"
```

---

## Self-Review Notes

- **Spec coverage:** demo-up toggle input + env (T1–T2), deploy.yml default-vllm env + create_secrets step (T3–T4), smoke comment refresh (T5), docs incl. operator GitHub config (T6), verification (T7). Operator-side Vars/Secret creation is explicitly out of scope (documented, not automated).
- **Naming consistency:** `MODEL_BACKEND`, `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`, secret `medical-qa-llm-key`, served model `medical-qa-llama-gdpo`, input name `backend` — identical across workflows, tests, and docs, and matching the scripts merged in `feat/dgx-vllm-cloudflare`.
- **Demo safety:** mock remains reachable (demo-up `backend: mock`, deploy.yml `MODEL_BACKEND=mock` Var). With `LLM_*` unset and backend=vllm, `deploy.sh` aborts loudly via its existing `:?` guards — no silent half-config.
