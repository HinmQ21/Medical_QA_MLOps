# LLMBackend Rename Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the serving `VllmBackend` to a generic `LLMBackend` (canonical `MODEL_BACKEND=llm`) with `vllm` kept as a back-compat alias, so the OpenAI-compatible client's name reflects that it talks to any `/v1` server (DGX vLLM or in-cluster llama.cpp) — without breaking existing deployments.

**Architecture:** Pure rename of one backend class/file + the factory selector, plus flipping user-facing defaults/docs to `llm`. The factory accepts both `llm` and `vllm`; `LLMBackend.name = "llm"` so responses report the canonical name. No behavior, request/response-shape, or env-var changes. The genuine vLLM *training-engine* references (`mlops/`, `params.yaml`, the DGX vLLM server) are explicitly untouched.

**Tech Stack:** Python 3.12 (httpx, dataclasses), pytest, Helm, bash, GitHub Actions YAML.

**Spec:** `docs/superpowers/specs/2026-06-08-llm-backend-rename-design.md`

**Execution prerequisites:**
- Repo `/home/vcsai/minhlbq/mlops-platform`, branch `feat/llm-backend-rename` (already created; spec already committed there). Work from this dir.
- Test runner `.venv/bin/pytest`. Project-local helm at `.tools/bin/helm` (present).
- `main` is local; do NOT push unless asked.

---

## File Structure

**Renamed (git mv, contents edited):**
- `src/medical_qa_platform/inference/vllm_backend.py` → `llm_backend.py` (class `VllmBackend`→`LLMBackend`, `name`→`"llm"`, docstring de-vendored)
- `tests/inference/test_vllm_backend.py` → `test_llm_backend.py`

**Modified:**
- `src/medical_qa_platform/inference/__init__.py` — factory branch accepts `("llm", "vllm")`
- `tests/inference/test_mock_backend.py` — factory tests (canonical + alias)
- `scripts/cloud/deploy.sh` — recognize `llm`|`vllm`, pass `$BACKEND` through
- `.github/workflows/demo-up.yml` — dropdown `[llm, mock]`, default `llm`
- `.github/workflows/deploy.yml` — `|| 'llm'`
- `tests/cloud/test_deploy_sh.py`, `tests/deploy/test_cloud_workflows.py` — flip + alias
- `tests/test_config.py`, `tests/app/test_ui_client.py` — canonical-name fixtures
- `README.md`, `docs/cloud-setup.md`, `docs/runbooks/dgx-vllm-cloudflare.md` — selector docs → `llm`

**Out of scope (do NOT touch):** `mlops/pipelines/*`, `params.yaml`, the vLLM-server prose in the runbook, `docs/plans/*`, `docs/specs/2026-06-03-*`.

---

## Task 1: Rename the backend class, file, and factory (with `vllm` alias)

**Files:**
- Rename: `src/medical_qa_platform/inference/vllm_backend.py` → `src/medical_qa_platform/inference/llm_backend.py`
- Modify: `src/medical_qa_platform/inference/__init__.py`
- Rename: `tests/inference/test_vllm_backend.py` → `tests/inference/test_llm_backend.py`
- Modify: `tests/inference/test_mock_backend.py`

- [ ] **Step 1: Rename the backend test file and rewrite its symbols.** First move it, then replace its contents:

```bash
git mv tests/inference/test_vllm_backend.py tests/inference/test_llm_backend.py
```

Replace the ENTIRE contents of `tests/inference/test_llm_backend.py` with (every `VllmBackend`→`LLMBackend`, import path `llm_backend`, name `"llm"`; behavior cases unchanged):

```python
import json

import httpx

from medical_qa_platform.inference.llm_backend import LLMBackend


def _backend(captured: dict, *, status: int = 200, json_body=None):
    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["method"] = request.method
        captured["headers"] = dict(request.headers)
        if request.method == "POST":
            captured["body"] = json.loads(request.content)
        return httpx.Response(
            status,
            json=json_body
            if json_body is not None
            else {"choices": [{"message": {"content": "<answer>B</answer>"}}]},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return LLMBackend(base_url="http://pod/v1", model="m", api_key="k", client=client)


def test_name():
    assert _backend({}).name == "llm"


def test_generate_returns_content():
    out = _backend({}).generate([{"role": "user", "content": "x"}])
    assert out == "<answer>B</answer>"


def test_posts_to_chat_completions_with_auth_model_and_max_tokens():
    captured = {}
    _backend(captured).generate([{"role": "user", "content": "x"}], max_tokens=2048)
    assert captured["url"] == "http://pod/v1/chat/completions"
    assert captured["headers"]["authorization"] == "Bearer k"
    assert captured["body"]["model"] == "m"
    assert captured["body"]["max_tokens"] == 2048
    assert captured["body"]["messages"] == [{"role": "user", "content": "x"}]


def test_from_env(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://x/v1")
    monkeypatch.setenv("LLM_MODEL", "llama")
    monkeypatch.setenv("LLM_API_KEY", "secret")
    backend = LLMBackend.from_env()
    assert backend.base_url == "http://x/v1"
    assert backend.model == "llama"
    assert backend.api_key == "secret"


def test_health_check_true_on_200_models():
    captured = {}
    ok = _backend(captured, json_body={"data": []}).health_check()
    assert ok is True
    assert captured["url"] == "http://pod/v1/models"
    assert captured["method"] == "GET"
    assert captured["headers"]["authorization"] == "Bearer k"


def test_health_check_false_on_bad_status():
    backend = _backend({}, status=503, json_body={"error": "down"})
    assert backend.health_check() is False


def test_health_check_false_on_transport_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    backend = LLMBackend(base_url="http://pod/v1", model="m", client=client)
    assert backend.health_check() is False
```

- [ ] **Step 2: Update the factory tests in `tests/inference/test_mock_backend.py`.** Replace the function `test_factory_returns_vllm_by_name` (currently imports `VllmBackend` from `vllm_backend` and asserts `get_backend("vllm")` is a `VllmBackend`) with these TWO functions:

```python
def test_factory_returns_llm_by_name():
    from medical_qa_platform.inference.llm_backend import LLMBackend

    assert isinstance(get_backend("llm"), LLMBackend)


def test_factory_vllm_alias_maps_to_llm():
    from medical_qa_platform.inference.llm_backend import LLMBackend

    # "vllm" is kept as a back-compat alias for the generic LLM/OpenAI backend.
    assert isinstance(get_backend("vllm"), LLMBackend)
```

(Leave `test_factory_unknown_raises`, `test_factory_no_longer_knows_runpod`, and `test_factory_no_longer_knows_kserve` untouched.)

- [ ] **Step 3: Run the inference tests to verify they FAIL**

Run: `.venv/bin/pytest tests/inference/ -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'medical_qa_platform.inference.llm_backend'` (the file and class don't exist yet).

- [ ] **Step 4: Rename the backend module and edit its symbols.** Move the file:

```bash
git mv src/medical_qa_platform/inference/vllm_backend.py src/medical_qa_platform/inference/llm_backend.py
```

Then make these four edits in `src/medical_qa_platform/inference/llm_backend.py`.

(a) Replace the module docstring (the first triple-quoted block) with:

```python
"""Generic OpenAI-compatible chat-completions backend.

Talks to any server exposing the OpenAI ``/v1`` API — a self-hosted vLLM server
on the DGX-Spark (reached over a Cloudflare Tunnel) or an in-cluster llama.cpp
server, among others. Configured via the ``LLM_BASE_URL``, ``LLM_MODEL`` and
``LLM_API_KEY`` environment variables.
"""
```

(b) Class declaration:

```python
class VllmBackend(ModelBackend):
```
→
```python
class LLMBackend(ModelBackend):
```

(c) The name attribute:

```python
    name = "vllm"
```
→
```python
    name = "llm"
```

(d) The `from_env` return annotation:

```python
    def from_env(cls) -> "VllmBackend":
```
→
```python
    def from_env(cls) -> "LLMBackend":
```

(Leave `__init__`, `_auth_headers`, `generate`, and `health_check` bodies exactly as they are.)

- [ ] **Step 5: Update the factory in `src/medical_qa_platform/inference/__init__.py`.** Replace the `vllm` branch:

```python
    if name == "vllm":
        from .vllm_backend import VllmBackend

        return VllmBackend.from_env()
```
with:
```python
    if name in ("llm", "vllm"):  # "vllm" is a back-compat alias for the generic LLM/OpenAI backend
        from .llm_backend import LLMBackend

        return LLMBackend.from_env()
```

- [ ] **Step 6: Run the inference tests to verify they PASS**

Run: `.venv/bin/pytest tests/inference/ -q`
Expected: PASS (test_llm_backend.py all green; `test_factory_returns_llm_by_name` and `test_factory_vllm_alias_maps_to_llm` pass; the kserve/runpod negative guards still pass).

- [ ] **Step 7: Confirm no stale references to the old symbol in source/tests**

Run: `grep -rnI "VllmBackend\|vllm_backend" src/ tests/`
Expected: no matches.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "refactor: rename serving VllmBackend -> generic LLMBackend

The backend is a generic OpenAI /v1 client (works against vLLM or llama.cpp).
Rename class/file/name to llm; factory accepts \"llm\" (canonical) and \"vllm\"
(back-compat alias). No behavior change.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Flip deploy.sh + workflows to `llm` canonical (alias still honored)

**Files:**
- Modify: `tests/cloud/test_deploy_sh.py`
- Modify: `tests/deploy/test_cloud_workflows.py`
- Modify: `scripts/cloud/deploy.sh`
- Modify: `.github/workflows/demo-up.yml`
- Modify: `.github/workflows/deploy.yml`

- [ ] **Step 1: Add the failing tests.** In `tests/cloud/test_deploy_sh.py`, add these two functions (next to the existing `test_dry_run_flips_to_vllm_backend_when_requested`, which you LEAVE as-is — it proves the `vllm` alias still flips):

```python
def test_dry_run_flips_to_llm_backend_when_requested():
    env = {
        **ENV,
        "MODEL_BACKEND": "llm",
        "LLM_BASE_URL": "https://llm.example/v1",
        "LLM_MODEL": "medical-qa-llama-gdpo",
    }
    out = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"], capture_output=True, text=True, env=env
    )
    assert out.returncode == 0, out.stderr
    o = out.stdout
    assert "env.modelBackend=llm" in o
    assert "env.llmBaseUrl=https://llm.example/v1" in o
    assert "env.llmModel=medical-qa-llama-gdpo" in o


def test_llm_backend_requires_base_url_and_model():
    env = {**ENV, "MODEL_BACKEND": "llm"}
    out = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"], capture_output=True, text=True, env=env
    )
    assert out.returncode != 0
```

In `tests/deploy/test_cloud_workflows.py`, edit the two existing assertions:
- In `test_demo_up_wires_vllm_backend_toggle`: change `assert inputs["backend"]["default"] == "vllm"` → `== "llm"`, and `assert "vllm" in inputs["backend"]["options"]` → `assert "llm" in inputs["backend"]["options"]`.
- In `test_deploy_auto_wires_vllm_and_ensures_llm_secret`: change `assert "MODEL_BACKEND: ${{ vars.MODEL_BACKEND || 'vllm' }}" in text` → `... || 'llm' }}" in text`.

- [ ] **Step 2: Run the edited/new tests to verify they FAIL**

Run: `.venv/bin/pytest tests/cloud/test_deploy_sh.py tests/deploy/test_cloud_workflows.py -q`
Expected: FAIL — `MODEL_BACKEND=llm` currently falls through to the mock path (no `env.modelBackend=llm`, no required-var guard), the demo-up default is still `vllm`, and deploy.yml still says `|| 'vllm'`.

- [ ] **Step 3: Update `scripts/cloud/deploy.sh`.** Replace the backend-flip block (lines ~36–47) — from:

```bash
# api defaults to the mock backend (GKE-only demo). Set MODEL_BACKEND=vllm plus
# LLM_BASE_URL / LLM_MODEL to point at the self-hosted vLLM server on the DGX-Spark
# (reached via Cloudflare Tunnel); the LLM_API_KEY secret is created separately.
API_SET=(--set image.tag="$IMAGE_TAG")
BACKEND="${MODEL_BACKEND:-mock}"
if [ "$BACKEND" = "vllm" ]; then
  : "${LLM_BASE_URL:?set LLM_BASE_URL (e.g. https://llm.example/v1) for the vllm backend}"
  : "${LLM_MODEL:?set LLM_MODEL (vllm --served-model-name) for the vllm backend}"
  API_SET+=(--set env.modelBackend=vllm \
            --set env.llmBaseUrl="$LLM_BASE_URL" \
            --set env.llmModel="$LLM_MODEL")
fi
```

to:

```bash
# api defaults to the mock backend (GKE-only demo). Set MODEL_BACKEND=llm plus
# LLM_BASE_URL / LLM_MODEL to point at an OpenAI /v1 server (self-hosted vLLM on the
# DGX-Spark via Cloudflare Tunnel, or the in-cluster llama.cpp InferenceService); the
# LLM_API_KEY secret is created separately. "vllm" is accepted as a back-compat alias.
API_SET=(--set image.tag="$IMAGE_TAG")
BACKEND="${MODEL_BACKEND:-mock}"
if [ "$BACKEND" = "llm" ] || [ "$BACKEND" = "vllm" ]; then
  : "${LLM_BASE_URL:?set LLM_BASE_URL (e.g. https://llm.example/v1) for the llm backend}"
  : "${LLM_MODEL:?set LLM_MODEL (the served model name) for the llm backend}"
  API_SET+=(--set env.modelBackend="$BACKEND" \
            --set env.llmBaseUrl="$LLM_BASE_URL" \
            --set env.llmModel="$LLM_MODEL")
fi
```

- [ ] **Step 4: Update `.github/workflows/demo-up.yml`.** Replace the `backend` input block:

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

with:

```yaml
      backend:
        description: Model backend (llm = real model via LLM_BASE_URL [DGX vLLM or in-cluster llama.cpp], mock = deterministic)
        required: true
        type: choice
        options:
          - llm
          - mock
        default: llm
```

- [ ] **Step 5: Update `.github/workflows/deploy.yml`.** Change the line:

```yaml
      MODEL_BACKEND: ${{ vars.MODEL_BACKEND || 'vllm' }}
```
to:
```yaml
      MODEL_BACKEND: ${{ vars.MODEL_BACKEND || 'llm' }}
```

- [ ] **Step 6: Run the tests + a both-ways dry-run to verify PASS**

Run:
```bash
.venv/bin/pytest tests/cloud/test_deploy_sh.py tests/deploy/test_cloud_workflows.py -q && \
PATH=/usr/bin:/bin GCP_PROJECT=demo MODEL_BACKEND=llm LLM_BASE_URL=https://llm.example/v1 LLM_MODEL=m \
  bash scripts/cloud/deploy.sh --dry-run | grep "env.modelBackend=llm" && \
PATH=/usr/bin:/bin GCP_PROJECT=demo MODEL_BACKEND=vllm LLM_BASE_URL=https://llm.example/v1 LLM_MODEL=m \
  bash scripts/cloud/deploy.sh --dry-run | grep "env.modelBackend=vllm" && echo FLIP_OK
```
Expected: tests pass, both greps print their line, and `FLIP_OK`. (`llm` flips to `env.modelBackend=llm`; `vllm` still flips via the alias to `env.modelBackend=vllm`.)

- [ ] **Step 7: Commit**

```bash
git add scripts/cloud/deploy.sh .github/workflows/demo-up.yml .github/workflows/deploy.yml \
        tests/cloud/test_deploy_sh.py tests/deploy/test_cloud_workflows.py
git commit -m "chore(deploy): make 'llm' the canonical MODEL_BACKEND value

deploy.sh recognizes llm|vllm and passes the chosen value through; demo-up
dropdown + deploy.yml default flip to llm. 'vllm' still works (alias).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Flip docs + remaining test fixtures to the canonical `llm`

**Files:**
- Modify: `tests/test_config.py`
- Modify: `tests/app/test_ui_client.py`
- Modify: `README.md`
- Modify: `docs/cloud-setup.md`
- Modify: `docs/runbooks/dgx-vllm-cloudflare.md`

These are canonical-name fixture/doc updates (the API now reports `backend: "llm"`). The fixture tests stay self-contained, so they pass before and after — run them to confirm green rather than to fail first.

- [ ] **Step 1: Update `tests/test_config.py`.** In `test_reads_env`, change:

```python
    monkeypatch.setenv("MODEL_BACKEND", "vllm")
```
→
```python
    monkeypatch.setenv("MODEL_BACKEND", "llm")
```
and:
```python
    assert settings.model_backend == "vllm"
```
→
```python
    assert settings.model_backend == "llm"
```

- [ ] **Step 2: Update `tests/app/test_ui_client.py`.** Four edits — the API response fixtures now carry the canonical name:
- In `test_predict_parses_success_and_sends_key`: change `"backend": "vllm",` → `"backend": "llm",` and `assert res.backend == "vllm"` → `assert res.backend == "llm"`.
- In `test_fetch_version_returns_json`: change `json={"backend": "vllm", "model_version": "smoke-dev", "contract_version": "v1"},` → `json={"backend": "llm", "model_version": "smoke-dev", "contract_version": "v1"},` and `assert out["backend"] == "vllm"` → `assert out["backend"] == "llm"`.

- [ ] **Step 3: Update `README.md`.** Two edits.

(a) In the KServe paragraph, change:
```markdown
from its `<answer>…</answer>` output. The API consumes it through the `vllm` backend:
set `MODEL_BACKEND=vllm` and point `LLM_BASE_URL` at the in-cluster predictor Service
```
to:
```markdown
from its `<answer>…</answer>` output. The API consumes it through the `llm` backend:
set `MODEL_BACKEND=llm` and point `LLM_BASE_URL` at the in-cluster predictor Service
```

(b) In the "Cloud Deploy" section, change:
```markdown
CI/CD. The CI demo uses the **mock** backend; flip to `MODEL_BACKEND=vllm`
(self-hosted vLLM on the DGX-Spark via Cloudflare Tunnel) for a real model —
```
to:
```markdown
CI/CD. The CI demo uses the **mock** backend; flip to `MODEL_BACKEND=llm`
(self-hosted vLLM on the DGX-Spark via Cloudflare Tunnel, or the in-cluster
llama.cpp InferenceService) for a real model —
```

- [ ] **Step 4: Update `docs/cloud-setup.md`.** Replace the "Real 3B model" bullet:

```markdown
- **Real 3B model (vLLM):** the deploy workflows default to the `vllm` backend.
  Set in GitHub repo settings: **Secret** `LLM_API_KEY` (the DGX vLLM api-key);
  **Vars** `LLM_BASE_URL` (e.g. `https://llm.<domain>/v1` — must end in `/v1`) and
  `LLM_MODEL` (e.g. `medical-qa-llama-gdpo`); optional Var `MODEL_BACKEND` to
  override the default. `Demo Up (GKE)` has a `backend` input (`vllm`|`mock`,
  default `vllm`); `Auto Deploy` uses `vllm` unless `MODEL_BACKEND` is set. Stand
```

with:

```markdown
- **Real model (`llm` backend):** the deploy workflows default to the `llm` backend
  (`vllm` still accepted as an alias). Set in GitHub repo settings: **Secret**
  `LLM_API_KEY` (the DGX vLLM api-key); **Vars** `LLM_BASE_URL` (e.g.
  `https://llm.<domain>/v1` — must end in `/v1`) and `LLM_MODEL` (e.g.
  `medical-qa-llama-gdpo`); optional Var `MODEL_BACKEND` to override the default.
  `Demo Up (GKE)` has a `backend` input (`llm`|`mock`, default `llm`); `Auto Deploy`
  uses `llm` unless `MODEL_BACKEND` is set. Stand
```

- [ ] **Step 5: Update `docs/runbooks/dgx-vllm-cloudflare.md`.** Five selector edits (the vLLM-server prose stays — only `MODEL_BACKEND`/dropdown selector wording changes).

(a) The manual-deploy snippet:
```bash
# deploy the api on the vllm backend (note the /v1 suffix)
export MODEL_BACKEND=vllm
```
→
```bash
# deploy the api on the llm backend (note the /v1 suffix)
export MODEL_BACKEND=llm
```

(b) The workflows paragraph:
```markdown
The `Demo Up (GKE)` and `Auto Deploy` workflows deploy the `vllm` backend by
default. One-time GitHub repo config (Settings → Secrets and variables → Actions):
```
→
```markdown
The `Demo Up (GKE)` and `Auto Deploy` workflows deploy the `llm` backend by
default (`vllm` still accepted as an alias). One-time GitHub repo config (Settings → Secrets and variables → Actions):
```

(c) The optional-var bullet:
```markdown
- **Var** `MODEL_BACKEND` (optional) — set to `mock` to make `Auto Deploy` skip the
  real model; otherwise it defaults to `vllm`.
```
→
```markdown
- **Var** `MODEL_BACKEND` (optional) — set to `mock` to make `Auto Deploy` skip the
  real model; otherwise it defaults to `llm`.
```

(d) The "Then:" line:
```markdown
Then: run **Demo Up (GKE)** (leave `backend: vllm`) to provision + deploy the real
```
→
```markdown
Then: run **Demo Up (GKE)** (leave `backend: llm`) to provision + deploy the real
```

(e) The "rolls the api" line:
```markdown
which re-applies the secrets and rolls the api on `vllm` — provided the DGX server
```
→
```markdown
which re-applies the secrets and rolls the api on `llm` — provided the DGX server
```

- [ ] **Step 6: Run the affected tests to confirm green**

Run: `.venv/bin/pytest tests/test_config.py tests/app/test_ui_client.py tests/test_readme_deploy_docs.py -q`
Expected: PASS. (The README doc-test checks for `KServe`/`llama.cpp`/etc., not `vllm`/`llm`, so the README edits don't disturb it.)

- [ ] **Step 7: Commit**

```bash
git add tests/test_config.py tests/app/test_ui_client.py README.md docs/cloud-setup.md docs/runbooks/dgx-vllm-cloudflare.md
git commit -m "docs: use canonical 'llm' MODEL_BACKEND value in docs + fixtures

API now reports backend=llm; flip the MODEL_BACKEND selector references in
README/cloud-setup/runbook and the UI-client/config fixtures. vLLM-server prose
and the genuine training-engine refs are untouched. 'vllm' still works (alias).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Full-suite verification

**Files:** none (verification only)

- [ ] **Step 1: Full test suite with the coverage gate**

Run: `.venv/bin/pytest --cov=medical_qa_platform --cov-report=term-missing --cov-fail-under=80`
Expected: all tests PASS, coverage ≥ 80%.

- [ ] **Step 2: Helm parity (api chart unaffected, run for safety)**

Run: `make helm-lint && make helm-template && echo HELM_OK`
Expected: clean, `HELM_OK`.

- [ ] **Step 3: Confirm the old symbol is gone and only legitimate `vllm` references remain**

Run: `grep -rnI "VllmBackend\|vllm_backend" src/ tests/`
Expected: no matches.

Run: `grep -rniI "vllm" src/ tests/ scripts/ .github/ README.md docs/cloud-setup.md`
Expected: the only matches are intentional — the factory alias branch + comment (`inference/__init__.py`), the deploy.sh alias condition (`= "vllm"`) + its comment, the README/cloud-setup "self-hosted vLLM"/"alias" prose, and `tests/cloud/test_deploy_sh.py`'s alias test (`MODEL_BACKEND=vllm`). There must be **no** `MODEL_BACKEND: ... || 'vllm'`, no `default: vllm` in demo-up, and no `VllmBackend`.

- [ ] **Step 4: Confirm clean tree + branch commits**

Run: `git status && git log --oneline main..HEAD`
Expected: clean tree; the spec commit + three task commits (Tasks 1–3) on `feat/llm-backend-rename`.

---

## Self-Review

- **Spec coverage:** (1) rename class/file/`name` → Task 1; (2) factory `llm` canonical + `vllm` alias → Task 1; (3) canonical-name reporting (`name="llm"`) → Task 1 (test_name) + Task 3 (fixtures); (4) flip deploy.sh/workflows defaults to `llm`, pass `$BACKEND` through, recognize alias → Task 2; (5) docs flip selector → `llm`, keep vLLM-server prose → Task 3; (6) no env-var/behavior change → enforced by reusing identical method bodies (Task 1 step 4d) and unchanged `LLM_*` vars; (7) out-of-scope (mlops/params/training-engine/historical docs) → never referenced in any task. All spec sections map to a task.
- **Placeholder scan:** every step has the literal code/edit and exact commands with expected output. No TBD/"similar to"/vague items.
- **Type/name consistency:** class `LLMBackend`, module `llm_backend`, `name = "llm"`, selector values `"llm"`/`"vllm"`, helm key `env.modelBackend`, test names (`test_factory_returns_llm_by_name`, `test_factory_vllm_alias_maps_to_llm`, `test_dry_run_flips_to_llm_backend_when_requested`, `test_llm_backend_requires_base_url_and_model`) are used consistently across Tasks 1–4. The `from_env` annotation `"LLMBackend"` matches the class.
