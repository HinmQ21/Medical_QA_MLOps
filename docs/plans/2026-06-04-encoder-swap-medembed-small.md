# Encoder Swap → MedEmbed-small (SP2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **NOTE — execution mode:** SP2 spans two repos (the `mlops-platform` git repo for code/config + the outer `minhlbq` repo for the 1-line `baseline/` edit) and includes heavy environment steps (a ~30-min CPU KG build and a golden regen in a baseline venv). Tasks 1–4 (mlops code/config) are ordinary TDD; Task 5 is a controller-run environment sequence. Inline execution is the pragmatic choice.

**Goal:** Switch the retrieval encoder to `abhinand/MedEmbed-small-v0.1` across the whole pipeline (KG build → serving → golden), establishing and verifying the new contract now while deferring the 3-stage re-train to the operator.

**Architecture:** Flip the mlops defaults (contract version, `KGRetrieval` default encoder) to small; make the baseline retrieval tool read its encoder from `KG_ENCODER_MODEL` (default large, back-compat); rebuild the KG with the small encoder into `artifacts/kg_small`; regenerate the SP1 golden from baseline `retrieve_v1` on the small encoder; wire `KG_ENCODER_MODEL` through the retrieval Helm chart; verify L1 + L2 parity. The SP1 `ranker.py` is unchanged (encoder-agnostic).

**Tech Stack:** Python 3.12, faiss + sentence-transformers (baseline venv, for the build/golden), pytest, Helm, PyYAML.

**Spec:** `docs/specs/2026-06-04-encoder-swap-medembed-small-design.md`

**Working dir:** mlops commands run from `/home/vcsai/minhlbq/mlops-platform` (`.venv/bin/python`). The baseline edit + KG build + golden regen run from `/home/vcsai/minhlbq/baseline` with `./training_venv312/bin/python`.

**Encoder:** `abhinand/MedEmbed-small-v0.1` (~384-d). KG → `mlops-platform/artifacts/kg_small/` (git-ignored).

---

## File Structure

**Modify (mlops-platform repo):**
- `src/medical_qa_platform/retrieval/contract.py` — `RETRIEVAL_CONTRACT_VERSION` → `v1-medembed-small`.
- `src/medical_qa_platform/retrieval/kg_backend.py` — default `KG_ENCODER_MODEL` → small.
- `scripts/gen_retrieval_golden.py` — `--embed-model` arg; set `KG_ENCODER_MODEL` env; record in manifest.
- `params.yaml` — `full`/`smoke_full` `build_kg.embed_model` → small.
- `deploy/helm/retrieval/values.yaml` + `templates/deployment.yaml` — `KG_ENCODER_MODEL` env.
- `tests/retrieval/test_contract.py`, `tests/mlops/test_full_config.py`, `tests/deploy/test_helm_retrieval_chart.py` — assertions updated to small.
- `tests/retrieval/golden/retrieve_v1_golden.json` — regenerated (small encoder).

**Modify (minhlbq repo):**
- `baseline/scripts/serve/retrieval_tool.py` — encoder from `KG_ENCODER_MODEL` (default large).

**Create (git-ignored, generated):**
- `mlops-platform/artifacts/kg_small/` — the small-encoder KG.

---

## Task 1: Flip mlops contract + default encoder to small

**Files:** Modify `src/medical_qa_platform/retrieval/contract.py`, `src/medical_qa_platform/retrieval/kg_backend.py`, `tests/retrieval/test_contract.py`.

- [ ] **Step 1: Update the failing test** — in `tests/retrieval/test_contract.py`, change the version assertion:

```python
def test_contract_version_is_v1_medembed_small():
    assert contract.RETRIEVAL_CONTRACT_VERSION == "v1-medembed-small"
```

(Replace the existing `test_contract_version_is_v1_medembed_large` function entirely.)

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/retrieval/test_contract.py -q`
Expected: FAIL — current value is `"v1-medembed-large"`.

- [ ] **Step 3: Update `contract.py`** — change the constant:

```python
RETRIEVAL_CONTRACT_VERSION = "v1-medembed-small"
```

- [ ] **Step 4: Update `kg_backend.py` default encoder** — change the default in `KGRetrieval.__init__`:

```python
        self.encoder_model = encoder_model or os.environ.get(
            "KG_ENCODER_MODEL", "abhinand/MedEmbed-small-v0.1"
        )
```

(Only the default string changes, `abhinand/MedEmbed-large-v0.1` → `abhinand/MedEmbed-small-v0.1`.)

- [ ] **Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests/retrieval/test_contract.py tests/retrieval/test_kg_backend_self_contained.py -q`
Expected: PASS. (The self-contained test fakes `SentenceTransformer` ignoring the model name, so the default change does not affect it; the guard still passes.)

- [ ] **Step 6: Commit**

```bash
git add src/medical_qa_platform/retrieval/contract.py src/medical_qa_platform/retrieval/kg_backend.py tests/retrieval/test_contract.py
git commit -m "feat(retrieval): switch contract + default encoder to MedEmbed-small"
```

---

## Task 2: Parametrize the golden generator's encoder

**Files:** Modify `scripts/gen_retrieval_golden.py`, Create `tests/mlops/test_gen_golden_encoder_arg.py`.

- [ ] **Step 1: Write the failing test** — create `tests/mlops/test_gen_golden_encoder_arg.py`:

```python
from pathlib import Path


def test_generator_exposes_embed_model_and_sets_env():
    src = Path("scripts/gen_retrieval_golden.py").read_text()
    assert "--embed-model" in src
    # the generator must export the encoder to the env the baseline tool reads
    assert 'os.environ["KG_ENCODER_MODEL"]' in src or "os.environ['KG_ENCODER_MODEL']" in src
    # manifest records the actual encoder, not a hardcoded large literal
    assert '"encoder_model": args.embed_model' in src
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/mlops/test_gen_golden_encoder_arg.py -q`
Expected: FAIL.

- [ ] **Step 3: Edit `scripts/gen_retrieval_golden.py`:**
  - Add `import os` to the imports block (after `import json`).
  - Add the argument (after the `--device` line):
    ```python
        parser.add_argument("--embed-model", default="abhinand/MedEmbed-large-v0.1")
    ```
  - Immediately after `args = parser.parse_args()`, set the env the baseline tool reads:
    ```python
        os.environ["KG_ENCODER_MODEL"] = args.embed_model
    ```
  - In the `manifest`, replace the hardcoded line `"encoder_model": "abhinand/MedEmbed-large-v0.1",` with:
    ```python
            "encoder_model": args.embed_model,
    ```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/mlops/test_gen_golden_encoder_arg.py -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/gen_retrieval_golden.py tests/mlops/test_gen_golden_encoder_arg.py
git commit -m "feat(golden): parametrize generator encoder via --embed-model + KG_ENCODER_MODEL"
```

---

## Task 3: Point pipeline KG build at the small encoder

**Files:** Modify `params.yaml`, `tests/mlops/test_full_config.py`.

- [ ] **Step 1: Update the failing test** — in `tests/mlops/test_full_config.py`, change the embed-model assertion in `test_load_full_profile_defaults`:

```python
    assert cfg.build_kg["embed_model"] == "abhinand/MedEmbed-small-v0.1"
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/mlops/test_full_config.py -q`
Expected: FAIL.

- [ ] **Step 3: Edit `params.yaml`** — under BOTH `full:` and `smoke_full:`, change `build_kg.embed_model`:

```yaml
  build_kg:
    embed_model: abhinand/MedEmbed-small-v0.1
    embed_device: cpu
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/mlops/test_full_config.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add params.yaml tests/mlops/test_full_config.py
git commit -m "feat(pipeline): pipeline build_kg uses MedEmbed-small encoder"
```

---

## Task 4: Wire KG_ENCODER_MODEL through the retrieval Helm chart

**Files:** Modify `deploy/helm/retrieval/values.yaml`, `deploy/helm/retrieval/templates/deployment.yaml`, `tests/deploy/test_helm_retrieval_chart.py`.

- [ ] **Step 1: Update/extend the failing test** — in `tests/deploy/test_helm_retrieval_chart.py`, add to the env-dict assertion block (where `env = {item["name"]: ...}` is built):

```python
    assert env["KG_ENCODER_MODEL"] == "abhinand/MedEmbed-small-v0.1"
```

And add to the values assertion (near `values["env"]["retrievalDevice"] == "cpu"`):

```python
    assert values["env"]["kgEncoderModel"] == "abhinand/MedEmbed-small-v0.1"
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/deploy/test_helm_retrieval_chart.py -q`
Expected: FAIL (KeyError `kgEncoderModel` / `KG_ENCODER_MODEL`).

- [ ] **Step 3a: Edit `deploy/helm/retrieval/values.yaml`** — add to the `env:` block:

```yaml
env:
  retrievalBackend: kg
  retrievalDevice: cpu
  kgEncoderModel: abhinand/MedEmbed-small-v0.1
  kgDataDir: /mnt/artifacts/smoke/kg
  hfHome: /mnt/artifacts/hf
  sentenceTransformersHome: /mnt/artifacts/hf/sentence-transformers
```

- [ ] **Step 3b: Edit `deploy/helm/retrieval/templates/deployment.yaml`** — add the env entry (after the `RETRIEVAL_DEVICE` block):

```yaml
            - name: KG_ENCODER_MODEL
              value: {{ .Values.env.kgEncoderModel | quote }}
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/deploy/test_helm_retrieval_chart.py -q`
Expected: PASS.

- [ ] **Step 5: Confirm the chart still renders** (if `make helm-template` tooling is installed)

Run: `make helm-template 2>&1 | tail -3 || echo "helm tools not installed; skipping (covered by render test)"`
Expected: clean render or the skip message.

- [ ] **Step 6: Commit**

```bash
git add deploy/helm/retrieval/values.yaml deploy/helm/retrieval/templates/deployment.yaml tests/deploy/test_helm_retrieval_chart.py
git commit -m "feat(serving): expose KG_ENCODER_MODEL in retrieval Helm chart (MedEmbed-small)"
```

---

## Task 5: Build the small KG + regenerate the golden + verify (controller-run environment sequence)

**Files:** Modify `baseline/scripts/serve/retrieval_tool.py` (minhlbq repo); Create `artifacts/kg_small/` (generated); Modify `tests/retrieval/golden/retrieve_v1_golden.json` (regenerated).

These steps run real models on the GB10 (CPU) and touch the `minhlbq` repo. They are not subagent unit tasks.

- [ ] **Step 1: Make the baseline encoder env-configurable (minhlbq repo, back-compat).**
  In `baseline/scripts/serve/retrieval_tool.py`:
  - Add `import os` to the stdlib import block (e.g. after `import json`).
  - Replace the encoder construction (currently line ~64-66):
    ```python
        inst.encoder = SentenceTransformer(
            'abhinand/MedEmbed-large-v0.1', device=device
        )
    ```
    with:
    ```python
        inst.encoder = SentenceTransformer(
            os.environ.get('KG_ENCODER_MODEL', 'abhinand/MedEmbed-large-v0.1'),
            device=device,
        )
    ```
  Verify back-compat (default unchanged): `grep -n "KG_ENCODER_MODEL" /home/vcsai/minhlbq/baseline/scripts/serve/retrieval_tool.py` shows the env read with the large default.
  Commit in the minhlbq repo (check its branch first; create a branch if on the default/detached HEAD):
  ```bash
  git -C /home/vcsai/minhlbq add baseline/scripts/serve/retrieval_tool.py
  git -C /home/vcsai/minhlbq commit -m "feat(retrieval_tool): encoder from KG_ENCODER_MODEL env (default large, back-compat)"
  ```

- [ ] **Step 2: Build the small-encoder KG** (~30 min, CPU; does not touch the GPU):

```bash
mkdir -p /home/vcsai/minhlbq/mlops-platform/artifacts/kg_small
cd /home/vcsai/minhlbq/baseline && ./training_venv312/bin/python -m scripts.build_kg.run_pipeline \
    --data-dir /home/vcsai/minhlbq/mlops-platform/artifacts/kg_small \
    --embed-model abhinand/MedEmbed-small-v0.1 \
    --embed-device cpu
```
Expected: writes `index_hyperedge.bin`, `index_entity.bin`, `hedge_ids.npy`, `entity_names.npy`, `medical_hg.json` into `artifacts/kg_small`. Confirm + record the embedding dim:
```bash
cd /home/vcsai/minhlbq/baseline && ./training_venv312/bin/python -c "import faiss; ix=faiss.read_index('/home/vcsai/minhlbq/mlops-platform/artifacts/kg_small/index_hyperedge.bin'); print('hedge index dim:', ix.d, 'ntotal:', ix.ntotal)"
```
Expected: a dim (≈384) and a non-zero ntotal.

- [ ] **Step 3: Regenerate the golden from baseline retrieve_v1 on the small encoder:**

```bash
cd /home/vcsai/minhlbq/baseline && ./training_venv312/bin/python \
    /home/vcsai/minhlbq/mlops-platform/scripts/gen_retrieval_golden.py \
    --baseline-root /home/vcsai/minhlbq/baseline \
    --data-dir /home/vcsai/minhlbq/mlops-platform/artifacts/kg_small \
    --embed-model abhinand/MedEmbed-small-v0.1 \
    --device cpu \
    --out /home/vcsai/minhlbq/mlops-platform/tests/retrieval/golden/retrieve_v1_golden.json
```
Expected: `wrote 8 cases to ...`. Confirm the manifest records the small encoder:
```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/python -c "import json; m=json.load(open('tests/retrieval/golden/retrieve_v1_golden.json'))['manifest']; print(m['encoder_model'], m['data_dir'])"
```
Expected: `abhinand/MedEmbed-small-v0.1 .../artifacts/kg_small`.

- [ ] **Step 4: L1 verify (deterministic, CI path) on the new golden:**

Run: `cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/python -m pytest tests/retrieval/test_contract_parity.py -q -rs`
Expected: 3 L1 tests PASS (the ported ranker reproduces baseline `retrieve_v1` on the small-encoder golden); L2 SKIPPED (no `KG_DATA_DIR`).

- [ ] **Step 5: L2 verify (full backend, small encoder, real artifacts):**

```bash
cd /home/vcsai/minhlbq/baseline && KG_DATA_DIR=/home/vcsai/minhlbq/mlops-platform/artifacts/kg_small RETRIEVAL_DEVICE=cpu KG_ENCODER_MODEL=abhinand/MedEmbed-small-v0.1 PYTHONPATH=/home/vcsai/minhlbq/mlops-platform/src ./training_venv312/bin/python - <<'PY'
import os, json
from medical_qa_platform.retrieval.kg_backend import KGRetrieval
g = json.load(open("/home/vcsai/minhlbq/mlops-platform/tests/retrieval/golden/retrieve_v1_golden.json"))
tk = g["manifest"]["top_k"]
b = KGRetrieval(data_dir="/home/vcsai/minhlbq/mlops-platform/artifacts/kg_small", device="cpu")
ok = all(b.search(c["query"], top_k=tk) == c["expected_ranked_descriptions"] for c in g["cases"])
print("L2 FULL-BACKEND PARITY (small):", "PASS %d/%d" % (len(g["cases"]), len(g["cases"])) if ok else "FAIL")
PY
```
Expected: `L2 FULL-BACKEND PARITY (small): PASS 8/8`. If it mismatches, do NOT edit the golden — the small-encoder front-half in `KGRetrieval` diverges from baseline; debug `kg_backend.search` vs baseline `retrieve_v1`.

- [ ] **Step 6: Commit the regenerated golden** (mlops repo):

```bash
cd /home/vcsai/minhlbq/mlops-platform
git add tests/retrieval/golden/retrieve_v1_golden.json
git commit -m "test(golden): regenerate SP1 parity fixture on MedEmbed-small encoder"
```

---

## Task 6: Full-suite + coverage gate

**Files:** none (verification only).

- [ ] **Step 1: Full suite + coverage**

Run: `cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/python -m pytest --cov --cov-report= -q 2>&1 | tail -3`
Expected: all tests pass; total coverage ≥ 80% (unchanged — only constants/config/golden changed; `ranker.py` is untouched).

- [ ] **Step 2: Confirm self-contained guard + parity still green**

Run: `.venv/bin/python -m pytest tests/retrieval -q 2>&1 | tail -2`
Expected: all green (L1 on the small golden; self-contained guard).

- [ ] **Step 3: Stop** — `git status` clean (mlops repo). SP2 code/config + verified small-encoder contract complete on branch `encoder-swap-medembed-small`; the baseline edit is committed in the `minhlbq` repo. The 3-stage re-train remains operator-run via the SP3 pipeline (`make full-pipeline`, GPU free).

---

## Notes for the implementer

- **`ranker.py` does not change.** SP2 only swaps the encoder/index/golden + config; the fusion logic is encoder-agnostic. L1 passing on the regenerated golden confirms this.
- **Two repos:** mlops code/config/golden commit to `mlops-platform`; the single back-compat encoder edit commits to `minhlbq` (`baseline/`). Check `minhlbq`'s branch before committing there.
- **Back-compat:** the baseline default stays `MedEmbed-large` — existing large-encoder runs/checkpoints are unaffected; small is selected only via `KG_ENCODER_MODEL` / the rebuilt index.
- **Determinism for L2:** run on CPU (`RETRIEVAL_DEVICE=cpu`) with the same encoder that built the index; the golden was generated CPU-side.
- **Never edit the golden to pass.** A mismatch means the `KGRetrieval` front-half diverges from baseline `retrieve_v1` on the new encoder.
- **Re-train is deferred.** Do not run `make full-pipeline` (full) here — that is the operator's multi-day GPU step.
