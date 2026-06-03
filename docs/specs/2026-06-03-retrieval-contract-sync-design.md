# Retrieval Contract Sync (SP1) — Design

**Sub-project 1 of the train–serve parity effort.** This is the foundational
step: make the `mlops-platform` serving retrieval reproduce, byte-for-byte, the
exact retrieval behavior that the `baseline/` training and evaluation pipeline
uses — so that what the model is served at inference time matches what it was
trained and evaluated against. Encoder-agnostic; verifiable today on the
existing MedEmbed-large index without any re-training.

## Goal

Bring `medical_qa_platform`'s serving-side retrieval to **behavioral parity**
with the baseline production retrieval (`search_medical_knowledge` →
`retrieve_v1`), define that behavior as an explicit **versioned retrieval
contract**, and **lock the parity in CI with golden fixtures** — without
importing any baseline code and without changing the encoder yet.

## Context

### The skew this fixes

The production retrieval path that **both training and eval** use is:

```
search_medical_knowledge(query)              # baseline/scripts/serve/retrieval_tool.py
  → _retrieve_cached(query)                  # functools.lru_cache(maxsize=512), top_k hardcoded to 5
  → retrieve(query, top_k=5)
  → retrieve_v1(query, top_k=5)              # dual-retrieval fusion ranker
  → "\n".join(f"- {r}" for r in results)     # or "No relevant knowledge found." if empty
```

The serving path actually deployed today is:

```
KGRetrieval.search(query, top_k)             # mlops-platform/.../retrieval/kg_backend.py
  ≈ retrieve_v0                              # simple hyperedge top-k + entity top-3 expansion
```

`retrieve_v0` and `retrieve_v1` are **different algorithms**. The model was
RL-trained against `retrieve_v1` evidence but is served `retrieve_v0`-style
evidence with a different `top_k` and (currently) a divergence even from `v0`
(missing the `[:2]` per-entity cap). This is a severe, multi-dimensional
train–serve skew. SP1 is therefore a **rewrite of `KGRetrieval`**, not a small
fix.

### Decisions locked during brainstorming

1. **One encoder for the whole pipeline.** Train and serve will share a single
   encoder, so retrieval skew is eliminated by construction. SP1 keeps the
   current `MedEmbed-large-v0.1` encoder; the encoder swap + index rebuild is
   **SP2**. SP1 is built encoder-agnostically so SP2 only changes one place.
2. **Baseline is reference, not a dependency.** `mlops-platform` stays
   self-contained: it does **not** import from `baseline/`. The baseline
   `retrieve_v1` logic is faithfully **ported** into mlops and kept honest by a
   golden parity test. The existing `tests/retrieval/test_kg_backend_self_contained.py`
   guard must continue to pass.
3. **Approach A — port + golden test** (chosen over vendor-copy and over
   redesigning to `v0`). The pure ranking logic is isolated from the heavy
   I/O adapter so it is unit-testable and CI-verifiable without faiss/encoder.

## The Retrieval Contract

The contract has three layers; SP1 reproduces all three.

### Layer 1 — Ranking algorithm (`retrieve_v1`)

Exact behavior to reproduce. Constants and weights are normative.

**Inputs per query:** normalized float32 query embedding; FAISS hyperedge index;
FAISS entity index; KG metadata.

**Candidate budgets** (with `top_k = 5`):
- `he_k = max(40, top_k * 8)` → 40 hyperedge candidates
- `ent_k = max(12, top_k * 4)` → 20 entity candidates
- `per_entity_limit = 8`

**Tokenization:** `re.findall(r"[a-z0-9]+", text.lower())` minus STOPWORDS.
STOPWORDS = `{a, an, and, are, as, at, be, by, does, for, from, how, in, is, of,
on, or, the, to, what, which, with}`.

**Helper scores:**
- `rrf(rank, k=10)` = `0.0` if `rank is None` else `1.0 / (k + rank + 1)`
- `lexical_score(query_tokens, hid)` = `|query_tokens ∩ hedge_tokens[hid]| /
  max(1, min(len(query_tokens), 6))`; `0.0` if either side empty.
- `entity_match_score(query_tokens, meta)` = max over `[anchor, *entities]` of
  `|query_tokens ∩ name_tokens| / len(name_tokens)`; `0.0` if no tokens.
- `expansion_priority(query_tokens, entity_name, entity_sim, hid)` =
  `0.60*entity_sim + 0.20*lexical + 0.10*entity_match + 0.10*anchor_bonus`,
  where `anchor_bonus = 1.0` if `hedge_meta[hid].anchor == entity_name` else `0.35`.

**Accumulation:**
- Hyperedge loop: for each `(rank, score, idx)` in the he search, set the
  candidate's `he_rank = rank` and `he_sim = max(he_sim, score)`.
- Entity loop: for each entity hit `(ent_rank, ent_score, idx)`, expand its
  `entity_to_hedges`, score each related hedge by `expansion_priority`, sort
  descending, take the top `per_entity_limit`. For each, set `ent_rank = min(...)`,
  `ent_sim = max(...)`, `exp_rank = min(...)`, `exp_score = max(...)`, and add the
  entity to `matched_entities`.

**Final fusion score per candidate:**
```
0.41*he_sim + 0.14*ent_sim
  + 0.14*rrf(he_rank) + 0.10*rrf(ent_rank) + 0.08*rrf(exp_rank)
  + 0.08*min(1.0, exp_score)
  + 0.10*lexical + 0.05*entity_match
```
Sort candidates by score descending; return the top-`top_k` hyperedge
descriptions (`hedge_by_id[hid]`).

**Supporting structures built at load:**
- `hedge_meta[hid]` = `{description, relation, type, anchor, entities}` from
  `hg['hyperedges']`.
- `hedge_by_id[hid]` = `hedge_meta[hid].description`.
- `entity_to_hedges` = `hg['entity_to_hedges']`.
- `entity_type_by_name[name]` = `hg['entities'][name].type`.
- `hedge_token_sets[hid]` = `set(tokenize(" ".join([description, relation, type,
  anchor, *entities])))`.
- `entity_token_sets[name]` = `set(tokenize(name))`.

The production `medical_hg.json` already carries every field above (verified:
top-level keys `entities` (dict, `name → {name, type}`), `hyperedges`
(`id, description, entities, type, relation, anchor`), `entity_to_hedges`).

### Layer 2 — Tool-response string

The text the model actually sees. `format_evidence(results)` reproduces it:
- non-empty: `"\n".join(f"- {r}" for r in results)`
- empty: `"No relevant knowledge found."`

`top_k` default for the serving retrieval contract is **5** (matching the
hardcoded baseline tool). The smoke pipeline's `top_k: 2` stays as-is — it only
governs the tiny smoke KG, not the production contract.

### Layer 3 — Prompt injection

`api/prompt.py:build_prompt` already prefixes each evidence line with `- `,
matching the per-line format. SP1 reconciles `SYSTEM_PROMPT` with the canonical
eval prompt (currently flagged by a NOTE in `prompt.py`; source of truth is
`baseline/scripts/benchmark/grpo_eval/grpo_eval.py` / `train_rl/data_prep.py`).
**Out of SP1 scope:** the structural difference between single-shot RAG
injection (current api) and the agentic tool round-trip (training) — that is a
later sub-project.

## Architecture & File Structure

Isolate the pure, deterministic ranking logic from the heavy I/O adapter so the
former is covered by unit tests and CI parity tests, and the latter (faiss +
encoder) stays excluded from coverage.

**Create:**
- `src/medical_qa_platform/retrieval/ranker.py` — **pure**, no faiss/ST imports.
  `tokenize`, `rrf`, `lexical_score`, `entity_match_score`, `expansion_priority`,
  and `fuse_candidates(...)` which takes the FAISS outputs (he scores/ids, ent
  scores/ids) plus the metadata structures and returns the ranked list of hedge
  ids. Constants (budgets, weights, STOPWORDS, default `top_k`) live here.
- `src/medical_qa_platform/retrieval/contract.py` — small, covered.
  `format_evidence(results) -> str` and `RETRIEVAL_CONTRACT_VERSION`
  (initial value `"v1-medembed-large"`; SP2 bumps it).
- `scripts/gen_retrieval_golden.py` — generator run **in a baseline venv** (with
  faiss/ST + real `baseline/data/`). Produces the golden fixtures. Not run in CI.
- `tests/retrieval/golden/` — checked-in golden fixtures (see Testing).
- `tests/retrieval/test_contract_parity.py` — the L1 and L2 parity tests.

**Modify:**
- `src/medical_qa_platform/retrieval/kg_backend.py` — rewrite `KGRetrieval` to
  load the richer artifacts, run FAISS with `he_k`/`ent_k`, and delegate ranking
  to `ranker.fuse_candidates`. Encoder model via `KG_ENCODER_MODEL`, device via
  `RETRIEVAL_DEVICE` (unchanged knobs). Stays `# pragma: no cover`.
- `src/medical_qa_platform/api/prompt.py` — reconcile `SYSTEM_PROMPT`; reuse the
  shared per-line evidence format.

**Untouched:** `baseline/` (reference only). `retrieval/service.py` (`/search`
already returns `list[str]` with `top_k` default 5) and `retrieval/client.py`
need no contract change.

## Data Flow

```
/predict → retrieval /search → KGRetrieval.search(q, 5)
              → encode(q) → faiss he_k/ent_k → ranker.fuse_candidates → list[str]
         → build_prompt injects (format aligned to format_evidence) → backend.generate → parse
```

`RETRIEVAL_CONTRACT_VERSION` is recorded for later MLflow pinning (SP3).

## Testing Strategy / Completion Gates

- **L1 — CI, no heavy deps (primary gate):** golden fixtures capture, for a fixed
  query set, the raw FAISS outputs (he scores/ids, ent scores/ids) **and** the
  metadata slices, plus the expected ranked ids from baseline `retrieve_v1`.
  The test replays the captured FAISS outputs through `ranker.fuse_candidates`
  and asserts the ranked ids match exactly. Fully deterministic; catches any
  fusion-logic porting bug without faiss/encoder/GPU.
- **L2 — opt-in (`@pytest.mark.runtime`, skipped unless runtime extra + real
  artifacts present):** run full `KGRetrieval.search(query, 5)` against
  `baseline/data/` and assert equality with baseline `search_medical_knowledge`
  output, on the **same device (CPU)** to avoid float nondeterminism. Run before
  merge.
- **Unit tests** for `ranker.py` helpers (rrf, tokenize/STOPWORDS, lexical,
  entity_match, expansion_priority) and `format_evidence` (incl. empty sentinel).
- **Gates:** `make test` green; coverage ≥ 80% (`ranker.py` + `contract.py`
  covered; `kg_backend.py` excluded as today); L1 parity green in CI; L2 parity
  green when run with real artifacts; `test_kg_backend_self_contained.py` still
  green (no baseline import).

## Out of Scope (later sub-projects)

- **SP2:** encoder swap to a smaller CPU model + KG index rebuild + retrieval
  quality re-eval; bump `RETRIEVAL_CONTRACT_VERSION` + regenerate goldens.
- **SP3:** full MLflow/DVC pipeline (KG build → Stage 1/1.5/2 → eval → register)
  with the contract version pinned to each model version.
- **SP4 / agentic:** serving infra changes and full prompt-structure parity
  (agentic tool round-trip vs single-shot injection).

## Risks / Open Questions

- **Golden regeneration discipline.** Goldens must be regenerated whenever the
  algorithm intentionally changes or the encoder swaps (SP2). The generator must
  record the artifact set + encoder + device it was produced with, embedded in
  the golden filename or a manifest, so stale goldens are detectable.
- **Float nondeterminism for L2.** CPU vs CUDA encoder embeddings differ
  slightly and can reorder near-tie candidates. L2 must pin device = CPU for
  both sides; L1 (replay of captured FAISS outputs) sidesteps this entirely and
  is the authoritative gate.
- **`lru_cache(512)` parity.** Baseline caches per-query results. Caching does
  not change outputs (deterministic), so it is a serving-side performance option,
  not part of the correctness contract; replicate only if desired.
- **System-prompt source of truth.** Confirm which file holds the canonical eval
  system prompt before reconciling (`grpo_eval.py` vs `data_prep.py`) — they must
  already agree per the project's working rules; if they diverge, that is a
  pre-existing bug to surface, not silently pick one.
