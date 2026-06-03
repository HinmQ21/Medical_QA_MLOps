# Retrieval Contract Sync (SP1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `medical_qa_platform`'s serving retrieval reproduce, byte-for-byte, the baseline production retrieval (`search_medical_knowledge` → `retrieve_v1` fusion ranker), defined as a versioned contract and locked in CI with golden fixtures.

**Architecture:** Split the pure, deterministic ranking logic (`ranker.py`) from the heavy faiss/encoder I/O adapter (`KGRetrieval` in `kg_backend.py`). The pure ranker is faithfully ported from baseline `retrieve_v1` and unit-tested plus golden-tested without faiss/encoder. A generator script (run once in a baseline venv) captures the FAISS front-half outputs and baseline's expected ranked descriptions into a checked-in golden fixture; a CI test (L1) replays those through the pure ranker. An opt-in test (L2) verifies the full backend against baseline on real artifacts. No `baseline/` import from the shipped package.

**Tech Stack:** Python 3.12, pytest, pytest-cov. Heavy deps (faiss, sentence-transformers, numpy) only in the optional `[runtime]` extra and the opt-in L2 test.

**Spec:** `docs/specs/2026-06-03-retrieval-contract-sync-design.md`

**Working directory:** all commands run from `mlops-platform/`. The package venv is `.venv` (`.venv/bin/python`).

---

## File Structure

**Create:**
- `src/medical_qa_platform/retrieval/ranker.py` — pure ranking logic + constants. No faiss/ST. Covered.
- `src/medical_qa_platform/retrieval/contract.py` — `format_evidence` + `RETRIEVAL_CONTRACT_VERSION`. Covered.
- `scripts/gen_retrieval_golden.py` — golden generator, run manually in a baseline venv. Not imported by the package; not run in CI.
- `tests/retrieval/test_ranker.py` — unit tests for ranker helpers + `fuse_candidates`.
- `tests/retrieval/test_contract.py` — unit tests for `contract.py`.
- `tests/retrieval/test_contract_parity.py` — L1 (CI) + L2 (opt-in) parity tests.
- `tests/retrieval/golden/retrieve_v1_golden.json` — checked-in golden fixture (produced by the generator).
- `tests/api/test_system_prompt_sync.py` — asserts mlops `SYSTEM_PROMPT` matches the canonical baseline text.

**Modify:**
- `src/medical_qa_platform/retrieval/kg_backend.py` — rewrite `KGRetrieval` to delegate ranking to `ranker.fuse_candidates`.
- `src/medical_qa_platform/api/prompt.py` — set `SYSTEM_PROMPT` to the canonical text; drop the stale NOTE.
- `pyproject.toml` — register the `runtime` pytest marker.

**Untouched:** everything under `baseline/` (reference only). `retrieval/service.py`, `retrieval/client.py` (already contract-compatible: `/search` returns `list[str]`, `top_k` default 5).

---

## Task 1: Pure ranker helpers

**Files:**
- Create: `src/medical_qa_platform/retrieval/ranker.py`
- Test: `tests/retrieval/test_ranker.py`

- [ ] **Step 1: Write the failing test**

Create `tests/retrieval/test_ranker.py`:

```python
import pytest

from medical_qa_platform.retrieval import ranker


def test_candidate_budgets():
    assert ranker.candidate_budgets(5) == (40, 20)
    assert ranker.candidate_budgets(2) == (40, 12)
    assert ranker.candidate_budgets(10) == (80, 40)


def test_tokenize_lowercases_splits_and_drops_stopwords():
    assert ranker.tokenize("What is the Treatment of Diabetes?") == [
        "treatment",
        "diabetes",
    ]
    assert ranker.tokenize("ACE-inhibitor, 10mg!") == ["ace", "inhibitor", "10mg"]


def test_rrf():
    assert ranker.rrf(None) == 0.0
    assert ranker.rrf(0) == pytest.approx(1.0 / 11)
    assert ranker.rrf(9) == pytest.approx(1.0 / 20)


def test_lexical_score():
    q = {"treatment", "diabetes"}
    assert ranker.lexical_score(q, {"diabetes", "drug", "treats"}) == pytest.approx(0.5)
    assert ranker.lexical_score(set(), {"diabetes"}) == 0.0
    assert ranker.lexical_score(q, set()) == 0.0


def test_entity_match_score():
    sets = {"insulin": {"insulin"}, "diabetes": {"diabetes"}}
    assert ranker.entity_match_score({"diabetes"}, ["insulin", "diabetes"], sets) == pytest.approx(1.0)
    assert ranker.entity_match_score(set(), ["diabetes"], sets) == 0.0
    # falls back to tokenizing the name when not in the provided sets
    assert ranker.entity_match_score({"foo"}, ["foo bar"], {}) == pytest.approx(0.5)


def test_expansion_priority():
    hedge_meta = {
        "h2": {
            "description": "insulin treats diabetes",
            "relation": "treats",
            "type": "drug",
            "anchor": "insulin",
            "entities": ["diabetes"],
        }
    }
    hedge_token_sets = {"h2": {"insulin", "treats", "diabetes", "drug"}}
    entity_token_sets = {"insulin": {"insulin"}, "diabetes": {"diabetes"}}
    score = ranker.expansion_priority(
        query_tokens={"diabetes"},
        entity_name="diabetes",
        entity_sim=1.0,
        hid="h2",
        hedge_meta=hedge_meta,
        hedge_token_sets=hedge_token_sets,
        entity_token_sets=entity_token_sets,
    )
    # 0.60*1.0 + 0.20*1.0 + 0.10*1.0 + 0.10*0.35
    assert score == pytest.approx(0.935)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/retrieval/test_ranker.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'medical_qa_platform.retrieval.ranker'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/medical_qa_platform/retrieval/ranker.py`:

```python
"""Pure retrieval ranking logic ported from baseline retrieve_v1.

No faiss / sentence-transformers imports: this module operates on FAISS
outputs that have already been resolved to (rank, hid/name, score) hits plus
plain-Python KG metadata, so it is fully unit-testable and CI-verifiable.

Behavioral parity with baseline/scripts/serve/retrieval_tool.py::retrieve_v1
is the contract. See docs/specs/2026-06-03-retrieval-contract-sync-design.md.
"""

import re

DEFAULT_TOP_K = 5
PER_ENTITY_LIMIT = 8

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "does", "for", "from",
    "how", "in", "is", "of", "on", "or", "the", "to", "what", "which", "with",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Final fusion weights (sum to 1.10, matching baseline retrieve_v1 verbatim).
W_HE_SIM = 0.41
W_ENT_SIM = 0.14
W_HE_RRF = 0.14
W_ENT_RRF = 0.10
W_EXP_RRF = 0.08
W_EXP_SCORE = 0.08
W_LEXICAL = 0.10
W_ENTITY_MATCH = 0.05

# expansion_priority weights
EXP_W_ENT_SIM = 0.60
EXP_W_LEXICAL = 0.20
EXP_W_ENTITY_MATCH = 0.10
EXP_W_ANCHOR = 0.10
ANCHOR_BONUS_HIT = 1.0
ANCHOR_BONUS_MISS = 0.35


def candidate_budgets(top_k):
    """FAISS candidate counts for hyperedge and entity searches."""
    return max(40, top_k * 8), max(12, top_k * 4)


def tokenize(text):
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def rrf(rank, k=10):
    if rank is None:
        return 0.0
    return 1.0 / (k + rank + 1)


def lexical_score(query_tokens, hedge_tokens):
    if not query_tokens or not hedge_tokens:
        return 0.0
    overlap = len(query_tokens & hedge_tokens)
    return overlap / max(1, min(len(query_tokens), 6))


def entity_match_score(query_tokens, names, entity_token_sets):
    if not query_tokens:
        return 0.0
    best = 0.0
    for name in names:
        name_tokens = entity_token_sets.get(name)
        if not name_tokens:
            name_tokens = set(tokenize(name))
        if not name_tokens:
            continue
        best = max(best, len(query_tokens & name_tokens) / len(name_tokens))
    return best


def expansion_priority(
    query_tokens,
    entity_name,
    entity_sim,
    hid,
    hedge_meta,
    hedge_token_sets,
    entity_token_sets,
):
    meta = hedge_meta[hid]
    lexical = lexical_score(query_tokens, hedge_token_sets.get(hid, set()))
    names = [meta.get("anchor", ""), *meta.get("entities", [])]
    entity_match = entity_match_score(query_tokens, names, entity_token_sets)
    anchor_bonus = ANCHOR_BONUS_HIT if meta.get("anchor") == entity_name else ANCHOR_BONUS_MISS
    return (
        EXP_W_ENT_SIM * entity_sim
        + EXP_W_LEXICAL * lexical
        + EXP_W_ENTITY_MATCH * entity_match
        + EXP_W_ANCHOR * anchor_bonus
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/retrieval/test_ranker.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/medical_qa_platform/retrieval/ranker.py tests/retrieval/test_ranker.py
git commit -m "feat(retrieval): pure ranker helpers ported from retrieve_v1"
```

---

## Task 2: `fuse_candidates` (the ported fusion ranker)

**Files:**
- Modify: `src/medical_qa_platform/retrieval/ranker.py`
- Test: `tests/retrieval/test_ranker.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/retrieval/test_ranker.py`:

```python
def _diabetes_kg():
    hedge_meta = {
        "h1": {
            "description": "metformin treats diabetes",
            "relation": "treats",
            "type": "drug",
            "anchor": "metformin",
            "entities": ["diabetes"],
        },
        "h2": {
            "description": "insulin treats diabetes",
            "relation": "treats",
            "type": "drug",
            "anchor": "insulin",
            "entities": ["diabetes"],
        },
    }
    hedge_token_sets = {
        "h1": {"metformin", "treats", "diabetes", "drug"},
        "h2": {"insulin", "treats", "diabetes", "drug"},
    }
    entity_token_sets = {
        "diabetes": {"diabetes"},
        "metformin": {"metformin"},
        "insulin": {"insulin"},
    }
    entity_to_hedges = {"diabetes": ["h2"], "metformin": ["h1"]}
    return hedge_meta, hedge_token_sets, entity_token_sets, entity_to_hedges


def test_fuse_candidates_ranks_direct_hyperedge_above_expansion():
    hedge_meta, hedge_token_sets, entity_token_sets, entity_to_hedges = _diabetes_kg()
    ranked = ranker.fuse_candidates(
        he_hits=[(0, "h1", 1.0)],
        ent_hits=[(0, "diabetes", 1.0)],
        hedge_meta=hedge_meta,
        hedge_token_sets=hedge_token_sets,
        entity_token_sets=entity_token_sets,
        entity_to_hedges=entity_to_hedges,
        query_tokens={"diabetes"},
        top_k=5,
    )
    # h1 (direct hyperedge hit, score ~0.573) ranks above h2 (expansion, ~0.381)
    assert ranked == ["h1", "h2"]


def test_fuse_candidates_respects_top_k_and_skips_entities_without_hedges():
    hedge_meta, hedge_token_sets, entity_token_sets, entity_to_hedges = _diabetes_kg()
    ranked = ranker.fuse_candidates(
        he_hits=[(0, "h1", 1.0)],
        ent_hits=[(5, "unknown_entity", 0.9)],  # not in entity_to_hedges -> skipped
        hedge_meta=hedge_meta,
        hedge_token_sets=hedge_token_sets,
        entity_token_sets=entity_token_sets,
        entity_to_hedges=entity_to_hedges,
        query_tokens={"diabetes"},
        top_k=1,
    )
    assert ranked == ["h1"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/retrieval/test_ranker.py -k fuse -q`
Expected: FAIL — `AttributeError: module ... has no attribute 'fuse_candidates'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/medical_qa_platform/retrieval/ranker.py`:

```python
def _blank_candidate():
    return {
        "he_rank": None,
        "he_sim": 0.0,
        "ent_rank": None,
        "ent_sim": 0.0,
        "exp_rank": None,
        "exp_score": 0.0,
    }


def fuse_candidates(
    *,
    he_hits,
    ent_hits,
    hedge_meta,
    hedge_token_sets,
    entity_token_sets,
    entity_to_hedges,
    query_tokens,
    top_k=DEFAULT_TOP_K,
    per_entity_limit=PER_ENTITY_LIMIT,
):
    """Return ranked hyperedge ids, byte-for-byte matching retrieve_v1.

    he_hits: list of (rank, hid, sim) in original FAISS order, already filtered
        to valid indices present in hedge_meta.
    ent_hits: list of (rank, entity_name, sim) in original FAISS order, filtered
        only to valid indices (the no-related-hedges check happens here).
    Candidate insertion order (hyperedge hits first, then expansion order) is
    preserved so that the stable descending sort breaks ties identically to
    baseline.
    """
    candidates = {}

    for rank, hid, sim in he_hits:
        cand = candidates.setdefault(hid, _blank_candidate())
        cand["he_rank"] = rank
        cand["he_sim"] = max(cand["he_sim"], sim)

    for ent_rank, entity_name, ent_sim in ent_hits:
        related_hids = entity_to_hedges.get(entity_name, [])
        if not related_hids:
            continue
        expanded = []
        for hid in related_hids:
            if hid not in hedge_meta:
                continue
            expanded.append(
                (
                    expansion_priority(
                        query_tokens,
                        entity_name,
                        ent_sim,
                        hid,
                        hedge_meta,
                        hedge_token_sets,
                        entity_token_sets,
                    ),
                    hid,
                )
            )
        expanded.sort(key=lambda x: x[0], reverse=True)
        for exp_rank, (exp_score, hid) in enumerate(expanded[:per_entity_limit]):
            cand = candidates.setdefault(hid, _blank_candidate())
            cand["ent_rank"] = ent_rank if cand["ent_rank"] is None else min(cand["ent_rank"], ent_rank)
            cand["ent_sim"] = max(cand["ent_sim"], ent_sim)
            cand["exp_rank"] = exp_rank if cand["exp_rank"] is None else min(cand["exp_rank"], exp_rank)
            cand["exp_score"] = max(cand["exp_score"], exp_score)

    ranked = []
    for hid, cand in candidates.items():
        lexical = lexical_score(query_tokens, hedge_token_sets.get(hid, set()))
        meta = hedge_meta[hid]
        names = [meta.get("anchor", ""), *meta.get("entities", [])]
        entity_match = entity_match_score(query_tokens, names, entity_token_sets)
        final_score = (
            W_HE_SIM * cand["he_sim"]
            + W_ENT_SIM * cand["ent_sim"]
            + W_HE_RRF * rrf(cand["he_rank"])
            + W_ENT_RRF * rrf(cand["ent_rank"])
            + W_EXP_RRF * rrf(cand["exp_rank"])
            + W_EXP_SCORE * min(1.0, cand["exp_score"])
            + W_LEXICAL * lexical
            + W_ENTITY_MATCH * entity_match
        )
        ranked.append((final_score, hid))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return [hid for _, hid in ranked[:top_k]]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/retrieval/test_ranker.py -q`
Expected: PASS (8 passed).

- [ ] **Step 5: Commit**

```bash
git add src/medical_qa_platform/retrieval/ranker.py tests/retrieval/test_ranker.py
git commit -m "feat(retrieval): port retrieve_v1 fusion ranker as fuse_candidates"
```

---

## Task 3: Output contract (`format_evidence` + version)

**Files:**
- Create: `src/medical_qa_platform/retrieval/contract.py`
- Test: `tests/retrieval/test_contract.py`

- [ ] **Step 1: Write the failing test**

Create `tests/retrieval/test_contract.py`:

```python
from medical_qa_platform.retrieval import contract


def test_format_evidence_empty_returns_sentinel():
    assert contract.format_evidence([]) == "No relevant knowledge found."


def test_format_evidence_joins_with_dash_prefix():
    assert contract.format_evidence(["a fact", "b fact"]) == "- a fact\n- b fact"


def test_contract_version_is_v1_medembed_large():
    assert contract.RETRIEVAL_CONTRACT_VERSION == "v1-medembed-large"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/retrieval/test_contract.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'medical_qa_platform.retrieval.contract'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/medical_qa_platform/retrieval/contract.py`:

```python
"""The retrieval contract: tool-response formatting + contract version.

format_evidence reproduces the exact string baseline search_medical_knowledge
returns, so the tokens the model sees at serving time match training.
"""

RETRIEVAL_CONTRACT_VERSION = "v1-medembed-large"

NO_RESULTS = "No relevant knowledge found."


def format_evidence(results):
    if not results:
        return NO_RESULTS
    return "\n".join(f"- {item}" for item in results)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/retrieval/test_contract.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/medical_qa_platform/retrieval/contract.py tests/retrieval/test_contract.py
git commit -m "feat(retrieval): add format_evidence + RETRIEVAL_CONTRACT_VERSION"
```

---

## Task 4: Rewrite `KGRetrieval` to delegate to the ranker

**Files:**
- Modify: `src/medical_qa_platform/retrieval/kg_backend.py` (full rewrite)
- Test (existing, must stay green): `tests/retrieval/test_kg_backend_self_contained.py`, `tests/retrieval/test_kg_backend_import.py`

There is no new unit test here: `kg_backend.py` is `# pragma: no cover` (faiss/encoder I/O). Its behavior is exercised by the existing self-contained test (which fakes faiss/numpy/ST) and by the opt-in L2 parity test (Task 7). The hand-traced result of the self-contained fixture under the v1 ranker is `["metformin treats diabetes", "insulin treats diabetes"]`, identical to the current assertion, so that test stays green.

- [ ] **Step 1: Rewrite the file**

Replace the entire contents of `src/medical_qa_platform/retrieval/kg_backend.py` with:

```python
"""Self-contained FAISS-backed medical knowledge retrieval (retrieve_v1 parity).

Heavy I/O only: load artifacts, encode the query, run FAISS, resolve indices to
(rank, hid/name, sim) hits, then delegate ranking to the pure `ranker` module.
"""

import json
import os
from pathlib import Path

from . import ranker
from .backends import RetrievalBackend
from .device import resolve_device


class KGRetrieval(RetrievalBackend):  # pragma: no cover
    """Load local KG artifacts and retrieve hyperedge descriptions via retrieve_v1."""

    def __init__(self, data_dir=None, device=None, encoder_model=None):
        self.data_dir = Path(data_dir or os.environ.get("KG_DATA_DIR", "data/"))
        self.device = resolve_device(device)
        self.encoder_model = encoder_model or os.environ.get(
            "KG_ENCODER_MODEL", "abhinand/MedEmbed-large-v0.1"
        )

        faiss, np, sentence_transformer = self._load_runtime_dependencies()
        self._np = np
        self.encoder = sentence_transformer(self.encoder_model, device=self.device)
        self.idx_he = faiss.read_index(str(self.data_dir / "index_hyperedge.bin"))
        self.idx_ent = faiss.read_index(str(self.data_dir / "index_entity.bin"))
        self.hedge_ids = np.load(
            self.data_dir / "hedge_ids.npy", allow_pickle=True
        ).tolist()
        self.ent_names = np.load(
            self.data_dir / "entity_names.npy", allow_pickle=True
        ).tolist()

        with open(self.data_dir / "medical_hg.json") as handle:
            graph = json.load(handle)
        self.hedge_meta = {
            item["id"]: {
                "description": item["description"],
                "relation": item.get("relation", ""),
                "type": item.get("type", ""),
                "anchor": item.get("anchor", ""),
                "entities": item.get("entities", []),
            }
            for item in graph["hyperedges"]
        }
        self.hedge_by_id = {
            hid: meta["description"] for hid, meta in self.hedge_meta.items()
        }
        self.entity_to_hedges = graph.get("entity_to_hedges", {})
        self.hedge_token_sets = {
            hid: set(
                ranker.tokenize(
                    " ".join(
                        [
                            meta["description"],
                            meta["relation"],
                            meta["type"],
                            meta["anchor"],
                            *meta["entities"],
                        ]
                    )
                )
            )
            for hid, meta in self.hedge_meta.items()
        }
        self.entity_token_sets = {
            name: set(ranker.tokenize(name)) for name in graph.get("entities", {})
        }

    @staticmethod
    def _load_runtime_dependencies():
        try:
            import faiss
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "KGRetrieval requires the optional runtime extra: "
                "pip install 'medical_qa_platform[runtime]'"
            ) from exc
        return faiss, np, SentenceTransformer

    @staticmethod
    def _first_row(values):
        if hasattr(values, "tolist"):
            values = values.tolist()
        if not values:
            return []
        return list(values[0])

    def _encode_query(self, query):
        embedding = self.encoder.encode([query], normalize_embeddings=True)
        if hasattr(embedding, "astype"):
            return embedding.astype(self._np.float32)
        return embedding

    def search(self, query, top_k=ranker.DEFAULT_TOP_K):
        if top_k <= 0:
            return []

        embedding = self._encode_query(query)
        he_k, ent_k = ranker.candidate_budgets(top_k)
        he_scores, he_ids = self.idx_he.search(embedding, he_k)
        ent_scores, ent_ids = self.idx_ent.search(embedding, ent_k)

        he_scores_row = self._first_row(he_scores)
        he_ids_row = self._first_row(he_ids)
        ent_scores_row = self._first_row(ent_scores)
        ent_ids_row = self._first_row(ent_ids)

        he_hits = []
        for rank, (score, raw_idx) in enumerate(zip(he_scores_row, he_ids_row)):
            idx = int(raw_idx)
            if idx < 0 or idx >= len(self.hedge_ids):
                continue
            hid = self.hedge_ids[idx]
            if hid not in self.hedge_meta:
                continue
            he_hits.append((rank, hid, float(score)))

        ent_hits = []
        for ent_rank, (score, raw_idx) in enumerate(zip(ent_scores_row, ent_ids_row)):
            idx = int(raw_idx)
            if idx < 0 or idx >= len(self.ent_names):
                continue
            ent_hits.append((ent_rank, self.ent_names[idx], float(score)))

        ranked_hids = ranker.fuse_candidates(
            he_hits=he_hits,
            ent_hits=ent_hits,
            hedge_meta=self.hedge_meta,
            hedge_token_sets=self.hedge_token_sets,
            entity_token_sets=self.entity_token_sets,
            entity_to_hedges=self.entity_to_hedges,
            query_tokens=set(ranker.tokenize(query)),
            top_k=top_k,
        )
        return [self.hedge_by_id[hid] for hid in ranked_hids]
```

- [ ] **Step 2: Run the existing backend tests to verify they stay green**

Run: `.venv/bin/python -m pytest tests/retrieval/test_kg_backend_self_contained.py tests/retrieval/test_kg_backend_import.py -q`
Expected: PASS (3 passed). In particular `test_kg_retrieval_loads_local_artifacts_without_baseline` still returns `["metformin treats diabetes", "insulin treats diabetes"]`, and `test_runtime_source_does_not_reference_baseline` confirms no `baseline` token leaked.

- [ ] **Step 3: Run the whole suite to confirm nothing else broke**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS (all tests).

- [ ] **Step 4: Commit**

```bash
git add src/medical_qa_platform/retrieval/kg_backend.py
git commit -m "refactor(retrieval): KGRetrieval delegates ranking to ported retrieve_v1"
```

---

## Task 5: Golden generator + fixture (run once in a baseline venv)

**Files:**
- Create: `scripts/gen_retrieval_golden.py`
- Create (produced output, committed): `tests/retrieval/golden/retrieve_v1_golden.json`

This script imports baseline as a reference to capture (a) the FAISS front-half hits and (b) baseline `retrieve_v1`'s expected ranked descriptions, for a fixed query set, into a golden fixture. It is **not** part of the shipped package and is **not** imported by any test — it only writes the fixture. It is run from the repo root with a baseline venv that has faiss/sentence-transformers.

- [ ] **Step 1: Create the generator**

Create `scripts/gen_retrieval_golden.py`:

```python
"""Generate the retrieve_v1 golden parity fixture from the baseline reference.

Run once (and after any intentional algorithm/encoder change) in a baseline
venv that has faiss + sentence-transformers, e.g.:

    cd /home/vcsai/minhlbq/baseline
    ./training_venv312/bin/python \
        ../mlops-platform/scripts/gen_retrieval_golden.py \
        --baseline-root /home/vcsai/minhlbq/baseline \
        --data-dir /home/vcsai/minhlbq/baseline/data \
        --device cpu \
        --out ../mlops-platform/tests/retrieval/golden/retrieve_v1_golden.json

The fixture captures, per query: the resolved FAISS hits (he_hits/ent_hits with
original ranks), baseline's expected ranked descriptions, and a minimal KG slice
covering every referenced hyperedge/entity so the pure ranker can be replayed in
CI without faiss/encoder.
"""

import argparse
import json
import subprocess
import sys

QUERIES = [
    "What is the mechanism of action of metformin in type 2 diabetes?",
    "first line treatment for essential hypertension",
    "classic symptoms of acute myocardial infarction",
    "cerebrospinal fluid findings in bacterial meningitis",
    "warfarin drug interactions and mechanism",
    "pathophysiology of asthma airway inflammation",
    "emergency management of severe hyperkalemia",
    "complications of diabetic ketoacidosis",
]

TOP_K = 5


def _git_commit(root):
    try:
        return (
            subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"])
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", required=True)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()

    sys.path.insert(0, args.baseline_root)
    from scripts.serve.retrieval_tool import MedicalKnowledgeTool

    tool = MedicalKnowledgeTool.load(data_dir=args.data_dir, device=args.device)
    he_k = max(40, args.top_k * 8)
    ent_k = max(12, args.top_k * 4)

    cases = []
    ref_hids = set()
    ref_entities = set()

    for query in QUERIES:
        q_emb = tool.encoder.encode([query], normalize_embeddings=True).astype("float32")
        he_scores, he_ids = tool.idx_he.search(q_emb, he_k)
        ent_scores, ent_ids = tool.idx_ent.search(q_emb, ent_k)

        he_hits = []
        for rank, (score, raw_idx) in enumerate(zip(he_scores[0], he_ids[0])):
            idx = int(raw_idx)
            if idx < 0 or idx >= len(tool.hedge_ids):
                continue
            hid = tool.hedge_ids[idx]
            if hid not in tool.hedge_meta:
                continue
            he_hits.append([rank, hid, float(score)])
            ref_hids.add(hid)

        ent_hits = []
        for ent_rank, (score, raw_idx) in enumerate(zip(ent_scores[0], ent_ids[0])):
            idx = int(raw_idx)
            if idx < 0 or idx >= len(tool.ent_names):
                continue
            name = tool.ent_names[idx]
            ent_hits.append([ent_rank, name, float(score)])
            ref_entities.add(name)
            for hid in tool.entity_to_hedges.get(name, []):
                if hid in tool.hedge_meta:
                    ref_hids.add(hid)

        expected = tool.retrieve_v1(query, args.top_k)
        cases.append(
            {
                "query": query,
                "baseline_query_tokens": sorted(set(tool._tokenize(query))),
                "he_hits": he_hits,
                "ent_hits": ent_hits,
                "expected_ranked_descriptions": expected,
            }
        )

    # entity_token_sets must also cover anchors/entities referenced by sliced hedges
    token_entities = set(ref_entities)
    for hid in ref_hids:
        meta = tool.hedge_meta[hid]
        token_entities.add(meta.get("anchor", ""))
        token_entities.update(meta.get("entities", []))

    kg_slice = {
        "hedge_meta": {hid: tool.hedge_meta[hid] for hid in sorted(ref_hids)},
        "entity_to_hedges": {
            name: tool.entity_to_hedges.get(name, []) for name in sorted(ref_entities)
        },
        "hedge_token_sets": {
            hid: sorted(tool.hedge_token_sets[hid]) for hid in sorted(ref_hids)
        },
        "entity_token_sets": {
            name: sorted(tool.entity_token_sets[name])
            for name in sorted(token_entities)
            if name in tool.entity_token_sets
        },
    }

    fixture = {
        "manifest": {
            "encoder_model": "abhinand/MedEmbed-large-v0.1",
            "device": args.device,
            "data_dir": args.data_dir,
            "top_k": args.top_k,
            "baseline_commit": _git_commit(args.baseline_root),
        },
        "cases": cases,
        "kg_slice": kg_slice,
    }

    with open(args.out, "w") as handle:
        json.dump(fixture, handle, indent=2, sort_keys=True)
    print(f"wrote {len(cases)} cases to {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the generator in a baseline venv to produce the fixture**

Run:
```bash
mkdir -p tests/retrieval/golden
cd /home/vcsai/minhlbq/baseline && ./training_venv312/bin/python \
    /home/vcsai/minhlbq/mlops-platform/scripts/gen_retrieval_golden.py \
    --baseline-root /home/vcsai/minhlbq/baseline \
    --data-dir /home/vcsai/minhlbq/baseline/data \
    --device cpu \
    --out /home/vcsai/minhlbq/mlops-platform/tests/retrieval/golden/retrieve_v1_golden.json
```
Expected stdout: `wrote 8 cases to .../retrieve_v1_golden.json`. If `training_venv312` lacks faiss/ST, use `./vllm_venv312/bin/python` instead.

- [ ] **Step 3: Sanity-check the fixture**

Run: `cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/python -c "import json; d=json.load(open('tests/retrieval/golden/retrieve_v1_golden.json')); print(len(d['cases']), 'cases'); print('hids', len(d['kg_slice']['hedge_meta'])); assert all(c['expected_ranked_descriptions'] for c in d['cases'])"`
Expected: prints `8 cases` and a non-zero hedge count; no assertion error.

- [ ] **Step 4: Commit**

```bash
cd /home/vcsai/minhlbq/mlops-platform
git add scripts/gen_retrieval_golden.py tests/retrieval/golden/retrieve_v1_golden.json
git commit -m "test(retrieval): golden generator + retrieve_v1 parity fixture"
```

---

## Task 6: L1 parity test (CI, no heavy deps)

**Files:**
- Create: `tests/retrieval/test_contract_parity.py`

- [ ] **Step 1: Write the test**

Create `tests/retrieval/test_contract_parity.py`:

```python
import json
from pathlib import Path

import pytest

from medical_qa_platform.retrieval import ranker

GOLDEN = Path(__file__).parent / "golden" / "retrieve_v1_golden.json"


def _load_golden():
    with open(GOLDEN) as handle:
        return json.load(handle)


def _kg_slice(fixture):
    s = fixture["kg_slice"]
    hedge_token_sets = {hid: set(toks) for hid, toks in s["hedge_token_sets"].items()}
    entity_token_sets = {name: set(toks) for name, toks in s["entity_token_sets"].items()}
    return s["hedge_meta"], hedge_token_sets, entity_token_sets, s["entity_to_hedges"]


def test_golden_fixture_present_and_nonempty():
    fixture = _load_golden()
    assert fixture["cases"], "golden fixture has no cases"


def test_tokenize_matches_baseline_for_every_query():
    fixture = _load_golden()
    for case in fixture["cases"]:
        assert sorted(set(ranker.tokenize(case["query"]))) == case["baseline_query_tokens"]


def test_fuse_candidates_reproduces_baseline_retrieve_v1():
    fixture = _load_golden()
    hedge_meta, hedge_token_sets, entity_token_sets, entity_to_hedges = _kg_slice(fixture)
    top_k = fixture["manifest"]["top_k"]

    for case in fixture["cases"]:
        he_hits = [tuple(hit) for hit in case["he_hits"]]
        ent_hits = [tuple(hit) for hit in case["ent_hits"]]
        ranked_hids = ranker.fuse_candidates(
            he_hits=he_hits,
            ent_hits=ent_hits,
            hedge_meta=hedge_meta,
            hedge_token_sets=hedge_token_sets,
            entity_token_sets=entity_token_sets,
            entity_to_hedges=entity_to_hedges,
            query_tokens=set(ranker.tokenize(case["query"])),
            top_k=top_k,
        )
        got = [hedge_meta[hid]["description"] for hid in ranked_hids]
        assert got == case["expected_ranked_descriptions"], f"mismatch for query: {case['query']!r}"
```

- [ ] **Step 2: Run the test**

Run: `.venv/bin/python -m pytest tests/retrieval/test_contract_parity.py -q`
Expected: PASS (3 passed). If `test_fuse_candidates_reproduces_baseline_retrieve_v1` fails, the port diverges from baseline — debug `ranker.fuse_candidates` against `baseline/scripts/serve/retrieval_tool.py::retrieve_v1` (do NOT edit the golden to match the port).

- [ ] **Step 3: Commit**

```bash
git add tests/retrieval/test_contract_parity.py
git commit -m "test(retrieval): L1 golden parity test for fuse_candidates"
```

---

## Task 7: L2 opt-in full-backend parity test

**Files:**
- Modify: `tests/retrieval/test_contract_parity.py`
- Modify: `pyproject.toml` (register the `runtime` marker)

- [ ] **Step 1: Register the pytest marker**

In `pyproject.toml`, add (or extend) the pytest config section. If `[tool.pytest.ini_options]` does not exist, add it:

```toml
[tool.pytest.ini_options]
markers = [
    "runtime: tests needing the [runtime] extra (faiss/sentence-transformers) and real KG artifacts; skipped by default",
]
```

If the section already exists, add only the `markers` entry (merge with existing keys).

- [ ] **Step 2: Append the L2 test**

Append to `tests/retrieval/test_contract_parity.py`:

```python
import os

RUNTIME_DATA_DIR = os.environ.get("KG_DATA_DIR")


@pytest.mark.runtime
@pytest.mark.skipif(
    not RUNTIME_DATA_DIR or not Path(RUNTIME_DATA_DIR).exists(),
    reason="set KG_DATA_DIR to a real KG artifact dir (with [runtime] extra) to run L2 parity",
)
def test_full_kgretrieval_matches_baseline_on_real_artifacts():
    """End-to-end parity: full KGRetrieval (incl. encode + faiss) vs the golden.

    Runs on CPU only (RETRIEVAL_DEVICE=cpu) to avoid float nondeterminism.
    Uses the golden queries + expected descriptions as the oracle.
    """
    os.environ.setdefault("RETRIEVAL_DEVICE", "cpu")
    from medical_qa_platform.retrieval.kg_backend import KGRetrieval

    fixture = _load_golden()
    top_k = fixture["manifest"]["top_k"]
    backend = KGRetrieval(data_dir=RUNTIME_DATA_DIR, device="cpu")

    for case in fixture["cases"]:
        got = backend.search(case["query"], top_k=top_k)
        assert got == case["expected_ranked_descriptions"], f"mismatch for query: {case['query']!r}"
```

- [ ] **Step 3: Verify L2 is collected but skipped by default**

Run: `.venv/bin/python -m pytest tests/retrieval/test_contract_parity.py -q -rs`
Expected: the 3 L1 tests PASS and the L2 test is SKIPPED (reason mentions `KG_DATA_DIR`). No marker warning.

- [ ] **Step 4: (Manual, before merge) Run L2 against real artifacts**

Run (requires the `[runtime]` extra installed and CPU artifacts present):
```bash
KG_DATA_DIR=/home/vcsai/minhlbq/baseline/data RETRIEVAL_DEVICE=cpu \
  .venv/bin/python -m pytest tests/retrieval/test_contract_parity.py::test_full_kgretrieval_matches_baseline_on_real_artifacts -q
```
Expected: PASS. Document the result in the PR. If it fails on near-tie reordering despite CPU, record which query and treat as a real parity gap to fix in `kg_backend.py` front-half (not by editing the golden).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml tests/retrieval/test_contract_parity.py
git commit -m "test(retrieval): L2 opt-in full-backend parity test + runtime marker"
```

---

## Task 8: Reconcile the serving system prompt with the canonical text

**Files:**
- Modify: `src/medical_qa_platform/api/prompt.py`
- Create: `tests/api/test_system_prompt_sync.py`

The canonical source of truth is `baseline/scripts/train_rl/data_prep.py::SYSTEM_PROMPT` (which must equal `sft_train.py` and `grpo_eval.py` per the project's working rules). Copy it verbatim. The single-shot-vs-agentic structural difference (tool round-trip vs pre-injected evidence) is explicitly out of SP1 scope and noted as a follow-up.

- [ ] **Step 1: Write the failing test**

Create `tests/api/test_system_prompt_sync.py`:

```python
from medical_qa_platform.api.prompt import SYSTEM_PROMPT

CANONICAL = (
    "You are a medical reasoning assistant with access to a "
    "search_medical_knowledge tool.\n\n"
    "Structure your response in this order:\n"
    "1. <think>Your initial reasoning about the question</think>\n"
    "2. (Optional) If you need to verify a medical fact, call "
    "search_medical_knowledge, then add another <think>...</think> "
    "incorporating the result.\n"
    "3. <answer>Your final answer</answer>\n\n"
    "IMPORTANT: In <answer> tags, write ONLY the option letter (e.g. A) "
    "or a short answer, NOT an explanation."
)


def test_system_prompt_matches_canonical_baseline():
    assert SYSTEM_PROMPT == CANONICAL


def test_build_prompt_renders_evidence_via_contract_format():
    """Lock the api's evidence rendering to the Layer-2 contract formatter."""
    from medical_qa_platform.api.prompt import build_prompt
    from medical_qa_platform.retrieval.contract import format_evidence

    msgs = build_prompt("Q?", {"A": "x", "B": "y"}, ["fact one", "fact two"])
    user = msgs[1]["content"]
    assert format_evidence(["fact one", "fact two"]) in user
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/api/test_system_prompt_sync.py -q`
Expected: FAIL (current `SYSTEM_PROMPT` is the placeholder wording).

- [ ] **Step 3: Update `prompt.py`**

In `src/medical_qa_platform/api/prompt.py`, replace the module docstring NOTE and the `SYSTEM_PROMPT` assignment. Old:

```python
"""Build chat messages for the medical MCQ task.

NOTE: SYSTEM_PROMPT must be reconciled with the canonical eval prompt before the
real-model demo. For mock/local testing the exact wording is not asserted.
"""

SYSTEM_PROMPT = (
    "You are a medical question-answering assistant. Read the question and "
    "options, reason step by step inside <think></think>, then give the single "
    "best option letter inside <answer></answer>."
)
```

New:

```python
"""Build chat messages for the medical MCQ task.

SYSTEM_PROMPT is kept verbatim in sync with the canonical training/eval prompt
(baseline/scripts/train_rl/data_prep.py::SYSTEM_PROMPT) so the model sees at
serving time the exact system message it was trained against. The structural
difference between single-shot evidence injection here and the agentic tool
round-trip used in training is tracked as a separate sub-project.
"""

SYSTEM_PROMPT = (
    "You are a medical reasoning assistant with access to a "
    "search_medical_knowledge tool.\n\n"
    "Structure your response in this order:\n"
    "1. <think>Your initial reasoning about the question</think>\n"
    "2. (Optional) If you need to verify a medical fact, call "
    "search_medical_knowledge, then add another <think>...</think> "
    "incorporating the result.\n"
    "3. <answer>Your final answer</answer>\n\n"
    "IMPORTANT: In <answer> tags, write ONLY the option letter (e.g. A) "
    "or a short answer, NOT an explanation."
)
```

Leave `build_prompt` unchanged (its per-line `- {item}` evidence format already matches `format_evidence`).

- [ ] **Step 4: Run the prompt tests**

Run: `.venv/bin/python -m pytest tests/api/test_system_prompt_sync.py tests/api/test_prompt.py -q`
Expected: PASS. If `tests/api/test_prompt.py` asserted the old wording, update those assertions to the canonical text (the prompt content moved; the message structure is unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/medical_qa_platform/api/prompt.py tests/api/test_system_prompt_sync.py
git commit -m "feat(api): sync SYSTEM_PROMPT verbatim with canonical training prompt"
```

---

## Task 9: Full-suite + coverage gate

**Files:** none (verification only)

- [ ] **Step 1: Run the full suite with coverage**

Run: `.venv/bin/python -m pytest --cov --cov-report= -q`
Expected: all tests pass; final line reports `Required test coverage of 80.0% reached. Total coverage: <>=80>%`. `ranker.py` and `contract.py` are counted (not in the coverage `omit` list); `kg_backend.py` stays omitted.

- [ ] **Step 2: Confirm the self-contained guard still holds**

Run: `.venv/bin/python -m pytest tests/retrieval/test_kg_backend_self_contained.py -q`
Expected: PASS — no `baseline` token leaked into `src/medical_qa_platform/**`.

- [ ] **Step 3: Final commit (if any config/lint touched); otherwise stop**

```bash
git status
# if clean, nothing to commit; the feature is complete on branch retrieval-contract-sync
```

---

## Notes for the implementer

- **Never edit the golden to make a test pass.** The golden is baseline's ground truth; a mismatch means the port (or the front-half capture) is wrong.
- **Determinism:** L1 is exact and CPU-only by construction. L2 must run with `RETRIEVAL_DEVICE=cpu` on both sides; CUDA embeddings can reorder near-ties.
- **Tie-breaking parity** depends on candidate insertion order (hyperedge hits first, then expansion order) feeding Python's stable descending sort. Do not reorder the loops in `fuse_candidates`.
- **No new tokenizer tokens, no encoder change** in SP1 — the encoder swap is SP2, which will bump `RETRIEVAL_CONTRACT_VERSION` and regenerate the golden.
- **`entity_type_by_name`** (listed among the spec's load-time structures) is intentionally **not** built: baseline `retrieve_v1` never references it, so it would be dead state. Confirm before adding it back if a later stage needs it.
- **Baseline prompt agreement:** the canonical `SYSTEM_PROMPT` lives in three baseline files (`data_prep.py`, `sft_train.py`, `grpo_eval.py`) and must agree per the project's working rules. If they diverge, surface it as a pre-existing baseline bug rather than silently picking one.
