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
