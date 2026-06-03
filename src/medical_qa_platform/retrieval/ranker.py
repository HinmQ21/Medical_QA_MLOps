"""Pure retrieval ranking logic ported from the reference retrieve_v1 ranker.

No faiss / sentence-transformers imports: this module operates on FAISS
outputs that have already been resolved to (rank, hid/name, score) hits plus
plain-Python KG metadata, so it is fully unit-testable and CI-verifiable.

Behavioral parity with the reference retrieve_v1 ranker is the contract.
See docs/specs/2026-06-03-retrieval-contract-sync-design.md.
"""

import re

DEFAULT_TOP_K = 5
PER_ENTITY_LIMIT = 8

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "does", "for", "from",
    "how", "in", "is", "of", "on", "or", "the", "to", "what", "which", "with",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Final fusion weights (sum to 1.10, matching the reference retrieve_v1 verbatim).
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
    the reference ranker.
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
