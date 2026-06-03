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
