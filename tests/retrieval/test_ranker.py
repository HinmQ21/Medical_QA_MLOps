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
